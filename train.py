import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model import Generator, Discriminator
from utils import normalize_m11, load_image, imresize, create_dir


class Trainer:
    def __init__(self,  
                 num_scales, 
                 num_iters, 
                 max_size, 
                 min_size, 
                 scale_factor,
                 learning_rate, 
                 checkpoint_dir, 
                 debug):

        self.num_scales = num_scales
        self.num_iters = num_iters
        self.num_filters = [32*pow(2, (scale//4)) for scale in range(self.num_scales)] # num_filters double for every 4 scales
        self.max_size = max_size
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.noise_amp_init = 0.1

        self.checkpoint_dir = checkpoint_dir
        self.G_dir = self.checkpoint_dir + '/G'
        self.D_dir = self.checkpoint_dir + '/D'

        self.learning_schedule = ExponentialDecay(learning_rate, decay_steps=4800, decay_rate=0.1, staircase=True) # 1600 * 3 steps
        self.build_model()

        self.debug = debug
        if self.debug:
            self.create_summary_writer()
            self.create_metrics()


    def build_model(self):
        """ Build initial model """
        create_dir(self.checkpoint_dir)
        self.generators = []
        self.discriminators = []
        for scale in range(self.num_scales):
            self.generators.append(Generator(num_filters=self.num_filters[scale]))
            self.discriminators.append(Discriminator(num_filters=self.num_filters[scale]))    


    def save_model(self, scale):
        """ Save weights and NoiseAmp """
        G_dir = self.G_dir + f'{scale}'
        D_dir = self.D_dir + f'{scale}'
        if not os.path.exists(G_dir):
            os.makedirs(G_dir)
        if not os.path.exists(D_dir):
            os.makedirs(D_dir)

        self.generators[scale].save_weights(G_dir + '/G', save_format='tf')
        self.discriminators[scale].save_weights(D_dir + '/D', save_format='tf')
        np.save(self.checkpoint_dir + '/NoiseAmp', self.NoiseAmp)


    def init_from_previous_model(self, scale):
        """ Initialize current model from the previous trained model """
        if self.num_filters[scale] == self.num_filters[scale-1]:
            self.generators[scale].load_weights(self.G_dir + f'{scale-1}/G')
            self.discriminators[scale].load_weights(self.D_dir + f'{scale-1}/D')    


    def train(self, training_image):
        """ Training """
        real_image = load_image(training_image, image_size=self.max_size)
        real_image = normalize_m11(real_image)
        reals = self.create_real_pyramid(real_image)

        self.Z_fixed = []
        self.NoiseAmp = []
        noise_amp = tf.constant(0.1)

        for scale in range(self.num_scales):
            print(scale)
            start = time.perf_counter()

            if scale > 0:
                self.init_from_previous_model(scale)
            g_opt = Adam(learning_rate=self.learning_schedule, beta_1=0.5, beta_2=0.999)
            d_opt = Adam(learning_rate=self.learning_schedule, beta_1=0.5, beta_2=0.999)

            """ Build with shape """
            prev_rec = tf.zeros_like(reals[scale])
            self.discriminators[scale](prev_rec)
            self.generators[scale](prev_rec, prev_rec)

            train_step = self.wrapper()

            for step in tf.range(self.num_iters):
                z_fixed, prev_rec, noise_amp, metrics = train_step(reals, prev_rec, noise_amp, scale, step, g_opt, d_opt)
    
            self.Z_fixed.append(z_fixed)
            self.NoiseAmp.append(noise_amp)
            self.save_model(scale)

            if self.debug:
                self.write_summaries(metrics, scale)
                self.update_metrics(metrics, scale)
                print(f'Time taken for scale {scale} is {time.perf_counter()-start:.2f} sec\n')


    def wrapper(self):
        @tf.function
        def train_step(reals, prev_rec, noise_amp, scale, step, g_opt, d_opt):
            real = reals[scale]
            z_rand = tf.random.normal(real.shape)

            if scale == 0:
                z_rec = tf.random.normal(real.shape)
            else:
                z_rec = tf.zeros_like(real)

            for i in range(6):
                if i == 0 and tf.equal(step, 0):
                    if scale == 0:
                        """ Coarsest scale is purely generative """
                        prev_rand = tf.zeros_like(real)
                        prev_rec = tf.zeros_like(real)
                        noise_amp = 1.0
                    else:
                        """ Finer scale takes noise and image generated from previous scale as input """
                        prev_rand = self.generate_from_coarsest(scale, reals, 'rand')
                        prev_rec = self.generate_from_coarsest(scale, reals, 'rec')
                        """ Compute the standard deviation of noise """
                        RMSE = tf.sqrt(tf.reduce_mean(tf.square(real - prev_rec)))
                        noise_amp = self.noise_amp_init * RMSE
                else:
                    prev_rand = self.generate_from_coarsest(scale, reals, 'rand')

                Z_rand = z_rand if scale == 0 else noise_amp * z_rand
                Z_rec = noise_amp * z_rec
                
                if i < 3:
                    with tf.GradientTape() as tape:
                        """ Only record the training variables """
                        fake_rand = self.generators[scale](prev_rand, Z_rand)

                        dis_loss = self.dicriminator_wgan_loss(self.discriminators[scale], real, fake_rand, 1)
    
                    dis_gradients = tape.gradient(dis_loss, self.discriminators[scale].trainable_variables)
                    d_opt.apply_gradients(zip(dis_gradients, self.discriminators[scale].trainable_variables))
                else:
                    with tf.GradientTape() as tape:
                        """ Only record the training variables """
                        fake_rand = self.generators[scale](prev_rand, Z_rand)
                        fake_rec = self.generators[scale](prev_rec, Z_rec)

                        gen_loss = self.generator_wgan_loss(self.discriminators[scale], fake_rand)
                        rec_loss = self.reconstruction_loss(real, fake_rec)
                        gen_loss = gen_loss + 10 * rec_loss

                    gen_gradients = tape.gradient(gen_loss, self.generators[scale].trainable_variables)
                    g_opt.apply_gradients(zip(gen_gradients, self.generators[scale].trainable_variables))


            metrics = (dis_loss, gen_loss, rec_loss)
            return z_rec, prev_rec, noise_amp, metrics
        return train_step


    def generate_from_coarsest(self, scale, reals, mode='rand'):
        """ Use random/fixed noise to generate from coarsest scale"""
        fake = tf.zeros_like(reals[0])
        if scale > 0:
            if mode == 'rand':
                for i in range(scale):
                    z_rand = tf.random.normal(reals[i].shape)
                    z_rand = self.NoiseAmp[i] * z_rand
                    fake = self.generators[i](fake, z_rand)
                    fake = imresize(fake, new_shapes=reals[i+1].shape)
    
            if mode == 'rec':
                for i in range(scale):
                    z_fixed = self.NoiseAmp[i] * self.Z_fixed[i]
                    fake = self.generators[i](fake, z_fixed)
                    fake = imresize(fake, new_shapes=reals[i+1].shape)
        return fake


    def create_real_pyramid(self, real_image):
        """ Create the pyramid of scales """
        reals = [real_image]
        for i in range(1, self.num_scales):
            reals.append(imresize(real_image, min_size=self.min_size, scale_factor=pow(0.75, i)))

        """ Reverse it to coarse-fine scales """
        reals.reverse()
        for real in reals:
            print(real.shape)
        return reals


    def generator_wgan_loss(self, discriminator, fake):
        """ Ladv(G) = -E[D(fake)] """
        return -tf.reduce_mean(discriminator(fake))


    def reconstruction_loss(self, real, fake_rec):
        """ Lrec = || G(z*) - real ||^2 """
        return tf.reduce_mean(tf.square(fake_rec - real))

 
    def dicriminator_wgan_loss(self, discriminator, real, fake, batch_size=1):
        """ Ladv(D) = E[D(fake)] - E[D(real)] + GradientPenalty"""
        dis_loss = tf.reduce_mean(discriminator(fake)) - tf.reduce_mean(discriminator(real))

        alpha = tf.random.uniform(shape=[batch_size,1,1,1], minval=0., maxval=1.)# real.shape
        interpolates = alpha * real + ((1 - alpha) * fake)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            dis_interpolates = discriminator(interpolates)
        gradients = tape.gradient(dis_interpolates, [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[3])) # compute pixelwise gradient norm; per image use [1, 2, 3]
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        dis_loss = dis_loss + 0.1 * gradient_penalty
        return dis_loss


    def create_metrics(self):
        self.dis_metric = Mean()
        self.gen_metric = Mean()
        self.rec_metric = Mean()
    

    def update_metrics(self, metrics, step):
        dis_loss, gen_loss, rec_loss = metrics

        self.dis_metric(dis_loss)
        self.gen_metric(gen_loss)
        self.rec_metric(rec_loss)

        print(f' dis_loss = {self.dis_metric.result():.3f}')
        print(f' gen_loss = {self.gen_metric.result():.3f}')
        print(f' rec_loss = {self.rec_metric.result():.3f}')

        self.dis_metric.reset_states()
        self.gen_metric.reset_states()
        self.rec_metric.reset_states()

    
    def create_summary_writer(self):
        import datetime
        self.summary_writer = tf.summary.create_file_writer(
            'log/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


    def write_summaries(self, metrics, scale):
        dis_loss, gen_loss, rec_loss = metrics

        with self.summary_writer.as_default():
            tf.summary.scalar('dis_loss', dis_loss, step=scale)
            tf.summary.scalar('gen_loss', gen_loss, step=scale)
            tf.summary.scalar('rec_loss', rec_loss, step=scale)
            # tf.summary.scalar('PSNR', psnr, step=epoch)