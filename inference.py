import os
import numpy as np
import tensorflow as tf

from model import Generator
from utils import load_image, imsave, imresize, normalize_m11, create_dir


class Inferencer:
    def __init__(self,
                 num_samples,
                 scale_factor,
                 start_scale,
                 result_dir,
                 checkpoint_dir):

        self.model = []
        self.NoiseAmp = []
        self.load_model(checkpoint_dir)
        self.num_samples = num_samples
        self.scale_factor = scale_factor
        self.start_scale = start_scale
        self.result_dir = result_dir


    def load_model(self, checkpoint_dir):
        """ Load generators and NoiseAmp from checkpoint_dir """
        self.NoiseAmp = np.load(checkpoint_dir + '/NoiseAmp.npy')
        dir = os.walk(checkpoint_dir)
        for path, dir_list, _ in dir:
            for dir_name in dir_list:
                network = dir_name[0]
                scale = int(dir_name[1])
                if network == 'G':
                    generator = Generator(num_filters=32*pow(2, (scale//4)))
                    generator.load_weights(os.path.join(path, dir_name) + '/G')
                    self.model.append(generator)


    def inference(self, mode, reference_image, image_size=250):
        """ Use SinGAN to do inference
        mode : Inference mode
        reference_image : Input image name
        image_size : Size of output image
        """
        reference_image = load_image(reference_image, image_size=image_size)
        reference_image = normalize_m11(reference_image)
        reals = self.create_real_pyramid(reference_image, num_scales=len(self.model))

        dir = create_dir(os.path.join(self.result_dir, mode))
        if mode == 'random_sample':
            z_fixed = tf.random.normal(reals[0].shape)
            for n in range(self.num_samples):
                fake = self.SinGAN_generate(reals, z_fixed, start_scale=self.start_scale)
                imsave(fake, dir + f'/random_sample_{n}.jpg') 

        elif (mode == 'harmonization') or (mode == 'editing') or (mode == 'paint2image'):
            fake = self.SinGAN_inject(reals, inject_scale=self.start_scale)
            imsave(fake, dir + f'/inject_at_{self.start_scale}.jpg') 

        else:
            print('Inference mode must be: random_sample, harmonization, paint2image, editing')


    def SinGAN_inject(self, reals, inject_scale=1):
        """ Inject reference image on given scale (inject_scale should > 0)"""
        fake = reals[inject_scale]

        for scale in range(inject_scale, len(reals)):
            fake = imresize(fake, new_shapes=reals[scale].shape)
            z = tf.random.normal(fake.shape)
            z = z * self.NoiseAmp[scale]
            fake = self.model[scale](fake, z)
    
        return fake


    @tf.function
    def SinGAN_generate(self, reals, z_fixed, start_scale=0):
        """ Use fixed noise to generate before start_scale """
        fake = tf.zeros_like(reals[0])
    
        for scale, generator in enumerate(self.model):
            fake = imresize(fake, new_shapes=reals[scale].shape)
            
            if scale > 0:
                z_fixed = tf.zeros_like(fake)

            if scale < start_scale:
                z = z_fixed
            else:
                z = tf.random.normal(fake.shape)

            z = z * self.NoiseAmp[scale]
            fake = generator(fake, z)

        return fake


    def create_real_pyramid(self, real_image, num_scales):
        """ Create the pyramid of scales """
        reals = [real_image]
        for i in range(1, num_scales):
            reals.append(imresize(real_image, scale_factor=pow(self.scale_factor, i)))
        
        """ Reverse it to coarse-fine scales """
        reals.reverse()
        for real in reals:
            print(real.shape)
        return reals