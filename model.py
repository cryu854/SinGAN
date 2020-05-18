import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU
from tensorflow.keras.layers import Add, Lambda, ZeroPadding2D
from utils import *


class instance_norm(Layer):
    def __init__(self, epsilon=1e-8):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))
        
        return self.gamma * x + self.beta
        

class ConvBlock(Layer):
    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2D = Conv2D(filters=self.num_filters,
                             kernel_size=3,
                             strides=1,
                             padding='valid',
                             kernel_initializer=self.initializer)
        self.instance_norm = instance_norm()

    def call(self, x):
        x = self.conv2D(x)
        x = self.instance_norm(x)
        x = LeakyReLU(alpha=0.2)(x)

        return x


class Generator(Model):
    def __init__(self, num_filters, name='Generator'):
        super(Generator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.padding = ZeroPadding2D(5)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = Conv2D(filters=3,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           activation='tanh',
                           kernel_initializer=self.initializer)

    def call(self, prev, noise):
        prev_pad = self.padding(prev)
        noise_pad = self.padding(noise)
        x = Add()([prev_pad, noise_pad])
        x = self.head(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.tail(x)
        x = Add()([x, prev])

        return x


class Discriminator(Model):
    def __init__(self, num_filters, name='Discriminator'):
        super(Discriminator, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.head = ConvBlock(num_filters)
        self.convblock1 = ConvBlock(num_filters)
        self.convblock2 = ConvBlock(num_filters)
        self.convblock3 = ConvBlock(num_filters)
        self.tail = Conv2D(filters=1,
                           kernel_size=3,
                           strides=1,
                           padding='valid',
                           kernel_initializer=self.initializer)

    def call(self, x):
        x = self.head(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.tail(x)

        return x