import os
import numpy as np
import tensorflow as tf
from PIL import Image


# -----------------------------------------------------------
#  OS
# -----------------------------------------------------------


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')  

    return dir


# -----------------------------------------------------------
#  Solve
# -----------------------------------------------------------


def load_image(image, image_size=None):
    """Load an image from directory into a tensor shape of [1,H,W,C] and value between [0, 255]
    image : Directory of image
    image_size : An integer number
    """
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    # image = tf.image.convert_image_dtype(image, tf.float32)   # to [0, 1]

    if image_size:
        image = tf.image.resize(image, (image_size, image_size),
                                method=tf.image.ResizeMethod.BILINEAR,
                                antialias=True,
                                preserve_aspect_ratio=True
                                )
    return image[tf.newaxis, ...]


def imresize(image, min_size=0, scale_factor=None, new_shapes=None):
    """ Expect input shapes [B, H, W, C] """
    if new_shapes:
        new_height = new_shapes[1]
        new_width = new_shapes[2]

    elif scale_factor:
        new_height = tf.maximum(min_size, 
                                tf.cast(image.shape[1]*scale_factor, tf.int32))
        new_width = tf.maximum(min_size, 
                               tf.cast(image.shape[2]*scale_factor, tf.int32))

    image = tf.image.resize(
                image, 
                (new_height, new_width),
                method=tf.image.ResizeMethod.BILINEAR,
                antialias=True
            )
    return image


def imsave(image, path):
    """ Expected input values [-1, 1] """
    image = denormalize_m11(image)
    image = clip_0_255(image)
    image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
    image.save(path)


# -----------------------------------------------------------
#  Processing
# -----------------------------------------------------------


def normalize_01(x):
    """ Normalizes RGB images to [0, 1]"""
    return x / 255.0

def normalize_m11(x):
    """ Normalizes RGB images to [-1, 1] """
    return x / 127.5 - 1 

def denormalize_m11(x):
    """ Inverse of normalize_m11 """
    return (x + 1) * 127.5

def clip_0_255(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)