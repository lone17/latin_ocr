import numpy as np

# Keras Core
from keras.layers import (Input, Dropout, Dense, Flatten, Activation, MaxPool2D,
                          Convolution2D, AveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model
from keras import backend as K
from keras import optimizers

if K.image_data_format() == 'channels_first':
    channel_axis = 1
else:
    channel_axis = -1

def conv2d_bn(x, filters, kernel_size, padding='same', strides=(1, 1),
              use_bias=False):

    x = Convolution2D(filters, kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      )(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('elu')(x)
    return x

def conv2d_factor(x, filters, kernel_size, padding='same', strides=(1, 1),
                  use_bias=False):

    x = Convolution2D(filters, (kernel_size[0],1),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      )(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('elu')(x)
    x = Convolution2D(filters, (1, kernel_size[1]),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      )(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('elu')(x)


    return x

def block_a(input, filters, pool_size, pool_strides=None, pool_padding='valid'):
    a1 = conv2d_bn(input, filters, (5,5), padding='same')
    a2 = conv2d_bn(input, filters, (5,7), padding='same')
    a3 = conv2d_bn(input, filters, (5,11), padding='same')

    x = concatenate([a1, a2, a3], axis=channel_axis)
    x = MaxPool2D(pool_size, strides=pool_strides, padding=pool_padding)(x)

    return x

def block_b(input, filters, pool_size, pool_strides=None, pool_padding='valid'):
    a0 = conv2d_bn(input, filters, (1,1))

    a1 = conv2d_bn(a0, filters, (5,5), padding='same')
    a2 = conv2d_bn(a0, filters, (5,7), padding='same')
    a3 = conv2d_bn(a0, filters, (5,11), padding='same')

    x = concatenate([a0, a1, a2, a3], axis=channel_axis)
    x = MaxPool2D(pool_size, strides=pool_strides, padding=pool_padding)(x)

    return x
