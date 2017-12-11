import glob
import os
import numpy as np
from os.path import basename, splitext, join, dirname
import json
import math
# np.random.seed(123)  # for reproducibility

import keras
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Reshape, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, LeakyReLU
from keras.utils import np_utils

from keras.layers import add
import keras.backend as K

fileName = 'label_data.csv'
modelName = 'infoGAN'

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")

# parameter
# modify as needed
code_dim = 4800
noise_dim = 100
image_dim = [3, 96, 96]
conv_layer = 4
gen_first_channel = 96
gen_first_kernel = 3
disc_first_channel = 64
dropRate = 0.3
text_compress = 256
lrelu_alpha = 0.2

# make model directory
os.makedirs(join(models_dir, modelName))

models_dir = join(models_dir, modelName)

# copy given model_generator.py to model's directory
os.system('cp ./model_generator.py ' + join(models_dir, 'model_generator.py'))


# Generator
inp_code = Input(shape = (code_dim,), name = 'code_input')
inp_noise = Input(shape = (noise_dim, ), name = 'noise_input')

x = Dense(units = text_compress)(inp_code)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = lrelu_alpha)(x)
inp = Concatenate()([x, inp_noise])
x = Dense(units = text_compress)(inp)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = lrelu_alpha)(x)

# first conv input: (batch_size, gen_first_channel, 6, 6)
# 96 -> 6 : image_dim[1]/pow(2, 4)
x = Dense(units = (gen_first_channel * int(image_dim[1]/pow(2, conv_layer)) * int(image_dim[2]/pow(2, conv_layer))))(x)

x = BatchNormalization()(x)
x = LeakyReLU(alpha = lrelu_alpha)(x)
x = Dropout(rate = dropRate)(x)
x = Reshape((gen_first_channel, int(image_dim[1]/pow(2, conv_layer)), int(image_dim[2]/pow(2, conv_layer))))(x)
for i in range(conv_layer - 1):
    out_channel_size = int(gen_first_channel / pow(2, i + 1))
    # cur_kernel_size = int(first_kernel/pow(2, i))
    cur_kernel_size = gen_first_kernel * pow(2, i)
    x = Conv2DTranspose(filters = out_channel_size, kernel_size = (cur_kernel_size, cur_kernel_size), strides = (2, 2), padding = 'same', name = 'gen_conv_' + str(i + 1), data_format = 'channels_first')(x)
    
    # x = Conv2D(filters = int(image_dim[0] * first_depth * pow(2, i)), kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'gen_conv_' + str(i + 1), data_format = 'channels_first')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = lrelu_alpha)(x)
    # x = UpSampling2D((2, 2), data_format =  "channels_first")(x)
    x = Dropout(rate = dropRate)(x)
cur_kernel_size *= 2
x = Conv2DTranspose(filters = image_dim[0], kernel_size = (cur_kernel_size, cur_kernel_size), strides = (2, 2), padding = 'same', name = 'gen_conv_' + str(conv_layer + 1), data_format = 'channels_first')(x)
x = BatchNormalization()(x)
# x = Activation('relu')(x)
x = Activation('sigmoid')(x)
# x = Activation('tanh')(x)
# x = Lambda(lambda x: x * 0.5 + 0.5)(x)
gen_model = Model(inputs = [inp_code, inp_noise], outputs = x)
gen_model.summary()
input('gen')
# Save model structure
model_architecture = gen_model.to_json()

with open(models_dir+'/gen_model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)


# Discriminator
inp_code = Input(shape = (code_dim, ))
c = Dense(units = text_compress)(inp_code)
c = BatchNormalization()(c)
c = LeakyReLU(alpha = lrelu_alpha)(c)
# c = Dropout(rate = dropRate)(c)
# Lambda function make operation possible, if not used -> form keras_tensor back to normal tensor and failed
c = Lambda(lambda x: K.expand_dims(x, axis = 2))(c)
c = Lambda(lambda x: K.expand_dims(x, axis = 3))(c)

def my_tile(x, dim_h, dim_w):
    return K.tile(x, [1, 1, dim_h, dim_w])
c = Lambda(my_tile, arguments = {'dim_h' : int(image_dim[1]/pow(2, conv_layer)), 'dim_w' : int(image_dim[2]/pow(2, conv_layer))})(c)
# c = K.tile(c, [1, 1,int(image_dim[1]/pow(2, conv_layer)),int(image_dim[2]/pow(2, conv_layer))])



inp_image = Input(shape = image_dim)
x = inp_image
for i in range(conv_layer):
    out_channel_size = disc_first_channel * pow(2, i)
    # cur_kernel_size = int(first_kernel/pow(2, 2 - i))
    cur_kernel_size = 5
    x = Conv2D(filters = out_channel_size, kernel_size = (cur_kernel_size, cur_kernel_size), strides = (2, 2), padding = 'same', name = 'dist_conv_' + str(i + 1), data_format =  "channels_first")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = lrelu_alpha)(x)
    x = Dropout(rate = dropRate)(x)

cur_kernel_size = 2
out_channel_size = 128
x = Concatenate(axis = 1)([x, c])
x = Conv2D(filters = out_channel_size, kernel_size = (cur_kernel_size, cur_kernel_size), strides = (2, 2), padding = 'same', name = 'dist_conv_' + str(i + 2), data_format =  "channels_first")(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha = lrelu_alpha)(x)
x = Dropout(rate = dropRate)(x)


x = Flatten()(x)
disc = Dense(units = 1)(x)
# disc = Activation("relu")(disc)


disc_model = Model(inputs = [inp_image, inp_code], outputs = disc)
# Save model structure
model_architecture = disc_model.to_json()
disc_model.summary()
input('disc')

with open(models_dir+'/disc_model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)

