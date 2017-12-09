import glob
import os
import numpy as np
from os.path import basename, splitext, join, dirname
import json
import math
# np.random.seed(123)  # for reproducibility

from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Reshape, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
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
conv_layer = 3
first_depth = 10
dropRate = 0.3
text_compress = 256

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
x = Activation("relu")(x)

inp = Concatenate()([x, inp_noise])
x = Dense(units = text_compress)(inp)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Dense(units = (image_dim[0] * first_depth * int(image_dim[1]/pow(2, conv_layer)) * int(image_dim[2]/pow(2, conv_layer))))(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(rate = dropRate)(x)
x = Reshape((image_dim[0] * first_depth, int(image_dim[1]/pow(2, conv_layer)), int(image_dim[2]/pow(2, conv_layer))))(x)
for i in range(conv_layer):
    x = Conv2D(filters = int(image_dim[0] * first_depth * pow(2, i)), kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'gen_conv_' + str(i + 1), data_format = 'channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2), data_format =  "channels_first")(x)
    x = Dropout(rate = dropRate)(x)

x = Conv2D(filters = image_dim[0], kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'gen_conv_' + str(conv_layer + 1), data_format = 'channels_first')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Activation('tanh')(x)
gen_model = Model(inputs = [inp_code, inp_noise], outputs = x)
gen_model.summary()
input('gen')
# Save model structure
model_architecture = gen_model.to_json()

with open(models_dir+'/gen_model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)


# Discriminator
inp_code = Input(shape = (code_dim, ))
c = Dense(units = 256)(inp_code)
c = BatchNormalization()(c)
c = Activation("relu")(c)
c = Dropout(rate = dropRate)(c)


inp_image = Input(shape = image_dim)
x = inp_image
for i in range(conv_layer):
    x = Conv2D(filters = first_depth * pow(2, (i)), kernel_size = (3, 3), strides = (1, 1), padding = 'same', name = 'dist_conv_' + str(i + 1), data_format =  "channels_first")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(x)
    x = Dropout(rate = dropRate)(x)

x = Flatten()(x)
x = Dense(units = 512)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(rate = dropRate)(x)

x = Concatenate()([x, c])

x = Dense(units = 256)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(rate = dropRate)(x)

disc = Dense(units = 2)(x)
disc = Activation("softmax")(disc)


disc_model = Model(inputs = [inp_image, inp_code], outputs = disc)
# Save model structure
model_architecture = disc_model.to_json()
disc_model.summary()
input('disc')

with open(models_dir+'/disc_model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)

