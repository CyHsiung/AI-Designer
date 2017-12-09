import numpy as np
from os.path import basename, splitext, join
import json
import math
import os
import h5py
import datetime

#from getData_xyz import read_train_file, read_test_file
#from test import test_model, test_model_prior, ROC_score


import tensorflow as tf
#keras import
import keras

from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Reshape, BatchNormalization, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
import keras.backend as K

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")

vectorFileName = 'test_vectors.hdf5'
imgFileName = 'test_imgs.hdf5'
#trainFileName = 'label_train_data_train.csv'
#testFileName = 'label_train_data_valid.csv'

loadWeight = False

loadModelName = 'infoGAN'
saveModelName = 'infoGAN_' + datetime.datetime.now().strftime('%y-%m-%dT%H:%M:%S')

models_dir = join(models_dir, loadModelName)
# parameter
# modify as needed
nEpoch = 100
code_dim = 4800
noise_dim = 100
image_dim = [3, 96, 96]
batch_size = 32
# train generator n times then train discriminator 1 time
gen_train_ratio = 10

gen_model_structure = join(models_dir, 'gen_model_structure')
disc_model_structure = join(models_dir, 'disc_model_structure')

# model_weight_path = join(models_dir, 'initialize_weight')


# Read file 
#X, Y = read_train_file(trainFileName, freq, secLength)


#trainLength = int(trainRatio * len(X))
#X_train = X
#Y_train = Y

#testList = read_test_file(testFileName, freq, secLength)
# new model directory for new parameter
models_dir = join(project_dir, "models")

# make model directory
os.makedirs(join(models_dir, saveModelName))
models_dir = join(models_dir, saveModelName)

# load generator model structure and weight
with open(gen_model_structure) as json_file:
    model_architecture = json.load(json_file)

gen_model = model_from_json(model_architecture)
# load discriminator model structure and weight
with open(disc_model_structure) as json_file:
    model_architecture = json.load(json_file)

disc_model = model_from_json(model_architecture)

'''
if loadWeight:
    model.load_weights(model_weight_path, by_name = True)
'''

gen_model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])

disc_model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])


def get_vector_data(filename):
    with h5py.File(join(corpus_dir, filename), "r") as f:
        vector_data = f['vectors'][()]
    return vector_data 

def get_img_data(filename):
    with h5py.File(join(corpus_dir, filename), "r") as f:
        img_data = f['imgs'][()]
    return img_data

label_data = get_vector_data(vectorFileName)
img_data = get_img_data(imgFileName)
print(label_data.shape, img_data.shape)

def get_gen_batch(label_data, batch_size, noise_dim):
    idx = 0
    dataLength = label_data.shape[0]
    while 1:
        if idx + batch_size >= dataLength:
            idx = 0
        c = label_data[idx: idx + batch_size, :]
        idx += batch_size
        noise = np.random.uniform(size = (batch_size, noise_dim))
        disc_out = np.zeros((batch_size, 2))
        disc_out[:, 1] = 1
        yield ([c, noise], [disc_out, c])

def get_disc_batch(img_data, label_data, gen_model, batch_size, code_dim, noise_dim):
    idx = 0
    dataLength = img_data.shape[0]
    while 1:
        if idx + batch_size >= dataLength:
            idx = 0
        # get image and label data
        image = img_data[idx : idx + batch_size, :, :, :]
        label = label_data[idx: idx + batch_size, :]
        idx += batch_size
        # generate noise
        noise = np.random.uniform(size = (batch_size, noise_dim))
        code = label
        # generate image
        global graph
        with graph.as_default():
            imgFake = gen_model.predict([code, noise])
        # concatenate fake and real image into a batch
        x_train = np.concatenate((imgFake, image), axis = 0)
        code = np.concatenate((label, label), axis = 0)
        disc = np.zeros((2 * batch_size, 2))
        # fake image label
        disc[:batch_size, 0] = 1
        # real image label
        disc[batch_size :, 1] = 1
        yield(x_train, [disc, code])
        
def train_gen_model(gen_model, disc_model, code_dim, noise_dim):
    inp_code = Input(shape = (code_dim,))
    inp_noise = Input(shape = (noise_dim, ))
    x = gen_model([inp_code, inp_noise])
    disc_out, code_out = disc_model(x)

    trainGenModel = Model(inputs=[inp_code, inp_noise],
            outputs=[disc_out, code_out])
    return trainGenModel  
        
    
# model.get_weights()
# model.summary()

# Save generator model structure
model_architecture = gen_model.to_json()

with open(models_dir+'/gen_model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)

# Save generator model structure
model_architecture = disc_model.to_json()

with open(models_dir+'/disc_model_structure', 'w') as outfile:
    json.dump(model_architecture, outfile)


# acc_highest = prior[0]
acc_highest = 0
print(acc_highest)
train_gen_model = train_gen_model(gen_model, disc_model, code_dim, noise_dim)
# cosine similarity
def cos_sim(y_true, y_pred):
    dot = K.sum(y_true * y_pred, axis = 1)
    u = K.sqrt(K.sum(K.square(y_true), axis = 1))
    v = K.sqrt(K.sum(K.square(y_pred), axis = 1))
    return 1 - dot / (u * v + 0.0001)

# loss for disc and code output
list_losses = ['binary_crossentropy', cos_sim]
list_weights = [1, 1]

disc_model.trainable = False
train_gen_model.compile(loss = list_losses, 
                        loss_weights = list_weights, 
                        optimizer = 'adam' # using the Adam optimiser)
                        )
disc_model.trainable = True
disc_model.compile(loss = list_losses, 
                        loss_weights = list_weights, 
                        optimizer = 'adam' # using the Adam optimiser)
                        )

#img_data: N * img_dim
#label_data: N * caption_vector_len
minLoss = float('Inf')
graph = tf.get_default_graph()
for i in range(nEpoch):
    disc_loss = disc_model.fit_generator(get_disc_batch(img_data, label_data, gen_model, batch_size, code_dim, noise_dim), steps_per_epoch = int(label_data.shape[0]/batch_size), epochs = 1) 
    # 0 is total loss
    disc_loss = disc_loss.history['loss']
    disc_model.trainable = False
    gen_loss = train_gen_model.fit_generator(get_gen_batch(label_data, batch_size, noise_dim), steps_per_epoch = int(label_data.shape[0]/batch_size), epochs = gen_train_ratio)
    # 0 is total loss
    gen_loss = gen_loss.history['loss'] 
    disc_model.trainable = True
    graph = tf.get_default_graph()
    
    if (i + 1) % 5 == 0:
        gen_model.save_weights(join(models_dir, 'gen_weight_'+ str(i + 1)))
        disc_model.save_weights(join(models_dir, 'disc_weight_' + str(i + 1)))
        print('Save model epoch: ', i + 1)


