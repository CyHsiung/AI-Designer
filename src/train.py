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
saveModelName = 'infoGAN_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S')

models_dir = join(models_dir, loadModelName)
# parameter
# modify as needed
nEpoch = 1000
code_dim = 2400
noise_dim = 100
image_dim = [3, 64, 64]
batch_size = 32
# train generator n times then train discriminator 1 time
gen_train_ratio = 2

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
'''
gen_model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])

disc_model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])
'''

def get_vector_data(filename):
    with h5py.File(join(corpus_dir, filename), "r") as f:
        vector_data = f['vectors'][()]
    return vector_data 

def get_img_data(filename):
    with h5py.File(join(corpus_dir, filename), "r") as f:
        img_data = f['imgs'][()]
    return img_data

label_data = get_vector_data(vectorFileName)
label_data = label_data[:, :code_dim]
img_data = get_img_data(imgFileName)
# normalize data between 0 ~ 1
#img_data = img_data/255
#print('data shape: ', label_data.shape, img_data.shape)

def get_gen_batch(label_data, batch_size, noise_dim):
    idx = 0
    dataLength = label_data.shape[0]
    while 1:
        if idx + batch_size > dataLength:
            idx = 0
        c = label_data[idx: idx + batch_size, :]
        idx += batch_size
        noise = np.random.uniform(low = -1.0, high = 1.0, size = (batch_size, noise_dim))
        # disc_out = np.random.uniform(low=0.7, high=1.2, size = (batch_size, 1))
        disc_out = np.ones((batch_size, 1))
        yield ([c, noise], [disc_out])

def get_disc_batch(img_data, label_data, gen_model, batch_size, code_dim, noise_dim):
    idx = 0
    dataLength = img_data.shape[0]
    while 1:
        if idx + batch_size >= dataLength:
            idx = 0
        # get image and label data
        image = img_data[idx : idx + batch_size, :, :, :]
        label = label_data[idx: idx + batch_size, :code_dim]
        idx += batch_size
        # generate noise
        noise = np.random.uniform(low = -1.0, high = 1.0, size = (batch_size, noise_dim))
        code = label
        # generate image
        global graph
        with graph.as_default():
            imgFake = gen_model.predict([code, noise])
        # concatenate fake and real image into a batch
        x_train = np.concatenate((imgFake, image), axis = 0)
        
        # random pick batch_size's random image
        idxList = np.random.randint(dataLength, size = batch_size)
        imgWrong = np.asarray([img_data[i, :, :, :] for i in idxList])
        x_train = np.concatenate((x_train, imgWrong), axis = 0)
        # code = np.concatenate((label, label), axis = 0)
        code = np.concatenate((label, label, label), axis = 0)
        disc = np.zeros((3 * batch_size, 1))
        # denote real image with correct word vector as 1
        disc[batch_size :2 * batch_size, 0] = 1

        # shuffle the order of input
        idxList = np.arange(3 * batch_size)
        np.random.shuffle(idxList)
        x_train = np.asarray([x_train[i, :, :, :] for i in idxList])
        code = np.asarray([code[i, :] for i in idxList])
        disc = np.asarray([disc[i, :] for i in idxList])
        yield([x_train, code], [disc]) 

        
def train_gen_model(gen_model, disc_model, code_dim, noise_dim):
    inp_code = Input(shape = (code_dim,))
    inp_noise = Input(shape = (noise_dim, ))
    x = gen_model([inp_code, inp_noise])
    AAA = disc_model([x, inp_code])
    disc_out  = disc_model([x, inp_code])

    trainGenModel = Model(inputs=[inp_code, inp_noise],
            outputs = disc_out)
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
#acc_highest = 0
#print(acc_highest)
train_gen_model = train_gen_model(gen_model, disc_model, code_dim, noise_dim)
# cosine similarity
def cos_sim(y_true, y_pred):
    dot = K.sum(y_true * y_pred, axis = 1)
    u = K.sqrt(K.sum(K.square(y_true), axis = 1))
    v = K.sqrt(K.sum(K.square(y_pred), axis = 1))
    return 1 - dot / (u * v + 0.0001)

# loss for disc and code output
list_losses = ['mean_squared_error']
list_weights = [1]

disc_model.trainable = False
train_gen_model.compile(loss = list_losses, 
                        loss_weights = list_weights, 
                        optimizer = 'adam'
                        )
disc_model.trainable = True
disc_model.compile(loss = list_losses, 
                        loss_weights = list_weights, 
                        optimizer = 'adam'
                        )

K.get_session().run(tf.global_variables_initializer())
#img_data: N * img_dim
#label_data: N * caption_vector_len
text_file = open(join(models_dir, "loss.txt"), "w")
minLoss = float('Inf')
graph = tf.get_default_graph()
for i in range(nEpoch):
    print('Epoch: ', i + 1)
    disc_loss = disc_model.fit_generator(get_disc_batch(img_data, label_data, gen_model, batch_size, code_dim, noise_dim), steps_per_epoch = int(label_data.shape[0]/batch_size), epochs = 1) 
    # 0 is total loss
    disc_loss = disc_loss.history['loss']
    disc_model.trainable = False
    gen_loss = train_gen_model.fit_generator(get_gen_batch(label_data, batch_size, noise_dim), steps_per_epoch = int(label_data.shape[0]/batch_size), epochs = gen_train_ratio)
    # 0 is total loss
    gen_loss = gen_loss.history['loss']
    disc_model.trainable = True
    graph = tf.get_default_graph() 
    text_file.write('epoch: %d generator_loss: %f discriminator_loss : %f\n' % ( i + 1, gen_loss[0], disc_loss[0]))
    if (i + 1) % 5 == 0:
        gen_model.save_weights(join(models_dir, 'gen_weight_'+ str(i + 1)))
        disc_model.save_weights(join(models_dir, 'disc_weight_' + str(i + 1)))
        print('Save model epoch: ', i + 1)

text_file.close()
