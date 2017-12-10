import numpy as np
from os.path import basename, splitext, join
import json
import math
import os
import h5py

#keras import
import keras

from keras.models import model_from_json
from keras.layers.wrappers import TimeDistributed
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback

import matplotlib.pylab as plt

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models")
feats_dir = join(project_dir, "feats")


def get_vector_data(filename):
    with h5py.File(join(corpus_dir, filename), "r") as f:
        vector_data = f['vectors'][()]
    return vector_data 

def get_img_data(filename):
    with h5py.File(join(corpus_dir, filename), "r") as f:
        img_data = f['imgs'][()]
    return img_data

def get_gen_batch(wordVector, imgNum, noise_dim):
    c = wordVector
    word_batch = [c] * imgNum
    word_batch = np.asarray(word_batch)
    print('word_batch shape: ', word_batch.shape)
    noise = np.random.uniform(size = (imgNum, noise_dim))
    print(noise.shape)
    return [word_batch, noise]

def display(imgArr):
    plt.figure()
    # imgArr = imgArr.reshape(imgArr.shape[0], imgArr.shape[2], imgArr.shape[3], imgArr.shape[1])
    imgNum = imgArr.shape[0]
    print(imgArr.shape)
    for i in range(imgNum):
        img = plt.subplot(imgNum, 1, i + 1)
        img.imshow(imgArr[i, :, :, :].transpose(1,2,0))
    plt.savefig(join(result_dir, "generate_image.png"))
    plt.show()

if __name__ == '__main__':
    
    # parameter
    noise_dim = 100
    imgNum = 4

    # model and weight name
    modelName ='infoGAN_171209_175038'
    weightName = 'gen_weight_90'
    models_dir = join(models_dir, modelName)

    # load variable (change with different model)
    vectorFileName = 'test_vectors.hdf5'

    model_structure = join(models_dir, 'gen_model_structure')

    model_weight_path = join(models_dir, weightName)

    # load model structure and weight
    with open(model_structure) as json_file:
        model_architecture = json.load(json_file)

    gen_model = model_from_json(model_architecture)

    gen_model.load_weights(model_weight_path, by_name=False)
    # load word vector file
    label_data = get_vector_data(vectorFileName)
    print(label_data.shape)
    word_vec = label_data[20]
    
    # generator produce image vectors
    img_vector = gen_model.predict(get_gen_batch(word_vec, imgNum, noise_dim))
    print(img_vector.shape)
    print(img_vector[0,:,:,:])
    display(img_vector)
    

    
    
