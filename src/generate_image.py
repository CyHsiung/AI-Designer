import numpy as np
from os.path import basename, join
import json
import math
import os
import h5py
import argparse
from skimage.io import imsave

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

NOISE_DIM = 100
CODE_DIM = 4800

def get_vector_data(path):
    with h5py.File(path, "r") as f:
        vector_data = f['vectors'][()]
    return vector_data 

def get_img_data(path):
    with h5py.File(path, "r") as f:
        img_data = f['imgs'][()]
    return img_data

def get_gen_batch(vector, n, noise_dim):
    word_batch = np.array([vector] * n)
    noise = np.random.uniform(size = (n, noise_dim))
    return [word_batch, noise]

if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',
                        type=str,
                        required=True)
    parser.add_argument('--epochs',
                        type=int,
                        nargs='+',
                        required=True)
    parser.add_argument('--input-vector',
                        type=str,
                        required=True)
    parser.add_argument('--num-images-per-vector',
                        type=int,
                        default=4)
    parser.add_argument('--output-path',
                        type=str,
                        default='')

    args = parser.parse_args()
    model_path = args.model_path
    epochs = args.epochs
    input_vector_path = args.input_vector
    num_img = args.num_images_per_vector
    output_path = args.output_path

    model_name = basename(model_path)

    # load word vector
    label_data = get_vector_data(input_vector_path)

    # load model structure and weight
    model_structure_path = join(model_path, 'gen_model_structure')
    with open(model_structure_path) as json_file:
        model_structure = json.load(json_file)
    gen_model = model_from_json(model_structure)

    for epoch in epochs:
        model_weight_path = join(model_path, 'gen_weight_{}'.format(epoch))
        gen_model.load_weights(model_weight_path, by_name=False)
        imgs_comb = []
        for i, word_vec in enumerate(label_data):
            img_tensor = gen_model.predict(get_gen_batch(word_vec, num_img, NOISE_DIM))
            img_split = np.split(img_tensor.transpose(0, 2, 3, 1), num_img, axis=0)
            imgs_comb.append(np.squeeze(np.concatenate(img_split, axis=2)))
        np.concatenate(imgs_comb, axis=0)
        imsave(join(output_path, '{}_epoch{}.png'.format(model_name, epoch)), np.concatenate(imgs_comb, axis=0))

