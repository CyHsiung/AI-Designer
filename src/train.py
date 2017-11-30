import numpy as np
from os.path import basename, splitext, join
import json
import math
import os

from getData_xyz import read_train_file, read_test_file
from test import test_model, test_model_prior, ROC_score

#keras import
import keras

from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback
import keras.backend as K

project_dir = "../"

result_dir = join(project_dir, "results")
corpus_dir = join(project_dir, "corpus")
models_dir = join(project_dir, "models_xyz")
feats_dir = join(project_dir, "feats")

trainFileName = 'label_train_data_train.csv'
testFileName = 'label_train_data_valid.csv'

loadWeight = False

loadModelName = 'CNN_dropout_xyz_2'
saveModelName = '1234_sub2'

models_dir = join(models_dir, loadModelName)
# parameter
nEpoch = 400
code_dim = 10
noise_dim = 40
image_dim = [3, 28, 28]

gen_model_structure = join(models_dir, 'gen_model_structure')
disc_model_structure = join(models_dir, 'disc_model_structure')

# model_weight_path = join(models_dir, 'initialize_weight')


# Read file 
X, Y = read_train_file(trainFileName, freq, secLength)

trainLength = int(trainRatio * len(X))
X_train = X
Y_train = Y

testList = read_test_file(testFileName, freq, secLength)
# new model directory for new parameter
models_dir = join(project_dir, "models_xyz")

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

model.compile(loss='mean_squared_error', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy'])

def get_gen_batch(label_data, batch_size, noise_dim):
    idx = 0
    dataLength = label.shape[0]
    while 1:
        if idx + batch_size >= dataLength:
            idx = 0
        c = label_data[idx: idx + batch_size, :]
        idx += batch_size
        noise = np.random.uniform(batch_size, noise_dim)
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
        noise = np.random.uniform(batch_size, noise_dim)
        code = label
        # generate image
        [imgFake] = gen(model.predict([code, noise]))
        # concatenate fake and real image into a batch
        x_train = np.concatenate((imgFake, image), axis = 0)
        code = concatenate((label, label), axis = 0)
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
model.summary()

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

# loss for disc and code output
list_losses = ['binary_crossentropy', 'categorical_crossentropy']
list_weights = [1, 1]

disc_model.trainable = False
train_gen_model.compile(loss = list_losses, 
                        loss_weights = list_weights, 
                        optimizer = 'adam', # using the Adam optimiser)
                        )
disc_model.trainable = True
disc_model.compile(loss = list_losses, 
                        loss_weights = list_weights, 
                        optimizer = 'adam', # using the Adam optimiser)


minLoss = float('Inf')
for i in range(nEpoch):
    disc_model.trainable = False
    gen_loss = train_gen_model.fit_generator(get_gen_batch(label_data, batch_size, noise_dim), steps_per_epoch = int(label_data.shape[0]/batch_size), epochs = 5)
    # 0 is total loss
    gen_loss = gen_loss[0] 
    disc_model.trainable = True
    
    disc_loss = train_gen_model.fit_generator(get_disc_batch(img_data, label_data, gen_model, batch_size, code_dim, noise_dim), steps_per_epoch = int(label_data.shape[0]/batch_size), epochs = 1) 
    # 0 is total loss
    disc_loss = disc_loss[0] 

    if disc_loss + gen_loss < minLoss:
        gen_model.save_weights(join(models_dir, 'gen_weight'))
        disc_model.save_weights(join(models_dir, 'disc_weight'))


