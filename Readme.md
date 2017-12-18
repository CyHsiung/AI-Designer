EECS545 F17 Final Project
=========================

### Prepare word vectors from tags
* Make drirectory `corpus` under the project root directory

### Preprocess image for training
* Untar `faces.tar.gz` under `corpus`
* Run `preprocess_img.py` under `src` to generate `train_imgs.hdf5`

### Generate model structure and train the model
* Run `model_generator.py` under `src` to generate model
* Run `train.py` under `src` to train the model

[Dataset](https://drive.google.com/drive/folders/0BwJmB7alR-AvMHEtczZZN0EtdzQ)