EECS545 F17 Final Project
=========================

### Prepare word vectors from tags
* Make the drirectory `corpus` under the project root directory
* Put `tags_clean.csv` under `corpus`
* Run `some script` to generate `training_captions.txt`
* Run `python skipthought_downloader.py` in `src/vectorEncoder` to download and construct the path of data
* Run `python generate_thought_vectors.py --caption-file <path-to-captions> --output-file <project_root>/corpus/train_vectors.hdf5` to generate training vectors.

### Preprocess images for training
* Untar `faces.tar.gz` under `corpus`
* Run `python preprocess_img.py` under `src` to generate `train_imgs.hdf5`

### Generate model structure and train the model
* Make the drirectory `models` under the project root directory
* Run `python model_generator.py` under `src` to generate model
* Run `python train.py` under `src` to train the model

### Generate images with custom captions using the trained model
* Create `custom_captions.txt` and write several captions to that file
* Run `python generate_thought_vectors.py --caption-file <path-to-caption> --output-file <project_root>/corpus/custom_caption_vectors.hdf5` to generate vectors corresponding to captions in `custom_captions.txt`
* Run the following command to generate images
```
python generate_image.py \
--model-path ../models/<model_dir> \
--epochs 100 200 300 \
--input-vector ../corpus/custom_caption_vectors.hdf5 \
--num-images-per-vector 5 \
--output-path ../corpus
```

### Dependencies
See `requirements.txt`

[Dataset](https://drive.google.com/drive/folders/0BwJmB7alR-AvMHEtczZZN0EtdzQ)