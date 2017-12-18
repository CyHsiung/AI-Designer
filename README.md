# EECS545 F17 Final Project

### Prepare word vectors from tags
* Make drirectory `corpus` under the project root directory
* Put `tags_clean.csv` under `corpus`
* Run `sed -e 's/^[0-9][0-9]*,//g' -e 's/:[0-9][0-9]*//g' -e 's/[[:space:]]/ /g' <tags_clean.csv >training_captions.txt` to generate `training_captions.txt`
* Run `python skipthought_downloader.py` in `src/vectorEncoder` to download and construct the path of data
* Run `python generate_thought_vectors.py --caption-file <project_root>/corpus/training_captions.txt --output-file <project_root>/corpus/train_vectors.hdf5` to generate training vectors.

### Preprocess images for training
* Untar `faces.tar.gz` under `corpus`
* Run `python preprocess_img.py` under `src` to generate `train_imgs.hdf5`

### Generate model structure and train the model
* Make drirectory `models` under the project root directory
* Run `python model_generator.py` under `src` to generate model
* Run `python train.py` under `src` to train the model

### Generate images with custom captions using the trained model
* Put trained model under `models` directory
* Create `custom_captions.txt` and write several captions to that file
* Run `python generate_thought_vectors.py --caption-file <path-to-caption> --output-file <project_root>/corpus/custom_caption_vectors.hdf5` to generate vectors corresponding to the captions in `custom_captions.txt`
* Make directory `result` under project root and run the following command under `src` to generate images
```
python generate_image.py \
--model-path ../models/<model_dir> \
--epochs 20 75 370 600\
--input-vector ../corpus/custom_caption_vectors.hdf5 \
--num-images-per-vector 5 \
--output-path ../result
```
* Follow the instructions in `reference_method/README.md` to generate reference images

### Dependencies
Python 3
See `requirements.txt` for used packages

[Dataset](https://drive.google.com/drive/folders/0BwJmB7alR-AvMHEtczZZN0EtdzQ)  
[Trained Models](https://drive.google.com/open?id=1Y5x9bMCg6ao22l-DKtnG6DXlDO0p0az2)