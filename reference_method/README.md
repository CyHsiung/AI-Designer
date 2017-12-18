To reproduce result, load the pre-trained model by the following steps using python2.7

Skip to step 5 if you have generated the skipthought vectors  
1. Make `Data` directory under `reference_method`
2. Run the `skipthought_downloader.py` to download the dependency of skipthought
3. Create a `sample_captions.txt` in `Data` and write one or multiple captions in that file
4. Run `generate_thought_vectors.py` it will generate a `sample_caption_vectors.hdf5` in the `./Data`  
**Congragulation, you have generated your own word vectors**
5. Download the pre-trained model from : 
	https://drive.google.com/drive/folders/1HsyfwdCQsmRWIIZGtYSm1EWf3d91joAC?usp=sharing
	and put them into `Data/Models`

6. Run the `generate_images.py` it will generate image you want.
(use --model_path to change the model you want to load)
