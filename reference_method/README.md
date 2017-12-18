To reproduce result, loading the pretrain model by following step.

1. Run the `skipthought_downloader.py` to download the dependency of skipthought.
2. Create a `sample_captions.txt` in the `./Data` and write one or multiple captions in that file.
3. Run `generate_thought_vectors.py` it will generate a `sample_caption_vectors.hdf5` in the `./Data`  
**Congragulation, you have generated your own word vectors**
4. Download the pre-trained model from : 
	https://drive.google.com/drive/folders/1HsyfwdCQsmRWIIZGtYSm1EWf3d91joAC?usp=sharing
	and put them into `./Data/Models`

5. Run the `generate_images.py` it will generate image you want.
(use --model_path to change the model you want to load)
