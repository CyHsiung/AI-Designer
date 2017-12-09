import numpy as np
import h5py
from os.path import join
import glob
import natsort
import imageio

IMG_DIR = '../corpus/faces'

def gen_train_imgs():
    img_file_list = glob.glob(join(IMG_DIR, '*.jpg'))
    img_file_list.sort(key=natsort.natsort_keygen())
    img_data = np.empty((len(img_file_list), 3, 96, 96))

    for i,img_file in enumerate(img_file_list):
        if i%100 == 0:
            print('Processing', img_file)
        img = imageio.imread(img_file)/255
        img_data[i,...] = img.transpose(2, 0, 1)

    with h5py.File('../corpus/train_imgs.hdf5', 'w') as f:
        f.create_dataset('imgs', data=img_data)

def test_train_imgs():
    with h5py.File('../corpus/train_imgs.hdf5', 'r') as f:
        dset = f['imgs']
        print(dset)

if __name__ == '__main__':
    print('Preparing to generate training imgs...')
    gen_train_imgs()
    test_train_imgs()
