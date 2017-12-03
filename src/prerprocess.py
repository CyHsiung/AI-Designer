import numpy as np
import h5py
from os.path import join
import scipy.ndimage

N = 100
IMG_SHAPE = (3,96,96)
IMG_DATA_SHAPE = tuple([N]+list(IMG_SHAPE))

def gen_test_vectors():
    fs = h5py.File('../corpus/data2_vectors.hdf5', 'r')
    src = fs['vectors']
    fd = h5py.File('../corpus/test_vectors.hdf5', 'w')
    fd.create_dataset('vectors', data=src[:N])
    fs.close()
    fd.close()

def gen_test_imgs():
    img_data = np.empty(IMG_DATA_SHAPE, dtype=np.uint8)
    with open('../corpus/avalist.txt','r') as f:
        for i in range(N):
            line = f.readline().strip()
            img = scipy.ndimage.imread(join('../corpus/faces', line))
            img_data[i,...] = img.transpose(2, 0, 1)
    with h5py.File('../corpus/test_imgs.hdf5', 'w') as f:
        f.create_dataset('imgs', data=img_data)

def test_read():
    with h5py.File('../corpus/img_data.hdf5', 'r') as f:
        dset = f['imgs']

def main():
    gen_test_vectors()
    gen_test_imgs()
    #test_read()

if __name__ == '__main__':
    main()