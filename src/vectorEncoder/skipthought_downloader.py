import os
import sys
import errno
import tarfile

if sys.version_info >= (3,):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

DATA_DIR = 'Data'


# http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_data_paths():
    if not os.path.isdir(DATA_DIR):
        raise EnvironmentError('Needs to be run from project directory containing ' + DATA_DIR)
    needed_paths = [
        os.path.join(DATA_DIR, 'samples'),
        os.path.join(DATA_DIR, 'val_samples'),
    ]
    for p in needed_paths:
        make_sure_path_exists(p)

def dl_progress_hook(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()

def download_dataset(data_name):

    if data_name == 'skipthoughts':
        print('== Skipthoughts models ==')
        SKIPTHOUGHTS_DIR = os.path.join(DATA_DIR, 'skipthoughts')
        SKIPTHOUGHTS_BASE_URL = 'http://www.cs.toronto.edu/~rkiros/models/'
        make_sure_path_exists(SKIPTHOUGHTS_DIR)

        # following https://github.com/ryankiros/skip-thoughts#getting-started
        skipthoughts_files = [
            'dictionary.txt', 'utable.npy', 'btable.npy', 'uni_skip.npz', 'uni_skip.npz.pkl', 'bi_skip.npz',
            'bi_skip.npz.pkl',
        ]
        for filename in skipthoughts_files:
            if os.path.exists(os.path.join(SKIPTHOUGHTS_DIR, filename)):
                print('Skip the downloaded file \'{}\''.format(filename))
                continue
            src_url = SKIPTHOUGHTS_BASE_URL + filename
            print('\nDownloading ' + src_url)
            urlretrieve(src_url, os.path.join(SKIPTHOUGHTS_DIR, filename),
                        reporthook=dl_progress_hook)

    elif data_name == 'nltk_punkt':
        import nltk
        print('== NLTK pre-trained Punkt tokenizer for English ==')
        nltk.download('punkt')
    else:
        raise ValueError('Unknown dataset name: ' + data_name)

if __name__ == '__main__':
    create_data_paths()
    # TODO: make configurable via command-line
    download_dataset('skipthoughts')
    download_dataset('nltk_punkt')