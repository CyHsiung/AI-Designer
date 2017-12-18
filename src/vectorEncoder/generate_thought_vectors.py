from __future__ import absolute_import
import os
import sys
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption-file', type=str, required=True,
					   help='caption file path')
	parser.add_argument('--output-file', type=str, required=True,
					   help='output file path')
		

	args = parser.parse_args()
	if os.path.exists(args.output_file):
		print('{} already exists'.format(args.output_file), file=sys.stderr)
		return

	with open( args.caption_file ) as f:
		captions = f.read().split('\n')

	captions = [cap for cap in captions if len(cap) > 0]
	print(captions)
	model = skipthoughts.load_model()
	caption_vectors = skipthoughts.encode(model, captions)

	h = h5py.File(args.output_file)
	h.create_dataset('vectors', data=caption_vectors)		
	h.close()

if __name__ == '__main__':
	main()