import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits

import muram as mio

def muram_binary_loader(path, iter, ranges=[]):

	# Loads the binary files from a muram file, and puts them into a suitable numpy array 
	# to feed into lw

	T = mio.MuramCube(path, iter, 'T')
	T = T.transpose(2,1,0)

	print ("info::muram_binary_loader::the original dimensions are: ", T.shape)

	if (len(ranges) == 0):
		xmin = 0
		xmax = T.shape[0]
		ymin = 0
		ymax = T.shape[1]
		zmin = 0
		zmax = T.shape[2]
	else if (len(ranges) == 6):
		xmin = ranges[0]
		xmax = ranges[1]
		ymin = ranges[2]
		ymax = ranges[3]
		zmin = ranges[4]
		zmax = ranges[5]


	return 0;