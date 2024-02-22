import pyana
import firtez_dz as frz 
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 

import sys


inputfile = sys.argv[1]

atmos = frz.read_model(inputfile)

h = atmos.z[0,0] * 1E5 # to cm

NX, NY, NZ = atmos.z.shape
print ('info:: atmosphere dimensions: ', NX, NY, NZ)

snapiatmos = np.zeros([12, NZ])

snapiatmos[0,:] = np.linspace(1,-7,NZ)
snapiatmos[1,:] = h[:]

snapiatmos[2,:] = atmos.tem
snapiatmos[3,:] = atmos.pg
snapiatmos[9,:] = -atmos.vz

snapiatmos = snapiatmos[:,::-1]

np.savetxt(inputfile[:-4]+'.dat',snapiatmos.T, fmt="%1.5e", header=str(NZ)+' 1dfirteztosnapi', comments='')
