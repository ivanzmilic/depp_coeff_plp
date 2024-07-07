import pyana
import firtez_dz as frz 
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 

import sys

atmossir = fits.open(sys.argv[1])[0].data

NP, NZ, NX, NY = atmossir.shape

print (atmossir.shape)

atmossnapi = np.zeros([12,NX,NY,NZ])

atmossir = atmossir.transpose(2,3,0,1)

atmossnapi[0,:,:,:] = atmossir[:,:,0,:] # tau
atmossnapi[1,:,:,:] = atmossir[:,:,8,:]*1E5 # z
atmossnapi[2,:,:,:] = atmossir[:,:,1,:]
atmossnapi[3,:,:,:] = atmossir[:,:,9,:]
atmossnapi[9,:,:,:] = -atmossir[:,:,5,:]

pyana.fzwrite(sys.argv[2], atmossnapi[:,:,:,::-1], 0, 'bla')





