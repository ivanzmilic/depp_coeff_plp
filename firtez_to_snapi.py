import pyana
import firtez_dz as frz 
import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 

import sys

start = int(sys.argv[1])
end = int(sys.argv[2])
NT = end-start+1

name = 'out_valc_it0_tau_'

atmos = frz.read_model(name+str(start)+'.bin')

h = atmos.z[0,0] * 1E5 # to cm

NX, NY, NZ = atmos.z.shape
print ('info:: atmosphere dimensions: ', NT, NX, NY, NZ)

snapiatmos = np.zeros([12, NX*NT, NY, NZ])

snapiatmos[1,:,:,:] = h[None,None,:]

snapiatmos[2,0:NX,:,:] = atmos.tem
snapiatmos[3,0:NX,:,:] = atmos.pg
snapiatmos[9,0:NX,:,:] = -atmos.vz

for i in range(1, NT):

	atmos = frz.read_model(name+str(start+i)+'.bin')

	snapiatmos[2,i*NX:(i+1)*NX,:,:] = atmos.tem
	snapiatmos[3,i*NX:(i+1)*NX,:,:] = atmos.pg
	snapiatmos[9,i*NX:(i+1)*NX,:,:] = -atmos.vz

pyana.fzwrite("out_valc_it0_tau_"+str(start)+'_'+str(end)+'.f0',snapiatmos[:,:,:,::-1],0,'temp')
