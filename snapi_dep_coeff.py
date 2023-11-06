import pyana
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits

import sys

atmos_file = sys.argv[1]
lte_file = sys.argv[2]
nlte_file = sys.argv[3]

Nt = int(sys.argv[4])

output_file = sys.argv[5]

atmos = pyana.fzread(atmos_file)["data"]
# full T why not
NP, NX, NY, NZ = atmos.shape
NX = NX//Nt

print (Nt, NX, NY, NZ)

T = atmos[2].reshape(Nt,NX,NY,NZ)

del(atmos)

# Load populations

ltepops = pyana.fzread(lte_file)["data"]
nltepops = pyana.fzread(nlte_file)["data"]
print(ltepops.shape)
print(nltepops.shape)

dep_coeffs = (nltepops/ltepops).reshape(Nt, NX, NY, NZ,-1)

kek = fits.PrimaryHDU(dep_coeffs)
bur = fits.ImageHDU(T)

hdulist = fits.HDUList([kek,bur])

hdulist.writeto(output_file, overwrite=True)