import pyana
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits 
import sys 

ltefile = sys.argv[1]
nltefile = sys.argv[2]

lte = pyana.fzread(ltefile)["data"]
nlte = pyana.fzread(nltefile)["data"]

print (lte.shape)
print (nlte.shape)

dc = nlte[0,0] / lte[0,0]

hyd = 'HI1   HI2   HI3   HI4   HI5   HII'
mag = '  MgI1   MgI2   MgI3   MgI4   MgI5   MgI6   MgII   MgIII'
calc = '  CaII1   CaII2   CaII3   CaII4   CaII5   CaIII'
iron = '  FeI1   FeI2   FeI3   FeI4   FeII   FeIII'
sod = '  NaI1   NaI2   NaI3   NaI4   NaI5   NaII   NaIII'



np.savetxt(sys.argv[3],dc[::-1,:],fmt="%1.5e", header = hyd + mag + calc + iron + sod + 'e', comments='')

