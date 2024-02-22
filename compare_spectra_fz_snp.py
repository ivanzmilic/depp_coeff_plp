import firtez_dz as frz 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 

snapifile = sys.argv[1]
firtezfile = sys.argv[2]
output = sys.argv[3]

snapi = np.loadtxt(snapifile,unpack=True)

fs = frz.read_profile(firtezfile)

fi = fs.stki[:,0,0]
fl = fs.wave

flfe = fl[:21]
fsfe = fi[:21]
print(flfe[0],flfe[-1])
flfe = flfe / 1000.0 + 6302.49

flca = fl[21:342]
fsca = fi[21:342]
print(flca[0],flca[-1])
flca = flca /1000.0 + 8542.08

flna = fl[342:503]
fsna = fi[342:503]
print(flna[0],flna[-1])
flna = flna/1000.0 + 5895.9

flmg = fl[503:]
fsmg = fi[503:]
print(flmg[0],flmg[-1])
flmg = flmg / 1000.0 + 5172.67

plt.figure(figsize=[12,8])
plt.subplot(221)
plt.plot(snapi[0]*1E8,snapi[1],label='snapi')
plt.plot(flfe, fsfe, 'o', label='firtez')
plt.legend()
plt.xlim([6301.0,6303.0])

plt.subplot(222)
plt.plot(snapi[0]*1E8,snapi[1],label='snapi')
plt.plot(flca, fsca, 'o', label='firtez')
plt.legend()
plt.xlim([8537.0,8547.0])

plt.subplot(223)
plt.plot(snapi[0]*1E8,snapi[1],label='snapi')
plt.plot(flna, fsna, 'o', label='firtez')
plt.legend()
plt.xlim([5893.0,5899.0])

plt.subplot(224)
plt.plot(snapi[0]*1E8,snapi[1],label='snapi')
plt.plot(flmg, fsmg, 'o', label='firtez')
plt.legend()
plt.xlim([5167.0,5177.0])

plt.tight_layout()
plt.savefig(output, bbox_inches='tight')
