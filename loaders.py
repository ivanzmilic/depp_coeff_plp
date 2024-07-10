import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
import sys

import muram as mio
import muram as muram

def muram_binary_loader_sub(path,iter,ranges,stokes=False):

	# Loads the output of a subsnap of a muram simulation into an array:

	snap=muram.MuramSubSnap(path,iter)

	T = snap.Temp.transpose(1,2,0)
	
	print ("info::muram_binary_loader::the original dimensions are: ", T.shape)

	if (len(ranges) == 0):
		xmin = 0
		xmax = T.shape[0]
		ymin = 0
		ymax = T.shape[1]
		zmin = 0
		zmax = T.shape[2]
	elif (len(ranges) == 6):
		xmin = ranges[0]
		xmax = ranges[1]
		ymin = ranges[2]
		ymax = ranges[3]
		zmin = ranges[4]
		zmax = ranges[5]
	else:
		print("info::muram_binary_loader:: wrong lenght of ranges... returnin zero")
		return 0;

	skip = 2

	T = snap.Temp[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	Tc = np.copy(T)
	Tc[np.where(T<3000.0)] = 3000.0
	Tc[np.where(T>50000.0)] = 50000.0
	z = np.arange(zmax-zmin) * 16E3
	p = snap.Pres[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	vz = snap.vx[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	data = {}
	data["dims"] = np.array([(xmax-xmin)//skip, (ymax-ymin)//skip, zmax-zmin])
	z_3d = np.zeros([(xmax-xmin)//skip, (ymax-ymin)//skip, zmax-zmin])
	z_3d[:,:,:] = z[None,None,:]
	data["z"] = z_3d[:,:,::-1]
	data["T"] = Tc.transpose(1,2,0)[:,:,::-1]
	data["p"] = p.transpose(1,2,0)[:,:,::-1] * 10.0
	data["vz"] = -vz.transpose(1,2,0)[:,:,::-1] / 1E2

	if (stokes):
		Bz = snap.Bx[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
		Bx = snap.By[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
		By = snap.Bz[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	
		B = np.sqrt(Bx**2.0 + By**2.0 + Bz**2.0) * np.sqrt(4.0*np.pi)
		theta = np.arccos(Bz/(B+0.00000001))
		phi = np.arctan(By/Bx)

		data["B"] = B.transpose(2,1,0)[:,:,::-1]
		data["theta"] = theta.transpose(2,1,0)[:,:,::-1]
		data["phi"] = phi.transpose(2,1,0)[:,:,::-1]

	return data;

def muram_binary_loader(path, iter, ranges=[], stokes=False):

	# Loads the binary files from a muram file, and puts them into a suitable numpy array 
	# to feed into lw

	T = mio.MuramCube(path, iter, 'Temp')
	T = T.transpose(2,1,0)

	print ("info::muram_binary_loader::the original dimensions are: ", T.shape)

	if (len(ranges) == 0):
		xmin = 0
		xmax = T.shape[0]
		ymin = 0
		ymax = T.shape[1]
		zmin = 0
		zmax = T.shape[2]
	elif (len(ranges) == 6):
		xmin = ranges[0]
		xmax = ranges[1]
		ymin = ranges[2]
		ymax = ranges[3]
		zmin = ranges[4]
		zmax = ranges[5]
	else:
		print("info::muram_binary_loader:: wrong lenght of ranges... returnin zero")
		return 0;

	skip = 4

	T = mio.MuramCube(path, iter, 'Temp')[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	Tc = np.copy(T)
	Tc[np.where(T<3000.0)] = 3000.0
	Tc[np.where(T>50000.0)] = 50000.0
	z = np.arange(zmax-zmin) * 16E3
	p = mio.MuramCube(path, iter, 'Pres')[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	vz = mio.MuramCube(path, iter, 'vx')[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	data = {}
	data["dims"] = np.array([(xmax-xmin)//skip, (ymax-ymin)//skip, zmax-zmin])
	z_3d = np.zeros([(xmax-xmin)//skip, (ymax-ymin)//skip, zmax-zmin])
	z_3d[:,:,:] = z[None,None,:]
	data["z"] = z_3d[:,:,::-1]
	data["T"] = Tc.transpose(2,1,0)[:,:,::-1]
	data["p"] = p.transpose(2,1,0)[:,:,::-1] * 10.0
	data["vz"] = -vz.transpose(2,1,0)[:,:,::-1] / 1E2

	if (stokes):
		Bz = mio.MuramCube(path, iter, 'Bx')[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
		Bx = mio.MuramCube(path, iter, 'By')[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
		By = mio.MuramCube(path, iter, 'Bz')[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip]
	
		B = np.sqrt(Bx**2.0 + By**2.0 + Bz**2.0) * np.sqrt(4.0*np.pi)
		theta = np.arccos(Bz/(B+0.00000001))
		phi = np.arctan(By/Bx)

		data["B"] = B.transpose(2,1,0)[:,:,::-1]
		data["theta"] = theta.transpose(2,1,0)[:,:,::-1]
		data["phi"] = phi.transpose(2,1,0)[:,:,::-1]

	return data;

def sir_format_loader(filename, stokes=False):

	input_atmos = fits.open(sys.argv[1])[0].data # reads from fits 
	input_atmos = input_atmos.transpose(2,3,0,1)
	data = {}
	data["z"] = input_atmos[:,:,8,::-1] * 1E3 # km to m
	data["T"] = input_atmos[:,:,1,::-1] # K
	data["p"] = input_atmos[:,:,9,::-1] * 10.0
	data["vz"] = input_atmos[:,:,5,::-1] * -1E-2 # cm to m
	print("info::sir_format_loader::stokes value is :", stokes)
	if (stokes):
		data["B"] = input_atmos[:,:,4,::-1] * 1E-5 # G to T
		data["theta"] = input_atmos[:,:,6,::-1] * np.pi/180.0 # inc, TOCHECK
		data["phi"] = input_atmos[:,:,7,::-1] * np.pi/180.0 # azi ,TOCHECK
	return data

def flatten_atmosarr(atmosarr, stokes):

	NX, NY, NZ = atmosarr["T"].shape
	print("info:flatten_atmosarr: original shape is: ", NX, NY, NZ)
	atmosarr["T"] = atmosarr["T"].reshape(NX*NY, NZ)
	atmosarr["p"] = atmosarr["p"].reshape(NX*NY, NZ)
	atmosarr["vz"] = atmosarr["vz"].reshape(NX*NY, NZ)

	if (len(atmosarr["z"].shape) == 3):
		atmosarr["z"] = atmosarr["z"].reshape(NX*NY, NZ)

	if (stokes):
		atmosarr["B"] = atmosarr["B"].reshape(NX*NY, NZ)
		atmosarr["theta"] = atmosarr["theta"].reshape(NX*NY, NZ)
		atmosarr["phi"] = atmosarr["phi"].reshape(NX*NY, NZ)

	return atmosarr
