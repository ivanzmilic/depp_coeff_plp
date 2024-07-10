from threadpoolctl import threadpool_limits

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
threadpool_limits(1)
import pickle
from enum import IntEnum

import lightweaver as lw
import numpy as np
from lightweaver.rh_atoms import (Al_atom, C_atom, CaII_atom, Fe23_atom, H_6_atom, He_9_atom, MgI_atom, N_atom, Na_atom, O_atom, S_atom, Si_atom)
from myatoms import Fe23_5250
from mpi4py import MPI
from tqdm import tqdm
import sys
threadpool_limits(1)
# NOTE(cmo): Numpy please, I beg you, only create 1 BLAS thread per process. 

# NOTE(cmo): Based on Andres' + Andreu's MPI Lightweaver worker

# various i/o stuff:
from astropy.io import fits

from loaders import sir_format_loader
from loaders import flatten_atmosarr
from loaders import muram_binary_loader
from loaders import muram_binary_loader_sub

# other files

def airtovac(lambda_air):

  s = 1E2/(lambda_air*1E8);
  n = 1.0 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s*s) + 0.0001599740894897 / (38.92568793293 - s*s);
  return lambda_air * n

def synth(atmos, conserve, prd, stokes, wave, mu, actives):
    
    '''
    Synthesize a spectral region with given parameters:
    
    Parameters
    ----------
    atmos : lw.Atmosphere - The atmospheric model in which to synthesise the line.
    
    conserve : bool - Whether to start from LTE electron density and conserve charge, or simply use from the electron density present in the atmospheric model.

    prd: bool - whether to use prd or no, most of the time it's no 

    stokes: bool - whether to synth all 4 Stokes parameters or I only - for tracking we start with I only
    
    wave : np.ndarray Array of wavelengths over which to resynthesise the final line profile

    mu : mu angle, gonna use 1.0 most of the times 

    actives: list of active species to synthesize

    Returns
    -------
    ctx : lw.Context -The Context object that was used to compute the equilibrium populations -> Gonna not return this
    
    Iwave : np.ndarray - The intensity at given mu and wave    '''
    
    # Configure the atmospheric angular quadrature - only matters for NLTE. Gonna use 3 for faster calc
    atmos.quadrature(1)
    # Replace this with atmos.rays ( specify mu ) - let's think how to use Stokes with it
    # ctx.single_stokes_fs
    
    # Configure the set of atomic models to use. Contrary to SNAPI you have to explicitly specify all species
    # Annoying, but since you have all the atoms in the file - that is fine.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe23_5250(), He_9_atom(), MgI_atom(), N_atom(), Na_atom(), S_atom()])
    
    # Set actives to the ones you have inputted.
    aSet.set_active(actives)
    
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # Calculate electron density in lte, we are never using the electron density from the model
    eqPops = aSet.iterate_lte_ne_eq_pops(atmos, direct=True)

    # Configure the Context which holds the state of the simulation for the  backend, and provides 
    # the python interface to the backend.
    # Feel free to increase Nthreads to increase the number of threads the program will use.
    # I would always stick to Nthreads = 1 as we are looking to mpi this one
    
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=1)
    
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver). Go test this one in order to calculate stuff in LTE!
    
    #lw.iterate_ctx_se(ctx, prd=prd)
    
    #lw.iterate_ctx_se(ctx, prd=prd, quiet=True)

    ctx.formal_sol_gamma_matrices()

    
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.
    #eqPops.update_lte_atoms_Hmin_pops(atmos)
    
    # Calculate the (stokes) spectru, at the provided wavegrid at the specified mu
    Iwave = ctx.compute_rays(wave, [mu], stokes=stokes) 

    # We will want to return some populations or so, at some point (Firtez pipeline)
    #return ctx, Iwave

    # For now we are only returning the intensity:
    if (stokes == False):
        Iwave = Iwave.reshape(1,-1)
    
    return Iwave

class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """ # Makes sense to me, but not sure what is the IntEnum class 
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def slice_tasks(atmosin, task_start, grain_size):
    
    task_end = min(task_start + grain_size, atmosin['T'].shape[0])

    #print (task_end)
    
    sl = slice(task_start, task_end) # this is a slice object, allowing us to access the specific thingy
    
    #print (sl)
    data = {}
    data['taskGrainSize'] = task_end - task_start

    if (len(atmosin['z'].shape)==1): # if it is just 1D 
        data['z'] = atmosin['z']
    else:
        data['z'] = atmosin['z'][sl,:]

    #print (data['z']/1E3)
    data['temperature'] = atmosin['T'][sl,:]
    data['pg'] =          atmosin['p'][sl,:]
    data['vlos'] =        atmosin['vz'][sl,:]

    # If we are Stokes, means we have magnetic field too:
    if (stokes):
        data['B'] =   atmosin['B'][sl,:]
        data['inc'] = atmosin['theta'][sl,:]
        data['azi'] = atmosin['phi'][sl,:]

    return data

def overseer_work(atmosarr, wave, stokes, task_grain_size=16):
    """ Function to define the work to do by the overseer """

    # Reshape the atmosphere:
    NX,NY,NZ = atmosarr["T"].shape
    atmosarr = flatten_atmosarr(atmosarr, stokes)
    
    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    data_size = 0 # Let's figure out what this is - total number of pixels?
    num_tasks = 0 # And this is data_size // 16? 
    file_idx_for_task = [] # does this have sth to do with reading from file?
    task_start_idx = [] # no idea
    task_writeback_range = [] # no idea
    
    cdf_size = atmosarr["T"].shape[0]
    print("info::overseer::cdf_size = ", cdf_size)

    num_cdf_tasks = int(np.ceil(cdf_size / task_grain_size)) # number of tasks = roundedup number of pixels / grain
    
    task_start_idx.extend(range(0, cdf_size, task_grain_size))
    
    task_writeback_range.extend([slice(data_size + i*task_grain_size, min(data_size + (i+1)*task_grain_size,
        data_size + cdf_size)) for i in range(num_cdf_tasks)])
    
    data_size = cdf_size
    num_tasks = num_cdf_tasks

    # Define the lists that will store the data of each feature-label pair - I hate lists, can I work with 
    # numpy array 
    spectra = [None] * data_size
    
    success = True
    task_status = [0] * num_tasks

    with tqdm(total=num_tasks, ncols=110) as progress_bar:
        
        # While we don't have more closed workers than total workers keep looping
        while closed_workers < num_workers:
            data_in = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:
                try:
                    task_index = task_status.index(0)
                    
                    # Slice out our task
                    data = slice_tasks(atmosarr, task_start_idx[task_index], task_grain_size)
                    data['index'] = task_index
                    data['wave'] = wave
                    data['stokes'] = stokes

                    # send the data of the task and put the status to 1 (done)
                    comm.send(data, dest=source, tag=tags.START)
                    task_status[task_index] = 1

                # If error, or no work left, kill the worker
                except:
                    comm.send(None, dest=source, tag=tags.EXIT)

            # If the tag is Done, receive the status, the index and all the data
            # and update the progress bar
            elif tag == tags.DONE:
                success = data_in['success']
                task_index = data_in['index']

                if not success:
                    task_status[task_index] = 0
                    print(f"Task: {task_index} failed")
                else:
                    task_writeback = task_writeback_range[task_index]
                    spectra[task_writeback] = data_in['spectrum']
                    progress_bar.update(1)

            # if the worker has the exit tag mark it as closed.
            elif tag == tags.EXIT:
                #print(" * Overseer : worker {0} exited.".format(source))
                closed_workers += 1

    # Once finished, dump all the data
    ns = 1
    if (stokes):
        ns = 4
    spectra = np.asarray(spectra)
    spectra = spectra.reshape(NX, NY, ns,-1)
    print("info::overseer::writing the spectra")
    spechdu = fits.PrimaryHDU(spectra)
    wavhdu = fits.ImageHDU(wave)
    to_output = fits.HDUList([spechdu, wavhdu])
    to_output.writeto(sys.argv[1]+'_lwsynth_'+str(wave[0])+'.fits', overwrite=True)

    
def worker_work(rank):
    # Function to define the work that the workers will do

    while True:
        # Send the overseer the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        # Receive the data with the index of the task, the atmosphere parameters and/or the tag
        data_in = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            # Receive the Radyn atmosphere
            task_index = data_in['index'] # I think we need this? - for what though (to keep track of what succeeeded where)
            z = data_in['z'].astype(float) # Can, in principle, be different
            temperature = data_in['temperature'].astype(float)
            vlos = data_in['vlos'].astype(float)
            pg = data_in['pg'].astype(float)
            task_size = data_in['taskGrainSize']
            wave = data_in['wave'].astype(float)
            ND = temperature.shape[-1]
            stokes = data_in['stokes']

            if (stokes):
                B = data_in['B'].astype(float)
                inc = data_in['inc'].astype(float)
                azi = data_in['azi'].astype(float)

            
            ns = 1
            if (stokes):
                ns = 4

            I = np.zeros([task_size, ns,len(wave)])
            
            for t in range(task_size): # Now, task size is not one, probably a good choice
                # Configure the context
                # Need to loop over task size
                atmos = 0
                # TODO - consider convertScales!!!!
                if (stokes):
                    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=z[t],
                                  temperature=temperature[t],  vlos=vlos[t], vturb=np.zeros(ND), Pgas=pg[t],
                                  B=B[t], gammaB=inc[t], chiB=azi[t], convertScales=False)
                else:
                    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=z[t],
                                  temperature=temperature[t],  vlos=vlos[t], vturb=np.zeros(ND), Pgas=pg[t], 
                                  convertScales=False)

                success = 1
                #try:
                    # This should work
                Itemp = synth(atmos, conserve=False, prd=False, stokes=stokes, wave=airtovac(wave), mu=1.0, actives='Fe')
                #except:
                    # NOTE(cmo): In this instance, the task should never fail
                    # for sane input.
                #    success = 0
                #    break
                I[t,:,:] = np.copy(Itemp)

            # Send the computed data
            # we do want to fill in tau too, but that can wait for the next step
            data_out =  {'index': task_index, 'success': success, 'spectrum': I}
            comm.send(data_out, dest=0, tag=tags.DONE)

        # If the tag is exit break the loop and kill the worker and send the EXIT tag to overseer
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)

if (__name__ == '__main__'):

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    #print(f"Node {rank}/{size} active", flush=True)

    if rank == 0:

        # --------------------------------------------------------------------
        stokes = sys.argv[2].lower() == 'true'
        atmos_format = sys.argv[3]
        atmosarr = 0
        print("info::overseer::stokes mode is: ", stokes)
        if (atmos_format == 'mrmfx'):
            print("info:overseer: opening the atmosphere in simple muram format...")
            atmosarr = load_muram_fixed_format(sys.argv[1],stokes)
            print("info:overseer: ...sucess!")

        elif (atmos_format == 'sir'):
            print("info:overseer: opening the atmosphere in SIR format...")
            atmosarr = sir_format_loader(sys.argv[1],stokes)
            print("info:overseer: ...sucess!")
        elif (atmos_format == 'muramb'):
            print("info:overseer: opening the atmosphere in muram binary format...")
            path = '/mnt/c/Users/ivanz/OneDrive/Documents/SSD_25_8Mm_16_pdmp_1_ISSI_flows'
            atmosarr = muram_binary_loader(path, int(sys.argv[1]), [0,512,0,512,380,480], stokes)
            print("info:overseer: ...sucess!")
        elif (atmos_format == 'muramsb'):
            print("info:overseer: opening the atmosphere in muram binary (sub) format...")
            path = '/home/milic/data/ISSI_trackings/SSD_25x8Mm_16_pdmp_1_ISSI_Flows/3D'
            atmosarr = muram_binary_loader_sub(path, int(sys.argv[1]), [0,1536,0,1536 ,0,121], stokes)
            print("info:overseer: ...sucess!")
        
        else:
            print("info:overseer: unknown file format. exiting..")
            exit();

        #atmosarr = atmosarr[i_start:i_end, j_start:j_end]

        #wave = np.linspace(516.9,517.6,351)
        wave = np.linspace(525.00,525.04,81)
        #wave = np.linspace(630.1,630.3,201)
        
        print("info::overseer::final atmos shape is: ", atmosarr['T'].shape)

        overseer_work(atmosarr, wave, stokes, task_grain_size = 16)
    else:
        worker_work(rank)
        pass