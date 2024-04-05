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
from lightweaver.rh_atoms import (Al_atom, C_atom, CaII_atom, Fe23_atom, H_6_atom, He_9_atom, MgII_atom, N_atom, Na_atom, O_atom, S_atom, Si_atom)
from mpi4py import MPI
from tqdm import tqdm
import sys
threadpool_limits(1)
# NOTE(cmo): Numpy please, I beg you, only create 1 BLAS thread per process. 

# NOTE(cmo): Based on Andres' + Andreu's MPI Lightweaver worker

# various i/o stuff:
from astropy.io import fits

# other files

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
    atmos.quadrature(3)
    # Replace this with atmos.rays ( specify mu ) - let's think how to use Stokes with it
    # ctx.single_stokes_fs
    
    # Configure the set of atomic models to use. Contrary to SNAPI you have to explicitly specify all species
    # Annoying, but since you have all the atoms in the file - that is fine.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe23_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])
    
    # Set actives to the ones you have inputted.
    aSet.set_active(actives)
    
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # Calculate electron density in lte, we are never using the electron density from the model
    eqPops = aSet.iterate_lte_ne_eq_pops(atmos)

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

def slice_tasks(atmosarr, task_start, grain_size):
    # atmosarr : NA x NP x NZ array of atmospheres coming from SIR, SNAPI, MURAM, whatever
    
    task_end = min(task_start + grain_size, atmosarr.shape[0])
    
    sl = slice(task_start, task_end) # this is a slice object, allowing us to access the specific thingy
    data = {}
    data['taskGrainSize'] = task_end - task_start
    data['z'] = atmosarr[sl,0,:]
    data['temperature'] = atmosarr[sl,1,:]
    data['pg'] = atmosarr[sl,2,:]
    data['vlos'] = atmosarr[sl,3,:]
    return data

def overseer_work(atmosarr, task_grain_size=16):
    """ Function to define the work to do by the overseer """
    
    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    data_size = 0 # Let's figure out what this is - total number of pixels?
    num_tasks = 0 # And this is data_size // 16? 
    file_idx_for_task = [] # does this have sth to do with reading from file?
    task_start_idx = [] # no idea
    task_writeback_range = [] # no idea
    
    cdf_size = atmosarr.shape[0]
    print("overseer::info:: cdf_size = ", cdf_size)

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
                print(" * Overseer : worker {0} exited.".format(source))
                closed_workers += 1

    # Once finished, dump all the data
    print("Overseer finishing")
    kek = fits.PrimaryHDU(spectra)
    kek.writeto("output_spectra.fits",overwrite=True)

    
def worker_work(rank):
    # Function to define the work that the workers will do

    #atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=np.linspace(100, 0, Nspace),
    #                              temperature=np.ones(Nspace)*5000, ne=np.ones(Nspace), 
    #                              vlos=np.ones(Nspace), vturb=np.ones(Nspace) * 2e3,
    #                              nHTot=np.ones(Nspace))
    
    while True:
        # Send the overseer the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        # Receive the data with the index of the task, the atmosphere parameters and/or the tag
        data_in = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            # Receive the Radyn atmosphere
            task_index = data_in['index'] # I think we need this? - for what though (to keep track of what succeeeded where)
            z = data_in['z'] # Can, in principle, be different
            temperature = data_in['temperature']
            vlos = data_in['vlos']
            pg = data_in['pg']
            task_size = data_in['taskGrainSize']
            ND = temperature.shape[-1]

            wave = np.linspace(630.1,630.3,201)

            stokes = False
            ns = 1
            if (stokes):
                ns = 4

            I = np.zeros([task_size, ns,len(wave)])
            
            for t in range(task_size): # Now, task size is not one, probably a good choice
                # Configure the context
                # Need to loop over task size
                atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=z[t],
                                  temperature=temperature[t],  vlos=vlos[t], vturb=np.zeros(ND), Pgas=pg[t])
                
                success = 1
                #try:
                    # This should work
                Itemp = synth(atmos, conserve=False, prd=False, stokes=stokes, wave=wave*1.000275, mu=1.0, actives='Fe')
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

    print(f"Node {rank}/{size} active", flush=True)

    if rank == 0:
       
        input_atmos = fits.open(sys.argv[1])[0].data
        input_atmos = input_atmos.transpose(2,3,0,1)
        NX, NY, NP, NZ = input_atmos.shape
        input_atmos = input_atmos.reshape(NX*NY,NP,NZ)

        atmosarr = np.zeros([NX*NY, 4, NZ])
        atmosarr[:,0,:] = input_atmos[:,8,:] * 1E3 # km to m
        atmosarr[:,1,:] = input_atmos[:,1,:] # K 
        atmosarr[:,2,:] = input_atmos[:,9,:] * 10.0
        atmosarr[:,3,:] = input_atmos[:,5,:] * -1E-2 # cm to m
        atmosarr = atmosarr[:,:,::-1] # flip from SIR to normal SA convention
        print("I like to debug kek lol bur")

        del(input_atmos) 
        print("info: final atmos shape is ", atmosarr.shape)

        overseer_work(atmosarr, task_grain_size = 16)
    else:
        worker_work(rank)
        pass