from threadpoolctl import threadpool_limits

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
threadpool_limits(1)
import pickle
from enum import IntEnum
import traceback
import datetime
import time

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

    s = 1E4/(lambda_air*1E1);
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
    atmos.quadrature(3)#, force3d=True)
    # See if you can force this to 
    # Replace this with atmos.rays ( specify mu ) - let's think how to use Stokes with it
    # ctx.single_stokes_fs
    
    # Configure the set of atomic models to use. Contrary to SNAPI you have to explicitly specify all species
    # Annoying, but since you have all the atoms in the file - that is fine.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(), Fe23_5250(), He_9_atom(), MgI_atom(), N_atom(), Na_atom(), S_atom()])
    
    # Set actives to the ones you want
    #aSet.set_active(actives)
    aSet.set_active('Ca')
    
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()

    # Calculate electron density in lte, we are never using the electron density from the model
    eqPops = aSet.iterate_lte_ne_eq_pops(atmos, direct=True)

    # Configure the Context which holds the state of the simulation for the  backend, and provides 
    # the python interface to the backend.
    # Feel free to increase Nthreads to increase the number of threads the program will use.
    # I would always stick to Nthreads = 1 as we are looking to mpi this one
    
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=False, Nthreads=1, formalSolver='piecewise_linear_1d')
    
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver). Go test this one in order to calculate stuff in LTE!
    
    #lw.iterate_ctx_se(ctx, prd=prd)
    
    lw.iterate_ctx_se(ctx, prd=prd, quiet=True)

    #ctx.formal_sol_gamma_matrices()

    
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
    
    return ctx, Iwave

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

def overseer_work(atmosarr, wave, stokes, use_ne = False, task_grain_size=16, task_info=None):
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
    pops = [None] * data_size
    
    success = True
    task_status = [0] * num_tasks
    failed_tasks = []  # Track failed tasks for reporting

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
                    #print(" * Overseer : sending task {0} to worker {1}".format(task_index, source))
                    task_status[task_index] = 1

                # If error, or no work left, kill the worker
                except:
                    comm.send(None, dest=source, tag=tags.EXIT)

            # If the tag is Done, receive the status, the index and all the data
            # and update the progress bar
            elif tag == tags.DONE:
                #print(" * Overseer : worker {0} finished task {1}".format(source, data_in['index']))
                success = data_in['success']
                task_index = data_in['index']

                if not success:
                    task_status[task_index] = 0
                    error_info = data_in.get('error_info', 'Unknown error')
                    print(f"!!! Task {task_index} FAILED: {error_info}")
                    print(f"    Check /tmp/lw_worker_{source}_errors.log for details")
                    failed_tasks.append({'task_index': task_index, 'worker': source, 'info': error_info})
                else:
                    task_writeback = task_writeback_range[task_index]
                    spectra[task_writeback] = data_in['spectrum']
                    pops[task_writeback] = data_in['pops']
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
    
    # THIS IS HARDCODED SO YOU MIGHT WANNA SKIP THIS IF YOU DONT WANT LEVEL POPULATIONS 
    # (But Ivan needs it for his Ca II project ;-))
    pops = np.asarray(pops)
    pops = pops.reshape(NX, NY, 6, -1)
    print("info::overseer::writing the spectra")
    spechdu = fits.PrimaryHDU(spectra)
    wavhdu = fits.ImageHDU(wave)
    popshdu = fits.ImageHDU(pops)
    to_output = fits.HDUList([spechdu, wavhdu, popshdu])
    if (task_info is not None):
        path, filename, number = task_info
        to_output.writeto(path+filename+'_'+str(number)+'_lwsynth_'+str(wave[0])+'.fits', overwrite=True)    
    else:
        to_output.writeto('/dat/milic/lw_synth_1d5D_output.fits', overwrite=True)

    return 0  
    
    

    
def worker_work(rank):
    # Function to define the work that the workers will do
    
    # Open a per-worker activity log
    activity_log = f'/tmp/lw_worker_{rank}_activity.log'
    with open(activity_log, 'w') as f:
        f.write(f"Worker {rank} started at {datetime.datetime.now()}\n")
    
    print(f"[Worker {rank}] Started, logging to {activity_log}", flush=True)

    while True:
        # Send the overseer the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        with open(activity_log, 'a') as f:
            f.write(f"[{datetime.datetime.now()}] READY signal sent, waiting for task...\n")
        
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
            
            task_start_time = time.time()
            with open(activity_log, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] Received task {task_index}, grain_size={task_size}, stokes={stokes}\n")
            print(f"[Worker {rank}] Starting task {task_index} with {task_size} pixels", flush=True)

            if (stokes):
                B = data_in['B'].astype(float)
                inc = data_in['inc'].astype(float)
                azi = data_in['azi'].astype(float)

            
            ns = 1
            if (stokes):
                ns = 4

            I = np.zeros([task_size, ns,len(wave)])
            pops = np.zeros([task_size, 6,z.shape[1]])
            
            for t in range(task_size): # Now, task size is not one, probably a good choice
                pixel_start_time = time.time()
                with open(activity_log, 'a') as f:
                    f.write(f"[{datetime.datetime.now()}] Task {task_index}, pixel {t}/{task_size} - starting synthesis\n")
                
                # Configure the context
                # Need to loop over task size
                atmos = 0
                # TODO - consider convertScales!!!!
                # Here you can provide vx and vy, so the code will reckognize the 3D field. 

                if (stokes):
                    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=z[t],
                                  temperature=temperature[t],  vlos=vlos[t], vturb=np.ones(ND)*5E3, Pgas=pg[t],
                                  B=B[t], gammaB=inc[t], chiB=azi[t], convertScales=False)
                else:
                    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=z[t],
                                  temperature=temperature[t],  vlos=vlos[t], vturb=np.ones(ND)*10E3, Pgas=pg[t], 
                                  convertScales=False)

                success = 1
                try:
                    with open(activity_log, 'a') as f:
                        f.write(f"[{datetime.datetime.now()}] Task {task_index}, pixel {t}/{task_size} - calling synth()\n")
                    
                    ctxtemp, Itemp = synth(atmos, conserve=False, prd=False, stokes=stokes, wave=airtovac(wave), mu=1.0, actives='Ca')
                    I[t,:,:] = np.copy(Itemp)
                    pops[t,:,:] = np.copy(ctxtemp.eqPops['Ca'])
                    
                    pixel_elapsed = time.time() - pixel_start_time
                    with open(activity_log, 'a') as f:
                        f.write(f"[{datetime.datetime.now()}] Task {task_index}, pixel {t}/{task_size} - SUCCESS in {pixel_elapsed:.2f}s\n")
                    
                except Exception as e:
                    # Log detailed failure information
                    error_msg = (
                        f"\n{'='*60}\n"
                        f"WORKER {rank} FAILURE at {datetime.datetime.now()}\n"
                        f"Task index: {task_index}, Pixel: {t}/{task_size}\n"
                        f"Exception: {type(e).__name__}: {str(e)}\n"
                        f"Atmospheric parameters:\n"
                        f"  T range: [{temperature[t].min():.1f}, {temperature[t].max():.1f}] K\n"
                        f"  Pg range: [{pg[t].min():.2e}, {pg[t].max():.2e}] dyne/cm^2\n"
                        f"  vlos range: [{vlos[t].min():.1f}, {vlos[t].max():.1f}] cm/s\n"
                        f"  z range: [{z[t].min():.2e}, {z[t].max():.2e}] cm\n"
                    )
                    
                    if stokes:
                        error_msg += (
                            f"  B range: [{B[t].min():.1f}, {B[t].max():.1f}] G\n"
                            f"  inc range: [{inc[t].min():.2f}, {inc[t].max():.2f}] rad\n"
                            f"  azi range: [{azi[t].min():.2f}, {azi[t].max():.2f}] rad\n"
                        )
                    
                    error_msg += f"Traceback:\n{traceback.format_exc()}\n{'='*60}\n"
                    
                    # Print to stdout (will appear in MPI output)
                    print(error_msg, flush=True)
                    
                    # Also save to a per-worker log file
                    with open(f'/tmp/lw_worker_{rank}_errors.log', 'a') as f:
                        f.write(error_msg)
                    
                    # Save the problematic atmospheric data for offline debugging
                    failed_data = {
                        'worker_rank': rank,
                        'task_index': task_index,
                        'pixel_in_task': t,
                        'z': z[t],
                        'temperature': temperature[t],
                        'pg': pg[t],
                        'vlos': vlos[t],
                        'wave': wave,
                        'exception_type': type(e).__name__,
                        'exception_msg': str(e),
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    if stokes:
                        failed_data.update({
                            'B': B[t],
                            'inc': inc[t],
                            'azi': azi[t]
                        })
                    
                    # Save to pickle file
                    pickle_file = f'/tmp/lw_failed_pixel_r{rank}_t{task_index}_p{t}.pkl'
                    with open(pickle_file, 'wb') as f:
                        pickle.dump(failed_data, f)
                    
                    print(f"Worker {rank}: Saved failed pixel data to {pickle_file}", flush=True)
                    
                    success = 0
                    break  # Exit the loop, this task has failed

                # we also want to pack scattering source function:
                # ctxtemp.background.sca # This should be a 2D array wavelength x depth 
                # ctxtemp.background.eta, ctxtemp.background.eta
                # 
                # TODO - output the level populations, rotate and then synthesize the spectrum using lw all the way. 
                # This way you will get reliable values of the intensity (we hope!) 
                # Check formal solver. 

            # Send the computed data
            # we do want to fill in tau too, but that can wait for the next step
            task_elapsed = time.time() - task_start_time
            
            if success == 0:
                data_out = {
                    'index': task_index, 
                    'success': success, 
                    'spectrum': I, 
                    'pops': pops,
                    'error_info': f'Worker {rank} failed at pixel {t}/{task_size}'
                }
                with open(activity_log, 'a') as f:
                    f.write(f"[{datetime.datetime.now()}] Task {task_index} FAILED after {task_elapsed:.2f}s\n")
                print(f"[Worker {rank}] Task {task_index} FAILED after {task_elapsed:.2f}s", flush=True)
            else:
                data_out = {'index': task_index, 'success': success, 'spectrum': I, 'pops': pops}
                with open(activity_log, 'a') as f:
                    f.write(f"[{datetime.datetime.now()}] Task {task_index} COMPLETED in {task_elapsed:.2f}s ({task_elapsed/task_size:.2f}s per pixel)\n")
                print(f"[Worker {rank}] Task {task_index} completed in {task_elapsed:.2f}s", flush=True)
            
            comm.send(data_out, dest=0, tag=tags.DONE)
            with open(activity_log, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] DONE signal sent for task {task_index}\n")

        # If the tag is exit break the loop and kill the worker and send the EXIT tag to overseer
        elif tag == tags.EXIT:
            with open(activity_log, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] Received EXIT signal\n")
            print(f"[Worker {rank}] Received EXIT signal, shutting down", flush=True)
            break

    comm.send(None, dest=0, tag=tags.EXIT)
    with open(activity_log, 'a') as f:
        f.write(f"[{datetime.datetime.now()}] Worker {rank} exiting, EXIT signal sent to overseer\n")
    print(f"[Worker {rank}] Exited", flush=True)

if (__name__ == '__main__'):

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    
    #print(f"Node {rank}/{size} active", flush=True)

    if rank == 0: # If I am the overseer process

        print("info::overseer::starting...")

        # --------------------------------------------------------------------
        path = sys.argv[1] # path where the data is
        filename = sys.argv[2]  # characteristic naming for the files, used differently depending on what 
                                # kind of simulation we are working with 
        number = int(sys.argv[3]) # number of the snapshot - again will be used differently for muram, co5bold, etc...
        stokes = sys.argv[4].lower() == 'true' # whether to synthesize Stokes I or all 4 components
        use_ne = sys.argv[5].lower() == 'true' # whether to use the electron density from the model or to compute it in LTE (or even nlte)
        atmos_format = sys.argv[6]
        
        atmosarr = 0
        
        print("info::overseer::stokes mode is: ", stokes)

        if (atmos_format == 'muramb'):
            print("info:overseer: opening the atmosphere in muram binary format...")
            atmosarr = muram_binary_loader(path, number, [0,1024,0,1024,15,316], 20E3, stokes, 4,4)
            print("info:overseer: ...sucess!")
            
        elif (atmos_format == 'muramsb'):
            print("info:overseer: opening the atmosphere in muram binary (sub) format...")
            path = '/dat/milic/3D/3D_full_subdomain/'
            atmosarr = muram_binary_loader_sub(path, int(sys.argv[1]), [512,53,0,512 ,11,90], stokes)
            print("info:overseer: ...sucess!")
            
        elif (atmos_format == 'mrmfx'):
            print("info:overseer: opening the atmosphere in simple muram format...")
            atmosarr = load_muram_fixed_format(sys.argv[1],stokes)
            print("info:overseer: ...sucess!")

        elif (atmos_format == 'sir'):
            print("info:overseer: opening the atmosphere in SIR format...")
            atmosarr = sir_format_loader(sys.argv[1],stokes)
            print("info:overseer: ...sucess!")
        
        else:
            print("info:overseer: unknown file format. exiting..")
            exit();

        #atmosarr = atmosarr[i_start:i_end, j_start:j_end]

        #wave = np.linspace(516.9,517.6,351)
        #wave0 = np.arange(30)*25.0 + 200.
        #wave = np.linspace(525.00,525.04,81)
        wave = np.linspace(392.8,394.8,1001)
        #wave = np.append(wave, wave0)
        #wave = np.sort(wave)
        #print("info::overseer:: this is our wavelength grid: ", wave)
        #wave = np.linspace(460.0, 461.5, 151)
        
        print("info::overseer::final atmos shape is: ", atmosarr['T'].shape)

        overseer_work(atmosarr, wave, stokes, use_ne, task_grain_size = 32, task_info = [path, filename, number])
    else:
        worker_work(rank)
        pass
