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
from lightweaver.rh_atoms import (Al_atom, C_atom, CaII_atom, Fe_atom,
                                  H_6_atom, He_9_atom, MgII_atom, N_atom,
                                  Na_atom, O_atom, S_atom, Si_atom)
from mpi4py import MPI
from radynpy.cdf import LazyRadynData
from tqdm import tqdm
threadpool_limits(1)
# NOTE(cmo): Numpy please, I beg you, only create 1 BLAS thread per process. 

# NOTE(cmo): Based on Andres' + Andreu's MPI Lightweaver worker

class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def slice_tasks(cdf, task_start, grain_size):
    task_end = min(task_start + grain_size, cdf.time.shape[0])
    sl = slice(task_start, task_end)
    data = {}
    data['taskGrainSize'] = task_end - task_start
    data['z1'] = cdf.z1[sl] / 1e2
    data['temperature'] = cdf.tg1[sl]
    data['ne'] = cdf.ne1[sl] * 1e6
    data['vlos'] = cdf.vz1[sl] / 1e2
    nHTot = cdf.d1[sl] / (lw.DefaultAtomicAbundance.massPerH * lw.Amu) * 1e3
    data['nHTot'] = nHTot
    data['HPops'] = ((cdf.n1[sl, :, :6, 0] * 1e6) / (cdf.n1[sl, :, :6, 0].sum(axis=2) * 1e6 / nHTot)[:, :, None]).transpose(0, 2, 1)
    data['CaPops'] = ((cdf.n1[sl, :, :6, 1] * 1e6) / (cdf.n1[sl, :, :6, 0].sum(axis=2) * 1e6 / nHTot)[:, :, None]).transpose(0, 2, 1)
    return data


def overseer_work(cdf_list, cdf_names, savedir, task_grain_size=16):
    """ Function to define the work to do by the overseer """
    # Index of the task to keep track of each job
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    data_size = 0
    num_tasks = 0
    meta = []
    file_idx_for_task = []
    task_start_idx = []
    task_writeback_range = []
    for file_idx, cdf in enumerate(cdf_list):
        cdf_size = cdf.time.shape[0]

        num_cdf_tasks = int(np.ceil(cdf_size / task_grain_size))
        file_idx_for_task.extend([file_idx] * num_cdf_tasks)
        task_start_idx.extend(range(0, cdf_size, task_grain_size))

        task_writeback_range.extend([slice(data_size + i*task_grain_size, 
                                           min(data_size + (i+1)*task_grain_size,
                                               data_size + cdf_size)
                                     ) for i in range(num_cdf_tasks)])

        meta.extend(zip([cdf_names[file_idx]] * cdf_size, range(cdf_size)))
        data_size += cdf_size
        num_tasks += num_cdf_tasks

    # Define the lists that will store the data of each feature-label pair
    spectra_6563 = [None] * data_size
    spectra_8542 = [None] * data_size

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
                    file_idx = file_idx_for_task[task_index]

                    # Slice out our task
                    data = slice_tasks(cdf_list[file_idx], task_start_idx[task_index], task_grain_size)
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
                    spectra_6563[task_writeback] = data_in['H_6563']
                    spectra_8542[task_writeback] = data_in['Ca_8542']
                    progress_bar.update(1)

            # if the worker has the exit tag mark it as closed.
            elif tag == tags.EXIT:
                print(" * Overseer : worker {0} exited.".format(source))
                closed_workers += 1

    # Once finished, dump all the data
    print("Overseer finishing")

    with open(os.path.join(savedir, 'H_6563.pkl'), 'wb') as f:
        pickle.dump(spectra_6563, f)

    with open(os.path.join(savedir, 'Ca_8542.pkl'), 'wb') as f:
        pickle.dump(spectra_8542, f)

    with open(os.path.join(savedir, 'LimbDarkeningMeta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

def worker_work(rank):
    # Function to define the work that the workers will do

    # Set up our atmosphere/context
    Nspace = 300
    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, depthScale=np.linspace(100, 0, Nspace),
                                  temperature=np.ones(Nspace)*5000, ne=np.ones(Nspace), 
                                  vlos=np.ones(Nspace), vturb=np.ones(Nspace) * 2e3,
                                  nHTot=np.ones(Nspace))
    atmos.quadrature(5)
    a_set = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(), Al_atom(), CaII_atom(),
                            Fe_atom(), He_9_atom(), MgII_atom(), N_atom(), Na_atom(), S_atom()])

    a_set.set_active('H', 'Ca')
    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops)

    half_width_8542 = 0.1
    half_width_6563 = 0.14
    line8542 = CaII_atom().lines[-1]
    line6563 = H_6_atom().lines[4]
    NwaveOut = 101
    wave_grid = lambda line, half_width: np.linspace(line.lambda0 - half_width, line.lambda0 + half_width, NwaveOut)
    grid_8542 = wave_grid(line8542, half_width_8542)
    grid_6563 = wave_grid(line6563, half_width_6563)
    combined_grid = np.concatenate((grid_6563, grid_8542))
    ray_grid = np.linspace(0.01, 1.0, 100)

    while True:
        # Send the overseer the signal that the worker is ready
        comm.send(None, dest=0, tag=tags.READY)
        # Receive the data with the index of the task, the atmosphere parameters and/or the tag
        data_in = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            # Receive the Radyn atmosphere
            task_index = data_in['index']
            z = data_in['z1']
            temperature = data_in['temperature']
            vlos = data_in['vlos']
            ne = data_in['ne']
            nHTot = data_in['nHTot']
            h_pops = data_in['HPops']
            ca_pops = data_in['CaPops']
            task_size = data_in['taskGrainSize']

            Iwave_6563 = []
            Iwave_8542 = []
            for t in range(task_size):
                # Configure the context
                # Need to loop over task size
                atmos.height[...] = z[t]
                atmos.temperature[...] = temperature[t]
                atmos.vlos[...] = vlos[t]
                atmos.ne[...] = ne[t]
                atmos.nHTot[...] = nHTot[t]
                eq_pops['H'][...] = h_pops[t]
                eq_pops['Ca'][...] = ca_pops[t]
                ctx.update_deps()

                success = 1
                try:
                    Iwave_combined = ctx.compute_rays(wavelengths=combined_grid, mus=ray_grid)
                    Iwave_6563.append(Iwave_combined[:NwaveOut])
                    Iwave_8542.append(Iwave_combined[NwaveOut:])
                except:
                    # NOTE(cmo): In this instance, the task should never fail
                    # for sane input.
                    success = 0
                    break

            # Send the computed data
            data_out =  {'index': task_index, 'success': success, 'H_6563': Iwave_6563, 'Ca_8542': Iwave_8542}
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
        output_dir = 'Data101_Angstrom'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        grid_path = '/data/p002/RadynGrid/'
        grid_files = [x for x in os.listdir(grid_path) if x.startswith('radyn_out')]
        grid_files_full_path = [grid_path + x for x in grid_files]

        cdfs = [LazyRadynData(f) for f in grid_files_full_path]
        overseer_work(cdfs, grid_files, output_dir)
    else:
        worker_work(rank)
        pass
