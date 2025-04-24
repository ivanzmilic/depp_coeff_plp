import os
import sys
import math
import time
import numpy as np

import lightweaver as lw
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, C_atom, O_atom, Si_atom, Al_atom, \
CaII_atom, Fe_atom, He_9_atom, MgII_atom, N_atom, Na_atom, S_atom
from lightweaver.utils import NgOptions, get_default_molecule_path,ExplodingMatrixError

import astropy.units as Unit
from scipy.interpolate import interp1d
import scipy.constants as ct

from mpi4py import MPI
import xarray as xr


def run_a_pixel(depthScale,temperature,vlos,vturb,Pgas,nHTot,wave,conserve=False, useNe=False,prd=False, hprd= False):
    atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Geometric, 
                                        depthScale= depthScale,
                                        temperature= temperature,  
                                        vlos= vlos, 
                                        vturb= vturb, 
                                        Pgas= Pgas, 
                                        nHTot = nHTot,
                                        convertScales=False)

    Iwave = synth(atmos, conserve=conserve, useNe=useNe, wave=lw.utils.air_to_vac(wave), prd=prd, hprd= hprd)
    return Iwave


def synth(atmos, conserve, useNe, wave, prd, hprd):
    '''
    Synthesise a spectral line for given atmosphere with different
    conditions.

    Parameters
    ----------
    atmos : lw.Atmosphere
        The atmospheric model in which to synthesise the line.
    conserve : bool
        Whether to start from LTE electron density and conserve charge, or
        simply use from the electron density present in the atomic model.
    useNe : bool
        Whether to use the electron density present in the model as the
        starting solution, or compute the LTE electron density.
    wave : np.ndarray
        Array of wavelengths over which to resynthesise the final line
        profile for muz=1.

    Returns
    -------
    ctx : lw.Context
        The Context object that was used to compute the equilibrium
        populations.
    Iwave : np.ndarray
        The intensity at muz=1 for each wavelength in `wave`.
    '''
    # Configure the atmospheric angular quadrature
    atmos.quadrature(5)
    # Configure the set of atomic models to use.
    aSet = lw.RadiativeSet([H_6_atom(), C_atom(), O_atom(), Si_atom(),
                            Al_atom(), CaII_atom(), Fe_atom(), He_9_atom(),
                            MgII_atom(), N_atom(), Na_atom(), S_atom()
                           ])
    # Set H and Ca to "active" i.e. NLTE, everything else participates as an
    # LTE background.
    aSet.set_active('H','Ca')
    # Compute the necessary wavelength dependent information (SpectrumConfiguration).
    spect = aSet.compute_wavelength_grid()
    # Either compute the equilibrium populations at the fixed electron density
    # provided in the model, or iterate an LTE electron density and compute the
    # corresponding equilibrium populations (SpeciesStateTable).


    molPaths = [get_default_molecule_path() + m + '.molecule' for m in ['H2']]
    mols = lw.MolecularTable(molPaths)

    if useNe:
        eqPops = aSet.compute_eq_pops(atmos, mols)
    else:
        eqPops = aSet.iterate_lte_ne_eq_pops(atmos, mols)

    # if useNe:
    #     eqPops = aSet.compute_eq_pops(atmos)
    # else:
    #     eqPops = aSet.iterate_lte_ne_eq_pops(atmos)

    # Configure the Context which holds the state of the simulation for the
    # backend, and provides the python interface to the backend.
    # Feel free to increase Nthreads to increase the number of threads the
    # program will use.
    ctx = lw.Context(atmos, spect, eqPops, conserveCharge=conserve, Nthreads=1)
    # Iterate the Context to convergence (using the iteration function now
    # provided by Lightweaver)
    if hprd:
        try:
            ctx.update_hprd_coeffs()
        except:
            pass
        
    ctx.depthData.fill = True
    try:
        lw.iterate_ctx_se(ctx, prd = prd, quiet=True)
    except ExplodingMatrixError:
        pass
    #iterate_ctx(ctx, atmos, eqPops, prd=prd, updateLte=False)
    # Update the background populations based on the converged solution and
    # compute the final intensity for mu=1 on the provided wavelength grid.

    eqPops.update_lte_atoms_Hmin_pops(atmos)
    try:
        Iwave = ctx.compute_rays(wave, [atmos.muz[-1]], stokes=False)
    except:
        Iwave=np.zeros(len(wave))
    return Iwave


def master(x,y,z,t,vz,gas_p,nh, wave,size):
    nx,ny,nz = np.shape(t)
    depthScale= np.ascontiguousarray(z).astype('float64')
    temperature= np.ascontiguousarray(t.reshape(nx*ny,nz)).astype('float64')
    vlos= np.ascontiguousarray(vz.reshape(nx*ny,nz)).astype('float64')
    vturb= np.ascontiguousarray(np.zeros(len(z))).astype('float64')
    Pgas= np.ascontiguousarray(gas_p.reshape(nx*ny,nz)).astype('float64')
    nHTot = np.ascontiguousarray(nh.reshape(nx*ny,nz)).astype('float64')    

    total_pixels = temperature.shape[0]
    total_nwave  = len(wave)
    pieces = np.array_split(np.arange(total_pixels),size)

    for irank, piece in enumerate(pieces):
        data = {}
        data['depthScale'] = depthScale[::-1]
        data['temperature'] = temperature[piece,::-1]
        data["temperature"][data["temperature"]<2000]=2000.
        data['vlos'] = vlos[piece,::-1]
        data['vturb'] = vturb[::-1]
        data['Pgas'] =  Pgas[piece,::-1]
        data['nHTot'] =  nHTot[piece,::-1]
        data["wave"]=wave

        comm.recv(source=irank+1, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        comm.send(data, dest=source, tag=3)

    Iwave = np.zeros((total_pixels,total_nwave))

    for irank, piece in enumerate(pieces):
       CRD_PRD = comm.recv(source=irank+1, tag=1, status=status)
       Iwave[piece,:]=CRD_PRD["Iwave"]
       #print("received synthesis from ",irank, "of size",np.shape(CRD_PRD["Iwaveprd_cobold_crd"]))

    ds = xr.Dataset(
        {
            "Intensity": (["x", "y", "wavelength"], Iwave.reshape(nx,ny,total_nwave)),
        },
        coords={
            "x": x,
            "y": y,
            "wavelength": wave,
        }
    )
    ds.to_netcdf("result.h5",mode="w")


def slave(rank):
    comm.send(None, dest=0, tag=0)
    data_in = comm.recv(source=0,tag=3, status=status)

    npixels = np.shape(data_in['temperature'])[0]
    nwave= np.shape(data_in['wave'])[0]
    Iwave = np.zeros((npixels,nwave))
    for ipixel in np.arange(npixels):
        Iwave[ipixel,:] = run_a_pixel(data_in["depthScale"],
                                    data_in["temperature"][ipixel,:],
                                    data_in["vlos"][ipixel,:],
                                    data_in["vturb"],
                                    data_in["Pgas"][ipixel,:],
                                    data_in["nHTot"][ipixel,:],
                                    data_in["wave"])
        
    CRD_PRD = {}
    CRD_PRD["Iwave"] = Iwave

    comm.send(CRD_PRD, dest=0, tag=1)

def get_quantity(datablock,convert=None):
    if convert:
        return (datablock*Unit.Unit(datablock.attrs["units"].replace("gm", "g").replace("^","**")).to(convert)).data.T
    else:
        return (datablock*Unit.Unit(datablock.attrs["units"].replace("gm", "g").replace("^","**"))).data.T

if (__name__ == '__main__'):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    inputfile = sys.argv[1]
    

    if rank == 0:
        
        print(f"Doing: {inputfile}")
        datablock = xr.open_dataset(inputfile)

        bx  = get_quantity(datablock["bx"], convert =Unit.Tesla)
        by  = get_quantity(datablock["by"], convert =Unit.Tesla)
        bz  = get_quantity(datablock["bz"], convert =Unit.Tesla)

        rho = get_quantity(datablock["rho"], convert =Unit.gram/Unit.m**3)

        vx  = get_quantity(datablock["vx"], convert =Unit.m/Unit.s)
        vy  = get_quantity(datablock["vy"], convert =Unit.m/Unit.s)
        vz  = get_quantity(datablock["vz"], convert =Unit.m/Unit.s)

        gas_p  = get_quantity(datablock["p"], convert =Unit.Newton/Unit.m**2)
        temperature  = get_quantity(datablock["T"])

        x = get_quantity(datablock["x"])
        y = get_quantity(datablock["y"])
        z = get_quantity(datablock["z"])

        local_z = np.copy(z)

        wght_per_h=1.4271

        nh = 0.001*rho / (wght_per_h * ct.atomic_mass)
        
        datablock.close()
        wave = np.linspace(393.36633-0.5,393.36633+.5, 1001)
        #wave=[450.]
        master(x,y,z,temperature,vz,gas_p,nh, wave,size-1)
    else:
        slave(rank)
        pass

