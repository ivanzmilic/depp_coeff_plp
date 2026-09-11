# lw_depp_coeff

Pipeline for synthesizing spectral lines (and departure coefficients) out of 3D
simulation cubes with [Lightweaver](https://github.com/Goobley/Lightweaver).
The main driver, `lw_synth_1d5D.py`, treats every horizontal pixel of a MURAM
cube as an independent 1D atmospheric column ("1.5D"), solves the NLTE problem in
each with Lightweaver, and writes the emergent spectra to a FITS file. Work is
distributed over MPI with a single overseer + many workers, so it scales across
cores/nodes.

Currently configured to synthesize the **Sr I 4607 Å** line.

---

## 1. Setup

### 1a. Python environment

```bash
conda env create -f environment.yml
conda activate lw
```

This installs Lightweaver and all numeric dependencies but **not** `mpi4py`
(see below).

### 1b. MPI (important)

`mpi4py` must be built against the **system** MPI that `mpirun` uses — not a
conda-bundled one. If they mismatch, `mpirun -n N` silently launches `N`
independent size-1 processes instead of one `N`-rank job, and nothing works.

On the KIS cluster:

```bash
module load mpi/openmpi-401_gcc-485                       # Open MPI 4.0.1
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py==3.1.5
```

Sanity check — this must print ranks `0..3` of **size 4** (four lines):

```bash
mpirun -n 4 python -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.rank, c.size)"
```

If instead you get four copies of `0 1`, the `mpi4py`/MPI pairing is wrong —
rebuild `mpi4py` with the module loaded.

### 1c. Local modules

The driver imports a few files that live in this repo (no install needed, just
run from the repo root): `loaders.py` (cube I/O), `muram.py` (MURAM reader),
`myatoms.py` and `custom_atoms.py` (atomic models, incl. `sr_tanausu`).

---

## 2. Running a synthesis

```bash
module load mpi/openmpi-401_gcc-485
conda activate lw
mpirun -n <N> python lw_synth_1d5D.py <path> <filename> <number> <stokes> <use_ne> <format>
```

Positional arguments:

| # | arg        | meaning                                                        | example                                |
|---|------------|---------------------------------------------------------------|----------------------------------------|
| 1 | `path`     | directory holding the simulation snapshots (also output dir)  | `/dat/milic/3D/3D_full_subdomain/`     |
| 2 | `filename` | tag used to name the output FITS file                         | `sr_synth`                             |
| 3 | `number`   | snapshot iteration number                                     | `4350`                                 |
| 4 | `stokes`   | `True` = full Stokes IQUV (needs B), `False` = Stokes I only  | `False`                                |
| 5 | `use_ne`   | currently ignored — `n_e` is always recomputed in LTE         | `False`                                |
| 6 | `format`   | atmosphere format; only `muramsb` (MURAM sub-snapshot) is wired up | `muramsb`                         |

Use `-n <N>` with **N ≥ 2** (1 overseer + N−1 workers). Quick single-snapshot run:

```bash
mpirun -n 6 python lw_synth_1d5D.py /dat/milic/3D/3D_full_subdomain/ sr_synth 4350 False False muramsb
```

### Batch runs

`runmultiples.sh` / `runmultiples2.sh` loop the command above over a list of
snapshot numbers. Edit the variables at the top (`DATAPATH`, `TAG`, `STOKES`,
`USE_NE`, `FORMAT`, `NPROC`) and the `for i in ...` list, then:

```bash
bash runmultiples.sh
```

---

## 3. Configuration knobs

Most run parameters are (still) hard-coded near the bottom of
`lw_synth_1d5D.py`, in the `__main__` / `overseer_work` section — edit them there:

- **Sub-domain & sampling** — the loader call
  `muram_binary_loader(path, number, [0,768,0,768,0,121], 16E3, stokes, 2, 2, sub=True)`:
  region `[xmin,xmax,ymin,ymax,zmin,zmax]`, vertical `dz`, and `skipx, skipy`
  (here `2,2` → a 768×768 region becomes 384×384 columns).
- **Wavelength grid** — `wave = np.linspace(460.72, 460.75, 601)` [nm, air; converted to vacuum internally].
- **Task granularity** — `task_grain_size = 32` columns per MPI task.
- **Active species** — hard-coded to `Sr` inside `synth()`.

**Sr abundance** has its own dedicated knob at the *top* of the module:

```python
SR_ABUNDANCE_DEX = 3.07   # dex, H == 12.0; Lightweaver default is 2.87; None = default table
```

Set it once here and the whole run uses it. It is applied both to the level
populations (`nTotal = abundance · nHTot`) and to the equation of state. Note the
override must be keyed by an `Element` internally — the module handles this via
`make_abundance()`; a plain `{'Sr': ...}` string key would be silently ignored.

---

## 4. Output

A FITS file is written to `<path><filename>_<number>_lwsynth_<wave0>.fits`, e.g.
`/dat/milic/3D/3D_full_subdomain/sr_synth_4350_lwsynth_460.72.fits`, with:

- **HDU 0** — spectra, shape `(NX, NY, nStokes, nWave)` (`nStokes` = 1 for I-only, 4 for full Stokes).
- **HDU 1** — the wavelength grid.

(Level-population writeback exists in the code but is currently commented out.)

---

## 5. Monitoring & troubleshooting

- **Hanging workers.** Individual columns can occasionally stall inside the NLTE
  solve. Each worker logs progress to `/tmp/lw_worker_<rank>_activity.log`; use
  `python monitor_workers.py --watch` to spot stuck ranks. See
  [`WORKER_MONITORING.md`](WORKER_MONITORING.md) for details.
- **`ImportError` on every rank at start-up** usually means a stale import or a
  local module missing from the run directory — the job dies before MPI starts
  and *looks* like an MPI failure. Test with a plain
  `python -c "import lw_synth_1d5D"` first.
- **`mpirun` launches size-1 singletons** → the `mpi4py`/system-MPI mismatch from
  §1b; rebuild `mpi4py` against the loaded module.

A worked debugging write-up lives in [`docs/mpirun_debug_report.pdf`](docs/mpirun_debug_report.pdf).
