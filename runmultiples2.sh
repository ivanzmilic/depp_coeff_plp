#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

# lw_synth_1d5D.py positional args: <path> <filename> <number> <stokes> <use_ne> <format>
DATAPATH="/dat/milic/3D/3D_full_subdomain/"
TAG="sr_synth"      # output naming tag: <path><TAG>_<number>_lwsynth_<wave0>.fits
STOKES="False"
USE_NE="False"
FORMAT="muramsb"
NPROC=128

for i in 4650 4800 4950 5100 5250 5400 5550 5700 5850 6000
do
  echo "I am running cube number $i"
  mpirun -n $NPROC python lw_synth_1d5D.py "$DATAPATH" "$TAG" $i $STOKES $USE_NE $FORMAT
done
