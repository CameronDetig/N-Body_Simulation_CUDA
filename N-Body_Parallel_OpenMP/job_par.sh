#!/bin/bash

sbatch --partition=Centaurus --chdir=`pwd` --time=00:10:00 --ntasks=1 --cpus-per-task=36 --job-name=nbody_par build_par.sh