#!/bin/bash

sbatch --partition=Centaurus --chdir=`pwd` --time=00:10:00 --ntasks=1 --cpus-per-task=1 --job-name=nbody_seq build_seq.sh