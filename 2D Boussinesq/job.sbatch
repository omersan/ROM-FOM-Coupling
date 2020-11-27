#!/bin/bash
#SBATCH -p batch
#SBATCH -t 120:00:00
#SBATCH --job-name="Boussinesq_FOM"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

#SBATCH --mail-user=shady.ahmed@okstate.edu
#SBATCH --mail-type=end

source activate py37
python 1_Boussinesq_Solver.py
