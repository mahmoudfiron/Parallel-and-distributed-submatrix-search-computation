#!/bin/bash
#SBATCH -J submatrix_c
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:10:00
srun ./build/pds_project_mpi_omp_c ${1:-data/input.txt} ${2:-output.txt}
