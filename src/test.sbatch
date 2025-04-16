#!/bin/bash
#SBATCH --partition=study
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --tmp=1G

export TMPDIR=$SLURM_JOB_TMP
echo "Loading Net1"
python load.py
echo "Showing content of $SLURM_JOB_TMP"
ls $SLURM_JOB_TMP
