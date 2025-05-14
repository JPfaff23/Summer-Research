#!/bin/bash
#SBATCH --job-name=worstof_100M_timing
#SBATCH --output=worstof_100M_%j.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --account=def-lstentof
#SBATCH --mail-user=jpfaff23@uwo.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.10 cuda/12.2 arrow/14.0.1
export PYTHONPATH=$SCRATCH/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SCRATCH
python -u ~/make_worst_of_100M.py \
       --rows 20 \
       --paths 100000000 \
       --steps 64 \
       --seed_offset 0 \
       --out restults_100M.parquet