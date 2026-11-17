#!/bin/bash
#SBATCH --job-name=forestsize-24w
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=forestsize-24w-%j.out

set -euo pipefail
cd /chemin/vers/forestsize

# Remplacer par le Python de l'environnement Linux où le projet est installé.
PYTHON=/chemin/vers/environnement/bin/python

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun uv run --frozen --no-sync python -u src/forestsize/benchmark.py \
    --trial 32 24 64 "$PREFETCH" 32 8192 128