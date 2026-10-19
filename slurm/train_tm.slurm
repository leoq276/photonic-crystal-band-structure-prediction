#!/bin/bash
# Submit from this directory with: sbatch slurm/train_tm.slurm
# Output and training artifacts stay inside this release folder.
#SBATCH --job-name=den-tm-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Activate the intended environment before sbatch, or set PYTHON_BIN explicitly.
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT_DIR"
"$PYTHON_BIN" train_tm.py \
  --data ../data/ev10-rand_uniform.mat \
  --device cuda \
  --output-dir outputs/training_uniform
