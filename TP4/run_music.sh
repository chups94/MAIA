#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J music-lstm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Activer l'environnement Python
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

# Informations sur le job
echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi

# Lancer l'entraînement
python -u train_music.py
