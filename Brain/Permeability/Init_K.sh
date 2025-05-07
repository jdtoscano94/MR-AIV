#!/bin/bash
#SBATCH -J Init_K
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=80:00:00
#SBATCH --mem=64GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o /users/jdtoscan/data/jdtoscan/References/MR-AVI/Results_log/Output/Init_K-%j.out
#SBATCH -e /users/jdtoscan/data/jdtoscan/References/MR-AVI/Results_log/Error/Init_K-%j.err

cd /users/jdtoscan/data/jdtoscan/References/MR-AVI/Brain/Permeability/|| exit

nvidia-smi
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate Instant_AIV_39

rm -rf ./__pycache__/

python3 -u Init_K.py  --Name 'Init_K'  