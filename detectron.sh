#!/bin/bash
#SBATCH -J detectron
#SBATCH -p high
#SBATCH -N 1
#SBATCH --chdir=/home/jreyes/face_detectron2
#SBATCH --mem=30GB
#SBATCH --gres=gpu:2
#SBATCH --time=infinite
#SBATCH -o /home/jreyes/face_detectron2/output/detectron_fe%J.%u.out # STDOUT
#SBATCH -e /home/jreyes/face_detectron2/output/detectron_fe%J.%u.err # STDERR

# Load CUDA module
module load CUDA/12.2.2

# Load Anaconda module
module load Anaconda3/2020.02

conda create -n detectron python=3.10.13
conda activate detectron

pip3 install torch torchvision
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

 
python3 ./scripts/main.py
