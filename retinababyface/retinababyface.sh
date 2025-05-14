#!/bin/bash
#SBATCH -J retinababyface
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jreyes/baby_face_detection/retinababyface/slurm_%j.out
#SBATCH --error=/home/jreyes/baby_face_detection/retinababyface/slurm_%j.err
#SBATCH --chdir=/home/jreyes/baby_face_detection/retinababyface

# Load modules
module load CUDA/12.1
module load Miniconda3/4.9.2

# Variables
ENV_NAME=babyface
PYTHON_VERSION=3.10.13
REQUIREMENTS_FILE=requirements.txt

#Enable the bash shell
eval "$(conda shell.bash hook)"

# Create Conda environment if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    source activate "$ENV_NAME"
    echo "Installing requirements..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Conda environment '$ENV_NAME' already exists."
    source activate "$ENV_NAME"
fi

echo "Environment is ready!"

# Run training with specified arguments
python main.py \
    --root_dir="/home/jreyes/obbabyface_rot" \
    --backbone="densenet121" \
    --epochs=100 \
    --lr=0.001 \
    --scheduler="OneCycle" \
    --patience=10 \
    --run_name="densenet121_10lambda_all" \
    --record_metrics \
    --batch_size=32 \
    --checkpoint_path="dense_10lambda_all.pt" \
    --clip_value=1.0 \
    --weight_decay=0 \
    --lambda_rot=10 \
    --lambda_cls=10 \
    --lambda_obb=10 \
    --inference_results="dense_10lambda_all" \
    --predictions_dir="pred_dense_10lambda_all" \