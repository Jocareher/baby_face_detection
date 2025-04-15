#!/bin/bash
#SBATCH -J yolo
#SBATCH -p high
#SBATCH -N 1
#### SBATCH --nodelist=node026
#SBATCH --chdir=/home/jreyes/Baby_face_detection
#SBATCH --gres=gpu:tesla:1
#SBATCH -o /home/jreyes/Baby_face_detection/yolo%J.%u.out # STDOUT
#SBATCH -e /home/jreyes/Baby_face_detection/yolo%J.%u.err # STDERR

# Load CUDA module
module load CUDA/12.2.2

# Load Anaconda module
module load Anaconda3/2020.02

# Check if the Conda environment 'yolo' already exists
if ! conda info --envs | grep -q yolo; then
    echo "Creating conda environment 'yolo'"
    conda create -n yolo python=3.9 -y
else
    echo "Conda environment 'yolo' already exists"
fi

source activate yolo

pip install ultralytics

# # Build a new model from YAML and start training from scratch
# yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

# # Start training from a pretrained *.pt model
# yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

# # Build a new model from YAML, transfer pretrained weights to it and start training
# yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640

yolo task=obb mode=train model=yolov8s-obb.pt data=./configs/yolo.yaml epochs=100 imgsz=640 batch=16
