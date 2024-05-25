# LHA-Net: A Lightweight and High-accuracy Network for Road Surface Defect Detection
## Requirements
matplotlib==3.7.1 numpy==1.23.5 opencv-python==4.6.0 Pillow==9.0.1 scipy==1.11.4 
thop==0.1.1 torch==1.9.1 torchvision==0.10.0 tqdm==4.64.0
## Datasets
- RDD-CC：https://drive.google.com/file/d/1QwGKqhLUDJtpeFiSEndkvw3cE4oOJhrn/view?usp=sharing
- RDD-SCD：https://drive.google.com/file/d/1KroTNZ2hoY8TMp3p8v4zcIHxiiyqnafr/view?usp=sharing
- RDD-SCM：https://drive.google.com/file/d/1RLheBWE3oWz6gnBjz7H4b49IDVYuyhcc/view?usp=sharing
## Training
- Set MODEL_DIR as the path to the model directory
- Set DATA_DIR as the path to the dataset directory
  
`python train.py cfg=${MODEL_DIR} data=${DATA_DIR}`
## Evaluation
- Set WEIGHT_PATH to the path of the correct checkpoint matching the choice of the model and dataset

`python val.py data=${DATA_DIR} weights=${WEIGHT_PATH}`
