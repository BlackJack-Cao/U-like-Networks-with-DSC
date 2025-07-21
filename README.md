# U-like-Networks-with-Dynamic-Skip-Connections
This code is based on U-Mamba and serves as the official implementation of our paper.

## Framework

Here is the overall framework:

![Framework](assets/framework.jpg)

## Installation

Requirements: `Ubuntu 20.04`, `CUDA 11.8`

1. Create a virtual environment:  `conda create -n ttt python=3.10 -y` and `conda activate ttt`
2. Install Pytorch 2.0.1: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3. Install Mamba: `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
4. Download code: `git clone https://github.com/your-username/U-like-Networks-with-Dynamic-Skip-Connections.git`
5. `cd U-like-Networks-with-Dynamic-Skip-Connections/umamba` and run `pip install -e .`

## Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Model Training

### Train 2D models with Dynamic Skip Connections (DSC)

Train 2D model with DSC:
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerXXXwDSC
```

### Train 3D models with Dynamic Skip Connections (DSC)

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerXXXwDSC
```

## Inference

Predict testing cases with models using DSC:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerXXXwDSC
```

`CONFIGURATION` can be `2d` and `3d_fullres` for 2D and 3D models, respectively.

## Acknowledgements

This project is based on U-Mamba. We acknowledge all the authors of the employed public datasets, as well as the authors of [U-Mamba](https://github.com/U-Mamba/U-Mamba) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for making their valuable code publicly available.
