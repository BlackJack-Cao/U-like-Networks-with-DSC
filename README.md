# U-like-Networks-with-Dynamic-Skip-Connections
Official repository for Enhancing Feature Fusion of U-like Networks with Dynamic Skip Connections. This code is based on U-Mamba and serves as the official implementation of our paper.

## Framework

Here is the overall framework :

![Framework](assets/framework.jpg)

## Installation

The setup for our work follows the same installation and configuration as U-Mamba. Please refer to the [U-Mamba repository](https://github.com/U-Mamba/U-Mamba) for detailed setup instructions.


Follow the installation steps provided in [U-Mamba](https://github.com/U-Mamba/U-Mamba).

## Model Training

The model training process for TTT-Unet also follows the same procedures as U-Mamba. For data preparation and model training, please refer to the [U-Mamba repository](https://github.com/U-Mamba/U-Mamba).

## Inference

Inference for our models also follows U-Mamba's setup. To generate predictions, use the `nnUNetv2_predict` command with the appropriate configuration. For further details, refer to the [U-Mamba repository](https://github.com/U-Mamba/U-Mamba).

## Paper

If you use our work in your research, please cite our paper as follows:

```
@article{zhou2024ttt,
  title={TTT-Unet: Enhancing U-Net with Test-Time Training Layers for biomedical image segment},
  author={Zhou, Rong and Yuan, Zhengqing and Yan, Zhiling and Sun, Weixiang and Zhang, Kai},
  journal={arXiv preprint arXiv:2409.11299},
  year={2024}
}
```

## Acknowledgements

This project is based on U-Mamba. We acknowledge all the authors of the employed public datasets, as well as the authors of [U-Mamba](https://github.com/U-Mamba/U-Mamba) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for making their valuable code publicly available.
