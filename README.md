# OD-GraspNet

---

#### We present a One-Shot Aggregation Dense Grasp Network for robotic grasping.

---
### Network structure contains the following methods:
- [One-shot aggregation dense network](https://arxiv.org/abs/1904.09730v1)
- [Cbam](https://arxiv.org/abs/1807.06521)
- [PixelShuffle](https://arxiv.org/abs/1609.05158)

---

## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

---
## Datasets

This repository supports both the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

---

## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

Example for Cornell dataset:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

Example for Jacquard dataset:

```bash
python train_network.py --dataset jacquard --dataset-path <Path To Dataset> --description training_jacquard --use-dropout 0 --input-size 300
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

Example for Cornell dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```

Example for Jacquard dataset:

```bash
python evaluate.py --network <Path to Trained Network> --dataset jacquard --dataset-path <Path to Dataset> --iou-eval
```

---

# Reference
### This repository is based on [Antipodal Robotic Grasping](https://github.com/skumra/robotic-grasping).

The repository contains the implementation of the Generative Residual Convolutional Neural Network (GR-ConvNet) from the paper:

#### Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

BibTeX entry:

```
@inproceedings{kumra2020antipodal,
  author={Kumra, Sulabh and Joshi, Shirin and Sahin, Ferat},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network}, 
  year={2020},
  pages={9626-9633},
  doi={10.1109/IROS45743.2020.9340777}}
}
```
