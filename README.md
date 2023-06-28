# Temporal-Evolution-Inference-with-Compact-Feature-Representation-for-Talking-Face-Video-Compression

This repository contains the source code for the paper [Beyond Keypoint Coding: Temporal Evolution Inference with Compact Feature Representation for Talking Face
Video Compression](https://ieeexplore.ieee.org/abstract/document/9810732) by Bolin Chen, Zhao Wang, Binzhe Li, Rongqun Lin, Shiqi Wang, and Yan Ye.

The DCC keynote video presented by Dr. Yan Ye can be found in https://www.youtube.com/watch?v=7en3YYT1QfU.

The overall implementation codes and pretrained checkpoint can be found under following link: [OneDrive](https://portland-my.sharepoint.com/:u:/g/personal/bolinchen3-c_my_cityu_edu_hk/Eb3aT-rdLhRLh99hRN6tkzwBAkMomwlX3GSJxCUt1tY8ZQ?e=1wbsQf). 

### Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

In addition, please activate the VVC codec run
```
sudo chmod -R 777 vtm
```

### Training

To train a model on [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

When finishing the downloading and pre-processing the dataset, you can train the model,
```
python run.py
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder. To check the loss values during training see ```log.txt```. You can also check training data reconstructions in the ```train-vis``` subfolder. You can change the training settings in corresponding ```./config/vox-256.yaml``` file.

### Inference

To encode a sequence, please put the provided testing sequence in ```./testing_data/``` file and run
```
python Encoder.py
```
After obtaining the bistream, please run
```
python Decoder.py
```
For the testing sequence, it should be in the format of ```RGB:444``` at the resolution of ```256*256```.


### Evaluate

In ```./evaluate/multiMetric.py``` file, we provide the corresponding quality measures, including DISTS, LPIPS, PSNR and SSIM.


### Additional notes

#### Reference

The training code refers to the FOMM: https://github.com/AliaksandrSiarohin/first-order-model.

The arithmetic-coding refers to https://github.com/nayuki/Reference-arithmetic-coding.

#### Citation:

```
@INPROCEEDINGS{CHEN_DCC2022,
  author={Chen, Bolin and Wang, Zhao and Li, Binzhe and Lin, Rongqun and Wang, Shiqi and Ye, Yan},
  booktitle={2022 Data Compression Conference (DCC)}, 
  title={Beyond Keypoint Coding: Temporal Evolution Inference with Compact Feature Representation for Talking Face Video Compression}, 
  year={2022},
  volume={},
  number={},
  pages={13-22},
  doi={10.1109/DCC52660.2022.00009}}
```
