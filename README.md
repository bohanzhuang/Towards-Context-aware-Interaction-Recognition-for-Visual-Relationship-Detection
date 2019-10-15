# Introduction
This is the pytorch implementation of the paper "Towards Context-Aware Interaction Recognition for Visual Relationship Detection"


***If you use this code in your research, please cite our paper:***

```
@InProceedings{Zhuang_2017_ICCV,
author = {Zhuang, Bohan and Liu, Lingqiao and Shen, Chunhua and Reid, Ian},
title = {Towards Context-Aware Interaction Recognition for Visual Relationship Detection},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```

## Dataset and Evaluation Metrics
Please download the VRD dataset and the corresponding evaluation metrics from https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection

Note that we first extract the data into individual files for training/testing. 


## Code
The code are written using Pytorch.

__utils.py__: provide necessary functions   
__new_layers.py__: provide self-defined layers     
__train.py__: main file, implementing training and testing  
__config.yaml__: define the necessary hyperparameters (e.g., data directory, GPU), please modify this file  
__model.py__: define network structures  


**If you want to evaluate the context-aware model independently without attention, find the code provided in ./no_attention subfolder.**
  

## Training

```
python train.py

```

## Testing
Follow the evaluation instructions in "https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection". Please extract the features by yourself after training and evaluate using relationship_phrase_detection.m and predicate_detection.m, respectively.


## Copyright

Copyright (c) Bohan Zhuang. 2017

** This code is for non-commercial purposes only. For commerical purposes,
please contact Chunhua Shen <chhshen@gmail.com> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

