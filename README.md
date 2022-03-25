# Representation-Analysis-of-Neural-Networks-on-Biased-and-OOD-data.

**_The github repository is yet to be updated. The required changes will be made soon_**

This repo is implementation of our [work](https://openreview.net/forum?id=BBSg-Wbsxfq).

## Getting Started

> **NOTE** - All the required libraries are in the **requirements.txt** .
> 
> A tutorial for our work has been provided at --**tutorials.ipynb**.
> 
> The six variant loss functions are provided in **lossfuntions.py** and training procedure is in **train.py**.


## Reproducibility 

To reproduce our work, please refer the [mega link](https://mega.nz/folder/B4FS0RDY#t6F8taiQ1QZ6Uxiodf-R4A) to download the trained weights for the network trained on various datasets and objective functions. 


### Code Snippet
``` python
#----------------Required Libraries-----------------#
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as  optim
import numpy as np
import pandas as pd
import matplotlib.pypot as plt
from lossfunctions import LossFunctions
from dataloader import LoadData

device = "cuda" if torch.cuda.is_available() else "cpu"

#------------------------MODEL-----------------------#
# if training via pipelining
model = models.resnet18(pretrained=False).to(device)
# if the weights the are available proceed to use the following set of lines
# model = torch.load("\PATH")
# model.load_state_dict(torch.load("\PATH"))
#----------------------DATASET-----------------------#
directory = r'\PATH'
dl = LoadData(directory)
#--------------------SET BATCH SIZE------------------#
batch_size = 256

#---------------------DATALOADING--------------------#
train_dataloader, test_dataloader, num_classes = dl.dataloader(dataset, batch_size)

#----------------------METRICS-----------------------#
lf = LossFuntions(num_classes)
# You could define any one of the 6 defined loss functions

# The defined loss functions are:
#   L1 = expectation_loss, L2 = mse_loss, Sum-Of-Squares = sos_loss, 
#   Cross-Entropy = cross_entropy_loss, Binary Cross-Entropy = bce_loss, Negative Log-Likelihood = neg_loglike_loss 

criterion = lf.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#--------------TRAINING AND EVALUATION---------------#
model.train()
model.eval()
```
### Models and Datasets utilized
We have utilized **ResNet18** and **ResNet50** as our model encoders, and these were trained on the following data.
| Generic Data | Biased Data | Out-Of-Distribution Data |
| --------------- | --------------- | --------------- |
| CIFAR-10  | Corrupted CIFAR-10  |-|
| MNIST Dataset | Colored MNIST  | Modified MNIST |
| ImageNet-200  |-| ImageNet-R (Renditions) |


## Code References
* Dataloaders with respect to Colored MNIST and Corrupted CIFAR have been cloned from [here](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled). The published paper referral have been provided in the succeeding section.[[1]](#1)
* The obtained results have been depicted via CKA (Centered kernel alignement) plots. The code for plotting this representation have cloned from [here](https://github.com/AntixK/PyTorch-Model-Compare). The published paper referral have been provided in the succeeding section.[[2]](#2)

## Citation 
Please cite out work if you have used it.

```
@unpublished{        
anonymous2022representation,        
title={Representation Analysis of Neural Networks Trained on Biased and Out-Of-Distribution Data},        
author={Anonymous},        
journal={OpenReview Preprint},        
year={2022},        
note={anonymous preprint under review}    
}

```
