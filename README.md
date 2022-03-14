# Representation-Analysis-of-Neural-Networks-on-Biased-and-OOD-data.
## Motive


## Interpretation of Loss Functions
We have analysed the below listed loss functions on datasets provided in the next section. We have trained and tested on the those datasets with the following datasets:<br/>
1. Softmax Cross Entropy - **SCE**
2. Except Loss - **L<sub>1</sub> Loss**
3. Mean Squared Error - **L<sub>2</sub> Loss**
4. Negative Log Likelihood Loss - **NLL**
5. Binary Cross Entropy - **BCE**
6. Sum of Squares - **SoS** 


## Code Snippet
``` python
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as  optim
import numpy as np
import pandas as pd
import matplotlib.pypot as plt
from lossfunctions import LossFunctions

# Switching from cuda to cpu, if cuda is available
device = "cuda" if torch.cuda.is_available() else "cpu"

#--------MODEL--------#
# if training via pipelining
model = models.resnet18(pretrained=False).to(device)
# if the weights the are available proceed to use the following set of lines
# model = torch.load("\PATH")
# model.load_state_dict(torch.load("\PATH"))

#------DEFINE BATCH SIZE-----#
batch_size = 256

#------DATASET-DATALOADING-----#
traindata_transforms = transforms.Compose([
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testdata_transforms = transforms.Compose([
                        transforms.Resize((64,64)),  
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

traindata = datasets.CIFAR10(root="\PATH", train=True, transform=traindata_transforms, download=True) 
testdata = datasets.CIFAR10(root="\PATH", train=False, transform=traindata_transforms, download=True)

train_dl = DataLoader(traindata, batch_size, shuffle=False) 
test_dl = DataLoader(testdata, batch_size, shuffle=False)

#----METRICS----#
num_classes = 10
lf = LossFuntions(num_classes)
# You could define any one of the 6 defined loss functions 
# The defined loss functions are:
#   L1 = , L2 = , Mean-Sqaured-Error = , Sum-Of-Squares = , Cross-Entropy = , Binary Cross-Entropy =  
criterion = lf.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#----TRAIN/TEST MODEL----#
```
## Datasets Utilized
* [CIFAR-10 Dataset](https://pytorch.org/vision/stable/datasets.html#cifar)
* [MNIST Dataset](https://pytorch.org/vision/stable/datasets.html#mnist)
* [ImageNet-200 Dataset]()
* [Corrupted CIFAR-10 Dataset](https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT)
* [Colored MNIST Dataset](https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT)
* [Modified MNIST](https://www.kaggle.com/balraj98/adversarial-discriminative-domain-adaptation/notebook)
* [ImageNet-R (Renditions) Dataset](https://github.com/hendrycks/imagenet-r)

> **NOTE** - The link corresponding to the dataset name would redirect you to the respective dataset retrieval.

## Models Utilized
We have considered to analyse on loss functions and also performed the task of cross-dataset genealization by utilizing residual neural networks as our model encoder. We have considered the 18 layered version of ResNet i.e., [ResNet18](https://arxiv.org/abs/1512.03385)  


> **NOTE** - All the required libraries have been depicted in the **requirements.txt** file.

## Our Results
We proved results for a cycle of 3 iterations for the following settings:<br/> 
Optimizer - **Adam** with learning rate of 10<sup>-3</sup><br/>
Number of epochs - 70<br/>
Early stopping patience - 12<br/>

_tables with results and plots_

## Code References
* The custom ViT model have been cloned from [here](https://github.com/lucidrains/vit-pytorch)
* Dataloaders with respect to Colored MNIST and Corrupted CIFAR have been cloned from [here](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled). The published paper referral have been provided in the succeeding section.[[1]](#1)
* The obtained results have been depicted via CKA (Centered kernel alignement) plots. The code for plotting this representation have cloned from [here](https://github.com/AntixK/PyTorch-Model-Compare). The published paper referral have been provided in the succeeding section.[[2]](#2)
 
## Paper References
<a id="1">[1]</a> [Kim, Eungyeup, Jungsoo Lee, Juyoung Lee, Jihyeon Lee and Jaegul Choo. “Learning Debiased Representation via Disentangled Feature Augmentation.” ArXiv abs/2107.01372 (2021): n. pag.](https://arxiv.org/abs/2107.01372)<br/>
<a id="2">[2]</a> [Kornblith, Simon, Mohammad Norouzi, Honglak Lee and Geoffrey E. Hinton. “Similarity of Neural Network Representations Revisited.” ArXiv abs/1905.00414 (2019): n. pag.](https://arxiv.org/abs/1905.00414)


