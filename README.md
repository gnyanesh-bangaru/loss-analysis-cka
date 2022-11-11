<p align="center">
  <img src="https://user-images.githubusercontent.com/67636257/184711258-ff61a650-1382-4c8d-8fd3-10fa8d6e838c.png" height="250" width="550">
</p>
<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" /></a>
    <a href="https://openreview.net/forum?id=BBSg-Wbsxfq" alt="OpenReview">
        <img src="https://img.shields.io/badge/Paper-OpenReview-red.svg" /></a>
    <a href="https://mega.nz/folder/B4FS0RDY#t6F8taiQ1QZ6Uxiodf-R4A" alt="Mega">
        <img src="https://img.shields.io/badge/Weights-Mega-brown.svg" /></a>
</p>
<h1 align="center">Representational Structure of Neural Networks Trained on Biased and Out-Of-Distribution Data</h1>

-------------------------------------------

<h2 align="center"> INTRODUCTION </h2> 

- This work is pertained towards addressing the representations attained by neural networks on standard image classification task. Neural Nets are observed to be
less robust to distributional shifts and pertain to certain levels of bias in representations. <br>
- We consider it to be predominant to know the appropriate objective function that is to be used for such data. There is very less literature that is focusing on choice of objective function and the representational structure attained by the neural nets.<br>
- Hence, we mainly focused on interpreting the representational structure of the intermediate neural networks when trained on such data. In order to withdraw the representations of the intermediate layers, we have utlized CKA-Centered Kernel Alignment to interpret similarities in the representations.<br>
- We examine the internal representational structure of convolutional-based neural networks (i.e., Residual Neural Networks) by training them on Generic, Biased and Out-of-Distribution data to understand the similarity in between the varying patterns, from the representations attained.<br>
- They were trained on different set of objective functions in multi-varied setting.<br>
- The details with respect to each dataset and individual loss function have been provided in the respective sections.<br>
- Our analysis reports that representations acquired by ResNets using Softmax Cross-Entropy (SCE) and Negative Log-Likelihood (NLL) as objectives are equally competent in providing superior performance and fine representations on OOD and biased data.

> **NOTE**:  - _This repository is implementation of our research work, the link for <ins>OpenReview paper</ins> is available as a badge above. Also, in order to reproduce our work, a badge for the <ins>Mega Link</ins> have been provided to download the trained weights for the network trained on various datasets and objective functions._ 

-------------

<h2 align="center"> CODE </h2> 

- The provided code can be implemented in various formats i.e., both by importing the downloaded snippets via class-based format or by parsing argument based methodology. 

- A tutorial on how to implement a specific setting with respect to individual loss function has been provided in a .ipynb format along with CKA visualizations.

- For visualizing CKA we have cloned code from this [particular repository](https://github.com/AntixK/PyTorch-Model-Compare).

``` python

#----------------REQUIRED LIBRARIES-----------------#
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as  optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lossfunctions import LossFunctions
from dataloader import LoadData
from train import Train
from CUDA_data_switch import get_default_device

device = get_default_device()

#------------------------MODEL-----------------------#
# if training via pipelining
model = models.resnet18(pretrained=False).to(device)
# if the weights the are available proceed to use the following set of lines
# model = torch.load("\PATH")
# model.load_state_dict(torch.load("\PATH"))


#----------------------DATASET-----------------------#
directory = r'\PATH'
dl = LoadData(directory)
# Pass the dataset name
dataset = '\cifar10' 

#--------------------SET BATCH SIZE------------------#
batch_size = 256

#---------------------DATALOADING--------------------#
train_dataloader, test_dataloader, num_classes = dl.dataloader(dataset, batch_size)

#----------------------METRICS-----------------------#
lf = LossFunctions(num_classes)
# You could define any one of the 6 user-defined loss functions
# The defined loss functions are:
#   L1 = expectation_loss, L2 = mse_loss, Sum-Of-Squares = sos_loss, 
#   Cross-Entropy = cross_entropy_loss, Binary Cross-Entropy = bce_loss, Negative Log-Likelihood = neg_loglike_loss 
# Declaring Loss Function and Optimizer
criterion = lf.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#--------------TRAINING AND EVALUATION---------------#
epochs = 15
modelname = 'resnet18'
# Train and Test
t = Train(optimizer = optimizer,
          loss = criterion,
          epochs = epochs,
          modelname = modelname,
          dataset = dataset,
          )
filename = dataset[1:] + modelname
history = t.train(
            train_dl = train_dataloader,
            test_dl = test_dataloader,
            num_classes = num_classes,
            filename = filename,
            dataset = dataset
            )
```

----------------------------------------------------------
<h2 align="center"> LOSS FUNCTIONS </h2> 
<h5 align="center"><ins>Table-1</ins></h5>
<h6 align="center">The below table illustrates all the objective functions that are experimented with in
this work.
</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184931453-157ca17b-ffbb-4cd0-bc23-c01055a641c8.jpg">
</p>


- We have categorized the implementation on specific loss functions into two categories and the denominations with respect to individual loss funcstions have been described in Table-1.

- The categorized loss functions are as follows-
  1) **Probablistic Loss Functions**: <br>
      a) Softmax Cross-Entropy (<i>L<sub>SCE</sub></i>)<br>
      b) Binary Cross-Entropy (<i>L<sub>BCE</sub></i>)<br>
      c) Negative Log-Likelihood (<i>L<sub>NLL</sub></i>)
    
  2) **Margin-Based Loss Functions**: <br>
       a) Mean Absolute Error (<i>L<sub>1</sub></i>)<br>
       b) Mean Squared Error (<i>L<sub>2</sub></i>)<br>
       c) Sum-of-Squares (<i>L<sub>SoS</sub></i>) 

----------------------------------------------------------

<h2 align="center"> DATASETS </h2> 

- The datasets have been imported from various sources and the appropriate links have been provided below.

    - CIFAR-10 and MNIST datasets have been imported via [torchvision pipeline](https://pytorch.org/vision/stable/datasets.html).<br>
    - Corrupted CIFAR and Colored-MNIST have been cloned and utilized as per this [work](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).<br>
    - MNIST-M have been cloned from this [work](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/mnistm.py).<br>
    - Tiny ImageNet have been downloaded from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip).<br>
    - ImageNet-R have been downloaded from [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar).

<h5 align="center"><ins>Table-2</ins></h5> 
<h6 align="center">The below table illustrates three kinds of data sets containing i.e., generic, biased,
and OOD samples.<br>’NA’ is specified to denote the absence of samples for that particular
split. The tag <i>X<sup>S</sup></i> denotes that the data set X is considered to be small.
</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184882764-3b447e99-67eb-4edc-aeb6-68ccf1c4599a.jpg"  height="250" width="500">
</p>

<br>

<h5 align="center"><ins>Figure-1</ins></h5>
<h6 align="center">The below figure represents the sample images with respect to each dataset on which our work was implemented on.
</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184953220-88e8f7fb-d8de-40d9-aa0a-16b50ba64818.jpg" >
</p>



----------------

<h2 align="center"> RESULTS </h2> 

- The results and appropriate reasoning have been elaborated in our work, in detail.<br>
- The below portrayed tables and visualizations are both generic and cross-data examinations with respect to individual setting.

<h5 align="center"><ins>Table-3</ins></h5>
<h6 align="center">The table below provides the empirical performance of the individual objective
functions for the generic, bias, and OOD data. The experiments were carried out without
any augmentations and did not use learnt weights (ImageNet1K or ImageNet21K) for the
training models. These experiments were conducted three times for each objective function
for a fair evaluation. The tabulated mean and standard deviation (mean ± std) in each cell
depicts the accuracy scores obtained after experimenting thrice with the ’test’ data. <b>Bold</b>
and underline represent the accuracy scores of first and second best performing models,
respectively.
</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184884510-5ff6e142-08d2-453c-97ca-0a778b5cf12f.jpg">
</p>

<br>

<h5 align="center">
<ins>Figure-2</ins></h5>
<h6 align="center">This figure illustrates CKA visualizations for probabilistic objective functions on
MNIST, C-MNIST, and MNIST-M. The First row, Second row, and Third row describe
the representations acquired by all objectives respectively for all three variants of data sets.
Each tile of the image consists of a color, indicating the strength of representations i.e., the
similarity of each layer representation. This corresponding color map is provided on the
right side of each image tile. For visualizing CKA, we considered all the layers of the neural
network (ResNet18) including activation, normalization, and fully-connected layers. The
matrix is formed by comparing the features acquired by each layer of the ResNet18 with
itself.
</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184944750-0287f4d6-8723-4d77-abed-e8861f59d8bc.jpg">
</p>

<br>

<h5 align="center"><ins>Table-4</ins></h5>
<h6 align="center">
These tables illustrate the performance of ResNets for both in-data and cross-data
generalization. All the results depicted in the table are test results that were experimented
thrice and the obtained the mean and standard deviation of test accuracy scores are noted.
The ’Train’ and ’Test’ technically mean that data were trained on the mentioned dataset and
the accuracy scores were obtained on the test datasets. We consider both the cases of training
ResNets with and without pre-trained weights and as a note, ImageNet1k weights are used
as pre-trained weights.
</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184930620-265af25a-f041-454a-b31b-12384983383c.jpg">
</p>

<br>

<h5 align="center">
<ins>Figure-3</ins></h5>
<h6 align="center">The figure visualizes the CKA plots for C-MNIST and MNIST-M on ResNet18 with and without ImageNet1k pre-trained weights. All these visualizations were obtained by training the model with <i>L<sub>SCE</sub></i> as objective.</h6>
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184946305-9f6d02c6-3880-444f-89f4-70ae68f3fa2b.jpg">
</p>
<br>


--------------------------------

<h2 align="center"> CITATION </h2> 
Please cite out work, if utilized.

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
