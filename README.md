<p align="center">
  <img src="https://user-images.githubusercontent.com/67636257/184711258-ff61a650-1382-4c8d-8fd3-10fa8d6e838c.png" height="250" width="550">
</p>
<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" /></a>
    <a href="https://openreview.net/forum?id=BBSg-Wbsxfq" alt="OpenReview">
        <img src="https://img.shields.io/badge/Paper-OpenReview-red.svg" /></a>
          </a>
</p>
<h1 align="center">Representation Analysis of <br>Neural Networks using Biased and OOD Data </h1>


This repo is implementation of our [work](https://openreview.net/forum?id=BBSg-Wbsxfq). The work is yet to be updated.

<h2 align="center"> Loss Functions </h2> 
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184931453-157ca17b-ffbb-4cd0-bc23-c01055a641c8.jpg">
</p>
<h6 align="center">Table-1: <ins>The above table illustrates all the objective functions that are experimented with in
this work.</ins>
</h6>




<h2 align="center"> Datasets </h2> 
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184882764-3b447e99-67eb-4edc-aeb6-68ccf1c4599a.jpg"  height="250" width="500">
</p>
<h6 align="center">Table-2: 
<ins>The above table illustrates three kinds of data sets containing i.e., generic, biased,
and OOD samples.</ins><br>’NA’ is specified to denote the absence of samples for that particular
split. The tag <i>X<sup>S</sup></i> denotes that the data set X is considered to be small.
</h6>

<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184953220-88e8f7fb-d8de-40d9-aa0a-16b50ba64818.jpg" >
</p>
<h6 align="center"><ins>Figure-1:</ins> The above figure represents the sample images with respect to each dataset on which our work was implemented on.
</h6>




<h2 align="center"> Results </h2> 
<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184884510-5ff6e142-08d2-453c-97ca-0a778b5cf12f.jpg">
</p>
<h6 align="center">Table-3: 
<ins>The table above provides the empirical performance of the individual objective
functions for the generic, bias, and OOD data. The experiments were carried out without
any augmentations and did not use learnt weights (ImageNet1K or ImageNet21K) for the
training models. These experiments were conducted three times for each objective function
for a fair evaluation. The tabulated mean and standard deviation (mean ± std) in each cell
depicts the accuracy scores obtained after experimenting thrice with the ’test’ data. <b>Bold</b>
and underline represent the accuracy scores of first and second best performing models,
respectively.</ins>
</h6>

<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184944750-0287f4d6-8723-4d77-abed-e8861f59d8bc.jpg">
</p>
<h6 align="center">
<ins>Figure-2:</ins> This figure illustrates CKA visualizations for probabilistic objective functions on
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
<img src= "https://user-images.githubusercontent.com/67636257/184930620-265af25a-f041-454a-b31b-12384983383c.jpg">
</p>
<h6 align="center">Table-4: 
<ins>These tables illustrate the performance of ResNets for both in-data and cross-data
generalization. All the results depicted in the table are test results that were experimented
thrice and the obtained the mean and standard deviation of test accuracy scores are noted.
The ’Train’ and ’Test’ technically mean that data were trained on the mentioned dataset and
the accuracy scores were obtained on the test datasets. We consider both the cases of training
ResNets with and without pre-trained weights and as a note, ImageNet1k weights are used
as pre-trained weights.</ins>
</h6>

<p align="center">
<img src= "https://user-images.githubusercontent.com/67636257/184946305-9f6d02c6-3880-444f-89f4-70ae68f3fa2b.jpg">
</p>
<h6 align="center">
<ins>Figure-3:</ins> The figure visualizes the CKA plots for C-MNIST and MNIST-M on ResNet18
with and without ImageNet1k pre-trained weights. All these visualizations were obtained by
training the model with <i>L<sub>SCE</sub></i> as objective.
</h6>


<h2 align="center"> Citation </h2> 
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
