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
We have considered to analyse on loss functions by having residual networks as our model encoder and the cross dataset generalization by both residual networks and visual transformers. We have considered the 18 layered version of ResNet i.e., [ResNet18](https://arxiv.org/abs/1512.03385)  and [ViT](https://arxiv.org/abs/2010.11929)-custom model.


## Requirements
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&url=<https://www.python.org/>&logo=python&logoColor=ffdd54)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)<br/>
We have used [PyTorch](https://pytorch.org/) framework, the utilized libraries from the same have been depicted below:<br/>

#### Required Libraries for computing loss:
* _torch_ <br/>
* _numpy_

#### Libraries utilized for our overall work:
* _torch_ version == 1.9.0 <br/>
* _torchvision_ version == 0.10.0<br/>
* _numpy_ version == 1.18.5<br/>
* _pandas_ version == 1.1.3<br/>
* _matplotlib_ version == 3.3.2<br/>
* _sklearn_ version == 0.23.2


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


