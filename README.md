# Representation-Analysis-of-Neural-Networks-on-Biased-and-OOD-data.
## Motive

## Interpretation of Loss Functions
We have analysed the below listed loss functions on datasets provided in the next section. We have trained and tested on the those datasets with the following datasets:<br/>
1. Softmax Cross Entropy - SCE
2. L<sub>1</sub> Loss - Except Loss
3. L<sub>2</sub> Loss - Mean squred error
4. Negative Log Likelihood Loss - NLL
5. Binary Cross Entropy - BCE 
6. Sum of Squares - SoS 

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

## Code References

## Paper References



