#--------Generic_Libraries---------#
import argparse


#---------Torch_Libraries----------#
import torch.optim as  optim
import torch.nn.functional as F
import torch.nn as nn


#------User-Defined_Libraries------#
from lossfunctions import LossFunctions 
from train import Train
from dataloader import LoadData

"""
For approporiate denominations and naming conventions, refer lossfunction.py and dataloader.py files.
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Code Implementation for Representation Analysis for biased and OOD data")
    parser.add_argument(
        "-dir",
        '--directory', 
        metavar="DIR", 
        type=str
        help="Path to dataset"
        )
    parser.add_argument(
        '-a',
        "--arch",
        metavar="ARCH",
        default="resnet18",
        type=str
        help="Model Architecture",
        )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
        help="Mini-batch size (default: 64)",
        )
    parser.add_argument(
        '-d',
        '--dataset',
        metavar='D',
        type=str
        default='\cifar10',
        help='Specify the dataset name (according to dataloader)'
        )
    parser.add_argument(
        '--epochs',
        metavar='E',
        default=30,
        type=int
        help="epochs for training (default-30)"
        )
    parser.add_argument(
        '-optim',
        '--optimizer',
        metavar='O',
        default=optim.Adam,
        type=str
        help="optimizer (default-Adam)"
        )
    parser.add_argument(
        '-loss',
        '--loss-function',
        metavar='L',
        default=None,
        help='loss function (default:categorical_cross_entropy)'
        )
    args = parser.parse_args()
    return args


def get_data():
    dl = LoadData(args.dir)
    batch_size = args.b
    dataset = args.dataset
    train_dataloader, test_dataloader, num_classes = dl.dataloader(dataset, batch_size)
    return train_dataloader, test_dataloader, num_classes


def get_loss():
    _, _, num_classes = get_data()
    lf = LossFunctions(num_classes)
    loss = lf.args.loss
    return loss


args = parse_args()
def main():    
    train_dataloader, test_dataloader, num_classes = get_data()
    loss = get_loss()
    optimizer = optim.Adam
    epochs = args.epochs
    modelname = args.a
    # Training 
    t = Train(optimizer = optimizer,
              loss = loss,
              epochs = epochs,
              modelname = modelname,
              dataset = args.d,
              )
    filename = args.d[1:] + modelname
    
    history = t.train(
                train_dl = train_dataloader,
                test_dl = test_dataloader,
                num_classes = num_classes,
                filename = filename,
                dataset = args.d
                )
    return history
