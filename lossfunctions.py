import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn


cudnn.benchmark = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class LossFunctions:
    def __init__(self, num_classes):
        super(LossFunctions, self).__init__()
        self.num_classes = int(num_classes)
    
    # Log (cross-entropy) Loss
    def cross_entropy_loss(self, x, y):
        loss = F.cross_entropy(x, y)
        return loss

    # Sum of sqaures
    def sos_loss(self, x, y):
        ones = torch.sparse.torch.eye(self.num_classes).to('cuda:0')
        y = ones.index_select(0, y)
        m = nn.Softmax(dim=1)
        criterion = nn.MSELoss(reduction='sum')
        loss = criterion(m(x), y)
        return loss
    
    # Mean Sqaured loss - L2 loss 
    def mse_loss(self, x, y):
        ones = torch.sparse.torch.eye(self.num_classes).to('cuda:0')
        y = ones.index_select(0, y)
        m = nn.Softmax(dim=1)
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(m(x), y)
        return loss
    
    # Negative log likelihood = logarithmic softmax
    def neg_loglike_loss(self, x, y):
        m = nn.LogSoftmax(dim=1)
        nll_loss = nn.NLLLoss()
        loss = nll_loss(m(x), y)
        return loss
    
    # Expectation Loss - L1 loss - Mean absoulte error
    def expectation_loss(self, x, y):
        ones = torch.sparse.torch.eye(self.num_classes).to('cuda:0')
        y = ones.index_select(0, y)
        m = nn.Softmax(dim=1)
        loss = F.l1_loss(m(x), y)
        return loss
    
    def bce_loss(self, x, y):
        ones = torch.sparse.torch.eye(self.num_classes).to('cuda:0')
        y = ones.index_select(0, y)
        loss = F.binary_cross_entropy_with_logits(x, y)
        return loss
