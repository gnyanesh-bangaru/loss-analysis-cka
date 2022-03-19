import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
class Train:
    def __init__(self,
                 optimizer,
                 loss,
                 epochs:int, 
                 modelname:str,
                 dataset:str):
        
        super(Train, self).__init__()
        self.optimizer = optimizer
        self.optimizer_name = str(optimizer)
        self.loss = loss
        self.epochs = epochs
        self.modelname = modelname
    
    def model_(self, modelname:str, num_classes:int, dataset:str):
        if modelname == 'resnet50':
            model = models.resnet50(pretrained=True).to(device)
            model.fc = nn.Sequential(
                        nn.Linear(2048, 1024, bias=True),
                        nn.Dropout(),
                        nn.Linear(1024, 512, bias=True),
                        nn.Dropout(),
                        nn.Linear(512, num_classes, bias=True)
                        ).to(device)
            return model
        elif modelname =='resnet18' and dataset=='\mnist':
            model = models.resnet18(pretrained=False).to(device)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
            model.fc = nn.Linear(512, num_classes, bias=True).to(device)
            return model
        elif modelname == 'resnet18':
            model = models.resnet18(pretrained=False).to(device)
            model.fc = nn.Linear(512, num_classes, bias=True).to(device)
            return model
        else:
            print('ERROR_model_or_dataset_is_not_found')
      
    def top1_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return acc
    
    @torch.no_grad()
    def test(self, model, test_dl, dataset):
        model.eval()
        for batch in test_dl:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = self.loss(outputs, labels)
            acc = self.top1_accuracy(outputs, labels)
        return loss, acc
    
    # Training Loop 
    def train(self, train_dl, test_dl, num_classes, filename, dataset):
        history = []
        since = time.time()
        model = self.model_(modelname=self.modelname,
                            num_classes = int(num_classes),
                            dataset = dataset)
        if self.optimizer_name == 'SGD':
            optimizer = self.optimizer(model.parameters(), lr=0.0001, momentum=0.93)
        else:
            optimizer = self.optimizer(model.parameters(), lr=0.001)
        
        for epoch in range(self.epochs):
            model.train()
            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []
            result = {}
            with tqdm(train_dl, unit="batch") as loop:
              for batch in loop:
                  inputs, labels = batch
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  outputs = model(inputs)
                  loss = self.loss(outputs, labels)
                  acc = self.top1_accuracy(outputs, labels)
                  train_acc.append(acc.cpu().detach().numpy())
                  train_loss.append(loss.cpu().detach().numpy())
                  loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                  loop.set_postfix(train_loss=np.average(train_loss),train_acc=np.average(train_acc))
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()
              
              test_losses,test_accu = self.test(model, test_dl, dataset)
              test_loss.append(test_losses.cpu().detach().numpy())
              test_acc.append(test_accu.cpu().detach().numpy())       
              result['train_loss'] = np.average(train_loss)
              result['train_acc'] = np.average(train_acc)
              result['test_loss'] = np.average(test_loss)
              result['test_acc'] = np.average(test_acc)
              print('\nEpoch',epoch,result)
              history.append(result)
            
        time_elapsed = time.time() - since
        print('Training Completed in {:.0f} min {:.0f} sec'.format(time_elapsed//60, time_elapsed%60))
        return history