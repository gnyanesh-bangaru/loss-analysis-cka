#--------Generic_Libraries---------#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from glob import glob
from PIL import Image
import errno
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from six.moves import urllib
import gzip
import pickle


#---------Torch_Libraries----------#
import torch
from torchvision.datasets import CIFAR10, MNIST
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision.datasets import ImageFolder


#------User-Defined_Libraries------#
from CUDA_data_switch import get_default_device, DeviceDataLoader


CUDA_LAUNCH_BLOCKING=1
cudnn.benchmark = True


class LoadData:
    """
    Dataset naming convention:
        - MNIST = \mnist
        - Colored = \cmnist
        - MNIST-M = \mnist-m
        - CIFAR10 = \cifar10
        - Corrupted CIFAR = \cifar10c
        - Tiny ImageNet 200 = \Tiny-imagenet-200
        - ImageNet-R = \imagnet-r
    """
    
    def __init__(self,
                data_dir:str):
        super(LoadData, self).__init__()
        self.data_dir = data_dir
    
    
    def dataloader(self, 
                   dataset:str,
                   batch_size:int):
        
        global transforms, datasets, DataLoader, Compose, ToTensor, Resize, Normalize
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Transforms utilized for CIFAR10, Corrupted CIFAR, Colored MNIST, Tiny Imagenet-200 and Imagenet-R
        traindata_transforms = Compose([
                        Resize((64, 64)),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        testdata_transforms = Compose([
                        Resize((64, 64)),  
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        
        # Transforms utilized for MNIST dataset
        mnist_traindata_transforms = Compose([
                        Resize((64, 64)),
                        ToTensor(),
                        Normalize(0.5, 0.5)
                        ])
        mnist_testdata_transforms = Compose([
                        Resize((64, 64)),  
                        ToTensor(),
                        Normalize(0.5, 0.5)
                        ])
        device = get_default_device()
        print(device)
        
        if self.dataset=='\cifar10':
            num_classes = 10
            train_dataset = CIFAR10(download=True, 
                                    root = '.\data', 
                                    transform = traindata_transforms, 
                                    train = True)
            test_dataset = CIFAR10(download=True, 
                                    root = '.\data', 
                                    transform = testdata_transforms, 
                                    train = False)
            train_dl = DataLoader(train_dataset, self.batch_size, shuffle=True) 
            test_dl = DataLoader(test_dataset, self.batch_size, shuffle=True) 
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dl, test_dl, num_classes
        
        
        elif self.dataset == '\mnist':
            num_classes = 10
            train_dataset = MNIST(download=True, 
                                    root = './data', 
                                    transform = mnist_traindata_transforms, 
                                    train = True)
            test_dataset = MNIST(download=True, 
                                    root = './data', 
                                    transform = mnist_testdata_transforms, 
                                    train = False)
            train_dl = DataLoader(train_dataset, self.batch_size, shuffle=True) 
            test_dl = DataLoader(test_dataset, self.batch_size, shuffle=True) 
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dataloader, test_dataloader, num_classes
        
        
        elif self.dataset == '\Tiny-imagenet-200':
            num_classes = 200
            train_data = r'E:\Analysis_Work\CODE\data\tiny-imagenet-200\train'
            test_data = r'E:\Tiny-ImageNet\tiny-imagenet-200\val\images'
            train_dataset = datasets.ImageFolder(os.path.join(train_data), traindata_transforms)
            test_dataset = datasets.ImageFolder(os.path.join(test_data), testdata_transforms)
            train_dl = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            test_dl = DataLoader(test_dataset, self.batch_size, shuffle=True) 
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dataloader, test_dataloader, num_classes
        
        
        elif self.dataset == '\mnist-m':
            num_classes = 10
            class MNISTM(Dataset):
                """`MNIST-M Dataset."""
            
                url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"
            
                raw_folder = "raw"
                processed_folder = "processed"
                training_file = "mnist_m_train.pt"
                test_file = "mnist_m_test.pt"
            
                def __init__(self, root, mnist_root="data", train=True, transform=None, target_transform=None, download=False):
                    """Init MNIST-M dataset."""
                    super(MNISTM, self).__init__()
                    self.root = os.path.expanduser(root)
                    self.mnist_root = os.path.expanduser(mnist_root)
                    self.transform = transform
                    self.target_transform = target_transform
                    self.train = train  # training set or test set
            
                    if download:
                        self.download()
            
                    if not self._check_exists():
                        raise RuntimeError("Dataset not found." + " You can use download=True to download it")
            
                    if self.train:
                        self.train_data, self.train_labels = torch.load(
                            os.path.join(self.root, self.processed_folder, self.training_file)
                        )
                    else:
                        self.test_data, self.test_labels = torch.load(
                            os.path.join(self.root, self.processed_folder, self.test_file)
                        )
            
                def __getitem__(self, index):
                    """Get images and target for data loader.
            
                    Args:
                        index (int): Index
            
                    Returns:
                        tuple: (image, target) where target is index of the target class.
                    """
                    if self.train:
                        img, target = self.train_data[index], self.train_labels[index]
                    else:
                        img, target = self.test_data[index], self.test_labels[index]
            
                    # doing this so that it is consistent with all other datasets
                    # to return a PIL Image
                    img = Image.fromarray(img.squeeze().numpy(), mode="RGB")
            
                    if self.transform is not None:
                        img = self.transform(img)
            
                    if self.target_transform is not None:
                        target = self.target_transform(target)
            
                    return img, target
            
                def __len__(self):
                    """Return size of dataset."""
                    if self.train:
                        return len(self.train_data)
                    else:
                        return len(self.test_data)
            
                def _check_exists(self):
                    return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
                        os.path.join(self.root, self.processed_folder, self.test_file)
                    )
            
                def download(self):
                    """Download the MNIST data."""
                    # check if dataset already exists
                    if self._check_exists():
                        return
                    # make data dirs
                    try:
                        os.makedirs(os.path.join(self.root, self.raw_folder))
                        os.makedirs(os.path.join(self.root, self.processed_folder))
                    except OSError as e:
                        if e.errno == errno.EEXIST:
                            pass
                        else:
                            raise
                    # download pkl files
                    print("Downloading " + self.url)
                    filename = self.url.rpartition("/")[2]
                    file_path = os.path.join(self.root, self.raw_folder, filename)
                    if not os.path.exists(file_path.replace(".gz", "")):
                        data_ = urllib.request.urlopen(self.url)
                        with open(file_path, "wb") as f:
                            f.write(data_.read())
                        with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                            out_f.write(zip_f.read())
                        os.unlink(file_path)
                    # process and save as torch files
                    print("Processing...")
                    # load MNIST-M images from pkl file
                    with open(file_path.replace(".gz", ""), "rb") as f:
                        mnist_m_data = pickle.load(f, encoding="bytes")
                    mnist_m_train_data = torch.ByteTensor(mnist_m_data[b"train"])
                    mnist_m_test_data = torch.ByteTensor(mnist_m_data[b"test"])
                    # get MNIST labels
                    mnist_train_labels = datasets.MNIST(root=self.mnist_root, train=True, download=True).train_labels
                    mnist_test_labels = datasets.MNIST(root=self.mnist_root, train=False, download=True).test_labels
                    # save MNIST-M dataset
                    training_set = (mnist_m_train_data, mnist_train_labels)
                    test_set = (mnist_m_test_data, mnist_test_labels)
                    with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
                        torch.save(training_set, f)
                    with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
                        torch.save(test_set, f)
                    print("Done!")
            
            train_dataset = MNISTM(root = './data',
                                    mnist_root = './data',
                                    download=True,
                                    train=True,
                                    transform = traindata_transforms)
            test_dataset =  MNISTM(root = './data',
                                    mnist_root = './data',
                                    download=True,
                                    train=False,
                                    transform = testdata_transforms)
            train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dl, test_dl, num_classes
        
        
        elif self.dataset == '\imagenet-r':
            num_classes = 200       
            def train_val_dataset(dataset, val_split=0.20):
                train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
                datasets = {}
                datasets['train'] = Subset(dataset, train_idx)
                datasets['test'] = Subset(dataset, val_idx)
                return datasets
            dataset = ImageFolder(r'E:\Analysis_Work\data\ImageNet Renditions\imagenet-r', transform=traindata_transforms)
            datasets = train_val_dataset(dataset)
            train_dataset =datasets['train']
            test_dataset = datasets['test']            
            train_dl = DataLoader(datasets['train'], self.batch_size, shuffle=True) 
            test_dl = DataLoader(datasets['test'], self.batch_size, shuffle=True)             
            train_dataloader = DeviceDataLoader(train_dl, device)
            test_dataloader = DeviceDataLoader(test_dl, device)
            return train_dataloader, test_dataloader, num_classes
        
        
        elif self.dataset == '\imagenet-r-test':
            num_classes = 200
            dataset = ImageFolder(r'E:\Analysis_Work\data\ImageNet Renditions\imagenet-r', transform=testdata_transforms)
            dl = DataLoader(dataset, self.batch_size, shuffle=True) 
            dataloader = DeviceDataLoader(dl, device)
            return dataloader, num_classes
        
        
        elif self.dataset == '\cifar10c':
            num_classes = 10
            class IdxDataset(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
    
                def __len__(self):
                    return len(self.dataset)
    
                def __getitem__(self, idx):
                    return (idx, *self.dataset[idx])
    
            class ZippedDataset(Dataset):
                def __init__(self, datasets):
                    super(ZippedDataset, self).__init__()
                    self.dataset_sizes = [len(d) for d in datasets]
                    self.datasets = datasets
            
                def __len__(self):
                    return max(self.dataset_sizes)
            
                def __getitem__(self, idx):
                    items = []
                    for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
                        items.append(self.datasets[dataset_idx][idx % dataset_size])
            
                    item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]
            
                    return item
            
            
            class CIFAR10Dataset(Dataset):
                def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
                    super(CIFAR10Dataset, self).__init__()
                    self.transform = transform
                    self.root = root
                    self.image2pseudo = {}
                    self.image_path_list = image_path_list
            
                    if split=='train':
                        self.align = glob(os.path.join(root, 'align',"*","*"))
                        self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
                        self.data = self.align + self.conflict
            
                    elif split=='valid':
                        self.data = glob(os.path.join(root,split,"*", "*"))
            
                    elif split=='test':
                        self.data = glob(os.path.join(root, '../test',"*","*"))
            
                def __len__(self):
                    return len(self.data)
            
                def __getitem__(self, index):
                    attr = torch.LongTensor(
                        [int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
                    image = Image.open(self.data[index]).convert('RGB')
            
                    if self.transform is not None:
                        image = self.transform(image)
            
                    return image, attr, self.data[index]
        
            transforms_preprcs = {
                "\cifar10c": {
                    "train": traindata_transforms,
                    "test": testdata_transforms,
                },
            } 

            
            def get_dataset(dataset, data_dir, dataset_split, 
                            transform_split, 
                            percent, 
                            use_preprocess=None, 
                            image_path_list=None, 
                            use_type0=None, 
                            use_type1=None):
                
                dataset_category = dataset.split("-")[0]
                transform = transforms_preprcs[dataset_category][transform_split]
                dataset_split = "test" if (dataset_split == "eval") else dataset_split
                root = data_dir + "/cifar10c/{}".format(percent)
                dataset = CIFAR10Dataset(root=root, split=dataset_split, transform=transform, image_path_list=image_path_list, use_type0=use_type0, use_type1=use_type1)
            
                return dataset
            
            train_dataset = get_dataset(
                dataset= self.dataset,
                data_dir=self.data_dir,
                dataset_split="train",
                transform_split="train",
                percent='5pct',
                use_preprocess=True
            )
            test_dataset = get_dataset(
                dataset=  self.dataset,
                data_dir=self.data_dir,
                dataset_split="test",
                transform_split="test",
                percent='5pct',
                use_preprocess=True
            )
            train_target_attr = []
            for d in train_dataset.data:
                train_target_attr.append(int(d.split('_')[-2]))
            train_target_attr = torch.FloatTensor(train_target_attr)
            attr_dims = []
            attr_dims.append(torch.amax(train_target_attr).item() + 1)
            num_classes = attr_dims[0]
            train_dataset = IdxDataset(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            return train_dataloader, test_dataloader, num_classes
        
        elif self.dataset == '\cmnist':
            num_classes = 10
            class IdxDataset(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
    
                def __len__(self):
                    return len(self.dataset)
    
                def __getitem__(self, idx):
                    return (idx, *self.dataset[idx])
    
    
            class ZippedDataset(Dataset):
                def __init__(self, datasets):
                    super(ZippedDataset, self).__init__()
                    self.dataset_sizes = [len(d) for d in datasets]
                    self.datasets = datasets
            
                def __len__(self):
                    return max(self.dataset_sizes)
            
                def __getitem__(self, idx):
                    items = []
                    for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
                        items.append(self.datasets[dataset_idx][idx % dataset_size])
            
                    item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]
            
                    return item
            
            
            class CMNISTDataset(Dataset):
                def __init__(self,root,split,transform=None, image_path_list=None):
                    super(CMNISTDataset, self).__init__()
                    self.transform = transform
                    self.root = root
                    self.image2pseudo = {}
                    self.image_path_list = image_path_list
            
                    if split=='train':
                        self.align = glob(os.path.join(root, 'align',"*","*"))
                        self.conflict = glob(os.path.join(root, 'conflict',"*","*"))
                        self.data = self.align + self.conflict
            
                    elif split=='valid':
                        self.data = glob(os.path.join(root,split,"*"))            
                    elif split=='test':
                        self.data = glob(os.path.join(root, '../test',"*","*"))
            
            
                def __len__(self):
                    return len(self.data)
            
                def __getitem__(self, index):
                    attr = torch.LongTensor([int(self.data[index].split('_')[-2]),int(self.data[index].split('_')[-1].split('.')[0])])
                    image = Image.open(self.data[index]).convert('RGB')
            
                    if self.transform is not None:
                        image = self.transform(image)
            
                    return image, attr, self.data[index]
                    
            transforms_preprcs = {
                "\cmnist": {
                    "train": traindata_transforms,
                    "test": testdata_transforms,
                },
            } 
            
            def get_dataset(dataset, data_dir, dataset_split, 
                            transform_split, 
                            percent, 
                            use_preprocess=None, 
                            image_path_list=None, 
                            use_type0=None, 
                            use_type1=None):
                
                dataset_category = dataset.split("-")[0]
                transform = transforms_preprcs[dataset_category][transform_split]
                dataset_split = "test" if (dataset_split == "eval") else dataset_split
                root = data_dir + "/cmnist/{}".format(percent)
                dataset = CMNISTDataset(root=root,split=dataset_split,transform=transform, image_path_list=image_path_list)            
                return dataset

            train_dataset = get_dataset(
                dataset= self.dataset,
                data_dir=self.data_dir,
                dataset_split="train",
                transform_split="train",
                percent='5pct',
                use_preprocess=True
            )
            test_dataset = get_dataset(
                dataset=self.dataset,
                data_dir=self.data_dir,
                dataset_split="test",
                transform_split="test",
                percent='5pct',
                use_preprocess=True
            )
            train_target_attr = []
            for d in train_dataset.data:
                train_target_attr.append(int(d.split('_')[-2]))
            train_target_attr = torch.FloatTensor(train_target_attr)
            attr_dims = []
            attr_dims.append(torch.amax(train_target_attr).item() + 1)
            num_classes = attr_dims[0]
            train_dataset = IdxDataset(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True
            )
            return train_dataloader, test_dataloader, num_classes
                    
        
        else:
            print('Invalid_Entry')
            
