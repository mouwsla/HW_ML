import torch 
from torchvision.datasets import CIFAR10
from torchvision import transforms 
from torch.utils.data import DataLoader

root = "./Data_10"

transformation = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 10

train_set = CIFAR10(train=True, transform=transformation, root= root,download = True)  
train_data_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

test_set = CIFAR10(train = False, transform = transformation, root = root, download = True)
test_data_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)