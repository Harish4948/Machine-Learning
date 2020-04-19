import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets,transforms,models
import numpy as np
import json
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb
import helper
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('--data_dir',type = str, default = './flowers', help = 'Path to dataset directory')
parser.add_argument('--save_dir',type = str, default = './', help = 'Path to checkpint save directory')
parser.add_argument('--arch',type = str, default = 'vgg', help = 'Tranfer learning model')
parser.add_argument('--lr',type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--epochs',type = int, default = 10, help = 'epochs')
parser.add_argument('--hidden_layers',type = int, default = 500, help = 'hidden units')
parser.add_argument('--gpu',type = str, default = 'cpu', help = 'GPU or CPU')

args=parser.parse_args()


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
validate_transforms=transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validate_data=datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader=torch.utils.data.DataLoader(validate_data, batch_size=64)

if args.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
elif args.arch == 'resnet':
    input_size = 2048
    model = models.alexnet(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
model.classifier=nn.Sequential(nn.Linear(input_size, args.hidden_layers),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(args.hidden_layers,102),
                                 nn.LogSoftmax(dim=1))
print(model)

criterion=nn.NLLLoss()
device=args.gpu
optimizer=optim.Adam(model.classifier.parameters(), args.lr)
loss,accuracy=helper.validate(model,criterion,testloader,device)
print(f"loss: {loss} \n Accuracy: {accuracy}")
epochs=args.epochs
model=helper.train(model,optimizer,criterion,epochs,trainloader,validloader,device)
helper.accuracy(model,testloader,device)
helper.save(model,train_data,args.arch,input_size,args.hidden_layers,epochs,args.lr)




