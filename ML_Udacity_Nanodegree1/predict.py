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
import helper

parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('--inp_image',type = str, default = 'flowers/valid/1/image_06755.jpg', help = 'Path to dataset directory')
parser.add_argument('--checkpoint',type=str,default='trained1.pth',help='Checkpoint')
parser.add_argument('--gpu',type=str,default='cpu',help='GPU')
parser.add_argument('--json_class',type=str,default='cat_to_name.json',help='JSON of key value')
parser.add_argument('--top_k',type=int,default=5,help='Top k classes and probabilities')
args=parser.parse_args()


class_to_name= helper.load_class(args.json_class)

model=helper.load(args.checkpoint)
print(model)

vals=torch.load(args.checkpoint)

image = helper.process_image(args.inp_image)


helper.imshow(image)

probs, classes = helper.predict(args.inp_image, model, args.top_k, args.gpu)  

print(probs)
print(classes)

helper.display_image(args.inp_image, class_to_name, classes,probs)

