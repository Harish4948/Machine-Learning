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

# data_dir = 'flowers'
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
# test_dir = data_dir + '/test'



# train_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                        transforms.RandomResizedCrop(224),
#                                        transforms.RandomHorizontalFlip(),
#                                        transforms.ToTensor(),
#                                        transforms.Normalize([0.485, 0.456, 0.406],
#                                                             [0.229, 0.224, 0.225])])

# test_transforms = transforms.Compose([transforms.Resize(255),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406],
#                                                            [0.229, 0.224, 0.225])])
# validate_transforms=transforms.Compose([transforms.Resize(255),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406],
#                                                            [0.229, 0.224, 0.225])])


# train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
# test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
# validate_data=datasets.ImageFolder(test_dir, transform=test_transforms)

# trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
# validloader=torch.utils.data.DataLoader(validate_data, batch_size=64)



def load_class(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    print(len(cat_to_name))
    return cat_to_name


# dir_name = '.'
# target_file_name = 'workspace_archive.tar'
# # List of files/directories to ignore
# ignore = {'.ipynb_checkpoints', '__pycache__', target_file_name}

# make_tar_file(dir_name, target_file_name, ignore)




# model=models.vgg16(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model



# model.classifier = nn.Sequential(nn.Linear(25088, 500),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.5),
#                                  nn.Linear(500,102),
#                                  nn.LogSoftmax(dim=1))


def validate(model,criterion,validloader,device):
    model.to(device)
    loss=0
    accuracy=0
    for inputs,labels in validloader:
        inputs,labels=inputs.to(device),labels.to(device)
        op=model(inputs)
        loss+=criterion(op,labels)
        probabilities=torch.exp(op)
        equality=(labels.data==probabilities.max(dim=1)[1])
        accuracy+=equality.type(torch.FloatTensor).mean()
    return loss,accuracy


def train(model,optimizer,criterion,epochs,trainloader,testloader,device):
    model.to(device)
    #optimizer=optim.Adam(model.classifier.parameters(), lr=0.001)
    #criterion=nn.NLLLoss()
    running_loss=0
    #train_losses, test_losses = [], []
    steps = 0
    running_loss = 0
    print_every = 40
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to("cuda"), labels.to("cuda")
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
    return model#     else:
#         test_loss* = 0
#         accuracy = 0
        
#         with torch.no_grad():
#             model.eval()
#             for images, labels in testloader:
#                 images, labels = images.to("cuda"), labels.to("cuda")
#                 log_ps = model(images)
#                 test_loss += criterion(log_ps, labels)
                
#                 ps = torch.exp(log_ps)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy += torch.mean(equals.type(torch.FloatTensor))
#         model.train()
                
#         train_losses.append(running_loss/len(trainloader))
#         test_losses.append(test_loss/len(testloader))

#         print("Epoch: {}/{}.. ".format(epoch, epochs),
#               "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
#               "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
#               "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
#         running_loss = 0
#             model.train()



# TODO: Do validation on the test set

def accuracy(model,testloader,device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        accuracy=0
        for images,labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits=model(images)
            probabilities=torch.exp(logits)
            equality = (labels.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        print("Testing Accuracy:",accuracy/len(testloader))

def save(model,training_data,transfer_model,input_layers,hidden_layers,epochs,lr):
    model.class_to_idx = training_data.class_to_idx
    checkpoint = {'transfer_model':transfer_model,
                    'input_layers':input_layers,
                    'hidden_layers':hidden_layers,
                    'class_to_idx': model.class_to_idx,
                    'model_state_dict': model.state_dict(),
                    "epochs":epochs,
                    "learning_rate":lr
                    }
    torch.save(checkpoint, 'trained2.pth')

def load(filepath):
    checkpoint = torch.load(filepath)
    
    if(checkpoint['transfer_model']=='vgg'):
        model = models.vgg16(pretrained=True)
    elif(checkpoint['transfer_model']=='resnet'):
        model=model.resnet152(pretrained=True)
    else:
        print("Unsupported model")  

    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_layers'], checkpoint["hidden_layers"]),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(checkpoint["hidden_layers"],102),
                                 nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# model = load('trained.pth')

# print(model)




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image=Image.open(image)
    pil_image=pil_image.resize(size=(256,256))
    
    bottom = (pil_image.height-224)/2    
    left = (pil_image.width-224)/2
    right = left + 224
    top= bottom + 224
    
    pil_image = pil_image.crop((left, bottom, right, top))
    
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# image = process_image('flowers/test/1/image_06754.jpg')
# imshow(image)





def predict(image_path, model, topk=5,device="cuda"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image=process_image(image_path)
    model.to(device)
    if device == "cuda":
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    output = model.forward(image)
    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk)
    
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes
    
# probs, classes = predict('flowers/test/97/image_07708.jpg', model)   
# print(probs)
# print(classes)




def display_image(image_path,cat_to_name,classes,probs):
    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)

    image = process_image(image_path)

    imshow(image, plot_1, title="flower_title");
    flower_names = [cat_to_name[i] for i in classes]
    plt.subplot(2,1,2)
    sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0]);
    plt.show()