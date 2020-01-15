# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:39:10 2020

@author: natas
"""

# tutorial from https://medium.com/diving-in-deep/facial-keypoints-detection-with-pytorch-86bac79141e4




#=========== IMPORTS ==============================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from torch import optim
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix



#===================================================


IMG_SIZE = 48 #image size is 48x48 pixels
data_dir = 'C:\\Users\\natas\\Desktop\\Facial Expression Recognition\\data\\'
MODEL_STORE_PATH = 'C:\\Users\\natas\\Desktop\\Facial Expression Recognition'

#Parameters
num_epochs = 25
num_classes = 6
batch_size = 128
learning_rate = 0.001

# percentage of training set to use as validation
valid_size = 0.2


#------- TRAINING DATA -------------

#read in trainingdata to dataframe
train_data = pd.read_csv(data_dir + 'train.csv')

#view training data
#print(train_data.T)

#get info on datatype etc.
#print(train_data.info())

# ----------------------------------

# ------ TEST DATA -----------------

#read in test data to dataframe
test_data = pd.read_csv(data_dir + 'test73.csv')

#view test data
#print(test_data)

#get info on datatype etc.
#print(test_data.info())

# ----------------------------------
# Get 20 random number for 20 random images used in the human experiment
random_images = np.random.randint(low=1, high=636, size=4)
print("random image id's")
print(random_images)

    
#function used to view specific images from data
def show_images(df, indxs, ncols=5, figsize=(15,10)):

    '''
    Args:
    df (DataFrame): data (M x N)
    indxs (iterators): list, Range, Indexes
    ncols (integer): number of columns to display
    figsize (float, float): width, height of image in inches
    '''
    
    #initialise plot and calculate number of rows
    plt.figure(figsize=figsize)
    nrows = len(indxs) // ncols + 1
    labels = []
    
    # plot images 
    for i, idx in enumerate(indxs):
        image = np.fromstring(df.loc[idx, 'pixels'], sep=' ').astype(np.float32).reshape(-1, IMG_SIZE)
        label = df.loc[idx, 'emotion']
#        plt.subplot(nrows, ncols, i + 1)
        # plt.title('__________________')
        plt.axis('off')
#        plt.tight_layout()
        plt.imshow(image, cmap='gray')
        labels.append(label)
        plt.savefig('Sample{}Label{}'.format(idx,label))

    print('Labels: {}.jpg'.format(labels))
    print('DONE WITH SAVING IMAGES')
    

# example of use:    
# show images from train data (show 9 images) (you can also input a list of image id's example [0,1,2,3045])
# show_images(test_data, random_images)
    
    
def plot_confusion_matrix(target, prediction, name):
   
    # create confusion matrix of valid:
    conf_matrix = confusion_matrix(target, prediction)
    
    #normalize confusion matrix
    cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    #round numbers in confusion matrix
    cm = np.around(cm, decimals=2)
        
    # confusion matrix dataframe
    df_cm_norm = pd.DataFrame(cm, index=["Angry", "Neutral", "Fear", "Happy", "Sad", "Surprise"], columns=["Angry", "Neutral", "Fear", "Happy", "Sad", "Surprise"])
    
    # plot confusion matrix
    sn.set(font_scale=1) # for label size
    sn.heatmap(df_cm_norm, annot=True, annot_kws={"size": 10}, linewidths=.2, cmap='PuBuGn') 
    plt.title('Model')
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.savefig(name)
    plt.show()
        
        
    

def plots(train_losses, valid_losses, train_accuracies, valid_accuracies, valid_loss, valid_accuracy):

    plt.plot(np.arange(1,num_epochs+1),valid_accuracies, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)  
    plt.xticks(np.arange(1,num_epochs+1))
    
    plt.plot(np.arange(1,num_epochs+1),train_accuracies, label='Training')
    plt.legend();
    plt.savefig('Accuracy{}.jpg'.format(valid_accuracy))
    plt.show()
    
    plt.plot(np.arange(1,num_epochs+1),valid_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1,num_epochs+1))
    
    plt.plot(np.arange(1,num_epochs+1),train_losses, label='Training')
    plt.legend();
    plt.savefig('Loss{}.jpg'.format(valid_loss))
    plt.show()
    
    


# CREATE DATASET CLASS
class FaceExpressionDataset(Dataset):
    
    def __init__(self, dataframe, train=True, transform=None):
        '''
        Args:
            dataframe (DataFrame): data in pandas dataframe format.
            train (Boolean) : Whether to train on data or not. Default is True
            transform (callable, optional): Optional transform to be applied on 
            sample
        '''
        self.dataframe = dataframe
        self.train = train
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx): 
        image = np.fromstring(self.dataframe.iloc[idx, -1], sep=' ')\
                .astype(np.float32).reshape(-1, IMG_SIZE)
        
        # if data should be trained, get labels
        if self.train:
            labels = self.dataframe.iloc[idx, :-1].values.astype(np.float32)
            
        else:
            labels = None

        sample = {'image': image, 'labels': labels}
        
        #apply transformation
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
# CREATE NORMALIZATION SCHEME
class Normalize(object):
    '''Normalize input images'''
    
    # scale image to [0, 1]
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        return {'image': image / 255., 'labels': labels} 
    
# CONVERT TO TENSOR
class ToTensor(object):

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.reshape(1, IMG_SIZE, IMG_SIZE)
        image = torch.from_numpy(image)
        
        # if there are labels, convert labels to long tensor
        if labels is not None: 
            labels = torch.from_numpy(labels)
            labels = labels.type(torch.LongTensor) #convert from float to long tensor
            
            return {'image': image, 'labels': labels}
            
        else:
            return {'image': image}
            
    
# create train_loader and valid_loaders
def prepare_train_valid_loaders(trainset, valid_size=valid_size, batch_size=batch_size):
    
    '''
    Split trainset data and prepare DataLoader for training and validation
    
    Args:
        trainset (Dataset): data 
        valid_size (float): validation size, default=0.2
        batch_size (int) : batch size, default=128
    ''' 
    
    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches (gets randomly assigned)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler)
    
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler)
    
    return train_loader, valid_loader


train_df = train_data
test_df = test_data  

# Define a transform to normalize the data
tsfm = transforms.Compose([Normalize(), ToTensor()])

# Load the training data and test data
trainset = FaceExpressionDataset(train_df, transform=tsfm)
testset = FaceExpressionDataset(test_df, transform=tsfm)

# prepare data loaders
train_loader, valid_loader = prepare_train_valid_loaders(trainset, 
                                                         valid_size,
                                                         batch_size)
# prepare test loader
test_loader = torch.utils.data.DataLoader(testset, batch_size=74)



# -------- CREATE Convolutional Network Model -----------------------------
        
# creating the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        #create first layer (number of input channels, number of output channels, kernel_size is filter proportions)
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        #create second layer
        self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        #create third layer
        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        #dropout layer to avoid overfitting model
        self.drop_out = nn.Dropout()
        
        #linear to create a fully connected layer
        self.fc1 = nn.Linear(6*6*96, 200)
        self.fc2 = nn.Linear(200, 6)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1) #flatten output
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

# ======= TRAIN THE NETWORK ======================
            
device = torch.device('cpu')

#train using ConvNet
model = ConvNet()

model = model.to(device)

#loss operation used to calculate the loss (crossentropyloss combines softmax and entropy loss function)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(train_loader, valid_loader, test_loader, model, criterion, optimizer, 
          n_epochs=num_epochs, saved_model='model.pt'):
    '''
    Train the model
    
    Args:
        train_loader (DataLoader): DataLoader for train Dataset
        valid_loader (DataLoader): DataLoader for valid Dataset
        model (nn.Module): model to be trained on
        criterion (torch.nn): loss funtion
        optimizer (torch.optim): optimization algorithms
        n_epochs (int): number of epochs to train the model
        saved_model (str): file path for saving model
    
    Return:
        tuple of train_losses, valid_losses, train_accuracies, valid_accuracies and conf_matrix(confusion matrix)
    '''
    
     # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf # set initial "min" to infinity
    
    # list of losses pr epoch
    train_losses = []
    valid_losses = []
    
    # list accuracy pr epoch
    train_accuracies = []
    valid_accuracies = []
    
    # total list of predictions
    train_prediction = []
    valid_prediction = []
    
    # total list of targetlabels
    train_target = []
    valid_target = []
#    test_target = []
#    
    epoch_counter = 0
    
    for epoch in range(n_epochs):
        epoch_counter += 1
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # list of accuracies pr batch
        train_accuracy = []
        valid_accuracy = []
        
        
        ###################
        # train the model #
        ###################
        
        model.train() # prep model for training
        
        
        for batch in train_loader:
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device))
            
            # labels
            labels = batch['labels'].to(device)
            
            # get correct dimensions for CrossEntropyLoss function
            labels = labels.squeeze_()
            
            # calculate the loss
            loss = criterion(output, labels)
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update running training loss
            train_loss += loss.item()*batch['image'].size(0)
            
            # calculate training accuracy
            total = labels.size(0)
            _, predicted = torch.max(output, 1)
            
            #collect predictions to list
            train_prediction.extend(predicted.tolist())
            
            correct = (predicted == labels).sum().item()
            # add accuracy to list of accuracies pr batch
            train_accuracy.append(correct/total)
            
            # keep track of target labels
            train_target.extend(labels.tolist())
 
        
            
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for batch in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device))
            
            # labels
            labels = batch['labels'].to(device)
            
            # get correct dimensions for CrossEntropyLoss function
            labels = labels.squeeze_()
            
            # calculate the loss
            loss = criterion(output, labels)
            
            # update running validation loss 
            valid_loss += loss.item()*batch['image'].size(0)
            
            # calculate valid accuracy
            total = labels.size(0)
            _, predicted = torch.max(output, 1)
            #collect predictions to list
            valid_prediction.extend(predicted.tolist())
            
            correct = (predicted == labels).sum().item()
            # add accuracy to list of accuracies pr batch
            valid_accuracy.append(correct/total)
            
            # keep track of target labels
            valid_target.extend(labels.tolist())
            
#        model.eval()
#        for batch in test_loader:
#            # forward pass: compute predicted outputs by passing inputs to the model
#            test_output = model(batch['image'].to(device))
#            
#             # labels
#            test_labels = batch['labels'].to(device)
#            
#             # get correct dimensions for CrossEntropyLoss function
#            test_labels = test_labels.squeeze_()
#            
#            # calculate valid accuracy
#            test_total = 74
#            _, test_predicted = torch.max(test_output, 1)
#            
#            
#            test_correct = (test_predicted == test_labels).sum().item()
#            
#            # add accuracy to list of accuracies pr batch
#            test_accuracy = test_correct/test_total
#            
#            # keep track of target labels
#            test_target = test_labels.tolist()
#            
 
            
        # print training/validation statistics 
        # calculate average Root Mean Square loss over an epoch
        train_loss = np.sqrt(train_loss/len(train_loader.sampler.indices))
        valid_loss = np.sqrt(valid_loss/len(valid_loader.sampler.indices))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        train_accuracy = np.mean(train_accuracy)
        valid_accuracy = np.mean(valid_accuracy)
        
        # list accuracy pr epoch
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
    

#        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining accuracy: {:.6f}% \tValidation accuracy: {:.6f}% \tTest accuracy: {:.6f}%'
#              .format(epoch+1, train_loss, valid_loss, train_accuracy*100, valid_accuracy*100, test_accuracy*100))

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining accuracy: {:.6f}% \tValidation accuracy: {:.6f}%'
              .format(epoch+1, train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))

     
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss
        
        print('train confusion matrix')
        plot_confusion_matrix(train_target, train_prediction, name='train confusion matrix.jpg')
        
        print('valid confusion matrix')
        plot_confusion_matrix(valid_target, valid_prediction, name='valid confusion matrix.jpg')
#        
#        print('test confusion matrix')
#        plot_confusion_matrix(test_target, test_predicted, name='test confusion matrix.jpg')
#            
#        
        

    return train_losses, valid_losses, train_accuracies, valid_accuracies, valid_accuracy, valid_loss



def test(test_loader, model): 
    
    model.eval()
    with torch.no_grad:
        for batch in test_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            test_output = model(batch['image'].to(device))
            
             # labels
            test_labels = batch['labels'].to(device)
            
             # get correct dimensions for CrossEntropyLoss function
            test_labels = test_labels.squeeze_()
            
            # calculate valid accuracy
            test_total = 74
            _, test_predicted = torch.max(test_output, 1)
            
            
            test_correct = (test_predicted == test_labels).sum().item()
            
            # add accuracy to list of accuracies pr batch
            test_accuracy = (test_correct/test_total)*100
            
            # keep track of target labels
            test_target = test_labels.tolist()
            
            print('Done testing. Test accuracy is: {}'.format(test_accuracy))
            
            print('test confusion matrix')
            plot_confusion_matrix(test_target, test_predicted, name='test confusion matrix.jpg')
            
            return test_accuracy
            
            
    
    
# save train_losses, valid_losses.... to variables
train_losses, valid_losses, train_accuracies, valid_accuracies, valid_accuracy, valid_loss = train(train_loader, valid_loader, test_loader,
                                   model,criterion, optimizer,
                                   n_epochs=num_epochs, saved_model='model.pt')

test_accuracy = test(test_loader, model)


plots(train_losses, valid_losses, train_accuracies, valid_accuracies, valid_loss, valid_accuracy)





## Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
## Print optimizer's state_dict
#print("Optimizer's state_dict:")
#for var_name in optimizer.state_dict():
#    print(var_name, "\t", optimizer.state_dict()[var_name])
#



