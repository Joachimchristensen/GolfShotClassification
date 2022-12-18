import os.path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio import transforms
from tqdm import tqdm
import wandb
from torch.backends import cudnn
from Loss_trigger import TriggerLoss 
from dataloader_Trigger import TriggerDataSet

from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.nn.functional as F

from D1Net import Net

from tqdm import tqdm


# Set up your default hyperparameters

hyperparameter_defaults = dict(
    learning_rate=0.001,
    batch_size=512,
    optimizer="adam",
    epochs=10,
    architecture = 'CNN'
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

    
mean = torch.tensor([0, 1.1983, 28.1437]).cuda(non_blocking=True)
std = torch.tensor([1.0000, 0.96511, 27.9314351395]).cuda(non_blocking=True)

def norm_labels(labels):
    #return labels/maxes
    return (labels-mean)/std

def inv_pred(inp):
    return inp*std[1:]+mean[1:]

channel_means = torch.tensor([[9.7679e-04], [7.8133e-05], 
                              [9.8849e-04], [1.0418e-03], 
                              [8.0620e-04], [8.6011e-04],
                              [5.5876e-04], [3.3484e-05]]).cuda(non_blocking=True)

channel_stds =  torch.tensor([[0.0070], [0.0069], 
                              [0.0071], [0.0071],
                              [0.0063], [0.0063], 
                              [0.0039], [0.0039]]).cuda(non_blocking=True)

def standardize(inp):
    return (inp-channel_means)/channel_stds

def train_loop(model: nn.Module, loss_func: TriggerLoss, optimizer, train_loader, epochs, device, batch_size, validation_loader=None):
    model.train()
    
    for epoch in tqdm(range(0, config["epochs"]),unit='epoch'):
        epoch_loss = 0
        total_size = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader, 0),total=len(train_loader)):
            samples, labels, weights, _, _ = data
            samples, labels, weights = samples.cuda(non_blocking=True), labels.cuda(non_blocking=True), weights.cuda(non_blocking=True)
            labels = norm_labels(labels) # - To be unleashed

            inputs = standardize(samples.to(torch.float32))
            optimizer.zero_grad()
            
            # Inference
            output_classification, output_regression = model(inputs)

            loss = loss_func(output_classification, output_regression, labels, weights)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            total_size += inputs.size(0)

        epoch_loss /= total_size
        
        msg = f"Epoch: {epoch + 1}, loss: {epoch_loss}"
        
        validation_loss, classification_accuracy, regression_accuracy = network_validation(model, validation_loader, loss_func, device)

        msg += f", validation loss: {validation_loss}"

        wandb.log({"loss": epoch_loss,
                  "validation_loss" : validation_loss,
                  "Learning rate" : optimizer.param_groups[0]["lr"],
                  "RMS toi" : regression_accuracy[0],
                  "RMS vr" : regression_accuracy[1],
                  "ACC" :  classification_accuracy,
                  "Epoch" : epoch
                   })
        

def network_validation(model: nn.Module, validation_loader, loss_func, device):
    with torch.no_grad():
        model.eval()
        classification_accuracy = torch.tensor(0.).to(device)
        regression_accuracy = torch.tensor([0., 0.]).to(device)
        triggers_size = 0
        total_loss = 0
        total_size = 0
        for i, data in enumerate(validation_loader, 0):
            samples, labels, weights, _, _ = data
            samples, labels, weights = samples.cuda(non_blocking=True), labels.cuda(non_blocking=True), weights.cuda(non_blocking=True)
            labels = norm_labels(labels)
            inputs = standardize(samples.to(torch.float32))

            output_classification, output_regression = model(inputs)
            loss = loss_func(output_classification, output_regression, labels, weights)

            total_loss += loss.item() * inputs.size(0)
            total_size += inputs.size(0)
            triggers_size += (labels[:, 0]>0).sum().item()
            
            output = output_classification.view(output_classification.size(0), -1)
            predicted = output.argmax(1)
            label_classes = torch.squeeze(labels[:, 0:1]) #LOOOOOK INTO THIS ,should be ,1 ?
            classification_accuracy += (label_classes==predicted).sum().item()

            reg_v = (labels[:, 0:1]>0).to(torch.int)
            
            regression_accuracy += ((reg_v * (inv_pred(output_regression)*weights[:, 1:] - inv_pred(labels[:,1:])))**2).sum(dim=0)
            output = F.softmax(output, dim=1)
            
        classification_accuracy = classification_accuracy / total_size
        regression_accuracy = (regression_accuracy / triggers_size).sqrt()
        total_loss /= total_size

    return total_loss, classification_accuracy, regression_accuracy

def get_loaders(dataset, batch_size, n_workers, pin_memory):
    validation_split = .15
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers = n_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers = n_workers, pin_memory=pin_memory)
    
    return train_loader, validation_loader


def build_optimizer(model, optimizer, learning_rate): #CHECK
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    return optimizer


def main(cuda_device=0):
    from torch.utils.data.sampler import SubsetRandomSampler

    torch.manual_seed(123) # Set seed for training 
    print(torch.cuda.is_available() )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
   
    # Define these settings for training
        
    pin_memory = True
    n_workers = 4

    #Get model
    model = Net()
    batch_size = int(config['batch_size'])
    

    model.to(device)
    
    optimizer = build_optimizer(model, config['optimizer'], config['learning_rate'])
    
    
    dataset_FS = TriggerDataSet("../../../data/raw/Train", "../../../data/010422_FS.txt", False)
    dataset_Put = TriggerDataSet("../../../data/raw/download","../../../data/labels_orb_07012022_putting.txt", True)
    dataset = torch.utils.data.ConcatDataset([dataset_FS, dataset_Put])

    train_loader, validation_loader = get_loaders(dataset, batch_size, n_workers, pin_memory)

    loss_func = TriggerLoss(1,1)
    print("Start training", flush=True)

    train_loop(model, loss_func, optimizer, train_loader, config['epochs'], 
               device, batch_size, validation_loader)
    
    print("Training done", flush=True)

if __name__ == "__main__":
    main()
