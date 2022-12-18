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
from D1Net import Net

from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.nn.functional as F


def save_checkpoint(model: nn.Module, optimizer, epoch, 
                    acc, toi, path, best_val_loss):
    save_dict = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_acc': acc,
                 'best_toi' : toi,
                 #'best_vr' : vr,
                 'best_val_loss' : best_val_loss
                }
    torch.save(save_dict, path)


# mean = torch.tensor([0, 1.1983, 28.1437]).cuda(non_blocking=True)
# std = torch.tensor([1.0000, 0.96511, 27.9314351395]).cuda(non_blocking=True)

mean = torch.tensor([0, 1.1983]).cuda(non_blocking=True)
std = torch.tensor([1.0000, 0.96511]).cuda(non_blocking=True)

def norm_labels(labels):
    #return labels/maxes
    return (labels-mean)/std

def inv_pred(inp):
    return inp*std[1:]+mean[1:]


channel_means = torch.tensor([[9.7679e-04], [7.8133e-05], 
                              [9.8849e-04], [1.0418e-03], 
                              [8.0620e-04], [8.6011e-04],
                              [5.5876e-04], [3.3484e-05]]).cuda(non_blocking=True)

channel_stds = torch.tensor([[0.0070], [0.0069],
                              [0.0071], [0.0071],
                              [0.0063], [0.0063], 
                              [0.0039], [0.0039]]).cuda(non_blocking=True)


def standardize(inp, mode='Train', batch_sisze=256):
    if mode == 'train':
        std_er = (0.8 - 8) * torch.rand(1) + 8
        mean_er = (0.5 - 2) * torch.rand(1) + 2
        return (inp-channel_means*mean_er)/(channel_stds*std_er)
    else:
        return (inp-channel_means)/channel_stds

sample_rate = 39062

#transform = transforms.Resample(sample_rate, sample_rate//4).cuda()

def local_std(inp):
    print(inp.shape)
    my = torch.mean(inp, axis=2)
    print(my.shape)
    return (inp.T - my)

def train_loop(model: nn.Module, loss_func: TriggerLoss, optimizer, train_loader, epochs, project, scheduler, device, batch_size, validation_loader=None, checkpoint_path=None):
    
    # wandb init stuff 
    config = {
      "init_learning_rate": optimizer.param_groups[0]["lr"],
      "epochs": epochs,
      "batch_size": batch_size,
      "Arc" : checkpoint_path 
    }
    
    wandb.init(project=project, entity="tm_orb", config = config)

    checkpoint_epoch = 0
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch'] + 1
        best_validation_loss = checkpoint['best_val_loss']
    else:
        print("No path exists, will train new")
        best_validation_loss = None

    print("Starting train loop",  flush=True)

    for epoch in tqdm(range(checkpoint_epoch, epochs),unit='epoch'):
        epoch_loss = 0
        total_size = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader, 0),total=len(train_loader)):
            samples, labels, weights, _, _ = data
            samples, labels, weights = samples.cuda(non_blocking=True), labels.cuda(non_blocking=True), weights.cuda(non_blocking=True)
            
            labels = norm_labels(labels) # - To be unleashed

            #inputs = standardize(samples.to(torch.float32))
            inputs = samples.to(torch.float32)#local_std(samples)
            #inputs = transform(inputs)
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
        
        if validation_loader is not None:
            validation_loss, classification_accuracy, regression_accuracy = network_validation(model, validation_loader, loss_func, device)

            msg += f", validation loss: {validation_loss}"
    
            if (best_validation_loss is None or validation_loss < best_validation_loss) and checkpoint_path is not None:
            
#                 save_checkpoint(model, optimizer, epoch, classification_accuracy, regression_accuracy[0], 
#                                 regression_accuracy[1], checkpoint_path + ".best", validation_loss)
                save_checkpoint(model, optimizer, epoch, classification_accuracy, regression_accuracy[0], 
                                checkpoint_path + ".best", validation_loss)
                
                best_validation_loss = validation_loss
                #export_onnx(model, checkpoint_path)
        
        if checkpoint_path is not None:
#             save_checkpoint(model, optimizer, epoch, classification_accuracy, regression_accuracy[0], 
#                             regression_accuracy[1], checkpoint_path, validation_loss)
            save_checkpoint(model, optimizer, epoch, classification_accuracy, regression_accuracy[0], 
                            checkpoint_path, validation_loss)

        print("Learning rate")
        print(optimizer.param_groups[0]["lr"])
        print("RMS {}".format(regression_accuracy))
        print(msg, flush=True)
        wandb.log({"loss": epoch_loss,
                  "validation_loss" : validation_loss,
                  "Learning rate" : optimizer.param_groups[0]["lr"],
                  "RMS toi" : regression_accuracy[0],
                  #"RMS vr" : regression_accuracy[1],
                  "ACC" :  classification_accuracy,
                  "Epoch" : epoch
                   })
        
        #img = log_img(data,model,epoch)
        #wandb.log({"examples": img})

        
        scheduler.step()

def network_validation(model: nn.Module, validation_loader, loss_func, device):
    with torch.no_grad():
        model.eval()
        classification_accuracy = torch.tensor(0.).to(device)
        regression_accuracy = torch.tensor([0.]).to(device)

        #regression_accuracy = torch.tensor([0., 0.]).to(device)
        triggers_size = 0
        total_loss = 0
        total_size = 0
        ground_truth = []
        predictions = []
        probs = []
        for i, data in enumerate(validation_loader, 0):
            samples, labels, weights, _, _ = data
            samples, labels, weights = samples.cuda(non_blocking=True), labels.cuda(non_blocking=True), weights.cuda(non_blocking=True)
            labels = norm_labels(labels)
            #inputs = standardize(samples.to(torch.float32),mode='Validation')
            #inputs = transform(inputs)
            inputs = samples.to(torch.float32)#local_std(samples)

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
            for i, f in enumerate(labels):
                ground_truth.append(labels[i,0].item())
                predictions.append(predicted[i].item())
                probs.append(output[i].cpu().numpy())

        classification_accuracy = classification_accuracy / total_size
        regression_accuracy = (regression_accuracy / triggers_size).sqrt()
        total_loss /= total_size

        wandb.log({"pr" : wandb.plot.pr_curve(ground_truth, np.asarray(probs),
                     labels=['NoTrigger','FullSwing', 'Putt'], classes_to_plot=None)})
        
        wandb.log({"roc": wandb.plot.roc_curve(ground_truth, np.asarray(probs),
                                              labels=['NoTrigger','FullSwing', 'Putt']
                                              )})

        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truth, preds=predictions,
                        class_names=['NoTrigger','FullSwing', 'Putt'])})
        
        cm = confusion_matrix(ground_truth, predictions)
        cm_df = pd.DataFrame(cm,
                             index = ['NoTrigger','FullSwing', 'Putt'], 
                             columns = ['NoTrigger','FullSwing', 'Putt'])
        
        plt.figure()
        sns.heatmap(cm_df, annot=True,fmt="d")
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        wandb.log({"chart": plt})

    return total_loss, classification_accuracy, regression_accuracy


def export_onnx(model, name):
    dummy_input = torch.randn(1, 8, 4096, device="cpu").to(torch.float32)
    model.eval()
    model.to('cpu')
    torch.onnx.export(model, 
                      dummy_input, 
                      name+".onnx",
                      export_params=True, 
                      verbose=True, 
                      input_names=['input'], 
                      output_names=['cls', 'reg'])
    wandb.save(name + ".onnx")
    model.to('cuda')

    

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


def main(cuda_device=0):
    cudnn.benchmark = True
    torch.manual_seed(123) # Set seed for training 
    print(torch.cuda.is_available() )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Define these settings for training
    #from MyInception import InceptionTime
    #model = InceptionTime(8)

    batch_size = 256
    pin_memory = True
    n_workers = 8
    
    #model = Net()#InceptionTime(8, 32)#mobilenetv2()
    from Mv21d import MobileNetV2
    model = Net()#MobileNetV2()
    model.to(device)

    attributes = 'AllSwings__nostd_1024'
    architecture = "CNN"
    epochs = 100
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    project_name = "TM4-TriggerNet"
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2",gamma=0.85)
    
    dataset_FSTtrain = TriggerDataSet("../../../data/raw/Train", "../../../data/12082022.txt", False)
    dataset_FSTest = TriggerDataSet("../../../data/raw/Test", "../../../data/12082022.txt", False)
    dataset_FSTVal = TriggerDataSet("../../../data/raw/Validation", "../../../data/12082022.txt", False)
    dataset_FSTEval = TriggerDataSet("../../../data/raw/eval", "../../../data/12082022.txt", False)

    dataset_Put = TriggerDataSet("../../../data/raw/download","../../../data/labels_orb_07012022_putting.txt", True)
    
    checkpoint_path = os.path.join("../../../models", "modelsTN", attributes +'_' + architecture + '_' + str(batch_size) +'_' + str(epochs) + ".nn")
    print(checkpoint_path)

    dataset = torch.utils.data.ConcatDataset([dataset_FSTtrain, dataset_FSTest, dataset_FSTVal, dataset_FSTEval, dataset_Put])
    #dataset  = torch.utils.data.Subset(dataset, range(0,10000))
    
    train_loader, validation_loader = get_loaders(dataset, batch_size, n_workers, pin_memory)

    loss_func = TriggerLoss(1,1)
   
    print("Start training", flush=True)
    
    train_loop(model, loss_func, optimizer, train_loader, epochs, project_name, scheduler, 
               device, batch_size, validation_loader, checkpoint_path)
    
    print("Training done", flush=True)

if __name__ == "__main__":
    main()
