import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import TriggerDataSet
import progressbar

device = torch.device('cuda:0')  # if 'cuda' else 'cpu'


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
        self.Xtr.to(device)
        self.ytr.to(device)

    def predict(self, X, distance='L1'):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        X.to(device)
        # lets make sure that the output type matches the input type
        #Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        Ypred = torch.zeros(num_test, dtype=float, device=device)
        #Ypred.to(device)
        # loop over all test rows
        for i in progressbar.progressbar(range(num_test)):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            X[i, :].to(device)
            #if distance == 'L1':
            distances = np.abs(self.Xtr - X[i, :]).sum((1, 2))
            # using the L2 distance (sum of absolute value differences)
            #if distance == 'L2':
             #   distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis=1))
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    path = r'/data/AIDatasets/TM4_bin'

    dataset_FS = TriggerDataSet(path+"/fullswing", path+"/12082022.txt", False)
    #dataset_Put = TriggerDataSet(path+"/putting",path+"/labels_orb_07012022_putting.txt", True)
    dataset = torch.utils.data.ConcatDataset([dataset_FS])#, dataset_Put])

    #################################################################################
    batch_size = 1000
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    ###################################################################################

    X_train, y_train, weights, tt, _ = next(iter(train_loader))
    X_val, y_val, weights, tt, _ = next(iter(validation_loader))


    nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
    nn.train(X_train, y_train)  # train the classifier on the training images and labels
    y_test_pred = nn.predict(X_val)  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print('accuracy: %f' % (np.mean([1 if pred == train else 0 for pred, train in zip(y_test_pred, y_val)])))