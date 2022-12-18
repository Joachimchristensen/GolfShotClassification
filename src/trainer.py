from pathlib import Path
import numpy as np
import pandas as pd
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
#from models import D1Net, InceptionTime
import torch
from torch import nn
import warnings
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .get_loaders import get_loaders
from .utils import *
from .models import ConvNet1D, ConvNet2D, InceptionTime
from src import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from kornia.losses import FocalLoss
import wandb
from typing import cast, Any, Dict, List, Tuple, Optional
from torchsummaryX import summary
import plotly


class Trainer:
    """Trains a TSC model on golf shot data.

    Attributes
    ----------
    The following need to be added by the initializer:
    model:
        The initialized inception model
    data_path:
        A path to the data folder - get_loaders should look here for the data
    model_dir:
        A path to where the model and its predictions should be saved

    The following don't:
    train_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    val_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    test_results:
        The evaluate function fills this in, evaluating the model on the test data
    """

    def __init__(self, model: nn.Module, data_path: str):

        self.model = model
        self.data_path = data_path

        self.save_path = Path('/home/students/joc/bachelor/bachelor_code/experiments/results')

        self.model_dir = self.save_path / 'models' / self.model.__class__.__name__
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.val_loss_ce: List[float] = []
        self.test_results: Dict[str, float] = {}

        self.random_seed = 42
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def fit(self, batch_size: int = 128, num_epochs: int = 50,
            val_size: float = 0.2, learning_rate: float = 0.001,
            patience: int = 10, use_wandb: bool = False, normalize: bool = True, gpu: str = 'cuda:0',
            model_name: str = "InceptionTime", verbose: bool = True, optimizer=None, scheduler=None, make_psd=False) -> None:
        """Trains a TSC model

        Arguments
        ----------
        batch_size:
            Batch size to use for training and validation
        num_epochs:
            Maximum number of epochs to train for
        val_size:
            Fraction of training set to use for validation
        learning_rate:
            Learning rate to use with Adam optimizer
        patience:
            Maximum number of epochs to wait without improvement before
            early stopping
        use_wandb:
            Boolean indicating if the fitting should be tracked via WandB
        """

        channel_means = torch.tensor([[1.0454e-03], [1.0913e-04],
                                      [1.0646e-03], [1.1150e-03],
                                      [8.4906e-04], [9.2392e-04],
                                      [5.8158e-04], [8.2257e-05]]).to(gpu, non_blocking=True)

        channel_stds = torch.tensor([[0.0079], [0.0077],
                                    [0.0080], [0.0080],
                                    [0.0071], [0.0071],
                                    [0.0045], [0.0045]]).to(gpu, non_blocking=True)

        train_loader, val_loader, _ = get_loaders(data_path=self.data_path, num_shots=20000, batch_size=batch_size,
                                                  normalize=True, make_psd=make_psd, gpu=gpu)
        device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=0.00001)
        focal_loss = FocalLoss(alpha=0.5, gamma=3, reduction='mean')
        #focal_loss = nn.CrossEntropyLoss()
        cross_entropy_loss = nn.CrossEntropyLoss()
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

        if use_wandb and True:
            wandb.init(project="Bachelor", entity="joachimcc")
            wandb.config = {"init_learning_rate": optimizer.param_groups[0]["lr"],
                            "epochs": num_epochs,
                            "batch_size": batch_size}
            config = wandb.config

        if verbose:
            print('Training in progress...')

        for epoch in range(num_epochs):

            self.model.train()
            epoch_train_loss = []

            for i, (train_batch, train_labels, train_file_names, train_idx, train_shot_ids, train_poi, train_ball_vr) in enumerate(train_loader):
                train_labels = train_labels.squeeze().long()
                train_batch, train_labels = train_batch.to(device, non_blocking=True), \
                                            train_labels.to(device, non_blocking=True)
                #if not make_psd:
                #    train_batch = (train_batch - channel_means) / channel_stds  # .to(torch.float32))
                optimizer.zero_grad()
                output = self.model(train_batch.float())
                train_loss = focal_loss(output, train_labels)
                #train_loss = cross_entropy_loss(output, train_labels)
                epoch_train_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                #if i % 100 == 0:
                #    print(f'Batch {i},\t Train loss: {train_loss.item():.3f}')
                # break
                if i == 1000:
                    break

            self.train_loss.append(np.mean(epoch_train_loss))

            epoch_val_loss = []
            #epoch_val_loss_ce = []
            true_list, preds_list = [], []
            val_shotid_list, val_poi_list, val_ballvr_list, = [], [], []
            self.model.eval()
            with torch.no_grad():
                for j, (val_batch, val_labels, val_file_names, val_idx, val_shot_ids, val_poi, val_ball_vr) in enumerate(val_loader):
                    val_labels = val_labels.squeeze().long()
                    val_batch = val_batch.to(device, non_blocking=True)
                    val_labels = val_labels.to(device, non_blocking=True)
                    # if not make_psd:
                    #    val_batch = (val_batch - channel_means) / channel_stds
                    output = self.model(val_batch.float())
                    val_loss = focal_loss(output, val_labels)
                    # val_loss = cross_entropy_loss(output, val_labels)
                    preds = torch.softmax(output, dim=-1)
                    epoch_val_loss.append(val_loss.item())
                    # epoch_val_loss_ce.append(val_loss_ce.item())
                    true_list.append(val_labels.detach().cpu().numpy())
                    preds_list.append(preds.detach().cpu().numpy())
                    val_shotid_list.append(val_shot_ids)
                    val_poi_list.append(val_poi)
                    val_ballvr_list.append(val_ball_vr)

                    if j == 100:
                        break

                self.val_loss.append(np.mean(epoch_val_loss))
                # self.val_loss_ce.append(np.mean(epoch_val_loss_ce))
                true_labels, pred_probs = np.concatenate(true_list), np.concatenate(preds_list)

                pred_labels = np.argmax(pred_probs, axis=-1)
                val_shot_ids = np.concatenate(val_shotid_list)
                val_poi = np.concatenate(val_poi_list)
                val_ball_vr = np.concatenate(val_ballvr_list)

                classification_metrics = get_classification_metrics(true_labels, pred_probs, val_shot_ids, val_poi, val_ball_vr)

            if use_wandb:
                #pr = wandb.plot.pr_curve(true_labels, pred_probs, labels=['NoTrigger', 'Putt', 'FullSwing'])
                #roc = wandb.plot.roc_curve(true_labels, pred_probs, labels=['NoTrigger', 'Putt', 'FullSwing'])
                log_dict = {"Epoch": epoch+1, "Learning Rate": optimizer.param_groups[0]["lr"],
                            "Train Loss": self.train_loss[-1],
                            "Validation FocalLoss": self.val_loss[-1],  # "Validation CELoss": self.val_loss_ce[-1],
                            #"Accuracy - Shots": acc_shot_basis,
                            #"Average output probabilities": output_prob_df,
                            #"Average Point of Impact in window": avg_poi_df,
                            #"Average Ballvr": avg_ballvr_df,
                            #"Number of unique misclassified shots": avg_shotid_df
                            }
                log_dict = {**classification_metrics, **log_dict}
                wandb.log(log_dict)
                #wandb.log({"Precision v. Recall": pr, "ROC": roc, "Epoch": epoch + 1, "Model": model_name})

            if scheduler is not None:
                scheduler.step(self.val_loss[-1])

            print(f'Epoch: {epoch + 1}, '
                  f'Train loss: {round(self.train_loss[-1], 3)}, '
                  f'Val loss: {round(self.val_loss[-1], 3)},'
                  f'Val accuracy: {round(classification_metrics["Accuracy - Windows"], 3)}')

            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    return None

        if verbose:
            print('\nTraining done!\n')

    def evaluate(self, batch_size: int = 128) -> None:

        _, _, test_loader = get_loaders(batch_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        true_list, preds_list = [], []
        with torch.no_grad():
            for test_batch, test_labels, test_file_names, test_idx, test_items_dict in test_loader:
                test_labels = test_labels.squeeze().long()
                test_batch, val_labels = test_batch.to(device), test_labels.to(device)
                true_list.append(test_labels.detach().numpy())
                preds = self.model(test_batch.float())
                preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.detach().cpu().numpy())

        true_labels, pred_probs = np.concatenate(true_list), np.concatenate(preds_list)
        classification_metrics = get_classification_metrics(true_labels, pred_probs)

        self.test_results.update(classification_metrics)

        print(f'ROC AUC score: {round(self.test_results["roc_auc_score"], 3)}')
        print(f'Accuracy score: {round(self.test_results["accuracy_score"], 3)}')

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        save_dict = {'model': {'model_class': self.model.__class__.__name__,
                               'state_dict': self.model.state_dict(),
                               'input_args': self.model.input_args}}
        if savepath is None:
            model_name = f'{self.model.__class__.__name__}_model.pkl'
            savepath = self.model_dir / model_name
        torch.save(save_dict, savepath)

        return savepath


def load_trainer(model_path: Path, data_path: str) -> Trainer:

    # data_path = model_path.resolve().parents[3]
    #data_path = r'/data/AIDatasets/TM4_bin'

    model_dict = torch.load(model_path)

    model_class = getattr(models, model_dict['model']['model_class'])
    model = model_class(**model_dict['model']['input_args'])
    model.load_state_dict(model_dict['model']['state_dict'])

    loaded_trainer = Trainer(model, data_path=data_path)
    loaded_trainer.encoder = model_dict['encoder']

    return loaded_trainer


if __name__ == "__main__":
    data_path = r'/data/AIDatasets/TM4_bin'
    use_wandb = False
    gpu = 'cuda:2'

    model = models.InceptionTime(in_channels=8, out_channels=32, kernel_size=40, depth=3, num_pred_classes=3)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model.to(device)
    print(summary(model, torch.rand([1, 8, 4096]).to(device)))
    #trainer = Trainer(model=model, data_path=data_path)
    #trainer.fit(use_wandb=use_wandb, gpu=gpu, num_epochs=1)

    #savepath = trainer.save_model()
    # new_trainer = load_trainer(model_path=savepath, data_path=data_path)
    # new_trainer.evaluate()
