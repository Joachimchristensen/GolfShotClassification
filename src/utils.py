

__all__ = ["get_confusion_matrix_plt", "get_classification_metrics", "build_optimizer"]

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from torch.autograd import Variable
from sklearn.preprocessing import label_binarize
import pywt
from matplotlib.colors import ListedColormap
from typing import Optional
from torch.autograd import Variable


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


def get_confusion_matrix_plt(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=['NoTrigger', 'Putt', 'FullSwing'], columns=['NoTrigger', 'Putt', 'FullSwing'])

    plt.figure()
    sns.heatmap(cm_df, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.axis('off')
    return plt


def get_confusion_matrix_histogram(true_labels, pred_labels, data_list, title='Distribution of classifications'):

    fig, axs = plt.subplot(3, 3, sharex=True)
    fig.suptitle(title)

    for i in range(3):
        for j in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                idxs = ((true_labels == i) * (pred_labels == j))
                axs[i, j].hist(data_list[idxs])
    axs[0, 0].set_ylabel('No Triggger')
    axs[1, 0].set_ylabel('Putt')
    axs[2, 0].set_ylabel('FullSwing')
    axs[2, 0].set_xlabel('No Triggger')
    axs[2, 1].set_xlabel('Putt')
    axs[2, 2].set_xlabel('FullSwing')

    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.axis('off')
    fig.tight_layout()
    return plt




def get_classification_metrics(true_labels, pred_probs, shot_ids, poi, ball_vr):

    pred_labels = np.argmax(pred_probs, axis=-1)
    true_one_hot_labels = label_binarize(true_labels, classes=[0, 1, 2])

    output_prob_mat = np.zeros((3, 3))
    out_prob_dict = {i: 0 for i in range(9)}
    avg_poi_mat = np.zeros((3, 3))
    avg_ballvr_mat = np.zeros((3, 3))
    avg_shotid_mat = np.zeros((3, 3))


    accuracy = accuracy_score(true_labels, pred_labels)
    cm_plot = get_confusion_matrix_plt(true_labels, pred_labels)

    c = 0
    for k in range(3):
        for l in range(3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                idxs = ((true_labels == k) * (pred_labels == l))
                output_prob_mat[k, l] = np.nanmean(pred_probs[idxs, l])
                out_prob_dict[c] = pred_probs[idxs, l]
                avg_poi_mat[k, l] = np.nanmean(poi[idxs])
                avg_ballvr_mat[k, l] = np.nanmean(ball_vr[idxs])
                shot_list = shot_ids[idxs]
                num_shots = len(shot_list)
                if num_shots > 0:
                    num_unique_shots = len(np.unique(shot_list))
                    avg_shotid_mat[k, l] = num_unique_shots  # / num_shots * 100
                c += 1

    class_names = ['No trigger', 'Putt', 'FullSwing']
    output_prob_df = pd.DataFrame(data=output_prob_mat, columns=class_names, index=class_names)
    avg_poi_df = pd.DataFrame(data=avg_poi_mat, columns=class_names, index=class_names)
    avg_ballvr_df = pd.DataFrame(data=avg_ballvr_mat, columns=class_names, index=class_names)
    avg_shotid_df = pd.DataFrame(data=avg_shotid_mat, columns=class_names, index=class_names)

    num_correctly_shots = 0
    for shot_id in np.unique(shot_ids):
        idx1 = (shot_ids == shot_id) * (true_labels == 0)
        idx2 = (shot_ids == shot_id) * (true_labels != 0)
        classes_correct = 0
        for idx in [idx1, idx2]:
            # The number of correctly classified windows should be bigger than either of the other two
            # misclassified classes
            correct_size = sum(true_labels[idx] == pred_labels[idx])
            if correct_size > sum(idx) / 2:
                classes_correct += 1
        # Only assign shot as correct if both its no trigger and trigger windows are correct
        if classes_correct == 2:
            num_correctly_shots += 1

    acc_shot_basis = num_correctly_shots / len(np.unique(shot_ids)) * 100

    probs_hist_matrix = get_confusion_matrix_histogram(true_labels, pred_labels, pred_probs, 'Distribution of Predicted Probabilities')
    toi_hist_matrix = get_confusion_matrix_histogram(true_labels, pred_labels, poi, 'Distribution of ToI')
    ballvr_hist_matrix = get_confusion_matrix_histogram(true_labels, pred_labels, ball_vr, 'Distribution of Ball Vr')

    classification_metrics = {"Accuracy - Windows": accuracy,
                              "Confusion Matrix": cm_plot,
                              "Probabilities distributions": probs_hist_matrix,
                              "ToI distributions": toi_hist_matrix,
                              "Ball Vr distributions": ballvr_hist_matrix,
                              "Accuracy - Shots": acc_shot_basis,
                              "Average output probabilities": output_prob_df,
                              "Average Point of Impact in window": avg_poi_df,
                              "Average Ballvr": avg_ballvr_df,
                              "Number of unique misclassified shots": avg_shotid_df
                              }

    return classification_metrics


def build_optimizer(model, optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=0.1, step_size_up=5,
                                                      mode="triangular2", gamma=0.85)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0.5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-7)
    return optimizer, scheduler


def channel_stats():
    nt_channel_means    = torch.tensor([[0.0010], [0.0001], [0.0011], [0.0011], [0.0008], [0.0009], [0.0005], [0.0001]])

    putt_channel_means  = torch.tensor([[0.0011], [0.0002], [0.0011], [0.0012], [0.0009], [0.0009], [0.0005], [0.0001]])

    fs_channel_means   = torch.tensor([[0.0011], [0.00002], [0.0011], [0.0011], [0.0008], [0.001], [0.0007], [0.00003]])

    fs_channel_means = torch.tensor([[1.0710e-03], [2.0657e-05], [1.0643e-03], [1.0674e-03], [7.8950e-04], [9.7558e-04], [7.0000e-04], [2.6018e-05]])


    nt_channel_stds     = torch.tensor([[0.0046], [0.0043], [0.0044], [0.0044], [0.0039], [0.0039], [0.0021], [0.0021]])

    putt_channel_stds   = torch.tensor([[0.0113], [0.0112], [0.0118], [0.0117], [0.0101], [0.0100], [0.0067], [0.0068]])

    fs_channel_stds     = torch.tensor([[0.0059], [0.0058], [0.0059], [0.0059], [0.0058], [0.0057], [0.0033], [0.0033]])


'''
def get_cmap():
    #X = np.loadtxt(r'C:\Projects\VisionSandbox\putting_trigger_nn\TriggerNets\notebooks\heat_map.txt', skiprows=1) #GIVE LOCATION HERE
    X = np.loadtxt(r'C:\Projects\VisionDevelopment\bachelor_code\heat_map.txt', skiprows=1)
    X = np.loadtxt(r'students\joc\bachelor\bachelor_code\data\heat_map.txt', skiprows=1)
    cmap = np.c_[X, np.ones(len(X))]
    New_cmap = ListedColormap(cmap)
    return New_cmap


tm_map = get_cmap()

# Sample rate of radar
sample_rate = 39062


def calc_psd(samples, fft_size, step_size, window, shape, out_shape):
    sliced_data = np.empty(shape, dtype=complex)
    for i in range(shape[0]):
        sliced_data[i, :] = samples[i * step_size: i * step_size + fft_size]
        sliced_data[i, :] *= window
    psd = np.abs(np.fft.fft(sliced_data, fft_size, axis=1)[:, :out_shape[1]])
    psd = 20. * np.log10(psd)
    return psd


def make_std_spectrogram(samples):
    fft_size = 128
    step_size = 16
    window = np.hanning(fft_size)
    N_samples = 4096
    shape = (int((N_samples - fft_size) / step_size + 1), fft_size)
    factor = 2
    out_shape = (shape[0], int(fft_size / factor))
    psd = calc_psd(samples, fft_size, step_size, window, shape, out_shape)

    return psd


def plot_spectogram(psd, toi_ax=False, fft_sz=512, step=64):
    if toi_ax:
        f = np.array([(n - 0.5) * sample_rate / fft_sz for n in range(psd.shape[1])])
        vr = (f * 3e8) / (2 * 24.1e9)
        tpsd = np.asarray([fft_sz / 2 + (n - 0.5) * step for n in range(psd.shape[0])]) / sample_rate
        n_t = tpsd
        n_f = vr
        plt.figure()
        plt.pcolormesh(n_t, n_f, psd.T, cmap=tm_map)
        plt.show()
    else:
        n_t = np.arange(0, psd.shape[0])
        n_f = np.arange(0, psd.shape[1])
        plt.figure()
        plt.pcolormesh(n_t, n_f, psd.T, cmap=tm_map)
        plt.show()'''








