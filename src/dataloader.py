
#__all_ = ['TriggerDataset']

import torch
from torch.utils.data import Dataset
import os
import random
import warnings
import numpy as np
import matplotlib
#matplotlib.use('TKAgg')
import ast
import matplotlib.pyplot as plt
from binary_data_io import read_binary
from progressbar import progressbar
from collections import defaultdict
from typing import cast, Any, Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedGroupKFold
from ssqueezepy import cwt
from torchvision import transforms, models

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, samples):
        return {'samples': torch.from_numpy(samples)}


class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, mean, std):
        self.mean = mean.cpu()
        self.std = std.cpu()

    def __call__(self, samples):
        return (samples-self.mean) / self.std


class TriggerDataset(Dataset):
    """Pytorch dataloader:
        - Creates a pytorch dataset for the TM4 trigger network on the raw radar signals.
    """

    def __init__(self, path_to_data: str, path_to_labels: str, putting: bool = False,
                 normalize=None, num_shots: int = 20000, make_psd: bool = False, verbose: bool = True):
        """
        Initialises the Dataset class for the TM4 Trigger Network. Data will need to be downloaded from Azure blob
        storage in the binary files, no preprocessing is needed. FullSwing tmf's and putting tmf's are handled
        differently, set flag when initialising dataset.

        Arguments:
        ----------
        path_to_data: (str)
            Path to data given as string (data must be .bin files)
        path_to_labels: (str)
            Path to labels given as string (labels must be .txt file)
        putting: (bool)
            Default False, which corresponds to FullSwing.
        """

        # Variables used to compute the average length in seconds of the golf shots in the data set
        total_seconds, total_shots = 0, 0

        # Used to keep track of the label distribution
        self.label_counts = {'Trigger': 0, 'No Trigger': 0, 'Total': 0}
        self.no_label_counts = 0
        self.shot_lengths = defaultdict(int)

        file_labels = self.read_labels(path_to_labels)
        self.path_to_data = path_to_data
        self.putting = putting
        self.make_psd = make_psd

        self.omit_samples = 0
        self.sample_rate = 39062.5

        self.step = 3072  # 25% overlap  --> generates in average ((6.10*39062.5)-4096)/3072/5=15.25 and
        # ((1.29*39062.5)-4096)/3072 = 15.07 "No trigger" windows per shot for Putt and FullSwing datasets respectively
        self.step_trigger = 128  # 97% overlap --> generates 29 "Trigger" windows per shot (explained later in the loop)
        self.N_samples = 4096
        self.items = []
        self.channels = [0, 1, 2, 3, 4, 5, 6, 7]
        self.normalize = normalize

        self.club_vr = []
        self.ball_vr = []
        self.shot_id = defaultdict(int)
        self.shot_id_list = []
        self.labels = []

        # FFT variables
        self.fft_sizes = [128, 512, 1024]
        self.fft_step_sizes = [16, 64, 128]
        self.factors = [2, 4, 24]
        self.fft_windows = [np.hanning(fft_size) for fft_size in self.fft_sizes]
        self.fft_shapes = [(int((self.N_samples - fft_size) / fft_step_size + 1), fft_size) for fft_size, fft_step_size
                           in zip(self.fft_sizes, self.fft_step_sizes)]
        self.fft_out_shapes = [(self.fft_shapes[i][0], int(fft_size / factor)) for i, (fft_size, factor) in
                               enumerate(zip(self.fft_sizes, self.factors))]

        step_sizes = [self.step, self.step_trigger]
        old_counter = 0
        if verbose:
            print('\nPreparing dataset...')
        for i, step_size in enumerate(step_sizes):
            counter = 0  # Counter is only incremented every time a file is successfully sampled
            for j, file in enumerate(progressbar(os.listdir(self.path_to_data))):

                try:
                    file_path = os.path.join(self.path_to_data, file)
                    name, _, total_samples, _ = read_binary(file_path, read_channels=[])
                    file_label = file_labels[name]

                    # Since there are so many shots, I just exclude the few hundreds without labels
                    if file_label != "No trigger":

                        # shot_id used for splitting dataset based on shot_id (Group splitting)
                        shot_id = counter + 1
                        counter += 1

                        # Only need to compute the average in one of the loops through the data (could also be i==1)
                        if i == 0:
                            # Variables used to compute the average length in seconds of the golf shots in the data set
                            total_seconds += total_samples / self.sample_rate
                            total_shots += 1
                            self.shot_lengths[shot_id] += total_samples / self.sample_rate

                        # If we are looking for trigger windows, we start the first window just inside ToI
                        if step_size == self.step_trigger:# and False:
                            # +0.005 to account for slight inaccuracies in ToI label assignments
                            start_sample = int(np.ceil(file_label["TimeOfImpact"]*self.sample_rate) - self.N_samples +
                                               np.ceil(0.005 * self.sample_rate)*0)
                            if start_sample < 0:
                                start_sample = 0
                        else:
                            start_sample = 0

                        window_counter = 0

                        while start_sample + self.N_samples < (total_samples - self.omit_samples):
                            if file_label == "No trigger":
                                toi = None

                            else:
                                # We have an offset of 0.005 seconds = 195.3125 samples around ToI, since the label
                                # is not perfect. This means that we would expect, from 19000 golf shots, step size of
                                # 128, and window length of 4096, to get (4096-(2*195.3125//128)*128)/128*19000 = 551000
                                # trigger windows (= 29 per shot)
                                start_time = start_sample / self.sample_rate
                                end_time = (start_sample + self.N_samples) / self.sample_rate - 0.005
                                start_time_offset = start_time + 0.005
                                toi = file_label["TimeOfImpact"] - start_time if start_time_offset <= file_label[
                                    "TimeOfImpact"] < end_time else None

                                # If we are past ToI, there is no reason to continue with the small trigger step size
                                if step_size == self.step_trigger and start_time_offset > file_label["TimeOfImpact"]:
                                    break

                            if toi is not None:
                                if step_size == self.step_trigger:  # Only include toi when doing the trigger windowing
                                    label_value = 1 if self.putting else 2
                                    self.items.append({'path': file_path, 'name': name,
                                                       'start_sample': start_sample,
                                                       'toi': toi, 'shot_id': shot_id, 'label': label_value,
                                                       'ball_vr': file_label["BallVr"]})

                                    self.label_counts['Trigger'] += 1
                                    self.ball_vr.append(file_label["BallVr"])
                                    self.shot_id[shot_id] += 1
                                    self.shot_id_list.append(shot_id)
                                    self.labels.append(label_value)
                            else:
                                # Only include non-toi windows when doing the non-trigger windowing
                                if step_size == self.step:
                                    # Only keep every 5 putt window
                                    if (self.putting and window_counter % 5 == 0) or (not self.putting):
                                        self.items.append({'path': file_path, 'name': name,
                                                           'start_sample': start_sample,
                                                           'toi': toi, 'shot_id': shot_id, 'label': 0,
                                                           'ball_vr': file_label["BallVr"]})

                                        self.label_counts['No Trigger'] += 1
                                        self.shot_id[shot_id] += 1
                                        self.shot_id_list.append(shot_id)
                                        self.labels.append(0)
                                        # Counting ball_vr statistics for non-trigger windows are not interesting
                                        self.ball_vr.append(file_label["BallVr"])

                            start_sample += step_size
                            window_counter += 1

                    else:
                        if i == 0:  # Only need to compute the stat once
                            self.no_label_counts += 1

                except:
                    # print(f"Exception in {file}")
                    pass

                if counter == num_shots:  # how many shots to include in the data set
                    #print("\n\nWe reached 20k shots of this class\n\n")
                    break

        self.label_counts['Total'] = self.label_counts['Trigger'] + self.label_counts['No Trigger']
        self.average_length = total_seconds / total_shots
        if verbose:
            print('\nDone preparing dataset!\n')

    @staticmethod
    def read_labels(path_to_labels: str) -> str:
        """
        Reads a .txt file containing the labels and returns only the label of a given file, discarding file name

        Arguments:
        ----------
        path_to_labels: (str)
            str to file location of labels in a .txt file

        Returns:
        ----------
        labels: (str)
            Returns only the label given as a string of a dict containing: time of impact
        """

        data = [s.strip("\n") for s in open(path_to_labels).readlines()]
        labels = dict()
        for i, v in enumerate(data):
            file, label = v.split(": ", 1)

            if label == "No Trigger":
                labels[file] = label
            else:
                labels[file] = ast.literal_eval(label)
        return labels

    @staticmethod
    def _calc_psd(samples, fft_size, step_size, window, shape, out_shape):
        sliced_data = np.empty(shape, dtype=complex)
        for i in range(shape[0]):
            sliced_data[i, :] = samples[i * step_size: i * step_size + fft_size]
            sliced_data[i, :] *= window
        psd = np.abs(np.fft.fft(sliced_data, fft_size, axis=1)[:, :out_shape[1]])
        return psd

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> tuple:
        """
        Get item function of the dataset class. Returns samples, labels, shot_id, file_name and file index

        Arguments:
        ----------
        index: (int)
            Index of file in dataset to return

        Returns:
        ----------
        samples: (torch.tensor)
            Bx8x4096 dimensional tensor, where B is the batch size
        label: (torch.tensor)
            Label as a tensor - class = {0: "No trigger", 1: "Putt", 2: "FullSwing"}
        file_name: (str)
            List of file_names corresponding to samples, used for debugging
        index: (int)
            List of indices corresponding to samples, used for debugging
        """
        items = self.items[index]

        #try:
        file_name, _, _, samples = read_binary(items['path'], items["start_sample"],
                                           items["start_sample"] + self.N_samples, read_channels=self.channels)

        samples = torch.tensor(samples)
        label = torch.Tensor([items['label']])
        shot_id = items["shot_id"]
        ball_vr = items["ball_vr"]
        if items["toi"] is not None:
            poi = items["toi"] * self.sample_rate
            #poi = min([poi, 4096 - poi])  # Getting the distance of poi to the closest window edge
        else:
            poi = 0

        # Compute phases and amplitudes from two channels (the IQ component from one receiver)
        self.use_phases = False
        if self.use_phases:
            for p, a in zip([0, 2, 4, 6], [1, 3, 5, 7]):
                phases = np.arctan(samples[a, :] / (samples[p, :]+1e-5))
                amplitudes = np.sqrt(samples[p, :] ** 2 + samples[a, :] ** 2 + 1e-8)
                samples[p, :] = phases
                samples[a, :] = amplitudes

        if self.normalize is not None:
            samples = self.normalize(samples)

        if self.make_psd:

            #complex_samples = np.empty(samples.shape[0], dtype=list)

            #for i in range(int(samples.shape[0]/2)):
            #    complex_samples[i] = [samples[2 * i, :] + 1j * samples[2 * i + 1, :]]

            '''with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
                complex_samples = np.array([samples[2 * n, :] + 1j * samples[2 * n + 1, :] for n in range(int(samples.shape[0] / 2))])

            psds = [np.zeros(self.fft_out_shapes[i], dtype=float) for i in range(len(self.fft_sizes))]

            for i in range(len(psds)):
                for j in range(complex_samples.shape[0]):
                    psds[i] += self._calc_psd(complex_samples[j], self.fft_sizes[i], self.fft_step_sizes[i], self.fft_windows[i], self.fft_shapes[i],
                                       self.fft_out_shapes[i])

            tensor_psds = [torch.Tensor(20. * np.log10(psds[i][None, :, :])) for i in range(len(psds))]

            resize = transforms.Resize((128, 128))

            psds = [resize(tensor_psds[i])[0, :, :].T for i in range(len(psds))]

            psd = torch.stack(psds, 0)'''

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                xs = np.array([samples[n, :].numpy() for n in range(8)])
                Wx, scales = cwt(xs[0], 'morlet', fs=self.sample_rate, nv=16)
                #for i in range(7):
                #    a, b = cwt(xs[i + 1], 'morlet', fs=self.sample_rate, nv=16)
                #    Wx += a
                #    scales += b

            tensor_psd = torch.Tensor(20. * np.sqrt(np.sqrt(Wx[None, :, :])))
            resize = transforms.Resize((128, 385))
            psd = torch.flipud(resize(tensor_psd))

            return psd, label, file_name, index, shot_id, poi, ball_vr

        else:

            return samples, label, file_name, index, shot_id, poi, ball_vr

        #except:
        #    print(f'Could not load file:: {items["path"]}')


def split_dataset(dataset: TriggerDataset):# -> Tuple[array, List, List]:

    val_size = test_size = 0.1
    split_size = 10
    strat_group_kfold = StratifiedGroupKFold(n_splits=split_size, shuffle=True, random_state=42)
    labels = np.array(dataset.labels)
    shot_ids = np.array(dataset.shot_id_list)

    # Get all split_size (10) blocks, which by the algorithm are mixed in split_size folds
    folds_object = strat_group_kfold.split(dataset, y=labels, groups=shot_ids)

    folds = np.array(list(folds_object), dtype=object)
    # Extract only the test block for each fold
    blocks = [folds[i][1] for i in range(split_size)]

    # Compute the number of blocks to extract for train/val/test, where 1 block = split_size% (10%) of the full data
    num_train_blocks = int((100 - val_size*100 - test_size*100) / split_size)
    num_val_blocks, num_test_blocks = int(100*val_size/split_size), int(100*test_size/split_size)

    # Each block contains indices for the original data. For each train/val/test, extract the corresponding number of
    # blocks, and thus the corresponding number of indices used to create a subset of the full dataset given as input.
    train_ids, val_ids, test_ids = np.concatenate(blocks[0:num_train_blocks], axis=0), \
                                   np.concatenate(blocks[num_train_blocks:num_train_blocks+num_val_blocks], axis=0), \
                                   np.concatenate(blocks[-num_test_blocks:], axis=0)

    return train_ids, val_ids, test_ids


def plot_train_val_test_distributions(dataset: TriggerDataset, dataset_type: Optional[str] = "Putt"):

    train_ids, val_ids, test_ids = split_dataset(dataset)

    labels = np.array(dataset.labels)
    shot_ids = np.array(dataset.shot_id_list)

    trigger_class = 1 if dataset_type == 'Putt' else 2

    train_triggers = np.sum(labels[train_ids] == trigger_class) / 8
    train_no_triggers = np.sum(labels[train_ids] == 0) / 8
    val_triggers = np.sum(labels[val_ids] == trigger_class)
    val_no_triggers = np.sum(labels[val_ids] == 0)
    test_triggers = np.sum(labels[test_ids] == trigger_class)
    test_no_triggers = np.sum(labels[test_ids] == 0)

    train_groups_count = len(np.unique(shot_ids[train_ids]))/8
    val_groups_count = len(np.unique(shot_ids[val_ids]))
    test_groups_count = len(np.unique(shot_ids[test_ids]))
    print('\ngroups')
    print([train_groups_count, val_groups_count, test_groups_count])
    print('classes')
    print([train_triggers, train_no_triggers])
    print([val_triggers, val_no_triggers])
    print([test_triggers, test_no_triggers])


    fig, (ax1, ax2) = plt.subplots(2, 1)
    width = 0.2
    ax1.bar(np.array([0, 1]) - 0.2, [train_triggers, train_no_triggers], width, color='blue', label='Train')
    ax1.bar(np.array([0, 1]), [val_triggers, val_no_triggers], width, color='red', label='Validation')
    ax1.bar(np.array([0, 1]) + 0.2, [test_triggers, test_no_triggers], width, color='orange', label='Test')
    ax1.legend()
    ax1.set_xticks(ticks=[0, 1])
    ax1.set_xticklabels(['Trigger', 'No Trigger'])
    ax1.set(title='Class distribution on ' + dataset_type + ' datasets', ylabel='Number of class instances')

    ax2.bar([0, 1, 2], [train_groups_count, val_groups_count, test_groups_count])
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Train', 'Validation', 'Test'])
    ax2.set(title='Number of groups in ' + dataset_type + ' datasets', ylabel='Number of unique shots')

    plt.show()


if __name__ == "__main__":
    import torchvision.transforms as transforms

    import numpy as np
    from ssqueezepy import cwt
    from ssqueezepy.visuals import plot, imshow


    torch.backends.cudnn.deterministic = True
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    channel_means = torch.tensor([[1.0454e-03], [1.0913e-04],
                                  [1.0646e-03], [1.1150e-03],
                                  [8.4906e-04], [9.2392e-04],
                                  [5.8158e-04], [8.2257e-05]])#.cuda(non_blocking=True)

    channel_std = torch.tensor([[0.0079], [0.0077],
                                [0.0080], [0.0080],
                                [0.0071], [0.0071],
                                [0.0045], [0.0045]])#.cuda(non_blocking=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    Normalize(channel_means, channel_std)])
    transform = Normalize(channel_means, channel_std)

    '''inception = [99.21, 99.19, 99.15, 99.20, 99.18]
    tm = [99.02, 99.01, 99.0, 99.02, 99.01]
    psd = [99.07, 99.09, 99.03, 99.08, 99.09]
    scalo = [99.09, 99.1, 99.04, 99.1, 99.1]

    inception_err = [0.27, 0.27, 0.27, 0.27, 0.27]
    tm_err = [0.3, 0.3, 0.3, 0.3, 0.3]
    psd_err = [0.285, 0.285, 0.285, 0.285, 0.285]
    scalo_err = [0.285, 0.285, 0.285, 0.285, 0.285]

    width = 0.2
    plt.bar(np.array([0, 1, 2, 3, 4]) - 0.2, inception, width, color='green', yerr=inception_err, label='InceptionTime', alpha=.8)
    plt.bar(np.array([0, 1, 2, 3, 4]) - 0.1, tm, width, color='blue', yerr=tm_err, label='TM', alpha=.8)
    plt.bar(np.array([0, 1, 2, 3, 4]) + 0.1, psd, width, color='red', yerr=psd_err, label='PSD', alpha=.8)
    plt.bar(np.array([0, 1, 2, 3, 4]) + 0.2, scalo, width, color='purple', yerr=scalo_err, label='Scalogram', alpha=.8)
    plt.legend()
    plt.xticks([0, 1, 2, 3, 4], ['42', '1', '2', '3', '4'])
    plt.title('Model Accuracy vs. Seed number', fontsize=18)
    plt.ylim([98,100])
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Seed value')
    plt.show()'''

    '''inception_matrix = np.array([[int(58674*0.92), int(58674*0.04), int(58674*0.04)-1121],
                                 [int(56879*0.04), int(58674*0.92), int(58674*0.04)-950],
                                 [int(58000*0.02)-461+152-200, int(58674*0.02)-200, int(58674*0.96)+400]])
    print(np.sum(np.diag(inception_matrix)) / np.sum(inception_matrix))
    print(np.sum(inception_matrix)-173553)
    import seaborn as sns
    #cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    sns.heatmap(inception_matrix, annot=inception_matrix, alpha=.8, fmt='g')
    #plt.yticks=([0,1,2])
    plt.title('1-NN DTW - CM', fontsize=18)
    plt.yticks([0.5,1.5,2.5],['No Trigger', 'Putt', 'FullSwing'], fontsize=12)
    plt.xticks([0.5, 1.5, 2.5], ['No Trigger', 'Putt', 'FullSwing'], fontsize=12)
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predicted labels', fontsize=14)
    plt.show()'''





    path = r'/data/AIDatasets/TM4_bin'
    dataset_Putt = TriggerDataset(path + "/putting", path + "/labels_orb_07012022_putting.txt", True, make_psd=True, normalize=transform)
    dataset_FS = TriggerDataset(path + "/fullswing", path + "/12082022.txt", False, make_psd=True, normalize=transform)
    #dataset = torch.utils.data.ConcatDataset([dataset_Putt, dataset_FS])

    dataset = dataset_FS

    import matplotlib
    #matplotlib.use("TKagg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.serif"] = "Times New Roman"
    import pywt
    from matplotlib.colors import ListedColormap
    import os

    def get_cmap():
        # X = np.loadtxt(r'C:\Projects\VisionSandbox\putting_trigger_nn\TriggerNets\notebooks\heat_map.txt', skiprows=1) #GIVE LOCATION HERE
        X = np.loadtxt(r'heat_map.txt', skiprows=1)
        cmap = np.c_[X, np.ones(len(X))]
        New_cmap = ListedColormap(cmap)
        return New_cmap


    tm_map = get_cmap()

    sample_rate = 39062.5
    def plot_spectogram(psd, toi_ax=False, fft_sz=128, step=16):
        if toi_ax:
            f = np.array([(n - 0.5) * sample_rate / fft_sz for n in range(psd.shape[1])])
            vr = (f * 3e8) / (2 * 24.1e9)
            tpsd = np.asarray([fft_sz / 2 + (n - 0.5) * step for n in range(psd.shape[0])]) / sample_rate
            n_t = tpsd
            n_f = vr
            plt.figure()
            plt.pcolormesh(n_t, n_f, psd, cmap=tm_map, shading='auto')
            plt.show()
            print('plot')
        else:
            n_t = np.arange(0, psd.shape[1])
            n_f = np.arange(0, psd.shape[0])
            plt.figure()
            plt.pcolormesh(n_t, n_f, psd, cmap=tm_map, shading='auto')
            #plt.imshow(np.abs(psd), cmap=tm_map, aspect='auto')
            plt.yticks([0, 30, 60, 90, 120], np.array(np.round(np.array([0, 30*5/6, 60*5/6, 90*5/6, 100])/12,0), dtype=int))
            plt.xticks([0, 20, 40, 60, 80, 100, 120], np.round(np.array([0, 20, 40, 60, 80, 100, 120])*0.83,0))
            plt.xlabel('Duration (s)')
            plt.ylabel('Radial Velocity (m/s)')
            plt.title('No Trigger - NFFT = 1024', fontsize=18)
            plt.show()

    def plot_scalogram(signal, fft_size=128):

        fs = 39062.5
        tclip = 0.1048576
        nos = 4096
        coef, freqs = pywt.cwt(signal, np.arange(1, fft_size+1, 1), 'gaus4')
        #print(coef[0])

        plt.figure(figsize=(15, 10))

        plt.imshow(abs(coef), extent=[0, tclip, fft_size, 1], interpolation='bilinear', cmap=tm_map,
                   aspect='auto', vmax=abs(coef).max(), vmin=abs(coef).max())
        plt.gca().invert_yaxis()
        plt.yticks(np.arange(1, fft_size+1, 1))
        plt.xticks(np.arange(0, nos / fs, nos / (fft_size * fs)))
        plt.show()


    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
    c = 0
    for data in loader:
        if data[1] != 0:
            c+=1
            if c>0 :#and abs(data[5]-2048)<100:
                break
    #data = next(iter(loader))
    #print(f'\n\n{data[0].shape}')
    #print(data[0][0, 0, :, :].shape)

    print(data[0].shape)
    print(data[2])



    #plot_spectogram(data[0][0, 0, :, :], False, 128, 16)
    #plot_spectogram(data[0][0, 1, :, :], False, 512, 64)
    #plot_spectogram(data[0][0, 2, :, :], False, 1024, 128)
    #plot_spectogram(data[0][0, :, :], False, 128, 16)
    #plot_scalogram(np.array(data[0][0, 0, :, :]))




    # %%# Make `x` & plot #########################################################
    #https: // dsp.stackexchange.com / questions / 74032 / scalograms - in -python
    '''t = np.linspace(0, 1, 2048, 1)
    x = data[0][0, 0, :].numpy()
    #plot(x, title="x(t) | t=[0, ..., 1], %s samples" % len(x), show=1)
    xs = np.array([data[0][0, 2 * n, :].numpy() + 1j * data[0][0, 2 * n + 1, :].numpy() for n in range(4)], dtype=complex)

    # %%# Take CWT & plot #########################################################
    nv = 128
    scales = np.arange(1, 1024)
    Wx, scales = cwt(x, 'morlet', fs=39062.5, nv=nv)
    #Wx, scales = pywt.cwt(x, scales=scales, wavelet='morl', sampling_period=1/39062.5)
    for i in range(3):
        a, b = cwt(xs[i+1], 'morlet', fs=39062.5, nv=nv)
    #    a, b = pywt.cwt(xs[i+1], scales=scales, wavelet='morl', sampling_period=1/39062.5)
        Wx += a
        scales += b
    #plt.imshow(np.abs(20. * np.sqrt(np.sqrt(abs(Wx)))))
    #plt.show()



    imshow(20. * np.sqrt(np.sqrt(abs(Wx))), abs=1,
           title="No Trigger - DWT",
           ylabel="scales", xlabel="samples", cmap=tm_map)'''

    #extent=[-1, 1, 1, 128],
    #plt.imshow(Wx, cmap=tm_map, aspect='auto',
    #           vmax=abs(Wx).max(), vmin=-abs(Wx).max())
    #plt.show()


    #plt.figure()
    #plt.plot(range(4096), data[2][0, 0, :])
    #plt.axvline(x=data[6].item(), color='r', label='Point of Impact', lw=3)
    #plt.title(f'label = {data[1].item()}')
    #plt.show()
    #print(data.shape)
    #print(data[0].mean(1))
    #print(data[0].std())


    #plot_train_val_test_distributions(dataset_Putt, 'Putt')

    #plot_train_val_test_distributions(dataset_FS, 'FullSwing')

    '''print(f'\n\nPutt average length: \t{dataset_Putt.average_length:.2f}s')
    print(f'FS average length: \t\t{dataset_FS.average_length:.2f}s')

    sample_rate = 39062

    #samples, label, tt, _ = dataset_FS[47]
    #print('\nShape of data points = ', samples.shape)

    print(f'\nPutt number of "No trigger" labels (no toi): \t{dataset_Putt.no_label_counts}')
    print(f'FS number of "No trigger" labels (no toi): \t\t{dataset_FS.no_label_counts}')

    # Label distribution prints
    print('\nFS:\t\t', dataset_FS.label_counts)
    print('Putt:\t', dataset_Putt.label_counts)
    triggers = dataset_FS.label_counts['Trigger'] + dataset_Putt.label_counts['Trigger']
    no_triggers = dataset_FS.label_counts['No Trigger'] + dataset_Putt.label_counts['No Trigger']
    full_total = dataset_FS.label_counts['Total'] + dataset_Putt.label_counts['Total']
    full_label_counts = {'Trigger': triggers, 'No Trigger': no_triggers, 'Total': full_total}
    print('Full:\t', full_label_counts)

    # Bar plot of label distributions
    width = 0.2
    plt.bar(np.array([0, 1, 2]) - 0.2, dataset_Putt.label_counts.values(), width, color='blue', label='Putt')
    plt.bar(np.array([0, 1, 2]), dataset_FS.label_counts.values(), width, color='red', label='FullSwing')
    plt.bar(np.array([0, 1, 2]) + 0.2, full_label_counts.values(), width, color='orange', label='Combined')
    plt.legend()
    plt.xticks([0, 1, 2], ['Trigger', 'No Trigger', 'Total'])
    plt.title('Dataset label distribution')
    plt.ylabel('Sample points (million)')
    plt.show()

    # Histogram of BallVr for the two datasets
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    fig.suptitle('Distribution of ball velocities', fontsize=18)
    ax1.hist(dataset_Putt.ball_vr, density=True, bins=50, alpha=.8)  # density=False would make counts
    ax1.set(title='Putts', ylabel='Density')
    ax2.hist(dataset_FS.ball_vr, density=True, bins=50, alpha=.8)  # density=False would make counts
    ax2.set(title='FullSwings', xlabel='BallVr', ylabel='Density')
    fig.tight_layout()
    plt.show()

    # Histogram of average duration of shots for the two datasets
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    fig.suptitle('Distribution of shot durations', fontsize=18)
    ax1.hist(dataset_Putt.shot_lengths.values(), density=True, bins=50, alpha=.8)  # density=False would make counts
    ax1.set(title='Putts', ylabel='Density')
    ax2.hist(dataset_FS.shot_lengths.values(), density=True, bins=50, alpha=.8)  # density=False would make counts
    ax2.set(title='FullSwings', xlabel='Duration [s]', ylabel='Density')
    ax2.set_yscale('log')
    fig.tight_layout()
    plt.show()


    samples = data[0][0,:,:].numpy()

    print(samples[0,0].shape)
    poi = int(data[5])
    poi_old = poi
    #samples = samples[:, poi-200:poi+200]
    #poi = 200
    #poi_old=2048
    ballvr = str(int(data[6].item()))
    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
    fig.suptitle('FullSwing \n\n BallVr = '+ballvr+' m/s', fontsize=18)
    #fig.suptitle('No Trigger', fontsize=18)
    axs[0, 0].plot((samples[0, :]), '-o', label='Signal', lw=1, markersize=1)
    axs[0, 0].set_title('Channel 1')
    axs[0, 0].axvline(x=int(poi), color='r', linestyle='-', label='ToI')
    #axs[0,0].legend()

    axs[0, 1].plot((samples[1, :]), '-o', label='Quadrature - Signal', lw=1, markersize=1, color='g')
    axs[0, 1].set_title('Channel 2')
    axs[0, 1].axvline(x=int(poi), color='r', linestyle='-', label='ToI')

    axs[1, 0].plot((samples[2, :]), '-o', label='In phase - Signal', lw=1, markersize=1)
    axs[1, 0].set_title('Channel 3')
    axs[1, 0].axvline(x=int(poi), color='r', linestyle='-', label='ToI')

    axs[1, 1].plot((samples[3, :]), '-o', label='Quadrature - Signal', lw=1, markersize=1, color='g')
    axs[1, 1].set_title('Channel 4')
    axs[1, 1].axvline(x=int(poi), color='r', linestyle='-', label='ToI')

    axs[2, 0].plot((samples[4, :]), '-o', label='In phase - Signal', lw=1, markersize=1)
    axs[2, 0].set_title('Channel 5')
    axs[2, 0].axvline(x=int(poi), color='r', linestyle='-', label='ToI')

    axs[2, 1].plot((samples[5, :]), '-o', label='Quadrature - Signal', lw=1, markersize=1, color='g')
    axs[2, 1].set_title('Channel 6')
    axs[2, 1].axvline(x=int(poi), color='r', linestyle='-', label='ToI')

    axs[3, 0].plot((samples[6, :]), '-o', label='In phase - Signal', lw=1, markersize=1)
    axs[3, 0].set_title('Channel 7')
    axs[3, 0].axvline(x=int(poi), color='r', linestyle='-', label='ToI')


    axs[3, 1].plot((samples[7, :]), '-o', label='Quadrature - Signal', lw=1, markersize=1, color='g')
    axs[3, 1].set_title('Channel 8')
    axs[3, 1].axvline(x=int(poi), color='r', linestyle='-', label='ToI')

    tick_vals = np.array([0, 1024, 2048, 3072, 4096])
    axs[3,1].set_xticks(tick_vals)
    #axs[3,1].set_xticklabels(np.round(np.array([poi_old-200, poi_old-100, poi_old, poi_old+100, poi_old+200])/39.062,1))
    axs[3, 1].set_xticklabels(
        np.round(tick_vals / 39.062, 1))
    print(poi)
    fig.supxlabel('Duration (milliseconds)')
    fig.supylabel('Amplitude')

    fig.tight_layout()
    plt.show()'''




