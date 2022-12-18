
__init__ = ['get_loaders']

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from .dataloader import TriggerDataset, ToTensor, Normalize
from sklearn.model_selection import StratifiedGroupKFold
from typing import Tuple

random_seed = 42


def load_dataset(data_path: str, putt: bool, normalize: bool = True,
                 num_shots: int = 20000, make_psd: bool = True, gpu='cuda') -> TriggerDataset:
    """Return TriggerDataset"""

    channel_means = torch.tensor([[1.0454e-03], [1.0913e-04],
                                  [1.0646e-03], [1.1150e-03],
                                  [8.4906e-04], [9.2392e-04],
                                  [5.8158e-04], [8.2257e-05]]).to(gpu, non_blocking=True)

    channel_std = torch.tensor([[0.0079], [0.0077],
                                [0.0080], [0.0080],
                                [0.0071], [0.0071],
                                [0.0045], [0.0045]]).to(gpu, non_blocking=True)

    transform = transforms.Compose([Normalize(channel_means, channel_std)]) if normalize else None
    transform = Normalize(channel_means, channel_std)

    if putt:
        return TriggerDataset(data_path + "/putting", data_path + "/labels_orb_07012022_putting.txt", True,
                              normalize=transform, num_shots=num_shots, make_psd=make_psd)
    else:
        return TriggerDataset(data_path + "/fullswing", data_path + "/12082022.txt", False,
                              normalize=transform, num_shots=num_shots, make_psd=make_psd)


def split_dataset(dataset: TriggerDataset, val_size: float = 0.1,
                  test_size: float = 0.1, verbose: bool = True) -> Tuple[Subset, Subset, Subset]:
    """Returns Stratified Group split dataset"""

    if verbose:
        print('\nSplitting dataset...')
    split_size = 10
    strat_group_kfold = StratifiedGroupKFold(n_splits=split_size, shuffle=True, random_state=random_seed)
    labels = np.array(dataset.labels)
    shot_ids = np.array(dataset.shot_id_list)

    kfolds_object = strat_group_kfold.split(dataset, y=labels, groups=shot_ids)
    kfolds = np.array(list(kfolds_object), dtype=object)

    blocks = [kfolds[i][1] for i in range(split_size)]

    num_train_blocks = int((100 - val_size*100 - test_size*100) / split_size)
    num_train_blocks = 2  # 1 block = 2k shots, so half of that is 1k shot
    num_val_blocks, num_test_blocks = int(100*val_size/split_size), int(100*test_size/split_size)

    train_ids, val_ids, test_ids = np.concatenate(blocks[0:num_train_blocks], axis=0), \
                                   np.concatenate(blocks[num_train_blocks:num_train_blocks+num_val_blocks], axis=0), \
                                   np.concatenate(blocks[-num_test_blocks:], axis=0)
    #train_ids = train_ids[0:int(len(train_ids)/2)]
    #unique_train_ids = np.unique(train_ids)
    #subset_unique_train_ids = unique_train_ids[0:int(len(unique_train_ids)/2)]
    #train_ids = train_ids[train_ids == subset_unique_train_ids]

    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)
    test_dataset = Subset(dataset, test_ids)
    #print(train_ids.shape)

    if verbose:
        print('Done splitting dataset!\n')

    return train_dataset, val_dataset, test_dataset


def merge_datasets(data_path: str, val_size: float = 0.1, test_size: float = 0.1, normalize: bool = False,
                   num_shots: int = 20000, make_psd: bool = False, gpu=None) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset]:
    """Return merged datasets for Putt and FullSwing"""

    dataset_Putt = load_dataset(data_path, num_shots=num_shots, putt=True, normalize=normalize, make_psd=make_psd, gpu=gpu)
    dataset_FS = load_dataset(data_path, num_shots=num_shots, putt=False, normalize=normalize, make_psd=make_psd, gpu=gpu)

    train_putt, val_putt, test_putt = split_dataset(dataset_Putt, val_size, test_size)
    train_fs, val_fs, test_fs = split_dataset(dataset_FS, val_size, test_size)

    train_dataset = ConcatDataset([train_putt, train_fs])
    val_dataset = ConcatDataset([val_putt, val_fs])
    test_dataset = ConcatDataset([test_putt, test_fs])

    return train_dataset, val_dataset, test_dataset


def get_loaders(data_path: str, num_shots: int = 20000, batch_size: int = 128, val_size: float = 0.1, test_size: float = 0.1,
                normalize: bool = False, shuffle: bool = True, make_psd: bool = False, verbose: bool = True, gpu=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders of merged Putt and FullSwing datasets

    Arguments
    ----------
    data_path: str
        Path to data
    batch_size: int
        The batch size each iteration of the dataloader should return
    val_size: float
        The fraction of training data to use for validation
    test_size: float
        The fraction of training data to use for testing
    normalize: bool
        Whether to normalize the training data or not

    Returns
    ----------
    Tuple of (train_loader, val_loader, test_loader)
    """

    if verbose:
        print('\nGetting data loaders...')
    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = merge_datasets(data_path, val_size, test_size, normalize,
                                                              num_shots=num_shots, make_psd=make_psd, gpu=gpu)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    if verbose:
        print('Done getting data loaders!\n')

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_path = r'/data/AIDatasets/TM4_bin'
    train_loader, val_loader, _ = get_loaders(data_path=data_path, batch_size=64, val_size=0.1, test_size=0.1,
                                              normalize=False)

