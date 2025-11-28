import torch
import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Type, Union

import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex

def time_to_frequency(data):
    """
    Convert time series data to frequency domain using PyTorch.
    Args:
    data (torch.Tensor): The time series data, expected shape (batch_size, sequence_length)
    Returns:
    torch.Tensor: The frequency domain representation of the data.
    """
    frequency_data = np.fft.fft(data)
    real_part = frequency_data.real
    imag_part = frequency_data.imag
    freq_feature = np.log(np.sqrt(real_part**2 + imag_part**2))
    return freq_feature

class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, augmentations=None):
        self.root = Path(root)
        self.data = np.load(Path(root), allow_pickle=True)
        self.augmentations = augmentations
    def __getitem__(self, index):
        x = self.data[index]
        x_fft = np.array()
        for i in range(x.shape[1]):
            column = x[:, i]
            x_fft = np.insert(x_fft, x_fft.shape[1], time_to_frequency(column), axis=1)
        train_item = [torch.from_numpy(x),torch.from_numpy(x_fft)]
        return train_item, -1 

    def __len__(self):
        return len(self.data)

class CustomDatasetWithLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.data = np.load(Path(root), allow_pickle=True)
        self.x = self.data[:, :self.data.shape[1]-1]
        self.y = self.data[:, self.data.shape[1]-1]

    def __getitem__(self, index):
        x = self.x[index]
        x_fft = np.array()
        for i in range(x.shape[1]):
            column = x[:, i]
            x_fft = np.insert(x_fft, x_fft.shape[1], time_to_frequency(column), axis=1)
        train_item = [torch.from_numpy(x),torch.from_numpy(x_fft)]
        return train_item, self.y[index]

    def __len__(self):
        return len(self.data)

def prepare_datasets(
    dataset: str,
    augmentations: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
    data_fraction: float = -1.0,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"
    if dataset in ["set1", "set2", "set3"]:
        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = CustomDatasetWithLabels

        train_dataset = dataset_with_index(dataset_class)(train_data_path, augmentations) 
    elif dataset in ["CrossPlatform-Android", "CICIoT2022", "ISCXTor2016","ISCXVPN2016","CrossPlatform-iOS","USTCTFC2016"]:
         # simple augmentation
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            ])
        train_dataset = ImageFolder(os.path.join(train_data_path, 'train'), transform=transform_train)
                     
    ## date count 
    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        from sklearn.model_selection import train_test_split

        if isinstance(train_dataset, CustomDatasetWithoutLabels):
            files = train_dataset.data
            (files,_,) = train_test_split(files, train_size=data_fraction, random_state=42)
            train_dataset.data = files
        else:
            data = train_dataset.samples
            files = [f for f, _ in data]
            labels = [l for _, l in data]
            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset

def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


if __name__ == "__main__":
    # Example usage
    # Create a sample time series data
    set_path = '/home/model-server/code/solo-learn/dataset/array.npy'
    dataset = np.load(set_path, allow_pickle=True)
    temp_data = torch.from_numpy(dataset[0][:1])
    frequency_data = torch.fft.fft(temp_data)
    real_part = frequency_data.real
    imag_part = frequency_data.imag
    freq_feature = torch.log(torch.sqrt(real_part**2 + imag_part**2))
    # train_dataset = prepare_datasets("set1", transform = None, train_data_path=set_path,no_labels = True)
    print(frequency_data)  # Should show (10, 256)