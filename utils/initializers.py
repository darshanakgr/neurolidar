import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from utils.datasets import VTWDataset
from utils.configs import TrainingConfig


def init_datasets(hdf5_file: str, metadata_file: str, clip: bool, downsample: float = 0.0) -> Dataset:
    df = pd.read_csv(metadata_file)
    train_df = df[df.split == "train"]
    test_df = df[df.split == "test"]
    train_dataset = VTWDataset(hdf5_file, train_df, clip, downsample)
    test_dataset = VTWDataset(hdf5_file, test_df, clip, downsample)

    return train_dataset, test_dataset

def init_dataloaders(train_dataset: Dataset, test_dataset: Dataset, config: TrainingConfig) -> tuple:
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, prefetch_factor=5)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, prefetch_factor=5)

    return train_loader, test_loader

def init_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    if config.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "nadam":
        return optim.NAdam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
