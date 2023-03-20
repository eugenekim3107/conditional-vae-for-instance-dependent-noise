import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file).values
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X = self.data[index, 1:].reshape(28, 28) / 255.
        y = torch.tensor(self.data[index, 0], dtype=torch.int64)
        if self.transform:
            X = self.transform(X)
        return X.to(torch.float32), y
    
class MNISTTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file).values
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X = self.data[index, :].reshape(28, 28) / 255.
        if self.transform:
            X = self.transform(X)
        return X.to(torch.float32)
    

class MNISTNoiseDataset(Dataset):
    def __init__(self, X_file, y_file):
        self.X = torch.load(X_file)
        self.y = torch.load(y_file)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        X = self.X[index] / 255.
        y = self.y[index]
        return X.to(torch.float32), y.to(torch.int64)