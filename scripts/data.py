import os
import numpy as np
from torch.utils.data import Dataset
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LargeTextDataset(Dataset):
    def __init__(self, bin_path: str, context_length: int = 256):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        # 1. Slice the memmap object. The OS reads only this small chunk from disk.
        chunk_np = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        # 2. Convert ONLY this tiny chunk to a PyTorch tensor.
        chunk_pt = torch.from_numpy(chunk_np)
        x = chunk_pt[:-1]
        y = chunk_pt[1:]
        return x, y


class TextDataset(Dataset):
    def __init__(self, data, context_length: int = 512):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        
        x = self.data[idx: idx+self.context_length]
        y = self.data[idx+1: idx+self.context_length+1]
        return x, y

