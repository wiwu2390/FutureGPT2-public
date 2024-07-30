import numpy as np
from torch import optim, nn, Tensor
from torch.nn import functional as F
import torch
import wandb
from transformers import GPT2Config, GPT2Model
import transformers
import lightning as L
from inspect import signature, _ParameterKind
import copy
import gc
import datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader

class SyntheticSeqDataset(Dataset):
    def __init__(self, size, seq_len=64, f=lambda x: torch.sin(10*x), conv=10, shift=0, p=1):
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.f = f
        #self.thresh = thresh
        self.conv = conv
        self.shift = shift
        self.p = p
        assert 0 <= self.p <= 1
        self.x = torch.normal(0, 1, (size, self.seq_len, 1))
        self.z = torch.bernoulli(torch.ones((size, self.seq_len, 1)) * self.p)
        

    def __len__(self):
        return self.size

    def __getitem__(self, value):
        x = self.x[value]
        z = self.z[value]
        if self.shift > 0:
            xsh = torch.concat([torch.zeros((self.shift, 1)), x[:-self.shift,:]], dim=0)
        else:
            xsh = x
        if self.conv == 0:
            y = self.f(xsh)
        elif self.conv >= self.seq_len:
            y = torch.cumsum(self.f(xsh), dim=0)
        else:
            fxpad = F.pad(
                self.f(xsh[:,0]),
                (self.conv-1, 0)
            ).unsqueeze(0).unsqueeze(0)
            y = F.conv1d(
                fxpad,
                torch.ones(1, 1, self.conv)
            )[0,0,:].unsqueeze(dim=1)
        y = z * y + (1 - z) * x
        # if 0 < self.p < 1:
        #     x = torch.concat([x, z], dim=-1)
        x = torch.concat([x, z], dim=-1)
        # y = torch.where(
        #     x[:,2] > self.thresh,
        #     fx1conv,
        #     x[:,0],
        # ).unsqueeze(-1)
        return {'x': x, 'y': y}
