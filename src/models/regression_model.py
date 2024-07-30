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

class LitGPT2RegModel(L.LightningModule):
    '''
    GPT2 for seq-to-seq regression
    '''
    def __init__(
        self,
        in_dim=1,
        out_dim=1,
        lr=1e-4,
        warmup=0.01,
        decay=0.01,
        **kwargs,
    ):
        super().__init__()
        args = vars()
        for param in list(signature(LitGPT2RegModel.__init__).parameters)[1:]:
            setattr(self, param, args[param])
        config = GPT2Config(**kwargs)
        self.model = GPT2Model(config)
        self.model.wte = nn.Linear(in_dim, config.n_embd)
        self.model.unembed = nn.Linear(config.n_embd, out_dim)
        self.loss_fn = nn.HuberLoss()
        self.decay = decay
        #self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, batch):
        # Unexpected shape for gpt2, so manually construct position_ids and inputs_embeds
        x = batch['x']
        out = self.model(
            inputs_embeds=self.model.wte(x),
            position_ids=torch.arange(0, x.shape[-2], 1, device=x.device),
        ).last_hidden_state
        return self.model.unembed(out)

    def loss(self, batch):
        y_hat = self.forward(batch)
        return self.loss_fn(y_hat, batch['y'])
        #torch.mean((y_hat - batch['y']) ** 2)

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('train_loss', loss.item(), on_step=True)
        self.log('global_step', self.trainer.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('val_loss', loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('test_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.decay,
        )
        # scheduler = transformers.get_constant_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=self.num_warmup_steps,
        #     #num_training_steps=9200 #1 epoch #self.trainer.estimated_stepping_batches,
        # )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.warmup * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        print('NUM TRAINING STEPS', self.trainer.estimated_stepping_batches)
        # HF's schedulers are on 'step' interval
        return(
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}]
        )