import numpy as np
from torch import optim, nn, Tensor
from torch.nn import functional as F
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoConfig
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
# from deepspeed.ops.adam import DeepSpeedCPUAdam

class LitGPTModel(L.LightningModule):
    def __init__(
        self,
        model_name='gpt2',
        pretrain=True,
        lr=6e-4,
        warmup=0.01,
        decay=0.01,
        deepspeed=False,
        loss_mask=None,
        acc_cutoff=None,
    ):
        super().__init__()
        args = vars()
        for param in list(signature(LitGPTModel.__init__).parameters)[1:]:
            setattr(self, param, args[param])
        config = AutoConfig.from_pretrained(model_name)
        config.upcast_attn = True
        if pretrain:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        else:
            self.model = AutoModelForCausalLM.from_config(config=config)
        if self.loss_mask is not None:
            self.loss_mask = nn.Parameter(torch.Tensor(self.loss_mask), requires_grad=False)
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model.forward(
            input_ids=batch['input_ids'],
            #attention_mask=batch['attention_mask'],
            # labels=batch['input_ids'],
            use_cache=True,
        )

    def _compute_loss(self, batch):
        labels = batch['input_ids']
        logits = self.forward(batch).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        if self.loss_mask is None:
            batch_loss_mask = torch.ones(flat_labels.shape, device=flat_labels.device)
        else:
            # self.loss_mask should be 1d of shape (seq_length,)
            # broadcast up to (batch_size, seq_length)
            batch_loss_mask = torch.broadcast_to(self.loss_mask, shift_labels.shape).contiguous().view(-1)
        return (loss_fct(flat_logits, flat_labels) * batch_loss_mask).sum() / batch_loss_mask.sum()

    def _compute_acc(self, batch):
        x = batch['input_ids']
        x_cut = x[:,:self.acc_cutoff]
        out = self.model.generate(
            x_cut, max_length=x.shape[-1],
            attention_mask=torch.ones(x_cut.shape, device=x.device),
            pad_token_id=50256, # tokenizer.eos_token_id
            use_cache=False, # for compatibility with myopic model
        )
        return (out == x).all(dim=1).to(torch.float).mean()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train_loss', loss.item(), on_step=True)
        self.log('global_step', self.trainer.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('val_loss', loss.item())
        if self.acc_cutoff is not None:
            acc = self._compute_acc(batch)
            self.log('val_acc', acc.item())
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('test_loss', loss.item())
        return loss

    def configure_optimizers(self):
        if self.deepspeed:
            optimizer = DeepSpeedCPUAdam(
                model_params=self.model.parameters(),
                lr=self.lr,
                weight_decay=self.decay,
            )
        else:
            optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=self.lr,
                weight_decay=self.decay,
            )
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
