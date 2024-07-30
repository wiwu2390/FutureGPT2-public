#!/usr/bin/env python
import wandb
import models.future_model
from lightning.pytorch.loggers import WandbLogger
import os
from lightning import Trainer
import datasets
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from models.necks import *
from models.future_model import *
from pprint import pprint
from datetime import datetime
from scripts.utils import *

# TODO: Set up argparse
# TODO: move params into a YAML or something

FAST_DEV_RUN = False

def kv2d(keys, value_list):
    return [
        {k: v for k, v in zip(keys, values)}
        for values in value_list
    ]

optim_keys = ['neck_lr', 'lr_scheduler_name', 'num_warmup_steps', 'num_restarts']
optim_values = (
    (4e-4, 'constant_with_warmup', 1000, 0),
    # (2e-3, 'constant_with_warmup', 5000, 0),
    # (2e-3, 'cosine_with_restarts', 5000, 0),
    # (2e-4, 'cosine_with_restarts', 5000, 0),
    # (2e-3, 'cosine_with_restarts', 500, 4),
)
optim_params = kv2d(optim_keys, optim_values)

MLP_keys = ['depth', 'layer_dims', 'use_next']
MLP_values = (
    (1, [], False),
    # (1, [], True),
    # (4, [], True),
    # (7, [], True),
    # (13, [], True),
    # (13, [256], True),
    # (13, [512], True),
    # (2, [256], True),
    # (4, [256], True),
)
MLP_params = kv2d(MLP_keys, MLP_values)

LSTM_keys = ['depth', 'neck_size', 'num_layers', 'use_next']
LSTM_values = [
    (1, 128, 2, True),
    (1, 128, 1, True),
    (4, 128, 1, True),
]
LSTM_params = kv2d(LSTM_keys, LSTM_values)

wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)

dataset = datasets.load_from_disk('/corpus/msmarco/msmarco_GPT2_64tokens_full').with_format('torch')
loaders = {
    split: DataLoader(dataset[split], batch_size=128)
    for split in ['train', 'val', 'test']
}

def do_run(name, model):
    print('STARTING RUN', name)
    wandb_logger = WandbLogger(
        name=name,
        project='LAISR_FUTURE_GPT2',
        log_model=False,   # Only save checkpoints locally
    )
    lr_callback = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath="/checkpoints/",
        # can't zero pad global_step b/c the logger casts it to a float >:(((
        # filename="{name}-{epoch:02d}-{global_step:.0f}-{val_total_loss:.2f}",
        filename=name + "-{epoch:02d}-{val_total_loss:.2f}",
        # every_n_train_steps=5_000,
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_total_loss',
        mode='min',
        # save_on_train_epoch_end=True,
    )
    early_stop_callback = EarlyStopping(
        monitor='train_self_loss_step',
        divergence_threshold=15,
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min',
    )
    trainer = Trainer(
        fast_dev_run=FAST_DEV_RUN,
        logger=wandb_logger,
        val_check_interval=.1,
        callbacks=[checkpoint_callback, lr_callback, early_stop_callback],
        max_epochs=2,
        enable_progress_bar=False,
    )
    wandb_logger.watch(model.future_neck, log='all')
    trainer.fit(
        model=model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val']
    )
    wandb.finish()

for op in optim_params:
    for mp in MLP_params:
        name = '_'.join(
            [
                'MLPNECK',
                run_id(),
                'OPTIM',
                '_'.join(f'{k}-{v}' for k, v in op.items()),
                'MLP',
                '_'.join(f'{k}-{v}' for k, v in mp.items()),
            ]
        )
        # neck = lambda h, v: MLPNeck(h, v, **mp)
        # model = LitFutureModel(neck, **op)
        model = LitMLPFutureModel(**{**mp, **op})
        do_run(name, model)


# for params in MLP_params:
    # name = 'MLP_gpt2_' + '_'.join(str(p) for p in params)
    # neck = lambda h, v: MLPNeck(h, v, *params)
    # model = LitFutureModel(neck, num_warmup_steps=5000)
    # do_run(name, model)

# for params in LSTM_params:
    # name = 'LSTM_gpt2_' + '_'.join(str(p) for p in params)
    # neck = lambda h, v: LSTMNeck(h, v, *params)
    # model = LitFutureModel(neck, num_warmup_steps=5000)
    # do_run(name, model)

