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


# Ampere and above; use tensor cores
if torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

FAST_DEV_RUN = False
BATCH_SIZE = 256

def kv2d(keys, value_list):
    return [
        {k: v for k, v in zip(keys, values)}
        for values in value_list
    ]

keys = ['kappa', 'reverse_kl']
values = (
    (1, False, 4e-5),
    (1, True, 4e-5),
)

params = kv2d(keys, values)

wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)

dataset = datasets.load_from_disk('/workspace/corpus/msmarco/msmarco_GPT2_64tokens_1m').with_format('torch')
loaders = {
    split: DataLoader(dataset[split], batch_size=BATCH_SIZE)
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
        dirpath="/workspace/checkpoints/",
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
        monitor='val_total_loss',
        divergence_threshold=15,
        min_delta=0.001,
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
        # max_time=timedelta(hours=2),
        enable_progress_bar=False,
    )
    wandb_logger.watch(model.future_neck, log='all')
    trainer.fit(
        model=model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val']
    )
    wandb.finish()

fixed_params = {
    'depth': 1,
    'layer_dims': [],
    'use_next': True,
    'neck_lr': 4e-4,
    # 1000 warmup steps (counted by batch) was found to be good for batch size 128
    'num_warmup_steps': (1_000 * 128) // BATCH_SIZE,
}
print('NUM WARMUP STEPS', (1_000 * 128) // BATCH_SIZE)

for p in params:
    name = '_'.join([
        'FINETUNE',
        run_id(),
        '_'.join(f'{k}-{v}' for k, v in p.items()),
    ])
    model = LitMLPFutureModel(**{**p, **fixed_params}, freeze_base=False)
    do_run(name, model)

