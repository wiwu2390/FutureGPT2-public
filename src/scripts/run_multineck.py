
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
from datetime import datetime, timedelta
import sys, yaml
from scripts.utils import *
from data.utils import *

# Ampere and above; use tensor cores
if torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

FAST_DEV_RUN = False
# FAST_DEV_RUN = True
BATCH_SIZE = 16
# MODEL = 'GPT2'
MODEL = 'MISTRAL'
# MODEL = 'LLAMA2'
PROJECT = f'LAISR_FUTURE_{MODEL}'
NAME = f'{MODEL}-MULTINECK-MLP'
PRECISION = '32'
DATASET = 'msmarco'

params = {
    # MODEL:
    'neck_lr': 4e-4,
    'num_warmup_steps': (1_000 * 128) // BATCH_SIZE,
    'freeze_base': True,
    'lam': 0,
    'base_lr': 1e-5,
    'base_model_name': MODEL_NAME_DICT[MODEL],
    'base_precision': PRECISION,
    'pretrained': True,
    # 'use_adam8bit': True,
}

necks = {
    f'hlayer{i}': lambda h, v, i=i: MLPNeck(h, v, i, [], 0, 0)  # capture the i
    for i in range(33)
}

if __name__ == '__main__':
    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
    id_ = run_id()
    name = '_'.join(
        [
            NAME,
            id_,
            # '_'.join(f'{k}-{v}' for k, v in params.items()),
        ]
    ).replace('/', '-')
    model = LitFutureModel(necks, **params)
    print('STARTING RUN', name)
    wandb_logger = WandbLogger(
        name=name,
        project=PROJECT,
        log_model=False,   # Only save checkpoints locally
    )
    lr_callback = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath="/workspace/checkpoints/",
        filename=name + "_{epoch:02d}-{val_total_loss:.2f}",
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_total_loss',
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor='val_total_loss',
        # divergence_threshold=10,
        min_delta=0.001,
        # verbose=False,
        verbose=True,
        mode='min',
        patience=3,
    )
    trainer = Trainer(
        fast_dev_run=FAST_DEV_RUN,
        logger=wandb_logger,
        val_check_interval=.1,
        callbacks=[checkpoint_callback, lr_callback, early_stop_callback],
        # max_epochs=3,
        # max_time=timedelta(hours=2),
        max_time=timedelta(hours=6),
        # max_time=timedelta(minutes=30),
        enable_progress_bar=False,
        precision=PRECISION,
        accumulate_grad_batches=128//BATCH_SIZE,  # Simulate batches of size 128
    )
    loaders = get_loader(DATASET, MODEL, BATCH_SIZE)
    trainer.fit(
        model=model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val'],
    )
    wandb.finish()
