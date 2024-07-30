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
from datetime import datetime, timedelta
import sys, yaml
from scripts.utils import *

# Ampere and above; use tensor cores
if torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

FAST_DEV_RUN = False
# FAST_DEV_RUN = True
PROJECT = 'LAISR_FUTURE_GPT2'
BATCH_SIZE = 128

dataset = datasets.load_from_disk('/workspace/corpus/msmarco/msmarco_GPT2_64tokens_1m').with_format('torch')
loaders = {
    split: DataLoader(dataset[split], batch_size=BATCH_SIZE)
    for split in ['train', 'val', 'test']
}

fixed_params = {
    'neck_lr': 4e-4,
    # 1000 warmup steps (counted by batch) was found to be good for batch size 128
    'num_warmup_steps': (2_000 * 128) // BATCH_SIZE,
    'freeze_base': True,
    'layer_dims': [],
    'use_next': True,
    'lam': 0,
}


def run():
    with wandb.init() as run:
        params = dict(wandb.config)  # config object not mutable...
        id_ = run_id()
        # top = 'LASSO-SWEEP'
        top = 'LAYER-SWEEP'
        name = '_'.join(
            [
                top,
                id_,
                '_'.join(f'{k}-{v}' for k, v in params.items()),
            ]
        )
        run.name = top + '_' + id_
        model = LitMLPFutureModel(**{**fixed_params, **params})
        print('STARTING RUN', name)
        wandb_logger = WandbLogger(
            name=name,
            project=PROJECT,
            log_model=False,   # Only save checkpoints locally
        )
        lr_callback = LearningRateMonitor()
        checkpoint_callback = ModelCheckpoint(
            dirpath="/workspace/checkpoints/",
            filename=name + "_{epoch:02d}-{val_self_loss:.2f}",
            every_n_epochs=1,
            save_top_k=1,
            monitor='val_self_loss',
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
            max_time=timedelta(hours=2),
            enable_progress_bar=False,
        )
        wandb_logger.watch(model.future_neck, log='all')
        trainer.fit(
            model=model,
            train_dataloaders=loaders['train'],
            val_dataloaders=loaders['val']
        )

def runcatch():
    try:
        run()
    except:
        import traceback
        print(traceback.print_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        sweep_config = yaml.safe_load(f)
    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
    try:
        sweep_id = sys.argv[2]
    except:
        sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    print('SWEEP ID:', sweep_id)
    wandb.agent(sweep_id=sweep_id, function=runcatch, count=1000, project=PROJECT)

