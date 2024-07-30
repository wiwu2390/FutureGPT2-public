
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

# TODO: there're way too many scripts/*.py. This one alone covers most use cases; can configure details in config/*.yaml.

# Ampere and above; use tensor cores
if torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

FAST_DEV_RUN = False
# FAST_DEV_RUN = True
BATCH_SIZE = 128
# MODEL = 'GPT2'
MODEL = 'MISTRAL'
# MODEL = 'LLAMA2'
PROJECT = f'LAISR_FUTURE_{MODEL}'
NAME = f'{MODEL}-BIGRAM'
PRECISION = '32'

# the only reason to put params in fixed_params instead of the yaml config is to prevent them from appearing in the filename
# but maybe it's cleaner to just not do this and live with very long filenames...
fixed_params = {
    # MODEL:
    'neck_lr': 4e-4,
    'num_warmup_steps': (1_000 * 128) // BATCH_SIZE,
    'freeze_base': True,
    # 'freeze_base': False,
    'lam': 0,
    'base_lr': 5e-5,  # 1e-5 for GPT2?
    'base_model_name': MODEL_NAME_DICT[MODEL],
    'base_precision': PRECISION,
    'pretrained': True,
    'dataset_name': 'msmarco',
    # 'use_adam8bit': True,
    # MLP:
    'layer_dims': [],
    # LSTM:
    # 'neck_size': 768,
    'neck_size': 4096,
    'num_layers': 1,
    'add_linear': True,
    'gru': False,
}


def run():
    with wandb.init() as run:
        params = split_config(wandb.config)
        id_ = run_id()
        name = '_'.join(
            [
                NAME,
                id_,
                '_'.join(f'{k}-{v}' for k, v in params.items()),
            ]
        ).replace('/', '-')
        run.name = NAME + '_' + id_
        wandb.config.update(fixed_params)  # so that fixed_params also show up on wandb
        params = {**fixed_params, **params} # note that this is done after name construction (o/w ckpt filename is too long)
        model = LitFutureModelWithNeck(**params)
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
            # max_time=timedelta(hours=2),
            max_time=timedelta(hours=3),
            enable_progress_bar=False,
            precision=PRECISION,
        )
        wandb_logger.watch(model.future_neck, log='all')
        loaders = get_loader(params['dataset_name'], MODEL, BATCH_SIZE)
        trainer.fit(
            model=model,
            train_dataloaders=loaders['train'],
            val_dataloaders=loaders['val'],
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

