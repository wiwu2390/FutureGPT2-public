#!/usr/bin/env python
import wandb
import models.future_model
from lightning.pytorch.loggers import WandbLogger
import os
from lightning import Trainer
import datasets
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from models.necks import *
from models.mlp_model import LitMLPModel
from models.bigram_model import LitBigramModel
from pprint import pprint
from scripts.utils import *

# TODO: Set up argparse
# TODO: move params into a YAML or something

FAST_DEV_RUN = False
BIGRAM = True

optim_keys = ['lr', 'lr_scheduler_name', 'num_warmup_steps', 'num_restarts']
optim_params = (
    # (2e-4, 'constant_with_warmup', 5000, 0),
    (2e-3, 'constant_with_warmup', 5000, 0),
    (2e-3, 'cosine_with_restarts', 500, 0),
    (2e-3, 'cosine_with_restarts', 500, 1),
    # (2e-3, 'cosine_with_restarts', 500, 4),
)
optim_params = [
    {k: v for k, v in zip(optim_keys, p)}
    for p in optim_params
]
print('OPTIM PARAMS')
pprint(optim_params)

# layer_dims
MLP_params = ([], [128], [256], [512])

wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)

dataset = datasets.load_from_disk('/corpus/msmarco/msmarco_GPT2_64tokens_full').with_format('torch')
loaders = {
    split: DataLoader(dataset[split], batch_size=32 if BIGRAM else 128)
    for split in ['train', 'val', 'test']
}

# TODO: don't repeat code from run.py
def do_run(name, model):
    print('STARTING RUN', name)
    wandb_logger = WandbLogger(
        name=name,
        project='LAISR_FUTURE_GPT2',
        log_model=False,
    )
    lr_callback = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath="/checkpoints/",
        # can't zero pad global_step b/c the logger casts it to a float >:(((
        # filename="{name}-{epoch:02d}-{global_step:.0f}-{val_total_loss:.2f}",
        filename=name + "-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        # save_on_train_epoch_end=True,
    )
    trainer = Trainer(
        fast_dev_run=FAST_DEV_RUN,
        logger=wandb_logger,
        val_check_interval=.05,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=1,
        enable_progress_bar=False,
        precision='16-true' if BIGRAM else '32-true'
    )
    # this might be storage-expensive in wandb?
    # wandb_logger.watch(model.layers, log='all')
    trainer.fit(
        model=model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['val']
    )
    wandb.finish()

if BIGRAM:
    model = LitBigramModel()
    do_run(run_id() + '_bigram', model)
else:
    for layer_dims in MLP_params:
        for op in optim_params:
            name = 'MLPBASE_' + '_'.join(str(p) for p in op.values()) + '_' + str(layer_dims)
            model = LitMLPModel('gpt2', layer_dims=layer_dims, **op)
            do_run(name, model)

