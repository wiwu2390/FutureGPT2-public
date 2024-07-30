import os
os.environ['HF_HOME'] = '/workspace/cache/huggingface/'

import numpy as np
from torch import optim, nn, Tensor
from torch.nn import functional as F
import torch
import wandb
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
from inspect import signature, _ParameterKind
import copy
import gc
import datasets
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
from itertools import islice

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import sys

from models.gpt_model import *
from models.myopic_model import *

if torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

wandb.login(key='os.environ[WANDB_API_KEY]', relogin=True)

# https://github.com/EleutherAI/pythia
LR_DICT = {
    '14m':  1.0e-3,
    '31m':  1.0e-3,
    '70m':  3.0e-4,
    '160m': 3.0e-4,
    '410m': 3.0e-4,
    '1b':   3.0e-4,
    '1.4b': 2.0e-4,
    '2.8b': 1.6e-4,
    '6.9b': 1.2e-4,
    '12b':  1.2e-4,
}

BATCH_DICT= {
    '14m':  512,
    '31m':  512,
    '70m':  512,
    '160m': 512,
    '410m': 256,
    '1b':   128,
    '1.4b': 128,
    '2.8b': 64,
    '6.9b': 64,
    '12b':  32,
}

fp16 = True
model_size = sys.argv[1]
assert sys.argv[2] in ['vanilla', 'myopic']
myopic = sys.argv[2]=='myopic'

params = {
    # 'model_name': f'EleutherAI/pythia-{model_size}-deduped',
    'model_name': f'EleutherAI/pythia-{model_size}', # 14m and 31m don't have deduped versions
    'lr': LR_DICT[model_size] * 0.4,
    'warmup': 0.05,
}
# batch_size = min(512, BATCH_DICT[model_size] * (2 if fp16 else 1))
batch_size = BATCH_DICT[model_size]
batch_accum = 512 // batch_size
if fp16:
    batch_size *= 2

# 5M examples sampled from the-pile. truncated to len 64
# train = load_dataset(
    # 'EleutherAI/pile-deduped-pythia-random-sampled',
    # split='train'
# )
# train = train.rename_column('Tokens', 'input_ids')
# train = train.remove_columns([c for c in train.column_names if c != 'input_ids'])
# train = train.cast_column('input_ids', datasets.Sequence(datasets.Value('int64')))
# train = train.with_format('torch', device='cuda')

train = load_from_disk('/workspace/pile_PYTHIA_64tokens_20M')['train'].select(range(10_000_000))
train = train.cast_column('input_ids', datasets.Sequence(datasets.Value('int64')))
train = train.with_format('torch', device='cuda')
# train = train.map(lambda x: {'input_ids': x['input_ids'][...,:512]})

train_loader = DataLoader(train, batch_size=batch_size)#, num_workers=96), #multiprocessing_context='spawn')

NAME = '_'.join(
    [f'PYTHIA-PILE10M64' + ('-MYOPIC' if myopic else '-VANILLA') + ('-fp16' if fp16 else '')] +
    [f'{k}_{v:.2e}' if isinstance(v, float) else f'{k}_{v}' for k, v in {**params}.items()]
).replace('EleutherAI/', '')
PROJ = 'LAISR_FUTURE_PYTHIA'
wandb_logger = WandbLogger(
    name=NAME,
    project=PROJ,
    log_model=False,   # Only save checkpoints locally
)

lr_monitor = LearningRateMonitor()
checkpoint_callback = ModelCheckpoint(
    dirpath="/workspace/checkpoints",
    filename=NAME + "_{global_step}_{train_loss:.2f}",
    every_n_epochs=1,
    save_top_k=1,
    monitor='train_loss',
    mode='min',
)
early_stop_callback = EarlyStopping(
    monitor='train_loss',
    divergence_threshold=15,
    min_delta=0.00,
    patience=10,
    verbose=False,
    mode='min',
)
trainer = L.Trainer(
    accelerator='gpu',
    fast_dev_run=False,
    logger=wandb_logger,
    #val_check_interval=.1,
    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    max_epochs=1,
    enable_progress_bar=True,
    # strategy="deepspeed_stage_2_offload",
    strategy="deepspeed_stage_2",
    # precision='32',
    precision='16-mixed' if fp16 else '32',
    # devices=1,
    accumulate_grad_batches=batch_accum,
)

if myopic:
    print('RUNNING MYOPIC')
    config = AutoConfig.from_pretrained(params['model_name'])
    config.upcast_attn = True
    myopic_model = AutoModelForCausalLM.from_pretrained(params['model_name'], config=config)
    model = LitMyopicModel(
        myopic_model=myopic_model,
        orig_model=None,    # set to None (default) for cutgrad training [use own detached hidden state or kv]
        loss_type='myopic_loss',
        to_myopic=to_myopic_neox,
        from_kv=False,
        layer_past = [None for _ in range(len(myopic_model.gpt_neox.layers))],
    )
    wandb_logger.watch(model.myopic_model, log='all', log_graph=False)
else:
    print('RUNNING VANILLA')
    model = LitGPTModel(**params, deepspeed=False)
    wandb_logger.watch(model.model, log='all', log_graph=False)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
)
