from torch import optim, nn, Tensor
from torch.nn import functional as F
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import transformers
import lightning as L
from inspect import getargspec
import copy
from models import utils
from evals.utils import EXAMPLES
import data.utils

# TODO: Don't use this
# There's less code reuse to just provide the future model a "neck" that ignores the hidden state

class LitMLPModel(L.LightningModule):
    def __init__(
        self,
        base_model_name='gpt2',
        vocab_size=None,
        layer_dims=[],
        act=nn.GELU(),
        freeze_base=True,
        lr=2e-3,
        lr_scheduler_name='constant_with_warmup',
        num_warmup_steps=10_000,
        num_restarts=0,
    ):
        '''
        Basic fully-connected architecture attached to base_model's existing encoder/decoders.
        (base_model itself is not used, except for its encoder/decoders)

        If base_model_name is None, then don't use any pre-trained encoder/decoders.
        I.e., input and output size = vocab_size.
        '''
        super().__init__()
        args = vars()
        for param in getargspec(args['self'].__init__).args[1:]:
            setattr(args['self'], param, args[param])
        base_config = utils.get_config(base_model_name)
        self.tokenizer = data.utils.get_tokenizer(base_model_name)
        base_model = utils.get_model(base_model_name, base_config)
        self.input_embeddings = base_model.get_input_embeddings()
        self.output_embeddings = base_model.get_output_embeddings()
        if freeze_base:
            for param in self.input_embeddings.parameters():
                param.requires_grad = False
            for param in self.output_embeddings.parameters():
                param.requires_grad = False

        d0 = base_config.n_embd
        self.layers = nn.ModuleList()
        for d1 in layer_dims:
            self.layers.append(nn.Linear(d0, d1, bias=True))
            d0 = d1
        self.layers.append(nn.Linear(d0, base_config.n_embd, bias=True))
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch):
        # returns just the logits
        embd = self.input_embeddings(batch['input_ids'])
        for layer in self.layers[:-1]:
            embd = self.act(layer(embd))
        embd = self.layers[-1](embd)
        return self.output_embeddings(embd)

    def _compute_loss(self, batch):
        logits = self(batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()
        return self.loss_func(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        if self.trainer.global_step % 50 == 0:
            self.eval_examples()
        return loss

    def eval_examples(self, prefix=''):
        for name, prompt, y1, y2 in EXAMPLES:
            # prompt with y1 added. have to be careful with whitespace chars.
            prompt += y1.replace("Ġ", " ").replace("Ċ", "\n")
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            ).to(self.device)
            logits = self(tokenized)
            t1 = self.tokenizer.get_vocab()[y1]
            t2 = self.tokenizer.get_vocab()[y2]
            logprobs = torch.log_softmax(logits, dim=-1)
            self.log(f'{prefix}{name}_loss_1', -logprobs[0,-2,t1])
            self.log(f'{prefix}{name}_loss_2', -logprobs[0,-1,t2])

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        # default for val is on_step=False, on_epoch=True
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        # default for test is on_step=False, on_epoch=True
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        # HF unified get_scheduler doesn't do num_restarts...  :(
        if self.lr_scheduler_name == 'cosine':
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_cycles=num_restarts,
            )
        else:
            scheduler = transformers.get_scheduler(
                name=self.lr_scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        # HF's schedulers are on 'step' interval (I think)
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}]
        )

