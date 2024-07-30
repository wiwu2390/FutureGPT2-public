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

class LitBigramModel(L.LightningModule):
    def __init__(
        self,
        base_model_name='gpt2',
    ):
        '''
        Bigram model.
        The key observation here is that a linear vocab_size x vocab_size model
        with linear loss trained with gradient descent will implement a bigram model.
        '''
        super().__init__()
        self.tokenizer = data.utils.get_tokenizer(base_model_name)
        config = utils.get_config(base_model_name)
        self.vocab_size = config.vocab_size
        self.bigram = nn.Embedding.from_pretrained(
            torch.zeros((self.vocab_size, self.vocab_size)),
            freeze=False,
            sparse=True
        )
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # returns frequencies
        return self.bigram(batch['input_ids'])

    def _compute_loss(self, batch):
        freqs = self(batch)
        # [batch_size, seq_length, vocab_size]
        shift_freqs = freqs[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()
        loss = -shift_freqs.gather(-1, shift_labels.unsqueeze(-1)).sum()
        return loss

    @staticmethod
    def _freqs_to_logprobs(freqs, eps=1e-6):
        # two sources of numerical issues:
        #   zero denominator when normalizing to get probs (fix with nan_to_num)
        #   log of zero (fix by adding eps)
        return torch.log((freqs / freqs.sum(dim=-1).unsqueeze(-1)).nan_to_num(0) + eps)

    def _cross_entropy_loss(self, batch):
        freqs = self(batch)
        eps = 1e-6
        logits = self._freqs_to_logprobs(freqs)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        if loss.isnan().any():
            import pdb; pdb.set_trace()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ce_loss = self._cross_entropy_loss(batch)
        self.log('train_loss', ce_loss, on_step=True, on_epoch=True)
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
            freqs = self(tokenized)
            t1 = self.tokenizer.get_vocab()[y1]
            t2 = self.tokenizer.get_vocab()[y2]
            logprobs = self._freqs_to_logprobs(freqs)
            self.log(f'{prefix}{name}_loss_1', -logprobs[0,-2,t1])
            self.log(f'{prefix}{name}_loss_2', -logprobs[0,-1,t2])

    def validation_step(self, batch, batch_idx):
        loss = self._cross_entropy_loss(batch)
        # default for val is on_step=False, on_epoch=True
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._cross_entropy_loss(batch)
        # default for test is on_step=False, on_epoch=True
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1)
        return optimizer

class LitPseudoBigram(L.LightningModule):
    def __init__(self, model_name, lr=1e-4):
        '''
        Borrows a base model's un/embedding and trains a linear model in the embed dim.
        '''
        super().__init__()
        self.model_name = model_name
        model = utils.get_model(model_name, precision='32')
        if model_name == 'gpt2':
            self.embed = model.transformer.wte
        else:
            self.embed = model.model.embed_tokens
        self.unembed = model.lm_head
        for param in self.embed.parameters():
            param.requires_grad=False
        for param in self.unembed.parameters():
            param.requires_grad=False
        self.save_hyperparameters()
        self.linear = nn.Linear(
            self.embed.embedding_dim,
            self.unembed.in_features
        )
        self.lr=lr

    def forward(self, batch):
        return self.unembed(self.linear(self.embed(batch['input_ids'])))

    def _compute_loss(self, batch):
        out = self.forward(batch)
        return nn.CrossEntropyLoss()(
            out.transpose(1, 2)[:,:,:-1],
            batch['input_ids'][:,1:],
        )
    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch)

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        print('val loss', loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self._compute_loss(batch)

    def configure_optimizers(self):
        return optim.Adam(params=self.linear.parameters())
