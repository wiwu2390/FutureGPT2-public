import numpy as np
from torch import optim, nn, Tensor
from torch.nn import functional as F
import torch
import wandb
from transformers import GPT2Config, GPT2LMHeadModel
import transformers
import lightning as L
from inspect import signature, _ParameterKind
import copy
from models import utils
from models.utils import dotdict, maybe_profile, size_mb, print_mem
from models.necks import *
from evals.utils import *
import data.utils
import torchvision.models as models
from itertools import chain
import bitsandbytes as bnb
import warnings

# TODO: Might be cleaner to have a FutureModel extend GPT2LMHeadModel
# (as in the original code.) And then wrap it in LitFutureModel.
# OTOH, the only useful method the GPT2LMHeadModel has anyways is forward??
class LitFutureModel(L.LightningModule):
    def __init__(
        self,
        # Model configs
        future_necks,
        base_model_name='gpt2',
        base_ckpt=None,
        pretrained=True,
        no_drop=True,
        base_precision='32',
        use_adam8bit=False,
        detach_attn=False,
        lora_rank=None,
        # Training configs
        freeze_base=True,
        k_self=1.,  # weight for neck self loss
        k_future=0., # weight for neck future loss
        k_kl=1., # weight for base/orig kl divergence (this doesn't do anything when freeze_base=True b/c kl loss is set to zero)
        k_base=0., # weight for base loss
        lam=0.,  # weight for neck regularization loss
        reverse_kl=False,
        base_lr=2e-5,
        neck_lr=4e-4,
        lr_scheduler_name='constant_with_warmup',
        num_warmup_steps=10_000,
        num_restarts=0,
        do_evals=False,
    ):
        '''
        future_necks: dict of name: nn.Module with constructor taking hidden_size, vocab_size
            (or a partial function that fakes this...)
        reverse_kl: if true, trains on kl(orig, base) [where base is the one being fine-tuned].
            Else, trains on kl(base, orig).
        '''
        # TODO: for now, can only pass in base_model and lr_scheduler by their
        # names, and then use HF's functions. might need to make more general in
        # the future?
        super().__init__()
        # All constructor args -> hyperparameters
        # Not sure this is good practice...
        args = vars()
        for param in list(signature(LitFutureModel.__init__).parameters)[1:]:
            setattr(self, param, args[param])
        # Should only save_hyperparameters in the most specific class
        # See https://github.com/Lightning-AI/lightning/issues/17889
        if self.__class__ is LitFutureModel:
            self.save_hyperparameters(ignore=['base_model', 'future_necks'])
        base_config = utils.get_config(base_model_name)
        self.tokenizer = data.utils.get_tokenizer(base_model_name)
        if self.no_drop:
            # disable dropout for smoother KL penalty
            # maybe sufficient to just do base_model.eval()?
            base_config.resid_pdrop = 0.0
            base_config.embd_pdrop = 0.0
            base_config.attn_pdrop = 0.0

        if base_ckpt:
            print('LOADING FROM CKPT', base_ckpt)
            self.base_model = LitFutureModelWithNeck.load_from_checkpoint(base_ckpt, strict=False).base_model
        else:
            self.base_model = utils.get_model(
                base_model_name, base_config, precision=base_precision,
                pretrained=pretrained,
                detach_attn=detach_attn, lora_rank=lora_rank,
            )
        # print_mem('A')
        warnings.warn(f'BASE SIZE: {size_mb(self.base_model)} MB')

        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif self.k_kl > 0:
            # When not freeze_base, need two copies of the base model:
            # one to fine-tune (self.base_model) and one to compare against (self.orig_model)
            # self.orig_model = utils.get_model(base_model_name, base_config, pretrained=pretrained)
            self.orig_model = copy.deep_copy(self.base_model)
            warnings.warn(f'ORIG SIZE: {size_mb(self.orig_model)} MB')
            for param in self.orig_model.parameters():
                param.requires_grad = False
            assert len(future_necks) == 1, 'Multi-neck supported only for frozen base.'

        try:
            hidden_size = base_config.hidden_size
        except AttributeError:
            # for GPT-2
            hidden_size = base_config.n_embd
        vocab_size = base_config.vocab_size
        self.future_necks = nn.ModuleDict({k: v(hidden_size, vocab_size) for k, v in future_necks.items()})
        if 'default' in self.future_necks:
            self.future_neck = self.future_necks['default'] # for backwards compatibility with old checkpoints
        warnings.warn(f'NECKS SIZE: {size_mb(self.future_necks)} MB')
        self.loss_func = nn.CrossEntropyLoss()

    def _log_prof(self, prof, name, batch_size):
        prof_metrics = ['cpu_time_total', 'cuda_time_total', 'flops', 'self_cpu_time_total', 'self_cuda_time_total']
        if prof:
            for k in prof_metrics:
                kr = k.replace('self', 'profiler').replace('_total', '')
                self.log(f'{name}_{kr}', sum([getattr(x, k) for x in prof.key_averages()]) / batch_size)

    def forward(self, batch, do_profile=False):
        # print_mem('B')
        with maybe_profile(do_profile) as prof:
            base_output = self.base_model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids'],
                output_hidden_states=True
            )
        # print_mem('C')
        batch_size = batch['input_ids'].size(-1)
        self._log_prof(prof, 'base', batch_size)
        # each hidden_state has shape (batch_size, seq_length, hidden_size)
        output = base_output
        hidden_states = base_output.hidden_states
        with maybe_profile(do_profile) as prof:
            # Do the shifting/padding inside the future neck
            tokens = self.base_model.get_input_embeddings()(batch['input_ids'])
        # print_mem('D')
        self._log_prof(prof, 'in_embed', batch_size)
        with maybe_profile(do_profile) as prof:
            future_outs = {k: v(hidden_states, tokens) for k, v in self.future_necks.items()}
        # print_mem('E')
        self._log_prof(prof, 'neck', batch_size)
        with maybe_profile(do_profile) as prof:
            output['future_logits_dict'] = {
                k: self.base_model.get_output_embeddings()(v) for k, v in future_outs.items()
            }
            if 'default' in self.future_necks:
                output['future_logits'] = output['future_logits_dict']['default']
        # print_mem('F')
        self._log_prof(prof, 'out_embed', batch_size)
        return output

    def orig_forward(self, batch):
        return self.orig_model.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids'],
            output_hidden_states=False
        )

    def _compute_loss(self, batch, do_profile=False):
        '''
        Returns dict of:
            base_loss: D_KL(base, orig) (only if freeze_base=self)
            base_loss_reverse: D_KL(orig, base) (only if freeze_base=self)
            self_loss: cross entropy between n+1 base output and n neck output
            future_loss: cross entropy between n neck output and n+2 label
            total_loss: base_loss + kappa * self_loss (this is the one used for training)
        '''
        loss = dotdict()
        output = self(batch, do_profile=do_profile)
        base_logits = output.logits
        loss.base_loss = output.loss

        # we want future_logits to move toward base_logits, so no grad on the latter
        with torch.no_grad():
            # discard first token and reindex. shape [batch_size, seq_length-1, vocab_size]
            shift_base_logits = base_logits[..., 1:, :].contiguous()
            shift_base_probs = torch.softmax(shift_base_logits, dim=2)

        # entropy == (cross entropy with self)
        loss.base_entropy = self.loss_func(
            shift_base_logits.view(-1, shift_base_logits.size(-1)),
            shift_base_probs.view(-1, shift_base_probs.size(-1))
        )
        for neck_name, future_logits in output['future_logits_dict'].items():
            loss[f'{neck_name}_reg_loss'] = self.future_necks[neck_name].reg_loss()

            # discard last token and reindex. shape [batch_size, seq_length-1, vocab_size]
            shift_future_logits = future_logits[..., :-1, :].contiguous()

            # Careful! nn.CrossEntropyLoss takes x as logits (or logprobs) but y as probs.
            # Also, it expects shape [batch_size, vocab_size, ...]
            shift_future_probs = torch.softmax(shift_future_logits, dim=2)
            shift_future_logprobs = torch.log_softmax(shift_future_logits, dim=2)
            loss[f'{neck_name}_self_loss'] = self.loss_func(
                shift_future_logits.view(-1, shift_future_logits.size(-1)),
                shift_base_probs.view(-1, shift_base_probs.size(-1)),
            )
            for k in [1, 5, 10]:
                loss[f'{neck_name}_self_precision_{k}'] = precision(shift_future_logits, shift_base_logits, k=k)
            loss[f'{neck_name}_surprisal'] = surprisal(shift_future_logits, shift_future_logprobs)

            loss[f'{neck_name}_future_entropy'] = self.loss_func(
                shift_future_logits.view(-1, shift_future_logits.size(-1)),
                shift_future_probs.view(-1, shift_future_probs.size(-1))
            )

            # Re-shift for future loss (n+2)
            shift_future_logits = future_logits[..., :-2, :].contiguous()
            shift_labels = batch['input_ids'][..., 2:].contiguous()
            loss[f'{neck_name}_future_loss'] = self.loss_func(
                shift_future_logits.view(-1, shift_future_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not self.freeze_base and self.k_kl > 0:
            orig_output = self.orig_forward(batch)
            loss.orig_loss = orig_output.loss
            orig_logits = orig_output.logits
            orig_probs = torch.softmax(orig_logits, dim=-1)
            orig_logprobs = torch.log_softmax(orig_logits, dim=-1)
            base_logprobs = torch.log_softmax(base_logits, dim=-1)
            kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
            # KLDivLoss's order of arguments are backwards from convention??
            # kl_div(p, q) = \sum_i q_i \log(q_i / p_i) = D_{KL}(q || p)
            # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
            loss.kl_base_orig = kl_div(orig_logprobs, base_logprobs)
            loss.kl_orig_base = kl_div(base_logprobs, orig_logprobs)
            loss.base_orig_surprisal = surprisal(base_logits, orig_logprobs)
            for k in [1, 5, 10]:
                loss[f'base_orig_precision_{k}'] = precision(base_logits, orig_logits, k=k)
            loss.orig_entropy = self.loss_func(
                orig_logits.view(-1, orig_output.logits.size(-1)),
                orig_probs.view(-1, orig_output.logits.size(-1)),
            )

            # shifting probaobly isn't necessary here...
            # shift_orig_logits = orig_output.logits[..., 1:, :].contiguous()
            # shift_orig_probs = torch.softmax(shift_orig_logits, dim=2)
            # loss.orig_entropy = self.loss_func(
                # shift_orig_logits.view(-1, shift_orig_logits.size(-1)),
                # shift_orig_probs.view(-1, shift_orig_probs.size(-1))
            # )
        else:
            loss.orig_loss = 0
            loss.kl_base_orig = 0
            loss.kl_orig_base = 0

        # Note that the default reduction is "mean"
        # This should be fine when further reducing over steps for e.g. epoch loss
        # Our (num tokens)/batch is constant so mean of means is mean.
        # (Because our dataset has the same number of tokens per sequence)
        loss.total_loss = (
            self.k_self * sum(loss[f'{k}_self_loss'] for k in self.future_necks) +
            self.k_future * sum(loss[f'{k}_future_loss'] for k in self.future_necks) +
            self.k_base * loss['base_loss'] +
            self.k_kl * (
                loss.kl_orig_base if self.reverse_kl else loss.kl_base_orig
            ) +
            self.lam * sum(loss[f'{k}_reg_loss'] for k in self.future_necks)
        )
        # Declutter for logging
        for k in list(loss.keys()):
            if k[:8] == 'default_':
                loss[k[8:]] = loss.pop(k)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        for k, v in loss.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True)
        self.log('global_step', self.trainer.global_step)
        # Want this on same freq as "on_step" logs (50 by default)
        if self.trainer.global_step % 50 == 0 and self.do_evals:
            self.eval_examples()
        return loss.total_loss

    def eval_examples(self, prefix=''):
        # Unfortunately this is a bit repetitive w/r/t _compute_loss
        for name, prompt in EXAMPLES:
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            ).to(self.device)
            tokenized_m1 = {k: v[...,:-1] for k, v in tokenized.items()}
            output = self(tokenized_m1)
            t1 = tokenized.input_ids[0,-2]
            t2 = tokenized.input_ids[0,-1]
            # shape is [batch_size, seq_length, vocab_size]
            base_logprobs = torch.log_softmax(output['logits'], dim=-1)
            base_probs = torch.softmax(output['logits'], dim=-1)
            for k in self.future_necks:
                future_logits = output[f'future_logits_dict'][k]
                future_logprobs = torch.log_softmax(future_logits, dim=-1)
                if not self.freeze_base:
                    orig_output = self.orig_forward(tokenized)
                    orig_logprobs = torch.log_softmax(orig_output['logits'], dim=-1)
                    self.log(f'{k}_{prefix}{name}_orig_loss_1', -orig_logprobs[0,-2,t1])
                    self.log(f'{k}_{prefix}{name}_orig_loss_2', -orig_logprobs[0,-1,t2])
                self.log(f'{k}_{prefix}{name}_future_loss_label', -future_logprobs[0,-2,t2])
                self.log(
                    f'{k}_{prefix}{name}_future_loss_self',
                    self.loss_func(future_logits[:,-2,:], base_probs[:,-1,:])
                )

            self.log(f'{prefix}{name}_base_loss_1', -base_logprobs[0,-2,t1])
            self.log(f'{prefix}{name}_base_loss_2', -base_logprobs[0,-1,t2])

    def _log_wgt_comp(self, prefix='', verbose=False):
        wandb_logger = self.logger.experiment
        if not self.freeze_base and self.k_kl > 0:
            wgt_comp = compare_weights(self.base_model, self.orig_model)
            wgt_diffs = list([v for v in wgt_comp.values() if np.isfinite(v)])
            table = wandb.Table(data=[[v] for v in wgt_diffs], columns=['wgt_diff'])
            wandb_logger.log({f'{prefix}wgt_comp': wandb.plot.histogram(table, 'wgt_diff')})
            self.log(f'{prefix}avg_wgt_diff', sum(wgt_diffs) / len(wgt_diffs))
            if verbose:
                for k, v in wgt_comp.items():
                    self.log(f'{prefix}wgt_comp_{k}', v)

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, do_profile=True)
        # default for val is on_step=False, on_epoch=True
        for k, v in loss.items():
            self.log(f'val_{k}', v)
        self._log_wgt_comp('val_')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, do_profile=True)
        # default for val is on_step=False, on_epoch=True
        for k, v in loss.items():
            self.log(f'test_{k}', v)
        self._log_wgt_comp('test_')
        return loss

    def configure_optimizers(self):
        optimizer_params = [{
                'params': self.future_necks.parameters(),
                'lr': self.neck_lr,
        }]
        if not self.freeze_base:
            optimizer_params.append({
                'params': self.base_model.parameters(),
                'lr': self.base_lr,
            })
        if self.use_adam8bit:
            optimizer = bnb.optim.Adam8bit(optimizer_params)
        else:
            optimizer = optim.Adam(optimizer_params)
        # HF unified get_scheduler doesn't do num_restarts...  :(
        if self.lr_scheduler_name == 'cosine':
            scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_cycles=self.num_restarts,
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

    def on_save_checkpoint(self, checkpoint):
        '''
        Remove orig_model (and base_model if freeze_base) from checkpoint to save space.
        NOTE: The resulting ckpt requires load_from_checkpoint(..., strict=False); o/w complains about missing keys.
        '''
        for k in list(checkpoint['state_dict'].keys()):
            if 'base_model' in k and self.freeze_base:
                checkpoint['state_dict'].pop(k)
            if 'orig_model' in k:
                checkpoint['state_dict'].pop(k)

class LitFutureModelWithNeck(LitFutureModel):
    '''
    Wrapper around LitFutureModel to handle neck construction
    '''
    def __init__(self, neck_cls, neck_ckpt=None, **kwargs):
        '''
        Either init a new neck with params from neck_params
        or use the future_neck from the LitFutureModel ckpt stored at neck_ckpt
        '''
        if isinstance(neck_cls, type):
            self.neck_cls = neck_cls
        else:
            self.neck_cls = {
                'mlp': MLPNeck,
                'lstm': LSTMNeck,
            }[neck_cls]
        neck_params = signature(self.neck_cls).parameters
        model_params = signature(super().__init__).parameters
        assert not any(
            p in [_ParameterKind.VAR_KEYWORD, _ParameterKind.VAR_POSITIONAL]
            for p in chain(neck_params.values(), model_params.values())
        ), \
            'In order to detect which params go to Neck vs FutureModel, these two cannot have any variadic args'
        self.neck_ckpt = neck_ckpt
        neck_params = {
            k: v for k, v in kwargs.items()
            if k in neck_params
        }
        model_params = {
            k: v for k, v in kwargs.items()
            if k in model_params
        }
        for k in kwargs.keys():
            if k not in neck_params and k not in model_params:
                warnings.warn(f'WARN: {k} present in config but not used in either neck or model!')
        for k, v in neck_params.items():
            # saving model_params to self is done already in super() constructor
            setattr(self, k, v)
        if neck_ckpt is None:
            assert neck_params is not None
            neck = lambda h, v: self.neck_cls(h, v, **neck_params)
        else:
            model = LitMLPFutureModel.load_from_checkpoint(neck_ckpt)
            neck = lambda h, v: model.future_neck
        super().__init__({'default': neck}, **model_params)
        self.save_hyperparameters(ignore=['base_model', 'future_neck'])

def foo():
    print('hi!')
