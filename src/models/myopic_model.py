import numpy as np
from torch import optim, nn, Tensor
from torch.nn import functional as F
import torch
import wandb
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import *
from types import MethodType
import gc
from transformers.pytorch_utils import Conv1D
from torch.cuda.amp import autocast

# modified from transformers.models.gpt_neox.modeling_gpt_neox
def myopic_attn_neox(
    query, key, value, past_key, past_value,
    attention_mask, head_mask,
    # originally from self:
    bias, attention_dropout, init_bias, norm_factor,
    # myopia params:
    offset=0, reverse=False, beta=0,
):
    if beta > 0:
        past_key = (1 - beta) * past_key + beta * key
        past_value = (1 - beta) * past_value + beta * value
    if reverse:
        key, past_key = past_key, key
        value, past_value = past_value, value
    assert offset <= 0, 'The causal mask is upper triangular, so it only makes sense to consider nonpositive diagonal offsets.'

    # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
    # compute causal mask from causal mask buffer
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    # dynamically increase the causal mask with the key length, if needed.
    if key_length > bias.shape[-1]:
        init_bias(key_length, device=key.device)
    causal_mask = bias[:, :, key_length - query_length : key_length, :key_length]

    query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
    key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    past_key = past_key.view(batch_size * num_attention_heads, key_length, attn_head_size)
    with autocast(enabled=False):
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=torch.float32,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query.float(),
            past_key.float().transpose(1, 2),
            beta=1.0,
            alpha=norm_factor,
        )
    # edge case: [:0] is nothing, not everything.
        noffset = offset if offset < 0 else query_length
        attn_scores.diagonal(dim1=1, dim2=2, offset=offset).copy_(norm_factor * (query[...,-offset:,:].float() * key[...,:noffset,:].float()).sum(dim=2))

    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    mask_value = torch.finfo(attn_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_scores = torch.where(causal_mask, attn_scores, mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_scores = attn_scores + attention_mask

    attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    attn_weights = attn_weights.to(value.dtype)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_weights = attention_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, past_value)
    attn_output[...,-offset:,:] += attn_weights.diagonal(dim1=2, dim2=3, offset=offset).unsqueeze(dim=3) * (value - past_value)[...,:noffset,:]
    return attn_output, attn_weights


def myopic_forward_neox(
    self,
    hidden_state,
    attention_mask,
    position_ids,
    head_mask,
    layer_past,
    output_attentions=False,
    offset=0,
    reverse=False,
    none_to_zero=False,
    beta=0,
    **kwargs,
):
    def to_qkv(h):
        qkv = self.query_key_value(h)
        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)
        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        q = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        k = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        v = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        q_rot = q[..., : self.rotary_ndims]
        q_pass = q[..., self.rotary_ndims :]
        k_rot = k[..., : self.rotary_ndims]
        k_pass = k[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = k.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)
        q = torch.cat((q, q_pass), dim=-1)
        k = torch.cat((k, k_pass), dim=-1)
        return q, k, v

    query, key, value = to_qkv(hidden_state)

    # this is pretty sloppy: we detect if layer_past is k/v state or hidden state
    # based on its type (tuple or tensor)
    if isinstance(layer_past, tuple):
        past_key, past_value = layer_past
        if past_key is None:
            if none_to_zero:
                past_key = torch.zeros(key.shape, device=key.device)
            else:
                past_key = torch.clone(key).detach()
        if past_value is None:
            if none_to_zero:
                past_value = torch.zeros(value.shape, device=value.device)
            else:
                past_value = torch.clone(value).detach()
        past_key, past_value = past_key.detach(), past_value.detach()
    else:
        if layer_past is None:
            if none_to_zero:
                layer_past = torch.zeros(hidden_state.shape, device=hidden_state.device)
            else:
                layer_past = torch.clone(hidden_state).detach()
        layer_past = layer_past.detach()
        _, past_key, past_value = to_qkv(layer_past)

    present = (key, value)

    # Compute attention
    attn_output, attn_weights = myopic_attn_neox(
        query, key, value, past_key, past_value,
        attention_mask, head_mask,
        bias=self.bias, attention_dropout=self.attention_dropout,
        norm_factor=self.norm_factor,
        init_bias=self._init_bias,
        offset=offset, reverse=reverse, beta=beta,
    )

    # Reshape outputs
    attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
    attn_output = self.dense(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs

def to_myopic_neox(model, all_layer_past, offset=0, reverse=False, none_to_zero=False, beta=0):
    for i, module in enumerate(model.gpt_neox.layers):
        layer_past = all_layer_past[i]
        def forward(self, *args, **kwargs):
            nonlocal offset
            nonlocal reverse
            nonlocal layer_past
            kwargs.pop('layer_past')
            return myopic_forward_neox(
                self, *args, **kwargs, layer_past=layer_past,
                offset=offset, reverse=reverse, none_to_zero=none_to_zero,
                beta=beta,
            )
        module.attention.forward = MethodType(forward, module.attention)
        module.attention.extra_repr = lambda: 'MYOPIC'
    return model


# modified from transformers.models.mistral.modeling_mistral.MistralAttention.forward
def myopic_forward_mistral(
    self,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_value,
    output_attentions,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    past_key_states = past_key_value.key_cache[self.layer_idx].detach()
    past_value_states = past_key_value.value_cache[self.layer_idx].detach()

    assert key_states.shape == past_key_states.shape, \
        f'past_key_states is wrong shape: {past_key_states.shape} instead of {key_states.shape}'
    assert value_states.shape == past_value_states.shape, \
        f'past_value_states is wrong shape: {past_value_states.shape} instead of {value_states.shape}'

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    past_key_states = repeat_kv(past_key_states, self.num_key_value_groups)
    past_value_states = repeat_kv(past_value_states, self.num_key_value_groups)
    #print('KEY DIFF', torch.norm(key_states - past_key_states).item())
    #print('VAL DIFF', torch.norm(value_states - past_value_states).item())

    # query @ past_key on off-diagonal
    attn_weights = torch.matmul(query_states, past_key_states.transpose(2, 3))
    # query @ key on diagonal
    #print('ATTN DIFF', torch.norm(attn_weights.diagonal(dim1=2, dim2=3)-(query_states * key_states).sum(dim=3)).item())
    attn_weights.diagonal(dim1=2, dim2=3).copy_((query_states * key_states).sum(dim=3))
    attn_weights /= math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # attn @ past_value on off-diagonal
    attn_output = torch.matmul(attn_weights, past_value_states)
    # attn @ value on diagonal
    #print('VAL DIFF', torch.norm(attn_weights.diagonal(dim1=2, dim2=3).unsqueeze(dim=3) * (value_states - past_value_states)).item())
    attn_output += attn_weights.diagonal(dim1=2, dim2=3).unsqueeze(dim=3) * (value_states - past_value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def to_myopic_mistral(model, past_key_value):
    past_key_value = DynamicCache.from_legacy_cache(past_key_value)
    def forward(*args, **kwargs):
        # This is very hacky, but otherwise it's hard to provide past_key_values
        # to myopic_forward without breaking a lot of MistralModel
        nonlocal past_key_value
        kwargs.pop('past_key_value')
        return myopic_forward(*args, **kwargs, past_key_value=past_key_value)
    
    for module in model.modules():
        if type(module) == MistralAttention:
            module.forward = MethodType(forward, module)
            module.extra_repr = lambda: 'MYOPIC'
            gc.collect()
            torch.cuda.empty_cache()
    return model

def myopic_attn_gpt2(
    query, key, value, past_key, past_value, attention_mask, head_mask,
    bias,
    attn_dropout,
    scale_attn_weights=True,
    offset=0,
    reverse=False,
    beta=0,
):
    if beta > 0:
        past_key = (1 - beta) * past_key + beta * key
        past_value = (1 - beta) * past_value + beta * value
    if reverse:
        key, past_key = past_key, key
        value, past_value = past_value, value
    assert offset <= 0, 'The causal mask is upper triangular, so it only makes sense to consider nonpositive diagonal offsets.'
    attn_weights = torch.matmul(query, past_key.transpose(-1, -2))
    # edge case: [:0] is nothing, not everything.
    noffset = offset if offset < 0 else query.shape[-2]
    attn_weights.diagonal(dim1=2, dim2=3, offset=offset).copy_((query[...,-offset:,:] * key[...,:noffset,:]).sum(dim=3))

    if scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = bias[:, :, key_length - query_length : key_length, :key_length]
    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, past_value)
    attn_output[...,-offset:,:] += attn_weights.diagonal(dim1=2, dim2=3, offset=offset).unsqueeze(dim=3) * (value - past_value)[...,:noffset,:]

    return attn_output, attn_weights

def myopic_forward_gpt2(
    self,
    hidden_states,
    layer_past,
    attention_mask=None,
    head_mask=None,
    output_attentions=False,
    offset=0,
    reverse=False,
    none_to_zero=False,
    beta=0,
    **kwargs,
):
    assert kwargs.get('encoder_hidden_states') is None, 'Only decoder is supported'
    
    query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    # this is pretty sloppy: we detect if layer_past is k/v state or hidden state
    # based on its type (tuple or tensor)
    if isinstance(layer_past, tuple):
        past_key, past_value = layer_past
        if past_key is None:
            if none_to_zero:
                past_key = torch.zeros(key.shape, device=key.device)
            else:
                past_key = torch.clone(key).detach()
        if past_value is None:
            if none_to_zero:
                past_value = torch.zeros(value.shape, device=value.device)
            else:
                past_value = torch.clone(value).detach()
        past_key, past_value = past_key.detach(), past_value.detach()
    else:
        if layer_past is None:
            if none_to_zero:
                layer_past = torch.zeros(hidden_states.shape, device=hidden_states.device)
            else:
                layer_past = torch.clone(hidden_states).detach()
        layer_past = layer_past.detach()
        _, past_key, past_value = self.c_attn(layer_past).split(self.split_size, dim=2)
        past_key = self._split_heads(past_key, self.num_heads, self.head_dim)
        past_value = self._split_heads(past_value, self.num_heads, self.head_dim)
    present = (key, value)

    assert not self.reorder_and_upcast_attn, 'Not supported!'
    assert not self.is_cross_attention, 'Not supported!'
    assert not self.scale_attn_by_inverse_layer_idx, 'Not supported!'
    attn_output, attn_weights = myopic_attn_gpt2(
        query, key, value, past_key, past_value, attention_mask, head_mask,
        self.bias, self.attn_dropout,
        scale_attn_weights=self.scale_attn_weights,
        offset=offset, reverse=reverse, beta=beta,
    )

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)

def to_myopic_gpt2(model, past_key_values, offset=0, reverse=False, none_to_zero=False, beta=0):    
    def forward(self, *args, **kwargs):
        nonlocal offset
        nonlocal reverse
        nonlocal past_key_values
        kwargs.pop('layer_past')
        return myopic_forward_gpt2(
            self, *args, **kwargs, layer_past=past_key_values[self.layer_idx], 
            offset=offset, reverse=reverse, none_to_zero=none_to_zero,
            beta=beta,
        )
    for name, module in model.named_modules():
        #if type(module) == GPT2Attention:  # type doesn't match? idk why
        if name.split('.')[-1] == 'attn':
            module.forward = MethodType(forward, module)
            module.extra_repr = lambda: 'MYOPIC'
    return model

def shrink_mlp_mistral(model, mlp_rank=2048, lora_rank=128):
    '''
    Replaces mlp layers with new layer with rank mlp_rank
    and adds lora adapters to attention weights with rank lora_rank
    '''
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    modules = dict(model.named_modules())
    for name, module in modules.items():
        parent, child = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        if 'mlp' in parent and child in ['gate_proj', 'up_proj']:
            setattr(modules[parent], child, nn.Linear(module.in_features, mlp_rank, bias=module.bias is not None))
        if 'mlp' in parent and child in ['down_proj']:
            setattr(modules[parent], child, nn.Linear(mlp_rank, module.out_features, bias=module.bias is not None))
    peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias='none',
        target_modules='.*(q|k|v|o)_proj',
    )
    for layer in model.model.layers:
        get_peft_model(layer.self_attn, peft_config)
        
    numel_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    numel_all = sum(p.numel() for p in model.parameters())
    print(f'SHRINK_MLP: {numel_train} trainable / {numel_all} total ({numel_train/numel_all})')
    return model

def shrink_mlp_gpt2(model, mlp_rank=256, lora_rank=128):
    '''
    Replaces mlp layers with new layer with rank mlp_rank
    and adds lora adapters to attention weights with rank lora_rank
    '''
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    modules = dict(model.named_modules())
    for name, module in modules.items():
        parent, child = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        if 'mlp' in parent and child == 'c_fc':
            setattr(
                modules[parent], 
                child, 
                Conv1D(mlp_rank, module.weight.shape[0])
            )
        if 'mlp' in parent and child in ['c_proj']:
            setattr(
                modules[parent], 
                child, 
                Conv1D(module.weight.shape[1], mlp_rank)
            )
            
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias='none',
        target_modules='.*c_(attn|proj)',
    )
    for layer in model.transformer.h:
        get_peft_model(layer.attn, peft_config)

    model.lm_head.weight.requires_grad = False  # shared with transformer.wte.weight
    model.transformer.wpe.weight.requires_grad = False
    numel_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    numel_all = sum(p.numel() for p in model.parameters())
    print(f'SHRINK_MLP: {numel_train} trainable / {numel_all} total ({numel_train/numel_all})')
    return model

class LitMyopicModel(L.LightningModule):
    def __init__(
        self,
        myopic_model=None,
        orig_model=None,
        layer_past=None,
        #model_name='gpt2',
        #downsample_func=shrink_mlp_gpt2,
        to_myopic=to_myopic_gpt2,
        loss_type='myopic_loss',
        from_kv=False,
        lr=6e-4,
        warmup=0.01,
        decay=0.01,
        deepspeed=False,
    ):
        '''
        Myopic attention training using attention weights from orig_model for all non-self tokens.
        If from_kv, uses kv state from orig_model. Else, uses orig_model's hidden state and own kv weights.
        '''
        super().__init__()
        args = vars()
        for param in list(signature(LitMyopicModel.__init__).parameters)[1:]:
            setattr(self, param, args[param])
        #self.orig_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        #self.myopic_model = downsample_func(copy.deepcopy(self.orig_model))
        #self.to_myopic = to_myopic_gpt2 if self.model_name=='gpt2' else to_myopic_mistral
        if not orig_model:
            assert layer_past is not None
            # print('WARNING: Running MyopicModel with no orig_model!')
        elif self.orig_model is not self.myopic_model:
            for param in self.orig_model.parameters():
                param.requires_grad = False
        self.save_hyperparameters()

    def forward(self, batch):
        if self.orig_model is not None:
            # self.orig_model.eval()
            orig_out = self.orig_model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=not self.from_kv,
                #labels=batch['input_ids'],
                use_cache=True,
            )
            cur_layer_past = orig_out.past_key_values if self.from_kv else orig_out.hidden_states
        else:
            assert self.layer_past is not None
            cur_layer_past = self.layer_past
        self.to_myopic(self.myopic_model, cur_layer_past)
        out = self.myopic_model.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            #labels=batch['input_ids'],
            use_cache=True,
        )
        if self.orig_model:
            for k in orig_out.keys():
                out[f'orig_{k}'] = orig_out[k]
        return out

    def _compute_loss(self, batch):
        loss = dotdict()
        out = self.forward(batch)
        ce = nn.CrossEntropyLoss()
        kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss['myopic_loss'] = ce(
            out.logits[...,:-1,:].contiguous().view(-1, out.logits.shape[-1]),
            batch['input_ids'][...,1:].contiguous().view(-1)
        )
        if self.orig_model:
            loss['orig_loss'] = ce(
                out.orig_logits[...,:-1,:].contiguous().view(-1, out.orig_logits.shape[-1]),
                batch['input_ids'][...,1:].contiguous().view(-1)
            )
            loss['kl_loss'] = kl_div(
                torch.log_softmax(out.logits, dim=-1).view(-1, out.logits.shape[-1]),
                torch.log_softmax(out.orig_logits, dim=-1).view(-1, out.orig_logits.shape[-1])
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        for k, v in loss.items():
            self.log(f'train_{k}', v)
        self.log(f'train_loss', loss[self.loss_type]) # for easier comparison
        self.log('global_step', self.trainer.global_step)
        return loss[self.loss_type]

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        for k, v in loss.items():
            self.log(f'val_{k}', v)
        self.log(f'val_loss', loss[self.loss_type]) # for easier comparison
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        for k, v in loss.items():
            self.log(f'test_{k}', v)
        self.log(f'test_loss', loss[self.loss_type]) # for easier comparison
        return loss
        return loss

    def configure_optimizers(self):
        if self.deepspeed:
            optimizer = DeepSpeedCPUAdam(
                model_params=self.myopic_model.parameters(),
                lr=self.lr,
                weight_decay=self.decay,
            )
        else:
            optimizer = optim.AdamW(
                params=self.myopic_model.parameters(),
                lr=self.lr,
                weight_decay=self.decay,
            )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.warmup * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        print('NUM TRAINING STEPS', self.trainer.estimated_stepping_batches)
        # HF's schedulers are on 'step' interval
        return(
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}]
        )

    def on_save_checkpoint(self, checkpoint):
        # Don't save orig_model
        for k in list(checkpoint['state_dict'].keys()):
            if 'orig_model' in k:
                checkpoint['state_dict'].pop(k)


