import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from torch import profiler as prf
from contextlib import contextmanager
from torch import nn

class dotdict(dict):
    """
    A dictionary that supports dot notation
    as well as dictionary access notation
    for getting and setting items.
    """
    def __init__(self, *args, **kwargs):
        super(dotdict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(dotdict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(dotdict, self).__delitem__(key)
        del self.__dict__[key]

class DetachLayer(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        # this assumes output of module is tuple and it's fine to just detach the tensors
        return (x.detach() if type(x) == torch.Tensor else x for x in out) 

def get_config(model_name):
    return AutoConfig.from_pretrained(model_name)

def get_model(model_name, config=None, precision=None, pretrained=True, detach_attn=False, lora_rank=None):
    if not isinstance(precision, torch.dtype):
        precision = {
            '32': torch.float32,
            'f32': torch.float32,
            'bf16': torch.bfloat16,
            '16': torch.float16,
            'f16': torch.float16,
            None: 'auto',
        }[precision]
    if config is None:
        config = get_config(model_name)
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=precision)
    else:
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=precision)
        print('USING RANDOM INITIALIZATION')
    if detach_attn:
        # Only tried this for Mistral-7B so far. Probably needs modifications for other architectures.
        for layer in model.model.layers:
            layer.self_attn = DetachLayer(layer.self_attn)
        print('DETACHING ATTENTION GRADIENTS')
    if lora_rank:
        from peft import get_peft_model, LoraConfig
        peft_config = LoraConfig(
            # Since our rank is const everywhere, should be fine to just keep lora_alpha/r=1 and tune this thru learning rate?
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias='all',
        )
        model = get_peft_model(model, peft_config)
        print('USING LORA')
        model.print_trainable_parameters()
    return model

@contextmanager
def maybe_profile(do_profile):
    if do_profile:
        with prf.profile(
            activities=[prf.ProfilerActivity.CPU, prf.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with prf.record_function("model_inference"):
                yield prof
    else:
        yield None

def size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def print_mem(name=''):
    print('--------------------------------------------------------------')
    print(name)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print('--------------------------------------------------------------')

