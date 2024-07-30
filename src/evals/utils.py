import torch
import pprint

import pprint
import torch
from math import prod
pp = pprint.PrettyPrinter(indent=4)

import data
from data.utils import get_token_dict
from models.utils import dotdict
import pandas as pd

# (name, example string)
# NOTE: we used to explictly seperate out the expected last two tokens
# but that's difficult to maintain across different tokenizers
# TODO: put this in an external YAML or something
EXAMPLES = [
    (
        'oppenheimer',
        'One of the most colorful characters at Los Alamos was J. Robert Oppenheimer',
    ),
    (
        'alphabet',
        'The letters of the alphabet, in order, are: A B C D E F G',
    ),
    (
        'primes',
        'The prime numbers, in order, are: 2 3 5 7 11 13',
    ),
    (
        'house1',
        'Alice lives in the blue house. Bob lives in the red house. Eve lives in the green house',
    ),
    (
        'house2',
        'Alice lives in a house that is blue. Bob lives in a house that is red. Eve lives in a house that is green',
    ),
    (
        'name',
        'Hello, it\'s nice to meet you. My name is Bob',
    ),
    (
        'redbook',
        'This store sells only red books and blue pens. Do you want a red book or a blue pen',
    ),
    (
        'friends',
        'These are my two friends Tom Smith and Bob Jones. Of the two, my favorite is Bob Jones',
    )
]

def describe(x, quantiles=[0.25, 0.5, 0.75]):
    '''
    Descriptive statistics for float tensor x.
    '''
    x = x.type(torch.double)
    stats = {
        'shape': x.shape,
        'num': prod(x.shape),
        'mean': x.mean(),
        'std': x.std(),
        'min': x.min(),
        'max': x.max(),
    }
    for q in quantiles:
        stats[f'quantile_{q:.2f}'] = x.quantile(q)
    return stats

def compare_weights(model, orig_model):
    state_dict_model = model.state_dict()
    state_dict_orig = orig_model.state_dict()
    ratios = {}

    for key in state_dict_orig.keys():
        if key in state_dict_model:
            W = state_dict_orig[key]
            W_prime = state_dict_model[key]
            # largest singular value
            # (different from p='fro' which is entrywise L^2 norm)
            diff_norm = torch.norm(W_prime - W, p=2)
            W_norm = torch.norm(W, p=2)

            if W_norm != 0:
                ratio = diff_norm / W_norm
            else:
                ratio = float('inf')

            ratios[key] = ratio.item()

    return ratios

def generate(prompt, model, tokenizer, max_length=64, do_sample=False, **kwargs):
    '''
    Generate conditioned on str text and return decoded str.
    '''
    Token = get_token_dict(tokenizer)
    out_list = []
    for n in range(max_length):
        tokenized = tokenizer(
            prompt,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)
        # out = model(dotdict({'input_ids': tokenized}))
        out = model(tokenized)
        # some of the base models return a tensor directly
        # others return an output dict. TODO: make this consistent?
        if hasattr(out, 'logits'):
            out = out.logits
        out = out[0][-1]
        next_token_id = out.argmax().item()
        next_token = Token[next_token_id]
        prompt += next_token
        out_list.append(next_token)

    return prompt, out_list

def generate_future(
    prompt,
    model,
    tokenizer,
    max_tokens=10,
    topk=10
):
    '''
    Outputs a Dataframe with columns:
    next_token, next_prob, future_token_1, future_prob_1, ..., future_token_k, future_prob_k
    where topk future_token are ranked by probability.
    '''
    #TODO: use past key values to speed up
    cols = sum([[f'base_token_{i}', f'base_prob_{i}'] for i in range(topk)], [])
    cols += sum([[f'future_token_{i}', f'future_prob_{i}'] for i in range(topk)], [])
    Token = get_token_dict(tokenizer)
    rows = []
    for n in range(max_tokens):
        tokenized = tokenizer(
            prompt,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)
        output = model(tokenized)
        logits = output.logits[0][-1]
        future_logits = output.future_logits[0][-2]
        probs = torch.softmax(logits,dim=-1)
        future_probs = torch.softmax(future_logits,dim=-1)
        next_token_id = logits.argmax().item()
        # future_token_id = future_logits.argmax().item()
        next_token = Token[next_token_id]
        # future_token = Token[future_token_id]
        row = []
        # row += [next_token, probs[next_token_id].item()]
        base_top = logits.topk(k=topk)
        for i in base_top.indices.cpu().numpy():
            row += [Token[i], probs[i].item()]
        future_top = future_logits.topk(k=topk)
        for i in future_top.indices.cpu().numpy():
            row += [Token[i], future_probs[i].item()]
        prompt += next_token
        rows.append(row)

    return pd.DataFrame(rows, columns=cols)

def generate_compare(
    prompt,
    model,
    base_model,
    max_tokens=10,
    topk=10
):
    #TODO: use past key values to speed up
    for n in range(max_tokens):
        print(prompt,end="")
        tokenized = tokenizer(
            prompt,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)
        output = model(tokenized)
        logits = output.logits[0][-1]
        base_logits = base_model(input_ids).logits[0][-1]
        probs = torch.softmax(logits,dim=-1)
        base_probs = torch.softmax(base_logits,dim=-1)
        next_token_id = logits.argmax().item()
        base_token_id = base_logits.argmax().item()
        next_token = Token[next_token_id]
        next_token_prob = 100*probs[next_token_id].item()
        base_next_token_prob = 100*base_probs[next_token_id].item()

        log_probs = logits.log_softmax(dim=-1)
        base_log_probs = base_logits.log_softmax(dim=-1)

        #extremes(log_probs - base_log_probs)

        KL = kl_div(base_log_probs,log_probs,log_target=True,reduction='sum')
        print(f'({next_token})  Model prob = {next_token_prob:.0f}, Base prob = {base_next_token_prob:.0f}, KL = {KL:.6f}')
        print()
        prompt += next_token
    return prompt, log_probs, base_log_probs

def precision(x, y, k=10):
    '''
    average of (1 if highest probabily token of x appears in top-k of y, else 0).
    See Din et al. 2023.
    Excepts shape (batch_size, seq_length, vocab_size)
    Doesn't matter if x and y are logits, probs, logprobs, etc; monotonically invariant.
    '''
    x = x.view(-1, x.size(-1))
    y = y.view(-1, y.size(-1))
    return (x.argmax(dim=1, keepdim=True) == y.topk(dim=1, k=k).indices).sum() / x.shape[0]


def surprisal(x, y):
    '''
    average negative log prob of y at argmax(x).
    see Din et al. 2023.
    Assumes y is already in the form of logprobs
    '''
    x = x.view(-1, x.size(-1))
    y = y.view(-1, y.size(-1))
    return -y.gather(dim=1, index=x.argmax(dim=1, keepdim=True)).sum() / x.shape[0]
