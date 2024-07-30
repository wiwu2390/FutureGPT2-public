import transformers
import datasets
import os, sys
from transformers import AutoTokenizer
from typing import Tuple, Dict
import requests
import tqdm
import zipfile
from torch.utils.data import Dataset, DataLoader
import torch

# TOKENS
def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def get_token_dict(tokenizer):
    '''
    Returns reverse dictionary: Token[k] is the k-th token
    '''
    return {
        # Spaces render as Ġ and newline as Ċ.
        v: k.replace("Ġ", " ").replace("Ċ", "\n")
        for k,v in tokenizer.get_vocab().items()
    }

def print_tokens(text, tokenizer):
    # note: the LLAMA tokenizer may insert an initial space to the string.
    Token = get_token_dict(tokenizer)
    for n in tokenizer.encode(text):
        print(Token[n],end=" | ")
    print()

# DOWNLAD DATASET
def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm.tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, 'r')
    zip_.extractall(path=out_dir)
    zip_.close()

def download_url_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split('/')[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        print(f'Downloading {dataset}...')
        download_url(url, zip_file, chunk_size)
    else:
        print(f'Dataset zip already exists at {zip_file}')

    unzipped = zip_file.replace('.zip', '')
    if not os.path.isdir(unzipped):
        print(f'Unzipping {dataset}...')
        unzip(zip_file, out_dir)
    else:
        print(f'Dataset dir already exists at {unzipped}')

    return unzipped

# LOAD DATASET
def dataset_from_path(path):
    return datasets.load_from_disk(path).with_format('torch')

CORPUS_PATH = '/corpus/{dataset_name}/{split}_{dataset_name}_{model_name}_{maxlen}_{size}'
# TODO: not sure if this needs its own method
def load_dataset(
    dataset_name,
    split,
    model_name,
    maxlen='64tokens',
    size='all'
):
    return dataset_from_path(
        CORPUS_PATH.format(
            dataset_name=dataset_name,
            split=split,
            model_name=model_name,
            maxlen=maxlen,
            size=size
        )
    )

VOCAB_SIZE_DICT = {
    'GPT2': 50257,
    'MISTRAL': 32000,
    'LLAMA2': 32000,
}

MODEL_NAME_DICT = {
    'GPT2': 'gpt2',
    'LLAMA2': 'daryl149/llama-2-7b-hf',
    'LLAMA2-CHAT': 'daryl149/llama-2-7b-chat-hf',
    'MISTRAL': 'mistralai/Mistral-7B-v0.1',
    'MISTRAL-INSTR': 'mistralai/Mistral-7B-Instruct-v0.1',
}

class RandDataset(Dataset):
    '''
    Dataset of uniform random token sequences
    '''
    def __init__(self, size, seq_length, vocab_size):
        super().__init__()
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, value):
        return {
            'input_ids': torch.randint(self.vocab_size, (self.seq_length,)),
            'attention_mask': torch.ones((self.seq_length,))
        }

def get_loader(name, model, batch_size=128, shuffle=True):   # todo: argument for dataset size?
    if name == 'rand':
        ds = {
            split: RandDataset(size=size, seq_length=64, vocab_size=VOCAB_SIZE_DICT[model])
            for split, size in [('train', 900_000), ('val', 10_000), ('test', 90_000)]
        }
    elif name == 'msmarco':
        ds = datasets.load_from_disk(f'/workspace/corpus/msmarco/msmarco_{model}_64tokens_1m').with_format('torch')
    else:
        ds = datasets.load_from_disk(f'/workspace/corpus/the_pile/the_pile_500k_64_{model}/{name}').with_format('torch')
    return {
        split: DataLoader(ds[split], batch_size=batch_size, shuffle=shuffle)
        for split in ['train', 'val', 'test']
    }

