import transformers
import torch

import os
import pathlib

import beir
import beir.datasets.data_loader
import datasets
import tqdm
from typing import Tuple, Dict

from data.utils import *
import random

# MODEL_NAME_SHORT = 'LLAMA2'
MODEL_NAME_SHORT = 'GPT2'
MODEL_NAME = MODEL_NAME_DICT[MODEL_NAME_SHORT]
SAVE_PATH = '/workspace/corpus/msmarco/'
BEIR_URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'
# MAX_LENGTH = 64
MAX_LENGTH = 1024
SPLIT_RATIO = {
    'train': 0.9,
    'val': 0.01,
    'test': 0.09,
}
SUBSETS = {
    'full': -1,
    '1m': 1_000_000,
    '100k': 100_000,
}

def load_msmarco(zip_path=os.getcwd()) -> Tuple[datasets.Dataset, datasets.Dataset, Dict[str, Dict[str, int]], Dict]:
    """Loads a BEIR test dataset through tools provided by BeIR.
    Returns:
        corpus (datasets.Dataset): Corpus of documents
            keys -- corpus_id, text
        queries (datasets.Dataset):  Corpus of queries
            keys -- query_id, text
        qrels
    """
    dataset = 'msmarco'
    split = 'train'
    #### Download msmarco.zip dataset and unzip the dataset
    url = BEIR_URL.format(dataset)
    #out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets')
    data_path = download_url_and_unzip(url, zip_path)
    corpus, queries, qrels = beir.datasets.data_loader.GenericDataLoader(data_path).load(split=split)
    corpus = datasets.Dataset.from_list(
        [{'id': k, 'text': v['text']} for k,v in corpus.items()]
        )
    return corpus, queries, qrels

if __name__ == '__main__':
    print('LOAD')
    corpus, _, _ = load_msmarco(SAVE_PATH)
    print('SHUFFLE')
    corpus = corpus.shuffle(seed=1729)

    print('MSMARCO SAMPLE:')
    for n in range(5):
       print(corpus[n]['text'])
       print()

    tokenizer = get_tokenizer(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # adds two fields to the dictionary: 'input_ids' and 'attention_mask'
    tokenize_text = lambda examples: tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    print('TOKENIZE')
    tokenized_msmarco = corpus.map(tokenize_text, batched=True, batch_size=1024)
    print(f'Tokenized {len(tokenized_msmarco)} texts.')

    print('FILTER')
    # LLAMA2 tokenizer: shorter texts are padded at the *beginning* with tokenizer.pad_token
    filtered_msmarco = tokenized_msmarco.filter(
        lambda x: x['input_ids'][-1] != tokenizer.pad_token_id and x['input_ids'][0] != tokenizer.pad_token_id
    )
    print(f'Truncated to {MAX_LENGTH} tokens and removed shorter texts.')
    print(f'Filtered from {len(tokenized_msmarco)} to {len(filtered_msmarco)} rows.')
    print(filtered_msmarco)

    for subset_name, subset_size in SUBSETS.items():
        if subset_size >= 0:
            subset = filtered_msmarco.select(
                range(min(subset_size, len(filtered_msmarco)))
            )
        else:
            subset = filtered_msmarco
        print(f'SUBSET {subset_name}: {len(subset)} rows')

        # SPLIT_RATIO values are absolute proportions.
        # Thus, at each iteration, split_size / denom is the correct fraction of
        # remainder to take.
        remainder = subset
        denom = 1.
        subset_dict = {}
        for split_name, split_size in SPLIT_RATIO.items():
            # round to avoid floating point errors
            train_size = round(split_size/denom, 3)
            if train_size < 1.:
                train_test = remainder.train_test_split(train_size=train_size)
                split, remainder = train_test['train'], train_test['test']
            else:
                split, remainder = remainder, None

            denom -= split_size
            subset_dict[split_name] = split
            print(f'SPLIT {split_name}: {split_size} of total; {len(split)} rows.')

        subset_dict = datasets.DatasetDict(subset_dict)
        save_path = f'{SAVE_PATH}/msmarco_{MODEL_NAME_SHORT}_{MAX_LENGTH}tokens_{subset_name}'
        print(f'Saving to {save_path}')
        subset_dict.save_to_disk(save_path)

