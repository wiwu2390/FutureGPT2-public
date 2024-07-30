from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import os
from data.utils import get_tokenizer, get_token_dict
import secrets
from itertools import islice

SUBSETS = [
    #'ArXiv',
    #'BookCorpus2',
    #'Books3',
    # 'DM Mathematics',
    'Enron Emails',
    # 'EuroParl',
    'FreeLaw',
    # 'Github',
    # 'Gutenberg (PG-19)',
    # 'HackerNews',
    # 'NIH ExPorter',
    # 'OpenSubtitles',
    # 'OpenWebText2',
    # 'PhilPapers',
    # 'Pile-CC',
    # 'PubMed Abstracts',
    # 'PubMed Central',
    # 'StackExchange',
    # 'UPSTO Backgrounds',
    # 'Wikipedia (en)',
    # 'YoutubeSubtitles',
    # 'Ubuntu IRC',
]

MODEL_NAME_SHORT = 'GPT2'
MODEL_NAME = {
    'GPT2': 'gpt2',
    'LLAMA2': 'daryl149/llama-2-7b-hf',
    'MISTRAL': 'mistralai/Mistral-7B-v0.1',
}[MODEL_NAME_SHORT]
SEQ_LENGTH = 64
SAVE_PATH = f'/workspace/corpus/the_pile/the_pile_500k_{SEQ_LENGTH}_{MODEL_NAME_SHORT}'
DATASET_NAME = 'ArmelR/the-pile-splitted'
SIZE = 500_000
SPLITS = {
    'train': 0.94,
    'val': 0.03,
    'test': 0.03
}

tokenizer = get_tokenizer(MODEL_NAME)

def tokenize_chunks(examples, seq_length=SEQ_LENGTH, max_chunks=5):
    tokens = tokenizer(
        examples['text'][:seq_length*max_chunks*15],  # guess that on average tokens are <15 chars
        truncation=True,
        max_length=seq_length * max_chunks,
        padding=False,
        # add_special_tokens=False,   # hopefully it's fine not to have bos tokens...
    )
    chunks = {
        'input_ids': [],
        'attention_mask': [],
        'text': [],   # decoded text for convenience
        'id': [],     # unique hex string for convenience
    }
    for input_ids, att_mask in zip(tokens['input_ids'], tokens['attention_mask']):
        while len(input_ids) >= seq_length:
            cur_ids, input_ids = input_ids[:seq_length], input_ids[seq_length:]
            cur_mask, att_mask = att_mask[:seq_length], att_mask[seq_length:]
            chunks['input_ids'].append(cur_ids)
            chunks['attention_mask'].append(cur_mask)
            chunks['text'].append(tokenizer.decode(cur_ids))
            chunks['id'].append(secrets.token_hex(16))
    return chunks

if __name__ == '__main__':
    for subset in SUBSETS:
        print(subset)
        ds_stream = iter(load_dataset(DATASET_NAME, subset, split='train', streaming=True))
        ds = Dataset.from_list([])
        # Book datasets have fewer rows and more tokens per row
        batch_size = 100 if 'Book' in subset else 1000
        # max number of chunks to take per original row
        # max_chunks = 50 if 'Book' in subset else 5
        max_chunks = 1

        while len(ds) < SIZE:
            batch = Dataset.from_list([next(ds_stream) for _ in range(batch_size)])
            ds = concatenate_datasets([
                ds, 
                batch.map(
                    lambda x: tokenize_chunks(x, max_chunks=max_chunks), 
                    batched=True, 
                    batch_size=batch_size, 
                    remove_columns=['text', 'meta', 'domain']
                )
            ])
            print(len(ds))

        # Redo the train/val/test split ourselves
        splits_d = {}
        ds_size = min(SIZE, len(ds))
        cur_idx = 0
        for split, split_p in SPLITS.items():
            split_size = int(ds_size * split_p)
            splits_d[split] = Dataset.from_dict(ds[cur_idx:cur_idx+split_size])
            cur_idx += split_size
            print(f'{subset} {split} size: {len(splits_d[split])}')
            print(splits_d[split]['text'][:5])
        DatasetDict(splits_d).save_to_disk(f'{SAVE_PATH}/{subset}')
