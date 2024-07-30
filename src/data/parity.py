from torch.utils.data import Dataset, DataLoader
import random

class ParityDataset(Dataset):
    def __init__(self, tokenizer, size, seq_len):
        super().__init__()
        self.size = size
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return self.size

    def __getitem__(self, value):
        x = ''.join(' ' + str(random.randint(0, 1)) for _ in range(self.seq_len))
        x += ' P ' + str(x.count('1') % 2)
        out = self.tokenizer(x, return_tensors='pt')
        out['input_ids'] = out['input_ids'].flatten()
        return out

