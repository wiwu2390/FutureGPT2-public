from torch.utils.data import Dataset, DataLoader
import random

class MultiplicationDataset(Dataset):
    def __init__(self, tokenizer, size, x_min_digits=1, x_max_digits=10, y_min_digits=1, y_max_digits=10, x_pad=None, y_pad=None):
        super().__init__()
        self.size = size
        self.x_min_digits = x_min_digits
        self.x_max_digits = x_max_digits
        self.y_min_digits = y_min_digits
        self.y_max_digits = y_max_digits
        self.x_pad = x_max_digits if x_pad is None else x_pad
        self.y_pad = y_max_digits if y_pad is None else y_pad
        self.tokenizer = tokenizer

    def __len__(self):
        return self.size

    def __getitem__(self, value):
        x_digits = random.randint(self.x_min_digits, self.x_max_digits)
        y_digits = random.randint(self.y_min_digits, self.y_max_digits)
        x = random.randrange(0, 10**x_digits)
        y = random.randrange(0, 10**y_digits)
        x_f = '{:0' + str(self.x_pad) + 'd}'
        y_f = '{:0' + str(self.y_pad) + 'd}'
        out_f = '{:0' + str(self.x_max_digits+self.y_max_digits) + 'd}'
        out_text = x_f.format(x)[::-1] \
            + '*' + y_f.format(y)[::-1] + '='\
            + out_f.format(x * y)[::-1]
        out_text = ''.join(b + a for a, b in zip(out_text, ' ' * len(out_text)))
        out = self.tokenizer(out_text, return_tensors='pt')
        #out['text'] = out_text
        out['input_ids'] = out['input_ids'].flatten()
        return out
