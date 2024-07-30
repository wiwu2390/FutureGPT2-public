import torch
from torch import nn, Tensor
import collections
from functools import partialmethod
from abc import ABC, abstractmethod
from torch.nn import functional as F

class Neck(ABC, nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_idxs, hidden_lb=0, token_lb=0):
        '''
        We allow __init__ to have varying signatures, and use partialcls to
        give LitFutureModel necks with __init__ signature (self, hidden_size, vocab_size)
        hidden_size: hidden dim of base transformer
        vocab_size: vocab size of base transformer
        hidden_idxs: base transformer hidden layer indices to use
        hidden_lb: lookback along hidden state sequence
            the indexing is a little confusing here; 0 means that at position n we look at just
            the hidden state at n; 1 means that at n we look at n and n-1, etc.
            set to -1 to ignore hidden state
            should be <= 0 for LSTM (because LSTM already does lookback itself.)
        token_lb: lookback along token sequence
            indexing starts at the *n+1*th token; 0 means that at position n we look at
            just the input token at n+1; 1 means at n we look at n+1 and n, etc.
            set to -1 to ignore token sequence
            should be <= 0 for LSTM
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hidden_idxs = hidden_idxs if isinstance(hidden_idxs, list) else [hidden_idxs]
        self.hidden_lb = max(hidden_lb, -1)
        self.token_lb = max(token_lb, -1)
        # Don't train our own embedding; just use the base model's
        # if use_next:
            # self.embedding = nn.Embedding(vocab_size, hidden_size)

    @abstractmethod
    def forward(self, hidden_states, tokens):
        '''
        LitFutureModel assumes this signature is respected.
        hidden_states is a list of the model's hidden states
        tokens is the token sequence up to (n+1)
        '''
        pass

    def input_size(self):
        # weird behaviour when input size is zero
        # so in this case, in form_input we return a size 1 zero tensor instead
        return max(1, self.hidden_size * (len(self.hidden_idxs) * (self.hidden_lb+1) + (self.token_lb+1)))

    def form_input(self, hidden_states, tokens):
        def shift(x, n, sdim=-2):
            '''
            shift(x, n)[...,i] = x[..., i+n] ( or 0 )
            '''
            # transpose (sdim, -1), shift, untranspose
            x = x.transpose(sdim, -1)
            if n > 0:
                x = F.pad(x[..., n:], (0, n))
            elif n < 0:
                x = F.pad(x[..., :n], (-n, 0))
            return x.transpose(sdim, -1)

        # Default init in linear case is torch.eye
        # so we need the top hidden state to be at index 0 for it to go into the identity.
        # thus, reverse sort
        hidden_idxs = sorted([idx % len(hidden_states) for idx in self.hidden_idxs], reverse=True)
        state = [shift(hidden_states[idx], -lb) for lb in range(self.hidden_lb+1) for idx in hidden_idxs]
        state += [shift(tokens, 1-lb) for lb in range(self.token_lb+1)]
        if state:
            try:
                state = torch.concat(state, dim=2)
            except Exception:
                import pdb; pdb.set_trace()
        else:
            state = torch.zeros(hidden_states[-1].shape[:2] + (1,)).cuda()
        return state

    def reg_loss(self):
        '''
        Regularization loss.
        '''
        return 0

# wikipedia says MLP is a "misnomer" :(
class MLPNeck(Neck):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        hidden_idxs,
        layer_dims,
        hidden_lb,
        token_lb,
        act=nn.GELU(),
        bias=True,
    ):
        '''
        hidden_size: hidden_size of base transformer model
        hidden_idxs: indices of hidden state layers to use in input to neck
        act: activation function between linear layers
        bias: whether to use trainable bias in the linear layers
        '''
        super().__init__(hidden_size, vocab_size, hidden_idxs, hidden_lb=hidden_lb, token_lb=token_lb)
        self.layers = nn.ModuleList()
        self.act = act
        d0 = self.input_size()
        if not isinstance(layer_dims, list):
            layer_dims = [layer_dims]
        for d1 in layer_dims:
            self.layers.append(nn.Linear(d0, d1, bias=bias))
            d0 = d1
        # Last layer's output dim must be hidden_size
        self.layers.append(nn.Linear(d0, hidden_size, bias=bias))
        # If single layer neck, makes sense to initialize to id
        # Else, default initialization is probably fine.
        if len(self.layers) == 1:
            nn.init.eye_(self.layers[0].weight.data)
            if bias:
                nn.init.zeros_(self.layers[0].bias.data)

    def forward(self, hidden_states, tokens):
        state = self.form_input(hidden_states, tokens)
        for layer in self.layers[:-1]:
            state = self.act(layer(state))
        return self.layers[-1](state)

    def reg_loss(self):
        wgt = self.layers[0].weight
        d0 = wgt.shape[1]
        return sum(torch.linalg.matrix_norm(wgt[:,i:i+self.hidden_size], ord='fro') for i in range(d0 // self.hidden_size))

class LSTMNeck(Neck):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        hidden_idxs,
        neck_size,
        num_layers,
        hidden_lb=0,
        token_lb=0,
        lstm_params=dict(),
        gru=False,
        add_linear=False,
    ):
        assert hidden_lb in [0, -1], 'hidden_lb for LSTM must be either 0 or -1'
        assert token_lb in [0, -1], 'token_lb for LSTM must be either 0 or -1'
        super().__init__(hidden_size, vocab_size, hidden_idxs, hidden_lb=hidden_lb, token_lb=token_lb)
        rnn_cls = nn.GRU if gru else nn.LSTM
        self.rnn = rnn_cls(
            self.input_size(),
            hidden_size=neck_size,
            num_layers=num_layers,
            batch_first=True,
            **lstm_params,
        )
        if neck_size != self.hidden_size or add_linear:
            self.linear = nn.Linear(neck_size, self.hidden_size, bias=True)
        else:
            self.linear = None

    def forward(self, hidden_states, tokens):
        out = self.rnn(self.form_input(hidden_states, tokens))[0]   # rnn returns (output, rnn_hidden_states)
        if self.linear is not None:
            out = self.linear(out)
        return out

