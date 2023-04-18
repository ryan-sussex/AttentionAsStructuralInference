import math
from itertools import combinations

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.register_buffer(
            "bias", 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    @staticmethod    
    def softmax(x):
        return F.softmax(x, dim=-1)

    @staticmethod
    def value(att, v):
        return att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

    def forward(self, x):
        B, T, C = x.size() 
        # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values 
        # for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)

        # causal self-attention; Self-attend:
        #  (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Create Attention Matrix
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Softmax
        att = self.softmax(att)
        y = self.value(att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        # re-assemble all head outputs side by side
        return y


class ThreeAttention(CausalSelfAttention):

    @staticmethod
    def softmax(x: torch.Tensor):
        """
        Return a probability per triple (x_i, (x_j, x_k))
        """
        # (B, nh, T, T)
        _, _, _, T = x.size()
        # # (B, nh, T, T, T)
        stacked = torch.stack([
            x + x.roll(1, dims=-1)
            for _ in range(T)
        ], dim=-1)
        stacked = stacked.flatten(start_dim=-2, end_dim=-1)
        return F.softmax(stacked, dim=-1)   # (B, nh, T, TxT)

    @staticmethod
    def value(att, v):
        _, _, T, _ = att.size()
        # (B, nh, T, TxT)
        V = torch.stack([
            v
            for _ in range(T)
        ], dim=-2)
        # .transpose(-3, -2)
        V = V.flatten(start_dim=-3, end_dim=-2)
        # .transpose(-2, -1)
        # (B, nh, TxT, hs)
        return att @ V
        # (B, nh, T, TxT) x (B, nh, TxT, hs) -> (B, nh, T, h_s)