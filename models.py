import math

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
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd + 1, bias=False)
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
        P = F.softmax(x, dim=-1)
        return P

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
        v = v.view(B, T, self.n_head, 1).transpose(1, 2)
        # (B, nh, T, hs)

        # causal self-attention; Self-attend:
        #  (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Create Attention Matrix
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Softmax
        att = self.softmax(att)
        y = self.value(att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, 1) 
        # re-assemble all head outputs side by side
        return y


class ExpandingAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # Beta distribution prior (for geometric)
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(5.))

    @staticmethod
    def get_truncated_window(alpha, beta):
        # batch size, sequence length, embedding dimensionality (n_embd)
        p = alpha / (alpha + beta)
        return 2 / p


    @staticmethod    
    def softmax(x, window: torch.Tensor):
        P = F.softmax(x[:, :, -window:], dim=-1)
        return P

    @staticmethod
    def beta_update(att: torch.Tensor):
        count = torch.range(att.size(-1) - 1, 0, step=-1) # T
        # print(count)
        beta_update = torch.einsum("bht, t -> bh", att, count)
        # print("beta update", beta_update)
        # print(beta)
        return beta_update

    def forward(self, x):
        B, T, C = x.size() 
        # calculate query, key, values 
        # for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, 1).transpose(1, 2)
        # (B, nh, T, hs)

        # causal self-attention; Self-attend:
        #  (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Create Attention Matrix
        att = (q @ k.transpose(-2, -1)) * (.001 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att[:, :, :, -1]
        # print(att.shape)
        att[:, :, -1] = float('-inf')
        # print(att)
        alpha = self.alpha
        beta = self.beta
        k_old = torch.tensor(0)
        ctr = 0
        # print("start")
        for i in range(30):
            # Softmax
            k = self.get_truncated_window(alpha, beta)
            window = torch.ceil(k)
            window = window.to(torch.int)
            att_iter = self.softmax(att, window)
            alpha = alpha + 1
            beta = beta + self.beta_update(att_iter)
            if k > T:
                break
            if k < k_old:
                break
            # if k_old.item() == window.item():
            #     ctr +=1
            # if ctr > 3:
            #     break
            k_old = k.detach()
            # if k > T or window <= k_old:
            #     break
            # if ((k + .5) <= k_old.item()):
            #     break

            # print("beta", beta)
            # print("window", window)
            # print("att", att_iter)
        # raise
        # print("window", window)
        att = att_iter
        self.record = {
            "attention": att,
            "window": window,
            "k_old": k_old,
            "iters": i,
            "k": k
        }
        # print(att)
        # print(v.shape)
        # print(window)
        y = self.value(att, v[:, :, -window:, :])
        y = y.transpose(1, 2).contiguous().view(B, 1, 1)
        # print(v)
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
        V = V.flatten(start_dim=-3, end_dim=-2)
        # (B, nh, TxT, hs)
        return att @ V
        # (B, nh, T, TxT) x (B, nh, TxT, hs) -> (B, nh, T, h_s)


class LongAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def softmax(x: torch.Tensor) -> torch.Tensor:
        """
        Return a probability per triple (x_i, (x_j, x_k))
        """
        B, nh, T, _ = x.size()
        # (B, nh, T, T)
        P = F.softmax(x, dim=-1)
        I = torch.eye(T).reshape(1, T, T).repeat(B, nh, 1, 1)
        paths = (I + P + P @ P) / 3
        return paths   # (B, nh, T, TxT)


# class StackedAttention(nn.Module): 
#     def __init__(self, config, attention_model, layers: int = 2) -> None:
#         self.attention = nn.Sequential(
#             [("attention_{i}", attention_model(config))]
#         )
    
#     def forward(self, x):
#         return self.attention(x)