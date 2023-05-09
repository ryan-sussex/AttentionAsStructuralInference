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
        att = (q @ k.transpose(-2, -1)) * (1. / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # Softmax
        att = self.softmax(att)
        y = self.value(att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, 1)
        # re-assemble all head outputs side by side
        self.record = {"attention": att[:, :, -1, ]}
        return y


class ExpandingAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # Beta distribution prior (for geometric)
        self.alpha = nn.Parameter(torch.tensor(.1))
        self.beta = nn.Parameter(torch.tensor(.9))

    @staticmethod
    def get_truncated_window(alpha, beta):
        # batch size, sequence length, embedding dimensionality (n_embd)
        p = alpha / (alpha + beta)
        window = torch.log(torch.Tensor([.05])) / torch.log(1 - p)
        # print(window)
        return window


    @staticmethod
    def softmax(x, window: torch.Tensor, k):
        _, _, n_xs = x[:, :, -window:].shape
        geo_probs = torch.tensor([- 1/k * i  for i in range(n_xs)]).flip(0)
        #
        P = F.softmax(
            x[:, :, -window:]
              + geo_probs,
            dim=-1
        )
        return P

    @staticmethod
    def beta_update(att: torch.Tensor, k):
        count = torch.range(att.size(-1) - 1, 0, step=-1) # T
        beta_update = torch.einsum("bht, t -> bh", att, count)
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
        att = (q @ k.transpose(-2, -1)) * (1. / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att[:, :, :, -1]
        # print(att.shape)
        att[:, :, -1] = float('-inf')
        # print(att)
        alpha = self.alpha
        beta = self.beta
        k_old = torch.tensor(0)
        # print("start")
        # Iterative inference
        tol = 0.01
        for i in range(100):
            # Softmax
            k = self.get_truncated_window(alpha, beta)
            # if i == 0:
            #     print(k)
            # print(k)
            window = torch.ceil(k)
            window = window.to(torch.int)
            att_iter = self.softmax(att, window, k)
            alpha = alpha + 1
            beta = beta + self.beta_update(att_iter, k)
            # Stopping criteria
            if k > T:  # We are looking at max window size
                break
            # if k < k_old:  # We don't need a bigger window
            #     break
            if (k - k_old) < tol:  # We don't need a bigger window
                break
            k_old = k.detach()

        att = att_iter
        self.record = {
            "attention": att,
            "window": window,
            "k_old": k_old,
            "iters": i,
            "k": k
        }

        y = self.value(att, v[:, :, -window:, :])
        y = y.transpose(1, 2).contiguous().view(B, 1, 1)
        return y



class EfficientExpandingAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # Beta distribution prior (for geometric)
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(2.))

    @staticmethod
    def get_truncated_window(alpha, beta):
        # batch size, sequence length, embedding dimensionality (n_embd)
        p = alpha / (alpha + beta)
        return 2 / p

    def get_dot_prod(self, k: torch.Tensor, q: torch.Tensor, i, T):
        # print(i)
        if i in self.attention_cache["dot_prods"]:
            return self.attention_cache["dot_prods"][i]
        dot_prod = q[:, :, -1, :] @ k[:, :, i:i+1, :]\
            .transpose(-2, -1) * (.01 / math.sqrt(k.size(-1)))
        dot_prod = dot_prod.squeeze(-1).squeeze(0)
        dot_prod = torch.exp(dot_prod)
        if i == (T-1):
            dot_prod = torch.zeros(dot_prod.shape)
        self.attention_cache["dot_prods"][i] = dot_prod
        self.attention_cache["sum"] = self.attention_cache["sum"] + dot_prod
        return dot_prod

    def get_window_stacked(self, k, q, window, T):
        if window in self.attention_cache["windows"]:
            return self.attention_cache["windows"][window]
        total_window = torch.stack(
            [self.get_dot_prod(k, q, i, T) for i in range(T-int(window), T)],
            dim=-1
        )
        self.attention_cache["windows"][window] = total_window
        return total_window


    def softmax(self, k, q, window: torch.Tensor, m, T):
        window = window.item()
        window = min(T, abs(window))
        # geo_probs = torch.tensor([ - i/m  for i in range(1, window+1)]).flip(0)
        # geo_probs = geo_probs[None, None, :]
        P = self.get_window_stacked(k, q, window, T)
        # + geo_probs
        P = P / (self.attention_cache["sum"])
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

        # We are still computing these for more values than necessary!!
        # For efficient implementation we need to split
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, 1).transpose(1, 2)
        # (B, nh, T, hs)

        alpha = self.alpha
        beta = self.beta
        m_old = torch.tensor(0)
        # Iterative inference
        self.attention_cache = {"dot_prods":{}, "windows": {}, "sum":0}
        for i in range(30):
            # Softmax
            m = self.get_truncated_window(alpha, beta)
            window = torch.ceil(m)
            window = window.to(torch.int)
            att_iter = self.softmax(k, q, window, m, T)
            alpha = alpha + 1
            beta = beta + self.beta_update(att_iter)
            # Stopping criteria
            if m > T:  # We are looking at max window size
                break
            if m < m_old:  # We don't need a bigger window
                break
            m_old = m.detach()

        att = att_iter
        self.record = {
            "attention": att,
            "window": window,
            "m_old": m_old,
            "iters": i,
            "m": m
        }

        y = self.value(att, v[:, :, -window:, :])
        y = y.transpose(1, 2).contiguous().view(B, 1, 1)
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
        P = F.softmax(x, dim=-2)
        # print(P[0, 0, :, :].sum(dim=-2))
        # raise
        I = torch.eye(T).reshape(1, T, T).repeat(B, nh, 1, 1)
        paths = (I + P + P @ P) / 3
        # print(paths[0, 0, :, :].sum(dim=-1))

        paths = paths.transpose(-1, -2)   # (B, nh, T, TxT)
        return paths
