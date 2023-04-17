import torch
from torch.nn import MSELoss
import torch.optim as optim

from data import Regression, SequenceRegression
from models import CausalSelfAttention, ThreeAttention


N_BATCH = 1
N_SEQ = 5
EMBED_DIM = 2


class AttentionConfig():
    n_embd=EMBED_DIM
    n_head=1
    block_size=1


attn = CausalSelfAttention(AttentionConfig())
attn = ThreeAttention(AttentionConfig())

regr_problem = Regression(input_dim=EMBED_DIM, output_dim=EMBED_DIM)
seq_regr_problem = SequenceRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)
optimizer = optim.Adam(attn.parameters(), lr=1e-3)
loss = MSELoss()

def linear_regression():
    # Example of simple linear regression
    N_SAMPLES = 500
    for _ in range(N_SAMPLES):
        X, y = regr_problem.sample(batch_size=1000)
        print(X.shape)
        output = attn(X)
        out: torch.Tensor = loss(output, y)
        out.backward()
        optimizer.step()
        print(out.item())

def seq_regression():
    N_SAMPLES = 500
    for _ in range(N_SAMPLES):
        X, y = seq_regr_problem.sample(batch_size=1000)
        output = attn(X)
        # output, attn_weights = attn(query, key, value)
        # Pick a single sequence element for y_prediction
        output_pred = output[:, 1, :]
        out: torch.Tensor = loss(output_pred, y)
        out.backward()
        optimizer.step()
        print(out.item())


if __name__ == "__main__":

    # linear_regression()
    seq_regression()
    # Example of regression where the input of the regression is chosen at random

    print(seq_regr_problem.weight_matrix)
    for param in attn.parameters():
        print(param)
