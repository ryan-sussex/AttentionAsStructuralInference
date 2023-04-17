import torch
from torch.nn import MSELoss
from torch.nn import Linear
import torch.optim as optim

from data import Regression, SequenceRegression, TwoVarSequenceRegression, AutoRegression
from models import CausalSelfAttention, ThreeAttention


N_BATCH = 1
BATCH_SIZE = 10000
N_SAMPLES = 5000

N_SEQ = 5
EMBED_DIM = 2
LR = 1e-3


class AttentionConfig():
    n_embd=EMBED_DIM
    n_head=1
    block_size=1

att_config = AttentionConfig()

attn = CausalSelfAttention(AttentionConfig())
mlp = Linear(att_config.n_embd, 1)
# attn = ThreeAttention(AttentionConfig())

regr_problem = Regression(input_dim=EMBED_DIM, output_dim=EMBED_DIM)
seq_regr_problem = SequenceRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)
auto_regr_problem = AutoRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)
seq_regr_problem_two_var = TwoVarSequenceRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)


optimizer = optim.Adam(attn.parameters(), lr=LR)
# optimizer = optim.SGD(attn.parameters(), lr=LR)

loss = MSELoss()

def linear_regression():
    # Example of simple linear regression
    for _ in range(N_SAMPLES):
        X, y = regr_problem.sample(batch_size=BATCH_SIZE)
        output = attn(X)
        out: torch.Tensor = loss(output, y)
        out.backward()
        optimizer.step()
        print(out.item())

def seq_regression(problem):
    for _ in range(N_SAMPLES):
        X, y = problem.sample(batch_size=BATCH_SIZE)
        # print(X[0] , y[0])
        # print(problem.weight_matrix)
        # raise
        output = attn(X)
        # output_pred = mlp(X)
        # print(output_pred.shape)
        # Pick a single sequence element for y_prediction
        output_pred = output[:, 0, :]
        out: torch.Tensor = loss(output_pred, y)
        out.backward()
        optimizer.step()
        print(out.item())


if __name__ == "__main__":

    # linear_regression()


    # seq_regression(seq_regr_problem)
    seq_regression(auto_regr_problem)

    # Example of regression where the input of the regression is chosen at random

    print(auto_regr_problem.weight_matrix)
    print(auto_regr_problem.autoregressive_matrix)

    for param in attn.parameters():
        print(param)
