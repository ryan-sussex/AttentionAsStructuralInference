import torch
from torch.nn import MSELoss
from torch.nn import Linear
import torch.optim as optim

from data import Regression, SequenceRegression, TwoVarSequenceRegression, AutoRegression
from models import CausalSelfAttention, ThreeAttention


N_BATCH = 1
BATCH_SIZE = 5000
N_SAMPLES = 5000

N_SEQ = 5
EMBED_DIM = 1
LR = 1e-3


class AttentionConfig():
    n_embd=EMBED_DIM
    n_head=1
    block_size=1

att_config = AttentionConfig()

# attn = CausalSelfAttention(AttentionConfig())
# mlp = Linear(att_config.n_embd, 1)
# attn = ThreeAttention(AttentionConfig())

regr_problem = Regression(input_dim=EMBED_DIM, output_dim=EMBED_DIM)
seq_regr_problem = SequenceRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)
auto_regr_problem = AutoRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ, ingroup_size=3)
seq_regr_problem_two_var = TwoVarSequenceRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)


# optimizer = optim.SGD(attn.parameters(), lr=LR)

loss = MSELoss(reduction="mean")

def linear_regression():
    # Example of simple linear regression
    attn = CausalSelfAttention(AttentionConfig())
    optimizer = optim.Adam(attn.parameters(), lr=LR)

    for _ in range(N_SAMPLES):
        X, y = regr_problem.sample(batch_size=BATCH_SIZE)
        output = attn(X)
        out: torch.Tensor = loss(output, y)
        out.backward()
        optimizer.step()
        if _ % 200 == 1:
            print(X[0] , y[0])
            # print(regr_problem.weight_matrix)
            print("pred", output[0].item(), "gt", y[0].item())
            # raise
            print(out.item())
    for param in attn.parameters():
        print(param)

def seq_regression(problem):
    # attn = CausalSelfAttention(AttentionConfig())
    attn = ThreeAttention(AttentionConfig())

    optimizer = optim.Adam(attn.parameters(), lr=LR, weight_decay=1)
    for _ in range(N_SAMPLES):
        X, y = problem.sample(batch_size=BATCH_SIZE)
        # print(X[0] , y[0])
        # print(problem.weight_matrix)
        # raise
        output = attn(X)
        # output_pred = mlp(X)
        # print(output_pred.shape)
        # Pick a single sequence element for y_prediction
        # print(output.size())
        # raise
        output_pred = output[:, 2, :]
        out: torch.Tensor = loss(output_pred, y)
        out.backward()
        optimizer.step()
        if _ % 1000 == 1:
            print(X[0] , y[0])
            print(regr_problem.weight_matrix)
            print("pred")
            print(output_pred[0])
            print("gt")
            print(y[0])
            # raise
            print(out.item())
    for param in attn.parameters():
        print(param)

if __name__ == "__main__":

    # linear_regression()


    # seq_regression(seq_regr_problem)
    seq_regression(auto_regr_problem)

    # # Example of regression where the input of the regression is chosen at random

    # print(auto_regr_problem.weight_matrix)
    # print(auto_regr_problem.autoregressive_matrix)

    # for param in attn.parameters():
    #     print(param)
