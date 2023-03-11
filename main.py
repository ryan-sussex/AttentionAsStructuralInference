from torch.nn import MultiheadAttention
import torch
from torch.nn import MSELoss
import torch.optim as optim

from data import Regression, SequenceRegression

N_BATCH = 1
N_SEQ = 5
EMBED_DIM = 2

attn = MultiheadAttention(embed_dim=EMBED_DIM, num_heads=1, batch_first=True)
regr_problem = Regression(input_dim=EMBED_DIM, output_dim=EMBED_DIM)
seq_regr_problem = SequenceRegression(
    input_dim=EMBED_DIM, output_dim=EMBED_DIM, sequence_length=N_SEQ)
optimizer = optim.Adam(attn.parameters(), lr=1e-3)
loss = MSELoss()

if __name__ == "__main__":

    # Example of simple linear regression
    N_SAMPLES = 100
    for _ in range(N_SAMPLES):
        X, y = regr_problem.sample(batch_size=1000)

        query = X
        key = query.clone()
        value = query.clone()
        output, attn_weights = attn(query, key, value)
        out: torch.Tensor = loss(output, y)
        out.backward()
        optimizer.step()
        print(out.item())

    # Example of regression where the input of the regression is chosen at random
    N_SAMPLES = 1000
    for _ in range(N_SAMPLES):
        X, y = seq_regr_problem.sample(batch_size=1000)

        query = X
        key = query.clone()
        value = query.clone()
        output, attn_weights = attn(query, key, value)
        # Pick a single sequence element for y_prediction
        output_pred = output[:, 1, :]
        out: torch.Tensor = loss(output_pred, y)
        out.backward()
        optimizer.step()
        print(out.item())

    print(seq_regr_problem.weight_matrix)
    for param in attn.parameters():
        print(param)
