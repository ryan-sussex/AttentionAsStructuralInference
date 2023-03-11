from torch.nn import MultiheadAttention
import torch
from torch.nn import MSELoss
import torch.optim as optim

from data import Regression

N_BATCH = 1
N_SEQ = 1
EMBED_DIM = 1

attn = MultiheadAttention(embed_dim=EMBED_DIM, num_heads=1)
regr_problem = Regression(input_dim=EMBED_DIM, output_dim=1)
optimizer = optim.Adam(attn.parameters(), lr=1e-3)
loss = MSELoss()

if __name__ == "__main__":

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