import torch
torch.manual_seed(8)

from torch.nn import MSELoss
import torch.optim as optim

from data import AutoRegression
from models import CausalSelfAttention, LongAttention



N_BATCH = 1
BATCH_SIZE = 200
N_SAMPLES = 20000
N_RUNS = 10

N_SEQ = 5
N_AUTOREGRESS = 3
EMBED_DIM = 3
OUTPUT_DIM = 1
LR = 1e-3

RECORD_EVERY = 1000


class AttentionConfig():
    n_embd=EMBED_DIM
    n_head=1
    block_size=1

att_config = AttentionConfig()




loss_fn = MSELoss(reduction="mean")

def seq_regression(problem, attention_model):
    optimizer = optim.Adam(attention_model.parameters(), lr=LR, weight_decay=2)

    training_history = []
    for batch_no in range(N_SAMPLES):
        optimizer.zero_grad()

        X, y = problem.sample(batch_size=BATCH_SIZE)
        output = attn(X)
        # Only use the last node for prediction
        output_pred = output[:, - 1, :]
        
        loss: torch.Tensor = loss_fn(output_pred, y)
        loss.backward()
        optimizer.step()
        if batch_no % RECORD_EVERY == 0:
            print(
                f"model:{repr(attention_model)} batch_no:{batch_no} mse:{loss}"
            )
            training_history.append(loss.item())
    return training_history


if __name__ == "__main__":
    multiple_training_dct = {}
    for i in range(N_RUNS):
        print("="*100)
        print(f"iteration {i}")
        torch.manual_seed(i)
        auto_regr_problem = AutoRegression(
            input_dim=EMBED_DIM, 
            output_dim=OUTPUT_DIM,
            sequence_length=N_SEQ, 
            ingroup_size=N_AUTOREGRESS
        )
        training_dct = {}
        attn = LongAttention(AttentionConfig())
        training_dct["long"] = seq_regression(auto_regr_problem, attention_model=attn)
        attn = CausalSelfAttention(AttentionConfig())
        training_dct["standard"] = seq_regression(auto_regr_problem, attention_model=attn)
        multiple_training_dct[i] = training_dct

    import json
    with open("./data/multiple_training.json", mode="w") as f:
        json.dump(multiple_training_dct, f)