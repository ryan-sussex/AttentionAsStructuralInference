import torch
torch.manual_seed(4)

from torch.nn import MSELoss
import torch.optim as optim

from data import AutoRegression
from models import CausalSelfAttention, LongAttention, ExpandingAttention, EfficientExpandingAttention



N_BATCH = 1
BATCH_SIZE = 1
N_SAMPLES = 2000

N_SEQ = 30
N_AUTOREGRESS = 2
EMBED_DIM = 20
OUTPUT_DIM = 1
LR = 1e-3

RECORD_EVERY = 200


class AttentionConfig():
    n_embd=EMBED_DIM
    n_head=1
    block_size=1

att_config = AttentionConfig()

auto_regr_problem = AutoRegression(
    input_dim=EMBED_DIM, 
    output_dim=OUTPUT_DIM,
    sequence_length=N_SEQ, 
    ingroup_size=N_AUTOREGRESS
)


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
            # if  loss.item() > 0:
            print(
                f"model:{repr(attention_model)} batch_no:{batch_no} mse:{loss}"
            )
            training_history.append(loss.item())
            #     print(X)
            #     print(y)
            #     print("window", attention_model.record["window"].item())
            #     print("iters", attention_model.record["iters"])
            #     # print("k", attention_model.record["k"])
            #     print(attention_model.record["attention"])

            # # print(problem.record)
            #     print(attention_model.alpha)
            #     print(attention_model.beta)

    return training_history


if __name__ == "__main__":
    training_dct = {}
    attn = EfficientExpandingAttention(AttentionConfig())
    training_dct["expanding"] = seq_regression(auto_regr_problem, attention_model=attn)
    # attn = CausalSelfAttention(AttentionConfig())
    # training_dct["standard"] = seq_regression(auto_regr_problem, attention_model=attn)
    attn = ExpandingAttention(AttentionConfig())
    training_dct["expanding"] = seq_regression(auto_regr_problem, attention_model=attn)
    attn = LongAttention(AttentionConfig())
    training_dct["long"] = seq_regression(auto_regr_problem, attention_model=attn)

    import json

    with open("training.json", mode="w") as f:
        json.dump(training_dct, f)
