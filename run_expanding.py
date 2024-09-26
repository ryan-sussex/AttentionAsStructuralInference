from typing import List
from dataclasses import dataclass, asdict

import numpy as np
import torch
RANDOM_SEED = 4
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from torch.nn import MSELoss
import torch.optim as optim

from data_expanding import AutoRegression
from models import CausalSelfAttention, ExpandingAttention



N_BATCH = 1
BATCH_SIZE = 1
N_SAMPLES = 10000

N_SEQ = 50
N_AUTOREGRESS = 2
EMBED_DIM = 10
OUTPUT_DIM = 1
LR = 5e-4

RECORD_EVERY = 200

@dataclass
class AttentionConfig():
    n_embd=EMBED_DIM
    n_head=1
    block_size=1
    n_tokens=N_SEQ

att_config = AttentionConfig()



loss_fn = MSELoss(reduction="mean")

def seq_regression(problem, attention_model):
    optimizer = optim.Adam(attention_model.parameters(), lr=LR, weight_decay=2)

    training_history = {"loss": [], "iters": [], "window":[]}
    for batch_no in range(N_SAMPLES):

        X, y = problem.sample(batch_size=BATCH_SIZE)
        output = attn(X)
        # Only use the last node for prediction
        output_pred = output[:, - 1, :]
        
        loss: torch.Tensor = loss_fn(output_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_history["loss"].append(loss.item())
        iters = attention_model.record.get("iters", 0)
        training_history["iters"].append(iters)
        window = attention_model.record.get("window", torch.Tensor([0.])).item()
        training_history["window"].append(window)
        if batch_no % RECORD_EVERY == 0:

            # if  loss.item() > 0:
            print(
                f"model:{repr(attention_model)} batch_no:{batch_no} mse:{loss}"
            )
            print("iters", iters)
            print("window", window)

            print(output_pred)
            print(y)
            print(problem.record["idx"].index(N_SEQ - 2))
            print(
                N_SEQ - attention_model.record["attention"].size(-1) 
                + torch.argmax(attention_model.record["attention"]).item()
            )

    return training_history


class AttentionType:
    expanding = "expanding"
    standard = "standard"


@dataclass
class RecordExperiment:
    attention_type: str 
    loss: List[float]
    iterations: List[float]
    window: List[float]
    geo_p: float = 0
    attention_config: AttentionConfig = att_config
    lr: float = LR
    record_every: int = RECORD_EVERY
    random_seed: int =  RANDOM_SEED

@dataclass
class RecordResult:
    expanding: RecordExperiment
    standard: RecordExperiment


@dataclass
class RecordMultipleResults:
    results: List[RecordResult]


# class HyperParamGrid
p_opts = [1/2, 1/5, 1/10, 1/25]


if __name__ == "__main__":

    all_results = []
    for p in p_opts:
        auto_regr_problem = AutoRegression(
            input_dim=EMBED_DIM, 
            output_dim=OUTPUT_DIM,
            sequence_length=N_SEQ, 
            ingroup_size=N_AUTOREGRESS,
            geo_p=p
        )

        attn = ExpandingAttention(AttentionConfig())
        results = seq_regression(
            auto_regr_problem, attention_model=attn
        )
        experiment_expanding = RecordExperiment(
            attention_type=AttentionType.expanding,
            loss=results["loss"],
            iterations=results["iters"],
            window=results["window"],
            geo_p=p
        )


        attn = CausalSelfAttention(
            AttentionConfig(), self_attend=False
        )
        results = seq_regression(
            auto_regr_problem, attention_model=attn
        )
        experiment_standard = RecordExperiment(
            attention_type=AttentionType.standard,
            loss=results["loss"],
            iterations=results["iters"],
            window=results["window"],
            geo_p=p
        )

        record = RecordResult(
            expanding=experiment_expanding, 
            standard=experiment_standard
        )
        all_results.append(asdict(record))


    # all_results = RecordMultipleResults(results=all_results)

    import json
    with open(f"./data/expanding_{EMBED_DIM}.json", mode="w") as f:
        json.dump(all_results, f)
