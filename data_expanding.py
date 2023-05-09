"""
Helper classes for simulating regression problems.
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

import random


STD = .1

class Regression():

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            weight_matrix: Optional[Tensor] = None
        ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_matrix is None:
            self.weight_matrix = torch.normal(
                torch.ones(output_dim, input_dim), std=1
            )

    def sample_X(self, batch_size: int) -> Tensor:
        return torch\
            .normal(mean=torch.ones(batch_size, self.input_dim), std=STD)

    def regr_func(self, X):
        return F.linear(self.weight_matrix, X)

    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            X = self.sample_X(batch_size)
            y = self.regr_func(X).t()

            # Put in empty sequence dim
            X = X[:, None, :]
            y = y[:, None]
        return X, y


class AutoRegression(Regression):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int = 10,
        weight_matrix: Optional[Tensor] = None,
        ingroup_size: int = 1
    ):
        super().__init__(input_dim, output_dim, weight_matrix)
        # self.autoregressive_matrix = torch.normal(
        #         torch.ones(input_dim, input_dim), std=1
        #     ) * 0.5
        self.autoregressive_matrix = torch.rand(
                (input_dim, input_dim)
            )
        # * 0.6
        self.sequence_length = sequence_length
        self.ingroup_size = ingroup_size

    def sample_X(self, batch_size: int) -> Tensor:
        outgroup_size = self.sequence_length - self.ingroup_size

        # in_group = [ torch.normal(
        #     mean=torch.ones(batch_size, 1, self.input_dim),
        #     std=10
        # )
        # ]

        in_group = [
            torch.rand(batch_size, 1, self.input_dim) * 10
        ]
        outgroup = torch.normal(
            mean=torch.zeros(batch_size, outgroup_size, self.input_dim),
            std=STD
        )

        for i in range(1, self.ingroup_size):
            new = (
                in_group[i - 1] @ self.autoregressive_matrix
                + torch.normal(
                    mean=torch.zeros(batch_size, 1, self.input_dim),
                    std=STD
                )
            )
            in_group.append(new)

        in_group = [outgroup] + in_group  
        X = torch.cat(in_group, dim=1)
        return X

    

    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            X = self.sample_X(batch_size)
            x = X[:, self.sequence_length - self.ingroup_size, :].squeeze(dim=0)
            y = self.regr_func(x).t()
        # randomly select a single y
        # shuffle X
        # print(X.size())
        idx = list(range(self.sequence_length - 1))
        random.shuffle(idx)
        # raise
        idx = idx + [self.sequence_length - 1]
        self.record = {"idx": idx}
        # print(idx)
        # raise
        X = X[:, idx, :]
        return X, y

