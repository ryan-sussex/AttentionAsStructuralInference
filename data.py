"""
Helper classes for simulating regression problems.
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


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
            self.weight_matrix = torch.rand(
                (output_dim, input_dim)
            )

    def sample_X(self, batch_size: int) -> Tensor:
        return torch\
            .normal(mean=torch.ones(batch_size, self.input_dim), std=0.01)

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


class SequenceRegression(Regression):
    """
    This class randomly selects the index to regress against.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int = 10,
        weight_matrix: Optional[Tensor] = None
    ):
        super().__init__(input_dim, output_dim, weight_matrix)
        self.sequence_length = sequence_length
        self.categorical = Categorical(torch.ones(sequence_length))

    def sample_X(self, batch_size: int) -> Tensor:
        return torch\
            .normal(
                mean=torch.ones(
                    batch_size, self.sequence_length, self.input_dim),
                std=0.01
            )

    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            X = self.sample_X(batch_size)
            selection = self.categorical.sample((batch_size,))
            selection = selection[:, None]
            placeholder = torch.zeros(X.shape)
            placeholder[:, 0] = selection
            placeholder = placeholder.to(torch.int64)
            # placeholder.int_repr()
            x = torch.gather(X, 1, placeholder.to())[:, 0]
            y = self.regr_func(x).t()
        # randomly select a single y
        return X, y
