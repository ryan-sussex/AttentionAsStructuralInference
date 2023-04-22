"""
Helper classes for simulating regression problems.
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


STD = 0.01

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
                (output_dim, 1)
            ) * 10

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
        self.autoregressive_matrix = (
            torch.randn(
                (input_dim, input_dim)
            ) * 10
        )
        self.sequence_length = sequence_length
        self.ingroup_size = ingroup_size

    def sample_X(self, batch_size: int) -> Tensor:
        outgroup_size = self.sequence_length - self.ingroup_size

        in_group = [ torch.normal(
            mean=torch.ones(batch_size, 1, self.input_dim),
            std=10
        )
        ]
        outgroup = torch.normal(
            mean=torch.ones(batch_size, outgroup_size, self.input_dim),
            std=10
        )

        for i in range(1, self.ingroup_size):
            new = (
                in_group[i - 1] @ self.autoregressive_matrix
                + torch.normal(
                    mean=torch.ones(batch_size, 1, self.input_dim),
                    std=STD
                )
            )
            in_group.append(new)

        in_group.append(outgroup)
        X = torch.cat(in_group, dim=1)
        return X

    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            X = self.sample_X(batch_size)
            x = X[:,0, -1]
            for i in range(1, self.ingroup_size):
                x = x + X[:,i, -1]

            x = x[:, None]
            y = self.regr_func(x).t()
        # randomly select a single y
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
            print(placeholder.shape)

            placeholder = placeholder.to(torch.int64)
            # placeholder.int_repr()
            x = torch.gather(X, 1, placeholder.to())[:, 0]
            y = self.regr_func(x).t()
        # randomly select a single y
        return X, y


class TwoVarSequenceRegression(SequenceRegression):
    """
    This class randomly selects the index to regress against.
    """

    # def __init__(
    #     self,
    #     input_dim: int,
    #     output_dim: int,
    #     sequence_length: int = 10,
    #     weight_matrix: Optional[Tensor] = None
    # ):
    #     super().__init__(input_dim, output_dim, weight_matrix)
    #     self.sequence_length = sequence_length
    #     self.categorical = Categorical(torch.ones(sequence_length))

    # def sample_X(self, batch_size: int) -> Tensor:
    #     return torch\
    #         .normal(
    #             mean=torch.ones(
    #                 batch_size, self.sequence_length, self.input_dim),
    #             std=0.01
    #         )

    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            X = self.sample_X(batch_size)
            selection_1 = self.categorical.sample((batch_size,))
            selection_1 = selection_1[:, None]
            placeholder_1 = torch.zeros(X.shape)
            placeholder_1[:, 0] = selection_1
            placeholder_1 = placeholder_1.to(torch.int64)
            
            selection_2 = self.categorical.sample((batch_size,))
            selection_2 = selection_2[:, None]
            placeholder_2 = torch.zeros(X.shape)
            placeholder_2[:, 0] = selection_1
            placeholder_2 = placeholder_2.to(torch.int64)            # placeholder.int_repr()
            x_1 = torch.gather(X, 1, placeholder_1.to())[:, 0]
            x_2 = torch.gather(X, 1, placeholder_2.to())[:, 0]
            x = x_1 + x_2
            y = self.regr_func(x).t()
        # randomly select a single y
        return X, y