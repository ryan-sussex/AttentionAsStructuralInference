"""
Helper classes for simulating regression problems.
"""
from typing import Optional
import torch
import torch.nn.functional as F


class Regression():

    def __init__(self, input_dim, output_dim, weight_matrix: Optional = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_matrix is None:
            self.weight_matrix = torch.rand(
                (output_dim, input_dim)
            )
    
    def sample(self, batch_size: int = 1):
        with torch.no_grad():
            X = torch.normal(mean=torch.ones(batch_size, self.input_dim), std=0.01)
            y = F.linear(self.weight_matrix, X)
        return X, y