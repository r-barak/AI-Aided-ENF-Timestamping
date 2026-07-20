import torch
import torch.nn as nn


class LinearFusion(nn.Module):
    """
    Linear fusion module that learns a weighted sum of input scores.

    Args:
        in_dim (int): The number of input scores.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(in_dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, scores):
        return (scores @ self.w) + self.b