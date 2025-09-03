import torch
from torch import Tensor
from torch import nn


class GaussianErrorLinearUnit(nn.Module):
    """
    GELU - a smooth activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return (
            0.5
            * inputs
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (inputs + 0.044715 * torch.pow(inputs, 3))
                )
            )
        )
