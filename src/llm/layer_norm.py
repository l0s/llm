from torch import nn, Tensor
import torch


class LayerNorm(nn.Module):
    """
    A normalization layer that improves the stability and efficiency of neural network training by speeding up the convergence to effective weights
    """

    def __init__(self, embedding_dimension: int, 𝜖: float = 1e-5):
        """
        :param embedding_dimension: the size of each vector representation of a token
        :param 𝜖: (epsilon) a small value to prevent division by zero during normalization
        """
        super().__init__()
        self.𝜖 = 𝜖
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True)
        normalized_input = (inputs - mean) / torch.sqrt(variance + self.ε)
        return (self.scale * normalized_input) + self.shift
