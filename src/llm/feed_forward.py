from torch import nn, Tensor

from llm.configuration import Configuration
from llm.gaussian_error_linear_unit import GaussianErrorLinearUnit


class FeedForward(nn.Module):
    """A small neural network with GELU activation"""

    def __init__(
        self, configuration: Configuration, hidden_layer_size_multiplier: int = 4
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                configuration.embedding_dimension,
                hidden_layer_size_multiplier * configuration.embedding_dimension,
            ),
            GaussianErrorLinearUnit(),
            nn.Linear(
                hidden_layer_size_multiplier * configuration.embedding_dimension,
                configuration.embedding_dimension,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)
