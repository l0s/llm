from torch import nn, Tensor
from collections.abc import Callable
from llm.gaussian_error_linear_unit import GaussianErrorLinearUnit

from jaxtyping import Float


class DeepNeuralNetwork(nn.Module):
    """
    A deep neural network with multiple layers, each consisting of a linear layer and an activation function
    """

    def __init__(
        self,
        layer_sizes: list[int],
        use_shortcut: bool = True,
        activation: Callable[[], nn.Module]
        | nn.Module = lambda: GaussianErrorLinearUnit(),
    ):
        """
        :param layer_sizes: a list of size _hidden layer count_ + 1. Each represents the output size of a hidden layer, except the last, which represents the output size of the network.
        :param use_shortcut: whether to create residual (skip) connections to mitigate the problem of vanishing gradients
        """
        super().__init__()
        self.use_shortcut = use_shortcut

        layers = [
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                activation if isinstance(activation, nn.Module) else activation(),
            )
            for i in range(0, len(layer_sizes) - 1)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(
        self, inputs: Float[Tensor, "batch_size embedding_dimensions"]
    ) -> Float[Tensor, "batch_size embedding_dimensions"]:
        for layer in self.layers:
            layer_output = layer(inputs)
            inputs = (
                inputs + layer_output
                if self.use_shortcut and inputs.shape == layer_output.shape
                else layer_output
            )
        return inputs
