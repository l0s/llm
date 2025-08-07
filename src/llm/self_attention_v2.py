import torch
from torch import nn
from torch import Tensor


class SelfAttention_v2(nn.Module):
    def __init__(self, input_dimensions: int, output_dimensions: int, bias=False):
        super().__init__()
        self.query_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.key_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.value_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )

    def forward(self, input: Tensor) -> Tensor:
        keys = self.key_weight_parameters(input)
        queries = self.query_weight_parameters(input)
        values = self.value_weight_parameters(input)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector
