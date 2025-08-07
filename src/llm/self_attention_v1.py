import torch.nn as nn
import torch
from torch import Tensor


class SelfAttention_v1(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        query_weight_parameters: nn.Parameter | None = None,
        key_weight_parameters: nn.Parameter | None = None,
        value_weight_parameters: nn.Parameter | None = None,
    ):
        super().__init__()

        self.query_weight_parameters = (
            query_weight_parameters
            if query_weight_parameters is not None
            else nn.Parameter(torch.rand(input_dimensions, output_dimensions))
        )
        self.key_weight_parameters = (
            key_weight_parameters
            if key_weight_parameters is not None
            else nn.Parameter(torch.rand(input_dimensions, output_dimensions))
        )
        self.value_weight_parameters = (
            value_weight_parameters
            if value_weight_parameters is not None
            else nn.Parameter(torch.rand(input_dimensions, output_dimensions))
        )

    def forward(self, input: Tensor) -> Tensor:
        keys = input @ self.key_weight_parameters
        queries = input @ self.query_weight_parameters
        values = input @ self.value_weight_parameters
        attention_scores = queries @ keys.T  # omega
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector
