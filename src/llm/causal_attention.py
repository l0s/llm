import torch
from torch import nn
from torch import Tensor


class CausalAttention(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        context_length: int,
        dropout_percentage: float,
        bias: bool = False,
    ):
        super().__init__()

        self.output_dimensions = output_dimensions
        self.query_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.key_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.value_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.dropout = nn.Dropout(dropout_percentage)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        _, num_tokens, input_dimensions = input.shape
        keys = self.key_weight_parameters(input)
        queries = self.query_weight_parameters(input)
        values = self.value_weight_parameters(input)

        # transpose dimensions 1 and 2
        # keeping the batch dimension at the first position
        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        context_vector = attention_weights @ values
        return context_vector
