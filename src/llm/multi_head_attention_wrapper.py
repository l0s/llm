import torch
from torch import nn, Tensor
from llm.causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        context_length: int,
        dropout_percentage: float,
        num_heads: int,
        bias: bool = False,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                CausalAttention(
                    input_dimensions,
                    output_dimensions,
                    context_length,
                    dropout_percentage,
                    bias,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, input: Tensor) -> Tensor:
        return torch.cat([head(input) for head in self.heads], dim=-1)
