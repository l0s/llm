from torch import nn, Tensor
import torch


class MultiHeadAttention(nn.Module):
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

        assert output_dimensions % num_heads == 0, (
            "output_dimensions must be divisible by num_heads"
        )

        self.output_dimensions = output_dimensions
        self.num_heads = num_heads

        # reduce the projection dimensions to match the desired output dimensions
        self.head_dimensions = output_dimensions // num_heads

        self.query_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.key_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )
        self.value_weight_parameters = nn.Linear(
            input_dimensions, output_dimensions, bias=bias
        )

        # use a Linear layer to combine head outputs
        self.output_projection = nn.Linear(output_dimensions, output_dimensions)

        self.dropout = nn.Dropout(dropout_percentage)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, input: Tensor) -> Tensor:
        batch_size, num_tokens, _input_dimensions = input.shape
        keys = self.key_weight_parameters(input)
        queries = self.query_weight_parameters(input)
        values = self.value_weight_parameters(input)

        # split the matrix by adding a num_heads dimension
        # then unroll the last dimension
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dimensions)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dimensions
        )
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dimensions
        )

        # transform from (batch_size, num_tokens, num_heads,  head_dimensions)
        #             to (batch_size, num_heads,  num_tokens, head_dimensions)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute dot product for each head
        attention_scores = queries @ keys.transpose(2, 3)

        # truncate masks to the number of tokens
        # and fill attention scores
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # shape: (batch_size, num_tokens, num_heads, head_dimensions)
        context_vector = (attention_weights @ values).transpose(1, 2)
        # combine heads
        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.output_dimensions
        )
        context_vector = self.output_projection(context_vector)
        return context_vector
