from llm.configuration import Configuration
from llm.feed_forward import FeedForward
from llm.layer_norm import LayerNorm
from llm.multi_head_attention import MultiHeadAttention
from torch import nn, Tensor


class TransformerBlock(nn.Module):
    """
    A neural network that performs multi-head attention, layer normalization, dropout, feed-forward layers, and GELU activations.
    This network handles complex data patterns by using self-attention to identify and analyze relationships between elements and using feed-forward to modify the data individually at each position.
    """

    def __init__(self, configuration: Configuration):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dimensions=configuration.embedding_dimension,
            output_dimensions=configuration.embedding_dimension,
            context_length=configuration.context_length,
            num_heads=configuration.head_count,
            dropout_percentage=configuration.attention_dropout_rate,
            bias=configuration.bias,
        )
        self.feed_forward = FeedForward(configuration)
        self.pre_attention_normalization = LayerNorm(configuration.embedding_dimension)
        self.pre_feed_forward_normalization = LayerNorm(
            configuration.embedding_dimension
        )
        self.dropout_shortcut = nn.Dropout(configuration.shortcut_dropout_rate)

    def forward(self, inputs: Tensor) -> Tensor:
        # attention
        shortcut = inputs
        inputs = self.pre_attention_normalization(inputs)
        inputs = self.attention(inputs)
        inputs = self.dropout_shortcut(inputs)
        inputs = inputs + shortcut

        # feed forward
        shortcut = inputs
        inputs = self.pre_feed_forward_normalization(inputs)
        inputs = self.feed_forward(inputs)
        inputs = self.dropout_shortcut(inputs)
        inputs = inputs + shortcut

        return inputs
