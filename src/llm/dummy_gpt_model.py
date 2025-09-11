import torch
import torch.nn as nn

from llm.configuration import Configuration

from torch import Tensor
from jaxtyping import Float


class DummyGptModel(nn.Module):
    def __init__(self, configuration: Configuration):
        super().__init__()

        self.token_embedding = nn.Embedding(
            configuration.vocabulary_size, configuration.embedding_dimension
        )
        self.position_embedding = nn.Embedding(
            configuration.context_length, configuration.embedding_dimension
        )
        self.dropout_embedding = nn.Dropout(configuration.embedding_dropout_rate)
        self.transformer_blocks = nn.Sequential(
            *[
                DummyTransformerBlock(configuration)
                for _ in range(configuration.layer_count)
            ]
        )
        self.final_norm = DummyLayerNorm(configuration.embedding_dimension)
        self.output_head = nn.Linear(
            configuration.embedding_dimension,
            configuration.vocabulary_size,
            bias=configuration.bias,
        )

    def forward(
        self, input_indices: Float[Tensor, "batch_size embedding_dimensions"]
    ) -> Float[Tensor, "batch_size embedding_dimensions"]:
        _batch_size, sequence_length = input_indices.shape
        token_embeddings = self.token_embedding(input_indices)
        position_embeddings = self.position_embedding(
            torch.arange(sequence_length, device=input_indices.device)
        )
        x = token_embeddings + position_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, configuration: Configuration):
        super().__init__()

    def forward(self, input):
        return input


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, 𝜖: float = 1e-5):
        super().__init__()

    def forward(
        self, input: Float[Tensor, "batch_size embedding_dimensions"]
    ) -> Float[Tensor, "batch_size embedding_dimensions"]:
        return input
