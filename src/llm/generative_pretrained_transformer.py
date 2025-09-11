import torch
from torch import nn, Tensor

from llm.configuration import Configuration
from llm.layer_norm import LayerNorm
from llm.transformer_block import TransformerBlock

from jaxtyping import Float, Int


class GenerativePretrainedTransformerModel(nn.Module):
    """
    A simple GPT model
    """

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
            *[TransformerBlock(configuration) for _ in range(configuration.layer_count)]
        )
        self.final_normalization = LayerNorm(configuration.embedding_dimension)
        self.output_head = nn.Linear(
            configuration.embedding_dimension, configuration.vocabulary_size, bias=False
        )

    def forward(
        self, input_indices: Int[Tensor, "num_batches words_per_batch"]
    ) -> Float[Tensor, "num_batches words_per_batch vocabulary_size"]:
        _batch_size, sequence_length = input_indices.shape
        token_embeddings: Int[Tensor, "num_batches words_per_batch embedding_size"] = (
            self.token_embedding(input_indices)
        )
        position_embeddings: Int[Tensor, "words_per_batch embedding_size"] = (
            self.position_embedding(
                torch.arange(sequence_length, device=input_indices.device)
            )
        )
        inputs: Int[Tensor, "num_batches words_per_batch embedding_size"] = (
            token_embeddings + position_embeddings
        )
        inputs = self.dropout_embedding(inputs)
        inputs = self.transformer_blocks(inputs)
        inputs = self.final_normalization(inputs)
        logits: Float[Tensor, "num_batches words_per_batch vocabulary_size"] = (
            self.output_head(inputs)
        )
        return logits
