import sys
import pytest
import torch
from torch import Tensor, nn
from llm.configuration import Configuration
from llm.deep_neural_network import DeepNeuralNetwork
from llm.dummy_gpt_model import DummyGptModel

import tiktoken

from llm.feed_forward import FeedForward
from llm.gaussian_error_linear_unit import GaussianErrorLinearUnit
from llm.generative_pretrained_transformer import GenerativePretrainedTransformerModel
from llm.layer_norm import LayerNorm
from llm.transformer_block import TransformerBlock

GPT_CONFIG_124M = Configuration()


class TestChapter4:
    def test_dummy_gpt_model(self, batch: Tensor):
        # given
        torch.manual_seed(123)
        model = DummyGptModel(GPT_CONFIG_124M)

        # when
        logits = model(batch)

        # then
        assert logits.shape == torch.Size(
            [batch.size(dim=0), 4, GPT_CONFIG_124M.vocabulary_size]
        )

    def test_manual_normalization(self):
        # given
        torch.manual_seed(123)
        batch = torch.randn(2, 5)
        layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
        output = layer(batch)

        # when
        mean = output.mean(dim=-1, keepdim=True)
        variance = output.var(dim=-1, keepdim=True)
        normalized_output = (output - mean) / torch.sqrt(variance)

        # then
        normalized_mean = normalized_output.mean(dim=-1, keepdim=True)
        normalized_variance = normalized_output.var(dim=-1, keepdim=True)
        assert torch.equal(
            torch.round(normalized_mean, decimals=4), torch.zeros([2, 1])
        )
        assert torch.equal(
            torch.round(normalized_variance, decimals=4), torch.ones([2, 1])
        )

    def test_normalization_layer(self):
        # given
        torch.manual_seed(123)
        batch = torch.randn(2, 5)
        layer = LayerNorm(embedding_dimension=batch.size()[-1])

        # when
        output = layer(batch)

        # then
        mean = output.mean(dim=-1, keepdim=True)
        variance = output.var(dim=-1, keepdim=True)
        assert torch.equal(
            torch.round(mean, decimals=4), torch.zeros([batch.size()[0], 1])
        )
        assert torch.equal(
            torch.round(variance, decimals=4), torch.ones([batch.size()[0], 1])
        )

    @pytest.mark.skip("needs to be run interactively to view plot")
    def test_activation_comparison(self):
        # given
        import matplotlib.pyplot as plot

        plot.figure(figsize=(8, 3))
        gelu, relu = GaussianErrorLinearUnit(), nn.ReLU()
        inputs = torch.linspace(-3, 3, 100)
        y_gelu, y_relu = gelu(inputs), relu(inputs)

        # when
        for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
            plot.subplot(1, 2, i)
            plot.plot(inputs, y)
            plot.title(f"{label} activation function")
            plot.xlabel("x")
            plot.ylabel(f"{label}(x)")
            plot.grid(True)
        plot.tight_layout()

        # then
        plot.show()

    def test_feed_forward_network(self):
        # given
        network = FeedForward(GPT_CONFIG_124M)
        inputs = torch.rand(2, 3, GPT_CONFIG_124M.embedding_dimension)

        # when
        outputs = network(inputs)

        # then
        assert outputs.shape == torch.Size([2, 3, GPT_CONFIG_124M.embedding_dimension])

    def print_gradients(self, model: nn.Module, inputs: Tensor):
        for name, gradient in self.calculate_gradients(model, inputs):
            print(f"{name} has a gradient with mean of {gradient}")

    def calculate_gradients(
        self, model: nn.Module, inputs: Tensor
    ) -> list[tuple[str, float]]:
        # execute a forward pass
        output = model(inputs)
        target = torch.tensor([[0.0]])

        # calculate the loss based on how close the target and output are
        loss = nn.MSELoss()
        loss = loss(output, target)

        # backward pass to calculate the gradients
        loss.backward()

        return [
            (name, param.grad.abs().mean().item())
            for name, param in model.named_parameters()
            if "weight" in name and param.grad is not None
        ]

    def test_deep_network_without_shortcuts(self):
        # given
        layer_sizes = [
            3,
            3,
            3,
            3,
            3,
            1,
        ]  # 5-layer neural network that outputs a single value
        input = torch.tensor([[1.0, 0.0, -1.0]])
        torch.manual_seed(123)

        # when
        model = DeepNeuralNetwork(layer_sizes, use_shortcut=False)

        # then
        gradients = torch.round(
            torch.tensor(
                [gradient for _, gradient in self.calculate_gradients(model, input)]
            )
        )
        assert torch.equal(
            gradients,
            torch.round(
                torch.tensor(
                    [
                        0.00020173587836325169,
                        0.0001201116101583466,
                        0.0007152041071094573,
                        0.0013988735154271126,
                        0.005049645435065031,
                    ]
                )
            ),
        )

    def test_deep_network_with_shortcuts(self):
        # given
        layer_sizes = [
            3,
            3,
            3,
            3,
            3,
            1,
        ]  # 5-layer neural network that outputs a single value
        input = torch.tensor([[1.0, 0.0, -1.0]])
        torch.manual_seed(123)

        # when
        model = DeepNeuralNetwork(layer_sizes, use_shortcut=True)

        # then
        gradients = torch.round(
            torch.tensor(
                [gradient for _, gradient in self.calculate_gradients(model, input)]
            ),
            decimals=4,
        )
        assert torch.equal(
            gradients,
            torch.tensor(
                [
                    0.2217,
                    0.2069,
                    0.3290,
                    0.2666,
                    1.3259,
                ]
            ),
        )

    def test_transformer(self):
        # given
        torch.manual_seed(123)
        inputs = torch.rand(2, 4, GPT_CONFIG_124M.embedding_dimension)
        transformer_block = TransformerBlock(GPT_CONFIG_124M)

        # when
        output = transformer_block(inputs)

        # then
        assert inputs.shape == output.shape

    def test_gpt_model(self, batch: Tensor):
        # given
        torch.manual_seed(123)
        model = GenerativePretrainedTransformerModel(GPT_CONFIG_124M)

        # when
        output = model(batch)

        # then
        assert output.shape == torch.Size(
            [batch.size(dim=0), 4, GPT_CONFIG_124M.vocabulary_size]
        )
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 163_009_536
        assert model.token_embedding.weight.shape == torch.Size(
            [GPT_CONFIG_124M.vocabulary_size, GPT_CONFIG_124M.embedding_dimension]
        )
        assert model.output_head.weight.shape == torch.Size(
            [GPT_CONFIG_124M.vocabulary_size, GPT_CONFIG_124M.embedding_dimension]
        )
        # number of trainable parameters considering weight tying
        # GPT-2 reuses the weights from the token embedding layer in its output layer
        total_params_gpt2 = total_params - sum(
            p.numel() for p in model.output_head.parameters()
        )
        # 124e6 parameters is "GPT-2 small"
        assert total_params_gpt2 == 124_412_160

        for transformer_block in model.transformer_blocks:
            if isinstance(transformer_block, TransformerBlock):
                feed_forward_parameters = sum(
                    p.numel() for p in transformer_block.feed_forward.parameters()
                )
                assert feed_forward_parameters == 4_722_432

                multi_head_attention_parameters = sum(
                    p.numel() for p in transformer_block.attention.parameters()
                )
                assert multi_head_attention_parameters == 2_360_064

        total_size_bytes = total_params * sys.getsizeof(
            torch.float32
        )  # note: the book uses 4 bytes per float
        total_size_mb = total_size_bytes / (1_024 * 1_024)
        assert round(total_size_mb) == 13_680

    def test_gpt2_medium(self):
        # given
        torch.manual_seed(123)
        configuration = Configuration(
            embedding_dimension=1_024,
            layer_count=24,
            head_count=16,
        )
        model = GenerativePretrainedTransformerModel(configuration)

        # when
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 406_212_608

    def test_gpt2_large(self):
        # given
        torch.manual_seed(123)
        configuration = Configuration(
            embedding_dimension=1_280,
            layer_count=36,
            head_count=20,
        )
        model = GenerativePretrainedTransformerModel(configuration)

        # when
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 838_220_800

    def test_generate_text(self, tokenizer: tiktoken.Encoding):
        # given
        torch.manual_seed(123)
        start_context = "Hello, I am"
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        model = GenerativePretrainedTransformerModel(GPT_CONFIG_124M)
        # disable training mode
        model.eval()

        # when
        output = self.generate_text_simple(
            model=model,
            index=encoded_tensor,
            max_new_tokens=6,
            context_size=GPT_CONFIG_124M.context_length,
        )

        # then
        assert torch.equal(
            output,
            torch.tensor(
                [[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]]
            ),
        )
        decoded_text = tokenizer.decode(output.squeeze(0).tolist())
        assert decoded_text == "Hello, I am Featureiman Byeswickattribute argue"

    def generate_text_simple(
        self,
        model: nn.Module,
        index: Tensor,
        max_new_tokens: int,
        context_size: int,
    ) -> Tensor:
        """
        :param model:
        :param index: Tensor[(batch, num_tokens)] in the current context
        :param max_new_tokens
        :param context_size
        """
        for _ in range(max_new_tokens):
            # crop the current context if it exceeds the supported context size
            index_condition = index[:, -context_size:]  # todo break out of loop?
            with torch.no_grad():
                logits = model(index_condition)
            # focus only on the last time step
            # (batch, num_tokens, vocabulary size) -> (batch, vocabulary size)
            logits = logits[:, -1, :]
            # (batch, vocabulary size)
            probas = torch.softmax(logits, dim=-1)
            next_index = torch.argmax(probas, dim=-1, keepdim=True)
            # append sampled index to the running sequence
            # (batch, num_tokens + 1)
            index = torch.cat((index, next_index), dim=-1)

        return index

    @pytest.fixture(scope="class")
    def batch(self, tokenizer: tiktoken.Encoding) -> Tensor:
        text1 = "Every effort moves you"
        text2 = "Every day holds a"
        texts = [text1, text2]
        batch = torch.stack(
            [torch.tensor(tokenizer.encode(text)) for text in texts], dim=0
        )
        return batch

    @pytest.fixture(scope="class")
    def tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.get_encoding("gpt2")
