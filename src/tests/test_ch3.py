import pytest

import torch
from torch import Tensor
from torch import nn, inf

from llm.self_attention_v1 import SelfAttention_v1
from llm.self_attention_v2 import SelfAttention_v2
from llm.causal_attention import CausalAttention
from llm.multi_head_attention_wrapper import MultiHeadAttentionWrapper


class TestChapter3:
    def test_calculate_attention_scores_for_single_query(self, inputs: Tensor):
        # given
        # query the second input token, "journey"
        query: Tensor = inputs[1]

        # when
        # calculate the attention scores using the dot product
        # the score measures alignment or similarity between
        # the query token and each other token
        attention_scores: Tensor = torch.empty(
            inputs.shape[0]
        )  # vector with one element per inputs row
        for index, input in enumerate(inputs):
            attention_scores[index] = torch.dot(input, query)

        # then
        assert torch.equal(
            torch.round(attention_scores, decimals=4),
            torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]),
        )

    def test_calculate_attention_weights_simple(self, inputs: Tensor):
        # given
        query: Tensor = inputs[1]
        attention_scores = self.attention_scores(inputs, query)

        # when
        attention_weights = attention_scores / attention_scores.sum()

        # then
        assert abs(attention_weights.sum() - 1.0) <= 0.0001

    def test_calculate_attention_weights_softmax_naïve(self, inputs: Tensor):
        # given
        def softmax_naïve(x: Tensor) -> Tensor:
            return torch.exp(x) / torch.exp(x).sum(dim=0)

        query: Tensor = inputs[1]
        attention_scores = self.attention_scores(inputs, query)

        # when
        attention_weights = softmax_naïve(attention_scores)

        # then
        assert abs(attention_weights.sum() - 1.0) <= 0.0001

    def test_calculate_attention_weights_softmax(self, inputs: Tensor):
        # given
        query: Tensor = inputs[1]
        attention_scores = self.attention_scores(inputs, query)

        # when
        attention_weights = torch.softmax(attention_scores, dim=0)

        # then
        assert abs(attention_weights.sum() - 1.0) <= 0.0001

    def test_calculate_context_vector(self, inputs: Tensor):
        # given
        query: Tensor = inputs[1]
        attention_weights = self.attention_weights(inputs, query)

        # when
        context_vector = torch.zeros(query.shape)
        for index, input in enumerate(inputs):
            context_vector += attention_weights[index] * input

        # then
        assert torch.equal(
            torch.round(context_vector, decimals=4),
            torch.tensor([0.4419, 0.6515, 0.5683]),
        )

    def test_calculate_context_vectors(self, inputs: Tensor):
        # given
        # when
        # slow version with nested loops
        # attention_scores = torch.empty(6, 6)
        # for i, query in enumerate(inputs):
        #     for j, input in enumerate(inputs):
        #         attention_scores[ i, j ] = torch.dot(query, input)

        attention_scores = inputs @ inputs.T
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vectors = attention_weights @ inputs

        # then
        assert torch.equal(
            torch.round(context_vectors, decimals=4),
            torch.tensor(
                [
                    [0.4421, 0.5931, 0.5790],
                    [0.4419, 0.6515, 0.5683],
                    [0.4431, 0.6496, 0.5671],
                    [0.4304, 0.6298, 0.5510],
                    [0.4671, 0.5910, 0.5266],
                    [0.4177, 0.6503, 0.5645],
                ]
            ),
        )

    def test_calculate_context_vectors_with_weight_parameters(self, inputs: Tensor):
        # given
        query_index = 1
        query = inputs[query_index]

        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2

        # initialize weight parameters that will be optimized during training
        torch.manual_seed(123)
        query_weight_parameters = torch.nn.Parameter(
            torch.rand(input_dimensions, output_dimensions), requires_grad=False
        )
        key_weight_parameters = torch.nn.Parameter(
            torch.rand(input_dimensions, output_dimensions), requires_grad=False
        )
        value_weight_parameters = torch.nn.Parameter(
            torch.rand(input_dimensions, output_dimensions), requires_grad=False
        )

        weighted_query = query @ query_weight_parameters
        # weighted_key = query @ key_weight_parameters
        # weighted_value = query @ value_weight_parameters

        keys = inputs @ key_weight_parameters
        values = inputs @ value_weight_parameters

        # query_key = keys[query_index]
        # attention_score = weighted_query.dot(query_key)
        attention_scores = weighted_query @ keys.T
        key_dimensions = keys.shape[-1]
        attention_weights = torch.softmax(
            attention_scores / key_dimensions**0.5, dim=-1
        )

        # when
        context_vector = attention_weights @ values

        # then
        assert torch.equal(
            torch.round(context_vector, decimals=4), torch.tensor([0.3061, 0.8210])
        )

    def test_self_attention_v1(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2

        torch.manual_seed(123)

        # when
        self_attention_v1 = SelfAttention_v1(input_dimensions, output_dimensions)

        # then
        assert torch.equal(
            torch.round(self_attention_v1(inputs), decimals=4),
            torch.tensor(
                [
                    [0.2996, 0.8053],
                    [0.3061, 0.8210],
                    [0.3058, 0.8203],
                    [0.2948, 0.7939],
                    [0.2927, 0.7891],
                    [0.2990, 0.8040],
                ]
            ),
        )

    def test_self_attention_v2(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2

        torch.manual_seed(789)

        # when
        self_attention_v2 = SelfAttention_v2(input_dimensions, output_dimensions)

        # then
        result = self_attention_v2(inputs)
        assert torch.equal(
            torch.round(result, decimals=4),
            torch.tensor(
                [
                    [-0.0739, 0.0713],
                    [-0.0748, 0.0703],
                    [-0.0749, 0.0702],
                    [-0.0760, 0.0685],
                    [-0.0763, 0.0679],
                    [-0.0754, 0.0693],
                ]
            ),
        )

    def test_v1_v2_comparison(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2

        torch.manual_seed(123)

        # when
        v2 = SelfAttention_v2(input_dimensions, output_dimensions)
        v1 = SelfAttention_v1(
            input_dimensions,
            output_dimensions,
            query_weight_parameters=nn.Parameter(v2.query_weight_parameters.weight.T),
            key_weight_parameters=nn.Parameter(v2.key_weight_parameters.weight.T),
            value_weight_parameters=nn.Parameter(v2.value_weight_parameters.weight.T),
        )

        # then
        assert torch.equal(v1(inputs), v2(inputs))

    def test_apply_causal_attention_mask(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2
        torch.manual_seed(789)

        sa_v2 = SelfAttention_v2(input_dimensions, output_dimensions)

        queries = sa_v2.query_weight_parameters(inputs)
        keys = sa_v2.key_weight_parameters(inputs)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # when
        # create a mask to hide future values
        context_length = attention_scores.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))

        # mask future values
        masked_simple = attention_weights * mask_simple

        # re-normalize the attention weights to sum to 1 again
        row_sums = masked_simple.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_simple / row_sums

        # then
        assert torch.equal(
            torch.round(masked_simple_norm, decimals=4),
            torch.tensor(
                [
                    [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
                    [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
                    [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
                    [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529],
                ]
            ),
        )

    def test_optimized_causal_attention_mask(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2
        torch.manual_seed(789)

        sa_v2 = SelfAttention_v2(input_dimensions, output_dimensions)

        queries = sa_v2.query_weight_parameters(inputs)
        keys = sa_v2.key_weight_parameters(inputs)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # when
        # create a mask to hide future values
        context_length = attention_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

        # mask future values
        masked = attention_scores.masked_fill(mask.bool(), -inf)
        attention_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)

        # then
        assert torch.equal(
            torch.round(attention_weights, decimals=4),
            torch.tensor(
                [
                    [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
                    [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
                    [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
                    [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529],
                ]
            ),
        )

    def test_causal_attention(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = inputs.shape[1]
        # output embedding size
        output_dimensions = 2
        batch = torch.stack((inputs, inputs), dim=0)

        torch.manual_seed(123)
        context_length = batch.shape[1]
        attention = CausalAttention(
            input_dimensions, output_dimensions, context_length, 0.0
        )

        # when
        context_vectors = attention(batch)

        # then
        assert torch.equal(
            torch.round(context_vectors, decimals=4),
            torch.tensor(
                [
                    [
                        [-0.4519, 0.2216],
                        [-0.5874, 0.0058],
                        [-0.6300, -0.0632],
                        [-0.5675, -0.0843],
                        [-0.5526, -0.0981],
                        [-0.5299, -0.1081],
                    ],
                    [
                        [-0.4519, 0.2216],
                        [-0.5874, 0.0058],
                        [-0.6300, -0.0632],
                        [-0.5675, -0.0843],
                        [-0.5526, -0.0981],
                        [-0.5299, -0.1081],
                    ],
                ]
            ),
        )

    def test_multi_head_attention(self, inputs: Tensor):
        # given
        # input embedding size
        input_dimensions = 3
        # output embedding size
        output_dimensions = 2
        batch = torch.stack((inputs, inputs), dim=0)

        torch.manual_seed(123)
        context_length = batch.shape[1]  # the number of tokens

        attention = MultiHeadAttentionWrapper(
            input_dimensions, output_dimensions, context_length, 0.0, num_heads=2
        )

        # when
        context_vectors = attention(batch)

        # then
        assert torch.equal(
            torch.round(context_vectors, decimals=4),
            torch.tensor(
                [
                    [
                        [-0.4519, 0.2216, 0.4772, 0.1063],
                        [-0.5874, 0.0058, 0.5891, 0.3257],
                        [-0.6300, -0.0632, 0.6202, 0.3860],
                        [-0.5675, -0.0843, 0.5478, 0.3589],
                        [-0.5526, -0.0981, 0.5321, 0.3428],
                        [-0.5299, -0.1081, 0.5077, 0.3493],
                    ],
                    [
                        [-0.4519, 0.2216, 0.4772, 0.1063],
                        [-0.5874, 0.0058, 0.5891, 0.3257],
                        [-0.6300, -0.0632, 0.6202, 0.3860],
                        [-0.5675, -0.0843, 0.5478, 0.3589],
                        [-0.5526, -0.0981, 0.5321, 0.3428],
                        [-0.5299, -0.1081, 0.5077, 0.3493],
                    ],
                ]
            ),
        )

    def test_multi_head_attention_with_weight_splits(self, inputs: Tensor):
        # given
        from llm.multi_head_attention import MultiHeadAttention

        torch.manual_seed(123)
        batch = torch.stack((inputs, inputs), dim=0)
        _batch_size, context_length, input_dimensions = batch.shape
        output_dimensions = 2

        attention = MultiHeadAttention(
            input_dimensions, output_dimensions, context_length, 0.0, num_heads=2
        )

        # when
        context_vector = attention(batch)

        # then
        assert torch.equal(
            torch.round(context_vector, decimals=4),
            torch.tensor(
                [
                    [
                        [0.3190, 0.4858],
                        [0.2943, 0.3897],
                        [0.2856, 0.3593],
                        [0.2693, 0.3873],
                        [0.2639, 0.3928],
                        [0.2575, 0.4028],
                    ],
                    [
                        [0.3190, 0.4858],
                        [0.2943, 0.3897],
                        [0.2856, 0.3593],
                        [0.2693, 0.3873],
                        [0.2639, 0.3928],
                        [0.2575, 0.4028],
                    ],
                ]
            ),
        )

    def attention_weights(self, inputs: Tensor, query: Tensor) -> Tensor:
        return torch.softmax(self.attention_scores(inputs, query), dim=0)

    def attention_scores(self, inputs: Tensor, query: Tensor) -> Tensor:
        result: Tensor = torch.empty(inputs.shape[0])
        for index, input in enumerate(inputs):
            result[index] = torch.dot(input, query)
        return result

    @pytest.fixture(scope="class")
    def inputs(self) -> Tensor:
        return torch.tensor(
            [
                [0.43, 0.15, 0.89],  # Your     (x^1)
                [0.55, 0.87, 0.66],  # journey  (x^2)
                [0.57, 0.85, 0.64],  # starts   (x^3)
                [0.22, 0.58, 0.33],  # with     (x^4)
                [0.77, 0.25, 0.10],  # one      (x^5)
                [0.05, 0.80, 0.55],  # step     (x^6)
            ]
        )
