import pytest

from torch import Tensor, arange, tensor, manual_seed, Size
from torch.nn import Embedding
from torch.utils.data import DataLoader


class TestChapter2:
    def test_simple_tokenizer_v1(self, all_tokens: list[str]):
        # given
        from simple_tokenizer_v1 import SimpleTokenizerV1

        vocabulary = {token: integer for integer, token in enumerate(all_tokens)}
        tokenizer = SimpleTokenizerV1(vocabulary)
        text = """"It's the last he painted, you know,"
               Mrs. Gisburn said with pardonable pride."""

        # when
        ids = tokenizer.encode(text)

        # then
        assert ids == [
            1,
            56,
            2,
            850,
            988,
            602,
            533,
            746,
            5,
            1126,
            596,
            5,
            1,
            67,
            7,
            38,
            851,
            1108,
            754,
            793,
            7,
        ]

        # when
        decoded = tokenizer.decode(ids)
        assert (
            decoded
            == '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
        )

    def test_simple_tokenizer_v2(self, all_tokens: list[str]):
        # given
        from simple_tokenizer_v2 import SimpleTokenizerV2

        all_tokens.extend(
            [SimpleTokenizerV2.END_OF_TEXT, SimpleTokenizerV2.UNKNOWN_TOKEN]
        )
        vocabulary = {token: integer for integer, token in enumerate(all_tokens)}
        tokenizer = SimpleTokenizerV2(vocabulary)
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = f" {SimpleTokenizerV2.END_OF_TEXT} ".join([text1, text2])

        # when
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        # then
        assert ids == [
            1131,
            5,
            355,
            1126,
            628,
            975,
            10,
            1130,
            55,
            988,
            956,
            984,
            722,
            988,
            1131,
            7,
        ]
        assert (
            decoded
            == "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
        )

    def test_byte_pair_encoding(self, preprocessed: list[str]):
        # given
        import tiktoken
        from simple_tokenizer_v2 import SimpleTokenizerV2

        tokenizer = tiktoken.get_encoding("gpt2")
        text = (
            f"Hello, do you like tea? {SimpleTokenizerV2.END_OF_TEXT} In the sunlit terraces"
            "of someunknownPlace."
        )

        # when
        ids = tokenizer.encode(text, allowed_special={SimpleTokenizerV2.END_OF_TEXT})
        decoded = tokenizer.decode(ids)

        # then
        assert ids == [
            15496,
            11,
            466,
            345,
            588,
            8887,
            30,
            220,
            50256,
            554,
            262,
            4252,
            18250,
            8812,
            2114,
            1659,
            617,
            34680,
            27271,
            13,
        ]
        assert (
            decoded
            == "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
        )

    def test_sliding_window(self, raw_text: str):
        # given
        dataloader = self.dataloader_v1(
            raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
        )

        # when
        data_iter = iter(dataloader)
        first_batch = next(data_iter)
        second_batch = next(data_iter)

        # then
        input_ids, target_ids = first_batch
        assert input_ids.tolist() == [
            [40, 367, 2885, 1464],
            [1807, 3619, 402, 271],
            [10899, 2138, 257, 7026],
            [15632, 438, 2016, 257],
            [922, 5891, 1576, 438],
            [568, 340, 373, 645],
            [1049, 5975, 284, 502],
            [284, 3285, 326, 11],
        ]
        assert target_ids.tolist() == [
            [367, 2885, 1464, 1807],
            [3619, 402, 271, 10899],
            [2138, 257, 7026, 15632],
            [438, 2016, 257, 922],
            [5891, 1576, 438, 568],
            [340, 373, 645, 1049],
            [5975, 284, 502, 284],
            [3285, 326, 11, 287],
        ]
        input_ids, target_ids = second_batch
        assert input_ids.tolist() == [
            [287, 262, 6001, 286],
            [465, 13476, 11, 339],
            [550, 5710, 465, 12036],
            [11, 6405, 257, 5527],
            [27075, 11, 290, 4920],
            [2241, 287, 257, 4489],
            [64, 319, 262, 34686],
            [41976, 13, 357, 10915],
        ]
        assert target_ids.tolist() == [
            [262, 6001, 286, 465],
            [13476, 11, 339, 550],
            [5710, 465, 12036, 11],
            [6405, 257, 5527, 27075],
            [11, 290, 4920, 2241],
            [287, 257, 4489, 64],
            [319, 262, 34686, 41976],
            [13, 357, 10915, 314],
        ]

    def test_token_embeddings(self):
        """
        2.7 Creating token embeddings
        """
        # given
        input_ids = tensor([2, 3, 5, 1])
        vocabulary_size = 6
        output_dimensions = 3

        # Create a weight matrix with small random values. There will be one
        # row per vocabulary entry and one column per output dimension. These
        # will be optimized later when the LLM is trained.
        manual_seed(123)
        embedding_layer = Embedding(vocabulary_size, output_dimensions)

        # when
        # apply the embedding layer to a token ID
        embedding_vector = embedding_layer(tensor([3]))
        embedding_layer(input_ids)

        # then
        assert embedding_vector.tolist()[0] == embedding_layer.weight.tolist()[3]

    def test_encoding_word_positions(self, raw_text: str):
        """
        2.8 Encoding word positions
        """
        # given

        # assume token ids were created by the BPE tokenizer
        vocabulary_size = 50_257
        output_dimensions = 256
        token_embedding_layer = Embedding(vocabulary_size, output_dimensions)

        max_length = 4
        dataloader = self.dataloader_v1(
            raw_text,
            batch_size=8,
            max_length=max_length,
            stride=max_length,
            shuffle=False,
        )
        data_iter = iter(dataloader)
        inputs, _ = next(data_iter)

        token_embeddings = token_embedding_layer(inputs)

        # absolute embedding approach
        context_length = max_length
        position_embedding_layer = Embedding(context_length, output_dimensions)

        # when
        position_embeddings = position_embedding_layer(arange(context_length))
        input_embeddings = token_embeddings + position_embeddings

        # then
        assert input_embeddings.shape == Size([8, 4, 256])

    def dataloader_v1(
        self,
        raw_text: str,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        import tiktoken
        from gpt_dataset_v1 import GPTDatasetV1

        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(raw_text, tokenizer, max_length, stride)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    @pytest.fixture(scope="class")
    def all_tokens(self, preprocessed: list[str]) -> list[str]:
        return sorted(set(preprocessed))

    @pytest.fixture(scope="class")
    def preprocessed(self, raw_text: str) -> list[str]:
        import re

        result = re.split(r'([.,:;?_!"()\']|--|\s)', raw_text)
        return [item.strip() for item in result if item.strip()]

    @pytest.fixture(scope="class")
    def raw_text(self) -> str:
        import urllib.request
        import os

        if not os.path.exists("the-verdict.txt"):
            url = (
                "https://raw.githubusercontent.com/rasbt/"
                "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
                "the-verdict.txt"
            )
            file_path = "the-verdict.txt"
            urllib.request.urlretrieve(url, file_path)
        with open("the-verdict.txt", "r", encoding="utf-8") as file:
            return file.read()
