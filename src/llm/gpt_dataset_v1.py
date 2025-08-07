import torch

from torch.utils.data import Dataset
from torch import Tensor
from tiktoken import Encoding


class GPTDatasetV1(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int):
        self.input_ids: list[Tensor] = []
        self.target_ids: list[Tensor] = []

        token_ids = tokenizer.encode(txt)

        chunks = (
            (token_ids[i : i + max_length], token_ids[i + 1 : i + max_length + 1])
            for i in range(0, len(token_ids) - max_length, stride)
        )
        ids = (
            (torch.tensor(input_chunk), torch.tensor(output_chunk))
            for input_chunk, output_chunk in chunks
        )
        self.input_ids, self.target_ids = map(list, zip(*ids))

    def __len__(self) -> int:
        """
        Returns: the number of items in the dataset
        """
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Returns: an input/target record from the dataset
        """
        return self.input_ids[index], self.target_ids[index]
