import re


class SimpleTokenizerV1:
    def __init__(self, vocabulary: dict[str, int]):
        self.str_to_int = vocabulary
        self.int_to_str = {i: s for s, i in vocabulary.items()}

    def encode(self, text) -> list[int]:
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text
