import re
from re import Pattern


class SimpleTokenizerV2:
    END_OF_TEXT: str = "<|endoftext|>"
    UNKNOWN_TOKEN: str = "<|unk|>"
    INPUT_DELIMITER: Pattern = re.compile(r'([,.?_!"()\']|--|\s)')
    OUTPUT_PUNCTUATION: Pattern = re.compile(r'\s+([,.:;?!"()\'])')

    def __init__(self, vocabulary: dict[str, int]):
        self.str_to_int = vocabulary
        self.int_to_str = {i: s for s, i in vocabulary.items()}

    def encode(self, text) -> list[int]:
        preprocessed = re.split(self.INPUT_DELIMITER, text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # flag unknown tokens
        preprocessed = [
            item if item in self.str_to_int else self.UNKNOWN_TOKEN
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(self.OUTPUT_PUNCTUATION, r"\1", text)
        return text
