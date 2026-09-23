import os

import numpy as np
import regex as re
import pickle
from typing import Iterable, Iterator

# Regex pattern for pre tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:

    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        if not vocab:
            raise ValueError("vocab must not be empty")
        
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens or []

        # Construct reverse vocab dict for encoding from bytes to int
        self.bytes_to_vocab_seq: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # Construct reverse merge dict for looking up pair -> order of creation
        self.merge_order: dict[tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(self.merges)}

        if self.special_tokens:
            for s in self.special_tokens:
                s_b = s.encode("utf-8")
                if s_b not in self.bytes_to_vocab_seq:
                    idx = len(self.vocab)
                    self.vocab[idx] = s_b
                    self.bytes_to_vocab_seq[s_b] = idx

        self.special_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        encoded_output = []

        # Pre-tokenize input text str
        if self.special_pattern:
            parts = re.split(f"({self.special_pattern})", text)
        else: 
            parts = [text]

        for part in parts:
            if not part:
                continue

            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                encoded_output.append(self.bytes_to_vocab_seq[token_bytes])
                continue

            for match in re.finditer(PAT, part):
                token = match.group()
                if not token:
                    continue
                token_bytes = list(bytes([b]) for b in token.encode("utf-8"))
                encoded_output.extend(self.bpe_merge(token_bytes))
        
        return encoded_output


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            if not text:
                continue

            for id in self.encode(text):
                yield id


    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""        
        missing = [i for i in ids if i not in self.vocab]
        if missing:
            raise ValueError("Few input id(s) not found in vocab")
        tokens = b"".join(self.vocab[i] for i in ids) 
        text = tokens.decode("utf-8", errors="replace")
        return text


    def bpe_merge(self, token: list[bytes]) -> list[int]:
        while len(token) > 1:
            best_pair = None
            best_order = None

            prev = token[0]
            for curr in token[1:]:
                pair = (prev, curr)
                order = self.merge_order.get(pair)
                if order is not None and (best_order is None or order < best_order):
                    best_order = order
                    best_pair = pair
                prev = curr

            if best_pair is None:
                break

            i = 0
            l = len(token)
            new_token: list[bytes] = []
            while i < l:
                if i < l - 1 and token[i] == best_pair[0] and token[i+1] == best_pair[1]:
                    new_token.append(best_pair[0] + best_pair[1])
                    i+=2
                else:
                    new_token.append(token[i])
                    i+=1

            token = new_token

        output = [self.bytes_to_vocab_seq[b] for b in token]
        return output


if __name__ == '__main__':
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    merges_path = os.path.join(output_dir, "merges.pkl")
    dataset_path = "data/TinyStoriesV2-GPT4-train.txt"
    encoded_dataset_path = os.path.join(output_dir, "encoded_dataset.npy")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

    with open(dataset_path, encoding="utf-8") as dataset:
        encoded_ids = np.fromiter(tokenizer.encode_iterable(dataset), dtype=np.uint16)
    np.save(encoded_dataset_path, encoded_ids)