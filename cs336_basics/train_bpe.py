#Byte-Pair Encoding Tokenizer (BPE) implementation

import cProfile
import pickle
import pstats

import regex as re
import os
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

# Regex pattern for pre tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize_chunk(args) -> dict[tuple[bytes, ...], int]:
    input_path, start, end, special_tokens, pattern = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:    
        delimiter = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
        parts = re.split(delimiter, chunk)
    else:
        parts = [chunk]

    count = {}
    # Store the counts of each pre-token where pre-token is a bytestring of individual characters of the entire word
    # eg: {('h','e','l','l','o'): 5, ...}
    for part in parts:
        for match in re.finditer(pattern, part):
            token = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            count[token] = count.get(token, 0) + 1

    return count
    

def pre_tokenize_file(
        input_path: str,
        special_tokens: list[str],
        pattern: str,
        num_processes: int = 4
) -> dict[tuple[bytes, ...], int]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    
    total_count = {}

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = [(input_path, start, end, special_tokens, pattern) for start, end in zip(boundaries[:-1], boundaries[1:])]

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        with Pool(num_processes) as pool:
            results = pool.map(pre_tokenize_chunk, chunks)

        for result in results: #dict[tuple[bytes, ...], int]
            for token, count in result.items():
                total_count[token] = total_count.get(token, 0) + count

    return total_count


def get_pairs(
        token_counts: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    pairs = {}
    for token, count in token_counts.items():
        for pair in zip(token, token[1:]):
            pairs[pair] = pairs.get(pair, 0) + count

    return pairs


def merge(
        token_counts: dict[tuple[bytes, ...], int],
        pairs: dict[tuple[bytes, bytes], int],
        merge_pair: tuple[bytes, bytes]
) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int]]:
    # Find and merge merge_pair in token_counts
    merged_token_counts = {}

    for token, count in token_counts.items():
        # If no action, just add token to new dict
        if merge_pair[0] not in token or merge_pair[1] not in token:
            merged_token_counts[token] = merged_token_counts.get(token, 0) + count
            continue

        # Merge pair, and check for adjacent bytes to the merged pair and update pairs dict accordingly
        new_token = []
        idx = 0
        while idx < len(token):
            if(idx < len(token) - 1 and token[idx] == merge_pair[0] and token[idx+1] == merge_pair[1]):
                # Decrease the frequency of old pairs, Add/Increase the frequency of new merged pairs
                if new_token:
                    old_pair = (new_token[-1], merge_pair[0])
                    pairs[old_pair] = pairs.get(old_pair, 0) - count
                    if(pairs[old_pair] <= 0):
                        del pairs[old_pair]

                    new_pair = (new_token[-1], merge_pair[0] + merge_pair[1])
                    pairs[new_pair] = pairs.get(new_pair, 0) + count

                if idx + 2 < len(token):
                    old_pair = (merge_pair[1], token[idx+2])
                    pairs[old_pair]  = pairs.get(old_pair, 0) - count
                    if(pairs[old_pair] <= 0):
                        del pairs[old_pair]
                    new_pair = (merge_pair[0] + merge_pair[1], token[idx+2])
                    pairs[new_pair] = pairs.get(new_pair, 0) + count

                pairs[merge_pair] = pairs.get(merge_pair, 0) - count
                if(pairs[merge_pair] <= 0):
                    del pairs[merge_pair]

                new_token.append(merge_pair[0] + merge_pair[1])
                idx += 2
            else:
                new_token.append(token[idx])
                idx += 1

        merged_token_counts[tuple(new_token)] = merged_token_counts.get(tuple(new_token), 0) + count
 
    return merged_token_counts, pairs


def bpe_tokenizer_model(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a input file path, vocab size and special tokens, train a BPE tokenizer model and return vocab and merges list
    """
    min_vocab_size = 256 + len(special_tokens)
    if vocab_size < min_vocab_size:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be >= 256 + len(special_tokens) ({min_vocab_size})"
        )
    
    # Pre-tokenize
    token_counts = pre_tokenize_file(input_path, special_tokens, PAT)

    # Get the consecutive pairs and their counts
    pairs = get_pairs(token_counts)

    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()
    num_merges = vocab_size - len(vocab)
    merges = []

    # Merge the most frequent pair, in case of tie, take the lexicographically greater pair
    for i in range(num_merges):
        # compare by count first (pairs[pair] --> count), if tied use the second factor (pair itself) for getting lexicographically greater pair
        top_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
        token_counts, pairs = merge(token_counts, pairs, top_pair)
        merges.append(top_pair)
        vocab[len(vocab)] = top_pair[0] + top_pair[1]

    return vocab, merges

if __name__ == "__main__":
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = bpe_tokenizer_model(input_path, vocab_size, special_tokens)
    profiler.disable()
    profiler.dump_stats("profile_output_owt.prof")
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab_owt.pkl")
    merges_path = os.path.join(output_dir, "merges_owt.pkl")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    assert len(vocab) == 32000
    assert len(merges) == 32000 - 256 - 1  # minus 256 base bytes, minus special tokens

    # All 256 raw bytes should be present as single-byte tokens
    for i in range(256):
        assert vocab[i] == bytes([i])

    # Special token should be present
    assert b"<|endoftext|>" in vocab.values()

    # No duplicate vocab entries
    assert len(set(vocab.values())) == len(vocab)