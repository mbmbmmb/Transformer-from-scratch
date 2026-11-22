"""Byte-level BPE: training (serial and multiprocess pre-tokenization) and the Tokenizer.

CLI: tokenizes the TinyStories train/valid text with a newly trained tokenizer.
    uv run python -m cs336_basics.train_bpe --config configs/bpe.yaml
Writes vocab.pkl, merges.pkl, train.bin, valid.bin (uint16 token ids) to paths.output_dir.
"""

import argparse
import os
import pickle
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import regex as re

from cs336_basics.pretokenization import find_chunk_boundaries
from cs336_basics.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# GPT-2 pre-tokenization pattern
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# ---------------------------------------------------------------- merge loop helpers


def get_all_pairs(tup):
    return [tup[i : i + 2] for i in range(len(tup) - 1)]


def get_token_pair_count(word_count):
    """Count every adjacent token pair, weighted by word frequency."""
    if len(word_count) == 0:
        return Counter()
    token_pairs_count = {}
    for word in word_count:
        if len(word) < 2:
            continue
        freq = word_count[word]
        for i in range(len(word) - 1):
            pair = tuple(word[i : i + 2])
            if pair in token_pairs_count:
                token_pairs_count[pair] += freq
            else:
                token_pairs_count[pair] = freq
    return Counter(token_pairs_count)


def get_top_pair_lex(token_pairs_count, vocab):
    """Most frequent pair; ties broken by the lexicographically greatest (bytes, bytes)."""
    if len(token_pairs_count) == 0:
        return None
    max_val = max(token_pairs_count.values())
    max_idx = [c for c, v in token_pairs_count.items() if v == max_val]
    top_pair = max(max_idx, key=lambda x: (vocab[x[0]], vocab[x[1]]))
    return top_pair


def replace_pair_in_tup(tup, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `tup` with `new_id`."""
    new_tup = []
    replace = False
    for i in range(len(tup) - 1):
        if replace:
            replace = False
            continue
        elif tup[i : i + 2] == pair:
            new_tup.append(new_id)
            replace = True
        else:
            new_tup.append(tup[i])
    # append the last token unless it was consumed by a merge
    if not replace:
        new_tup.append(tup[-1])
    return tuple(new_tup)


def get_keys_to_replace(word_count, top_pair, new_id):
    """Map each word containing `top_pair` to its merged form."""
    keys_to_replace = {}
    for key in word_count:
        if top_pair[0] not in key or top_pair[1] not in key or len(key) < 2:
            continue
        else:
            new_key = replace_pair_in_tup(key, top_pair, new_id)
            if new_key != key:
                keys_to_replace[key] = new_key
    return keys_to_replace


def update_pair_count(token_pairs_count, word_count, keys_to_replace):
    """Update pair counts only for the words that changed (delta of their pairs)."""
    for old_key in keys_to_replace:
        old_key_count = word_count[old_key]
        new_key = keys_to_replace[old_key]
        old_pairs = get_all_pairs(old_key)  # e.g. [(32, 116), (116, 111)]
        new_pairs = get_all_pairs(new_key)  # e.g. [(258, 111)]
        old_pairs_counts = Counter(old_pairs)
        delta = Counter(new_pairs)
        delta.subtract(old_pairs_counts)

        for pair, d in delta.items():
            token_pairs_count[pair] += d * old_key_count

        if new_key not in word_count:
            word_count[new_key] = word_count[old_key]
        else:
            word_count[new_key] += word_count[old_key]
        del word_count[old_key]
    return token_pairs_count, word_count


# ---------------------------------------------------------------- pre-tokenization


def get_chunk_word_count(chunk, special_tokens):
    """Split on special tokens (they never take part in merges), pre-tokenize with PAT,
    and count each pre-token as a tuple of byte values."""
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_specials)
    docs = re.split(pattern, chunk)
    wc = Counter()
    for d in docs:
        wc.update(tuple(s.encode("utf-8")) for s in PAT.findall(d))
    return wc


def pretokenize_chunk(args, input_path, special_tokens):
    input_path, start, end = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return get_chunk_word_count(chunk, special_tokens)


def parallel_pretokenize(input_path, num_processes=10, special_tokens=None):
    """Pre-tokenize file chunks (split at the first special token) in a process pool."""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    boundary_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, boundary_token)

    tasks = list(zip([input_path] * (len(boundaries) - 1), boundaries[:-1], boundaries[1:]))
    worker = partial(pretokenize_chunk, input_path=input_path, special_tokens=special_tokens)

    with Pool(num_processes) as pool:
        per_chunk = pool.map(worker, tasks)

    word_count = Counter()
    for c in per_chunk:
        word_count.update(c)
    return word_count


# ---------------------------------------------------------------- BPE training


def train_bpe(input_path, vocab_size=500, special_tokens=["<|endoftext|>"], **kwargs):
    """Single-process BPE training. Returns (vocab: dict[int, bytes], merges: list[(bytes, bytes)])."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {n: bytes([n]) for n in range(256)}
    for t in range(len(special_tokens)):
        vocab[t + 256] = special_tokens[t].encode("utf-8")
    with open(input_path, encoding="utf-8") as f:
        lines = f.read()

    # use the special tokens as hard boundaries: split them out of the text so they never merge
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_specials)
    docs = re.split(pattern, lines)
    # further split each doc into pre-tokens with the GPT-2 pattern
    docs_splits = [re.findall(PAT, d) for d in docs]

    # count pre-token frequency, keyed by the tuple of its utf-8 byte values
    word_table = Counter()
    for doc in docs_splits:
        word_table.update(doc)
    word_count = Counter({tuple(s.encode("utf-8")): word_table[s] for s in word_table})

    # repeatedly merge the top pair and add it to the vocab as a new token
    token_pairs_count = get_token_pair_count(word_count)
    merges = []
    while len(vocab) < vocab_size:
        top_pair = get_top_pair_lex(token_pairs_count, vocab)
        if top_pair is None or token_pairs_count[top_pair] <= 0:
            break
        merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        keys_to_replace = get_keys_to_replace(word_count, top_pair, new_id)
        token_pairs_count, word_count = update_pair_count(token_pairs_count, word_count, keys_to_replace)
    return vocab, merges


def train_bpe_parallel(input_path, vocab_size=500, special_tokens=["<|endoftext|>"], num_processes=10, **kwargs):
    """Same as train_bpe, but pre-tokenization runs in a process pool over file chunks."""
    vocab = {n: bytes([n]) for n in range(256)}
    for t in range(len(special_tokens)):
        vocab[t + 256] = special_tokens[t].encode("utf-8")
    word_count = parallel_pretokenize(input_path, num_processes=num_processes, special_tokens=special_tokens)

    # repeatedly merge the top pair and add it to the vocab as a new token
    token_pairs_count = get_token_pair_count(word_count)
    merges = []
    while len(vocab) < vocab_size:
        top_pair = get_top_pair_lex(token_pairs_count, vocab)
        if top_pair is None or token_pairs_count[top_pair] <= 0:
            break
        merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        keys_to_replace = get_keys_to_replace(word_count, top_pair, new_id)
        token_pairs_count, word_count = update_pair_count(token_pairs_count, word_count, keys_to_replace)
    return vocab, merges


# ---------------------------------------------------------------- tokenizer (encode / decode)


def save_pkl(param, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(param, f)


def read_pkl(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as f:
        return pickle.load(f)


def split_special(lines, special_tokens, keep_split=False):
    """Split text on special tokens (longest first). keep_split=True keeps the tokens
    as separate list items, so even indices are plain text and odd indices are special tokens."""
    if not special_tokens:
        return [lines]
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in sorted_specials)
    if keep_split:
        docs = re.split("(" + pattern + ")", lines)
    else:
        docs = re.split(pattern, lines)
    return docs


def ecd(s):
    return s.encode("utf-8")


def string_to_byte_tup(string):
    """'hi' -> (b'h', b'i')"""
    return tuple([bytes([i]) for i in ecd(string)])


def get_merge_pair(encoded_tup, merges_rank):
    """Find the adjacent pair with the lowest merge rank and every index where it occurs."""
    min_pair, min_rank, merge_idx = None, float("inf"), None
    i = 0
    while i < len(encoded_tup):
        pair = encoded_tup[i : i + 2]
        if pair in merges_rank:
            cur_rank = merges_rank[pair]
            if cur_rank < min_rank:
                min_pair = pair
                min_rank = cur_rank
                merge_idx = [i]
            elif cur_rank == min_rank:
                merge_idx.append(i)
        i += 1
    return min_pair, min_rank, merge_idx


def merge_single_idx(encoded_tup, idx):
    merged = encoded_tup[idx] + encoded_tup[idx + 1]
    return encoded_tup[:idx] + (merged,) + encoded_tup[idx + 2 :]


def merge_tuple(encoded_tup, merge_idx):
    """Merge every (idx, idx + 1) for idx in merge_idx.
    Edge case: if idx and idx + 1 are both listed, only idx is merged (idx + 1 is skipped)."""
    tmp = encoded_tup
    merged = 0
    prev = None
    for i in merge_idx:
        if prev is not None and prev == i - 1:
            continue
        # shift by the number of merges already applied to the left
        tmp = merge_single_idx(tmp, i - merged)
        prev = i
        merged += 1
    return tmp


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self.merges_rank = {merges[i]: i for i in range(len(merges))}
        # pre-token string -> merged byte tuple
        self.merge_cache = {}
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # add special tokens that are missing from the vocab
        if self.special_tokens is not None:
            for token in special_tokens:
                b_token = token.encode("utf-8")
                if b_token not in self.inv_vocab:
                    ls = len(self.vocab)
                    self.vocab[ls] = b_token
                    self.inv_vocab[b_token] = ls

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        vocab = read_pkl(vocab_filepath)
        merges = read_pkl(merges_filepath)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str, use_cache=True):
        output_encoded_lis = []
        text_lis = split_special(text, self.special_tokens, keep_split=True)
        for i in range(len(text_lis)):
            if i % 2 == 0:
                # plain text: pre-tokenize, then apply merges in rank order within each pre-token
                doc_split = re.findall(self.PAT, text_lis[i])
                for j in range(len(doc_split)):
                    if use_cache and doc_split[j] in self.merge_cache:
                        encoded_tup = self.merge_cache[doc_split[j]]
                    else:
                        encoded_tup = string_to_byte_tup(doc_split[j])
                        while True:
                            min_pair, min_rank, merge_idx = get_merge_pair(encoded_tup, self.merges_rank)
                            if min_pair is None:
                                break
                            encoded_tup = merge_tuple(encoded_tup, merge_idx)
                        self.merge_cache[doc_split[j]] = encoded_tup
                    for token in encoded_tup:
                        output_encoded_lis.append(self.inv_vocab[token])
            else:
                # special token: get its id directly from the vocab
                output_encoded_lis.append(self.inv_vocab[text_lis[i].encode("utf-8", errors="replace")])
        return output_encoded_lis

    def encode_iterable(self, iterable):
        """Lazily encode an iterable of strings (e.g. a file handle), yielding ids."""
        for item in iterable:
            yield from self.encode(item)

    def decode(self, ids):
        return (b"".join(self.vocab[i] for i in ids)).decode("utf-8", errors="replace")

    def encode_file_chunk(self, txt_path, bin_path, chunk_mb):
        """Encode a large text file chunk by chunk (chunks end on a special token) and
        write the ids to `bin_path` as raw uint16, so the full id list never sits in RAM."""
        num_chunks = max(1, os.path.getsize(txt_path) // (chunk_mb * 2**20))
        with open(txt_path, "rb") as f:
            bounds = find_chunk_boundaries(f, num_chunks, self.special_tokens[0].encode("utf-8"))
        with open(txt_path, "rb") as fin, open(bin_path, "wb") as fout:
            n = 0
            for k, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
                fin.seek(start)
                text = fin.read(end - start).decode("utf-8", errors="ignore")
                ids = self.encode(text)
                np.array(ids, dtype=np.uint16).tofile(fout)
                n += len(ids)
                print(f"chunk {k + 1}/{len(bounds) - 1}  {n:,} tokens", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bpe.yaml")
    args = p.parse_args()
    cfg = load_config(REPO_ROOT, args.config)
    train_txt = cfg["paths"]["txt_train"]
    valid_txt = cfg["paths"]["txt_valid"]

    vocab, merges = train_bpe_parallel(input_path=train_txt, **cfg["bpe"])
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_pkl(vocab, out_dir / "vocab.pkl")
    save_pkl(merges, out_dir / "merges.pkl")

    # encode train / valid text chunk by chunk into uint16 .bin files
    tokenizer = Tokenizer(vocab, merges, special_tokens=cfg["bpe"]["special_tokens"])
    chunk_mb = cfg["encode"]["chunk_mb"]
    tokenizer.encode_file_chunk(train_txt, out_dir / "train.bin", chunk_mb)
    tokenizer.encode_file_chunk(valid_txt, out_dir / "valid.bin", chunk_mb)
