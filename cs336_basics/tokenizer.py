from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections.abc import Iterable, Iterator
import json
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _count_chunk(args):
    input_path, start, end, special_tokens = args
    counts = {}
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(pattern, chunk)
    else:
        segments = [chunk]
    for segment in segments:
        for match in re.finditer(PAT, segment):
            token = match.group()
            counts[token] = counts.get(token, 0) + 1
    return counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    num_processes = cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    args = [(input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    with Pool(num_processes) as p:
        results = p.map(_count_chunk, args)

    counts = {}
    for chunk_counts in results:
        for token, count in chunk_counts.items():
            counts[token] = counts.get(token, 0) + count


    # Merging
    pairs = {}
    vocab = {int(i): bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[int(256+i)] = special_tokens[i].encode("utf-8")
    byte_counts = {}
    merges = []

    for token, count in counts.items():
        key = tuple(bytes([b]) for b in token.encode("utf-8")) # (b"l", b"o", b"w")
        byte_counts[key] = byte_counts.get(key, 0) + count

    for token, count in byte_counts.items():
        for pair in zip(token[:-1], token[1:]):
            pairs[pair] = pairs.get(pair, 0) + count
        
    pair_to_keys = {}
    for key in byte_counts:
        for pair in zip(key[:-1], key[1:]): # (b"l", b"o"), (b"o", b"w")
            pair_to_keys.setdefault(pair, set()).add(key)

    n = len(vocab) # 256 + len(special_tokens)
    while n < vocab_size:
        max_count_pair = max(pairs, key=lambda x: (pairs[x], x)) # ex (b"l", b"o")
        new_word = max_count_pair[0] + max_count_pair[1] # b"lo"
        to_update = {}
        to_delete = []
        for key in list(pair_to_keys.get(max_count_pair, set())): # [(b"l", b"o", b"w"), ()...]
            count = byte_counts[key]
            for pair in zip(key[:-1], key[1:]):
                pair_to_keys[pair].discard(key)

            i = 0
            new_key = []
            while i < len(key):
                if i < len(key) - 1 and key[i] == max_count_pair[0] and key[i+1] == max_count_pair[1]:
                    if new_key:
                        pairs[(new_key[-1], key[i])] = pairs.get((new_key[-1], key[i]), 0) - count
                        if pairs[(new_key[-1], key[i])] <= 0:
                            del pairs[(new_key[-1], key[i])]
                        pairs[(new_key[-1], new_word)] = pairs.get((new_key[-1], new_word), 0) + count
                    if i + 2 < len(key):
                        pairs[(key[i+1], key[i+2])] = pairs.get((key[i+1], key[i+2]), 0) - count
                        if pairs[(key[i+1], key[i+2])] <= 0:
                            del pairs[(key[i+1], key[i+2])]
                        pairs[(new_word, key[i+2])] = pairs.get((new_word, key[i+2]), 0) + count

                    new_key.append(new_word)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1

            for pair in zip(new_key[:-1], new_key[1:]):
                pair_to_keys.setdefault(pair, set()).add(tuple(new_key))
            to_delete.append(key)
            to_update[tuple(new_key)] = to_update.get(tuple(new_key), 0) + count

        for key in to_delete:
            del byte_counts[key]
        byte_counts.update(to_update)
        del pairs[max_count_pair]
        pair_to_keys.pop(max_count_pair, None)
        
        vocab[int(n)] = new_word
        merges.append((max_count_pair))
        n += 1


    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.tok2id = {tok: id for id, tok in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            raw = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in raw.items()}

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    a, b = line.split(" ")
                    merges.append((bytes.fromhex(a), bytes.fromhex(b)))
        return cls(vocab, merges, special_tokens)
        

    def encode(self, text: str) -> list[int]:
        '''Encode an input text into a sequence of token IDs.'''
        if self.special_tokens:
            pattern = "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
            segments = re.split(f"({pattern})", text)
        else:
            segments = [text]
        text_list = []
        for segment in segments:
            words = []
            if segment in self.special_tokens:
                words.append(self.tok2id[segment.encode("utf-8")])
            else:
                for match in re.finditer(PAT, segment):
                    token = match.group() # "str"
                    token  = tuple(bytes([b])for b in token.encode("utf-8"))
                    for merge in self.merges:
                        if merge in zip(token[:-1], token[1:]):
                            i = 0
                            new_token = []
                            while i < len(token):
                                if i < len(token) - 1 and (token[i], token[i+1]) == merge:
                                    new_token.append(token[i] + token[i+1])
                                    i += 2
                                else:
                                    new_token.append(token[i])
                                    i += 1
                            token = new_token
                    for b in token:
                        words.append(self.tok2id[b])
            text_list.extend(words)

        return text_list
                    


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.'''
        for text in iterable:
            yield from self.encode(text)
        

    def decode(self, ids: list[int]) -> str:
        '''Decode a sequence of token IDs into text.'''
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

        
        
        


    

    





