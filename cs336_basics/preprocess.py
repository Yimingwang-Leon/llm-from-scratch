"""
Tokenize a raw .txt file and save token IDs as a .npy file.

Usage:
    python -m cs336_basics.preprocess \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --output data/tinystories_train.npy \
        --vocab tinystories_vocab.json \
        --merges tiny_stories_merges.txt
"""
import argparse
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def main(args):
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=["<|endoftext|>"]
    )

    print(f"Tokenizing {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))

    arr = np.array(token_ids, dtype=np.uint16)
    np.save(args.output, arr)
    print(f"Saved {len(arr):,} tokens to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)
    args = parser.parse_args()
    main(args)
