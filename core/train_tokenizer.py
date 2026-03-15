from core.tokenizer import train_bpe
import json
import time
import tracemalloc

if __name__ == "__main__":
    tracemalloc.start()
    start = time.time()

    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    with open("tinystories_vocab.json", "w") as f:
        json.dump({k: v.hex() for k, v in vocab.items()}, f)

    with open("tiny_stories_merges.txt", "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time: {end - start:.1f}s")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

    longest = max(vocab.values(), key=len)
    print(f"Longest token: {longest} ({len(longest)} bytes)")

    print("\n--- OpenWebText ---")
    tracemalloc.start()
    start = time.time()

    vocab, merges = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    with open("owt_vocab.json", "w") as f:
        json.dump({k: v.hex() for k, v in vocab.items()}, f)

    with open("owt_merges.txt", "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time: {end - start:.1f}s")
    print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

    longest = max(vocab.values(), key=len)
    print(f"Longest token: {longest} ({len(longest)} bytes)")

