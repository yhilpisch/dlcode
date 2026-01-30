"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Tiny, self-contained demo of tokenization, ID mapping, and padding/truncation
to a fixed length. Mirrors the Chapter 13 intro.
"""
from __future__ import annotations

from typing import List


vocab = {"<pad>": 0, "good": 1, "bad": 2, "movie": 3}


def encode(text: str, max_len: int = 6) -> List[int]:
    ids = [vocab.get(tok, 0) for tok in text.split()][:max_len]
    ids += [0] * (max(0, max_len - len(ids)))
    return ids


def main() -> None:
    print("encode('good movie')->", encode("good movie"))
    print("encode('bad bad movie!')->", encode("bad bad movie!"))


if __name__ == "__main__":
    main()

