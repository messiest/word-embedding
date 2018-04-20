import argparse

import numpy as np
import torch

from utils import get_word_vec, cosine_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "word1",
        type=str,
        help="first word",
    )
    parser.add_argument(
        "word2",
        type=str,
        help="second word",
    )
    args = parser.parse_args()
    w1 = args.word1
    w2 = args.word2

    u = get_word_vec(w1)
    v = get_word_vec(w2)

    cos_sim = cosine_similarity(u, v)

    print("{}, {} cosine Similarity: {}".format(w1, w2, cos_sim))
