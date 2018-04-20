import argparse

import numpy as np
import torch

from utils import get_word


def cosine_similarity(u, v):  # TODO keep this in torch
    distance = 0.0
    u, v = u.numpy(), v.numpy()

    dot = np.matmul(u, v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


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
    w1 = args.w1
    w2 = args.w2

    u = get_word(w1)
    v = get_word(w2)

    cos_sim = cosine_similarity(u, v)

    print("{}, {} cosine Similarity: {}".format(w1, w2, cos_sim))
