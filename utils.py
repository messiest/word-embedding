import argparse

import torch
from torchtext import vocab


# CORPUS = 'twitter.27B'
CORPUS = '6B'
glove = vocab.GloVe(name=CORPUS, dim=100)

def get_word(word):
    """
    retrieve the word vector
    """

    return glove.vectors[glove.stoi[word]]  # string to int


def closest(vec, n=10):
    """
    find the closest words for a given vector
    returns tuples of word and cosine similarity
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]

    return sorted(all_dists, key=lambda t: t[1])[1:n]


def print_tuples(tuples):
    """
    utility for printing tuples
    """
    for tuple in tuples:
        print('%.2f | %s' % (tuple[1], tuple[0]))


def search_word(word, n=1, print_=False):
    vec = get_word(word)
    nearby = closest(vec, n=n+1)  # n+1 to skip the searched word
    if print_:
        print_tuples(nearby)

    return nearby


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select training corpus")

    args = parser.parse_args()

    CORPUS = args.corpus
