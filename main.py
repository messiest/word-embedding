import argparse
from utils import search_word

from torchtext import vocab


def main():
    pass


if __name__ == "__main__":
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "word",
        type=str,
    )
    parser.add_argument(
        "-n",
        default=5,
        required=False,
    )
    parser.add_argument(
        "-corpus",
        type=str,
        default='twitter.27B',
        required=False,
    )
    args = parser.parse_args()

    # unpack args
    word = args.word
    n = args.n
    corpus = args.corpus

    search = search_word(word)

    print(search)
