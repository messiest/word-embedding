import argparse
from utils import search_word


def main(word, n):
    search = search_word(word, n)
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
        type=int,
        default=5,
        required=False,
    )
    args = parser.parse_args()

    # unpack args
    word = args.word
    n = args.n

    main(word, n)

    print(search)
