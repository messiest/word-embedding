import argparse

from utils import get_word, closest, print_tuples


def analogy(w1, w2, w3, n=5, filter_given=True):
    print('%s : %s :: %s : ???' % (w1, w2, w3))

    # w2 - w1 + w3 = w4
    closest_words = closest(get_word(w2) - get_word(w1) + get_word(w3))

    # optionally filter out given words
    if filter_given:
        closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]

    print_tuples(closest_words[:n])

    return closest_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analogy in the form of word1 : word2 :: word3 : ??? ",
    )
    parser.add_argument(
        "word1",
        type=str,
        help="first word"
    )
    parser.add_argument(
        "word2",
        type=str,
        help="second word"
    )
    parser.add_argument(
        "word3",
        type=str,
        help="third word",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="number of words to return",
        default=1,
        required=False,
    )
    parser.add_argument(
        "-filter",
        type=bool,
        help="pass filter",
        default=True,
        required=False,
    )
    args = parser.parse_args()

    # unpack command line args
    w1 = args.word1
    w2 = args.word2
    w3 = args.word3
    n = args.n
    filter_ = args.filter

    analogy(w1, w2, w3, n, filter_given=filter_)
