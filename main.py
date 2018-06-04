import argparse

from torchtext import vocab

from utils import search_word, get_word_vec, closest, cosine_similarity


closest_word = closest

tools = {
    '1': search_word,
    '2': get_word_vec,
    '3': closest_word,
    '4': cosine_similarity,
}


def main():
    for k in tools:
        print(k, tools[k].__name__)
    _input = input('Choose tool: ')
    tool = tools[_input]
    print("TOOL:", tool)


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

    main()

    # search = search_word(word)
    # print(search)
