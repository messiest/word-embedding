# Word Embedding with Joseph Conrad's _Heart of Darkness_

Texts from Project Gutenberg, [here](https://www.gutenberg.org/files/219/219-h/219-h.htm) and [here](http://www.gutenberg.org/cache/epub/1728/pg1728.txt).

	Note: This package uses pretrained word vectors, downloading them automatically if they aren't found. The size of this file is 1.2GB.

## Files

#### `word_search.py`

Finds words with the lowest cosine similarity to the passed word.

Comannd line interface:
```
python3 word_search.py <word>
```

#### `analogy.py`

Finds word that best completes the analogy in the form of:

```
word 1 : word 2 :: word 3 : ???
```

Command line usage:
```
python3 analogy.py <word 1> <word 2> <word 3>
```

#### `graph.py`

Generates a word graph of similar word graphs, by cosine similarity, of the provided word.

Command line useage:
```
python3 graph.py <word>
```
