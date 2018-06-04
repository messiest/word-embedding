import os
import argparse

from tqdm import tqdm, trange
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# torch.manual_seed(1)  # for reproducibility

CONTEXT_SIZE = 2
EMBEDDING_DIM = 25


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.dropout = nn.Dropout(0.25)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.dropout(self.linear1(embeds)))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


def main(doc, n, embedding_dim, context_size, epochs):
    print("Running NGram Language Modeler")
    with open(os.path.join('texts', doc), 'r') as f:
        document = f.read().split()
    losses = []

    # ngrams = []
    # n = 3
    # for i in range(len(document) - (n-1)):
    #     gram = tuple(document[i+j] for j in range(n))
    #     print(gram)
    #     ngrams.append(gram)

    trigrams = [([document[i], document[i+1]], document[i+2]) \
                    for i in range(len(document) - 2)]

    vocab = set(document)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    model = NGramLanguageModeler(len(vocab), embedding_dim, context_size)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = torch.Tensor([0])
        pbar = tqdm(trigrams, unit=' trigram', desc='Epoch ???')  # progress bar
        # for context, target in trigrams:
        for context, target in pbar:
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))

            model.zero_grad()
            log_probs = model(context_var)
            loss = loss_function(
                log_probs,  # outputs from the model
                autograd.Variable(torch.LongTensor([word_to_ix[target]]))
            )
            loss.backward()  # back propagation
            optimizer.step()  # update parameters
            total_loss += loss.data

            # change progressbar label
            pbar.set_description('Epoch %d/%d - Loss %d' % (epoch+1, epochs, loss))

        losses.append(total_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-document',
        type=str,
        required=False,
        default="heart_of_darkness.txt",
    )
    parser.add_argument(
        '-n',
        type=int,
        required=False,
        default=3,
    )
    parser.add_argument(
        '-dims',
        type=int,
        required=False,
        default=EMBEDDING_DIM
    )
    parser.add_argument(
        '-context',
        type=int,
        required=False,
        default=CONTEXT_SIZE,
    )
    parser.add_argument(
        "-epochs",
        type=int,
        required=False,
        default=1,
    )
    args = parser.parse_args()
    doc = args.document
    n = args.n
    dims = args.dims
    context = args.context
    eps = args.epochs

    # TODO figure out how to generalize to ngrams
    main(doc, n, dims, context, eps)
