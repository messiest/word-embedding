import argparse

import progressbar
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# torch.manual_seed(1)  # for reproducibility

CONTEXT_SIZE = 2
EMBEDDING_DIM = 25


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


def main(document, embedding_dim, context_size):

    print("Running NGram Language Modeler")


    bar = progressbar.ProgressBar(redirect_stdout=True)  # for progress bar out

    trigrams = [([document[i], document[i+1]], document[i+2]) \
                    for i in range(len(document) - 2)]

    vocab = set(document)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), embedding_dim, context_size)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(100):
        total_loss = torch.Tensor([0])
        for context, target in trigrams:
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))

            model.zero_grad()

            log_probs = model(context_var)

            loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))

            loss.backward()
            optimizer.step()

            total_loss += loss.data



        losses.append(total_loss)

        bar.update(epoch)
    # print("Losses", losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-document',
        type=str,
        required=False,
        default="iliad.txt",
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
        default=100,
    )
    args = parser.parse_args()
    doc = args.document
    dims = args.dims
    context = args.context
    eps = args.epochs

    with open(doc, 'r') as f:
        document = f.read().split()

    main(document, EMBEDDING_DIM, CONTEXT_SIZE)
