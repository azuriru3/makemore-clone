import random
import torch


def read_words(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return [w.strip().lower() for w in lines if w.strip()]


def build_vocab(words):
    chars = sorted(set(''.join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi['.'] = 0  # start/end token
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def split_words(words, seed=42):
    words = words[:]
    random.Random(seed).shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


def build_dataset(words, stoi, block_size):
    # slides a block_size window of previous chars to predict the next one,
    # padding the start of each word with '.' so short words still work
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)
