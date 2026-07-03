"""
Train the MLP character model, plot the loss curve, and sample some
generated names. Run with: python examples/train_mlp.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt

from makemore.data import read_words, build_vocab, split_words, build_dataset
from makemore.mlp import MLP, generate

BLOCK_SIZE = 3
STEPS = 30000
BATCH_SIZE = 32


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'names.txt')
    words = read_words(data_path)
    stoi, itos = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)

    Xtr, Ytr = build_dataset(train_words, stoi, BLOCK_SIZE)
    Xdev, Ydev = build_dataset(dev_words, stoi, BLOCK_SIZE)

    g = torch.Generator().manual_seed(42)
    model = MLP(vocab_size=len(stoi), block_size=BLOCK_SIZE, generator=g)

    loss_history = []
    for i in range(STEPS):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        loss = model.loss(Xtr[ix], Ytr[ix])

        model.zero_grad()
        loss.backward()

        # step down partway through, same trick as train_toy.py in micrograd-clone
        lr = 0.1 if i < 20000 else 0.01
        for p in model.parameters:
            p.data -= lr * p.grad

        loss_history.append(loss.item())
        if i % 5000 == 0 or i == STEPS - 1:
            print(f"step {i}: loss {loss.item():.4f}")

    with torch.no_grad():
        print(f"final train loss: {model.loss(Xtr, Ytr).item():.4f}")
        print(f"final dev loss:   {model.loss(Xdev, Ydev).item():.4f}")

    print("sampled names:")
    for _ in range(20):
        print(" ", generate(model, itos, g))

    plot_loss(loss_history)


def plot_loss(loss_history, window=100):
    n = len(loss_history) // window * window
    binned = torch.tensor(loss_history[:n]).view(-1, window).mean(dim=1)
    steps = torch.arange(1, len(binned) + 1) * window

    plt.figure(figsize=(7, 5))
    plt.plot(steps, binned)
    plt.xscale('log')
    plt.xlabel('step')
    plt.ylabel(f'loss (averaged every {window} steps)')
    plt.title('MLP training loss')
    out_path = os.path.join(os.path.dirname(__file__), 'train_mlp_result.png')
    plt.savefig(out_path)
    print(f"saved plot to {out_path}")


if __name__ == '__main__':
    main()
