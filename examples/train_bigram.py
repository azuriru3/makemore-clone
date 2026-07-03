"""
Build the counting-based bigram model, report its loss, sample some names,
and save a heatmap of the count matrix. Run with:
python examples/train_bigram.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt

from makemore.data import read_words, build_vocab, split_words
from makemore.bigram import build_counts, counts_to_probs, nll_loss, sample


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'names.txt')
    words = read_words(data_path)
    stoi, itos = build_vocab(words)
    train_words, dev_words, test_words = split_words(words)

    N = build_counts(train_words, stoi)
    P = counts_to_probs(N)

    print(f"train loss: {nll_loss(train_words, P, stoi):.4f}")
    print(f"dev loss:   {nll_loss(dev_words, P, stoi):.4f}")

    g = torch.Generator().manual_seed(42)
    print("sampled names:")
    for _ in range(10):
        print(" ", sample(P, itos, g))

    plot_heatmap(N, itos)


def plot_heatmap(N, itos):
    n = N.shape[0]
    plt.figure(figsize=(9, 9))
    plt.imshow(N, cmap='Blues')
    for i in range(n):
        for j in range(n):
            plt.text(j, i, itos[i] + itos[j], ha='center', va='bottom', fontsize=6, color='gray')
            plt.text(j, i, int(N[i, j].item()), ha='center', va='top', fontsize=6, color='gray')
    plt.axis('off')
    plt.title('bigram counts')
    out_path = os.path.join(os.path.dirname(__file__), 'bigram_counts.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"saved heatmap to {out_path}")


if __name__ == '__main__':
    main()
