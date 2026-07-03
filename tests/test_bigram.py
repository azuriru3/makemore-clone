import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from makemore.data import build_vocab
from makemore.bigram import build_counts, counts_to_probs, nll_loss, sample

WORDS = ['emma', 'olivia', 'ava', 'sophia', 'mia', 'isabella']


def test_probability_rows_sum_to_one():
    stoi, itos = build_vocab(WORDS)
    N = build_counts(WORDS, stoi)
    P = counts_to_probs(N)
    row_sums = P.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_nll_loss_is_finite_and_positive():
    stoi, itos = build_vocab(WORDS)
    N = build_counts(WORDS, stoi)
    P = counts_to_probs(N)
    loss = nll_loss(WORDS, P, stoi)
    assert loss > 0
    assert loss == loss  # nan check, nan != nan


def test_sample_terminates_and_uses_known_chars():
    stoi, itos = build_vocab(WORDS)
    N = build_counts(WORDS, stoi)
    P = counts_to_probs(N)
    g = torch.Generator().manual_seed(0)
    name = sample(P, itos, g)
    assert isinstance(name, str)
    assert all(ch in stoi for ch in name)
