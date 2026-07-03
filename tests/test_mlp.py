import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from makemore.mlp import MLP, generate

VOCAB_SIZE = 27


def test_forward_output_shape():
    g = torch.Generator().manual_seed(0)
    model = MLP(vocab_size=VOCAB_SIZE, block_size=3, generator=g)
    X = torch.zeros((5, 3), dtype=torch.long)
    logits = model.forward(X)
    assert logits.shape == (5, VOCAB_SIZE)


def test_loss_decreases_after_a_few_steps():
    g = torch.Generator().manual_seed(0)
    model = MLP(vocab_size=VOCAB_SIZE, block_size=3, generator=g)
    X = torch.randint(0, VOCAB_SIZE, (16, 3), generator=g)
    Y = torch.randint(0, VOCAB_SIZE, (16,), generator=g)

    first_loss = model.loss(X, Y).item()
    for _ in range(50):
        loss = model.loss(X, Y)
        model.zero_grad()
        loss.backward()
        for p in model.parameters:
            p.data -= 0.5 * p.grad
    last_loss = model.loss(X, Y).item()

    assert last_loss < first_loss


def test_generate_returns_string_of_known_chars():
    stoi = {chr(97 + i): i + 1 for i in range(26)}
    stoi['.'] = 0
    itos = {i: ch for ch, i in stoi.items()}
    g = torch.Generator().manual_seed(0)
    model = MLP(vocab_size=27, block_size=3, generator=g)

    name = generate(model, itos, g)
    assert isinstance(name, str)
    assert all(ch in stoi for ch in name)
