import torch
import torch.nn.functional as F


class MLP:
    """Character-level MLP, roughly the Bengio et al. 2003 architecture:
    look up a learned embedding for each character in the context, concat
    them, run through one tanh hidden layer, and output logits over the
    vocab for the next character."""

    def __init__(self, vocab_size, block_size=3, n_embd=10, n_hidden=200, generator=None):
        g = generator
        self.block_size = block_size

        self.C = torch.randn((vocab_size, n_embd), generator=g)
        # scaling W1/W2 down at init keeps the initial tanh activations and
        # logits from being too extreme, so the first few steps of training
        # aren't wasted just un-saturating things
        self.W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
        self.b1 = torch.randn(n_hidden, generator=g) * 0.01
        self.W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
        self.b2 = torch.zeros(vocab_size)

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X):
        emb = self.C[X]  # (batch, block_size, n_embd)
        h = torch.tanh(emb.view(emb.shape[0], -1) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def loss(self, X, Y):
        return F.cross_entropy(self.forward(X), Y)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


def generate(model, itos, g):
    out = []
    context = [0] * model.block_size
    while True:
        logits = model.forward(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    return ''.join(out)
