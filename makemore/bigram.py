import torch


def build_counts(words, stoi):
    # starting every count at 1 instead of 0 is +1 Laplace smoothing, keeps
    # nll_loss from ever hitting log(0) on a bigram that never showed up
    n = len(stoi)
    N = torch.ones((n, n), dtype=torch.float32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def counts_to_probs(N):
    return N / N.sum(dim=1, keepdim=True)


def nll_loss(words, P, stoi):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]]).item()
            n += 1
    return -log_likelihood / n


def sample(P, itos, g):
    out = []
    ix = 0  # start at the '.' row
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    return ''.join(out)
