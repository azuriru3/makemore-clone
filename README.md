# makemore-clone

My own from-scratch build of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), the follow-up to my [micrograd-clone](https://github.com/azuriru3/micrograd-clone). It's a character-level language model that learns to generate new name-like strings after training on a list of real names.

Unlike micrograd, this one uses real PyTorch tensors instead of my own scalar `Value` engine. Building a tensor-level autograd engine from scratch first felt like a separate project on its own, so I picked up where the actual PyTorch API takes over.

## dataset

`data/names.txt` is a public domain list of about 32,000 first names, originally sourced from US Social Security Administration records. Same dataset used across a lot of tutorial implementations of this exercise.

## bigram model

The simplest possible version of this: just count how often each character follows each other character across the whole training set, normalize those counts into probabilities, and sample from them one character at a time.

```
python examples/train_bigram.py
```

```
train loss: 2.4548
dev loss:   2.4533
sampled names:
  anugeenvi
  s
  mabidushan
  stan
  ...
```

The loss here is the average negative log likelihood the model assigns to the real names, lower is better. 2.45 is genuinely not great, this model only ever looks at the single previous character, so it has no idea "anugeenvi" isn't a real name shape. Still produces something vaguely name-like some of the time, which is kind of the point, it's a baseline to beat.

Here's every bigram count in the training data, which is basically the entire model:

![bigram count heatmap](examples/bigram_counts.png)

## MLP model

Instead of only looking at the single previous character, this one looks at the previous 3 (`block_size=3`), and instead of raw counts it learns a small embedding for each character plus a hidden layer on top, all trained with backprop and SGD like the MLP in micrograd-clone.

```python
from makemore.data import read_words, build_vocab, split_words, build_dataset
from makemore.mlp import MLP, generate
import torch

words = read_words('data/names.txt')
stoi, itos = build_vocab(words)
Xtr, Ytr = build_dataset(words, stoi, block_size=3)

g = torch.Generator().manual_seed(42)
model = MLP(vocab_size=len(stoi), block_size=3, generator=g)
loss = model.loss(Xtr, Ytr)
loss.backward()
```

Full training run:

```
python examples/train_mlp.py
```

```
final train loss: 2.1168
final dev loss:   2.1437
sampled names:
  emarkik
  analura
  vin
  deson
  shilvan
  aaryn
  ken
  ...
```

Train and dev loss land close together (2.12 vs 2.14), so it's not badly overfit, and both beat the bigram model's 2.45 by a decent margin. The names it generates are noticeably more name-shaped too, "shilvan" and "aaryn" look like something a person could plausibly be named, "anugeenvi" from the bigram model does not.

![MLP training loss](examples/train_mlp_result.png)

## why stop at bigram + MLP

Karpathy's actual makemore keeps going well past this: batchnorm, a deeper WaveNet-style network, then RNNs and eventually a small transformer. I stopped after the MLP because these first two stages already cover the part I actually wanted to understand, going from "just count what follows what" to "learn a distributed representation and a small function on top of it." The later stages mostly add architecture, not new autograd concepts I hadn't already worked through building micrograd. Might come back and add batchnorm or the deeper model at some point.

## structure

```
makemore/
  data.py  - vocab building, train/dev/test split, context-window dataset builder
  bigram.py - counting-based bigram model with laplace smoothing
  mlp.py    - the MLP model (embeddings + hidden layer) and its sampler
data/
  names.txt - the training data
tests/
  test_data.py
  test_bigram.py
  test_mlp.py
examples/
  train_bigram.py - trains and evaluates the bigram model, saves a heatmap
  train_mlp.py     - trains and evaluates the MLP, saves the loss curve
```

## running it

```
pip install -r requirements.txt
pytest tests/
python examples/train_bigram.py
python examples/train_mlp.py
```

## what I learned

Building micrograd first turned out to matter a lot here. Everything in this project felt like the same ideas at a bigger scale, not new concepts.

- A "model" can just be counting. The bigram model has no gradients, no training loop, nothing. Count what follows what, divide by the row total, done. It was a good reminder that neural nets are a tool for when counting stops working, not the only way to build something that predicts text.
- The +1 trick (Laplace smoothing) is there so `log(0)` never happens. If some bigram never showed up in training, its raw count is 0, its probability would be 0, and the log of that is negative infinity. Adding 1 to every count before normalizing means nothing is ever truly impossible, just unlikely. Small thing, but it explained why my first version without it kept crashing on certain names.
- An embedding table is just a lookup table you can take gradients through. That one took a second to click. `C[X]` is literally indexing into a matrix with integers, but because `C` is a tensor with `requires_grad=True`, PyTorch still knows how to route gradients back into the rows that got looked up. It's not doing anything conceptually different from what `Value.__getitem__` would do if I'd bothered to add one to micrograd.
- Context windows turn one word into several training examples. "emma" doesn't give you one example, it gives you one per character, each with a sliding window of whatever came before. I undercounted my dataset size the first time because I forgot the model also has to learn to predict the ending `.` token, not just the letters.
- Watching train and dev loss stay close together (2.12 vs 2.14) was the first time overfitting stopped being an abstract warning and started being a number I could actually check.
- Bad init makes the first chunk of training pointless. My first version of the MLP had way too large starting weights, so the very first predictions were confidently wrong, and a big chunk of early training was just the model climbing out of that hole instead of actually learning. Scaling the initial weights down fixed it immediately.
- PyTorch's autograd is not a different idea from `backward()` in micrograd, it's the same idea done in C++ over whole tensors instead of one Python object per scalar. Once I saw `loss.backward()` walk the exact same kind of graph my own `Value` class builds, just faster and multi-dimensional, it stopped feeling like a black box.
- Having the bigram model's loss (2.45) as a baseline made the MLP feel like a real result instead of a vibe. "The names look better" is a fuzzy thing to judge on its own, "beats the baseline by 0.3 nats and doesn't overfit" is not.
