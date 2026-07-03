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

## structure

(filling in as I build out the MLP model)

## what I learned

*(filling this in later)*
