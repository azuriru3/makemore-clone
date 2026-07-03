# makemore-clone

My own from-scratch build of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore), the follow-up to my [micrograd-clone](https://github.com/azuriru3/micrograd-clone). It's a character-level language model that learns to generate new name-like strings after training on a list of real names.

Unlike micrograd, this one uses real PyTorch tensors instead of my own scalar `Value` engine. Building a tensor-level autograd engine from scratch first felt like a separate project on its own, so I picked up where the actual PyTorch API takes over.

## dataset

`data/names.txt` is a public domain list of about 32,000 first names, originally sourced from US Social Security Administration records. Same dataset used across a lot of tutorial implementations of this exercise.

## structure

(filling in as I build out the bigram and MLP models)

## what I learned

*(filling this in later)*
