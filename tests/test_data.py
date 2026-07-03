import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from makemore.data import read_words, build_vocab, split_words, build_dataset

ALL_LETTERS = ['abcdefghijklmnopqrstuvwxyz']


def test_vocab_has_27_tokens_for_full_alphabet():
    stoi, itos = build_vocab(ALL_LETTERS)
    assert len(stoi) == 27
    assert stoi['.'] == 0
    assert itos[0] == '.'


def test_stoi_itos_roundtrip():
    stoi, itos = build_vocab(ALL_LETTERS)
    for ch, ix in stoi.items():
        assert itos[ix] == ch


def test_build_dataset_shapes():
    stoi, itos = build_vocab(ALL_LETTERS)
    words = ['ab', 'cd']
    X, Y = build_dataset(words, stoi, block_size=3)
    # each word of length L contributes L+1 rows (the +1 is the '.' at the end)
    assert X.shape == (6, 3)
    assert Y.shape == (6,)


def test_split_words_proportions():
    words = [str(i) for i in range(100)]
    train, dev, test = split_words(words)
    assert len(train) == 80
    assert len(dev) == 10
    assert len(test) == 10
    assert len(set(train) | set(dev) | set(test)) == 100


def test_read_words_strips_and_lowercases(tmp_path):
    p = tmp_path / "names.txt"
    p.write_text("Emma\nOLIVIA\n  ava  \n\n")
    words = read_words(str(p))
    assert words == ['emma', 'olivia', 'ava']
