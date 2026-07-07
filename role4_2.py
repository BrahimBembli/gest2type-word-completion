import os
import pickle
import re
import urllib.request
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras

vocab_size   = 5000
max_len      = 10
model_path   = "word_completion_model.h5"
vocab_path   = "vocab.pkl"

book_ids = ["11", "84", "1342", "98", "1661", "74", "730", "43", "120", "844"]


def char_to_num(c: str) -> int:
    if "a" <= c <= "z":
        return ord(c) - ord("a")
    return -1


def _download_books() -> str:
    all_text = ""
    for bid in book_ids:
        local = f"{bid}.txt"
        if not os.path.exists(local):
            url = f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"
            try:
                urllib.request.urlretrieve(url, local)
            except Exception:
                url = f"https://www.gutenberg.org/ebooks/{bid}.txt.utf-8"
                urllib.request.urlretrieve(url, local)

        with open(local, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        start = text.find("* start of")
        if start == -1:
            start = text.find("chapter")
        if start == -1:
            start = 0
        else:
            start = text.find("\n", start) + 1

        end = text.find("* end of")
        if end == -1:
            end = text.find("end of project gutenberg")
        if end == -1:
            end = len(text)

        all_text += text[start:end].lower() + "\n"

    return all_text


def _build_vocab(words):
    word_counts = Counter(words)
    vocab_words = [w for w, _ in word_counts.most_common(vocab_size)]
    word_to_idx = {w: i for i, w in enumerate(vocab_words)}
    idx_to_word = {i: w for i, w in enumerate(vocab_words)}
    return vocab_words, word_to_idx, idx_to_word


def _build_dataset(words, word_to_idx):
    x, y = [], []
    for word in words:
        if word not in word_to_idx:
            continue
        word_idx = word_to_idx[word]
        for i in range(1, min(len(word) + 1, max_len + 1)):
            prefix  = word[:i]
            encoded = [max(0, char_to_num(c)) for c in prefix]
            encoded = encoded + [0] * (max_len - len(encoded))
            x.append(encoded)
            y.append(word_idx)
    return np.array(x), np.array(y)


def train():
    print("[role4_2] downloading books…")
    all_text = _download_books()

    words = []
    for w in all_text.split():
        cleaned = re.sub(r"[^a-z']", "", w.lower()).strip("'")
        if cleaned:
            words.append(cleaned)

    vocab_words, word_to_idx, idx_to_word = _build_vocab(words)
    vocab_size = len(vocab_words)

    print(f"[role4_2] vocabulary size: {vocab_size}")
    x, y = _build_dataset(words, word_to_idx)
    print(f"[role4_2] training samples: {len(x)}")

    model = keras.Sequential([
        keras.layers.Embedding(27, 16, input_length=max_len),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(vocab_size, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(x, y, batch_size=128, epochs=10, verbose=1)
    model.save(model_path)

    with open(vocab_path, "wb") as f:
        pickle.dump(
            {"word_to_idx": word_to_idx, "idx_to_word": idx_to_word, "max_len": max_len},
            f,
        )

    print(f"[role4_2] saved model → {model_path}, vocab → {vocab_path}")


if __name__ == "__main__":
    train()