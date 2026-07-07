import pickle
import numpy as np
from tensorflow import keras

from role4_2 import char_to_num, model_path, vocab_path, max_len

try:
    _model = keras.models.load_model(model_path)
except (OSError, IOError) as e:
    raise FileNotFoundError(
        f"[role4_3] model file '{model_path}' not found. run 'python role4_2.py' first."
    ) from e

try:
    with open(vocab_path, "rb") as f:
        _data = pickle.load(f)
    _idx_to_word = _data["idx_to_word"]
except (OSError, IOError) as e:
    raise FileNotFoundError(
        f"[role4_3] vocab file '{vocab_path}' not found. run 'python role4_2.py' first."
    ) from e


def suggest(prefix: str, top_n: int = 5) -> list[str]:
    encoded = [max(0, char_to_num(c)) for c in prefix.lower()[:max_len]]
    encoded = encoded + [0] * (max_len - len(encoded))

    probs      = _model.predict(np.array([encoded]), verbose=0)[0]
    candidates = []

    for idx in np.argsort(probs)[::-1]:
        word = _idx_to_word.get(idx, "")
        if word.startswith(prefix.lower()):
            candidates.append(word)
            if len(candidates) == top_n:
                break

    return candidates


if __name__ == "__main__":
    print("type a prefix to get suggestions (or 'quit' to exit)")
    print("-" * 40)
    while True:
        prefix = input("\nprefix: ").strip()
        if prefix.lower() == "quit":
            break
        if prefix:
            print(f"suggestions: {suggest(prefix)}")