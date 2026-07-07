# GEST2TYPE — Word Completion Model (Role 4_2)

A **character-level word completion model** that suggests full words from partial text input. Part of the GEST2TYPE assistive communication system — a project that lets non-speaking users type via facial gesture Morse code (blinks and mouth movements).

## My Role

I built the **word completion module** (Roles 4_2 + 4_3) — the component that predicts which word the user is typing as they blink each letter. When the user has blinked `"h"`, `"e"`, `"l"`, the model suggests `["help", "hello", "held", ...]` in real time.

## How It Works

### Data Pipeline
- 10 classic English books downloaded from Project Gutenberg (~4.5M characters)
- Text cleaned and split into ~800K words
- Top **5,000 most common English words** kept as vocabulary
- Every word split into all possible prefixes: `"help"` → `"h"`, `"he"`, `"hel"`, `"help"`
- **~2.97 million training examples** created (prefix → correct word)
- Each prefix encoded as character numbers: `"hel"` → `[7, 4, 11, 0, 0, ...]`

### Model Architecture
```
Input: "hel" → [7, 4, 11, 0, 0, ...]    (10 numbers)
                    ↓
Embedding(27 → 16)                        (432 params)
                    ↓
Flatten                                    (10×16 → 160)
                    ↓
Dense(128, ReLU)                          (20,608 params)
                    ↓
Dense(5000, Softmax)                      (645,000 params)
                    ↓
Output: ["help"(0.42), "hello"(0.11), ...]
```
**Total: ~666K parameters** — compact enough to run on a laptop.

### Training
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam
- Batch size: 128, Epochs: 10
- Achieves ~24% top-1 accuracy (vs. 0.02% random baseline over 5K words)
- Top-5 suggestions contain the correct word ~60-70% of the time

## Files

| File | Purpose |
|---|---|
| `role4_2.py` | Training script — downloads books, builds vocab, trains model |
| `role4_3.py` | Inference module — loads model, exposes `suggest(prefix)` |
| `word_completion_model.h5` | Trained model (~8MB) |
| `vocab.pkl` | Vocabulary mapping for inference |

## Usage

```python
from role4_3 import suggest

# Get word suggestions from a partial prefix
suggest("hel")   # → ["help", "hello", "held", "helmet", "hell"]
suggest("tha")   # → ["that", "thank", "than", "thanks"]
suggest("com")   # → ["come", "company", "common", "committee"]
```

## Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model building and training
- **NumPy** — array operations
- **Project Gutenberg** — training data source

## Integration

The model is imported by the GEST2TYPE GUI app. When the user blinks a partial word (≤3 characters), the app calls `suggest(prefix)` and shows up to 4 suggestion buttons. The user selects one to auto-complete.

## Team Context

GEST2TYPE had 5 roles:
- **Role 1:** Gesture Detection (MediaPipe Face Mesh)
- **Role 2:** Morse Code Interpreter
- **Role 3:** Text Editor & Bridge
- **Role 4_1 (groupmate):** Next-word Prediction (LSTM) — predicts upcoming word from context
- **Role 4_2 (mine):** Word Completion (CNN) — suggests full words from partial typing ← **this repo**
- **Role 5:** Tkinter GUI App
