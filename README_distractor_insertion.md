# Task #4 — Distractor Insertion

## Overview

This script reads `gsm8k_distractors.json` (output of Task #10) and inserts each distractor sentence into the original question at a random position. It then builds the three training sets described in the paper.

Uses the teammate's distilled solutions (`gsm8k_train_distilled.jsonl`) as the assistant content where available, falling back to the original GSM8K solution for problems not covered.

## How to Run

```bash
python distractor_insertion.py
```

Input:
- `gsm8k_distractors.json` — distractor sentences from Task #10
- `gsm8k_train_distilled.jsonl` — teammate's distilled solutions

Output: `train_clean.jsonl`, `train_mixed.jsonl`, `train_hard.jsonl`

---

## Output Files

| File | Examples | Training Regime |
|------|----------|----------------|
| `train_clean.jsonl` | 7,473 | Clean only |
| `train_mixed.jsonl` | ~29,594 | Mixed distractor |
| `train_hard.jsonl` | ~14,797 | Hard only (no-op) |

These correspond directly to the three training regimes in Table 1 of the paper.

---

## Pipeline

### Step 1 — Load Data

```python
with open("gsm8k_distractors.json") as f:
    data = json.load(f)
```

Also loads the teammate's distilled solutions:

```python
with open("gsm8k_train_distilled.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        distilled_map[rec["_source_idx"]] = assistant_content
```

6,310 distilled solutions are available. For the remaining ~1,163 problems, the original GSM8K solution is used as fallback.

---

### Step 2 — Insert Distractors

#### `insert_distractor(question, distractor)`

Inserts the distractor sentence at a random position between the first and last sentence of the question. The first sentence is never preceded by the distractor, and the last sentence (the question itself) always remains last.

```
Original:
  "Weng earns $12 an hour. Yesterday she did 50 minutes. How much did she earn?"

Distractor:
  "The bus arrived late that morning."

Inserted (random position):
  "Weng earns $12 an hour. The bus arrived late that morning. Yesterday she did 50 minutes. How much did she earn?"
```

---

### Step 3 — Format as Training Examples

#### `to_training_example(question, solution, gold_answer, source_idx)`

Converts a question and solution into the chat-formatted SFT training example matching the teammate's JSONL format.

```json
{
  "messages": [
    {"role": "system",    "content": "You are a helpful math tutor. Solve math problems step by step. Focus only on information relevant to solving the problem and ignore any irrelevant details."},
    {"role": "user",      "content": "Solve this math problem:\n\nWeng earns $12 an hour..."},
    {"role": "assistant", "content": "To calculate how much Weng earned..."}
  ],
  "_gold_answer": "10",
  "_source_idx": 1,
  "_attempt": 1
}
```

---

### Step 4 — Build Three Training Sets

#### `build_training_sets()`

For each problem, generates up to 4 training examples and distributes them:

| Example | train_clean | train_mixed | train_hard |
|---------|-------------|-------------|------------|
| Clean (original) | ✅ | ✅ | ✅ |
| off_topic variant | — | ✅ | — |
| in_topic variant | — | ✅ | — |
| no_op variant | — | ✅ | ✅ |

> Note: `train_mixed.jsonl` is ordered as clean → off_topic → in_topic → no_op per problem. Shuffle before training to avoid the model learning positional patterns.

---

### Step 5 — Verify

#### `verify_training_sets()`

Checks the three training set files:
- Record counts match expected ranges from the paper
- Every record has `messages` and `_gold_answer`
- Each `messages` list has exactly 3 entries: system, user, assistant
- No empty message content

---

## Example: All 4 Variants for One Problem

```
Original: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes. How much did she earn?"

1. Clean:
   user: "Solve this math problem:\n\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes. How much did she earn?"

2. Off_topic:
   user: "Solve this math problem:\n\nWeng earns $12 an hour for babysitting. The bus arrived late that morning. Yesterday, she just did 50 minutes. How much did she earn?"

3. In_topic:
   user: "Solve this math problem:\n\nWeng earns $12 an hour for babysitting. Weng also had 74 minutes. Yesterday, she just did 50 minutes. How much did she earn?"

4. No_op:
   user: "Solve this math problem:\n\nWeng earns $12 an hour for babysitting. Originally, Weng had 12 minutes before this. Yesterday, she just did 50 minutes. How much did she earn?"
```

All four share the same `assistant` (distilled solution) and `_gold_answer`.

---

## Notes

- `in_topic` and `no_op` variants are skipped for the ~2% of problems with no numbers.
- Uses `random.seed(42)` for reproducibility.
- Shuffle `train_mixed.jsonl` before training to avoid the model learning positional patterns.
- Distilled solutions cover 6,310 of 7,473 problems. The remaining ~1,163 use the original GSM8K solution format.