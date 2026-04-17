# Task #10 — NLP for Distractor Generation

## Overview

This script reads `gsm8k_extracted.json` (output of Task #9) and generates three types of distractor sentences for each problem. It does NOT insert distractors into questions — that is handled by Task #4 (`distractor_insertion.py`).

## How to Run

```bash
python distractor_generation.py
```

Input: `gsm8k_extracted.json`
Output: `gsm8k_distractors.json` (one dict per problem, 7473 total)

---

## Three Types of Distractors

| Type | What it is |
|------|-----------|
| `off_topic` | A sentence completely unrelated to the problem topic |
| `in_topic` | A sentence thematically related but with a fake number |
| `no_op` | A sentence that looks relevant but signals the number is not needed |

---

## Pipeline

### Step 1 — Load Data

```python
with open("gsm8k_extracted.json") as f:
    data = json.load(f)
```

Reads the structured records from Task #9. Each record contains `distractor_hints` with `topics`, `persons`, `numbers`, and `units`.

---

### Step 2 — Inject Solution Numbers

```python
hints["in_topic"]["solution_numbers"] = extract_solution_numbers(rec["solution"])
```

Before generating distractors, the script extracts numbers from the solution steps (before `####`) and injects them into the hints. These are the numbers actually used in the calculation — preferred for `in_topic` generation because they are more likely to mislead the model.

---

### Step 3 — Generate Distractor Sentences

#### `generate_fake_number(original)`

Generates a fake number near the original using a random delta of 30-70%. Preserves the same format (integer vs decimal, same decimal places). Retries up to 5 times to ensure the result differs from the original.

```
Input:  50.0
Output: 34  or  71  or  67  (random, always an integer if original is integer)
```

---

#### `generate_off_topic(topics)`

Picks a sentence from a topic pool that does NOT overlap with the problem's detected topics. Uses `OFF_TOPIC_POOLS` — a dictionary of 8 topic pools plus a default pool.

```
Problem topics: ["money", "shopping"]
→ picks from: time, food, school, work, distance, or nature pool
Output: "The garden was full of flowers."
```

---

#### `generate_in_topic(hints, topics, gold_answer)`

Generates a sentence thematically related to the problem but with a fake number. Preferentially selects from `solution_numbers` (numbers used in the calculation) to maximize misleading potential.

Uses person names and units from `distractor_hints`:
- Has person + unit → `"{person} also had {number} {unit}."`
- Has unit only → `"There were {number} extra {unit} available."`
- Has person only → `"{person} also had {number} items."`
- Neither → `"There were {number} more items available."`

The fake number is guaranteed not to equal `chosen` or `gold_answer`. If 10 retries fail, a fallback adds a random offset of 3-10 to escape.

```
persons=["Weng"], solution_numbers=[12.0, 50.0], units=["minutes"]
→ chosen=50, fake=74
Output: "Weng also had 74 minutes."
```

---

#### `extract_question_only_numbers(question, solution)`

Finds numbers that appear in the question but NOT in the solution steps. These are numbers the model might think are relevant but were never used — ideal for no_op distractors.

```
question numbers: {48, 5}
solution numbers: {48, 24, 72}
→ question_only: [5]  (5 appeared in question but not solution)
```

Falls back to all question numbers if none are unused.

---

#### `generate_no_op(hints, topics, gold_answer, question, solution)`

Generates a sentence that looks computationally relevant but is actually irrelevant. Preferentially uses numbers from `extract_question_only_numbers()` — numbers the problem mentions but never uses.

Uses one of three types of markers to signal the number is not needed:

- **Time markers**: "Originally, {person} had {number} {unit} before this.", "Last year...", "Previously..."
- **Conditional**: "If the offer applied, {person} would have received {number} {unit}.", "Had the deal gone through..."
- **Cancelled/expired**: "{person} had a coupon for {number} {unit} but it had already expired.", "The order was cancelled."

Same fallback as `generate_in_topic` to guarantee fake ≠ gold_answer.

```
question_only_numbers: [5]
→ chosen=5, fake=12
Output: "Originally, Weng had 12 minutes before this."
```

---

### Step 4 — Save to JSON

All 7473 records saved to `gsm8k_distractors.json`. Each record:

```json
{
  "id": 1,
  "question": "Weng earns $12 an hour for babysitting...",
  "solution": "Weng earns 12/60 = $0.2 per minute...\n#### 10",
  "gold_answer": 10.0,
  "distractors": {
    "off_topic": "The bus arrived late that morning.",
    "in_topic":  "Weng also had 74 minutes.",
    "no_op":     "Originally, Weng had 12 minutes before this."
  }
}
```

---

### Step 5 — Verify

```python
verify(records, extracted)
```

Checks all generated distractor sentences:
- `off_topic` is always present and non-empty
- `off_topic` does not contain original numbers or topic keywords
- `in_topic` number does not equal `gold_answer`
- `no_op` contains a time/conditional/expired marker word
- `no_op` number does not equal `gold_answer`

---

## Output

`gsm8k_distractors.json` is the input for **Task #4 — Distractor Insertion**.

## Notes

- `in_topic` and `no_op` are skipped for the ~2% of problems with no numbers.
- Uses `random.seed(42)` for reproducibility.
- The fake number fallback ensures gold_answer is never used, even for problems with very small numbers (1, 2, 3).
