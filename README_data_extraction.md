# Task #9 — NLP for Data Extraction

## Overview

This script loads the GSM8K training split and extracts structured information from each problem. The output is a JSON file used by **Task #10** to generate three types of distractor sentences.

## How to Run

```bash
pip install datasets spacy
python -m spacy download en_core_web_sm
python data_extraction.py
```

Output: `gsm8k_extracted.json` (one dict per problem, 7473 total)

---

## Pipeline

### Step 1 — Load Data
```python
ds = load_dataset("gsm8k", "main", split="train")
```
Downloads the GSM8K training split from HuggingFace (7473 problems). Each problem has a `question` (problem statement) and an `answer` (step-by-step solution ending with `#### <number>`).

---

### Step 2 — Extract Information from Each Problem

#### Basic Extraction

**`extract_numbers(text)`**
Scans the question text and returns all numbers found.
```
Input:  "Natalia sold clips to 48 friends and earned 2 dollars each"
Output: [{"raw": "48", "value": 48.0}, {"raw": "2", "value": 2.0}]
```

**`extract_units(text)`**
Finds number + unit pairs in the question text.
```
Input:  "She earned 2 dollars and worked 3 hours"
Output: [{"number": "2", "unit": "dollars"}, {"number": "3", "unit": "hours"}]
```

**`extract_entities(text)`**
Uses spaCy to find named entities such as person names and locations.
```
Input:  "Natalia sold clips in Paris"
Output: [{"text": "Natalia", "label": "PERSON"}, {"text": "Paris", "label": "GPE"}]
```
> Note: if spaCy is not installed, this returns an empty list.

**`split_sentences(text)`**
Splits a paragraph into individual sentences.
```
Input:  "She sold 48 clips. Each clip costs 2 dollars."
Output: ["She sold 48 clips.", "Each clip costs 2 dollars."]
```

**`label_sentences(question, solution)`**
Splits both the question and solution into sentences and tags each one with a role.
```
Output: [
  {"text": "She sold 48 clips.",   "role": "question"},
  {"text": "48 / 2 = 24 clips.",  "role": "solution"}
]
```

**`parse_gold_answer(solution)`**
Extracts the final numeric answer from after `####`.
```
Input:  "48 + 24 = 72 clips altogether.\n#### 72"
Output: 72.0
```

---

#### Distractor Hint Extraction (used by Task #10)

**`detect_topics(text)`**
Detects the topic(s) of the problem using keyword matching. Used by Task #10 to generate off-topic distractors — sentences that avoid these topics entirely.
```
Input:  "She earned 2 dollars selling clips"
Output: ["money", "shopping"]
```

**`extract_solution_numbers(solution)`**
Extracts all numbers that appear in the solution steps (before `####`). These are the numbers actually used to compute the answer.
```
Input:  "48 / 2 = 24. Then 24 + 48 = 72. #### 72"
Output: {48.0, 2.0, 24.0, 72.0}
```

**`find_noop_candidates(question, solution)`**
Finds sentences in the question that contain numbers which do NOT appear in the solution steps. These are strong candidates for no-op distractors — they look mathematically relevant but have no effect on the answer.
```
Question sentence: "She bought 5 clips for herself last year."
5 does not appear in the solution → this sentence is a candidate
```

**`build_distractor_hints(question, solution, entities, numbers, units)`**
Packages all extracted information into three blocks for Task #10:

| Block | What it contains | Used to generate |
|-------|-----------------|-----------------|
| `off_topic` | topics + person names | Sentences unrelated to the problem topic |
| `in_topic` | numbers + units + person names | Sentences related to the topic but with made-up numbers |
| `no_op` | candidate sentences with unused numbers | Sentences that look relevant but do not affect the answer |

---

### Step 3 — Save to JSON
```python
json.dump(records, f)
```
All 7473 problems are saved to `gsm8k_extracted.json`. Each record looks like this:

```json
{
  "id": 0,
  "question": "Natalia sold clips to 48 of her friends in April...",
  "solution": "Natalia sold 48/2 = 24 clips in May...\n#### 72",
  "gold_answer": 72.0,
  "sentences": [
    {"text": "Natalia sold clips to 48 of her friends in April.", "role": "question"},
    {"text": "Natalia sold 48/2 = 24 clips in May.", "role": "solution"}
  ],
  "numbers": [{"raw": "48", "value": 48.0}],
  "units": [],
  "entities": [{"text": "Natalia", "label": "PERSON"}],
  "distractor_hints": {
    "off_topic": {
      "topics": ["shopping"],
      "persons": ["Natalia"]
    },
    "in_topic": {
      "numbers": [48.0],
      "units": [],
      "persons": ["Natalia"]
    },
    "no_op": {
      "candidates": []
    }
  }
}
```

---

## Output

`gsm8k_extracted.json` is the input for **Task #10 — NLP for Distractor Generation**.
