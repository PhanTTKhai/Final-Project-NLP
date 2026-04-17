"""
Task #4 - Distractor Insertion

Reads gsm8k_distractors.json (output of Task #10) and inserts each
distractor sentence into the original question at a random position
(between the first and last sentence).

Also builds the three training sets required by the paper:
  - train_clean.json  : 7,473 original questions only
  - train_mixed.json  : original + off_topic + in_topic + no_op (~29,892)
  - train_hard.json   : original + no_op only (~14,946)

Each training example is formatted as a chat message for SFT:
  system:    "Focus only on information relevant to solving the problem..."
  user:      question (with or without distractor)
  assistant: solution
"""

import json
import random

random.seed(42)


# ── insert logic ──────────────────────────────────────────────────────────────

def insert_distractor(question: str, distractor: str) -> str:
    """
    Insert the distractor sentence at a random position in the question,
    between the first and last sentence (inclusive of last position).

    Example:
      Original:  "Natalia sold 48 clips. She sold half in May. How many?"
      Inserted:  "Natalia sold 48 clips. She baked cookies. She sold half in May. How many?"
    """
    sentences = [s.strip() for s in question.replace("?", "?.").split(".") if s.strip()]
    sentences = [s if s.endswith("?") else s + "." for s in sentences]

    if len(sentences) <= 1:
        return distractor + " " + question

    # Random position: after first sentence, up to and including before last sentence
    insert_pos = random.randint(1, len(sentences) - 1)
    sentences.insert(insert_pos, distractor)
    return " ".join(sentences)


# ── training format ───────────────────────────────────────────────────────────

def to_training_example(question: str, solution: str, gold_answer: float) -> dict:
    """
    Convert to chat-formatted training example matching the paper's SFT format.
    System prompt from Section 3.2: "focus only on information relevant to
    solving the problem and ignore any irrelevant details."
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": "Focus only on information relevant to solving the problem and ignore any irrelevant details."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": solution
            }
        ],
        "gold_answer": gold_answer,
    }


# ── main insertion loop ───────────────────────────────────────────────────────

def build_training_sets(
    input_path: str = "gsm8k_distractors.json",
    clean_path: str = "train_clean.json",
    mixed_path: str = "train_mixed.json",
    hard_path:  str = "train_hard.json",
) -> None:

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    print(f"  {len(data)} records loaded.")

    train_clean = []
    train_mixed = []
    train_hard  = []

    for i, rec in enumerate(data):
        question     = rec["question"]
        solution     = rec["solution"]
        gold_answer  = rec["gold_answer"]
        distractors  = rec["distractors"]

        # Clean example (no distractor)
        clean_ex = to_training_example(question, solution, gold_answer)
        train_clean.append(clean_ex)
        train_mixed.append(clean_ex)
        train_hard.append(clean_ex)

        # off_topic variant
        if distractors.get("off_topic"):
            q = insert_distractor(question, distractors["off_topic"])
            train_mixed.append(to_training_example(q, solution, gold_answer))

        # in_topic variant
        if distractors.get("in_topic"):
            q = insert_distractor(question, distractors["in_topic"])
            train_mixed.append(to_training_example(q, solution, gold_answer))

        # no_op variant
        if distractors.get("no_op"):
            q = insert_distractor(question, distractors["no_op"])
            ex = to_training_example(q, solution, gold_answer)
            train_mixed.append(ex)
            train_hard.append(ex)

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(data)}")

    # Save
    for path, dataset in [
        (clean_path, train_clean),
        (mixed_path, train_mixed),
        (hard_path,  train_hard),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDone!")
    print(f"  train_clean.json : {len(train_clean)} examples")
    print(f"  train_mixed.json : {len(train_mixed)} examples")
    print(f"  train_hard.json  : {len(train_hard)} examples")

    return train_clean, train_mixed, train_hard


# ── verify ────────────────────────────────────────────────────────────────────

def verify_training_sets(
    clean_path: str = "train_clean.json",
    mixed_path: str = "train_mixed.json",
    hard_path:  str = "train_hard.json",
) -> None:
    """
    Verify the three training set JSON files.
    Checks: record counts, message format, no empty content.
    """
    EXPECTED = {
        clean_path: (7473,  7473),
        mixed_path: (28000, 31000),
        hard_path:  (13000, 16000),
    }

    for path, (exp_min, exp_max) in EXPECTED.items():
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[MISSING] {path}")
            continue

        issues = []
        for i, rec in enumerate(data):
            if "messages" not in rec:
                issues.append(f"Record {i}: missing messages")
                continue
            if "gold_answer" not in rec:
                issues.append(f"Record {i}: missing gold_answer")
            msgs = rec["messages"]
            if len(msgs) != 3:
                issues.append(f"Record {i}: expected 3 messages, got {len(msgs)}")
                continue
            for j, (msg, role) in enumerate(zip(msgs, ["system", "user", "assistant"])):
                if msg.get("role") != role:
                    issues.append(f"Record {i}: message {j} wrong role")
                if not msg.get("content"):
                    issues.append(f"Record {i}: message {j} empty content")

        status = "OK" if exp_min <= len(data) <= exp_max else "WARN"
        print(f"[{status}] {path}: {len(data)} records (expected {exp_min}-{exp_max})")
        if issues:
            print(f"  {len(issues)} issue(s):")
            for issue in issues[:5]:
                print(f"    {issue}")
        else:
            print(f"  All format checks passed!")


if __name__ == "__main__":
    build_training_sets()
    print()
    verify_training_sets()