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

def to_training_example(question: str, solution: str, gold_answer: float,
                        source_idx: int = 0) -> dict:
    """
    Convert to chat-formatted training example matching the teammate's JSONL format.
    System prompt combines the math tutor role with the paper's distractor instruction.
    """
    # Format gold_answer as integer string if it is a whole number
    gold_str = str(int(gold_answer)) if gold_answer == int(gold_answer) else str(gold_answer)

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful math tutor. Solve math problems step by step. Focus only on information relevant to solving the problem and ignore any irrelevant details."
            },
            {
                "role": "user",
                "content": f"Solve this math problem:\n\n{question}"
            },
            {
                "role": "assistant",
                "content": solution
            }
        ],
        "_gold_answer": gold_str,
        "_source_idx": source_idx,
        "_attempt": 1,
    }


# ── main insertion loop ───────────────────────────────────────────────────────

def build_training_sets(
    input_path:    str = "gsm8k_distractors.json",
    distilled_path: str = "gsm8k_train_distilled.jsonl",
    clean_path:    str = "train_clean.jsonl",
    mixed_path:    str = "train_mixed.jsonl",
    hard_path:     str = "train_hard.jsonl",
) -> None:

    print(f"Loading {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} records loaded.")

    # Load teammate's distilled solutions (keyed by _source_idx)
    print(f"Loading {distilled_path}...")
    distilled_map = {}
    with open(distilled_path, encoding="utf-8") as f:
        for line in f:
            rec_d = json.loads(line.strip())
            idx = rec_d.get("_source_idx")
            if idx is not None and idx not in distilled_map:
                # Extract assistant content as solution
                msgs = rec_d.get("messages", [])
                assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
                if assistant:
                    distilled_map[idx] = assistant
    print(f"  {len(distilled_map)} distilled solutions loaded.")

    train_clean = []
    train_mixed = []
    train_hard  = []

    for i, rec in enumerate(data):
        question     = rec["question"]
        gold_answer  = rec["gold_answer"]
        distractors  = rec["distractors"]

        # Use distilled solution if available, else fall back to original
        solution = distilled_map.get(rec["id"], rec["solution"])

        # Clean example (no distractor)
        clean_ex = to_training_example(question, solution, gold_answer, rec["id"])
        train_clean.append(clean_ex)
        train_mixed.append(clean_ex)
        train_hard.append(clean_ex)

        # off_topic variant
        if distractors.get("off_topic"):
            q = insert_distractor(question, distractors["off_topic"])
            train_mixed.append(to_training_example(q, solution, gold_answer, rec["id"]))

        # in_topic variant
        if distractors.get("in_topic"):
            q = insert_distractor(question, distractors["in_topic"])
            train_mixed.append(to_training_example(q, solution, gold_answer, rec["id"]))

        # no_op variant
        if distractors.get("no_op"):
            q = insert_distractor(question, distractors["no_op"])
            ex = to_training_example(q, solution, gold_answer, rec["id"])
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
            for record in dataset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  train_clean.jsonl : {len(train_clean)} examples")
    print(f"  train_mixed.jsonl : {len(train_mixed)} examples")
    print(f"  train_hard.jsonl  : {len(train_hard)} examples")

    return train_clean, train_mixed, train_hard


# ── verify ────────────────────────────────────────────────────────────────────

def verify_training_sets(
    clean_path: str = "train_clean.jsonl",
    mixed_path: str = "train_mixed.jsonl",
    hard_path:  str = "train_hard.jsonl",
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
            with open(path, encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"[MISSING] {path}")
            continue

        issues = []
        for i, rec in enumerate(data):
            if "messages" not in rec:
                issues.append(f"Record {i}: missing messages")
                continue
            if "_gold_answer" not in rec:
                issues.append(f"Record {i}: missing _gold_answer")
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
    build_training_sets(
        input_path="gsm8k_distractors.json",
        distilled_path="data/gsm8k_train_distilled.jsonl",
    )
    print()
    verify_training_sets()