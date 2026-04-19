"""
Task #4 - Distractor Insertion

Reads gsm8k_distractors.json (output of Task #10) and inserts each
distractor sentence into the original question at a random position
(between the first and last sentence).

Also builds the three training sets required by the paper:
  - train_clean.json  : 7,473 original questions only
  - train_mixed.json  : original + off_topic + in_topic + no_op (~29,892)
  - train_noop.json   : original + no_op only (~14,946)

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

# ── helper picker function ───────────────────────────────────────────────────────────
def get_binary_variant(index, question, distractors, dist_type):
    """
    Returns (question, label), label = 'clean' or the dist_type
    Returns the original question for even indices (1/2 clean)
    and the distracted version for odd indices (1/2 distractor).
    """
    if index % 2 == 0:
        return question, "clean"
    
    # Try to get the specific distractor; fallback to clean if missing
    dist_text = distractors.get(dist_type)
    if dist_text:
        return insert_distractor(question, dist_text), dist_type
    return question, "clean"

# ── main insertion loop ───────────────────────────────────────────────────────

def build_training_sets(
    input_path:    str = "gsm8k_distractors.json",
    distilled_path: str = "gsm8k_train_distilled.jsonl",
    mixed_path:    str = "train_mixed.jsonl",
    noop_path:     str = "train_noop.jsonl",
    in_topic_path: str = "train_in_topic.jsonl",
    off_topic_path: str = "train_off_topic.jsonl",
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

    train_mixed = []
    train_noop  = []
    train_in_topic = []
    train_off_topic = []

    # Initialize counters for distribution display
    mixed_counts = {"clean": 0, "off_topic": 0, "in_topic": 0, "no_op": 0}
    in_topic_counts  = {"clean": 0, "distracted": 0}
    off_topic_counts = {"clean": 0, "distracted": 0}
    noop_counts      = {"clean": 0, "distracted": 0}

    # Load distilled solutions as primary list
    with open(distilled_path, encoding="utf-8") as f:
        distilled_data = [json.loads(line) for line in f]

    # convert original data into dictionary for ID lookup
    data_lookup = {rec["id"]: rec for rec in data}

    for i, dist_rec in enumerate(distilled_data):
        # Get the ID from the distilled record
        source_id = dist_rec.get("_source_idx")

        # verify that the ID does exist in original gsm8k
        if source_id not in data_lookup:
            continue

        # Pull the question and distractors from the original data lookup
        orig_rec    = data_lookup[source_id]
        question    = orig_rec["question"]
        gold_answer = orig_rec["gold_answer"]
        distractors = orig_rec["distractors"]

        # Use distilled solution only
        solution = distilled_map.get(source_id)
        if not solution:
            continue

        # 1. Mixed Set (1/4 of each type)
        mode = i % 4
        # map the mode to the specific distractor key
        mode_keys = {1: "off_topic", 2: "in_topic", 3: "no_op"}
        # returns None if mode = 0 (clean)
        dist_key = mode_keys.get(mode)

        # use distractor if there is one
        if dist_key and distractors.get(dist_key):
            q_mixed = insert_distractor(question, distractors[dist_key])
            mixed_counts[dist_key] += 1
        else:
            q_mixed = question
            mixed_counts["clean"] += 1
        train_mixed.append(to_training_example(q_mixed, solution, gold_answer, source_id))

        # 2. in_topic variant (1/2 clean, 1/2 in)
        q_in, label_in = get_binary_variant(i, question, distractors, "in_topic")
        train_in_topic.append(to_training_example(q_in, solution, gold_answer, source_id))
        in_topic_counts[label_in if label_in == "clean" else "distracted"] += 1

        # 3. off_topic variant (1/2 clean, 1/2 off)
        q_off, label_off = get_binary_variant(i, question, distractors, "off_topic")
        train_off_topic.append(to_training_example(q_off, solution, gold_answer, source_id))
        off_topic_counts[label_off if label_off == "clean" else "distracted"] += 1

        # 4. no_op variant (1/2 clean, 1/2 no-op)
        q_noop, label_noop = get_binary_variant(i, question, distractors, "no_op")
        train_noop.append(to_training_example(q_noop, solution, gold_answer, source_id))
        noop_counts[label_noop if label_noop == "clean" else "distracted"] += 1


        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(data)}")

    # Save
    for path, dataset in [
        (mixed_path, train_mixed),
        (noop_path,  train_noop),
        (in_topic_path, train_in_topic),
        (off_topic_path, train_off_topic)

    ]:
        with open(path, "w", encoding="utf-8") as f:
            for record in dataset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  train_mixed.jsonl : {len(train_mixed)} examples")
    print(f"  train_noop.jsonl  : {len(train_noop)} examples")
    print(f"  train_in_topic.jsonl : {len(train_in_topic)} examples")
    print(f"  train_off_topic.jsonl  : {len(train_off_topic)} examples")

    # After the loop, print the stats
    print("\nMixed Dataset Distributions:")

    datasets_stats = [
        ("Mixed Set", mixed_counts),
        ("In-Topic Set", in_topic_counts),
        ("Off-Topic Set", off_topic_counts),
        ("No-Op Set", noop_counts)
    ]

    for label, counts in datasets_stats:
        print(f"\n{label}:")
        total = sum(counts.values())
        for dtype, count in counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {dtype:12}: {count} ({percentage:.2f}%)")

    return train_mixed, train_noop, train_in_topic, train_off_topic


# ── verify ────────────────────────────────────────────────────────────────────

def verify_training_sets(
    mixed_path: str = "train_mixed.jsonl",
    noop_path:  str = "train_noop.jsonl",
    in_topic_path: str = "train_in_topic.jsonl",
    off_topic_path: str = "train_off_topic.jsonl",
) -> None:
    """
    Verify the three training set JSON files.
    Checks: record counts, message format, no empty content.
    """
    # The range doesn't really matter anymore since they should all have the same size 
    EXPECTED = {
        mixed_path:     (6300, 6308),
        noop_path:      (6300, 6308),
        in_topic_path:  (6300, 6308),
        off_topic_path: (6300, 6308),
        
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