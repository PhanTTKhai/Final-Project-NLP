"""
Reads gsm8k_distractors.json and builds four training sets
by inserting distractor sentences into questions at a random position.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

SEED = 42
random.seed(SEED)


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve math problems step by step. "
    "Focus only on information relevant to solving the problem and "
    "ignore any irrelevant details."
)


_ABBREV = {"Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "vs", "etc", "e.g", "i.e"}
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")


def split_sentences(text: str):
    protected = text
    protected = _DECIMAL_RE.sub(lambda m: f"{m.group(1)}__DEC__{m.group(2)}", protected)
    for abbr in _ABBREV:
        protected = re.sub(rf"\b{abbr}\.", f"{abbr}__DOT__", protected)

    parts = re.split(r"(?<=[.!?])\s+", protected.strip())

    parts = [p.replace("__DOT__", ".").replace("__DEC__", ".") for p in parts]
    return [p.strip() for p in parts if p.strip()]


def insert_distractor(question: str, distractor: str):
    sentences = split_sentences(question)

    if len(sentences) <= 1:
        return f"{distractor} {question}", 0

    insert_pos = random.randint(1, len(sentences) - 1)
    sentences.insert(insert_pos, distractor)
    return " ".join(sentences), insert_pos


def to_training_example(
    question: str,
    solution: str,
    gold_answer,
    source_idx: int,
    distractor_type: str = "clean",
    distractor_sentence = None,
    distractor_position = None,
):
    if isinstance(gold_answer, (int, float)):
        gold_str = str(int(gold_answer)) if gold_answer == int(gold_answer) else str(gold_answer)
    else:
        gold_str = str(gold_answer)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Solve this math problem:\n\n{question}"},
            {"role": "assistant", "content": solution},
        ],
        "_gold_answer": gold_str,
        "_source_idx": source_idx,
        "_attempt": 1,
        "_distractor_type": distractor_type,
        "_distractor_sentence": distractor_sentence,
        "_distractor_position": distractor_position,
    }


def _get_solution_from_distilled(rec: dict):
    for msg in rec.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content")
    return None


def _write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_training_sets(
    input_path: str = "gsm8k_distractors.json",
    distilled_path: str = "data/gsm8k_train_distilled.jsonl",
    mixed_path: str = "train_mixed.jsonl",
    noop_path: str = "train_noop.jsonl",
    in_topic_path: str = "train_in_topic.jsonl",
    off_topic_path: str = "train_off_topic.jsonl",
):
    rng = random.Random(SEED)


    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} distractor records loaded.")

    data_lookup = {rec["id"]: rec for rec in data}

    print(f"Loading {distilled_path}...")
    distilled_records: list[dict] = []
    with open(distilled_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                distilled_records.append(json.loads(line))
    print(f"  {len(distilled_records)} distilled records loaded.")

    solution_by_id: dict[int, str] = {}
    valid_ids: list[int] = []
    seen_ids: set[int] = set()

    for rec in distilled_records:
        source_id = rec.get("_source_idx")
        if source_id is None:
            continue
        if source_id in seen_ids:
            continue
        if source_id not in data_lookup:
            continue

        solution = _get_solution_from_distilled(rec)
        if not solution:
            continue

        solution_by_id[source_id] = solution
        valid_ids.append(source_id)
        seen_ids.add(source_id)

    print(f"  {len(valid_ids)} usable source ids after alignment.")

    available: dict[str, list[int]] = {
        "off_topic": [],
        "in_topic": [],
        "no_op": [],
    }

    for source_id in valid_ids:
        distractors = data_lookup[source_id].get("distractors", {})
        for dtype in available:
            if distractors.get(dtype):
                available[dtype].append(source_id)

    for dtype, ids in available.items():
        pct = 100 * len(ids) / len(valid_ids) if valid_ids else 0.0
        print(f"  {dtype:10}: {len(ids)} / {len(valid_ids)} ({pct:.1f}%)")

    def make_record(source_id: int, distractor_type: str) -> dict:
        orig = data_lookup[source_id]
        question = orig["question"]
        solution = solution_by_id[source_id]
        gold = orig["gold_answer"]

        if distractor_type == "clean":
            return to_training_example(
                question=question,
                solution=solution,
                gold_answer=gold,
                source_idx=source_id,
                distractor_type="clean",
                distractor_sentence=None,
                distractor_position=None,
            )

        distractor_sentence = orig["distractors"][distractor_type]
        modified_question, pos = insert_distractor(question, distractor_sentence)
        return to_training_example(
            question=modified_question,
            solution=solution,
            gold_answer=gold,
            source_idx=source_id,
            distractor_type=distractor_type,
            distractor_sentence=distractor_sentence,
            distractor_position=pos,
        )

    def build_exact_binary(dtype: str) -> tuple[list[dict], dict[str, int]]:

        distract_ids = available[dtype]
        max_k = min(len(distract_ids), len(valid_ids) // 2)

        chosen_clean: list[int] = []
        chosen_dist: list[int] = []

        for k in range(max_k, 0, -1):
            dist_sample = rng.sample(distract_ids, k)
            dist_set = set(dist_sample)

            clean_pool = [sid for sid in valid_ids if sid not in dist_set]
            if len(clean_pool) < k:
                continue

            clean_sample = rng.sample(clean_pool, k)
            chosen_clean = clean_sample
            chosen_dist = dist_sample
            break

        rows: list[dict] = []
        for source_id in chosen_clean:
            rows.append(make_record(source_id, "clean"))
        for source_id in chosen_dist:
            rows.append(make_record(source_id, dtype))

        rng.shuffle(rows)
        counts = {"clean": len(chosen_clean), dtype: len(chosen_dist)}
        return rows, counts

    def build_exact_mixed() -> tuple[list[dict], dict[str, int]]:

        max_k = min(
            len(valid_ids) // 4,
            len(available["off_topic"]),
            len(available["in_topic"]),
            len(available["no_op"]),
        )

        chosen: dict[str, list[int]] = {
            "clean": [],
            "off_topic": [],
            "in_topic": [],
            "no_op": [],
        }

        order = ["in_topic", "no_op", "off_topic"]

        for k in range(max_k, 0, -1):
            remaining = set(valid_ids)
            temp: dict[str, list[int]] = {
                "clean": [],
                "off_topic": [],
                "in_topic": [],
                "no_op": [],
            }
            ok = True

            for dtype in order:
                pool = [sid for sid in available[dtype] if sid in remaining]
                if len(pool) < k:
                    ok = False
                    break
                sample = rng.sample(pool, k)
                temp[dtype] = sample
                remaining.difference_update(sample)

            if not ok:
                continue

            if len(remaining) < k:
                continue

            temp["clean"] = rng.sample(list(remaining), k)
            chosen = temp
            break

        rows: list[dict] = []
        for source_id in chosen["clean"]:
            rows.append(make_record(source_id, "clean"))
        for source_id in chosen["off_topic"]:
            rows.append(make_record(source_id, "off_topic"))
        for source_id in chosen["in_topic"]:
            rows.append(make_record(source_id, "in_topic"))
        for source_id in chosen["no_op"]:
            rows.append(make_record(source_id, "no_op"))

        rng.shuffle(rows)
        counts = {key: len(value) for key, value in chosen.items()}
        return rows, counts

    train_mixed, mixed_counts = build_exact_mixed()
    train_in_topic, in_counts = build_exact_binary("in_topic")
    train_off_topic, off_counts = build_exact_binary("off_topic")
    train_noop, noop_counts = build_exact_binary("no_op")

    _write_jsonl(mixed_path, train_mixed)
    _write_jsonl(in_topic_path, train_in_topic)
    _write_jsonl(off_topic_path, train_off_topic)
    _write_jsonl(noop_path, train_noop)

    print(f"\n{'=' * 70}")
    print("Training sets created:")
    print(f"{'=' * 70}")
    print(f"  {mixed_path}:     {len(train_mixed)} examples")
    print(f"  {in_topic_path}:  {len(train_in_topic)} examples")
    print(f"  {off_topic_path}: {len(train_off_topic)} examples")
    print(f"  {noop_path}:      {len(train_noop)} examples")

    print(f"\n{'=' * 70}")
    print("Distribution breakdown:")
    print(f"{'=' * 70}")

    for label, counts in [
        ("Mixed set", mixed_counts),
        ("In-topic set", in_counts),
        ("Off-topic set", off_counts),
        ("No-op set", noop_counts),
    ]:
        total = sum(counts.values())
        print(f"\n{label}:")
        for dtype, count in counts.items():
            pct = 100 * count / total if total else 0.0
            print(f"  {dtype:12}: {count} ({pct:.1f}%)")


def verify_training_sets(
    mixed_path: str = "train_mixed.jsonl",
    noop_path: str = "train_noop.jsonl",
    in_topic_path: str = "train_in_topic.jsonl",
    off_topic_path: str = "train_off_topic.jsonl",
):
    print(f"\n{'=' * 70}")
    print("Verifying training set format...")
    print(f"{'=' * 70}")

    paths = [mixed_path, noop_path, in_topic_path, off_topic_path]

    for path in paths:
        if not Path(path).exists():
            print(f"[MISSING] {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        issues = []
        for i, rec in enumerate(data):
            if "messages" not in rec:
                issues.append(f"Record {i}: missing 'messages'")
                continue
            if "_gold_answer" not in rec:
                issues.append(f"Record {i}: missing '_gold_answer'")
            if "_distractor_type" not in rec:
                issues.append(f"Record {i}: missing '_distractor_type'")
            msgs = rec["messages"]
            if len(msgs) != 3:
                issues.append(f"Record {i}: expected 3 messages, got {len(msgs)}")
                continue
            for j, (msg, role) in enumerate(zip(msgs, ["system", "user", "assistant"])):
                if msg.get("role") != role:
                    issues.append(f"Record {i}: message {j} wrong role")
                if not msg.get("content"):
                    issues.append(f"Record {i}: message {j} empty content")

        status = "OK" if not issues else "ISSUES"
        print(f"[{status}] {path}: {len(data)} records")
        if issues:
            for issue in issues[:5]:
                print(f"    {issue}")
            if len(issues) > 5:
                print(f"    ... and {len(issues) - 5} more")

def to_test_example(
    question: str,
    solution: str,
    gold_answer,
    source_idx: int,
    distractor_type: str,
    distractor_sentence = None,
    distractor_position = None,
):
    return {
        "id": source_idx,
        "question": question,
        "solution": solution,
        "gold_answer": gold_answer,
        "distractor_type": distractor_type,
        "distractor_sentence": distractor_sentence,
        "distractor_position": distractor_position,
    }


def build_test_sets(
    input_path: str = "gsm8k_test_distractors.json",
    off_topic_path: str = "test_off_topic.jsonl",
    in_topic_path: str = "test_in_topic.jsonl",
    noop_path: str = "test_noop.jsonl",
):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} test distractor records loaded.")

    test_off_topic: list[dict] = []
    test_in_topic: list[dict] = []
    test_noop: list[dict] = []

    for rec in data:
        source_id = rec["id"]
        question = rec["question"]
        solution = rec["solution"]
        gold = rec["gold_answer"]
        distractors = rec.get("distractors", {})

        d_off = distractors.get("off_topic")
        if d_off:
            q_off, pos_off = insert_distractor(question, d_off)
            test_off_topic.append(to_test_example(
                question=q_off,
                solution=solution,
                gold_answer=gold,
                source_idx=source_id,
                distractor_type="off_topic",
                distractor_sentence=d_off,
                distractor_position=pos_off,
            ))

        d_in = distractors.get("in_topic")
        if d_in:
            q_in, pos_in = insert_distractor(question, d_in)
            test_in_topic.append(to_test_example(
                question=q_in,
                solution=solution,
                gold_answer=gold,
                source_idx=source_id,
                distractor_type="in_topic",
                distractor_sentence=d_in,
                distractor_position=pos_in,
            ))

        d_noop = distractors.get("no_op")
        if d_noop:
            q_noop, pos_noop = insert_distractor(question, d_noop)
            test_noop.append(to_test_example(
                question=q_noop,
                solution=solution,
                gold_answer=gold,
                source_idx=source_id,
                distractor_type="no_op",
                distractor_sentence=d_noop,
                distractor_position=pos_noop,
            ))

    _write_jsonl(off_topic_path, test_off_topic)
    _write_jsonl(in_topic_path, test_in_topic)
    _write_jsonl(noop_path, test_noop)

    print("Test sets created:")

    print(f"  {off_topic_path}: {len(test_off_topic)} examples")
    print(f"  {in_topic_path}:  {len(test_in_topic)} examples")
    print(f"  {noop_path}:      {len(test_noop)} examples")

if __name__ == "__main__":
    build_training_sets(
        input_path="data/distractors/gsm8k_test_distractors.json",
        distilled_path="data/training/gsm8k_train_distilled.jsonl",
        mixed_path="train_mixed.jsonl",
        noop_path="train_noop.jsonl",
        in_topic_path="train_in_topic.jsonl",
        off_topic_path="train_off_topic.jsonl",
    )
    verify_training_sets()

    print()

    build_test_sets(
        input_path="data/distractors/gsm8k_test_distractors.json",
        off_topic_path="test_off_topic.jsonl",
        in_topic_path="test_in_topic.jsonl",
        noop_path="test_noop.jsonl",
    )