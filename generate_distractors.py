from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from distractor_utils import insert_distractors

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve math problems step by step. "
    "Focus only on information relevant to solving the problem and "
    "ignore any irrelevant details."
)

def build_mixed_matched(records: list[dict[str, Any]], seed: int = 42):
    clean = [r for r in records if r.get("distractor_type") == "none"]
    off_topic = [r for r in records if r.get("distractor_type") == "off_topic"]
    in_topic = [r for r in records if r.get("distractor_type") == "in_topic"]
    noop = [r for r in records if r.get("distractor_type") == "noop"]

    if not clean:
        raise ValueError("No clean examples found.")
    if not off_topic or not in_topic or not noop:
        raise ValueError("Missing one or more distractor types for mixed_matched.")

    rng = random.Random(seed)

    n_clean = len(clean)
    base = n_clean // 3
    remainder = n_clean % 3

    n_off_topic = base + (1 if remainder > 0 else 0)
    n_in_topic = base + (1 if remainder > 1 else 0)
    n_noop = base

    mixed_matched = (
        list(clean)
        + rng.sample(off_topic, n_off_topic)
        + rng.sample(in_topic, n_in_topic)
        + rng.sample(noop, n_noop)
    )

    rng.shuffle(mixed_matched)
    return mixed_matched

def generate_distractors_for_dataset(
    input_path: Path,
    output_path: Path,
    n_distractors: int = 1,
):
    distractor_types = ["off_topic", "in_topic", "noop"]

    examples = [json.loads(line) for line in input_path.open()]
    print(f"Generating distractors for {len(examples)} examples...")

    augmented: list[dict[str, Any]] = []
    for ex in tqdm(examples):
        clean = dict(ex)
        clean.update({
            "has_distractor": False,
            "distractor_type": "none",
            "distractor_position": [],
            "n_distractors": 0,
        })
        augmented.append(clean)

        for dtype in distractor_types:
            modified_question, positions = insert_distractors(
                ex["question"], distractor_type=dtype, n_distractors=n_distractors
            )
            distracted = dict(ex)
            distracted["question"] = modified_question
            distracted["has_distractor"] = True
            distracted["distractor_type"] = dtype
            distracted["distractor_position"] = positions
            distracted["n_distractors"] = n_distractors
            augmented.append(distracted)

    random.shuffle(augmented)
    with output_path.open("w") as f:
        for ex in augmented:
            f.write(json.dumps(ex) + "\n")

    print(
        f"Saved {len(augmented)} examples "
        f"({len(examples)} clean + {len(examples) * len(distractor_types)} distracted) to {output_path}"
    )

    hard_path = output_path.parent / output_path.name.replace(".jsonl", "_hard.jsonl")
    hard_examples = [ex for ex in augmented if ex["distractor_type"] in ("none", "noop")]
    random.shuffle(hard_examples)
    with hard_path.open("w") as f:
        for ex in hard_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(hard_examples)} hard-distractor examples to {hard_path}")

    return augmented


def create_training_prompt(example: dict[str, Any]) -> dict[str, Any]:
    question = example["question"]
    solution = example.get("solution", "")
    answer = example["answer"]

    prompt = (
        "Solve the following math problem step by step. "
        "Ignore any information that is not needed to solve the problem.\n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )
    completion = f" {solution.replace('####', '\n####').strip()}" if solution else f" The answer is {answer}."

    return {
        "prompt": prompt,
        "completion": completion,
        "answer": example["answer"],
        "distractor_type": example.get("distractor_type", "none"),
        "distractor_position": example.get("distractor_position", []),
        "n_distractors": example.get("n_distractors", 0),
    }


def create_chat_format(example: dict[str, Any]):
    question = example["question"]
    solution = example.get("solution", "")
    answer = example["answer"]
    assistant_msg = solution if solution else f"The answer is {answer}."

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Solve this math problem:\n\n{question}"},
            {"role": "assistant", "content": assistant_msg},
        ],
        "answer": answer,
        "distractor_type": example.get("distractor_type", "none"),
        "distractor_position": example.get("distractor_position", []),
        "n_distractors": example.get("n_distractors", 0),
    }


def prepare_training_files(data_path: Path, output_dir: Path, label: str):
    examples = [json.loads(line) for line in data_path.open()]

    completion_path = output_dir / f"train_completion_{label}.jsonl"
    with completion_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(create_training_prompt(ex)) + "\n")
    print(f"Saved completion format to {completion_path}")

    chat_path = output_dir / f"train_chat_{label}.jsonl"
    with chat_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(create_chat_format(ex)) + "\n")
    print(f"Saved chat format to {chat_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distractor-augmented GSM8K training data")
    parser.add_argument("--input", type=str, default="./data/gsm8k_train.jsonl")
    parser.add_argument("--output", type=str, default="./data/gsm8k_train_distracted.jsonl")
    parser.add_argument("--n_distractors", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    augmented = generate_distractors_for_dataset(
        input_path=input_path,
        output_path=output_path,
        n_distractors=args.n_distractors,
    )

    mixed_matched = build_mixed_matched(augmented, seed=args.seed)
    mixed_matched_path = output_path.parent / output_path.name.replace(".jsonl", "_mixed_matched.jsonl")

    with mixed_matched_path.open("w") as f:
        for ex in mixed_matched:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(mixed_matched)} size-matched mixed examples to {mixed_matched_path}")

    prepare_training_files(input_path, output_path.parent, label="clean")
    prepare_training_files(output_path, output_path.parent, label="mixed")
    prepare_training_files(mixed_matched_path, output_path.parent, label="mixed_matched")

    hard_path = output_path.parent / output_path.name.replace(".jsonl", "_hard.jsonl")
    prepare_training_files(hard_path, output_path.parent, label="hard")


if __name__ == "__main__":
    main()
