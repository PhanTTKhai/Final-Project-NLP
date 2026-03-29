from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from distractor_utils import insert_distractors

#Some Keywords that suggest that the question is info-dense and that it cant be removed
INFO_DENSE_HINTS = {
    "each", "every", "per", "times", "more", "less", "left", "remaining",
    "total", "altogether", "then", "after", "before", "gave", "bought", "sold",
    "twice", "half", "double", "sum", "difference",
}


def extract_answer(answer_text: str):
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return str(answer_text).strip()


def count_steps(solution: str) -> int:
    lines = [
        line.strip()
        for line in solution.strip().split("\n")
        if line.strip() and not line.strip().startswith("####")
    ]
    return len(lines)


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def sentence_has_quantity_or_relation(sentence: str):
    lowered = sentence.lower()
    has_number = bool(re.search(r"\b\d+\.?\d*\b", sentence))
    has_hint = any(token in lowered for token in INFO_DENSE_HINTS)
    return has_number or has_hint


def is_info_dense_candidate(example: dict[str, Any]):
    if example.get("n_steps", 0) < 4:
        return False
    sentences = split_sentences(example["question"])
    if len(sentences) < 2:
        return False
    return all(sentence_has_quantity_or_relation(s) for s in sentences)


def format_gsm8k_example(example: dict[str, Any], split: str):
    return {
        "question": example["question"],
        "solution": example["answer"],
        "answer": extract_answer(example["answer"]),
        "n_steps": count_steps(example["answer"]),
        "source": "gsm8k",
        "split": split,
        "has_distractor": False,
        "distractor_type": "none",
        "distractor_position": [],
        "n_distractors": 0,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]):
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1
    return count


def normalize_benchmark_example(ex: dict[str, Any], source: str, extra: dict[str, Any] | None = None):
    question = ex.get("question") or ex.get("input") or ex.get("problem") or ex.get("prompt")
    raw_answer = ex.get("answer")
    if raw_answer is None:
        raw_answer = ex.get("target") or ex.get("final_answer") or ex.get("output")
    if question is None or raw_answer is None:
        return None

    row = {
        "question": str(question),
        "answer": extract_answer(str(raw_answer)),
        "solution": str(ex.get("solution", ex.get("rationale", ""))),
        "source": source,
        "split": "test",
        "has_distractor": True,
        "distractor_type": ex.get("distractor_type", source),
        "distractor_position": ex.get("distractor_position", []),
        "n_distractors": ex.get("n_distractors", 1),
    }
    if extra:
        row.update(extra)
    return row


def download_gsm8k(output_dir: Path):
    ds = load_dataset("openai/gsm8k", "main")

    train_examples = [format_gsm8k_example(ex, "train") for ex in ds["train"]]
    test_examples = [format_gsm8k_example(ex, "test") for ex in ds["test"]]

    train_path = output_dir / "gsm8k_train.jsonl"
    test_path = output_dir / "gsm8k_test.jsonl"
    print(f"  Saved {write_jsonl(train_path, train_examples)} train examples to {train_path}")
    print(f"  Saved {write_jsonl(test_path, test_examples)} test examples to {test_path}")
    return train_examples, test_examples


def create_template_test_set(test_examples: list[dict[str, Any]], output_dir: Path, n_distractors: int = 1):
    rows: list[dict[str, Any]] = []
    for ex in test_examples:
        for dtype in ["off_topic", "in_topic", "noop"]:
            modified_question, positions = insert_distractors(
                ex["question"], distractor_type=dtype, n_distractors=n_distractors
            )
            new_ex = dict(ex)
            new_ex["question"] = modified_question
            new_ex["source"] = "template_distracted"
            new_ex["has_distractor"] = True
            new_ex["distractor_type"] = dtype
            new_ex["distractor_position"] = positions
            new_ex["n_distractors"] = n_distractors
            rows.append(new_ex)

    path = output_dir / "gsm8k_test_template_distracted.jsonl"
    print(f"  Saved {write_jsonl(path, rows)} template-distracted test examples to {path}")


def create_info_dense_subset(test_examples: list[dict[str, Any]], output_dir: Path, overwrite_verified: bool = False):
    candidates = [ex for ex in test_examples if is_info_dense_candidate(ex)]
    candidate_path = output_dir / "gsm8k_test_info_dense_candidates.jsonl"
    final_path = output_dir / "gsm8k_test_info_dense.jsonl"

    print(f"Saved {write_jsonl(candidate_path, candidates)} info-dense candidates to {candidate_path}")
    if overwrite_verified or not final_path.exists():
        print(f"{final_path.name} Manually verify this subset")
        write_jsonl(final_path, candidates)
    else:
        print(f"  Keeping existing verified info-dense subset at {final_path}")


def download_gsm_plus(output_dir: Path):
    rows: list[dict[str, Any]] = []

    try:
        ds = load_dataset("qintongli/GSM-Plus")
        for split_name in ds.keys():
            for ex in ds[split_name]:
                ptype = ex.get("perturbation_type") or ex.get("type") or ex.get("category")
                if ptype != "distractor_insertion":
                    continue
                row = normalize_benchmark_example(
                    ex,
                    source="gsm_plus_distractor",
                    extra={"perturbation_type": ptype},
                )
                if row is not None:
                    rows.append(row)
    except Exception as exc:
        print(f" failed to load: {exc}")

    path = output_dir / "gsm_plus_distractor.jsonl"
    if rows:
        print(f"Saved {write_jsonl(path, rows)} GSM-Plus distractor examples to {path}")
    else:
        print("GSM-Plus distractor subset could not be downloaded.")
    return rows


def download_gsm_dc(output_dir: Path, local_path: str | None = None):
    rows: list[dict[str, Any]] = []

    if local_path:
        local_file = Path(local_path)
        if local_file.exists():
            for line in local_file.open():
                ex = json.loads(line)
                row = normalize_benchmark_example(ex, source="gsm_dc")
                if row is not None:
                    rows.append(row)

    if not rows:
        candidate_ids = ["yminglai/GSM-DC"]
        for dataset_id in candidate_ids:
            try:
                ds = load_dataset(dataset_id)
                for split_name in ds.keys():
                    for ex in ds[split_name]:
                        row = normalize_benchmark_example(ex, source="gsm_dc")
                        if row is not None:
                            rows.append(row)
                if rows:
                    break
            except Exception as exc:
                print(f"  Warning: failed to load {dataset_id}: {exc}")

    path = output_dir / "gsm_dc.jsonl"
    if rows:
        print(f"Saved {write_jsonl(path, rows)} GSM-DC examples to {path}")
    else:
        print("GSM-DC could not be downloaded automatically")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for the irrelevant-context project")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument(
        "--overwrite_info_dense",
        action="store_true",
    )
    parser.add_argument(
        "--gsm_dc_local",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples, test_examples = download_gsm8k(output_dir)
    create_template_test_set(test_examples, output_dir)
    create_info_dense_subset(test_examples, output_dir, overwrite_verified=args.overwrite_info_dense)
    download_gsm_plus(output_dir)
    download_gsm_dc(output_dir, local_path=args.gsm_dc_local)

    for file_path in sorted(output_dir.glob("*.jsonl")):
        n_rows = sum(1 for _ in file_path.open())
        print(f"  {file_path.name}: {n_rows} examples")


if __name__ == "__main__":
    main()
