"""Evaluate models on clean, distractor, and over-filtering conditions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve math problems step by step. "
    "Focus only on information relevant to solving the problem and ignore any irrelevant details."
)


def extract_numeric_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip().replace(",", "").rstrip(".")
    match = re.search(r"(?:the answer is|answer:)\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace(",", "").rstrip(".")
    match = re.search(r"=\s*([\d,]+\.?\d*)\s*$", text)
    if match:
        return match.group(1).strip().replace(",", "").rstrip(".")
    match = re.search(r"\\boxed\{(.+?)\}", text)
    if match:
        return match.group(1).strip().replace(",", "").rstrip(".")
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else text.strip()


def normalize_answer(answer: str) -> str:
    answer = answer.strip().replace(",", "").rstrip(".")
    try:
        val = float(answer)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return answer.lower().strip()


def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512, max_input_length: int = 1024) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this math problem:\n\n{question}"},
    ]

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = (
            "Solve the following math problem step by step. Ignore any information that is not needed to solve the problem.\n\n"
            f"Problem: {question}\n\nSolution:"
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_on_dataset(model, tokenizer, data_path: str, max_examples: int | None = None, max_new_tokens: int = 512) -> dict[str, Any]:
    examples = [json.loads(line) for line in Path(data_path).open()]
    if max_examples:
        examples = examples[:max_examples]

    correct = 0
    results: list[dict[str, Any]] = []
    per_type: dict[str, dict[str, int]] = {}

    for ex in tqdm(examples, desc=f"Evaluating on {Path(data_path).name}"):
        gold_answer = normalize_answer(str(ex["answer"]))
        response = generate_answer(model, tokenizer, ex["question"], max_new_tokens=max_new_tokens)
        predicted = normalize_answer(extract_numeric_answer(response))
        is_correct = predicted == gold_answer
        correct += int(is_correct)

        dtype = ex.get("distractor_type", "unknown")
        if dtype not in per_type:
            per_type[dtype] = {"correct": 0, "total": 0}
        per_type[dtype]["correct"] += int(is_correct)
        per_type[dtype]["total"] += 1

        results.append({
            "question": ex["question"][:200],
            "gold": gold_answer,
            "predicted": predicted,
            "correct": is_correct,
            "response": response[:500],
            "source": ex.get("source", "unknown"),
            "distractor_type": dtype,
            "distractor_position": ex.get("distractor_position", []),
            "n_distractors": ex.get("n_distractors", 0),
        })

    total = len(examples)
    accuracy = correct / total if total else 0.0
    per_type_metrics = {
        dtype: {
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            "correct": counts["correct"],
            "total": counts["total"],
        }
        for dtype, counts in per_type.items()
    }
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results, "per_type": per_type_metrics}


def infer_scale_key(model_name: str) -> str:
    lowered = model_name.lower()
    if "15b" in lowered or "1.5b" in lowered:
        return "15b"
    if "05b" in lowered or "0.5b" in lowered:
        return "05b"
    return "other"


def find_scale_baseline(model_names: list[str], scale_key: str) -> str | None:
    for name in model_names:
        lowered = name.lower()
        if infer_scale_key(name) == scale_key and "clean" in lowered:
            return name
    return None


def build_eval_dataset_map(data_dir: Path) -> dict[str, str]:
    dataset_files = {
        "gsm8k_clean": "gsm8k_test.jsonl",
        "gsm_plus_distractor": "gsm_plus_distractor.jsonl",
        "gsm_dc": "gsm_dc.jsonl",
        "template_distracted": "gsm8k_test_template_distracted.jsonl",
        "info_dense": "gsm8k_test_info_dense.jsonl",
    }
    found: dict[str, str] = {}
    for dataset_name, filename in dataset_files.items():
        path = data_dir / filename
        if path.exists():
            found[dataset_name] = str(path)
    return found


def compute_derived_metrics(pivot: pd.DataFrame) -> pd.DataFrame:
    model_names = list(pivot.index)
    rows: list[dict[str, Any]] = []

    for model_name in model_names:
        clean_acc = pivot.loc[model_name].get("gsm8k_clean")
        scale_key = infer_scale_key(model_name)
        baseline_name = find_scale_baseline(model_names, scale_key)
        baseline_info_dense = pivot.loc[baseline_name].get("info_dense") if baseline_name else None

        row: dict[str, Any] = {
            "model": model_name,
            "baseline_model": baseline_name,
            "gsm8k_clean": clean_acc,
            "info_dense_raw": pivot.loc[model_name].get("info_dense"),
            "info_dense_baseline": baseline_info_dense,
        }

        for dataset_name in ["gsm_plus_distractor", "gsm_dc", "template_distracted"]:
            noisy_acc = pivot.loc[model_name].get(dataset_name)
            row[dataset_name] = noisy_acc
            if pd.notna(clean_acc) and clean_acc not in (None, 0) and pd.notna(noisy_acc):
                row[f"R_{dataset_name}"] = noisy_acc / clean_acc
            else:
                row[f"R_{dataset_name}"] = None

        info_dense_acc = pivot.loc[model_name].get("info_dense")
        if baseline_name and pd.notna(info_dense_acc) and pd.notna(baseline_info_dense) and baseline_info_dense not in (None, 0):
            row["O_info_dense"] = info_dense_acc / baseline_info_dense
        else:
            row["O_info_dense"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def run_evaluation(model_paths: list[str], data_dir: str, output_dir: str, max_examples: int | None = None) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir_path = Path(data_dir)
    eval_datasets = build_eval_dataset_map(data_dir_path)

    print(f"\nEvaluation datasets found: {list(eval_datasets.keys())}")
    print(f"Models to evaluate: {model_paths}")

    all_results: list[dict[str, Any]] = []
    per_type_rows: list[dict[str, Any]] = []

    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\n{'=' * 60}\nEvaluating: {model_name}\n{'=' * 60}")

        model, tokenizer = load_model(model_path)

        for dataset_name, dataset_path in eval_datasets.items():
            print(f"\n  Dataset: {dataset_name}")
            metrics = evaluate_on_dataset(model, tokenizer, dataset_path, max_examples=max_examples)
            all_results.append({
                "model": model_name,
                "dataset": dataset_name,
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            })
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

            if dataset_name == "template_distracted":
                for dtype, dtype_metrics in metrics["per_type"].items():
                    per_type_rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "distractor_type": dtype,
                        "accuracy": dtype_metrics["accuracy"],
                        "correct": dtype_metrics["correct"],
                        "total": dtype_metrics["total"],
                    })
                    print(
                        f"    {dtype}: {dtype_metrics['accuracy']:.4f} "
                        f"({dtype_metrics['correct']}/{dtype_metrics['total']})"
                    )

            detail_path = output_path / f"{model_name}_{dataset_name}_details.jsonl"
            with detail_path.open("w") as f:
                for row in metrics["results"]:
                    f.write(json.dumps(row) + "\n")

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    pivot = df.pivot(index="model", columns="dataset", values="accuracy")
    summary_path = output_path / "summary.csv"
    pivot.to_csv(summary_path)

    derived = compute_derived_metrics(pivot)
    derived_path = output_path / "derived_metrics.csv"
    derived.to_csv(derived_path, index=False)

    if per_type_rows:
        per_type_df = pd.DataFrame(per_type_rows)
        per_type_path = output_path / "template_per_distractor_type.csv"
        per_type_df.to_csv(per_type_path, index=False)
    else:
        per_type_path = None

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    if not derived.empty:
        print("\n" + "-" * 40)
        print("DERIVED METRICS")
        print("-" * 40)
        display_cols = [
            c for c in [
                "baseline_model",
                "gsm8k_clean",
                "R_gsm_plus_distractor",
                "R_gsm_dc",
                "R_template_distracted",
                "info_dense_raw",
                "O_info_dense",
            ] if c in derived.columns
        ]
        print(derived.set_index("model")[display_cols].to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan"))

    if per_type_rows:
        print("\n" + "-" * 40)
        print("TEMPLATE DISTRACTOR TYPE BREAKDOWN")
        print("-" * 40)
        display_df = pd.DataFrame(per_type_rows).pivot(index="model", columns="distractor_type", values="accuracy")
        print(display_df.to_string(float_format=lambda x: f"{x:.4f}"))

    full_path = output_path / "all_results.json"
    with full_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    print(f"Derived metrics saved to {derived_path}")
    if per_type_path is not None:
        print(f"Per-distractor-type metrics saved to {per_type_path}")
    print(f"Full results saved to {full_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on clean, distractor, and over-filtering conditions")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to model directories")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(
        model_paths=args.models,
        data_dir=args.data_dir,
        output_dir=args.output,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
