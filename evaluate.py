"""Evaluate fine-tuned Qwen models across GSM8K, GSM-DC, GSM-Plus, and custom benchmarks."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve math problems step by step. "
    "Focus only on information relevant to solving the problem and "
    "ignore any irrelevant details."
)


# Answer extraction

def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from model output.

    Tries several patterns in priority order:
      1. #### <number>          (GSM8K convention)
      2. "the answer is <number>"
      3. \\boxed{<number>}
      4. last number in the text (fallback)
    """
    # Normalise commas inside numbers: "1,234" -> "1234"
    def _clean(num_str: str) -> str:
        return num_str.replace(",", "").strip().rstrip(".")

    # Pattern 1: #### <number>
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return _clean(m.group(1))

    # Pattern 2: "the answer is <number>"
    m = re.search(r"(?:the\s+)?answer\s+is[:\s]*([+-]?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        return _clean(m.group(1))

    # Pattern 3: \boxed{<number>}
    m = re.search(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}", text)
    if m:
        return _clean(m.group(1))

    # Fallback: last number in text
    nums = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if nums:
        return _clean(nums[-1])

    return None


def normalize_answer(answer: str | int | float) -> str:
    """Normalize a gold answer to a comparable string."""
    s = str(answer).replace(",", "").strip().rstrip(".")
    # Remove the #### prefix if present
    s = re.sub(r"^#+\s*", "", s)
    return s


def answers_match(predicted: str | None, gold: str) -> bool:
    """Compare predicted and gold answers with tolerance for floats."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-3
    except ValueError:
        return predicted.strip().lower() == gold.strip().lower()


# Benchmark loaders

def load_gsm8k_test() -> list[dict]:
    """Load the standard GSM8K test split from HuggingFace."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for row in ds:
        # Gold answer is after ####
        gold = row["answer"].split("####")[-1].strip()
        examples.append({"question": row["question"], "gold": gold, "source": "gsm8k"})
    return examples


def load_jsonl_benchmark(path: str, source_name: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            question = ex.get("question") or ex.get("problem") or ex.get("input", "")

            # Explicitly check each field; 0.0 must not fall through
            gold_raw = None
            for key in ("gold_answer", "answer", "gold", "target", "solution"):
                if key in ex and ex[key] is not None:
                    gold_raw = ex[key]
                    break

            if isinstance(gold_raw, str) and "####" in gold_raw:
                gold_raw = gold_raw.split("####")[-1].strip()

            examples.append({
                "question": question,
                "gold": str(gold_raw),
                "source": source_name,
            })
    return examples


# Generation

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """Generate completions for a batch of prompts."""
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    gen_kwargs = dict(
        **encodings,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.95 if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )

    output_ids = model.generate(**gen_kwargs)

    # Decode only the new tokens
    responses = []
    for i, out in enumerate(output_ids):
        prompt_len = encodings["input_ids"].shape[1]
        response_ids = out[prompt_len:]
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return responses


def format_prompt(tokenizer, question: str) -> str:
    """Build the chat-template prompt for a single question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this math problem:\n\n{question}"},
    ]
    # apply_chat_template returns a string ready for tokenization
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# Evaluation loop

def evaluate_benchmark(
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> dict:
    """Run evaluation on a list of examples and return metrics."""
    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(examples), batch_size), desc=f"Evaluating {examples[0]['source']}"):
        batch = examples[i : i + batch_size]
        prompts = [format_prompt(tokenizer, ex["question"]) for ex in batch]
        responses = generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)

        for ex, response in zip(batch, responses):
            predicted = extract_answer(response)
            gold = normalize_answer(ex["gold"])
            is_correct = answers_match(predicted, gold)
            correct += int(is_correct)
            total += 1
            results.append({
                "question": ex["question"],  # truncate for readability
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
                "response": response,
            })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "source": examples[0]["source"] if examples else "unknown",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models on math benchmarks")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model or HF model name")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help=(
            "Benchmarks to evaluate on. Use 'gsm8k' for HF GSM8K test set, "
            "or provide paths to JSONL files. Format: name:path (e.g. gsm_dc:data/gsm_dc.jsonl) "
            "or just a path (name derived from filename)."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default=None, help="Path to save detailed results JSON")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # important for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load benchmarks
    all_benchmarks: list[tuple[str, list[dict]]] = []

    for spec in args.benchmarks:
        if spec == "gsm8k":
            all_benchmarks.append(("gsm8k", load_gsm8k_test()))
        elif ":" in spec:
            name, path = spec.split(":", 1)
            all_benchmarks.append((name, load_jsonl_benchmark(path, name)))
        else:
            name = Path(spec).stem
            all_benchmarks.append((name, load_jsonl_benchmark(spec, name)))

    # Evaluate
    summary = {}
    all_results = {}

    for name, examples in all_benchmarks:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {name} ({len(examples)} examples)")
        print(f"{'=' * 60}")

        t0 = time.time()
        metrics = evaluate_benchmark(
            model, tokenizer, examples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0

        summary[name] = {
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "time_seconds": round(elapsed, 1),
        }
        all_results[name] = metrics["results"]

        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"  Time: {elapsed:.1f}s")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Benchmark':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 55)
    for name, s in summary.items():
        print(f"{name:<20} {s['accuracy']:>10.4f} {s['correct']:>10} {s['total']:>10}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "model": args.model,
            "summary": summary,
            "detailed_results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
