"""Self-distillation: use base Qwen2.5-Instruct to regenerate GSM8K training
solutions in its natural verbose style, keep only the correct ones, and write
them as chat-format JSONL ready for train.py.

Method: Rejection-sampling fine-tuning (RFT) / self-distillation.
  1. For each train question, generate up to N attempts (1 greedy + N-1 sampled).
  2. Extract the predicted answer from each attempt.
  3. Keep the first attempt that matches the gold answer.
  4. Save as chat-format JSONL with the new verbose solution as the assistant turn.

Output: data/gsm8k_train_distilled.jsonl (chat format, drop-in for train.py).
"""
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


# answer extraction (copied from evaluate.py for consistency)

def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from model output."""
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


def answers_match(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-3
    except ValueError:
        return predicted.strip().lower() == gold.strip().lower()


def extract_gold_from_gsm8k(answer_text: str) -> str:
    """GSM8K gold answers end in '#### <number>'."""
    return answer_text.split("####")[-1].replace(",", "").strip()


# generation

def format_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this math problem:\n\n{question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
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
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    output_ids = model.generate(**gen_kwargs)

    prompt_len = encodings["input_ids"].shape[1]
    responses = []
    for out in output_ids:
        response_ids = out[prompt_len:]
        responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))
    return responses


# main loop

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-distill GSM8K training data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Teacher model for generating solutions")
    parser.add_argument("--output", type=str, default="data/gsm8k_train_distilled.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--attempts_per_question", type=int, default=3,
                        help="Total attempts per question (1 greedy + rest sampled)")
    parser.add_argument("--sampling_temperature", type=float, default=0.7,
                        help="Temperature for non-greedy attempts")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N examples (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions already in output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load data  
    print(f"Loading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.limit:
        ds = ds.select(range(args.limit))
    total = len(ds)
    print(f"  {total} training questions")

    # Build list of (idx, question, gold) to process
    questions = []
    for i, row in enumerate(ds):
        gold = extract_gold_from_gsm8k(row["answer"])
        questions.append({"idx": i, "question": row["question"], "gold": gold})

    # Resume support  
    already_done: set[int] = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_done.add(rec["_source_idx"])
                except Exception:
                    pass
        print(f"  Resuming: {len(already_done)} examples already in {args.output}")
        questions = [q for q in questions if q["idx"] not in already_done]

    # Load model  
    print(f"Loading teacher model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Main loop: for each attempt, batch-generate over unsolved questions  
    # Track solved questions; once solved, stop generating for them.
    solved: dict[int, dict] = {}  # idx -> record to write
    unsolved = list(questions)

    # Open output in append mode so we stream results
    output_file = open(args.output, "a", encoding="utf-8")

    total_attempts_made = 0
    t0 = time.time()

    for attempt_num in range(args.attempts_per_question):
        if not unsolved:
            break
        temperature = 0.0 if attempt_num == 0 else args.sampling_temperature
        style = "greedy" if temperature == 0 else f"sampled (T={temperature})"
        print(f"\n--- Attempt {attempt_num + 1}/{args.attempts_per_question} ({style}) "
              f"on {len(unsolved)} unsolved questions ---")

        newly_solved = 0
        pbar = tqdm(range(0, len(unsolved), args.batch_size),
                    desc=f"Attempt {attempt_num + 1}")
        for i in pbar:
            batch = unsolved[i : i + args.batch_size]
            prompts = [format_prompt(tokenizer, q["question"]) for q in batch]
            responses = generate_batch(
                model, tokenizer, prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
            )
            total_attempts_made += len(batch)

            for q, response in zip(batch, responses):
                predicted = extract_answer(response)
                if answers_match(predicted, q["gold"]):
                    # Build chat-format record, matching download_data.py schema
                    record = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Solve this math problem:\n\n{q['question']}"},
                            {"role": "assistant", "content": response.strip()},
                        ],
                        "_source_idx": q["idx"],
                        "_gold_answer": q["gold"],
                        "_attempt": attempt_num + 1,
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    solved[q["idx"]] = record
                    newly_solved += 1

            pbar.set_postfix(solved_this_attempt=newly_solved,
                             total_solved=len(solved) + len(already_done))

        # Rebuild unsolved list
        unsolved = [q for q in unsolved if q["idx"] not in solved]
        elapsed = time.time() - t0
        print(f"  Attempt {attempt_num + 1}: solved {newly_solved} new, "
              f"{len(unsolved)} still unsolved. Elapsed: {elapsed/60:.1f} min")

    output_file.close()

    # Summary  
    total_solved = len(solved) + len(already_done)
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"{'=' * 60}")
    print(f"  Questions processed:     {total}")
    print(f"  Solved:                  {total_solved} ({100 * total_solved / total:.1f}%)")
    print(f"  Unsolved (discarded):    {len(unsolved)}")
    print(f"  Total generation calls:  {total_attempts_made}")
    print(f"  Output:                  {args.output}")
    print()
    print(f"Next step: train.py --data {args.output}")


if __name__ == "__main__":
    main()
