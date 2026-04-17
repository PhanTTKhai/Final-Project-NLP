from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve math problems step by step. "
    "Focus only on information relevant to solving the problem and "
    "ignore any irrelevant details."
)


def load_chat_data(data_path: str, val_ratio: float = 0.05) -> tuple[Dataset, Dataset]:
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            if "messages" in ex:
                examples.append(ex)
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Solve this math problem:\n\n{ex['question']}"},
                    {
                        "role": "assistant",
                        "content": ex.get("solution", f"The answer is {ex['answer']}."),
                    },
                ]
                examples.append({"messages": messages})

    dataset = Dataset.from_list(examples)
    split = dataset.train_test_split(test_size=val_ratio, seed=42)
    print(f"Loaded {len(split['train'])} train / {len(split['test'])} val examples from {data_path}")
    return split["train"], split["test"]


def train_full(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-5,
    max_seq_length: int = 1024,
    val_ratio: float = 0.05,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Full Fine-Tuning: {model_name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    train_dataset, val_dataset = load_chat_data(data_path, val_ratio=val_ratio)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        max_length=max_seq_length,
        report_to="none",
        seed=42,
        dataloader_num_workers=4,
        # validation
        eval_strategy="epoch",
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models on GSM8K + distractors")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (chat-format JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction held out for validation")
    args = parser.parse_args()

    train_full(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
