"""
Task #5 - Prepare Benchmark Evaluation Sets

Prepares evaluation sets used in the paper.

ACTIVE:
  1. GSM-Plus (distractor-insertion subset)  - OOD evaluation
  2. GSM-DC                                  - OOD evaluation
  3. Info-Dense GSM8K subset                 - over-filtering evaluation

TODO:
  4. Template-distracted GSM8K test set      - needs Task #10
"""

import json
import re
from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("eval_sets")
OUTPUT_DIR.mkdir(exist_ok=True)

ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUMERIC_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_answer_from_text(text: str) -> float | None:
    """Extract numeric answer after ####. Returns None if not found."""
    m = ANSWER_PATTERN.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def coerce_to_float(value) -> float | None:
    """Convert anything-like-a-number to float, handling #### format and commas."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parsed = parse_answer_from_text(value)
        if parsed is not None:
            return parsed
        m = NUMERIC_PATTERN.search(value)
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                pass
    return None

def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save a list of dicts as JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---- Word-form number extraction (for info-dense filter) -------------------
# Cardinal number words common in GSM8K. Fractional words ('half', 'quarter')
# are intentionally EXCLUDED because they express relations (/2, /4) rather
# than absolute quantities, and rarely appear literally in solution arithmetic.

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "dozen": 12}

_ALL_NUM_WORDS = list(_ONES) + list(_TENS) + list(_SCALES)
_WORD_NUM_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ALL_NUM_WORDS + ["and", "a"], key=len, reverse=True)) + r")"
    r"(?:[-\s]+(?:" + "|".join(sorted(_ALL_NUM_WORDS + ["and", "a"], key=len, reverse=True)) + r"))*\b",
    re.IGNORECASE,
)
_DIGIT_NUM_RE = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)")
_ABBREV = {"Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr"}


def _parse_word_number(phrase: str) -> float | None:
    """Convert a phrase like 'twenty-three' or 'a dozen' to a number."""
    tokens = re.split(r"[-\s]+", phrase.lower())
    tokens = [t for t in tokens if t and t not in {"and"}]
    if not tokens or tokens == ["a"]:
        return None

    total = 0
    current = 0
    saw_number = False
    for tok in tokens:
        if tok == "a":
            current = current or 1
            continue
        if tok in _ONES:
            current += _ONES[tok]
            saw_number = True
        elif tok in _TENS:
            current += _TENS[tok]
            saw_number = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            current = (current or 1) * scale
            total += current
            current = 0
            saw_number = True
        else:
            return None
    return float(total + current) if saw_number else None


def extract_numbers(text: str) -> set[float]:
    """Extract all numbers (digit + word form) from text as a set of floats."""
    nums: set[float] = set()

    for m in _DIGIT_NUM_RE.finditer(text):
        try:
            nums.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass

    for m in _WORD_NUM_RE.finditer(text):
        parsed = _parse_word_number(m.group(0))
        if parsed is not None and parsed > 0:
            nums.add(parsed)

    return nums


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    protected = text
    for abbr in _ABBREV:
        protected = re.sub(rf"\b{abbr}\.", f"{abbr}__DOT__", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    return [p.replace("__DOT__", ".") for p in parts if p.strip()]


def is_question_sentence(sentence: str) -> bool:
    """True if this sentence is asking the question (vs providing info)."""
    s = sentence.strip()
    if s.endswith("?"):
        return True
    s_lower = s.lower()
    interrogative_starts = (
        "how many", "how much", "how far", "how long", "how often",
        "what is", "what are", "what was", "what will",
        "find ", "calculate ", "determine ", "compute ",
    )
    return s_lower.startswith(interrogative_starts)


# ---- 1. GSM-Plus (distractor-insertion subset) -----------------------------

def prepare_gsm_plus(output_path: Path) -> list[dict]:
    """
    Load GSM-Plus from HuggingFace and keep only the distractor-insertion subset.

    GSM-Plus has 8 perturbation types across 10,552 rows. The distractor-insertion
    subset has 1,319 rows (one per original GSM8K test question).

    Schema: question, solution, answer, perturbation_type, seed_*.
    Paper: Li et al. 2024 (ACL).
    """
    print("Loading GSM-Plus...")
    ds = load_dataset("qintongli/GSM-Plus", split="test")
    print(f"  Total examples: {len(ds)}")
    print(f"  Columns: {ds.column_names}")

    if "perturbation_type" not in ds.column_names:
        raise KeyError(
            f"Expected 'perturbation_type' field in GSM-Plus; got {ds.column_names}. "
            "Inspect ds[0] and adjust the filter key."
        )

    unique_perturbations = sorted(set(ds["perturbation_type"]))
    print(f"  Perturbation types found: {unique_perturbations}")

    # Main dataset uses space; formatted mirrors use underscore. Accept both.
    target_values = {"distraction insertion", "distraction_insertion"}

    distractor_subset = []
    for ex in ds:
        pt = ex.get("perturbation_type")
        if pt not in target_values:
            continue
        distractor_subset.append({
            "id": len(distractor_subset),       # running counter -> dense 0..N-1
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_answer": coerce_to_float(ex["answer"]),
            "perturbation_type": pt,
        })

    if not distractor_subset:
        raise RuntimeError(
            "GSM-Plus filter produced 0 rows. Check perturbation_type values above "
            "against the target set {distraction insertion, distraction_insertion}."
        )

    with_gold = sum(1 for r in distractor_subset if r["gold_answer"] is not None)
    print(f"  Distractor subset: {len(distractor_subset)} examples "
          f"({with_gold} with parseable gold answer)")

    save_jsonl(distractor_subset, output_path)
    print(f"  Saved -> {output_path}")
    return distractor_subset


# ---- 2. GSM-DC -------------------------------------------------------------

def prepare_gsm_dc(output_path: Path) -> list[dict]:
    """
    Load GSM-DC from HuggingFace.
    DAG-based math problems with controlled distractor-node injection.

    Paper: Yang et al. 2025 (EMNLP).
    HuggingFace: YMinglai/GSM-DC-Dataset-Sample (6,300 problems).
    """
    print("Loading GSM-DC...")
    try:
        ds = load_dataset("YMinglai/GSM-DC-Dataset-Sample", split="train")
    except Exception as e:
        print(f"  [error] Could not load GSM-DC: {e}")
        print("  Verify dataset name at https://github.com/yminglai/GSM-DC")
        save_jsonl([], output_path)
        return []

    print(f"  Total examples: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Sample row (truncated): {str(ds[0])[:500]}...")

    records = []
    for i, ex in enumerate(ds):
        problem = ex.get("problem_text") or ex.get("problem") or ex.get("question", "")
        solution = ex.get("solution_text") or ex.get("solution", "")
        if isinstance(problem, list):
            problem = " ".join(problem)
        if isinstance(solution, list):
            solution = " ".join(solution)

        info = ex.get("problem_info") or {}
        gold_raw = info.get("final_answer") if isinstance(info, dict) else None
        if gold_raw is None:
            gold_raw = ex.get("final_answer") or ex.get("answer")

        gold = coerce_to_float(gold_raw)

        records.append({
            "id": i,
            "question": problem,
            "solution": solution,
            "answer": gold,  # added for evaluate.py
            "gold_answer": gold,
            "irrelevant_nodes_count": ex.get("irrelevant_nodes_count"),
        })

    with_gold = sum(1 for r in records if r["gold_answer"] is not None)
    print(f"  Parsed: {len(records)} examples ({with_gold} with parseable gold answer)")
    if with_gold < len(records) * 0.9:
        print(f"  [warn] Low gold-answer coverage. Check printed schema above.")

    save_jsonl(records, output_path)
    print(f"  Saved -> {output_path}")
    return records


# ---- 3. Info-Dense GSM8K subset --------------------------------------------

def is_info_dense(
    question: str,
    solution: str,
    min_sentences: int = 3,
    max_setup_words: int = 15,
) -> tuple[bool, str]:
    """
    Check if every non-question sentence in the question contributes to the solution.

    A sentence contributes if:
      - It is the question sentence (ends with '?' or starts with interrogative)
      - It contains at least one number (digit or word-form) used in the solution
      - It has no numbers AND is short (<=max_setup_words) - treated as setup text

    A sentence fails if:
      - It has numbers, none of which appear in the solution (information added
        but not used)
      - It has no numbers but is long narrative (>max_setup_words words)

    Returns (is_dense, reason). `reason` is useful for debugging/auditing.
    """
    sol_text = solution.split("####")[0]
    sol_numbers = extract_numbers(sol_text)
    sentences = split_sentences(question)

    if len(sentences) < min_sentences:
        return False, f"only {len(sentences)} sentences (need {min_sentences}+)"

    for sent in sentences:
        if is_question_sentence(sent):
            continue

        sent_numbers = extract_numbers(sent)
        word_count = len(sent.split())

        if sent_numbers:
            if sent_numbers.isdisjoint(sol_numbers):
                return False, f"unused numbers {sent_numbers} in: '{sent[:80]}'"
        else:
            if word_count > max_setup_words:
                return False, f"long narrative ({word_count} words), no numbers: '{sent[:80]}'"

    return True, "all sentences contribute"


def prepare_info_dense(
    output_path: Path,
    min_sentences: int = 3,
    max_setup_words: int = 15,
) -> list[dict]:
    """
    Filter GSM8K test to problems where every sentence contributes to the solution.

    Used to test over-filtering: if a distractor-trained model learns to ignore
    relevant info, its accuracy on this set drops vs a clean-trained model.
    """
    print("Loading GSM8K test split for Info-Dense subset...")
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"  Total test examples: {len(ds)}")

    info_dense = []
    rejection_reasons: dict[str, int] = {}

    for i, ex in enumerate(ds):
        ok, reason = is_info_dense(
            ex["question"], ex["answer"],
            min_sentences=min_sentences,
            max_setup_words=max_setup_words,
        )
        if ok:
            info_dense.append({
                "id": i,
                "question": ex["question"],
                "answer": ex["answer"],
                "gold_answer": parse_answer_from_text(ex["answer"]),
            })
        else:
            # Categorize rejection reasons
            if "only" in reason and "sentences" in reason:
                category = "too few sentences"
            elif "unused numbers" in reason:
                category = "unused numbers"
            elif "long narrative" in reason:
                category = "long narrative, no numbers"
            else:
                category = "other"
            rejection_reasons[category] = rejection_reasons.get(category, 0) + 1

    print(f"  Info-Dense subset: {len(info_dense)} / {len(ds)} "
          f"({100 * len(info_dense) / len(ds):.1f}%)")
    print(f"  Rejection breakdown:")
    for cat, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    save_jsonl(info_dense, output_path)
    print(f"  Saved -> {output_path}")
    return info_dense


# ---- 4. Template-distracted GSM8K test set (TODO) --------------------------

# TODO: Implement after Task #10 (distractor generation) is done.

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing benchmark evaluation sets...")
    print("=" * 60)

    prepare_gsm_plus(OUTPUT_DIR / "gsm_plus_distractor.jsonl")
    print()

    prepare_gsm_dc(OUTPUT_DIR / "gsm_dc.jsonl")
    print()

    prepare_info_dense(OUTPUT_DIR / "info_dense.jsonl")
    print()
    print("Output files in ./eval_sets/")
    print("  gsm_plus_distractor.jsonl")
    print("  gsm_dc.jsonl")
    print("  info_dense.jsonl")

