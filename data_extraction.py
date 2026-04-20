"""
Extracts structured info from GSM8K training split:
  - sentences (with role labels)
  - numbers + units
  - named entities (variables, people, objects)
  - gold answer
  - distractor_hints: info needed by #10 to generate the 3 distractor types:
      off_topic  → topic keywords + person names (to generate unrelated sentences)
      in_topic   → numbers + units used in the problem (to generate plausible but wrong sentences)
      no_op      → candidate sentences that mention numbers NOT used in the gold solution
                   (look computationally relevant but are actually irrelevant)
Output: gsm8k_extracted.json  (one dict per problem)
"""

import re
import json
from datasets import load_dataset

# optional: spaCy for NER
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False
    print("[warn] spaCy not available – NER will be skipped. "
          "Run:  pip install spacy && python -m spacy download en_core_web_sm")


# regex patterns
NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)                        # not preceded by a word char
    (\d{1,3}(?:,\d{3})*|\d+)       # integer, optionally comma-grouped
    (?:\.\d+)?                     # optional decimal part
    (?!\w)                         # not followed by a word char
    """,
    re.VERBOSE,
)

UNIT_PATTERN = re.compile(
    r"""
    (\d[\d,.]*)\s*                 # leading number
    (dollars?|cents?|pounds?|kg|g|km|miles?|
     hours?|minutes?|seconds?|days?|weeks?|
     apples?|oranges?|boxes?|bags?|
     \$|€|£|%)                     # common units in GSM8K
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_numbers(text: str) -> list[dict]:
    """Return list of {value: float, raw: str, start: int, end: int}."""
    results = []
    for m in NUMBER_PATTERN.finditer(text):
        raw = m.group(0)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        results.append({"raw": raw, "value": value, "start": m.start(), "end": m.end()})
    return results


def extract_units(text: str) -> list[dict]:
    """Return list of {number: str, unit: str}."""
    return [{"number": m.group(1), "unit": m.group(2)} for m in UNIT_PATTERN.finditer(text)]


def extract_entities(text: str) -> list[dict]:
    """Return spaCy named entities if available, else []."""
    if not USE_SPACY:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter that handles math-word-problem quirks.
    Falls back to spaCy sentencizer if available.
    """
    if USE_SPACY:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # basic fallback: split on ". ", "? ", "! "
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def label_sentences(question: str, solution: str) -> list[dict]:
    """
    Assign each sentence a role:
      question  – from the problem statement
      solution  – from the solution text (before ####)
    """
    labeled = []
    for sent in split_sentences(question):
        labeled.append({"text": sent, "role": "question"})
    # solution text is everything before ####
    sol_text = solution.split("####")[0].strip()
    for sent in split_sentences(sol_text):
        labeled.append({"text": sent, "role": "solution"})
    return labeled


def parse_gold_answer(solution: str) -> float | None:
    """Extract the numeric answer after #### ."""
    m = ANSWER_PATTERN.search(solution)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


# distractor hint extraction

# common GSM8K topics, used to tag question theme for off-topic distractor gen
TOPIC_KEYWORDS = {
    "money":    r"\b(dollar|cent|price|cost|pay|earn|spend|sale|discount|profit|cheap|expensive|\$)\b",
    "time":     r"\b(hour|minute|second|day|week|month|year|morning|afternoon|evening|schedule)\b",
    "food":     r"\b(apple|orange|banana|cake|cookie|pizza|meal|eat|drink|food|fruit|vegetable)\b",
    "distance": r"\b(mile|km|kilometer|meter|walk|drive|run|travel|distance|far|close)\b",
    "school":   r"\b(student|teacher|class|school|grade|homework|exam|lesson|study)\b",
    "work":     r"\b(worker|employee|job|hour|salary|shift|factory|office|task|company)\b",
    "shopping": r"\b(buy|sell|store|shop|item|product|order|customer|bag|box)\b",
    "nature":   r"\b(tree|flower|garden|farm|animal|bird|fish|river|lake|mountain)\b",
}

def detect_topics(text: str) -> list[str]:
    """Return list of matched topic labels for off-topic distractor generation."""
    text_lower = text.lower()
    return [
        topic for topic, pattern in TOPIC_KEYWORDS.items()
        if re.search(pattern, text_lower, re.IGNORECASE)
    ]


def extract_solution_numbers(solution: str) -> set[float]:
    """
    Extract all numbers that appear in the solution steps (before ####).
    These are the numbers actually USED to reach the answer.
    """
    sol_text = solution.split("####")[0]
    used = set()
    for m in NUMBER_PATTERN.finditer(sol_text):
        try:
            used.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return used


def find_noop_candidates(question: str, solution: str) -> list[dict]:
    """
    Find question sentences that contain numbers but whose numbers
    do NOT appear in the solution steps — strong candidates for no-op distractors.
    These sentences look mathematically relevant but are not used.
    """
    used_numbers = extract_solution_numbers(solution)
    candidates = []
    for sent in split_sentences(question):
        sent_numbers = {
            float(m.group(0).replace(",", ""))
            for m in NUMBER_PATTERN.finditer(sent)
            if m.group(0).replace(",", "").replace(".", "").isdigit()
        }
        if sent_numbers and sent_numbers.isdisjoint(used_numbers):
            candidates.append({
                "text": sent,
                "numbers_in_sentence": list(sent_numbers),
                "reason": "numbers not used in solution"
            })
    return candidates


def extract_persons(text: str) -> list[str]:
    """
    Extract person names from question text using capitalization rules.
    More reliable than spaCy for GSM8K — spaCy often misclassifies names
    like 'Natalia' as GPE or 'Betty' as ORG in math problem context.

    Logic: collect all capitalized words not in a known non-name list.
    This includes sentence-start words since names like 'Weng' or 'James'
    may only appear once at the start of a sentence.
    """
    NON_NAMES = {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "I", "A", "An", "The", "He", "She", "They", "We", "It", "His", "Her",
        "Yesterday", "Today", "Tomorrow", "Now", "Then", "How", "What",
        "Each", "Every", "All", "Some", "This", "That", "There", "If",
        "After", "Before", "When", "While", "Since", "By", "At", "In",
        "On", "For", "To", "Of", "With", "From", "About",
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
        "And", "But", "So", "Next", "Finally", "Also", "Over", "Under",
        "Last", "First", "Second", "Third", "Fourth", "Fifth",
        "New", "Old", "Big", "Small", "Large", "Long", "Short", "High", "Low",
        "Mr", "Mrs", "Ms", "Dr", "Sir",
        "DVD", "TV", "PC", "PE", "ATM",
        "No", "Yes", "Not", "Just", "Only", "Even", "Still", "Already",
        "During", "Between", "Around", "Across", "Along", "Upon",
        "Ships", "Grade", "However", "Although", "Because", "Therefore",
        "Another", "Other", "Same", "Different", "Total", "Average",
        "Striploin", "Ribeye", "Sirloin",
    }
    sentences = split_sentences(text)
    persons = set()
    for sent in sentences:
        words = sent.split()
        for word in words:
            clean = re.sub(r"[^a-zA-Z]", "", word)
            if not clean or not clean[0].isupper():
                continue
            if len(clean) <= 2:
                continue  # skip single letters and two-letter words
            if clean.isupper():
                continue  # skip all-caps abbreviations like FICA, NY, LLC
            if clean not in NON_NAMES:
                persons.add(clean)
    return list(persons)


def build_distractor_hints(question: str, solution: str, entities: list[dict],
                           numbers: list[dict], units: list[dict]) -> dict:
    """
    Package everything #10 needs to generate all 3 distractor types.

    off_topic  → needs: topics + person names → generate unrelated sentence
    in_topic   → needs: numbers + units in the problem → swap values, keep theme
    no_op      → needs: candidate sentences with unused numbers → insert as-is or paraphrase
    """
    persons = extract_persons(question)
    topics  = detect_topics(question)

    return {
        "off_topic": {
            "topics":   topics,       # e.g. ["money", "school"], avoid these in distractor
            "persons":  persons,      # e.g. ["Natalia"], can reuse names for naturalness
        },
        "in_topic": {
            "numbers":  [n["value"] for n in numbers],   # numbers to swap/perturb
            "units":    [u["unit"]  for u in units],     # units to keep consistent
            "persons":  persons,
        },
        "no_op": {
            "candidates": find_noop_candidates(question, solution),
            # sentences with numbers irrelevant to the solution
            # can insert these as distractors directly
        },
    }


# main extraction loop
def extract_gsm8k(split: str = "train", output_path: str = "gsm8k_extracted.json") -> list[dict]:
    print(f"Loading GSM8K ({split} split)...")
    ds = load_dataset("gsm8k", "main", split=split)
    print(f"  {len(ds)} examples found.")

    records = []
    for i, example in enumerate(ds):
        question: str = example["question"]
        solution: str = example["answer"]

        numbers  = extract_numbers(question)
        units    = extract_units(question)
        entities = extract_entities(question)

        record = {
            "id": i,
            "question": question,
            "solution": solution,
            "gold_answer": parse_gold_answer(solution),
            "sentences": label_sentences(question, solution),
            "numbers": numbers,
            "units": units,
            "entities": entities,
            "distractor_hints": build_distractor_hints(
                question, solution, entities, numbers, units
            ),
        }
        records.append(record)

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(ds)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(records)} records → {output_path}")
    return records


# quick sanity check
def print_sample(records: list[dict], n: int = 2) -> None:
    for rec in records[:n]:
        print("=" * 60)
        print(f"ID           : {rec['id']}")
        print(f"Question     : {rec['question'][:120]}...")
        print(f"Gold answer  : {rec['gold_answer']}")
        print()
        print(f"Numbers ({len(rec['numbers'])}):")
        for num in rec['numbers'][:5]:
            print(f"  {num['raw']} = {num['value']} (pos {num['start']}-{num['end']})")
        print(f"Units ({len(rec['units'])}):")
        for u in rec['units'][:3]:
            print(f"  {u['number']} {u['unit']}")
        print(f"Entities ({len(rec['entities'])}):")
        for e in rec['entities'][:5]:
            print(f"  {e['text']} [{e['label']}]")
        print()
        print(f"Sentences ({len(rec['sentences'])}):")
        for s in rec['sentences']:
            print(f"  [{s['role']:8}] {s['text'][:80]}")
        print()
        hints = rec["distractor_hints"]
        print(f"Distractor hints:")
        print(f"  off_topic  → topics={hints['off_topic']['topics']}")
        print(f"             → persons={hints['off_topic']['persons']}")
        print(f"  in_topic   → numbers={hints['in_topic']['numbers'][:5]}")
        print(f"             → units={hints['in_topic']['units'][:3]}")
        print(f"             → persons={hints['in_topic']['persons']}")
        noop = hints['no_op']['candidates']
        print(f"  no_op      → {len(noop)} candidate sentence(s)")
        for c in noop[:2]:
            print(f"    '{c['text'][:70]}' | nums={c['numbers_in_sentence']}")


def validate(records: list[dict]) -> None:
    """
    Validate all records and report any data quality issues.
    Checks: gold_answer, numbers, units, entities, sentences, distractor_hints.
    """
    issues = []

    for rec in records:
        id = rec["id"]

        # gold_answer must not be None
        if rec["gold_answer"] is None:
            issues.append(f"ID {id}: gold_answer is None")

        # numbers must have valid value and position
        for n in rec["numbers"]:
            if n["value"] is None:
                issues.append(f"ID {id}: number has None value: {n}")
            if n["start"] >= n["end"]:
                issues.append(f"ID {id}: number has invalid position: {n}")

        # units must have non-empty number and unit
        for u in rec["units"]:
            if not u["number"] or not u["unit"]:
                issues.append(f"ID {id}: unit has empty field: {u}")

        # entities must have text and label
        for e in rec["entities"]:
            if not e["text"] or not e["label"]:
                issues.append(f"ID {id}: entity has empty field: {e}")

        # sentences must have text and valid role
        for s in rec["sentences"]:
            if not s["text"]:
                issues.append(f"ID {id}: sentence has empty text")
            if s["role"] not in ("question", "solution"):
                issues.append(f"ID {id}: sentence has invalid role: {s['role']}")

        # distractor_hints must have all three blocks
        hints = rec["distractor_hints"]
        for block in ("off_topic", "in_topic", "no_op"):
            if block not in hints:
                issues.append(f"ID {id}: distractor_hints missing block: {block}")

        # no_op candidates must have text and numbers
        for c in hints.get("no_op", {}).get("candidates", []):
            if not c["text"]:
                issues.append(f"ID {id}: no_op candidate has empty text")
            if not c["numbers_in_sentence"]:
                issues.append(f"ID {id}: no_op candidate has no numbers")

    print(f"Validated {len(records)} records.")
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues[:20]:
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more.")
    else:
        print("All checks passed!")


if __name__ == "__main__":
    print("\n=== Extracting TRAIN split ===")
    train_records = extract_gsm8k(
        split="train",
        output_path="gsm8k_train_extracted.json",
    )
    print_sample(train_records)
    validate(train_records)

    print("\n=== Extracting TEST split ===")
    test_records = extract_gsm8k(
        split="test",
        output_path="gsm8k_test_extracted.json",
    )
    print_sample(test_records)
    validate(test_records)