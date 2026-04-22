"""
Breaks down math problems in the GSM8K into machine-readable components
through an extraction pipeline using Regex and spaCy.

"""
import re
import json
from datasets import load_dataset

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except Exception:
    USE_SPACY = False


NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)
    (\d{1,3}(?:,\d{3})*|\d+)
    (?:\.\d+)?
    (?!\w)
    """,
    re.VERBOSE,
)

UNIT_PATTERN = re.compile(
    r"""
    (\d[\d,.]*)\s*
    (dollars?|cents?|pounds?|kg|g|km|miles?|
     hours?|minutes?|seconds?|days?|weeks?|
     apples?|oranges?|boxes?|bags?|
     \$|€|£|%)
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# pull data from text:
def extract_numbers(text: str):
    results = []
    for m in NUMBER_PATTERN.finditer(text):
        raw = m.group(0)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        results.append({"raw": raw, "value": value, "start": m.start(), "end": m.end()})
    return results


def extract_units(text: str):
    return [{"number": m.group(1), "unit": m.group(2)} for m in UNIT_PATTERN.finditer(text)]


def extract_entities(text: str):
    if not USE_SPACY:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


# split & catrgorize sentences:
def split_sentences(text: str):
    if USE_SPACY:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def label_sentences(question: str, solution: str):
    labeled = []
    for sent in split_sentences(question):
        labeled.append({"text": sent, "role": "question"})
    sol_text = solution.split("####")[0].strip()
    for sent in split_sentences(sol_text):
        labeled.append({"text": sent, "role": "solution"})
    return labeled


def parse_gold_answer(solution: str):
    m = ANSWER_PATTERN.search(solution)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None

# topic classification:
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

def detect_topics(text: str):
    text_lower = text.lower()
    return [
        topic for topic, pattern in TOPIC_KEYWORDS.items()
        if re.search(pattern, text_lower, re.IGNORECASE)
    ]


# finds all numbers used within the solution
def extract_solution_numbers(solution: str):
    sol_text = solution.split("####")[0]
    used = set()
    for m in NUMBER_PATTERN.finditer(sol_text):
        try:
            used.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return used

# finds questions that contains numbers unused in the solution
def find_noop_candidates(question: str, solution: str):
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


def extract_persons(text: str):
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
                continue
            if clean.isupper():
                continue
            if clean not in NON_NAMES:
                persons.add(clean)
    return list(persons)


def build_distractor_hints(question: str, solution: str, entities: list[dict],
                           numbers: list[dict], units: list[dict]):
    persons = extract_persons(question)
    topics  = detect_topics(question)

    return {
        "off_topic": {
            "topics":   topics,
            "persons":  persons,
        },
        "in_topic": {
            "numbers":  [n["value"] for n in numbers],
            "units":    [u["unit"]  for u in units],
            "persons":  persons,
        },
        "no_op": {
            "candidates": find_noop_candidates(question, solution),
        },
    }


# main pipeline to load, process & validate
def extract_gsm8k(split: str = "train", output_path: str = "gsm8k_extracted.json"):
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

def validate(records: list[dict]):
    issues = []

    for rec in records:
        id = rec["id"]

        if rec["gold_answer"] is None:
            issues.append(f"ID {id}: gold_answer is None")

        for n in rec["numbers"]:
            if n["value"] is None:
                issues.append(f"ID {id}: number has None value: {n}")
            if n["start"] >= n["end"]:
                issues.append(f"ID {id}: number has invalid position: {n}")

        for u in rec["units"]:
            if not u["number"] or not u["unit"]:
                issues.append(f"ID {id}: unit has empty field: {u}")

        for e in rec["entities"]:
            if not e["text"] or not e["label"]:
                issues.append(f"ID {id}: entity has empty field: {e}")

        for s in rec["sentences"]:
            if not s["text"]:
                issues.append(f"ID {id}: sentence has empty text")
            if s["role"] not in ("question", "solution"):
                issues.append(f"ID {id}: sentence has invalid role: {s['role']}")

        hints = rec["distractor_hints"]
        for block in ("off_topic", "in_topic", "no_op"):
            if block not in hints:
                issues.append(f"ID {id}: distractor_hints missing block: {block}")

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
    train_records = extract_gsm8k(
        split="train",
        output_path="gsm8k_train_extracted.json",
    )
    validate(train_records)

    test_records = extract_gsm8k(
        split="test",
        output_path="gsm8k_test_extracted.json",
    )
    validate(test_records)