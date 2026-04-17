"""
Task #10 - NLP for Distractor Generation

Reads gsm8k_extracted.json (output of Task #9) and generates three types
of distractor sentences for each problem:
  - off_topic:  a sentence unrelated to the problem topic
  - in_topic:   a sentence related to the topic but with a fake number
  - no_op:      a sentence that looks computationally relevant but isn't,
                using time/context markers to signal the number is not needed

Output: gsm8k_distractors.json
Each record contains the original question and three distractor sentences.
Does NOT insert distractors into questions — that is done by Task #4 (distractor_insertion.py).
"""

import json
import random

random.seed(42)

# ── off_topic templates ───────────────────────────────────────────────────────

OFF_TOPIC_POOLS = {
    "money": [
        "The weather was sunny and warm that day.",
        "It was a quiet morning outside.",
        "The sky was clear and blue.",
        "Birds were singing in the trees.",
    ],
    "time": [
        "She enjoyed painting on weekends.",
        "He had a pet dog named Max.",
        "The garden was full of flowers.",
        "They went for a walk in the park.",
    ],
    "food": [
        "The bus arrived late that morning.",
        "It was a busy day at the office.",
        "She liked reading books in the evening.",
        "He played soccer with his friends.",
        "The post office was closed on Sunday.",
        "She went for a morning jog.",
    ],
    "school": [
        "The market was crowded on weekends.",
        "She bought a new pair of shoes.",
        "He enjoyed cycling in the park.",
        "The river flowed gently through the valley.",
        "She painted a picture in her free time.",
        "He went for a jog in the morning.",
    ],
    "shopping": [
        "The children played outside all day.",
        "It was a quiet evening at home.",
        "She watered the plants every morning.",
        "He fixed his bicycle in the garage.",
    ],
    "work": [
        "The lake was calm and peaceful.",
        "She visited her grandmother on Sunday.",
        "He cooked dinner for the family.",
        "The flowers bloomed in the spring.",
        "She took a long walk in the afternoon.",
        "He enjoyed reading mystery novels.",
        "They played chess in the evening.",
        "She knitted a scarf for her friend.",
    ],
    "distance": [
        "She baked cookies for the neighbors.",
        "He watched a movie with his family.",
        "The cat slept on the windowsill.",
        "They played board games after dinner.",
    ],
    "nature": [
        "She finished her homework before dinner.",
        "He practiced piano every afternoon.",
        "The library was open until nine.",
        "They decorated the room with balloons.",
        "The mountain trail was peaceful and quiet.",
        "He enjoyed swimming at the local pool.",
        "She learned to play the guitar.",
        "He read a book in the afternoon.",
        "She painted a picture on the weekend.",
        "They watched a movie together.",
    ],
    "default": [
        "The weather was pleasant that afternoon.",
        "She enjoyed her morning coffee.",
        "He took a short nap after lunch.",
        "The neighborhood was quiet and peaceful.",
    ],
}

# ── in_topic templates ────────────────────────────────────────────────────────

PERSON_UNIT_TEMPLATES = [
    "{person} also had {number} {unit}.",
    "{person} bought {number} more {unit}.",
    "{person} gave away {number} {unit} yesterday.",
    "{person} found {number} extra {unit} at home.",
]
UNIT_ONLY_TEMPLATES = [
    "There were {number} extra {unit} available.",
    "Another {number} {unit} were added.",
    "{number} more {unit} were left over.",
    "A total of {number} {unit} were not counted.",
]
PERSON_ONLY_TEMPLATES = [
    "{person} also had {number} items.",
    "{person} counted {number} more things.",
    "{person} found {number} extra objects.",
]
GENERIC_TEMPLATES = [
    "There were {number} more items available.",
    "Another {number} things were added.",
    "{number} extra objects were left over.",
]

# ── no_op templates ───────────────────────────────────────────────────────────

NO_OP_PERSON_UNIT_TEMPLATES = [
    "{person} had bought {number} {unit} the previous month.",
    "Originally, {person} had {number} {unit} before this.",
    "Last year, {person} had {number} {unit} in total.",
    "{person} started with {number} {unit} at the beginning.",
    "Before this happened, {person} already had {number} {unit}.",
    "If the offer applied, {person} would have received {number} {unit}.",
    "Had the deal gone through, {person} would have had {number} {unit}.",
    "{person} had a coupon for {number} {unit} but it had already expired.",
    "A discount of {number} {unit} was offered to {person} but not applied.",
    "{person} had ordered {number} extra {unit} but the order was cancelled.",
]
NO_OP_UNIT_ONLY_TEMPLATES = [
    "There were originally {number} {unit} before this.",
    "Last week, there were {number} {unit} in total.",
    "Previously, {number} {unit} had already been counted.",
    "At the start, there were {number} {unit} available.",
    "If the promotion had applied, there would have been {number} {unit}.",
    "Had the shipment arrived, there would be {number} more {unit}.",
    "A bulk order of {number} {unit} was placed but later cancelled.",
    "An offer of {number} {unit} was available but had already expired.",
]
NO_OP_PERSON_ONLY_TEMPLATES = [
    "{person} had previously counted {number} items.",
    "Originally, {person} had {number} things in total.",
    "Before this, {person} already owned {number} objects.",
    "Last month, {person} had {number} items.",
    "If the plan had worked, {person} would have had {number} more.",
    "Had things gone differently, {person} would have counted {number} items.",
    "{person} had a voucher worth {number} but it was not valid.",
    "An additional {number} items were reserved for {person} but not delivered.",
]
NO_OP_GENERIC_TEMPLATES = [
    "There were originally {number} items before this.",
    "Previously, {number} things had already been counted.",
    "At the beginning, there were {number} objects in total.",
    "Last week, {number} items were already accounted for.",
    "If the conditions had been met, there would have been {number} more.",
    "Had the discount applied, the total would have been {number}.",
    "A promotional offer of {number} was available but had since expired.",
    "An order for {number} extra items was placed but later cancelled.",
]

# Generic subjects by topic
TOPIC_SUBJECTS = {
    "school":   ["A student", "A teacher", "A classmate"],
    "shopping": ["A customer", "A shopper", "A buyer"],
    "work":     ["A worker", "An employee", "A colleague"],
    "food":     ["A chef", "A customer", "A visitor"],
    "money":    ["A customer", "A client", "A vendor"],
    "time":     ["Someone", "A person", "A participant"],
    "distance": ["A traveler", "A runner", "A driver"],
    "nature":   ["A farmer", "A visitor", "A gardener"],
    "default":  ["Someone", "A person", "A participant"],
}


# ── helper functions ──────────────────────────────────────────────────────────

def generate_fake_number(original: float) -> float:
    """
    Generate a fake number near the original but different.
    Preserves the same format (integer vs decimal, same decimal places).
    Uses a random delta of 30-70% of the original to avoid fixed ratios.
    Retries up to 5 times to ensure the result differs from the original.
    """
    if original == int(original):
        for _ in range(5):
            delta = random.uniform(0.3, 0.7) * original
            fake = original + delta if random.random() < 0.5 else original - delta
            result = max(1, int(round(fake)))
            if result != int(original):
                return result
        # fallback: add or subtract a random amount between 1 and max(2, original//2)
        offset = random.randint(1, max(2, int(original // 2)))
        return max(1, int(original) + random.choice([-1, 1]) * offset)
    else:
        decimal_places = len(str(original).rstrip("0").split(".")[-1])
        min_val = 10 ** (-decimal_places)
        for _ in range(5):
            delta = random.uniform(0.3, 0.7) * original
            fake = original + delta if random.random() < 0.5 else original - delta
            result = round(max(min_val, fake), decimal_places)
            if result != original:
                return result
        return round(original + min_val, decimal_places)


def get_person(persons, topics):
    if persons:
        return random.choice(persons)
    key = topics[0] if topics else "default"
    return random.choice(TOPIC_SUBJECTS.get(key, TOPIC_SUBJECTS["default"]))


def extract_solution_numbers(solution: str) -> list[float]:
    """Extract numbers from solution steps (before ####) — these are the
    numbers actually used in the calculation, preferred for distractor generation."""
    import re
    sol_text = solution.split("####")[0]
    nums = set()
    for m in re.findall(r"(?<!\w)(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?!\w)", sol_text):
        try:
            nums.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return list(nums)


def extract_question_only_numbers(question: str, solution: str) -> list[float]:
    """
    Find numbers that appear in the question but NOT in the solution steps.
    These are numbers the model might think are relevant but are never used.
    Ideal for no_op distractors — using these makes the distractor more deceptive.
    Falls back to all question numbers if none are found.
    """
    import re
    def get_nums(text):
        nums = set()
        for m in re.findall(r"(?<!\w)(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?!\w)", text):
            try:
                nums.add(float(m.replace(",", "")))
            except ValueError:
                pass
        return nums

    q_nums  = get_nums(question)
    sol_nums = get_nums(solution.split("####")[0])
    unused = list(q_nums - sol_nums)
    return unused if unused else list(q_nums)


def generate_off_topic(topics: list[str]) -> str:
    avoid = set(topics)
    available = [t for t in OFF_TOPIC_POOLS if t not in avoid and t != "default"]
    chosen = random.choice(available) if available else "default"
    return random.choice(OFF_TOPIC_POOLS[chosen])


def generate_in_topic(hints: dict, topics: list[str], gold_answer: float = None) -> str | None:
    numbers = hints["numbers"]
    sol_numbers = hints.get("solution_numbers", [])
    if not numbers:
        return None
    # Prefer solution numbers — more likely to mislead the model
    candidates = sol_numbers if sol_numbers else numbers
    chosen = random.choice(candidates)
    fake = generate_fake_number(chosen)
    # Retry until fake differs from chosen number and gold_answer
    forbidden = {chosen, gold_answer} if gold_answer else {chosen}
    for _ in range(10):
        if fake not in forbidden:
            break
        fake = generate_fake_number(chosen)
    # Fallback: if still in forbidden, jump by a random offset of 3-10
    if fake in forbidden:
        fake = float(int(fake) + random.randint(3, 10))
        while fake in forbidden:
            fake += 1
    person = get_person(hints["persons"], topics)
    unit   = random.choice(hints["units"]) if hints["units"] else None
    if person and unit:
        return random.choice(PERSON_UNIT_TEMPLATES).format(person=person, number=fake, unit=unit)
    elif unit:
        return random.choice(UNIT_ONLY_TEMPLATES).format(number=fake, unit=unit)
    elif person:
        return random.choice(PERSON_ONLY_TEMPLATES).format(person=person, number=fake)
    else:
        return random.choice(GENERIC_TEMPLATES).format(number=fake)


def generate_no_op(hints: dict, topics: list[str], gold_answer: float = None,
                   question: str = None, solution: str = None) -> str | None:
    numbers = hints["numbers"]
    if not numbers:
        return None
    # Prefer numbers that appear in question but NOT in solution steps
    # — these look relevant but were never used, making the distractor more deceptive
    if question and solution:
        candidates = extract_question_only_numbers(question, solution)
    else:
        candidates = numbers
    chosen = random.choice(candidates) if candidates else random.choice(numbers)
    fake = generate_fake_number(chosen)
    # Retry until fake differs from chosen number and gold_answer
    forbidden = {chosen, gold_answer} if gold_answer else {chosen}
    for _ in range(10):
        if fake not in forbidden:
            break
        fake = generate_fake_number(chosen)
    # Fallback: if still in forbidden, jump by a random offset of 3-10
    if fake in forbidden:
        fake = float(int(fake) + random.randint(3, 10))
        while fake in forbidden:
            fake += 1
    person = get_person(hints["persons"], topics)
    unit   = random.choice(hints["units"]) if hints["units"] else None
    if person and unit:
        return random.choice(NO_OP_PERSON_UNIT_TEMPLATES).format(person=person, number=fake, unit=unit)
    elif unit:
        return random.choice(NO_OP_UNIT_ONLY_TEMPLATES).format(number=fake, unit=unit)
    elif person:
        return random.choice(NO_OP_PERSON_ONLY_TEMPLATES).format(person=person, number=fake)
    else:
        return random.choice(NO_OP_GENERIC_TEMPLATES).format(number=fake)


# ── main generation loop ──────────────────────────────────────────────────────

def generate_distractors(
    input_path:  str = "gsm8k_extracted.json",
    output_path: str = "gsm8k_distractors.json",
) -> list[dict]:

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    print(f"  {len(data)} records loaded.")

    records = []
    for i, rec in enumerate(data):
        hints  = rec["distractor_hints"]
        topics = hints["off_topic"]["topics"]

        # Inject solution numbers into in_topic hints for better distractor selection
        hints["in_topic"]["solution_numbers"] = extract_solution_numbers(rec["solution"])

        record = {
            "id":          rec["id"],
            "question":    rec["question"],
            "solution":    rec["solution"],
            "gold_answer": rec["gold_answer"],
            "distractors": {
                "off_topic": generate_off_topic(topics),
                "in_topic":  generate_in_topic(hints["in_topic"], topics, rec["gold_answer"]),
                "no_op":     generate_no_op(hints["in_topic"], topics, rec["gold_answer"],
                                           rec["question"], rec["solution"]),
            }
        }
        records.append(record)

        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(data)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(records)} records -> {output_path}")
    print(f"  off_topic: {sum(1 for r in records if r['distractors']['off_topic'])}")
    print(f"  in_topic:  {sum(1 for r in records if r['distractors']['in_topic'])}")
    print(f"  no_op:     {sum(1 for r in records if r['distractors']['no_op'])}")
    return records


# ── verify ────────────────────────────────────────────────────────────────────

def verify(records: list[dict], extracted: list[dict]) -> None:
    import re
    TOPIC_KEYWORDS = {
        "money":    ["dollar", "cent", "price", "cost", "pay", "spend", "profit", "wage"],
        "time":     ["hour", "minute", "second", "month", "year"],
        "food":     ["apple", "orange", "banana", "cake", "pizza", "meal", "food", "drink"],
        "distance": ["mile", "km", "kilometer", "meter", "drive", "traveled", "distance"],
        "school":   ["student", "teacher", "class", "school", "grade", "exam", "lesson"],
        "work":     ["worker", "employee", "job", "salary", "shift", "factory", "hired", "workplace"],
        "shopping": ["buy", "sell", "store", "shop", "item", "product", "order"],
        "nature":   ["farm", "animal", "fish", "forest", "river", "crop", "wildlife"],
    }
    NO_OP_MARKERS = [
        "previously", "originally", "last year", "last week", "last month",
        "previous month", "previous", "before this", "at the start",
        "at the beginning", "already",
        "if the", "had the", "would have",
        "expired", "cancelled", "not applied", "not valid", "not delivered",
    ]
    extracted_map = {r["id"]: r for r in extracted}
    issues = []

    for rec in records:
        id     = rec["id"]
        dist   = rec["distractors"]
        orig   = rec["question"]
        topics = extracted_map[id]["distractor_hints"]["off_topic"]["topics"]
        orig_nums = set(float(m.replace(",","")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", orig))

        if not dist.get("off_topic"):
            issues.append(f"ID {id}: off_topic is empty")
        else:
            d = dist["off_topic"]
            d_nums = set(float(m.replace(",","")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", d))
            if d_nums & orig_nums:
                issues.append(f"ID {id}: off_topic contains original number")
            for topic in topics:
                for kw in TOPIC_KEYWORDS.get(topic, []):
                    if kw.lower() in d.lower():
                        issues.append(f"ID {id}: off_topic contains topic keyword '{kw}'")
                        break

        # in_topic: number must not equal gold_answer
        if dist.get("in_topic"):
            d = dist["in_topic"]
            d_nums = set(float(m.replace(",","")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", d))
            gold = rec.get("gold_answer")
            if gold and gold in d_nums:
                issues.append(f"ID {id}: in_topic contains gold_answer {gold}")

        # no_op: must have marker, number must not equal gold_answer
        if dist.get("no_op"):
            d = dist["no_op"]
            if not any(m in d.lower() for m in NO_OP_MARKERS):
                issues.append(f"ID {id}: no_op missing marker: {d}")
            d_nums = set(float(m.replace(",","")) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", d))
            gold = rec.get("gold_answer")
            if gold and gold in d_nums:
                issues.append(f"ID {id}: no_op contains gold_answer {gold}")

    print(f"Verified {len(records)} records.")
    print(f"  off_topic: {sum(1 for r in records if r['distractors']['off_topic'])}")
    print(f"  in_topic:  {sum(1 for r in records if r['distractors']['in_topic'])}")
    print(f"  no_op:     {sum(1 for r in records if r['distractors']['no_op'])}")
    print()
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues[:20]:
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more.")
    else:
        print("All checks passed!")


if __name__ == "__main__":
    records = generate_distractors(
        input_path="gsm8k_extracted.json",
        output_path="gsm8k_distractors.json",
    )
    with open("gsm8k_extracted.json") as f:
        extracted = json.load(f)
    verify(records, extracted)