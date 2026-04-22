"""
Reads gsm8k_extracted.json and generates three types
of distractor sentences for each problem:
  - off_topic:  a sentence unrelated to the problem topic
  - in_topic:   a sentence related to the topic but with a fake number
  - no_op:      a sentence that looks computationally relevant but isn't
"""
from __future__ import annotations

import json
import random
import re

SEED = 42
random.seed(SEED)


OFF_TOPIC_BY_DOMAIN = {
    "weather": [
        "The sky was overcast with a light breeze that afternoon.",
        "A gentle rain fell throughout the morning hours.",
        "The weather turned unexpectedly warm over the weekend.",
        "Fog rolled in from the coast by early evening.",
        "Sunshine broke through the clouds just after lunch.",
    ],
    "hobby": [
        "She had recently taken up watercolor painting as a weekend hobby.",
        "He enjoyed learning new chess openings in his free time.",
        "Their book club met every other Thursday at the local cafe.",
        "She had been collecting vintage postcards since childhood.",
        "He practiced the guitar for an hour each evening.",
    ],
    "travel": [
        "The trip to the coast last summer had been especially memorable.",
        "They were planning a weekend hike in the nearby mountains.",
        "A cousin was visiting from out of state for the holidays.",
        "The train station had been renovated the previous spring.",
        "Their vacation photos were still sitting on the kitchen table.",
    ],
    "household": [
        "The cat had fallen asleep on the sunlit windowsill.",
        "A new rug had arrived for the living room last week.",
        "The old clock in the hallway needed a fresh battery.",
        "The garden hose had sprung a small leak near the spigot.",
        "A stack of library books sat waiting by the front door.",
    ],
    "community": [
        "The local park was hosting a music festival on Saturday.",
        "A new bakery had opened on the corner of Main Street.",
        "The neighborhood watch meeting was rescheduled to Tuesday.",
        "The community garden was looking for new volunteers.",
        "A charity run was scheduled for the following weekend.",
    ],
    "event": [
        "Her birthday party had gone late into the evening.",
        "The family reunion was happening next month.",
        "A wedding announcement arrived in the mail yesterday.",
        "His retirement celebration was scheduled for Friday.",
        "The graduation ceremony lasted longer than expected.",
    ],
}

TOPIC_AVOID_DOMAINS = {
    "money":    {"community", "event"},
    "shopping": {"community", "household"},
    "work":     {"community", "event"},
    "time":     {"event"},
    "school":   {"event", "community"},
    "distance": {"travel"},
    "nature":   {"household", "weather"},
    "food":     {"household", "event"},
}


UNIT_CATEGORY = {
    # money
    "dollar": "money", "dollars": "money",
    "cent": "money", "cents": "money",
    "$": "money", "€": "money", "£": "money",
    # time
    "second": "time", "seconds": "time",
    "minute": "time", "minutes": "time",
    "hour": "time", "hours": "time",
    "day": "time", "days": "time",
    "week": "time", "weeks": "time",
    "month": "time", "months": "time",
    "year": "time", "years": "time",
    # distance
    "mile": "distance", "miles": "distance",
    "km": "distance", "kilometer": "distance", "kilometers": "distance",
    "m": "distance", "meter": "distance", "meters": "distance",
    "cm": "distance",
    # weight
    "pound": "weight", "pounds": "weight", "lb": "weight", "lbs": "weight",
    "kg": "weight", "g": "weight", "mg": "weight",
    "oz": "weight",
    # volume (treated like weight for template purposes)
    "l": "weight", "ml": "weight",
    # percent
    "%": "percent",
}


def classify_unit(unit: str):
    if not unit:
        return None
    u = unit.strip().lower()
    if u in UNIT_CATEGORY:
        return UNIT_CATEGORY[u]
    if u.isalpha() and len(u) >= 3:
        return "count"
    return None


def allow_weight_templates(raw_unit, topics: list[str], question: str):
    if not raw_unit:
        return False
    u = raw_unit.strip().lower()
    q = question.lower()
    strong_weight_words = {"weigh", "weight", "pound", "pounds", "kg", "gram", "grams", "ounce", "ounces", "liter", "liters", "ml"}
    if any(w in q for w in strong_weight_words):
        return True
    if any(t in {"food", "shopping", "nature"} for t in topics):
        return True
    if u in {"g", "mg", "ml"}:
        return False
    return True


WEIGHT_SUBSTANCES = {
    "kg":      ["flour", "rice", "feed", "produce", "grain"],
    "g":       ["coffee", "tea", "herbs", "cocoa"],
    "mg":      ["medicine"],
    "l":       ["water", "juice", "milk", "oil"],
    "ml":      ["oil", "vinegar", "syrup"],
    "oz":      ["cheese", "coffee", "tea"],
    "lb":      ["flour", "potatoes", "produce"],
    "lbs":     ["flour", "potatoes", "produce"],
    "pound":   ["flour", "potatoes", "produce"],
    "pounds":  ["flour", "potatoes", "produce"],
}


def pluralize(word: str, n):
    try:
        n_val = float(n)
    except (TypeError, ValueError):
        n_val = 2

    if word.endswith("s") and not word.endswith("ss"):
        if n_val == 1:
            if word.endswith("ies"):
                return word[:-3] + "y"
            if word.endswith("es") and len(word) > 3:
                return word[:-2]
            return word[:-1]
        return word
    if n_val == 1:
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def build_phrase(unit: str, number, category: str):
    u = unit.strip().lower()

    if category == "money":
        if u in {"$", "€", "£"}:
            return f"{u}{_fmt(number)}"
        return f"{_fmt(number)} {pluralize(u, number)}"

    if category == "time":
        return f"{_fmt(number)} {pluralize(u, number)}"

    if category == "distance":
        if u in {"km", "cm", "m"}:
            return f"{_fmt(number)} {u}"
        return f"{_fmt(number)} {pluralize(u, number)}"

    if category == "weight":
        substances = WEIGHT_SUBSTANCES.get(u)
        if substances:
            subst = random.choice(substances)
            return f"{_fmt(number)} {u} of {subst}"
        return None

    if category == "count":
        return f"{_fmt(number)} {pluralize(u, number)}"

    return None


IN_TOPIC_MONEY_PERSON = [
    "{person} had paid {phrase} for a similar purchase last week.",
    "{person} had set aside {phrase} for a different expense.",
    "On a separate occasion, {person} had budgeted {phrase}.",
    "A neighbor of {person} had recently spent {phrase}.",
    "At another shop, {person} saw the same item priced at {phrase}.",
]
IN_TOPIC_MONEY_NO_PERSON = [
    "At another store, the same kind of item was priced at {phrase}.",
    "A similar purchase elsewhere had cost {phrase}.",
    "A different vendor had set the price at {phrase}.",
    "The market average that week was around {phrase}.",
]

IN_TOPIC_TIME_PERSON = [
    "{person} had spent {phrase} on a different task earlier that week.",
    "On another occasion, {person} had taken {phrase} to finish a similar job.",
    "{person} had scheduled {phrase} for a separate activity that day.",
    "An earlier plan had allowed {person} {phrase} for another task.",
]
IN_TOPIC_TIME_NO_PERSON = [
    "A similar task elsewhere had taken {phrase}.",
    "Another group had spent {phrase} on a comparable project.",
    "In a different setting, the same work required {phrase}.",
    "An earlier estimate had put the duration at {phrase}.",
]

IN_TOPIC_DISTANCE_PERSON = [
    "{person} had walked {phrase} on a different outing last week.",
    "On a separate trip, {person} had driven {phrase}.",
    "{person} had traveled {phrase} for an unrelated errand that day.",
    "An earlier route connected with {person} covered {phrase}.",
]
IN_TOPIC_DISTANCE_NO_PERSON = [
    "A different route in the area covered {phrase}.",
    "Another traveler had gone {phrase} that same day.",
    "A nearby trail was known to be {phrase} long.",
    "The alternate path was roughly {phrase}.",
]

IN_TOPIC_WEIGHT_PERSON = [
    "{person} had bought {phrase} at a different store that week.",
    "On another trip, {person} had weighed out {phrase} for a separate order.",
    "{person} had ordered {phrase} for a different delivery.",
    "A previous purchase linked to {person} included {phrase}.",
]
IN_TOPIC_WEIGHT_NO_PERSON = [
    "Another batch contained {phrase} of similar quality.",
    "A different supplier delivered {phrase} that same week.",
    "The nearby market had {phrase} available at the time.",
    "A parallel order included {phrase} as well.",
]

IN_TOPIC_COUNT_PERSON = [
    "{person} had seen {phrase} at another store that day.",
    "{person} had considered buying {phrase} last week.",
    "At one point, {person} had counted {phrase} in a different collection.",
    "A separate batch connected to {person} included {phrase}.",
]
IN_TOPIC_COUNT_NO_PERSON = [
    "At a different store, there were {phrase} on display.",
    "Another supplier had {phrase} in stock.",
    "A nearby shop had {phrase} that week.",
    "A separate batch contained {phrase} in total.",
]

IN_TOPIC_PERCENT = [
    "An earlier survey showed {number}% of respondents agreed.",
    "A recent study found {number}% of participants benefited.",
    "Approximately {number}% of the samples were set aside.",
    "A report noted that {number}% of cases were reviewed separately.",
]


NO_OP_MONEY_PERSON = [
    "A coupon worth {phrase} had been given to {person}, but it had expired.",
    "{person} had been offered a discount of {phrase}, which went unused.",
    "A refund of {phrase} had been promised to {person}, but never processed.",
    "If the earlier offer had applied, {person} would have saved {phrase}.",
    "{person} had budgeted {phrase} previously, though that plan was cancelled.",
]
NO_OP_MONEY_NO_PERSON = [
    "A previous promotion offered {phrase} off, but it had since expired.",
    "An earlier refund of {phrase} had been promised but never paid out.",
    "Had the old pricing remained, the total would have been {phrase}.",
    "A past invoice for {phrase} was issued but later voided.",
]

NO_OP_TIME_PERSON = [
    "{person} had originally scheduled {phrase} for this task, but the plan changed.",
    "An earlier estimate gave {person} {phrase}, though that no longer applies.",
    "If the deadline had held, {person} would have had {phrase} to finish.",
    "{person} had set aside {phrase} previously, but the meeting was cancelled.",
]
NO_OP_TIME_NO_PERSON = [
    "An earlier schedule had allowed {phrase} for this step, but it was revised.",
    "If the original timeline had held, the process would have taken {phrase}.",
    "A previous plan reserved {phrase}, though that estimate is now outdated.",
    "Had the deadline not shifted, there would have been {phrase} available.",
]

NO_OP_DISTANCE_PERSON = [
    "{person} had originally planned to walk {phrase}, but took a shortcut.",
    "An earlier route for {person} covered {phrase}, though it was abandoned.",
    "If {person} had taken the long way, they would have traveled {phrase}.",
    "{person} had previously driven {phrase}, but that trip is unrelated here.",
]
NO_OP_DISTANCE_NO_PERSON = [
    "The original route was {phrase}, but it was changed before departure.",
    "An earlier map marked the distance as {phrase}, which proved inaccurate.",
    "Had the longer path been chosen, the trip would have been {phrase}.",
    "A previous estimate put the distance at {phrase}, but it no longer applies.",
]

NO_OP_WEIGHT_PERSON = [
    "{person} had ordered {phrase} the previous week, but it was returned.",
    "An earlier shipment of {phrase} had been sent to {person}, but was recalled.",
    "Had the supplier delivered, {person} would have received {phrase}.",
    "{person} had weighed out {phrase} earlier, though that batch was discarded.",
]
NO_OP_WEIGHT_NO_PERSON = [
    "A previous shipment of {phrase} had been returned some time ago.",
    "An earlier batch of {phrase} had been set aside and later discarded.",
    "Had the delivery arrived, there would have been {phrase} more in stock.",
    "A past order for {phrase} was placed but never fulfilled.",
]

NO_OP_COUNT_PERSON = [
    "{person} had owned {phrase} the previous month but sold them.",
    "At an earlier point, {person} had {phrase}, though that was before this.",
    "{person} had initially ordered {phrase}, but the order was cancelled.",
    "Before any of this, {person} had held {phrase}, unrelated to the current count.",
]
NO_OP_COUNT_NO_PERSON = [
    "Previously, there had been {phrase}, but those were used up long ago.",
    "A past inventory recorded {phrase}, which no longer applies here.",
    "An older shipment of {phrase} had been returned some time ago.",
    "Had the order gone through, there would have been {phrase}.",
]

NO_OP_PERCENT = [
    "Earlier, a discount of {number}% had been offered, but it had since expired.",
    "A previous survey showed {number}% agreement, though that data is outdated.",
    "Had conditions been different, {number}% of the supply would have been reserved.",
    "A past report mentioned {number}%, but that figure no longer applies.",
]


IN_TOPIC_TEMPLATES = {
    "money":    (IN_TOPIC_MONEY_PERSON, IN_TOPIC_MONEY_NO_PERSON),
    "time":     (IN_TOPIC_TIME_PERSON, IN_TOPIC_TIME_NO_PERSON),
    "distance": (IN_TOPIC_DISTANCE_PERSON, IN_TOPIC_DISTANCE_NO_PERSON),
    "weight":   (IN_TOPIC_WEIGHT_PERSON, IN_TOPIC_WEIGHT_NO_PERSON),
    "count":    (IN_TOPIC_COUNT_PERSON, IN_TOPIC_COUNT_NO_PERSON),
}

NO_OP_TEMPLATES = {
    "money":    (NO_OP_MONEY_PERSON, NO_OP_MONEY_NO_PERSON),
    "time":     (NO_OP_TIME_PERSON, NO_OP_TIME_NO_PERSON),
    "distance": (NO_OP_DISTANCE_PERSON, NO_OP_DISTANCE_NO_PERSON),
    "weight":   (NO_OP_WEIGHT_PERSON, NO_OP_WEIGHT_NO_PERSON),
    "count":    (NO_OP_COUNT_PERSON, NO_OP_COUNT_NO_PERSON),
}


TOPIC_SUBJECTS = {
    "school":   ["A student", "A classmate", "A teacher"],
    "shopping": ["A customer", "A shopper", "A passerby"],
    "work":     ["A colleague", "An employee", "A coworker"],
    "food":     ["A cook", "A patron", "A diner"],
    "money":    ["A client", "A vendor", "An accountant"],
    "time":     ["Someone nearby", "A participant", "A bystander"],
    "distance": ["A traveler", "A commuter", "A cyclist"],
    "nature":   ["A gardener", "A hiker", "A farmhand"],
}

PERSON_REJECT = {
    "Western", "Eastern", "Northern", "Southern", "Central",
    "iPhone", "iPad", "Android", "Google", "Amazon", "Netflix",
    "Striploin", "Ribeye", "Sirloin", "Brisket", "Chuck",
    "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Mr", "Mrs", "Ms", "Dr", "Jr", "Sr",
}


def is_good_person(name: str):
    if not name or len(name) < 3:
        return False
    if name in PERSON_REJECT:
        return False
    if name.isupper() or name.islower():
        return False
    if not name[0].isupper():
        return False
    return True


def get_person(persons: list[str], topics: list[str]):
    good_names = [p for p in persons if is_good_person(p)]
    if good_names:
        return random.choice(good_names)
    key = topics[0] if topics else None
    if key and key in TOPIC_SUBJECTS:
        return random.choice(TOPIC_SUBJECTS[key])
    return None


_NUM_RE = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)")


def _get_numbers(text: str):
    nums: set[float] = set()
    for m in _NUM_RE.finditer(text):
        try:
            nums.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return nums


def extract_solution_numbers(solution: str):
    return _get_numbers(solution.split("####")[0])


def extract_question_only_numbers(question: str, solution: str):
    q_nums = _get_numbers(question)
    sol_nums = extract_solution_numbers(solution)
    unused = list(q_nums - sol_nums)
    return unused if unused else list(q_nums)


def generate_fake_number(original: float, forbidden: set[float]):
    if original == int(original):
        base = int(original)
        for _ in range(20):
            delta = random.randint(max(1, base // 3), max(2, base + 3))
            fake = float(base + random.choice([-1, 1]) * delta)
            if fake > 0 and fake not in forbidden:
                return fake
        return float(base + random.randint(5, 20))
    else:
        decimals = len(str(original).rstrip("0").split(".")[-1])
        for _ in range(20):
            delta = round(random.uniform(0.3, 0.9) * abs(original), decimals)
            fake = round(original + random.choice([-1, 1]) * delta, decimals)
            if fake > 0 and fake not in forbidden:
                return fake
        return round(original + 10 ** (-decimals) * 3, decimals)


def contains_phrase_from_question(distractor: str, question: str):
    d_tokens = distractor.lower().split()
    q_lower = question.lower()
    for i in range(len(d_tokens) - 5):
        span = " ".join(d_tokens[i:i + 6])
        if span in q_lower:
            return True
    return False


REJECT_PATTERNS = [
    re.compile(r"\bitems?\b", re.IGNORECASE),
    re.compile(r"\bthings?\b", re.IGNORECASE),
    re.compile(r"\bobjects?\b", re.IGNORECASE),
    re.compile(r"\bstuff\b", re.IGNORECASE),
    re.compile(r"\d+\s*%(?!\s+(of|agreement|agreed|participants|respondents|cases|samples|voters|supply))", re.IGNORECASE),
    re.compile(r"\d+\s*g\b(?!\s+of\s+\w)", re.IGNORECASE),
    re.compile(r"\d+\s*kg\b(?!\s+of\s+\w)", re.IGNORECASE),
    re.compile(r"\d+\s*ml\b(?!\s+of\s+\w)", re.IGNORECASE),
    re.compile(r"\d+\s*oz\b(?!\s+of\s+\w)", re.IGNORECASE),
    re.compile(r"\d+\s*lbs?\b(?!\s+of\s+\w)", re.IGNORECASE),
    re.compile(r"\d+\s*mg\b(?!\s+of\s+\w)", re.IGNORECASE),
    re.compile(r"\b(friend|coworker) of\b", re.IGNORECASE),
    re.compile(r"\bkept\b.*\bat home\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*g\s+of\s+(sugar|spice)\b", re.IGNORECASE),
    re.compile(r"coupon\s+(worth|for)\s+\d+\s*(minutes?|hours?|days?|weeks?|months?|years?|miles?|km|meters?|m)\b", re.IGNORECASE),
    re.compile(r"\b(received|gift|present)\s+.*\b\d+\s*(minutes?|hours?|miles?|km|kg|g|lbs?|pounds?)\b.*\b(as a gift|as a present)\b", re.IGNORECASE),
    re.compile(r"mentioned\s+having\s+\d+\s*(minutes?|miles?|km|meters?|hours?)\b", re.IGNORECASE),
]


def passes_reject_filter(distractor: str):
    for pat in REJECT_PATTERNS:
        if pat.search(distractor):
            return False
    return True


def passes_filters(
    distractor: str,
    forbidden_numbers: set[float],
    question: str,
    min_words: int = 5,
):
    if not distractor:
        return False
    if len(distractor.split()) < min_words:
        return False
    if _get_numbers(distractor) & forbidden_numbers:
        return False
    if contains_phrase_from_question(distractor, question):
        return False
    if not passes_reject_filter(distractor):
        return False
    return True


def _fmt(n: float):
    if n == int(n):
        return str(int(n))
    return str(n)


def generate_off_topic(topics: list[str], question: str, max_retries: int = 10):
    forbidden_domains = set()
    for t in topics:
        forbidden_domains.update(TOPIC_AVOID_DOMAINS.get(t, set()))

    available = [d for d in OFF_TOPIC_BY_DOMAIN if d not in forbidden_domains]
    if not available:
        available = list(OFF_TOPIC_BY_DOMAIN.keys())

    q_nums = _get_numbers(question)
    for _ in range(max_retries):
        domain = random.choice(available)
        candidate = random.choice(OFF_TOPIC_BY_DOMAIN[domain])
        if passes_filters(candidate, q_nums, question, min_words=5):
            return candidate
    return None


def _render_by_category(
    category: str,
    templates_bundle: dict,
    phrase: str,
    person,
):
    if category not in templates_bundle:
        return None
    person_tpls, no_person_tpls = templates_bundle[category]
    pool = person_tpls if person else no_person_tpls
    tpl = random.choice(pool)
    if person:
        return tpl.format(person=person, phrase=phrase)
    return tpl.format(phrase=phrase)


def generate_in_topic(
    hints: dict,
    topics: list[str],
    gold_answer,
    question: str,
    solution: str,
    max_retries: int = 15,
):
    numbers = hints.get("numbers", [])
    if not numbers:
        return None

    sol_nums = extract_solution_numbers(solution)
    forbidden = set(sol_nums)
    forbidden.update(_get_numbers(question))
    if gold_answer is not None:
        forbidden.add(float(gold_answer))

    pool = list(sol_nums) if sol_nums else [float(n) for n in numbers]

    raw_unit = random.choice(hints["units"]) if hints.get("units") else None
    category = classify_unit(raw_unit) if raw_unit else None
    person = get_person(hints.get("persons", []), topics)

    if category == "weight" and not allow_weight_templates(raw_unit, topics, question):
        return None

    if category == "weight" and not allow_weight_templates(raw_unit, topics, question):
        return None

    if category == "percent":
        for _ in range(max_retries):
            seed_num = random.choice(pool)
            fake = generate_fake_number(seed_num, forbidden)
            if fake > 100:
                fake = fake % 100 + 1
            candidate = random.choice(IN_TOPIC_PERCENT).format(number=_fmt(fake))
            if passes_filters(candidate, forbidden, question, min_words=6):
                return candidate
        return None

    if category is None:
        return None

    for _ in range(max_retries):
        seed_num = random.choice(pool)
        fake = generate_fake_number(seed_num, forbidden)

        phrase = build_phrase(raw_unit, fake, category)
        if not phrase:
            continue

        candidate = _render_by_category(category, IN_TOPIC_TEMPLATES, phrase, person)
        if not candidate:
            continue

        if passes_filters(candidate, forbidden, question, min_words=6):
            return candidate
    return None


def generate_no_op(
    hints: dict,
    topics: list[str],
    gold_answer: float | None,
    question: str,
    solution: str,
    max_retries: int = 15,
):
    numbers = hints.get("numbers", [])
    if not numbers:
        return None

    sol_nums = extract_solution_numbers(solution)
    forbidden = set(sol_nums)
    if gold_answer is not None:
        forbidden.add(float(gold_answer))

    candidates = extract_question_only_numbers(question, solution)
    pool = candidates if candidates else [float(n) for n in numbers]

    raw_unit = random.choice(hints["units"]) if hints.get("units") else None
    category = classify_unit(raw_unit) if raw_unit else None
    person = get_person(hints.get("persons", []), topics)

    if category == "percent":
        for _ in range(max_retries):
            seed_num = random.choice(pool)
            fake = generate_fake_number(seed_num, forbidden)
            if fake > 100:
                fake = fake % 100 + 1
            candidate = random.choice(NO_OP_PERCENT).format(number=_fmt(fake))
            if passes_filters(candidate, forbidden, question, min_words=8):
                return candidate
        return None

    if category is None:
        return None

    for _ in range(max_retries):
        seed_num = random.choice(pool)
        fake = generate_fake_number(seed_num, forbidden)

        phrase = build_phrase(raw_unit, fake, category)
        if not phrase:
            continue

        candidate = _render_by_category(category, NO_OP_TEMPLATES, phrase, person)
        if not candidate:
            continue

        if passes_filters(candidate, forbidden, question, min_words=8):
            return candidate
    return None

def generate_distractors(
    input_path: str = "gsm8k_extracted.json",
    output_path: str = "gsm8k_distractors.json",
):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    fail_counts = {"off_topic": 0, "in_topic": 0, "no_op": 0}

    for i, rec in enumerate(data):
        hints = rec["distractor_hints"]
        topics = hints["off_topic"]["topics"]
        question = rec["question"]
        solution = rec["solution"]
        gold_answer = rec.get("gold_answer")

        in_topic_hints = dict(hints["in_topic"])
        if not in_topic_hints.get("persons"):
            in_topic_hints["persons"] = hints["off_topic"].get("persons", [])

        dist_off = generate_off_topic(topics, question)
        dist_in = generate_in_topic(in_topic_hints, topics, gold_answer, question, solution)
        dist_noop = generate_no_op(in_topic_hints, topics, gold_answer, question, solution)

        if dist_off is None: fail_counts["off_topic"] += 1
        if dist_in is None: fail_counts["in_topic"] += 1
        if dist_noop is None: fail_counts["no_op"] += 1

        records.append({
            "id": rec["id"],
            "question": question,
            "solution": solution,
            "gold_answer": gold_answer,
            "distractors": {
                "off_topic": dist_off,
                "in_topic": dist_in,
                "no_op": dist_noop,
            },
        })

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(data)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Coverage:")
    for dtype in ("off_topic", "in_topic", "no_op"):
        filled = sum(1 for r in records if r["distractors"][dtype])
        print(f"  {dtype:10} {filled:5}/{len(records)} ({100 * filled / len(records):.1f}%), "
              f"failed: {fail_counts[dtype]}")
    return records


def verify(records: list[dict]) -> None:
    NO_OP_MARKERS = [
        "previously", "originally", "last year", "last week", "last month",
        "before", "earlier", "at one point", "at an earlier", "older",
        "if ", "had ", "would have",
        "expired", "cancelled", "returned", "discarded", "outdated",
        "no longer", "unrelated", "voided", "recalled", "abandoned",
    ]
    issues = []
    reject_hits = 0

    for rec in records:
        id_ = rec["id"]
        dist = rec["distractors"]
        gold = rec.get("gold_answer")
        sol_nums = extract_solution_numbers(rec["solution"])

        for dtype in ("off_topic", "in_topic", "no_op"):
            d = dist.get(dtype)
            if not d:
                continue
            d_nums = _get_numbers(d)
            if gold is not None and float(gold) in d_nums:
                issues.append(f"ID {id_} [{dtype}]: contains gold answer {gold}")
            if dtype in ("in_topic", "no_op") and (d_nums & sol_nums):
                issues.append(f"ID {id_} [{dtype}]: collides with solution numbers")
            if dtype == "no_op" and not any(m in d.lower() for m in NO_OP_MARKERS):
                issues.append(f"ID {id_} [no_op]: missing marker in: {d[:80]}")
            if not passes_reject_filter(d):
                reject_hits += 1
                issues.append(f"ID {id_} [{dtype}]: reject pattern leaked: {d[:80]}")

    print(f"\nVerified {len(records)} records.")
    if reject_hits:
        print(f"[!] {reject_hits} reject-pattern leaks (bug - filter should have caught)")
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues[:20]:
            print(f"  {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more.")
    else:
        print("All checks passed!")

def generate_for_both_splits() -> None:
    train_records = generate_distractors(
        input_path="gsm8k_train_extracted.json",
        output_path="gsm8k_train_distractors.json",
    )
    verify(train_records)

    test_records = generate_distractors(
        input_path="gsm8k_test_extracted.json",
        output_path="gsm8k_test_distractors.json",
    )
    verify(test_records)


if __name__ == "__main__":
    generate_for_both_splits()
