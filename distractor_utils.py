from __future__ import annotations

import random
import re
from typing import Any

NAMES = [
    "Alex", "Jordan", "Taylor", "Sam", "Casey", "Riley", "Morgan", "Jamie",
    "Avery", "Parker", "Drew", "Blake", "Quinn", "Cameron", "Logan", "Hayden",
]

OFF_TOPIC_TEMPLATES = [
    "{name} scored {n} points in the basketball game yesterday.",
    "The weather forecast said there would be {n} inches of snow tomorrow.",
    "{name} watched {n} episodes of a TV show last weekend.",
    "There are {n} birds sitting on the fence outside the house.",
    "{name} owns {n} different board games at home.",
    "The movie started at {n} o'clock in the evening.",
    "{name} planted {n} flowers in the garden last summer.",
    "The temperature outside was {n} degrees.",
    "{name} drove {n} miles to get to work.",
    "There are {n} {item} in {name}'s garden.",
    "{name} has been working at the company for {n} years.",
    "The store across the street has {n} employees.",
]

IN_TOPIC_TEMPLATES = [
    "Each {item} weighs about {n} grams.",
    "The {item} come in {n} different colors.",
    "The store also sells {n} types of {item2}.",
    "{name} considered buying {n} {item} but decided against it.",
    "The {item} were on sale for {n} days last month.",
    "There were {n} other customers in the store.",
    "{name} originally wanted {n} {item} but changed the order.",
    "The {item} are delivered in boxes of {n}.",
    "The warranty on each {item} lasts {n} months.",
    "The store has been selling {item} for {n} years.",
]

NOOP_TEMPLATES = [
    "{name} also had a coupon for {n}% off, but it had already expired.",
    "If each {item} were {n}% heavier, the total weight would be different, but that is not the case here.",
    "{name} thought about buying {n} more {item}, but did not.",
    "The price of {item} had increased by ${n} the previous week, but has since returned to normal.",
    "There is a {n}-item limit per customer, but {name} was well under that.",
    "{name} had ${n} in savings, though this was not used for this purchase.",
    "The store offers a {n}% discount for orders over $1000, but this order does not qualify.",
    "Each {item} comes with {n} accessories, which do not affect the price.",
    "The delivery fee is ${n}, but {name} gets free delivery.",
    "{name} earns ${n} per hour at work, which is unrelated to this problem.",
]

ITEMS = [
    "apples", "books", "pencils", "chairs", "shirts", "cookies",
    "toys", "flowers", "stamps", "marbles", "stickers", "cards",
    "bottles", "boxes", "plates", "candles", "ribbons", "badges",
]


def extract_entities_from_question(question: str) -> dict[str, list[Any]]:
    words = question.split()
    names_found: list[str] = []
    for i, word in enumerate(words):
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if clean and clean[0].isupper() and len(clean) > 2 and i > 0:
            names_found.append(clean)

    numbers_found = re.findall(r"\b\d+\.?\d*\b", question)
    numbers = [float(n) if "." in n else int(n) for n in numbers_found]

    return {
        "names": list(dict.fromkeys(names_found)) or [random.choice(NAMES)],
        "numbers": numbers or [random.randint(2, 50)],
    }


def choose_template_pool(distractor_type: str) -> list[str]:
    if distractor_type == "off_topic":
        return OFF_TOPIC_TEMPLATES
    if distractor_type == "in_topic":
        return IN_TOPIC_TEMPLATES
    if distractor_type == "noop":
        return NOOP_TEMPLATES
    if distractor_type == "mixed":
        return OFF_TOPIC_TEMPLATES + IN_TOPIC_TEMPLATES + NOOP_TEMPLATES
    raise ValueError(f"Unknown distractor_type: {distractor_type}")


def build_distractor_sentence(question: str, distractor_type: str) -> str:
    entities = extract_entities_from_question(question)
    template = random.choice(choose_template_pool(distractor_type))

    if distractor_type == "off_topic":
        available_names = [n for n in NAMES if n not in entities["names"]]
        name = random.choice(available_names) if available_names else random.choice(NAMES)
    else:
        name = random.choice(entities["names"])

    if entities["numbers"]:
        base = random.choice(entities["numbers"])
        n = random.randint(max(1, int(base * 0.5)), int(base * 2.0) + 1)
    else:
        n = random.randint(2, 100)

    item = random.choice(ITEMS)
    item2 = random.choice([i for i in ITEMS if i != item])
    return template.format(name=name, n=n, item=item, item2=item2)


def insert_distractors(
    question: str,
    distractor_type: str,
    n_distractors: int = 1,
) -> tuple[str, list[int]]:
    sentences = re.split(r"(?<=[.!?])\s+", question.strip())
    sentences = [s for s in sentences if s]
    positions: list[int] = []

    if len(sentences) <= 1:
        distractors = [build_distractor_sentence(question, distractor_type) for _ in range(n_distractors)]
        modified = question.strip() + " " + " ".join(distractors)
        positions = list(range(1, 1 + len(distractors)))
        return modified.strip(), positions

    for _ in range(n_distractors):
        distractor = build_distractor_sentence(question, distractor_type)
        insert_pos = random.randint(1, max(1, len(sentences) - 1))
        sentences.insert(insert_pos, distractor)
        positions.append(insert_pos)

    return " ".join(sentences).strip(), positions
