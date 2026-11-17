"""Dataset, prompts and quality checks for the Product Description Enhancer.

Pure Python, standard library only, so everything here can be tested without
PyTorch. The LoRA model code lives in llm.py.

An example is a plain dict:
    {"name": ..., "category": ..., "specs": {"Battery life": "12 hours", ...},
     "keywords": ["wireless earbuds", ...], "tone": "playful", "description": ...}
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
from collections import Counter

TONES = ("professional", "playful", "luxury")
DEFAULT_TONE = "professional"
MIN_DESCRIPTION_CHARS = 20
MAX_DESCRIPTION_CHARS = 1200

# --------------------------------------------------------------------------
# Specs, keywords and the prompt format
# --------------------------------------------------------------------------


def parse_specs(text: str) -> dict[str, str]:
    """Parse 'Battery life: 12 hours' lines (or ';'-separated items) into a dict."""
    specs: dict[str, str] = {}
    for item in re.split(r"[;\n]+", text or ""):
        item = item.strip()
        if not item:
            continue
        separator = ":" if ":" in item else "=" if "=" in item else None
        if separator is None:
            raise ValueError(f"spec '{item}' needs a name and a value, like 'Battery life: 12 hours'")
        key, value = (part.strip() for part in item.split(separator, 1))
        if not key or not value:
            raise ValueError(f"spec '{item}' needs a name and a value, like 'Battery life: 12 hours'")
        specs[key] = value
    return specs


def format_specs(specs: dict[str, str]) -> str:
    return "; ".join(f"{key}: {value}" for key, value in specs.items())


def parse_keywords(value) -> list[str]:
    """Accept 'a, b; c' or a list. Duplicates are dropped; at most 6 are kept."""
    if value is None:
        return []
    parts = value if isinstance(value, (list, tuple)) else re.split(r"[,;\n]+", str(value))
    seen, keywords = set(), []
    for part in parts:
        keyword = str(part).strip()
        if keyword and keyword.lower() not in seen:
            seen.add(keyword.lower())
            keywords.append(keyword)
    return keywords[:6]


def build_prompt(name: str, category: str, specs: dict[str, str], keywords, tone: str) -> str:
    """The text the model is trained on, and prompted with. It ends at 'Description:'."""
    return "\n".join([
        f"Product: {name.strip()}",
        f"Category: {(category or 'general').strip()}",
        f"Specs: {format_specs(specs)}",
        f"Keywords: {', '.join(keywords) if keywords else 'none'}",
        f"Tone: {(tone or DEFAULT_TONE).strip().lower()}",
        "Description:",
    ])


def example_prompt(example: dict) -> str:
    return build_prompt(example["name"], example["category"], example["specs"],
                        example["keywords"], example["tone"])


def example_completion(example: dict) -> str:
    """What the model should write after 'Description:' (it starts with a space)."""
    return " " + example["description"].strip()


# --------------------------------------------------------------------------
# Reading datasets (JSONL or CSV)
# --------------------------------------------------------------------------


def normalise_record(record: dict, require_description: bool = True):
    """Turn a raw row into an example. Returns (example, []) or (None, problems)."""
    rec = {str(k).strip().lower(): v for k, v in record.items() if k is not None}
    problems: list[str] = []

    name = str(rec.get("name") or "").strip()
    if not name:
        problems.append("missing name")

    raw_specs = rec.get("specs")
    specs: dict[str, str] = {}
    specs_error = False
    if isinstance(raw_specs, dict):
        specs = {str(k).strip(): str(v).strip() for k, v in raw_specs.items()
                 if str(k).strip() and str(v).strip()}
    else:
        try:
            specs = parse_specs(str(raw_specs or ""))
        except ValueError as exc:
            problems.append(str(exc))
            specs_error = True
    if not specs and not specs_error:
        problems.append("missing specs")

    description = " ".join(str(rec.get("description") or "").split())
    if require_description:
        if len(description) < MIN_DESCRIPTION_CHARS:
            problems.append("description is missing or too short")
        elif len(description) > MAX_DESCRIPTION_CHARS:
            problems.append("description is too long")

    if problems:
        return None, problems
    return {
        "name": name,
        "category": str(rec.get("category") or "general").strip() or "general",
        "specs": specs,
        "keywords": parse_keywords(rec.get("keywords")),
        "tone": str(rec.get("tone") or DEFAULT_TONE).strip().lower() or DEFAULT_TONE,
        "description": description,
    }, []


def parse_dataset_text(text: str, kind: str = "jsonl"):
    """Parse a JSONL or CSV dataset. Returns (examples, problems); bad rows are skipped."""
    kind = kind.lower().lstrip(".")
    rows: list[tuple[str, dict]] = []
    problems: list[str] = []
    if kind in ("jsonl", "ndjson"):
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"line {number}: not valid JSON")
                continue
            if not isinstance(record, dict):
                problems.append(f"line {number}: expected a JSON object")
                continue
            rows.append((f"line {number}", record))
    elif kind == "csv":
        for number, row in enumerate(csv.DictReader(io.StringIO(text)), 2):
            rows.append((f"row {number}", row))
    else:
        raise ValueError(f"unsupported dataset format '{kind}' - use .jsonl or .csv")

    examples = []
    for label, record in rows:
        example, issues = normalise_record(record)
        if example:
            examples.append(example)
        else:
            problems.extend(f"{label}: {issue}" for issue in issues)
    return examples, problems


def split_examples(examples: list, val_fraction: float = 0.1, seed: int = 0):
    """Shuffle (reproducibly) and hold some examples out for validation."""
    if len(examples) < 2:
        raise ValueError("need at least 2 examples to hold some out for validation")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    n_val = min(len(examples) - 1, max(1, round(len(examples) * val_fraction)))
    val_ids = set(order[:n_val])
    train = [e for i, e in enumerate(examples) if i not in val_ids]
    val = [e for i, e in enumerate(examples) if i in val_ids]
    return train, val


def dataset_summary(examples: list) -> dict:
    return {
        "examples": len(examples),
        "categories": dict(Counter(e["category"] for e in examples)),
        "tones": dict(Counter(e["tone"] for e in examples)),
        "average_words": round(sum(len(e["description"].split()) for e in examples) / max(1, len(examples)), 1),
    }


# --------------------------------------------------------------------------
# A synthetic starter dataset
# --------------------------------------------------------------------------
# Each product type has its own keyword phrases. Each category has spec values and
# sentence templates that quote the spec value, so a description always mentions
# the specs it was given. Openers and closers vary by tone.

BRANDS = ["Aero", "Nova", "Luma", "Terra", "Zenith", "Pico", "Orbit", "Vivo", "Kora", "Mira",
          "Atlas", "Ember", "Solis", "Nimbus", "Alder", "Quill"]
SUFFIXES = ["", " Pro", " Lite", " One", " Plus", " Air", " Max"]

CATALOG = {
    "audio": {
        "types": {
            "Wireless Earbud Set": ["wireless earbuds", "bluetooth earbuds", "sports earbuds", "true wireless earbuds"],
            "Over-Ear Headset": ["over-ear headphones", "wireless headphones", "bluetooth headphones", "comfortable headphones"],
            "Portable Bluetooth Speaker": ["portable speakers", "bluetooth speakers", "waterproof speakers", "outdoor speakers"],
        },
        "specs": {
            "Battery life": (["8 hours", "12 hours", "20 hours", "30 hours", "40 hours"], [
                "Up to {v} of battery life keeps the music going from the first coffee to the last commute.",
                "With {v} of battery life, you can leave the charger at home."]),
            "Bluetooth": (["5.0", "5.2", "5.3"], [
                "Bluetooth {v} delivers a fast, stable connection to your phone and laptop.",
                "Pairing is quick and dropouts are rare thanks to Bluetooth {v}."]),
            "Water resistance": (["IPX4", "IPX5", "IPX7"], [
                "An {v} rating protects against sweat and splashes, so it keeps up with your day.",
                "Rated {v} for water resistance, it handles rain and workouts with ease."]),
            "Charging": (["USB-C", "USB-C fast charging", "USB-C and wireless charging"], [
                "Top up with {v} and spend less time plugged in and more time listening.",
                "Charging is simple with {v}, so it is always ready when you are."]),
        },
    },
    "kitchen": {
        "types": {
            "Electric Kettle": ["electric kettles", "fast boil kettles", "stainless steel kettles", "tea kettles"],
            "Countertop Blender": ["countertop blenders", "smoothie blenders", "powerful blenders", "kitchen blenders"],
            "Air Fryer": ["air fryers", "healthy air fryers", "family air fryers", "countertop air fryers"],
        },
        "specs": {
            "Capacity": (["1.0 L", "1.5 L", "1.7 L", "3.5 L", "5 L"], [
                "A {v} capacity is big enough for the whole household and still compact on the counter.",
                "With {v} of capacity, there is room for everything you want to make."]),
            "Power": (["1200 W", "1500 W", "1800 W"], [
                "Rated at {v}, it gets up to speed in moments so you spend less time waiting.",
                "{v} of power means fast, even results every time."]),
            "Material": (["stainless steel", "BPA-free Tritan", "borosilicate glass", "brushed aluminium"], [
                "Built from {v}, it looks great and is made to last.",
                "The {v} body is easy to clean and tough enough for daily use."]),
            "Safety": (["auto shut-off", "boil-dry protection", "cool-touch handle", "overheat protection"], [
                "Peace of mind comes standard with {v}.",
                "Built-in {v} keeps busy mornings stress-free."]),
        },
    },
    "fitness": {
        "types": {
            "Yoga Mat": ["yoga mats", "non-slip yoga mats", "eco-friendly yoga mats", "thick yoga mats"],
            "Pilates Mat": ["pilates mats", "exercise mats", "studio pilates mats", "non-slip pilates mats"],
            "Exercise Mat": ["exercise mats", "workout mats", "gym mats", "fitness mats"],
        },
        "specs": {
            "Thickness": (["4 mm", "6 mm", "8 mm", "10 mm"], [
                "At {v} thick, it cushions your joints without costing you balance.",
                "Its {v} thickness strikes the right balance between comfort and stability."]),
            "Material": (["natural rubber", "TPE", "cork-top natural rubber", "PVC-free foam"], [
                "Made from {v}, it grips the floor and stays put through every flow.",
                "The {v} surface feels great underfoot and is easy to wipe clean."]),
            "Length": (["173 cm", "183 cm", "190 cm"], [
                "At {v} long, there is plenty of room to stretch out.",
                "Its {v} length suits taller practitioners as well."]),
            "Weight": (["1.0 kg", "1.5 kg", "2.2 kg"], [
                "Weighing just {v}, it rolls up and travels easily to class.",
                "At {v}, it is light enough to carry and sturdy enough to stay put."]),
        },
    },
    "backpacks": {
        "types": {
            "Hiking Backpack": ["hiking backpacks", "trekking backpacks", "waterproof backpacks", "lightweight hiking packs"],
            "Daypack": ["daypacks", "lightweight backpacks", "everyday backpacks", "commuter backpacks"],
            "Travel Backpack": ["travel backpacks", "carry-on backpacks", "weekender backpacks", "durable travel packs"],
        },
        "specs": {
            "Capacity": (["20 L", "28 L", "35 L", "45 L"], [
                "A {v} capacity holds everything you need without feeling bulky.",
                "With {v} of space, packing for the day or the weekend is easy."]),
            "Weight": (["0.8 kg", "1.1 kg", "1.4 kg"], [
                "Weighing only {v}, it keeps your load light on long days.",
                "At {v}, it stays comfortable mile after mile."]),
            "Material": (["recycled nylon", "ripstop polyester", "water-resistant canvas"], [
                "Made from {v}, it is built to handle rough trails and rushed commutes.",
                "The {v} fabric is tough, light and ready for the weather."]),
            "Pockets": (["7 pockets", "9 pockets", "12 pockets"], [
                "With {v}, everything has its place and nothing gets lost.",
                "{v} keep your gear organised and within easy reach."]),
        },
    },
    "lighting": {
        "types": {
            "LED Desk Lamp": ["led desk lamps", "dimmable desk lamps", "office lamps", "study lamps"],
            "Floor Lamp": ["floor lamps", "modern floor lamps", "reading lamps", "living room lamps"],
            "Bedside Lamp": ["bedside lamps", "night lamps", "touch lamps", "dimmable bedside lamps"],
        },
        "specs": {
            "Brightness": (["400 lumens", "800 lumens", "1200 lumens"], [
                "Up to {v} of light fills your space without harsh glare.",
                "A bright {v} output keeps reading, working and relaxing comfortable."]),
            "Colour temperature": (["2700K-6500K", "3000K-5000K", "2700K-4000K"], [
                "Choose any colour temperature across {v} to suit your mood or task.",
                "A {v} colour range lets you switch from cosy evenings to focused mornings."]),
            "Power": (["8 W", "12 W", "18 W"], [
                "Using only {v}, it saves energy while keeping the room bright.",
                "Drawing just {v}, it keeps running costs low and the room bright."]),
            "Control": (["touch dimming", "USB charging port", "memory function", "timer"], [
                "Handy {v} puts convenience right at your fingertips.",
                "Thanks to {v}, using it is effortless."]),
        },
    },
    "drinkware": {
        "types": {
            "Insulated Water Bottle": ["insulated water bottles", "stainless steel bottles", "leak-proof bottles", "reusable water bottles"],
            "Travel Mug": ["travel mugs", "insulated travel mugs", "coffee mugs", "leak-proof travel mugs"],
            "Coffee Tumbler": ["coffee tumblers", "insulated tumblers", "hot drink tumblers", "double-wall tumblers"],
        },
        "specs": {
            "Capacity": (["350 ml", "500 ml", "750 ml", "1 L"], [
                "A {v} capacity is just right for a day of sipping.",
                "Holding {v}, it fits your cup holder and your routine."]),
            "Insulation": (["cold for 24 hours", "hot for 12 hours", "hot for 8 hours and cold for 18 hours"], [
                "Insulated to keep drinks {v}, so the last sip is as good as the first.",
                "Double-wall insulation keeps drinks {v}."]),
            "Material": (["18/8 stainless steel", "BPA-free Tritan", "double-wall stainless steel"], [
                "Made from {v}, it is tough, safe and built for daily use.",
                "The {v} body resists dents and never holds on to flavours."]),
            "Lid": (["leak-proof", "flip-top", "screw-on"], [
                "The {v} lid keeps your bag dry and your drink secure.",
                "A {v} lid makes every sip simple."]),
        },
    },
}

OPENERS = {
    "professional": [
        "Introducing the {name}, a dependable choice among {kw1}.",
        "The {name} stands out among {kw1} for its consistent, reliable performance.",
        "Looking for quality {kw1}? The {name} delivers.",
        "Designed for everyday use, the {name} sets a new standard for {kw1}.",
    ],
    "playful": [
        "Say hello to the {name}, proof that {kw1} can be fun!",
        "Searching for {kw1} with personality? Meet the {name}!",
        "The {name} is here to make {kw1} your new favourite thing.",
        "Who said {kw1} have to be boring? Not the {name}!",
    ],
    "luxury": [
        "Crafted with exceptional care, the {name} redefines {kw1}.",
        "The {name} brings understated elegance to {kw1}.",
        "Discover the {name}, where refined design meets effortless performance in {kw1}.",
        "For those who expect the finest in {kw1}, there is the {name}.",
    ],
}
CLOSERS = {
    "professional": [
        "A practical, long-lasting choice for anyone shopping for {kw2}.",
        "Order the {name} today and see why it is a top pick for {kw2}.",
        "The {name}: dependable value in {kw2}.",
    ],
    "playful": [
        "Grab yours today and join everyone raving about {kw2}!",
        "Add it to your cart and let the good times roll with {kw2} you will love.",
        "Treat yourself to {kw2} that bring a smile every single day!",
    ],
    "luxury": [
        "Experience {kw2} the way they were meant to be.",
        "Own the {name} and enjoy a standard in {kw2} that few can match.",
        "An investment in {kw2} that you will appreciate every day.",
    ],
}


def _synthetic_example(rng: random.Random, category: str) -> dict:
    entry = CATALOG[category]
    product_type = rng.choice(list(entry["types"]))
    name = f"{rng.choice(BRANDS)}{rng.choice(SUFFIXES)} {product_type}"
    spec_names = list(entry["specs"])
    chosen = sorted(rng.sample(spec_names, 3), key=spec_names.index)
    specs, sentences = {}, []
    for key in chosen:
        values, templates = entry["specs"][key]
        value = rng.choice(values)
        specs[key] = value
        sentences.append(rng.choice(templates).format(v=value))
    kw1, kw2 = rng.sample(entry["types"][product_type], 2)
    tone = rng.choice(TONES)
    opener = rng.choice(OPENERS[tone]).format(name=name, kw1=kw1)
    closer = rng.choice(CLOSERS[tone]).format(name=name, kw2=kw2)
    return {
        "name": name,
        "category": category,
        "specs": specs,
        "keywords": [kw1, kw2],
        "tone": tone,
        "description": " ".join([opener, *sentences, closer]),
    }


def synthesize_dataset(n: int = 240, seed: int = 7) -> list[dict]:
    """A reproducible starter dataset, balanced across categories."""
    rng = random.Random(seed)
    categories = list(CATALOG)
    return [_synthetic_example(rng, categories[i % len(categories)]) for i in range(n)]


# --------------------------------------------------------------------------
# Tokenization helpers (work with any tokenizer that has encode() and eos_token_id)
# --------------------------------------------------------------------------


def tokenize_example(tokenizer, prompt: str, completion: str, max_length: int = 256) -> dict:
    """Token ids for prompt + completion. Labels are -100 on the prompt, so the
    model is only trained to write the description (and then the end token)."""
    prompt_ids = list(tokenizer.encode(prompt))
    completion_ids = list(tokenizer.encode(completion)) + [tokenizer.eos_token_id]
    if len(prompt_ids) >= max_length:
        raise ValueError("a product prompt is longer than max_length - shorten the specs or raise it")
    completion_ids = completion_ids[: max_length - len(prompt_ids)]
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def pad_batch(items: list[dict], pad_id: int) -> dict:
    """Right-pad a list of tokenized examples to the same length."""
    longest = max(len(item["input_ids"]) for item in items)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in items:
        pad = longest - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
        batch["attention_mask"].append([1] * len(item["input_ids"]) + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    return batch


# --------------------------------------------------------------------------
# Cleaning generated text and measuring quality
# --------------------------------------------------------------------------

_STOP_MARKERS = ("<|endoftext|>", "\n\n", "\nProduct:", "\nCategory:", "\nSpecs:",
                 "\nKeywords:", "\nTone:", "\nDescription:")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MIN_WORDS, MAX_WORDS = 25, 150


def finish_description(text: str) -> str:
    """Cut a raw generation at the first sign of a new example, then drop a dangling fragment."""
    text = text.lstrip()
    for marker in _STOP_MARKERS:
        position = text.find(marker)
        if position != -1:
            text = text[:position]
    text = " ".join(text.split())
    if text and text[-1] not in ".!?":
        last_end = max(text.rfind(c) for c in ".!?")
        if last_end >= 0:
            text = text[: last_end + 1]
    return text


def word_count(text: str) -> int:
    return len(text.split())


def numbers_in(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def spec_coverage(description: str, specs: dict[str, str]):
    """Share of the specs whose value appears in the description, and which are missing."""
    if not specs:
        return 1.0, []
    lowered = description.casefold()
    found = numbers_in(description)
    missing = []
    for key, value in specs.items():
        numbers = _NUMBER.findall(value)
        if value.casefold() in lowered:
            continue
        if numbers and all(number in found for number in numbers):
            continue
        missing.append(key)
    return 1 - len(missing) / len(specs), missing


def keyword_coverage(description: str, keywords):
    if not keywords:
        return 1.0, []
    lowered = description.casefold()
    missing = [k for k in keywords if k.casefold() not in lowered]
    return 1 - len(missing) / len(keywords), missing


def unsupported_numbers(description: str, allowed_texts) -> list[str]:
    """Numbers in the description that appear nowhere in the input (a hallucination sign)."""
    allowed = numbers_in(" ".join(allowed_texts))
    return sorted(numbers_in(description) - allowed)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.lower())


def repetition_rate(text: str, n: int = 3) -> float:
    """Share of word n-grams that repeat inside the description (0 = none)."""
    words = _words(text)
    grams = list(zip(*[words[i:] for i in range(n)]))
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def ngram_set(texts, n: int = 5) -> set:
    grams: set = set()
    for text in texts:
        words = _words(text)
        grams.update(zip(*[words[i:] for i in range(n)]))
    return grams


def copy_rate(text: str, reference_ngrams: set, n: int = 5) -> float:
    """Share of the description's word n-grams that appear verbatim in the training data."""
    words = _words(text)
    grams = list(zip(*[words[i:] for i in range(n)]))
    return sum(g in reference_ngrams for g in grams) / len(grams) if grams else 0.0


def assess(description: str, specs: dict[str, str], keywords=(), name: str = "",
           category: str = "", reference_ngrams: set | None = None) -> dict:
    """Automatic relevance and quality checks. They support, but do not replace, a manual read."""
    words = word_count(description)
    coverage, missing_specs = spec_coverage(description, specs)
    keyword_cov, missing_keywords = keyword_coverage(description, list(keywords))
    allowed = [name, category, format_specs(specs), ", ".join(keywords)]
    bad_numbers = unsupported_numbers(description, allowed)
    repetition = repetition_rate(description)

    flags = []
    if words < MIN_WORDS:
        flags.append(f"short ({words} words)")
    if words > MAX_WORDS:
        flags.append(f"long ({words} words)")
    if missing_specs:
        flags.append("leaves out specs: " + ", ".join(missing_specs))
    if missing_keywords:
        flags.append("leaves out keywords: " + ", ".join(missing_keywords))
    if bad_numbers:
        flags.append("numbers not in the input: " + ", ".join(bad_numbers))
    if repetition > 0.15:
        flags.append("repeats itself")

    result = {
        "words": words,
        "spec_coverage": coverage,
        "missing_specs": missing_specs,
        "keyword_coverage": keyword_cov,
        "missing_keywords": missing_keywords,
        "unsupported_numbers": bad_numbers,
        "repetition": repetition,
        "novelty": None if reference_ngrams is None else 1 - copy_rate(description, reference_ngrams),
        "flags": flags,
        "ok": not flags,
    }
    return result


# --------------------------------------------------------------------------
# Review report
# --------------------------------------------------------------------------


def average_metrics(assessments: list[dict]) -> dict:
    """Averages over a list of assess() results."""
    count = max(1, len(assessments))
    novelties = [a["novelty"] for a in assessments if a["novelty"] is not None]
    return {
        "words": sum(a["words"] for a in assessments) / count,
        "spec_coverage": sum(a["spec_coverage"] for a in assessments) / count,
        "keyword_coverage": sum(a["keyword_coverage"] for a in assessments) / count,
        "unsupported_numbers": sum(len(a["unsupported_numbers"]) for a in assessments) / count,
        "repetition": sum(a["repetition"] for a in assessments) / count,
        "novelty": sum(novelties) / len(novelties) if novelties else None,
        "pass_rate": sum(a["ok"] for a in assessments) / count,
    }


def format_review(rows: list[dict], meta: dict) -> str:
    """Markdown report for the manual review: outputs side by side, metrics, blank score lines.

    Each row: {"example": example dict, "base": {"text", "metrics"}, "tuned": {"text", "metrics"}}.
    """
    lines = ["# Product description review", ""]
    for key, value in meta.items():
        lines.append(f"- **{key}:** {value}")
    lines += ["", "## Automatic checks (averages)", "",
              "| check | base model | fine-tuned |", "|---|---|---|"]
    base_avg = average_metrics([r["base"]["metrics"] for r in rows])
    tuned_avg = average_metrics([r["tuned"]["metrics"] for r in rows])
    labels = [("spec_coverage", "specs mentioned", "{:.0%}"), ("keyword_coverage", "keywords used", "{:.0%}"),
              ("unsupported_numbers", "numbers not in input (per description)", "{:.2f}"),
              ("repetition", "repetition", "{:.0%}"), ("novelty", "novelty vs training text", "{:.0%}"),
              ("words", "words", "{:.0f}"), ("pass_rate", "descriptions with no flags", "{:.0%}")]
    for key, label, fmt in labels:
        cells = ["n/a" if avg[key] is None else fmt.format(avg[key]) for avg in (base_avg, tuned_avg)]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Examples for manual review", "",
              "Score each description from 1 to 5 for **relevance** (does it match the specs, and",
              "invent nothing?) and **creativity** (is it engaging, not just a template?).", ""]
    for number, row in enumerate(rows, 1):
        example = row["example"]
        lines += [f"### {number}. {example['name']} ({example['tone']})", "",
                  f"Specs: {format_specs(example['specs'])}",
                  f"Keywords: {', '.join(example['keywords']) or 'none'}", ""]
        for label, key in (("Base model", "base"), ("Fine-tuned", "tuned")):
            result = row[key]
            flags = "; ".join(result["metrics"]["flags"]) or "none"
            lines += [f"**{label}:** {result['text'] or '(empty)'}", "", f"Flags: {flags}", ""]
        lines += ["Relevance (1-5): ___   Creativity (1-5): ___   Notes: ", ""]
    return "\n".join(lines)
