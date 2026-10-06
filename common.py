"""Shared schema, prompt, output parsing and scoring for training, evaluation and the demo."""

import json
import re

import transformers

INTENTS = [
    "delivery_delay",
    "refund_request",
    "return_exchange",
    "cancel_order",
    "payment_issue",
    "damaged_product",
    "account_issue",
    "general_question",
]
SENTIMENTS = ["positive", "neutral", "negative"]
URGENCIES = ["low", "medium", "high"]
LANGUAGES = ["en", "ar", "mixed"]
FIELDS = ["intent", "order_id", "sentiment", "urgency", "language"]

SYSTEM_PROMPT = f"""You extract structured data from customer support messages written in English, Arabic (any dialect), Franco-Arabic, or a mix.
Reply with one JSON object only, with exactly these keys:
- "intent": one of {INTENTS}
- "order_id": the order number as a string with Western digits (e.g. "4521" or "ORD-88213"), or null if none is given
- "sentiment": one of {SENTIMENTS}
- "urgency": one of {URGENCIES}
- "language": "en", "ar", or "mixed" (mixed = Franco-Arabic or Arabic and English together)"""


def build_messages(text: str, answer: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    return messages


def parse_output(text: str) -> dict | None:
    """Pull the first JSON object out of the model's reply. Returns None if invalid."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def score(preds: list[dict | None], golds: list[dict]) -> dict:
    """Per-field accuracy, valid-JSON rate and exact match (all fields correct)."""
    n = len(golds)
    result = {"valid_json": sum(p is not None for p in preds) / n}
    for field in FIELDS:
        result[field] = sum(p is not None and p.get(field) == g[field] for p, g in zip(preds, golds)) / n
    result["exact_match"] = sum(
        p is not None and all(p.get(f) == g[f] for f in FIELDS) for p, g in zip(preds, golds)
    ) / n
    return result


def dtype_kwargs(dtype) -> dict:
    """transformers 4.56 renamed `torch_dtype` to `dtype`; support both."""
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
