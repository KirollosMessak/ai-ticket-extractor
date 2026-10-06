import json
from pathlib import Path

from common import FIELDS, INTENTS, LANGUAGES, SENTIMENTS, URGENCIES, parse_output, score

DATA = Path(__file__).parent.parent / "data"


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l]


def test_parse_output_handles_extra_text_and_garbage():
    assert parse_output('Sure! {"intent": "cancel_order"} done') == {"intent": "cancel_order"}
    assert parse_output("no json here") is None
    assert parse_output("{broken") is None


def test_score_counts_fields_and_exact_match():
    gold = {"intent": "cancel_order", "order_id": "1", "sentiment": "neutral", "urgency": "low", "language": "en"}
    wrong_intent = {**gold, "intent": "refund_request"}
    m = score([gold, wrong_intent, None], [gold, gold, gold])
    assert m["valid_json"] == 2 / 3
    assert m["intent"] == 1 / 3
    assert m["order_id"] == 2 / 3
    assert m["exact_match"] == 1 / 3


def test_all_labels_use_allowed_values():
    allowed = {"intent": INTENTS, "sentiment": SENTIMENTS, "urgency": URGENCIES, "language": LANGUAGES}
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for row in load(name):
            assert set(row["label"]) == set(FIELDS)
            for field, values in allowed.items():
                assert row["label"][field] in values, (name, row)
            oid = row["label"]["order_id"]
            assert oid is None or oid.isascii(), "order IDs must use Western digits"


def test_order_ids_appear_in_text():
    to_arabic = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    for name in ("train.jsonl", "test.jsonl"):
        for row in load(name):
            oid = row["label"]["order_id"]
            if oid:
                digits = oid.removeprefix("ORD-")
                assert digits in row["text"] or digits.translate(to_arabic) in row["text"], row


def test_no_test_examples_leak_into_training():
    test = {r["text"] for r in load("test.jsonl")}
    train = {r["text"] for r in load("train.jsonl") + load("val.jsonl")}
    assert not test & train
