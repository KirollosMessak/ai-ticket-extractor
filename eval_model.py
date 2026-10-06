"""Score a model on the hand-written test set.

Usage:
    python eval_model.py --model Qwen/Qwen2.5-0.5B-Instruct --name base
    python eval_model.py --model outputs/merged --name finetuned
Results are saved to results/<name>.json (metrics + every prediction).
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import FIELDS, build_messages, dtype_kwargs, parse_output, score


def predict(model, tokenizer, texts: list[str], batch_size: int = 16) -> list[str]:
    tokenizer.padding_side = "left"  # required for batched generation
    outputs = []
    for i in range(0, len(texts), batch_size):
        prompts = [
            tokenizer.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True)
            for t in texts[i : i + batch_size]
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l]
    rows = rows[: args.limit] if args.limit else rows

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **dtype_kwargs(torch.float16 if device == "cuda" else torch.float32)
    ).to(device)
    model.eval()

    raw = predict(model, tokenizer, [r["text"] for r in rows])
    preds = [parse_output(r) for r in raw]
    golds = [r["label"] for r in rows]
    metrics = score(preds, golds)

    by_lang = {}
    for lang in ("en", "ar", "mixed"):
        idx = [i for i, g in enumerate(golds) if g["language"] == lang]
        if idx:
            by_lang[lang] = score([preds[i] for i in idx], [golds[i] for i in idx])["exact_match"]

    Path("results").mkdir(exist_ok=True)
    out = {
        "model": args.model,
        "metrics": metrics,
        "exact_match_by_language": by_lang,
        "predictions": [
            {"text": r["text"], "gold": g, "raw_output": o, "parsed": p}
            for r, g, o, p in zip(rows, golds, raw, preds)
        ],
    }
    Path(f"results/{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{args.name} ({args.model}) on {len(rows)} test examples")
    for k in ["valid_json", *FIELDS, "exact_match"]:
        print(f"  {k:<12} {metrics[k]:.0%}")
    print("  exact match by language:", {k: f"{v:.0%}" for k, v in by_lang.items()})


if __name__ == "__main__":
    main()
