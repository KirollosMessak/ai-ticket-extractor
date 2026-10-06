"""Gradio demo for Hugging Face Spaces (runs on the free CPU tier).

Set the MODEL_ID variable in the Space settings to your fine-tuned model repo.
"""

import json
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import build_messages, dtype_kwargs, parse_output

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **dtype_kwargs(torch.float32))
model.eval()


def extract(message: str) -> str:
    if not message.strip():
        return "{}"
    prompt = tokenizer.apply_chat_template(build_messages(message), tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=120, do_sample=False)
    raw = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    parsed = parse_output(raw)
    return json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else raw


demo = gr.Interface(
    fn=extract,
    inputs=gr.Textbox(lines=3, label="Customer message (English, Arabic, or Franco-Arabic)"),
    outputs=gr.Code(language="json", label="Extracted ticket"),
    title="🎫 AI Support Ticket Extractor (EN + AR)",
    description=f"A fine-tuned {MODEL_ID.split('/')[-1]} that turns messy customer messages into structured tickets.",
    examples=[
        ["My order #4521 still hasn't arrived after 10 days, this is unacceptable!"],
        ["الأوردر رقم ١٢٣٤٥ لسه موصلش وأنا محتاجه بكرة ضروري"],
        ["3ayez a3mel return lel jacket, el size kbeer. order 88120"],
        ["أبي ألغي الطلب ORD-60018 لو سمحتوا"],
        ["Thanks for the great service! Do you ship to Alexandria?"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
