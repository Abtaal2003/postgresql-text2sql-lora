"""Generate SQL for the test set, with or without the LoRA adapter.

Usage:
    python -m text2sql.generate --out outputs/preds_base.jsonl
    python -m text2sql.generate --adapter outputs/lora --out outputs/preds_lora.jsonl
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.prompts import build_messages, extract_sql
from text2sql.train import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", help="path or Hub id of a LoRA adapter")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # LoRA does not change the tokenizer, so always take it from the base model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # decoder-only models need left padding for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    with open(args.test) as f:
        rows = [json.loads(line) for line in f][: args.n]

    with open(args.out, "w") as out:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(build_messages(r["context"], r["question"]), tokenize=False, add_generation_prompt=True)
                for r in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            replies = tokenizer.batch_decode(generated[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for r, reply in zip(batch, replies):
                record = {"id": r["id"], "question": r["question"], "context": r["context"], "gold": r["sql"], "raw": reply, "pred": extract_sql(reply)}
                out.write(json.dumps(record) + "\n")
            print(f"{start + len(batch)}/{len(rows)}")


if __name__ == "__main__":
    main()
