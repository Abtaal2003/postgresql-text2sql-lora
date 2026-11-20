"""LoRA fine-tuning with TRL's SFTTrainer.

The data is given to TRL in "prompt / completion" form, so the loss is only
computed on the SQL the model should write, not on the schema and question.

Usage (Kaggle T4 or any CUDA GPU):
    python -m text2sql.train --n-train 20000 --out outputs/lora
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from text2sql.prompts import build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def to_prompt_completion(row):
    return {
        "prompt": build_messages(row["context"], row["question"]),
        "completion": [{"role": "assistant", "content": row["sql"]}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--out", default="outputs/lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="cap steps, for quick tests")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_dataset("json", data_files={"train": args.train, "val": args.val})
    train = data["train"].select(range(min(args.n_train, len(data["train"]))))
    val = data["val"]
    train = train.map(to_prompt_completion, remove_columns=train.column_names)
    val = val.map(to_prompt_completion, remove_columns=val.column_names)

    # T4 GPUs (Kaggle, Colab) have no native bf16, so keep the base weights in fp32
    # and let fp16 autocast do the speed-up. Ampere (compute capability 8.0) and newer use bf16.
    # torch.cuda.is_bf16_supported() is not enough here: it also returns True for emulated bf16
    # on a T4, and TRL's fused loss kernel then fails there.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        max_length=args.max_length,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": dtype},
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(model=args.model, args=config, train_dataset=train, eval_dataset=val, peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()
    print("final eval:", trainer.evaluate())

    trainer.save_model(args.out)  # saves only the LoRA adapter (about 35 MB)
    trainer.processing_class.save_pretrained(args.out)


if __name__ == "__main__":
    main()
