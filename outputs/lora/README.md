---
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
datasets:
- b-mc2/sql-create-context
tags:
- lora
- sft
- text-to-sql
- postgresql
---

# Qwen2.5-Coder-0.5B-Instruct: PostgreSQL text-to-SQL LoRA

A LoRA adapter (rank 16, all linear layers, about 8.8M trainable parameters) for [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). It turns a question plus a `CREATE TABLE` schema into a single PostgreSQL query.

Trained with supervised fine-tuning (TRL) for one epoch on 20,000 examples from a Postgres-validated version of [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context): every gold query was converted to Postgres syntax and kept only if it ran on Postgres 16. One epoch took 16.5 minutes on a single Kaggle T4.

## Results

1,000 test questions on schemas not seen in training, scored by running the queries on Postgres:

| Model | Exact match | Executes on Postgres | Execution match |
|:--|--:|--:|--:|
| Base model | 47.8% | 88.9% | 37.9% |
| + this adapter | 88.1% | 99.5% | 88.3% |

Part of the gain comes from learning the dataset's conventions (for example lowercase values). With letter case ignored, execution match is 59.3% for the base model and 90.7% with the adapter. The schemas are small and mostly single-table, so this is not a measure of performance on real production databases.

Code, data pipeline and evaluation: the repository this folder belongs to (`text2sql/`). The prompt format is in `text2sql/prompts.py`; use the base model's tokenizer.
