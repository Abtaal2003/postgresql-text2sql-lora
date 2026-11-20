# Postgres Text-to-SQL with LoRA

Fine-tuning a small open LLM (Qwen2.5-Coder-0.5B-Instruct) with LoRA to turn plain English questions into **PostgreSQL** queries, and evaluating it by running the queries on a real Postgres server instead of just comparing strings.

## Why this project

Most open text-to-SQL data comes from WikiSQL and Spider, which were written for SQLite. I wanted a model for Postgres, and found that a large share of the gold answers in the popular `b-mc2/sql-create-context` dataset do not run on Postgres at all:

- **Double-quoted strings.** SQLite accepts `WHERE team = "boston celtics"`. In Postgres double quotes mean a column name, so the query fails with `column "boston celtics" does not exist`.
- **Type strictness.** Every WikiSQL column is `VARCHAR`, so `WHERE pick = 7` or `SUM(attendance)` fail on Postgres, which does not compare or add text and numbers implicitly.
- **Reserved words** like `order` used as column names.

So the pipeline first converts the dataset to Postgres, then keeps only the examples whose gold query actually executes on Postgres 16. Training on queries that would fail in production would teach the model the wrong dialect.

## Pipeline

```
sql-create-context (HF)
        │  prepare.py: fix quoting, run every gold query on Postgres, keep what executes,
        │              split by schema so test tables are never seen in training
        ▼
data/train.jsonl, val.jsonl, test.jsonl, stats.json
        │  train.py (Kaggle T4): LoRA on all linear layers, loss on the SQL only
        ▼
LoRA adapter (about 35 MB)
        │  generate.py: base model vs base + adapter, same prompt, greedy decoding
        ▼
outputs/preds_base.jsonl, preds_lora.jsonl
        │  evaluate.py: run on Postgres, compare result sets
        ▼
results/results.md, metrics.json
```

## Results

<!-- results:start -->
| Model | Test questions | Exact match | Executes on Postgres | Execution match |
|:--|--:|--:|--:|--:|
| Qwen2.5-Coder-0.5B-Instruct (base) | 1000 | 47.8% | 88.9% | 37.9% |
| + LoRA fine-tune | 1000 | 88.1% | 99.5% | 88.3% |

Most common Postgres errors in the predictions:

- Qwen2.5-Coder-0.5B-Instruct (base): `UndefinedColumn` (58), `UndefinedFunction` (30), `UndefinedTable` (9), `SyntaxError` (8), `GroupingError` (5)
- + LoRA fine-tune: `UndefinedColumn` (3), `InvalidTextRepresentation` (2)

Dataset: 57,688 of 78,577 examples (73.42%) have a gold query that runs on Postgres. 62,959 gold queries needed their quoting rewritten. Splits: 56,188 train / 500 val / 1,000 test.
<!-- results:end -->

Both models get the same prompt, the same 1,000 held-out questions (on schemas never seen in training) and greedy decoding.

### What the numbers mean

Part of the gain is real Postgres skill: the base model's queries failed on Postgres 111 times out of 1,000 (mostly made-up columns and comparing text columns to numbers, e.g. `WHERE frequency = 101.1` on a `VARCHAR`), while the fine-tuned model's failed 5 times.

Part of it is the model learning this dataset's conventions. WikiSQL stores most values in lowercase, so the gold answer is `location = 'hague'` even when the question says "Hague", and it prefers `COUNT(column)` over `COUNT(*)`. To separate the two effects I re-ran the execution check with both queries lowercased (`evaluate.py --ignore-case`):

| Model | Execution match | Case-insensitive execution match |
|:--|--:|--:|
| Base | 37.9% | 59.3% |
| + LoRA | 88.3% | 90.7% |

So about 21 points of the base model's misses are only about letter case, and the fine-tuned model is still 31 points ahead once casing is ignored. The gold answers also contain some noise (for example a "how many" question whose gold query selects a pick number), which puts a ceiling below 100% on any model. Full numbers, error counts and failure examples are in `results/metrics.json`; which examples were dropped from the dataset and why is in `data/stats.json`.

### Metrics

- **Exact match**: same query text after lowercasing and collapsing whitespace. Strict: `COUNT(*)` vs `COUNT(id)` counts as wrong.
- **Executes on Postgres**: the predicted query runs on the schema without an error (catches made-up columns, type errors, SQLite-only syntax).
- **Execution match**: both queries are run on the same tables filled with synthetic rows, and must return the same rows. Rows are seeded mostly from the literals in the two queries, so `WHERE` clauses actually match something instead of every query "agreeing" on an empty result. Order only matters when the gold query has `ORDER BY`.

Failing predictions are grouped by Postgres error class (`UndefinedColumn`, `UndefinedFunction`, `SyntaxError`, ...), which shows *how* each model fails, not just how often.

## Use the model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.prompts import build_messages, extract_sql

base = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), "outputs/lora")

schema = "CREATE TABLE orders (id INTEGER, customer VARCHAR, total INTEGER)"
messages = build_messages(schema, "Which customers spent more than 100 in total?")
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(extract_sql(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
```

## How to run

### Everything in one go (Kaggle)

`notebooks/kaggle_run_all.ipynb` reproduces the whole project on a free Kaggle T4: it installs Postgres, builds the dataset, trains, generates, evaluates, writes the results into this README and zips the finished repo. Import it on Kaggle, set **Accelerator: GPU T4 x2** (it uses one of them) and **Internet: on**, then **Save Version > Save & Run All**. The notebook is generated from the source files by `scripts/build_notebook.py`.

### Step by step on your own machine

#### 1. Set up (no GPU needed)

```bash
uv sync --python 3.12 --extra dev
docker compose up -d          # Postgres 16 on localhost:5432
```

#### 2. Build the Postgres dataset

```bash
uv run python -m text2sql.prepare --out data
uv run pytest                 # unit tests; set PG_DSN to also run the Postgres test
```

#### 3. Train and generate (CUDA GPU)

```bash
uv sync --python 3.12 --extra train
uv run python -m text2sql.train --n-train 20000 --out outputs/lora
uv run python -m text2sql.generate --out outputs/preds_base.jsonl
uv run python -m text2sql.generate --adapter outputs/lora --out outputs/preds_lora.jsonl
```

#### 4. Evaluate

```bash
uv run python -m text2sql.evaluate outputs/preds_base.jsonl outputs/preds_lora.jsonl \
    --labels "Qwen2.5-Coder-0.5B-Instruct (base)" "+ LoRA fine-tune" --stats data/stats.json --update-readme README.md
```

## Design choices

- **Qwen2.5-Coder-0.5B-Instruct**: small enough to train on a free T4 in one sitting (1,250 steps took 16.5 minutes), and already code-aware, so the experiment measures what LoRA adds on top of a reasonable baseline.
- **LoRA, rank 16 on all linear layers**: trains under 2% of the weights (about 9M of 494M). The adapter is about 35 MB (vs about 1 GB for the full model), so it is easy to share and swap.
- **Completion-only loss**: the data is passed to TRL as prompt/completion pairs, so the model is trained on the SQL it should write, not on reproducing the schema.
- **fp32 weights + fp16 autocast on T4**: T4s do not support bf16, so the base weights stay in fp32 and fp16 autocast does the speed-up. On newer GPUs the script switches to bf16 automatically.
- **Split by schema, not by row**: WikiSQL has many questions per table. A random split would put the same table in train and test and inflate the scores.
- **Prediction cleanup is the same for both models**: the base model likes to add ```` ```sql ```` fences and explanations, so `extract_sql` takes the first statement for every model. Otherwise the comparison would mostly measure formatting.

## Limitations

- The schemas are tiny (usually one table) and every WikiSQL column is `VARCHAR`, which is far from a real production database.
- Execution match on synthetic data can still call two different queries equal if the data happens not to separate them. It is a much better signal than string match, but not proof.
- Keeping only gold queries that run on Postgres removes the examples where SQLite and Postgres disagree, which are exactly the hard ones. A next step would be to rewrite them (for example casting `VARCHAR` columns before numeric comparisons) instead of dropping them.

## Credits

- Dataset: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (CC BY 4.0), built from WikiSQL and Spider.
- Base model: [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) (Apache 2.0).
- Training: Hugging Face TRL and PEFT.

Code is MIT licensed (see `LICENSE`). The adapter is a derivative of Qwen2.5-Coder-0.5B-Instruct (Apache 2.0).
