"""Build notebooks/kaggle_run_all.ipynb from the source files in this repo.

The notebook writes every project file with %%writefile, so it runs on
Kaggle without cloning anything. Rebuild it after changing any source file:

    python scripts/build_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBEDDED = [
    "pyproject.toml",
    "docker-compose.yml",
    ".gitignore",
    "README.md",
    "text2sql/__init__.py",
    "text2sql/postgresify.py",
    "text2sql/pg.py",
    "text2sql/prompts.py",
    "text2sql/prepare.py",
    "text2sql/train.py",
    "text2sql/generate.py",
    "text2sql/evaluate.py",
    "tests/test_text2sql.py",
    "tests/fixtures/sample.jsonl",
    "scripts/build_notebook.py",
]


PG_SETUP = r'''import subprocess

import psycopg


def pg_ready():
    try:
        psycopg.connect(os.environ["PG_DSN"], connect_timeout=3).close()
        return True
    except Exception:
        return False


if not pg_ready():
    for cmd in [
        "apt-get -qq update",
        "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y postgresql > /dev/null",
        "service postgresql start",
        """su postgres -c "psql -q -c \\"ALTER USER postgres PASSWORD 'postgres'\\"" """,
        """su postgres -c "createdb text2sql" """,
    ]:
        subprocess.run(cmd, shell=True, check=True)
print("Postgres ready:", pg_ready())'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def build() -> dict:
    cells = [
        md(
            "# Postgres Text-to-SQL with LoRA: full run\n\n"
            "Runs the whole project on a Kaggle T4: installs Postgres, builds the dataset, fine-tunes "
            "Qwen2.5-Coder-0.5B-Instruct with LoRA, generates SQL with the base and fine-tuned models, "
            "scores both on Postgres, writes the results into the README and zips the finished repo.\n\n"
            "Before running: **Accelerator: GPU T4 x2** (one is used) and **Internet: on**. "
            "Use **Save Version > Save & Run All** so it keeps running with the browser closed, "
            "then download `postgresql-text2sql-lora-complete.zip` from the version's Output tab."
        ),
        code(
            "# Settings\n"
            'MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"\n'
            "N_TRAIN = 20000\n"
            "N_VAL = 500\n"
            "N_TEST = 1000\n"
            "EPOCHS = 1\n"
            'SOURCE = ""  # empty: download b-mc2/sql-create-context from Hugging Face\n'
            'HF_REPO = "Abtaal2003/qwen2.5-coder-0.5b-postgres-text2sql-lora"  # pushed only if an HF_TOKEN secret exists'
        ),
        code(
            "import os\n\n"
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU keeps the Trainer simple\n'
            'os.environ["PG_DSN"] = "postgresql://postgres:postgres@localhost:5432/text2sql"\n'
            'PROJECT = os.path.abspath("postgresql-text2sql-lora")\n'
            'for d in ["text2sql", "tests/fixtures", "scripts", "notebooks"]:\n'
            "    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)\n"
            "os.chdir(PROJECT)\n"
            "print(os.getcwd())"
        ),
        md("## Project files"),
    ]
    for rel in EMBEDDED:
        content = (ROOT / rel).read_text()
        cells.append(code(f"%%writefile {rel}\n{content}"))

    cells += [
        md("## Install and start Postgres"),
        code('!pip install -q -U "psycopg[binary]>=3.2" "datasets>=3.0" "transformers>=5.0" "trl>=1.0" "peft>=0.17" "accelerate>=1.0" pytest'),
        code(PG_SETUP),
        code("!python -m pytest -q"),
        md("## 1. Build the Postgres dataset"),
        code(
            'source_arg = f"--source {SOURCE}" if SOURCE else ""\n'
            "!python -m text2sql.prepare --out data --n-val {N_VAL} --n-test {N_TEST} {source_arg}"
        ),
        md("## 2. Fine-tune with LoRA"),
        code(
            "import torch\n\n"
            'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none, this will be very slow")'
        ),
        code("!python -m text2sql.train --model {MODEL} --n-train {N_TRAIN} --epochs {EPOCHS} --out outputs/lora"),
        md("## 3. Generate SQL with the base and the fine-tuned model"),
        code(
            "!python -m text2sql.generate --model {MODEL} --n {N_TEST} --out outputs/preds_base.jsonl\n"
            "!python -m text2sql.generate --model {MODEL} --adapter outputs/lora --n {N_TEST} --out outputs/preds_lora.jsonl"
        ),
        md("## 4. Score both on Postgres"),
        code(
            'base_label = MODEL.split("/")[-1] + " (base)"\n'
            'lora_label = "+ LoRA fine-tune"\n'
            "!python -m text2sql.evaluate outputs/preds_base.jsonl outputs/preds_lora.jsonl --labels \"{base_label}\" \"{lora_label}\" --stats data/stats.json --update-readme README.md"
        ),
        code(
            "import json\n\n"
            "base = [json.loads(l) for l in open(\"outputs/preds_base.jsonl\")]\n"
            "lora = [json.loads(l) for l in open(\"outputs/preds_lora.jsonl\")]\n"
            "for b, l in list(zip(base, lora))[:5]:\n"
            '    print("Q:   ", b["question"])\n'
            '    print("gold:", b["gold"])\n'
            '    print("base:", b["pred"])\n'
            '    print("lora:", l["pred"], "\\n")'
        ),
        md("## 5. Package the finished repo"),
        code(
            "import shutil\n\n"
            "out_zip = os.path.join(os.path.dirname(PROJECT), \"postgresql-text2sql-lora-complete\")\n"
            "shutil.make_archive(out_zip, \"zip\", os.path.dirname(PROJECT), \"postgresql-text2sql-lora\")\n"
            'print(out_zip + ".zip", round(os.path.getsize(out_zip + ".zip") / 1e6, 1), "MB")'
        ),
        md("## Optional: publish the adapter on Hugging Face\n\nAdd a write token as a Kaggle secret named `HF_TOKEN` (Add-ons > Secrets). Skipped otherwise."),
        code(
            "try:\n"
            "    from kaggle_secrets import UserSecretsClient\n"
            "    from huggingface_hub import HfApi\n\n"
            '    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))\n'
            "    api.create_repo(HF_REPO, exist_ok=True)\n"
            '    api.upload_folder(folder_path="outputs/lora", repo_id=HF_REPO)\n'
            '    print(f"https://huggingface.co/{HF_REPO}")\n'
            "except Exception as e:\n"
            '    print("Skipped the Hugging Face upload:", e)'
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


if __name__ == "__main__":
    path = ROOT / "notebooks" / "kaggle_run_all.ipynb"
    path.write_text(json.dumps(build(), indent=1))
    print(f"wrote {path}")
