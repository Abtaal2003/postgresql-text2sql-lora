"""Build the Postgres version of sql-create-context.

1. Download b-mc2/sql-create-context from Hugging Face.
2. Convert each gold query to Postgres (see postgresify.py).
3. Run it on a real Postgres server against its schema and keep it only if it executes.
4. Split by schema (the exact CREATE TABLE text), so no schema appears in more than one split.

Usage:
    python -m text2sql.prepare --out data
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from text2sql import pg
from text2sql.postgresify import to_postgres


def load_examples(source: str | None) -> list[dict]:
    if source:  # a local .jsonl, handy for tests
        with open(source) as f:
            return [json.loads(line) for line in f]
    from datasets import load_dataset

    return list(load_dataset("b-mc2/sql-create-context", split="train"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="local .jsonl instead of the Hugging Face dataset")
    parser.add_argument("--out", default="data")
    parser.add_argument("--dsn", default=pg.DEFAULT_DSN)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_examples(args.source)
    conn = pg.connect(args.dsn)

    kept, seen = [], set()
    dropped = Counter()
    quote_fixes = 0
    for i, ex in enumerate(examples):
        key = (ex["context"], ex["question"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        sql = to_postgres(ex["answer"], ex["context"])
        quote_fixes += sql != ex["answer"].strip().rstrip(";")
        status, error, _ = pg.run(conn, ex["context"], sql)
        if status == "ok":
            kept.append({"question": ex["question"], "context": ex["context"], "sql": sql, "original_sql": ex["answer"]})
        else:
            dropped[f"{status}:{error}"] += 1

        if (i + 1) % 5000 == 0:
            print(f"{i + 1}/{len(examples)} checked, {len(kept)} kept")

    # Group by schema text so the same CREATE TABLE never appears in two splits.
    groups = defaultdict(list)
    for row in kept:
        groups[row["context"]].append(row)
    contexts = sorted(groups)
    random.Random(args.seed).shuffle(contexts)

    test, val, train = [], [], []
    for context in contexts:
        if len(test) < args.n_test:
            test += groups[context]
        elif len(val) < args.n_val:
            val += groups[context]
        else:
            train += groups[context]
    random.Random(args.seed).shuffle(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        for j, row in enumerate(rows):
            row["id"] = f"{name}-{j}"
        write_jsonl(out / f"{name}.jsonl", rows)

    stats = {
        "source_examples": len(examples),
        "kept": len(kept),
        "kept_pct": round(100 * len(kept) / max(len(examples), 1), 2),
        "gold_queries_rewritten_for_postgres": quote_fixes,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "dropped": dict(dropped.most_common()),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
