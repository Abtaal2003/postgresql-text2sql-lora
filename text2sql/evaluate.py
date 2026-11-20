"""Score prediction files against a real PostgreSQL server.

Metrics, for each prediction file:
- exact match: same query text after lowercasing and collapsing whitespace
- executes: the predicted query runs on the schema without an error
- execution match: on tables filled with synthetic rows, the predicted query
  returns the same rows as the gold query (order only matters if the gold
  query has ORDER BY)

It also counts which Postgres errors the failing queries hit.

Usage:
    python -m text2sql.evaluate outputs/preds_base.jsonl outputs/preds_lora.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from text2sql import pg


def normalise(sql: str) -> str:
    return " ".join(sql.lower().replace(";", " ").split())


def score_file(conn, path: Path, ignore_case: bool = False) -> dict:
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    exact = executes = matches = gold_non_empty = 0
    errors = Counter()
    failures = []
    for i, r in enumerate(rows):
        gold, pred = r["gold"], r["pred"]
        if ignore_case:  # identifiers in this dataset are lowercase, so this only changes values
            gold, pred = gold.lower(), pred.lower()
        exact += normalise(gold) == normalise(pred)

        # Only run read queries. Something like COMMIT or SET from a confused model
        # would interfere with the rollback-only transaction the checks rely on.
        if not pred.lower().lstrip("( ").startswith(("select", "with")):
            status, error = "query_error", "NotASelect"
        else:
            status, error, _ = pg.run(conn, r["context"], pred)
        if status != "ok":
            errors[error] += 1
            if len(failures) < 15:
                failures.append({"question": r["question"], "gold": gold, "pred": pred, "error": error})
            continue
        executes += 1

        # same synthetic rows for both queries (same seed, same literal pool)
        _, _, gold_rows = pg.run(conn, r["context"], gold, fill_with=[gold, pred], seed=i)
        _, _, pred_rows = pg.run(conn, r["context"], pred, fill_with=[gold, pred], seed=i)
        gold_non_empty += bool(gold_rows)
        if gold_rows is not None and pred_rows is not None:
            matches += pg.same_result(gold_rows, pred_rows, ordered="order by" in gold.lower())

    n = len(rows)
    return {
        "file": path.name,
        "n": n,
        "exact_match": round(100 * exact / n, 1),
        "executes": round(100 * executes / n, 1),
        "execution_match": round(100 * matches / n, 1),
        "gold_non_empty_pct": round(100 * gold_non_empty / max(executes, 1), 1),
        "errors": dict(errors.most_common()),
        "failure_examples": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preds", nargs="+", type=Path)
    parser.add_argument("--dsn", default=pg.DEFAULT_DSN)
    parser.add_argument("--out", default="results", type=Path)
    parser.add_argument("--labels", nargs="+", help="display names, one per predictions file")
    parser.add_argument("--stats", type=Path, help="data/stats.json, to add dataset numbers to the summary")
    parser.add_argument("--update-readme", type=Path, help="replace the results block in this README")
    parser.add_argument("--ignore-case", action="store_true", help="lowercase both queries first, to see how much of a gap is only letter case")
    args = parser.parse_args()

    conn = pg.connect(args.dsn)
    results = [score_file(conn, p, args.ignore_case) for p in args.preds]
    labels = args.labels or [p.name for p in args.preds]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))

    lines = [
        "| Model | Test questions | Exact match | Executes on Postgres | Execution match |",
        "|:--|--:|--:|--:|--:|",
    ]
    for label, r in zip(labels, results):
        lines.append(f"| {label} | {r['n']} | {r['exact_match']}% | {r['executes']}% | {r['execution_match']}% |")
    lines += ["", "Most common Postgres errors in the predictions:", ""]
    for label, r in zip(labels, results):
        top = ", ".join(f"`{k}` ({v})" for k, v in list(r["errors"].items())[:5]) or "none"
        lines.append(f"- {label}: {top}")
    if args.stats and args.stats.exists():
        s = json.loads(args.stats.read_text())
        lines += [
            "",
            f"Dataset: {s['kept']:,} of {s['source_examples']:,} examples ({s['kept_pct']}%) have a gold query that runs on Postgres. "
            f"{s['gold_queries_rewritten_for_postgres']:,} gold queries needed their quoting rewritten. "
            f"Splits: {s['splits']['train']:,} train / {s['splits']['val']:,} val / {s['splits']['test']:,} test.",
        ]
    summary = "\n".join(lines)
    (args.out / "results.md").write_text(summary + "\n")
    print(summary)

    if args.update_readme:
        text = args.update_readme.read_text()
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            after = rest.split(end, 1)[1]
            args.update_readme.write_text(f"{before}{start}\n{summary}\n{end}{after}")
            print(f"updated {args.update_readme}")


if __name__ == "__main__":
    main()
