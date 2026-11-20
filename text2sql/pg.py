"""Running queries against a real PostgreSQL server.

Every check happens inside a transaction that is always rolled back, so the
database stays empty and the same table names can be reused for every example.
"""

import os
import random
import re
from decimal import Decimal

import psycopg

from text2sql.postgresify import NUMERIC_TYPES, TEXT_TYPES, parse_tables, split_schema

DEFAULT_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/text2sql")
MAX_ROWS = 1000


def connect(dsn: str = DEFAULT_DSN) -> psycopg.Connection:
    return psycopg.connect(dsn)


def literals(sql: str) -> tuple[list[str], list[float]]:
    """String and number literals that appear in a query."""
    strings = [s.replace("''", "'") for s in re.findall(r"'((?:[^']|'')*)'", sql)]
    without_strings = re.sub(r"'(?:[^']|'')*'", "", sql)
    numbers = [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", without_strings)]
    return strings, numbers


def fill_tables(cur, context: str, queries: list[str], n_rows: int, seed: int) -> None:
    """Insert synthetic rows so that queries return something to compare.

    Values are drawn mostly from the literals used in the queries themselves,
    which makes WHERE clauses actually match some rows. Empty tables would make
    almost any two queries "agree" on an empty result.
    """
    rng = random.Random(seed)
    strings, numbers = [], []
    for q in queries:
        s, n = literals(q)
        strings += s
        numbers += n
    strings = strings + ["alpha", "beta", "gamma", "1", "2", "10"]
    numbers = numbers + [0, 1, 2, 5, 10, 100]

    for table, columns in parse_tables(context).items():
        cols = ", ".join(f'"{c}"' for c, _ in columns)
        marks = ", ".join(["%s"] * len(columns))
        rows = []
        for _ in range(n_rows):
            row = []
            for _, col_type in columns:
                if col_type.startswith(TEXT_TYPES):
                    row.append(rng.choice(strings))
                elif col_type.startswith(NUMERIC_TYPES):
                    value = rng.choice(numbers) + rng.choice([0, 0, 0, rng.randint(-3, 3)])
                    row.append(int(value) if col_type.startswith("int") else value)
                else:
                    row.append(None)
            rows.append(row)
        cur.executemany(f'INSERT INTO "{table}" ({cols}) VALUES ({marks})', rows)


def _normalise(value):
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return round(float(value), 4)
    return value


def run(conn, context: str, sql: str, fill_with: list[str] | None = None, seed: int = 0):
    """Create the schema, optionally add rows, run `sql`, roll everything back.

    Returns (status, error_name, rows), where status is one of
    "ok", "query_error" or "schema_error".
    """
    with conn.transaction(force_rollback=True):
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '3s'")
            try:
                with conn.transaction():
                    for stmt in split_schema(context):
                        cur.execute(stmt)
                    if fill_with is not None:
                        fill_tables(cur, context, fill_with, n_rows=20, seed=seed)
            except psycopg.Error as e:
                return "schema_error", type(e).__name__, None
            try:
                with conn.transaction():
                    cur.execute(sql)
                    rows = cur.fetchmany(MAX_ROWS) if cur.description else []
            except psycopg.Error as e:
                return "query_error", type(e).__name__, None
    rows = [tuple(_normalise(v) for v in row) for row in rows]
    return "ok", None, rows


def same_result(gold_rows, pred_rows, ordered: bool) -> bool:
    if ordered:
        return gold_rows == pred_rows
    key = repr
    return sorted(gold_rows, key=key) == sorted(pred_rows, key=key)
