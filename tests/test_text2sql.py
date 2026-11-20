import os

import pytest

from text2sql.postgresify import fix_string_quotes, parse_tables, to_postgres
from text2sql.prompts import extract_sql

CONTEXT = "CREATE TABLE table_name_95 (position VARCHAR, team VARCHAR)"


def test_double_quoted_value_becomes_string_literal():
    sql = 'SELECT position FROM table_name_95 WHERE team = "boston celtics"'
    assert to_postgres(sql, CONTEXT) == "SELECT position FROM table_name_95 WHERE team = 'boston celtics'"


def test_double_quoted_column_stays_identifier():
    sql = 'SELECT "Position" FROM table_name_95 WHERE team = "x"'
    assert to_postgres(sql, CONTEXT) == "SELECT \"position\" FROM table_name_95 WHERE team = 'x'"


def test_apostrophe_inside_value_is_escaped():
    assert fix_string_quotes('WHERE singer = "o\'brien"', set()) == "WHERE singer = 'o''brien'"


def test_single_quoted_strings_are_left_alone():
    sql = "WHERE song = 'say \"hi\"'"
    assert fix_string_quotes(sql, set()) == sql


def test_parse_tables_handles_multiple_statements():
    context = "CREATE TABLE a (x VARCHAR, y INTEGER); CREATE TABLE b (z VARCHAR)"
    assert parse_tables(context) == {"a": [("x", "varchar"), ("y", "integer")], "b": [("z", "varchar")]}


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("SELECT a FROM t", "SELECT a FROM t"),
        ("```sql\nSELECT a\nFROM t;\n```\nThis query selects a.", "SELECT a FROM t"),
        ("SELECT a FROM t;\n\nExplanation: ...", "SELECT a FROM t"),
    ],
)
def test_extract_sql(reply, expected):
    assert extract_sql(reply) == expected


@pytest.mark.skipif("PG_DSN" not in os.environ, reason="needs a running Postgres (set PG_DSN)")
def test_runs_on_postgres_and_compares_results():
    from text2sql import pg

    conn = pg.connect(os.environ["PG_DSN"])
    gold = "SELECT position FROM table_name_95 WHERE team = 'boston celtics'"
    status, _, _ = pg.run(conn, CONTEXT, gold)
    assert status == "ok"

    status, error, _ = pg.run(conn, CONTEXT, "SELECT positon FROM table_name_95")
    assert (status, error) == ("query_error", "UndefinedColumn")

    no_filter = "SELECT position FROM table_name_95"
    _, _, gold_rows = pg.run(conn, CONTEXT, gold, fill_with=[gold, no_filter], seed=1)
    _, _, pred_rows = pg.run(conn, CONTEXT, no_filter, fill_with=[gold, no_filter], seed=1)
    assert not pg.same_result(gold_rows, pred_rows, ordered=False)
