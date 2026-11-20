"""Turn sql-create-context examples into valid PostgreSQL.

The dataset was built from WikiSQL and Spider, which were written for SQLite.
SQLite is forgiving in ways Postgres is not. The biggest one: SQLite accepts
"double quotes" around string values, while in Postgres double quotes always
mean an identifier (a table or column name). So a gold answer like

    SELECT position FROM table_name_95 WHERE team = "boston celtics"

fails on Postgres with `column "boston celtics" does not exist`.

This module rewrites those string literals to single quotes, using the schema
to tell real identifiers apart from values.
"""

import re

TEXT_TYPES = ("char", "text", "varchar")
NUMERIC_TYPES = ("int", "real", "float", "double", "decimal", "numeric", "number")


def split_schema(context: str) -> list[str]:
    """Split the context field into individual CREATE TABLE statements."""
    parts = re.split(r";\s*|,\s*(?=CREATE TABLE)", context, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def parse_tables(context: str) -> dict[str, list[tuple[str, str]]]:
    """Return {table_name: [(column_name, column_type), ...]} from the schema."""
    tables = {}
    for stmt in split_schema(context):
        m = re.match(r"CREATE TABLE\s+(\w+)\s*\((.*)\)\s*$", stmt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        columns = []
        for col_def in m.group(2).split(","):
            bits = col_def.split()
            if bits:
                col_type = bits[1].lower() if len(bits) > 1 else "varchar"
                columns.append((bits[0].lower(), col_type))
        tables[m.group(1).lower()] = columns
    return tables


def schema_names(context: str) -> set[str]:
    """All table and column names in the schema, lowercased."""
    names = set()
    for table, columns in parse_tables(context).items():
        names.add(table)
        names.update(col for col, _ in columns)
    return names


def fix_string_quotes(sql: str, names: set[str]) -> str:
    """Rewrite "double quoted" values as 'single quoted' ones.

    Walks the query character by character so that text already inside
    single quotes is left alone. A double-quoted token is kept as an
    identifier only if it matches a table or column name in the schema.
    """
    out = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            # copy a single-quoted string as-is ('' is an escaped quote)
            j = i + 1
            while j < n:
                if sql[j] == "'" and j + 1 < n and sql[j + 1] == "'":
                    j += 2
                elif sql[j] == "'":
                    break
                else:
                    j += 1
            out.append(sql[i : j + 1])
            i = j + 1
        elif ch == '"':
            j = sql.find('"', i + 1)
            if j == -1:  # unbalanced quote, give up and leave the rest
                out.append(sql[i:])
                break
            inner = sql[i + 1 : j]
            if inner.lower() in names:
                out.append(f'"{inner.lower()}"')
            else:
                out.append("'" + inner.replace("'", "''") + "'")
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def to_postgres(sql: str, context: str) -> str:
    """Full conversion of a gold answer to Postgres (currently just quotes)."""
    return fix_string_quotes(sql.strip().rstrip(";"), schema_names(context))
