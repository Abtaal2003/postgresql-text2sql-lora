"""Prompt format shared by training, generation and evaluation."""

import re

from text2sql.postgresify import split_schema

SYSTEM_PROMPT = (
    "You are a PostgreSQL expert. Given a database schema and a question, "
    "write one PostgreSQL query that answers the question. Reply with the SQL only."
)


def build_messages(context: str, question: str) -> list[dict]:
    schema = ";\n".join(split_schema(context)) + ";"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]


def extract_sql(text: str) -> str:
    """Pull the query out of a model reply.

    The base model often wraps the answer in ```sql fences or adds an
    explanation, so take the fenced block if there is one, then keep only
    the first statement.
    """
    fenced = re.search(r"```(?:sql|postgresql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = fenced.group(1) if fenced else text
    sql = sql.strip().split("\n\n")[0]
    sql = sql.split(";")[0]
    return " ".join(sql.split())
