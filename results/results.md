| Model | Test questions | Exact match | Executes on Postgres | Execution match |
|:--|--:|--:|--:|--:|
| Qwen2.5-Coder-0.5B-Instruct (base) | 1000 | 47.8% | 88.9% | 37.9% |
| + LoRA fine-tune | 1000 | 88.1% | 99.5% | 88.3% |

Most common Postgres errors in the predictions:

- Qwen2.5-Coder-0.5B-Instruct (base): `UndefinedColumn` (58), `UndefinedFunction` (30), `UndefinedTable` (9), `SyntaxError` (8), `GroupingError` (5)
- + LoRA fine-tune: `UndefinedColumn` (3), `InvalidTextRepresentation` (2)

Dataset: 57,688 of 78,577 examples (73.42%) have a gold query that runs on Postgres. 62,959 gold queries needed their quoting rewritten. Splits: 56,188 train / 500 val / 1,000 test.
