# `benchmark-coverage` — Benchmark-wiring coverage ledger

Which benchmarkable NumPy-compatibility APIs actually have a wired benchmark. It audits the benchmark
suite's routes against the authoritative API surface in the `inventory` dataset
(`coverage/generated/coverage.json`) and reports measured / aliased / unmatched. This answers
"is there a benchmark for this op?", **not** "does the API exist?" (that is the `inventory` dataset).

## Files in a snapshot

| File | What |
|------|------|
| `coverage.json` | Per-API benchmark-wiring rows + summary |
| `coverage.csv` | Flat spreadsheet form |
| `summary.md` | Human-readable ledger (**linked under the toc "Raw Reports → API Coverage Ledger"**) |

## Consumed by the docs site

- The toc "Benchmarks → Raw Reports → API Coverage Ledger" links `summary.md`.

## Regenerate

```bash
python benchmark/scripts/audit_coverage.py          # writes benchmark/coverage/generated/
python benchmark/scripts/audit_coverage.py --check  # CI gate: fail if the committed ledger is stale
```

## Publish to this branch

```bash
python tools/dashboard_data/publish.py --type benchmark-coverage \
    --from benchmark/coverage/generated \
    --branch-worktree ../NumSharp-master-code-data \
    --commit
```

(CI does this automatically in `.github/workflows/benchmark.yml` after a benchmark run.)
