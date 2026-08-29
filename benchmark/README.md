# `benchmark` — NumSharp-vs-NumPy benchmark report

The official performance comparison: the op × dtype × N matrix (C# BenchmarkDotNet vs a warm
NumPy 2.4.2 process, unified Managed + OpenBLAS profiles) plus the five appended subsystem sheets
(nditer, layout, operand, cast, fusion). Convention is **NPY/NS** (ratio = NumPy_ms ÷ NumSharp_ms,
`>1` = NumSharp faster).

A snapshot is the same set `benchmark/run_benchmark.py` writes into `benchmark/history/<date>_<sha>/`.

## Files in a snapshot

| File | What |
|------|------|
| `benchmark-report.json` | Merged machine-readable op-matrix (**the Benchmarks dashboard's Function Explorer fetches this**) |
| `benchmark-report.managed.json` / `.openblas.json` | The two backend profiles, same schema |
| `benchmark-report.csv` | Spreadsheet form of the merged matrix |
| `benchmark-report.md` | The canonical text report (op-matrix + appended subsystem sections) |
| `MANIFEST.md` | Provenance, environment, methodology, headline geomeans |
| `cast_results.{md,tsv}` · `fusion_results.md` · `layout_results.{md,tsv}` · `nditer_results.{md,tsv}` · `operand_results.{md,tsv}` | The appended subsystem sheets |
| `openblas_results.{md,tsv,managed.json,openblas.json}` | OpenBLAS / LAPACK backend routes |
| `numpy-results.json` | Raw NumPy timings (merge input) |
| `cards/` | The two NDIter summary PNG cards (`ops.png`, `cat.png`) |

## Consumed by the docs site

- `benchmarks-dashboard.md` fetches `benchmark-report.json` (Function Explorer heatmaps).
- The toc "Raw Reports" node links `MANIFEST.md`, `benchmark-report.md`, and the subsystem `*_results.md`.

## Regenerate

```bash
python benchmark/run_benchmark.py          # full official run; writes benchmark/history/<date>_<sha>/ + latest
```

See `benchmark/CLAUDE.md` for options (`--suites`, `--skip-build`, subsystem toggles) and methodology.

## Publish to this branch

```bash
python tools/dashboard_data/publish.py --type benchmark \
    --from benchmark/history/latest \
    --branch-worktree ../NumSharp-master-code-data \
    --commit
```

(CI does this automatically in `.github/workflows/benchmark.yml` after a benchmark run.)
