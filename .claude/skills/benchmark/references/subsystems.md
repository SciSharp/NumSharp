# The five matrix subsystems (and how to add one)

Beyond the op/dtype/N matrix, `run_benchmark.py` appends five subsystems that fill axes the op matrix can't
express. Each has its OWN result model and is **appended, not merged** into the op matrix. They live in
`benchmark/{nditer,layout,operand,cast,fusion}/`.

| Subsystem | Dir | What it adds the op matrix lacks | Result model |
|-----------|-----|----------------------------------|--------------|
| **nditer** | `benchmark/nditer/` | iterator machinery (construction, traversal, reductions, selection, dtypes, pathologies) | aspect × cache-tier |
| **layout** | `benchmark/layout/` | reduction / copy / elementwise across the 8 memory layouts (C/F/T/sliced/strided/negrow/negcol/bcast) — the op matrix is C-contiguous only | op × layout × dtype |
| **operand** | `benchmark/operand/` | 1-D (contig/strided/reversed), scalar operand, mixed operand layouts (C+F, C+T), binary broadcast | case × dtype |
| **cast** | `benchmark/cast/` | full `astype` src→dst × 8 layouts at 1M — no op-matrix coverage at all | 15×15 per-layout matrices |
| **fusion** | `benchmark/fusion/` | `np.evaluate` fused vs unfused np.* chains (+ NumPy context) | fixed-expression report |

## The shared shape (every subsystem is the same three files)

1. **`<name>_bench.cs`** — a NumSharp file-based app fed on stdin via `dotnet run -c Release -`. Emits rows with the
   subsystem's keys. Its author-absolute `#:project` path is rewritten to the running checkout by the driver.
2. **`<name>_bench.py`** — the NumPy twin emitting IDENTICAL keys.
3. **`<name>_sheet.py`** — merges the two and renders `<name>_results.md` (+ `.tsv`), in the **NPY/NS** convention.

The build/run/parse plumbing is shared in `benchmark/scripts/bench_common.py`. `run_benchmark.py` calls each
`*_sheet.py` and appends its `*_results.md` as a report section; `--skip-<name>` opts out.

## Adding a new op to an EXISTING subsystem

Most common. E.g. to cover a new op across memory layouts, add it to `benchmark/layout/{reduce,copy,elementwise}_
layout_bench.cs` AND the `.py` twin with the same key (op name + layout + dtype), then rerun
`python benchmark/layout/layout_sheet.py` (or `run_benchmark.py --suites '' ` with only that subsystem, or just the
sheet script). Keep the two sides' keys byte-identical or they won't merge.

## Adding a WHOLE new subsystem

Rare. Mirror an existing one (the `cast` subsystem is the cleanest template):
1. Create `benchmark/<name>/<name>_bench.cs`, `<name>_bench.py`, `<name>_sheet.py` following the `bench_common.py`
   contract (same row keys both sides; NPY/NS in the sheet).
2. Wire it into `run_benchmark.py` (add the append step + a `--skip-<name>` flag) next to the other four.
3. Ensure the sheet writes `benchmark/<name>/<name>_results.md` (+ `.tsv`) — those are what the history snapshot
   collects.

## Convention & pitfalls (same as the op matrix)

- **NPY/NS everywhere** (ratio = NumPy_ms / NumSharp_ms, `>1` = NumSharp faster). The sheet renders it; don't invert.
- **`-c Release`** — the `.cs` benches run through `dotnet run -c Release -`; Debug would taint hand-written kernels
  ~2× (see `run-and-report.md`).
- **Committed output** is the subsystem's `*_results.md`/`.tsv`, collected into `benchmark/history/<date>_<sha>/`;
  the raw run under `results/<ts>/` is gitignored.

The full subsystem descriptions are in `benchmark/CLAUDE.md` → "Running Benchmarks → Official run" and each
subsystem's own `README.md` (e.g. `benchmark/nditer/README.md`).
