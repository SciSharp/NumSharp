---
name: benchmark
description: >-
  NumSharp's NumPy-vs-NumSharp performance harness — the op/dtype/N matrix (C# BenchmarkDotNet vs a
  warm NumPy process) plus five appended subsystems (nditer, layout, operand, cast, fusion), all in
  the NPY/NS convention. Use this whenever you add a benchmark for an np.* op, wire a C# benchmark to
  its NumPy twin, run the official suite or a subset, read/interpret the ratio matrix or history
  snapshots, add a whole subsystem, or debug a suspicious measurement (the Debug-taint 2x pitfall,
  the InProcessEmit toolchain). Trigger on: "benchmark", "add a benchmark", "how fast is <op> vs
  numpy", "run_benchmark.py", "BenchmarkDotNet", "NPY/NS ratio", "perf comparison", "benchmark
  <op>", "benchmark-report", "history snapshot", "why is my timing 2x slow". Reach for it before
  quoting any NumSharp-vs-NumPy speed number.
---

# NumSharp Benchmark Harness

The exhaustive guide is **`benchmark/CLAUDE.md`** (architecture, every suite, config, troubleshooting). This skill
is the distilled map + the actionable playbooks. Read `benchmark/CLAUDE.md` when you need depth beyond this.

## THE convention: NPY/NS (memorize this)

> **ratio = NumPy_ms / NumSharp_ms.** `>1` = NumSharp **faster**, `<1` = slower, `=1` = parity. **Higher is better.**

Used everywhere — matrices, geomeans, commit messages, every `*_sheet.py`. Icons: ✅ `≥1.0` · 🟡 `≥0.5` · 🟠 `≥0.2`
· 🔴 `<0.2`. (The legacy `run-benchmarks.ps1` prints the INVERSE NS/NPY — prefer NPY/NS for anything new.)

## THE pitfall: Debug taints timings ~2×

Ad-hoc `dotnet run file.cs` / `dotnet_run` (file-based apps) compile **both the script AND any `#:project`
NumSharp.Core in Debug** (`DebuggableAttribute(DisableOptimizations)`), which the JIT honors even over
`[MethodImpl(AggressiveOptimization)]`. Hand-written C# hot loops run ~2× slow; IL-emitted kernels look normal.
**Every timing script MUST run `dotnet run -c Release - < script.cs`.** `#:property Optimize=true` fixes only the
script assembly, not Core. The BenchmarkDotNet projects are exempt (they mandate `-c Release`).

## Structure — two sides, joined on (op, dtype, N)

| Side | Where | What |
|------|-------|------|
| C# | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/<Category>/*.cs` | BenchmarkDotNet classes; `[Benchmark(Description="np.foo(a)")]` methods. |
| NumPy | `benchmark/NumSharp.Benchmark.Python/numpy_benchmark.py` | `run_<suite>_benchmarks(...)` emitting `BenchmarkResult` rows. |
| Merge | `benchmark/scripts/merge-results.py` | Joins on `(normalize_op_name(name), dtype, N)`. |
| Orchestrator | `benchmark/run_benchmark.py` | Builds C#, runs each suite (BDN) + warm NumPy across 1K/100K/10M, merges, snapshots. |

The join is by **normalized op name**: `normalize_op_name` strips the dtype tag, `[annotations]`, and
identifier-only arg lists — so C# `"np.foo(a)"` and NumPy `"np.foo"` both collapse to `np.foo` and join. Get the
names to normalize identically or the row shows as "C# not run" / "NumPy only".

## The matrix + subsystems

- **Op matrix** — 14 comparison suites, each a C# namespace filter in `run_benchmark.py`'s `SUITES` map
  (`arithmetic, unary, reduction, broadcast, creation, manipulation, slicing, comparison, bitwise, logic,
  statistics, sorting, linalg, selection`). Swept over 1K/100K/10M × the 15 dtypes.
- **Five appended subsystems** (own result models, appended not merged): `nditer` (iterator machinery),
  `layout` (op × 8 memory layouts × dtype), `operand` (1-D/scalar/mixed/broadcast), `cast` (astype 15×15 × layout),
  `fusion` (`np.evaluate`). Each is a `*_bench.{cs,py}` pair + a `*_sheet.py` renderer.

## Playbook — add a benchmark for a new op

The most common task. Full worked example in **`references/add-benchmark.md`**. In brief:

1. **C# side** — add a `[Benchmark(Description = "np.foo(a)")]` method to the class in
   `Benchmarks/<Category>/` that fits (or a new class in that namespace so the suite's `*Benchmarks.<Category>.*`
   filter auto-includes it). Use `BenchmarkBase` (single-dtype, float64) like the manipulation classes, or
   `TypedBenchmarkBase` (dtype-swept) like arithmetic. `[Params(Medium, Large)]` for size.
2. **NumPy twin** — append to the matching `run_<suite>_benchmarks(...)` in `numpy_benchmark.py`, setting
   `r.name, r.category, r.suite, r.dtype`. Make `.name` normalize to the C# Description (`"np.foo"` ↔ `"np.foo(a)"`).
3. **Smoke it** (this is usually the right scope — a full measured run is the post-release CI job):
   `dotnet build -c Release`; `dotnet run -c Release --no-build -f net10.0 -- --list flat | grep <Class>` to confirm
   BenchmarkDotNet discovers it; `python numpy_benchmark.py --suite <suite> --quick` to confirm the NumPy rows emit.
4. **Full numbers** come from `python run_benchmark.py` (or the `benchmark.yml` post-release workflow).

## Other tasks → where to go

- **Run the suite (official / subset), interpret the report, snapshots** → `references/run-and-report.md`.
- **Add or edit a matrix subsystem (nditer/layout/operand/cast/fusion)** → `references/subsystems.md`.
- **Everything else (all suites, config internals, troubleshooting, type map)** → `benchmark/CLAUDE.md`.

## Gotchas

- **BenchmarkDotNet's out-of-process toolchain fails here** ("project names need to be unique") because sibling
  `.claude/worktrees/` checkouts hold same-named benchmark projects. The official run uses **InProcessEmit**
  (`OfficialBenchmarkConfig`). For a smoke check use `--list flat` (reflection only, no toolchain) rather than
  trying to run BDN ad-hoc.
- **What we commit is `benchmark/history/<date>_<sha>/`**, not the gitignored `benchmark/results/<ts>/` scratch.
  Reference `benchmark/history/latest/benchmark-report.md`.
- **These are mostly view ops → sub-µs.** flip/rot90/transpose-aliases are O(1) views; their benchmark tracks
  allocation/dispatch overhead, not throughput. Ops doing real work (trim_zeros, reductions) are where ratios
  are meaningful.

## References

- `references/add-benchmark.md` — the detailed add-a-benchmark playbook (C# + NumPy twin + join-key rules + smoke).
- `references/run-and-report.md` — running the official run / subsets, the report + history snapshots, InProcessEmit.
- `references/subsystems.md` — the five subsystems and how to add one (`*_bench.{cs,py}` + `*_sheet.py`).
