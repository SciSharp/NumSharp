# Benchmark snapshot — 2026-06-23 · e3b7c268

Official NumSharp-vs-NumPy 3-size comparison + the five matrix subsystems, persisted for provenance.

## Provenance
| | |
|---|---|
| Run timestamp | `20260623-065155` |
| Git HEAD (report commit) | `e3b7c268` |
| Benchmarked tree | branch HEAD `b8623db6` **+ uncommitted engine WIP** (Cast / Reduction.Axis.Simd / Scan / Search / MatMul / Unary / Comparison / Binary / BitwiseOp / UnmanagedStorage.Cloning) — built clean in Release |
| Date | 2026-06-23 |

## Environment
| | |
|---|---|
| CPU | 13th Gen Intel Core i9-13900K |
| OS | Windows 11 Pro N (10.0.26200) |
| .NET SDK | 10.0.101 (net10.0, Release) |
| Python | 3.12.12 |
| NumPy | 2.4.2 |

## Convention
**Ratio = NumPy_ms ÷ NumSharp_ms (NPY/NS) → `>1.0×` = NumSharp faster** (higher is better).
> The 2026-06-05 snapshot reported the *inverse* (NS/NPY, lower = better). The project has since standardized on **NPY/NS**, which every report in this snapshot uses.

## Methodology
- **C#:** BenchmarkDotNet, `OfficialBenchmarkConfig` — InProcessEmit toolchain, 50 measured
  iterations / 5 warmup, iteration time capped at 25 ms (so µs–ms array ops don't get the
  nanosecond-microbenchmark invocation ramp). MemoryDiagnoser on.
- **NumPy:** 50 timed iterations / 10 warmup per op (warm long-lived interpreter).
- **Sizes:** 1,000 / 100,000 / 10,000,000 elements. Same seeds both sides.
- Join keyed on (op, dtype, N): 1,851 ops → 1,398 credible comparisons (384 negligible
  rows, <1 µs or >20×, excluded as non-comparable).
- **Subsystems** appended to `benchmark-report.md` (own size/dtype grids, not part of the
  op/dtype/N matrix): NpyIter (aspect × tier), Layout, Operand, Cast (astype src→dst ×
  layout), Fusion (np.evaluate fused vs unfused).

## Headline — op-matrix geomean by size (NPY/NS, >1 = NumSharp faster)
| Size | geomean | %NumPy🕐 | ✅ / 🟡 / 🟠 / 🔴 |
|---|--:|--:|---|
| 1,000 | 1.14× | 87% | 115 / 69 / 27 / 13 |
| 100,000 | 0.90× | 111% | 280 / 138 / 119 / 48 |
| **10,000,000** | **1.26×** | **80%** | **397 / 150 / 31 / 11** |

Overall: **1,851 ops | ✅ 792 · 🟡 357 · 🟠 177 · 🔴 72 · ▫ 384 · ⚪ 69**.

## Subsystem headlines (NPY/NS)
| subsystem | headline |
|---|---|
| NpyIter | 1.18× geomean op-matrix · reductions 1.75× · construction 3.3× · `bcast_reduce` **538×** · `any(F)` 2.74× · `forder_out` 1.28× |
| Cast | **1,439 / 1,568** cells win · per-layout geomean 1.47–2.21× · f16→narrow 3.77× |
| Operand | every class wins f64/f32/i32/i64/c128 (geomean 1.34–2.18×) · f16 0.52–0.86 (no `Vector<Half>`) |
| Fusion | `a*b+c` 1.56× · `(a-b)/(a+b)` **4.16×** · F/T 1.85×/1.74× · bcast 3.60× |

## Files
| file | what |
|---|---|
| `benchmark-report.md` | human-readable report: op-matrix (per-(op,dtype,N) ratio) + appended NpyIter / Layout / Operand / Cast / Fusion sections |
| `benchmark-report.json` | machine-readable unified op-matrix results |
| `benchmark-report.csv` | spreadsheet form |
| `numpy-results.json` | raw NumPy timings (all sizes) — merge input |

Raw BenchmarkDotNet per-class JSON (~tens of MB) is **not** persisted here (regenerable).
Reproduce with `python benchmark/run_benchmark.py` against the benchmarked tree above.
The five subsystem sheets (`*_results.md` / `.tsv`) live committed under their
`benchmark/<subsystem>/` dirs and are mirrored into `benchmark-report.md` here.
