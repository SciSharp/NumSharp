# Benchmark snapshot — 2026-06-05 · 6038990f

Official NumSharp-vs-NumPy 3-size comparison, persisted for provenance.

## Provenance
| | |
|---|---|
| Run timestamp | `20260605-125639` |
| Git HEAD (report + suite) | `6038990f` |
| Benchmarked NumSharp.Core | `d01f1d63` (fused strided-SIMD unary; last Core change before the run) |
| Date | 2026-06-05 |

## Environment
| | |
|---|---|
| CPU | 13th Gen Intel Core i9-13900K |
| OS | Windows 11 Pro N (10.0.26200) |
| .NET SDK | 10.0.101 (net10.0, Release) |
| Python | 3.12.12 |
| NumPy | 2.4.2 |

## Methodology
- **C#:** BenchmarkDotNet, `OfficialBenchmarkConfig` — InProcessEmit toolchain, 50 measured
  iterations / 5 warmup, iteration time capped at 25 ms (so µs–ms array ops don't get the
  nanosecond-microbenchmark invocation ramp). MemoryDiagnoser on.
- **NumPy:** 50 timed iterations / 10 warmup per op (warm long-lived interpreter).
- **Sizes:** 1,000 / 100,000 / 10,000,000 elements. Same seeds both sides.
- Join keyed on (op, dtype, N). 1,813 C# measurements → 1,111 matched comparisons.

## Headline — geomean ratio (NumSharp ÷ NumPy, lower = better)
| Size | geomean | faster / close / slower / much |
|---|---|---|
| 1,000 | 1.96× | 102 / 53 / 128 / 84 |
| 100,000 | 1.83× | 109 / 66 / 121 / 75 |
| **10,000,000** | **1.00× (parity)** | **166 / 171 / 20 / 16** |

## Per-suite geomean by size
| suite | 1K | 100K | 10M |
|---|--:|--:|--:|
| Statistics | 0.19× | 0.68× | 0.48× |
| Sorting | 0.41× | 1.13× | 0.45× |
| Reduction | 0.48× | 0.94× | 0.91× |
| Comparison | 1.27× | 2.22× | 0.50× |
| Bitwise | 8.16× | 1.16× | 0.61× |
| Arithmetic | 3.09× | 2.62× | 1.25× |
| Unary | 3.50× | 4.44× | 1.53× |
| Creation | 12.26× | 2.92× | 2.24× |
| LinearAlgebra | 2.76× | 1.66× | 4.02× |

## Files
| file | what |
|---|---|
| `benchmark-report.md` | human-readable 3-size ratio matrix (per-suite, with N column) |
| `benchmark-report.json` | machine-readable unified results (1,233 rows) |
| `benchmark-report.csv` | spreadsheet form |
| `numpy-results.json` | raw NumPy timings (all sizes) — merge input |

Raw BenchmarkDotNet per-class JSON (~34 MB) is **not** persisted here (regenerable).
Reproduce with: `python benchmark/run_benchmark.py` at this commit.
