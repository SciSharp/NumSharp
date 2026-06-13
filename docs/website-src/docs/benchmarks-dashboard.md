# Dashboard — NumSharp vs NumPy (operation matrix)

> _Dense, numbers-first view of the full op × dtype × N comparison — companion to the narrative [Benchmarks vs NumPy](benchmarks.md) and the [iterator sheet](benchmark-iterator.md). Auto-generated each release. speedup = NumPy ÷ NumSharp (**>1.0× = NumSharp faster**); %NumPy🕐 = (NumSharp ÷ NumPy)×100 = the share of NumPy's time NumSharp uses (30% = takes only 30% as long)._

```
NumSharp vs NumPy — operation matrix · 2026-06-14 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
1386 credible comparisons of 1851 ops · 389 negligible + 76 no-data excluded · BenchmarkDotNet vs NumPy 2.4.2
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (30% = takes only 30% as long; <100% = faster)

HEADLINE — 1.08× geomean · 93%🕐 of NumPy's time · over 1386 cells · 797 faster / 589 slower

BY ARRAY-SIZE TIER  (geomean over all credible ops at that size)
          slower ◄───────── 1.0 (parity) ─────────► faster
1K           █████████▎ .........   0.93×   107%🕐  (  96 win / 119 lose)  ◄ SLOWER
100K         ████████▉ ..........   0.89×   112%🕐  ( 275 win / 311 lose)  ◄ SLOWER
10M          █████████████▋ .....   1.37×    73%🕐  ( 426 win / 159 lose)
ALL          ██████████▊ ........   1.08×    93%🕐  ( 797 win / 589 lose)

BY SUITE  (geomean, ranked fastest → slowest)
          slower ◄───────── 1.0 (parity) ─────────► faster
reduction    ████████████████▊ ..   1.68×    60%🕐  ( 343 win / 141 lose)
statistics   ████████████████▍ ..   1.64×    61%🕐  (  31 win /  18 lose)
broadcasting ███████████▌ .......   1.15×    87%🕐  (   3 win /   0 lose)
arithmetic   █████████▏ .........   0.92×   109%🕐  ( 183 win / 157 lose)  ◄ SLOWER
bitwise      █████████ ..........   0.91×   110%🕐  (  56 win /  57 lose)  ◄ SLOWER
logic        ████████▋ ..........   0.87×   115%🕐  (  22 win /  19 lose)  ◄ SLOWER
unary        ████████▍ ..........   0.84×   119%🕐  (  92 win / 119 lose)  ◄ SLOWER
selection    ███████▊ ...........   0.78×   129%🕐  (   2 win /   3 lose)  ◄ SLOWER
comparison   ███████▋ ...........   0.77×   131%🕐  (  24 win /  24 lose)  ◄ SLOWER
sorting      ██████ .............   0.61×   165%🕐  (  22 win /  14 lose)  ◄ SLOWER
linearalgebra█████▊ .............   0.58×   173%🕐  (   2 win /   6 lose)  ◄ SLOWER
manipulation ███▋ ...............   0.37×   269%🕐  (   1 win /   1 lose)  ◄ SLOWER
creation     ███▏ ...............   0.32×   315%🕐  (  16 win /  30 lose)  ◄ SLOWER

BY DTYPE  (geomean over all credible ops of that type)
          slower ◄───────── 1.0 (parity) ─────────► faster
uint8        ███████████████████▶   2.52×    40%🕐  (  53 win /   8 lose)
int8         ███████████████████▶   2.29×    44%🕐  (  50 win /  10 lose)
int16        ███████████████████▋   1.96×    51%🕐  (  51 win /  13 lose)
uint16       ██████████████████▍    1.85×    54%🕐  (  50 win /  14 lose)
uint32       ███████████████▎ ...   1.53×    66%🕐  (  49 win /  17 lose)
int32        ███████████▏ .......   1.12×    89%🕐  (  69 win /  41 lose)
float32      ██████████▍ ........   1.04×    96%🕐  ( 126 win / 101 lose)
float16      █████████▋ .........   0.97×   103%🕐  ( 111 win / 115 lose)  ◄ SLOWER
float64      █████████▎ .........   0.93×   108%🕐  ( 130 win / 124 lose)  ◄ SLOWER
uint64       ████████▍ ..........   0.84×   119%🕐  (  29 win /  34 lose)  ◄ SLOWER
int64        ███████▋ ...........   0.77×   130%🕐  (  57 win /  61 lose)  ◄ SLOWER
complex128   ████ ...............   0.41×   243%🕐  (  22 win /  43 lose)  ◄ SLOWER
bool         ██▉ ................   0.29×   342%🕐  (   0 win /   8 lose)  ◄ SLOWER

STATUS MIX  (NumSharp ÷ NumPy bands; credible only)
✅ faster   ≤100% NumPy  ██████████████████ 797
🟡 close    100–200%     ███████            289
🟠 slower   200–500%     █████              219
🔴 much     >500%        ██                 81

TOP 12 FASTEST  (NumPy ÷ NumSharp — biggest NumSharp wins)
  operation                      dtype       N     NumPy   NumSharp    NP/NS   %NumPy🕐
  np.sum axis=1 (uint16)         uint16    10M     8.379 →    0.421 ms   19.88×     5%🕐
  np.nanstd(a) (float64)         float64    1K     0.030 →    0.002 ms   19.48×     5%🕐
  np.sum axis=0 (int16)          int16     10M     9.299 →    0.495 ms   18.79×     5%🕐
  np.sum axis=1 (int16)          int16     10M     7.118 →    0.400 ms   17.79×     6%🕐
  np.dot(a, b) (float64)         float64  100K     0.108 →    0.007 ms   15.49×     6%🕐
  np.nanstd(a) (float32)         float32    1K     0.022 →    0.002 ms   14.96×     7%🕐
  np.nanquantile(a, 0.5) (float… float32    1K     0.036 →    0.002 ms   14.95×     7%🕐
  np.prod (float64)              float64  100K     2.349 →    0.173 ms   13.60×     7%🕐
  np.nanpercentile(a, 50) (floa… float64    1K     0.030 →    0.002 ms   12.84×     8%🕐
  np.quantile(a, 0.5) (float32)  float32    1K     0.029 →    0.002 ms   12.64×     8%🕐
  np.sum axis=0 (uint8)          uint8     10M     4.488 →    0.363 ms   12.36×     8%🕐
  np.percentile(a, 50) (float32) float32    1K     0.028 →    0.002 ms   12.23×     8%🕐

TOP 12 SLOWEST  (smallest NumPy ÷ NumSharp = optimization priorities)
  operation                      dtype       N     NumPy   NumSharp    NP/NS   %NumPy🕐
  np.zeros (int64)               int64     10M     0.011 →   11.099 ms   0.001×  100153%🕐
  np.zeros (float64)             float64   10M     0.011 →   11.197 ms   0.001×  97991%🕐
  np.zeros (int32)               int32     10M     0.011 →    2.966 ms   0.004×  26069%🕐
  np.zeros (float32)             float32   10M     0.012 →    2.981 ms   0.004×  25357%🕐
  np.mean axis=0 (complex128)    complex128 100K     0.017 →    1.093 ms   0.016×  6407%🕐
  np.sum axis=0 (complex128)     complex128 100K     0.015 →    0.808 ms   0.019×  5208%🕐
  np.mean axis=1 (complex128)    complex128 100K     0.033 →    1.050 ms   0.031×  3203%🕐
  np.argsort(a) (int64)          int64    100K     0.486 →   13.027 ms   0.037×  2682%🕐
  np.argsort(a) (int32)          int32    100K     0.407 →   10.562 ms   0.038×  2598%🕐
  np.sum axis=1 (complex128)     complex128 100K     0.032 →    0.786 ms   0.041×  2460%🕐
  np.mean axis=0 (complex128)    complex128  10M     7.632 →  185.728 ms   0.041×  2434%🕐
  np.right_shift(a, 2) (int64)   int64      1K     0.001 →    0.021 ms   0.048×  2084%🕐

note · speedup = NumPy ÷ NumSharp on one runner (>1.0× = NumSharp faster) · %NumPy🕐 = share of
       NumPy's time NumSharp uses · negligible rows (<1µs / >20× = overhead, lazy alloc, views) excluded
```
