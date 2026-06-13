# Dashboard — NumSharp vs NumPy (operation matrix)

> _Dense, numbers-first view of the full op × dtype × N comparison — the companion to the narrative [Benchmarks vs NumPy](benchmarks.md) page and the [iterator sheet](benchmark-iterator.md). Auto-generated each release. Ratio = NumSharp ÷ NumPy (**<1.0× = NumSharp faster**, 1.0 = parity), matching the [full report](benchmark-matrix.md)._

```
NumSharp vs NumPy — operation matrix · 2026-06-13 · ratio = NumSharp ÷ NumPy (<1.0× = NumSharp faster, 1.0 = parity)
832 credible comparisons of 1233 ops · 275 negligible + 126 no-data excluded · BenchmarkDotNet vs NumPy 2.4.2

HEADLINE — 1.36× geomean (NumSharp ÷ NumPy) over 832 cells · 305 faster / 527 slower

BY ARRAY-SIZE TIER  (geomean over all credible ops at that size)
          faster ◄───────── 1.0 (parity) ─────────► slower
1K           █████████████▋ .....   1.37×   (  41 faster /  68 slower)   ◄ SLOWER
100K         ██████████████████▏    1.82×   ( 104 faster / 255 slower)   ◄ SLOWER
10M          ██████████▏ ........   1.02×   ( 160 faster / 204 slower)   ◄ PARITY
ALL          █████████████▌ .....   1.36×   ( 305 faster / 527 slower)   ◄ SLOWER

BY SUITE  (geomean, ranked fastest → slowest)
          faster ◄───────── 1.0 (parity) ─────────► slower
statistics   ████▍ ..............   0.44×   (  21 faster /  11 slower)   ◄ FASTER
broadcasting ████████▏ ..........   0.82×   (   3 faster /   0 slower)   ◄ FASTER
reduction    ████████▎ ..........   0.83×   ( 137 faster /  86 slower)   ◄ FASTER
bitwise      ██████████ .........   1.01×   (  56 faster /  43 slower)   ◄ PARITY
comparison   ██████████▌ ........   1.05×   (  24 faster /  24 slower)   ◄ SLOWER
selection    █████████████▏ .....   1.32×   (   1 faster /   4 slower)   ◄ SLOWER
arithmetic   ███████████████████    1.90×   (  32 faster / 232 slower)   ◄ SLOWER
sorting      ███████████████████▊   1.99×   (  10 faster /  14 slower)   ◄ SLOWER
linearalgebra███████████████████▶   2.46×   (   2 faster /   6 slower)   ◄ SLOWER
unary        ███████████████████▶   2.63×   (   1 faster /  79 slower)   ◄ SLOWER
creation     ███████████████████▶   2.83×   (  18 faster /  28 slower)   ◄ SLOWER

BY DTYPE  (geomean over all credible ops of that type)
          faster ◄───────── 1.0 (parity) ─────────► slower
uint8        █████████▎ .........   0.93×   (  35 faster /  15 slower)   ◄ FASTER
uint32       ███████████▊ .......   1.18×   (  21 faster /  30 slower)   ◄ SLOWER
int16        ████████████▉ ......   1.30×   (  23 faster /  31 slower)   ◄ SLOWER
float64      █████████████▎ .....   1.33×   (  64 faster / 125 slower)   ◄ SLOWER
float32      █████████████▎ .....   1.33×   (  60 faster / 116 slower)   ◄ SLOWER
uint16       █████████████▌ .....   1.36×   (  22 faster /  33 slower)   ◄ SLOWER
uint64       ██████████████▏ ....   1.42×   (  14 faster /  36 slower)   ◄ SLOWER
int32        ██████████████▉ ....   1.49×   (  36 faster /  65 slower)   ◄ SLOWER
int64        ████████████████ ...   1.60×   (  30 faster /  68 slower)   ◄ SLOWER
bool         ███████████████████▶   3.55×   (   0 faster /   8 slower)   ◄ SLOWER

STATUS MIX  (NumSharp ÷ NumPy bands; credible only)
✅ faster   ≤1.0×    ██████████████████ 305
🟡 close    1–2×     ███████████████    255
🟠 slower   2–5×     ██████████         169
🔴 much     >5×      ██████             103

TOP 12 FASTEST  (NumSharp ÷ NumPy, smallest = most ahead of NumPy)
  operation                      dtype       N     NumPy   NumSharp    NS/NP
  np.nansum(a) (float64)         float64  100K     0.242 →    0.019 ms    0.079× ( 12.6× faster)
  np.percentile(a, 50) (float64) float64    1K     0.025 →    0.002 ms    0.094× ( 10.7× faster)
  np.percentile(a, 50) (float32) float32    1K     0.025 →    0.002 ms    0.097× ( 10.3× faster)
  np.average(a) (float32)        float32   10M     9.598 →    0.937 ms    0.098× ( 10.2× faster)
  np.quantile(a, 0.5) (float64)  float64    1K     0.023 →    0.002 ms    0.099× ( 10.1× faster)
  np.quantile(a, 0.5) (float32)  float32    1K     0.024 →    0.002 ms    0.100× ( 10.0× faster)
  np.nanprod(a) (float32)        float32   10M    18.515 →    1.904 ms    0.103× (  9.7× faster)
  np.nansum(a) (float32)         float32   10M    14.349 →    1.488 ms    0.104× (  9.6× faster)
  np.nanprod(a) (float64)        float64  100K     0.287 →    0.032 ms    0.111× (  9.0× faster)
  np.average(a) (float32)        float32  100K     0.018 →    0.002 ms    0.119× (  8.4× faster)
  np.count_nonzero(a) (float32)  float32  100K     0.038 →    0.005 ms    0.121× (  8.2× faster)
  np.nanstd(a) (float32)         float32    1K     0.020 →    0.003 ms    0.123× (  8.1× faster)

TOP 12 SLOWEST  (largest NumSharp ÷ NumPy = optimization priorities)
  operation                      dtype       N     NumPy   NumSharp    NS/NP
  np.zeros (int64)               int64     10M     0.012 →   10.747 ms    880.9× (  881× slower)
  np.zeros (int32)               int32     10M     0.011 →    5.622 ms    520.6× (  521× slower)
  np.zeros (float64)             float64   10M     0.021 →   10.755 ms    507.3× (  507× slower)
  np.zeros (float32)             float32   10M     0.017 →    5.673 ms    333.7× (  334× slower)
  np.argsort(a) (int64)          int64    100K     0.472 →   12.893 ms     27.3× (   27× slower)
  np.argsort(a) (int32)          int32    100K     0.442 →   10.404 ms     23.5× (   24× slower)
  np.left_shift(a, 2) (int64)    int64      1K     0.001 →    0.020 ms     19.9× (   20× slower)
  np.right_shift(a, 2) (int64)   int64      1K     0.001 →    0.019 ms     19.5× (   20× slower)
  a * 2 (literal) (float32)      float32  100K     0.007 →    0.129 ms     19.2× (   19× slower)
  np.sum axis=1 (uint8)          uint8     10M     3.115 →   49.741 ms     16.0× (   16× slower)
  np.right_shift(a, 2) (int32)   int32      1K     0.001 →    0.017 ms     15.6× (   16× slower)
  np.sum axis=0 (uint16)         uint16    10M     4.620 →   71.694 ms     15.5× (   16× slower)

note · ratio = NumSharp ÷ NumPy on one runner (<1.0× = NumSharp faster) · negligible rows
       (<1µs work or >20× = call overhead / lazy alloc / views) excluded · ratios hold, ms drift
```
