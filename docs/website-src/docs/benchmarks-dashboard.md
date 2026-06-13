# Dashboard — NumSharp vs NumPy (operation matrix)

> _Dense, numbers-first view of the full op × dtype × N comparison — the companion to the narrative [Benchmarks vs NumPy](benchmarks.md) page and the [iterator sheet](benchmark-iterator.md). Auto-generated each release; speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)._

```
NumSharp vs NumPy — operation matrix · 2026-06-13 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
832 credible comparisons of 1233 ops · 275 negligible + 126 no-data excluded · BenchmarkDotNet vs NumPy 2.4.2

HEADLINE — 0.74× geomean over 832 credible cells · 305 win / 527 lose

BY ARRAY-SIZE TIER  (geomean over all credible ops at that size)
          slower ◄───────── 1.0 (parity) ─────────► faster
1K           ███████▎ ...........   0.73×   (  41 win /  68 lose)   ◄ SLOWER
100K         █████▌ .............   0.55×   ( 104 win / 255 lose)   ◄ SLOWER
10M          █████████▊ .........   0.98×   ( 160 win / 204 lose)   ◄ PARITY
ALL          ███████▎ ...........   0.74×   ( 305 win / 527 lose)   ◄ SLOWER

BY SUITE  (geomean, ranked fastest → slowest)
          slower ◄───────── 1.0 (parity) ─────────► faster
statistics   ███████████████████▶   2.28×   (  21 win /  11 lose)
broadcasting ████████████▏ ......   1.22×   (   3 win /   0 lose)
reduction    ████████████ .......   1.21×   ( 137 win /  86 lose)
bitwise      █████████▉ .........   0.99×   (  56 win /  43 lose)   ◄ PARITY
comparison   █████████▌ .........   0.95×   (  24 win /  24 lose)   ◄ SLOWER
selection    ███████▌ ...........   0.76×   (   1 win /   4 lose)   ◄ SLOWER
arithmetic   █████▎ .............   0.53×   (  32 win / 232 lose)   ◄ SLOWER
sorting      █████ ..............   0.50×   (  10 win /  14 lose)   ◄ SLOWER
linearalgebra████ ...............   0.41×   (   2 win /   6 lose)   ◄ SLOWER
unary        ███▊ ...............   0.38×   (   1 win /  79 lose)   ◄ SLOWER
creation     ███▌ ...............   0.35×   (  18 win /  28 lose)   ◄ SLOWER

BY DTYPE  (geomean over all credible ops of that type)
          slower ◄───────── 1.0 (parity) ─────────► faster
uint8        ██████████▋ ........   1.07×   (  35 win /  15 lose)
uint32       ████████▍ ..........   0.85×   (  21 win /  30 lose)   ◄ SLOWER
int16        ███████▋ ...........   0.77×   (  23 win /  31 lose)   ◄ SLOWER
float64      ███████▌ ...........   0.75×   (  64 win / 125 lose)   ◄ SLOWER
float32      ███████▌ ...........   0.75×   (  60 win / 116 lose)   ◄ SLOWER
uint16       ███████▎ ...........   0.74×   (  22 win /  33 lose)   ◄ SLOWER
uint64       ███████ ............   0.70×   (  14 win /  36 lose)   ◄ SLOWER
int32        ██████▋ ............   0.67×   (  36 win /  65 lose)   ◄ SLOWER
int64        ██████▏ ............   0.62×   (  30 win /  68 lose)   ◄ SLOWER
bool         ██▊ ................   0.28×   (   0 win /   8 lose)   ◄ SLOWER

STATUS MIX  (NumSharp ÷ NumPy bands; credible only)
✅ faster   ≥1.0×      ██████████████████ 305
🟡 close    0.5–1.0×   ███████████████    255
🟠 slower   0.2–0.5×   ██████████         169
🔴 much     <0.2×      ██████             103

TOP 12 WINS  (NumSharp fastest vs NumPy)
  operation                      dtype       N     NumPy   NumSharp     speedup
  np.nansum(a) (float64)         float64  100K     0.242 →    0.019 ms   12.63×
  np.percentile(a, 50) (float64) float64    1K     0.025 →    0.002 ms   10.65×
  np.percentile(a, 50) (float32) float32    1K     0.025 →    0.002 ms   10.33×
  np.average(a) (float32)        float32   10M     9.598 →    0.937 ms   10.24×
  np.quantile(a, 0.5) (float64)  float64    1K     0.023 →    0.002 ms   10.09×
  np.quantile(a, 0.5) (float32)  float32    1K     0.024 →    0.002 ms   10.00×
  np.nanprod(a) (float32)        float32   10M    18.515 →    1.904 ms    9.72×
  np.nansum(a) (float32)         float32   10M    14.349 →    1.488 ms    9.64×
  np.nanprod(a) (float64)        float64  100K     0.287 →    0.032 ms    8.99×
  np.average(a) (float32)        float32  100K     0.018 →    0.002 ms    8.43×
  np.count_nonzero(a) (float32)  float32  100K     0.038 →    0.005 ms    8.24×
  np.nanstd(a) (float32)         float32    1K     0.020 →    0.003 ms    8.12×

TOP 12 LOSSES  (NumSharp slowest vs NumPy)
  operation                      dtype       N     NumPy   NumSharp     speedup
  np.zeros (int64)               int64     10M     0.012 →   10.747 ms   0.001× (881× slower)
  np.zeros (int32)               int32     10M     0.011 →    5.622 ms   0.002× (521× slower)
  np.zeros (float64)             float64   10M     0.021 →   10.755 ms   0.002× (507× slower)
  np.zeros (float32)             float32   10M     0.017 →    5.673 ms   0.003× (334× slower)
  np.argsort(a) (int64)          int64    100K     0.472 →   12.893 ms   0.037× (27× slower)
  np.argsort(a) (int32)          int32    100K     0.442 →   10.404 ms   0.042× (24× slower)
  np.left_shift(a, 2) (int64)    int64      1K     0.001 →    0.020 ms   0.050× (20× slower)
  np.right_shift(a, 2) (int64)   int64      1K     0.001 →    0.019 ms   0.051× (20× slower)
  a * 2 (literal) (float32)      float32  100K     0.007 →    0.129 ms   0.052× (19× slower)
  np.sum axis=1 (uint8)          uint8     10M     3.115 →   49.741 ms   0.063× (16× slower)
  np.right_shift(a, 2) (int32)   int32      1K     0.001 →    0.017 ms   0.064× (16× slower)
  np.sum axis=0 (uint16)         uint16    10M     4.620 →   71.694 ms   0.064× (16× slower)

note · speedup = NumPy ÷ NumSharp on one runner · negligible rows (<1µs work or >20× = call
       overhead / lazy alloc / views) excluded · absolute ms drift by hardware, ratios hold
```
