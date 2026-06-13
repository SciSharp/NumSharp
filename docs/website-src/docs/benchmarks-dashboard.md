# Dashboard — NumSharp vs NumPy (operation matrix)

> _Dense, numbers-first view of the full op × dtype × N comparison — companion to the narrative [Benchmarks vs NumPy](benchmarks.md) and the [iterator sheet](benchmark-iterator.md). Auto-generated each release. speedup = NumPy ÷ NumSharp (**>1.0× = NumSharp faster**); %NumPy🕐 = (NumSharp ÷ NumPy)×100 = the share of NumPy's time NumSharp uses (30% = takes only 30% as long)._

```
NumSharp vs NumPy — operation matrix · 2026-06-13 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
832 credible comparisons of 1233 ops · 275 negligible + 126 no-data excluded · BenchmarkDotNet vs NumPy 2.4.2
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (30% = takes only 30% as long; <100% = faster)

HEADLINE — 0.74× geomean · 136%🕐 of NumPy's time · over 832 cells · 305 faster / 527 slower

BY ARRAY-SIZE TIER  (geomean over all credible ops at that size)
          slower ◄───────── 1.0 (parity) ─────────► faster
1K           ███████▎ ...........   0.73×   137%🕐  (  41 win /  68 lose)  ◄ SLOWER
100K         █████▌ .............   0.55×   182%🕐  ( 104 win / 255 lose)  ◄ SLOWER
10M          █████████▊ .........   0.98×   102%🕐  ( 160 win / 204 lose)  ◄ PARITY
ALL          ███████▎ ...........   0.74×   136%🕐  ( 305 win / 527 lose)  ◄ SLOWER

BY SUITE  (geomean, ranked fastest → slowest)
          slower ◄───────── 1.0 (parity) ─────────► faster
statistics   ███████████████████▶   2.28×    44%🕐  (  21 win /  11 lose)
broadcasting ████████████▏ ......   1.22×    82%🕐  (   3 win /   0 lose)
reduction    ████████████ .......   1.21×    83%🕐  ( 137 win /  86 lose)
bitwise      █████████▉ .........   0.99×   101%🕐  (  56 win /  43 lose)  ◄ PARITY
comparison   █████████▌ .........   0.95×   105%🕐  (  24 win /  24 lose)  ◄ SLOWER
selection    ███████▌ ...........   0.76×   132%🕐  (   1 win /   4 lose)  ◄ SLOWER
arithmetic   █████▎ .............   0.53×   190%🕐  (  32 win / 232 lose)  ◄ SLOWER
sorting      █████ ..............   0.50×   198%🕐  (  10 win /  14 lose)  ◄ SLOWER
linearalgebra████ ...............   0.40×   247%🕐  (   2 win /   6 lose)  ◄ SLOWER
unary        ███▊ ...............   0.38×   263%🕐  (   1 win /  79 lose)  ◄ SLOWER
creation     ███▌ ...............   0.35×   283%🕐  (  18 win /  28 lose)  ◄ SLOWER

BY DTYPE  (geomean over all credible ops of that type)
          slower ◄───────── 1.0 (parity) ─────────► faster
uint8        ██████████▋ ........   1.07×    93%🕐  (  35 win /  15 lose)
uint32       ████████▌ ..........   0.85×   118%🕐  (  21 win /  30 lose)  ◄ SLOWER
int16        ███████▋ ...........   0.77×   130%🕐  (  23 win /  31 lose)  ◄ SLOWER
float64      ███████▌ ...........   0.75×   133%🕐  (  64 win / 125 lose)  ◄ SLOWER
float32      ███████▌ ...........   0.75×   133%🕐  (  60 win / 116 lose)  ◄ SLOWER
uint16       ███████▎ ...........   0.74×   136%🕐  (  22 win /  33 lose)  ◄ SLOWER
uint64       ███████ ............   0.70×   142%🕐  (  14 win /  36 lose)  ◄ SLOWER
int32        ██████▋ ............   0.67×   149%🕐  (  36 win /  65 lose)  ◄ SLOWER
int64        ██████▏ ............   0.62×   161%🕐  (  30 win /  68 lose)  ◄ SLOWER
bool         ██▊ ................   0.28×   356%🕐  (   0 win /   8 lose)  ◄ SLOWER

STATUS MIX  (NumSharp ÷ NumPy bands; credible only)
✅ faster   ≤100% NumPy  ██████████████████ 305
🟡 close    100–200%     ███████████████    255
🟠 slower   200–500%     ██████████         169
🔴 much     >500%        ██████             103

TOP 12 FASTEST  (NumPy ÷ NumSharp — biggest NumSharp wins)
  operation                      dtype       N     NumPy   NumSharp    NP/NS   %NumPy🕐
  np.nansum(a) (float64)         float64  100K     0.242 →    0.019 ms   12.65×     8%🕐
  np.percentile(a, 50) (float64) float64    1K     0.025 →    0.002 ms   10.50×    10%🕐
  np.percentile(a, 50) (float32) float32    1K     0.025 →    0.002 ms   10.30×    10%🕐
  np.average(a) (float32)        float32   10M     9.598 →    0.937 ms   10.24×    10%🕐
  np.quantile(a, 0.5) (float32)  float32    1K     0.024 →    0.002 ms   10.01×    10%🕐
  np.quantile(a, 0.5) (float64)  float64    1K     0.023 →    0.002 ms    9.89×    10%🕐
  np.nanprod(a) (float32)        float32   10M    18.515 →    1.904 ms    9.72×    10%🕐
  np.nansum(a) (float32)         float32   10M    14.349 →    1.488 ms    9.64×    10%🕐
  np.nanprod(a) (float64)        float64  100K     0.287 →    0.032 ms    8.98×    11%🕐
  np.average(a) (float32)        float32  100K     0.018 →    0.002 ms    8.32×    12%🕐
  np.count_nonzero(a) (float32)  float32  100K     0.038 →    0.005 ms    8.26×    12%🕐
  np.nanstd(a) (float32)         float32    1K     0.020 →    0.003 ms    8.08×    12%🕐

TOP 12 SLOWEST  (smallest NumPy ÷ NumSharp = optimization priorities)
  operation                      dtype       N     NumPy   NumSharp    NP/NS   %NumPy🕐
  np.zeros (int64)               int64     10M     0.012 →   10.747 ms   0.001×  87957%🕐
  np.zeros (int32)               int32     10M     0.011 →    5.622 ms   0.002×  51820%🕐
  np.zeros (float64)             float64   10M     0.021 →   10.755 ms   0.002×  50765%🕐
  np.zeros (float32)             float32   10M     0.017 →    5.673 ms   0.003×  33403%🕐
  np.argsort(a) (int64)          int64    100K     0.472 →   12.893 ms   0.037×  2734%🕐
  np.argsort(a) (int32)          int32    100K     0.442 →   10.404 ms   0.042×  2354%🕐
  a * 2 (literal) (float32)      float32  100K     0.007 →    0.129 ms   0.052×  1937%🕐
  np.left_shift(a, 2) (int64)    int64      1K     0.001 →    0.020 ms   0.052×  1911%🕐
  np.right_shift(a, 2) (int64)   int64      1K     0.001 →    0.019 ms   0.052×  1920%🕐
  np.sum axis=1 (uint8)          uint8     10M     3.115 →   49.741 ms   0.063×  1597%🕐
  np.sum axis=0 (uint16)         uint16    10M     4.620 →   71.694 ms   0.064×  1552%🕐
  np.right_shift(a, 2) (int32)   int32      1K     0.001 →    0.017 ms   0.064×  1561%🕐

note · speedup = NumPy ÷ NumSharp on one runner (>1.0× = NumSharp faster) · %NumPy🕐 = share of
       NumPy's time NumSharp uses · negligible rows (<1µs / >20× = overhead, lazy alloc, views) excluded
```
