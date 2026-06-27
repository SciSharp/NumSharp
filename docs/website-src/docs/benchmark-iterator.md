# Iterator Benchmark — NDIter vs NumPy (full sheet)

> _Auto-generated after each release by the [Benchmark workflow](https://github.com/SciSharp/NumSharp/blob/master/.github/workflows/benchmark.yml) — do not edit by hand. This is the canonical sheet the cards on [Benchmarks vs NumPy](benchmarks.md) render from. speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)._

```
NumSharp NDIter — canonical benchmark · 2026-06-13 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.17× geomean · 85%🕐 of NumPy's time · 77 win / 53 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ███████████▍ .......   1.14×    87%🕐  ( 11 win / 15 lose)
1K         ███████████▍ .......   1.14×    88%🕐  ( 14 win / 12 lose)
100K       ███████████▏ .......   1.12×    89%🕐  ( 17 win /  9 lose)
1M         █████████████▍ .....   1.34×    75%🕐  ( 18 win /  8 lose)
10M        ███████████▎ .......   1.13×    88%🕐  ( 17 win /  9 lose)
ALL        ███████████▋ .......   1.17×    85%🕐  ( 77 win / 53 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████▏ .......   1.12×    89%🕐  ( 22 win / 18 lose)
reductions █████████████████▉     1.80×    56%🕐  ( 34 win /  6 lose)
selection  (no data)
copy/cast  ██████▌ ............   0.65×   153%🕐  (  9 win / 16 lose)  ◄ SLOWER
index-math ██████▉ ............   0.70×   144%🕐  (  5 win /  5 lose)  ◄ SLOWER
dtypes     ███████████████▉ ...   1.59×    63%🕐  (  7 win /  8 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     0.97×    1.31×    1.23×    1.02×    1.12×
reductions      4.21×    2.72×    1.20×    1.32×    1.04×
selection           -        -        -        -        -
copy/cast       0.47×    0.36×    0.46×    1.34×    1.15×
index-math      0.23×    0.58×    1.20×    1.14×    0.89×
dtypes          0.71×    0.82×    3.15×    3.22×    1.71×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          0.99×    0.51×    1.05×    0.68×    1.12×     0.83×
  sqrt         0.82×    0.54×    1.04×    1.30×    1.11×     0.92×
  copy         0.86×    1.61×    1.37×    1.43×    2.40×     1.45×
  strided      0.91×    0.76×    1.04×    0.91×    0.97×     0.91×
  bcast        0.92×    2.26×    1.07×    0.95×    0.92×     1.14×
  reversed     0.86×    1.65×    0.83×    1.04×    0.83×     1.00×
  castbuf      1.42×    3.06×    1.84×    1.79×    1.13×     1.74×
  mixbuf       1.09×    2.19×    2.01×    0.59×    0.97×     1.22×
-- reductions
  sum          1.79×    2.34×    2.30×    2.13×    1.62×     2.02×
  sum ax0      1.71×    2.56×    1.14×    1.07×    1.18×     1.45×
  sum ax1      3.82×    2.42×    1.35×    3.58×    1.50×     2.32×
  sum dt=      3.11×    2.76×    1.18×    1.09×    1.07×     1.64×
  amin         2.57×    1.62×    0.43×    0.24×    0.74×     0.79×
  cumsum       1.89×    1.72×    2.11×    2.51×    1.15×     1.82×
  any(F)      24.56×    6.15×    0.18×    0.08×    0.09×     0.71×
  any(hit)    22.84×    4.35×    6.22×   22.07×    6.03×     9.62×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.40×    0.16×    0.26×    2.20×    1.11×     0.52×
  astype       0.29×    0.25×    0.71×    1.80×    1.57×     0.68×
  ravel.T      0.41×    0.28×    0.62×    1.71×    1.08×     0.67×
  in-place     1.02×    0.65×    0.47×    1.07×    1.23×     0.84×
  less->b      0.49×    0.83×    0.38×    0.59×    0.85×     0.60×
-- index-math
  unravel      0.30×    0.49×    1.36×    1.04×    0.76×     0.69×
  ravel_mi     0.17×    0.70×    1.07×    1.26×    1.05×     0.70×
-- dtypes
  complex      0.73×    0.60×    1.39×    2.67×    2.52×     1.33×
  float16      0.73×    0.63×    0.97×    0.99×    0.61×     0.77×
  int8         0.67×    1.47×   23.21×   12.59×    3.26×     3.93×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          █████████▋ .........   0.96×   104%🕐  (  0 win /  1 lose)  ◄ SLOWER
3op_exl      ███████████████████▶   2.44×    41%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   2.91×    34%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   7.24×    14%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   3.96×    25%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   3.78×    26%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   2.64×    38%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.31×    43%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.00×    33%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   2.88×    35%🕐  (  8 win /  1 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ███████▎ ...........   0.73×   136%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████▉ ........   1.09×    92%🕐  (  1 win /  0 lose)
w=64         ███████████▍ .......   1.15×    87%🕐  (  1 win /  0 lose)
w=256        █████████████▎ .....   1.33×    75%🕐  (  1 win /  0 lose)
w=1024       █████████████▉ .....   1.40×    71%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce      0.02×   (51.8× slower, SLOWER)
  allocate          0.57×   (1.8× slower, SLOWER)
  overlap_copy      0.73×   (1.4× slower, SLOWER)
  forder_out        0.46×   (2.2× slower, SLOWER)
  zerodim           0.92×   (1.1× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           13.10×    6.90×    2.30×    4.02×    1.76×   vs chained 6× add
reuse            6.75×    6.32×    1.25×    0.87×    1.01×   vs rebuild each call
par8                 -    1.67×    8.04×    3.55×    6.28×   vs single-thread

biggest NumSharp wins: anyff@1 24.56× · i8@100K 23.21× · anyeh@1 22.84× · anyeh@1M 22.07× · i8@1M 12.59×
most behind:           anyff@1M 0.08× · anyff@10M 0.09× · flatten@1K 0.16× · ravelmi@1 0.17× · anyff@100K 0.18×
```
