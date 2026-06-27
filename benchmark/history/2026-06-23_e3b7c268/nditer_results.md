```
NumSharp NDIter — canonical benchmark · 2026-06-23 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.18× geomean · 85%🕐 of NumPy's time · 72 win / 58 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████▉ ........   1.10×    91%🕐  ( 12 win / 14 lose)
1K         ███████████▊ .......   1.19×    84%🕐  ( 15 win / 11 lose)
100K       ██████████▊ ........   1.08×    93%🕐  ( 12 win / 14 lose)
1M         █████████████ ......   1.31×    77%🕐  ( 17 win /  9 lose)
10M        ████████████▎ ......   1.23×    81%🕐  ( 16 win / 10 lose)
ALL        ███████████▊ .......   1.18×    85%🕐  ( 72 win / 58 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████▊ .......   1.18×    85%🕐  ( 28 win / 12 lose)
reductions █████████████████▍     1.75×    57%🕐  ( 28 win / 12 lose)
selection  (no data)
copy/cast  ███████▎ ...........   0.73×   137%🕐  (  8 win / 17 lose)  ◄ SLOWER
index-math ███████▌ ...........   0.75×   133%🕐  (  3 win /  7 lose)  ◄ SLOWER
dtypes     ████████████▏ ......   1.22×    82%🕐  (  5 win / 10 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.05×    1.54×    1.18×    1.09×    1.11×
reductions      2.67×    1.99×    1.51×    1.44×    1.42×
selection           -        -        -        -        -
copy/cast       0.61×    0.59×    0.40×    1.39×    1.06×
index-math      0.32×    0.51×    0.97×    1.22×    1.22×
dtypes          0.71×    0.85×    1.97×    1.54×    1.47×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.01×    1.48×    1.03×    0.88×    1.01×     1.06×
  sqrt         0.85×    1.15×    1.00×    1.01×    1.02×     1.00×
  copy         0.88×    2.59×    1.78×    1.33×    1.72×     1.56×
  strided      0.89×    1.12×    1.00×    1.02×    0.99×     1.00×
  bcast        0.89×    1.13×    1.02×    0.98×    1.03×     1.01×
  reversed     0.85×    1.28×    0.90×    0.99×    1.00×     0.99×
  castbuf      1.98×    2.29×    1.65×    1.35×    1.16×     1.64×
  mixbuf       1.49×    1.94×    1.40×    1.24×    1.09×     1.40×
-- reductions
  sum          1.84×    1.78×    2.79×    2.21×    1.76×     2.04×
  sum ax0      1.90×    0.86×    0.96×    1.00×    0.94×     1.08×
  sum ax1      1.85×    0.86×    1.51×    1.83×    1.57×     1.47×
  sum dt=      1.97×    1.47×    0.49×    0.47×    0.55×     0.82×
  amin         1.70×    1.61×    0.71×    0.70×    0.82×     1.02×
  cumsum       1.47×    1.09×    1.06×    1.80×    1.68×     1.39×
  any(F)       8.89×    8.41×    2.12×    0.98×    1.00×     2.74×
  any(hit)     9.01×    8.50×    8.50×    7.87×    8.22×     8.41×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.43×    0.44×    0.17×    2.17×    0.90×     0.57×
  astype       0.30×    0.53×    0.59×    1.97×    1.90×     0.81×
  ravel.T      0.45×    0.73×    0.48×    2.11×    1.01×     0.80×
  in-place     1.77×    0.81×    0.81×    1.06×    1.02×     1.05×
  less->b      0.81×    0.52×    0.26×    0.54×    0.76×     0.54×
-- index-math
  unravel      0.33×    0.50×    0.95×    1.01×    0.97×     0.68×
  ravel_mi     0.32×    0.52×    0.99×    1.49×    1.53×     0.82×
-- dtypes
  complex      0.74×    0.63×    1.01×    0.76×    0.89×     0.80×
  float16      0.72×    0.65×    0.62×    0.62×    0.62×     0.65×
  int8         0.67×    1.47×   12.09×    7.70×    5.78×     3.51×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ██████████████████▋    1.86×    54%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.43×    23%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   4.98×    20%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.49×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.56×    39%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.26×    19%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   2.94×    34%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.65×    38%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.35×    30%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   3.33×    30%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ███████ ............   0.71×   141%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████▏ ........   1.02×    98%🕐  (  1 win /  0 lose)  ◄ PARITY
w=64         ███████████▍ .......   1.15×    87%🕐  (  1 win /  0 lose)
w=256        █████████████▍ .....   1.34×    75%🕐  (  1 win /  0 lose)
w=1024       ███████████████ ....   1.51×    66%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    538.56×   (538.6× faster, faster)
  allocate          1.10×   (1.1× faster, faster)
  overlap_copy      1.78×   (1.8× faster, faster)
  forder_out        1.28×   (1.3× faster, faster)
  zerodim           1.26×   (1.3× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.65×    3.80×    1.39×    1.62×    2.01×   vs chained 6× add
reuse            5.63×    5.30×    0.97×    1.04×    1.06×   vs rebuild each call
par8                 -    0.66×    2.70×    3.09×    4.25×   vs single-thread

biggest NumSharp wins: i8@100K 12.09× · anyeh@1 9.01× · anyff@1 8.89× · anyeh@100K 8.50× · anyeh@1K 8.50×
most behind:           flatten@100K 0.17× · lessbool@100K 0.26× · astype@1 0.30× · ravelmi@1 0.32× · unravel@1 0.33×
```
