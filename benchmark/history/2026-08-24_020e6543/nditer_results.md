```
NumSharp NDIter — canonical benchmark · 2026-08-24 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (40 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across elementwise.

HEADLINE — operation matrix: 1.30× geomean · 77%🕐 of NumPy's time · 72 win / 53 lose over 125 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     █████████▎ .........   0.93×   108%🕐  (  9 win / 16 lose)  ◄ SLOWER
1K         ███████████▋ .......   1.17×    86%🕐  ( 13 win / 12 lose)
100K       █████████████▊ .....   1.38×    72%🕐  ( 15 win / 10 lose)
1M         ████████████████▍ ..   1.64×    61%🕐  ( 17 win /  8 lose)
10M        ███████████████▏ ...   1.52×    66%🕐  ( 18 win /  7 lose)
ALL        █████████████ ......   1.30×    77%🕐  ( 72 win / 53 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise(no data)
reductions ███████████████████▶   2.55×    39%🕐  ( 38 win /  2 lose)
selection  ███████████▋ .......   1.17×    86%🕐  ( 19 win / 16 lose)
copy/cast  ███████▍ ...........   0.74×   134%🕐  (  6 win / 19 lose)  ◄ SLOWER
index-math ██████▊ ............   0.68×   146%🕐  (  5 win /  5 lose)  ◄ SLOWER
dtypes     ██████████▉ ........   1.09×    91%🕐  (  4 win / 11 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise         -        -        -        -        -
reductions      3.80×    3.26×    2.15×    2.07×    1.95×
selection       0.57×    1.12×    1.29×    1.64×    1.61×
copy/cast       0.47×    0.48×    0.69×    1.29×    1.15×
index-math      0.18×    0.45×    1.20×    1.09×    1.41×
dtypes          0.64×    0.70×    1.76×    1.74×    1.15×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add             NA       NA       NA       NA       NA
  sqrt            NA       NA       NA       NA       NA
  copy            NA       NA       NA       NA       NA
  strided         NA       NA       NA       NA       NA
  bcast           NA       NA       NA       NA       NA
  reversed        NA       NA       NA       NA       NA
  castbuf         NA       NA       NA       NA       NA
  mixbuf          NA       NA       NA       NA       NA
-- reductions
  sum          1.79×    1.84×    2.76×    2.18×    1.89×     2.06×
  sum ax0      1.87×    1.14×    1.02×    1.17×    1.04×     1.22×
  sum ax1      4.35×    1.30×    1.45×    2.06×    1.65×     1.95×
  sum dt=      2.08×    2.52×    1.14×    1.10×    1.06×     1.48×
  amin         2.24×    2.04×    0.85×    0.73×    1.29×     1.30×
  cumsum       1.53×    1.36×    1.25×    1.91×    1.68×     1.53×
  any(F)      17.93×   26.03×    3.54×    1.88×    1.38×     5.33×
  any(hit)    23.57×   25.79×   25.82×   21.86×   19.84×    23.26×
-- selection
  where        0.62×    0.87×    0.81×    1.33×    1.05×     0.91×
  a[mask]      0.21×    1.01×    2.35×    2.07×    1.65×     1.11×
  a[mask]=     0.18×    2.69×    7.56×    5.84×    4.88×     2.54×
  count_nz     0.77×    2.08×    2.71×    3.57×    1.14×     1.78×
  argwhere     0.98×    2.32×    1.07×    2.78×    3.90×     1.92×
  a[idx]       0.65×    0.32×    0.33×    0.41×    0.93×     0.48×
  a[idx]=      1.67×    0.62×    0.43×    0.48×    0.80×     0.70×
-- copy/cast
  flatten      0.43×    0.19×    0.97×    2.70×    1.29×     0.77×
  astype       0.31×    0.28×    0.57×    1.19×    1.87×     0.65×
  ravel.T      0.43×    0.68×    0.73×    1.47×    0.96×     0.79×
  in-place     0.91×    0.88×    1.05×    0.93×    0.98×     0.95×
  less->b      0.44×    0.75×    0.37×    0.81×    0.87×     0.61×
-- index-math
  unravel      0.29×    0.36×    1.13×    0.94×    1.25×     0.68×
  ravel_mi     0.11×    0.57×    1.28×    1.26×    1.59×     0.69×
-- dtypes
  complex      0.66×    0.56×    0.88×    0.90×    0.80×     0.75×
  float16      0.63×    0.42×    0.62×    0.62×    0.65×     0.58×
  int8         0.64×    1.44×   10.07×    9.38×    2.94×     3.04×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████████████▏   1.92×    52%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   5.07×    20%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   5.63×    18%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   9.06×    11%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   7.12×    14%🕐  (  1 win /  0 lose)
8op          ███████████████████▶  10.08×    10%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   6.90×    14%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   5.37×    19%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   4.98×    20%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   5.72×    17%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ████████▌ ..........   0.86×   116%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ███████████▉ .......   1.20×    84%🕐  (  1 win /  0 lose)
w=64         ████████████▋ ......   1.27×    79%🕐  (  1 win /  0 lose)
w=256        ███████████████▉ ...   1.59×    63%🕐  (  1 win /  0 lose)
w=1024       ███████████████████▊   1.98×    50%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    489.52×   (489.5× faster, faster)
  allocate          1.10×   (1.1× faster, faster)
  overlap_copy      1.79×   (1.8× faster, faster)
  forder_out        1.51×   (1.5× faster, faster)
  zerodim           1.73×   (1.7× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           13.72×    7.03×    2.54×    3.83×    1.67×   vs chained 6× add
reuse            7.40×    9.38×    1.31×    0.77×    1.03×   vs rebuild each call
par8                 -    1.52×    5.99×    3.42×    6.61×   vs single-thread

biggest NumSharp wins: anyff@1K 26.03× · anyeh@100K 25.82× · anyeh@1K 25.79× · anyeh@1 23.57× · anyeh@1M 21.86×
most behind:           ravelmi@1 0.11× · bassign@1 0.18× · flatten@1K 0.19× · bread@1 0.21× · astype@1K 0.28×
```
