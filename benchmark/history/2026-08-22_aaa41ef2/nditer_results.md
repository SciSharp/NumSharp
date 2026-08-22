```
NumSharp NDIter — canonical benchmark · 2026-08-22 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.40× geomean · 72%🕐 of NumPy's time · 83 win / 47 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████▊ ........   1.08×    93%🕐  ( 16 win / 10 lose)
1K         ████████████▉ ......   1.29×    78%🕐  ( 14 win / 12 lose)
100K       █████████████ ......   1.31×    76%🕐  ( 15 win / 11 lose)
1M         ███████████████████▶   2.02×    49%🕐  ( 19 win /  7 lose)
10M        ██████████████▎ ....   1.44×    70%🕐  ( 19 win /  7 lose)
ALL        █████████████▉ .....   1.40×    72%🕐  ( 83 win / 47 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████████████▶   2.24×    45%🕐  ( 39 win /  1 lose)
reductions █████████████████▍     1.74×    57%🕐  ( 29 win / 11 lose)
selection  (no data)
copy/cast  ███████▋ ...........   0.77×   130%🕐  ( 10 win / 15 lose)  ◄ SLOWER
index-math ██████ .............   0.61×   164%🕐  (  1 win /  9 lose)  ◄ SLOWER
dtypes     ██████████▏ ........   1.02×    98%🕐  (  4 win / 11 lose)  ◄ PARITY

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.46×    2.73×    1.92×    3.61×    2.05×
reductions      2.48×    1.87×    1.45×    1.61×    1.49×
selection           -        -        -        -        -
copy/cast       0.42×    0.46×    0.61×    2.13×    1.10×
index-math      0.18×    0.52×    0.86×    0.93×    1.11×
dtypes          0.80×    0.67×    1.73×    1.22×    0.95×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.43×    2.66×    0.99×    2.68×    3.94×     2.09×
  sqrt         1.42×    4.15×    4.63×    5.81×    3.32×     3.50×
  copy         1.50×    3.16×    2.38×    7.39×    3.93×     3.18×
  strided      1.47×    1.77×    1.14×    1.99×    2.49×     1.71×
  bcast        1.47×    2.53×    1.63×    3.19×    1.95×     2.07×
  reversed     1.51×    1.58×    1.18×    2.97×    1.00×     1.53×
  castbuf      1.63×    3.49×    3.10×    4.25×    1.13×     2.43×
  mixbuf       1.31×    3.57×    2.46×    3.14×    1.13×     2.10×
-- reductions
  sum          1.58×    1.57×    2.82×    2.84×    1.65×     2.01×
  sum ax0      1.49×    0.89×    0.87×    0.97×    1.06×     1.04×
  sum ax1      1.39×    0.85×    1.28×    3.15×    1.75×     1.53×
  sum dt=      1.49×    1.32×    0.49×    0.55×    0.64×     0.80×
  amin         1.65×    1.38×    0.65×    0.90×    0.87×     1.03×
  cumsum       1.22×    0.94×    1.03×    1.86×    1.73×     1.31×
  any(F)      12.48×    8.48×    2.21×    1.19×    1.03×     3.10×
  any(hit)    11.80×    8.69×    8.59×    4.58×    7.84×     7.94×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.43×    0.16×    0.30×    2.61×    1.09×     0.57×
  astype       0.27×    0.24×    1.13×    3.01×    1.98×     0.85×
  ravel.T      0.34×    0.72×    0.72×    1.72×    0.96×     0.78×
  in-place     0.78×    0.94×    1.00×    2.08×    1.00×     1.09×
  less->b      0.42×    0.74×    0.34×    1.54×    0.76×     0.66×
-- index-math
  unravel      0.26×    0.42×    0.83×    0.94×    0.97×     0.61×
  ravel_mi     0.13×    0.64×    0.88×    0.92×    1.27×     0.61×
-- dtypes
  complex      0.89×    0.59×    0.87×    0.63×    0.77×     0.74×
  float16      0.83×    0.32×    0.58×    0.27×    0.62×     0.48×
  int8         0.70×    1.60×   10.24×   10.82×    1.78×     2.95×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ██████████████████▋    1.87×    53%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.38×    23%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   4.77×    21%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.51×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.48×    40%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.07×    20%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   2.66×    38%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.20×    46%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.08×    32%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   3.16×    32%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          █████████▏ .........   0.92×   108%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████████▋ ....   1.47×    68%🕐  (  1 win /  0 lose)
w=64         █████████████▊ .....   1.38×    72%🕐  (  1 win /  0 lose)
w=256        █████████████████▎     1.73×    58%🕐  (  1 win /  0 lose)
w=1024       ███████████████████▶   2.07×    48%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    530.73×   (530.7× faster, faster)
  allocate          1.00×   (1.0× slower, parity)
  overlap_copy      1.76×   (1.8× faster, faster)
  forder_out        1.34×   (1.3× faster, faster)
  zerodim           1.51×   (1.5× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.85×    3.36×    1.55×    2.34×    2.15×   vs chained 6× add
reuse            6.05×    5.09×    1.21×    1.97×    1.30×   vs rebuild each call
par8                 -    0.75×    3.22×    5.69×    5.27×   vs single-thread

biggest NumSharp wins: anyff@1 12.48× · anyeh@1 11.80× · i8@1M 10.82× · i8@100K 10.24× · anyeh@1K 8.69×
most behind:           ravelmi@1 0.13× · flatten@1K 0.16× · astype@1K 0.24× · unravel@1 0.26× · f16@1M 0.27×
```
