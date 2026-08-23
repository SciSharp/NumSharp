```
NumSharp NDIter — canonical benchmark · 2026-08-22 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.32× geomean · 76%🕐 of NumPy's time · 80 win / 50 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ████████████▋ ......   1.27×    78%🕐  ( 16 win / 10 lose)
1K         █████████████▊ .....   1.38×    73%🕐  ( 16 win / 10 lose)
100K       ████████████▋ ......   1.27×    79%🕐  ( 14 win / 12 lose)
1M         ██████████████ .....   1.41×    71%🕐  ( 18 win /  8 lose)
10M        ████████████▌ ......   1.25×    80%🕐  ( 16 win / 10 lose)
ALL        █████████████▏ .....   1.32×    76%🕐  ( 80 win / 50 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████████ ....   1.51×    66%🕐  ( 34 win /  6 lose)
reductions ███████████████████▶   2.42×    41%🕐  ( 34 win /  6 lose)
selection  (no data)
copy/cast  ██████ .............   0.61×   164%🕐  (  5 win / 20 lose)  ◄ SLOWER
index-math ██████▎ ............   0.64×   157%🕐  (  4 win /  6 lose)  ◄ SLOWER
dtypes     ██████████▌ ........   1.06×    94%🕐  (  3 win / 12 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.80×    2.37×    1.24×    1.38×    1.07×
reductions      3.85×    3.29×    2.09×    1.84×    1.70×
selection           -        -        -        -        -
copy/cast       0.42×    0.36×    0.57×    0.94×    1.04×
index-math      0.17×    0.48×    0.99×    1.14×    1.12×
dtypes          0.65×    0.62×    1.58×    1.70×    1.25×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.64×    2.86×    1.14×    1.73×    1.05×     1.58×
  sqrt         1.62×    4.88×    1.04×    1.03×    1.01×     1.53×
  copy         1.58×    2.16×    1.39×    1.68×    1.70×     1.68×
  strided      1.49×    1.28×    1.10×    0.92×    0.98×     1.13×
  bcast        1.51×    2.36×    1.13×    1.64×    0.96×     1.45×
  reversed     1.53×    1.29×    0.84×    1.37×    0.98×     1.17×
  castbuf      1.93×    3.13×    1.83×    1.67×    0.98×     1.79×
  mixbuf       3.91×    2.75×    1.77×    1.27×    1.04×     1.91×
-- reductions
  sum          1.67×    2.33×    2.98×    2.04×    1.69×     2.09×
  sum ax0      1.66×    1.26×    0.98×    0.99×    0.87×     1.12×
  sum ax1      4.51×    1.20×    1.33×    1.46×    1.51×     1.74×
  sum dt=      2.04×    2.46×    1.12×    1.05×    1.02×     1.43×
  amin         2.24×    1.88×    0.80×    0.73×    0.75×     1.13×
  cumsum       1.56×    1.46×    1.23×    1.66×    1.43×     1.46×
  any(F)      20.94×   24.01×    3.48×    1.66×    1.35×     5.23×
  any(hit)    25.91×   24.34×   23.92×   20.94×   21.59×    23.27×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.40×    0.16×    0.26×    2.22×    1.05×     0.52×
  astype       0.26×    0.21×    0.94×    0.94×    1.37×     0.58×
  ravel.T      0.28×    0.26×    0.69×    1.08×    0.99×     0.56×
  in-place     0.91×    0.90×    0.94×    0.57×    1.01×     0.85×
  less->b      0.49×    0.74×    0.39×    0.58×    0.86×     0.59×
-- index-math
  unravel      0.26×    0.38×    1.00×    1.02×    1.08×     0.64×
  ravel_mi     0.11×    0.60×    0.98×    1.28×    1.16×     0.63×
-- dtypes
  complex      0.66×    0.54×    0.78×    0.89×    0.92×     0.75×
  float16      0.67×    0.53×    0.60×    0.62×    0.63×     0.61×
  int8         0.62×    0.81×    8.34×    8.89×    3.33×     2.62×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          █████████▌ .........   0.96×   104%🕐  (  0 win /  1 lose)  ◄ SLOWER
3op_exl      ███████████████████▶   2.60×    38%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   2.99×    33%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   7.22×    14%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   4.15×    24%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   4.04×    25%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   2.79×    36%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.39×    42%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.26×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   3.00×    33%🕐  (  8 win /  1 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ███████▍ ...........   0.74×   135%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         █████████▊ .........   0.98×   102%🕐  (  0 win /  1 lose)  ◄ PARITY
w=64         ██████████▊ ........   1.09×    92%🕐  (  1 win /  0 lose)
w=256        █████████████▌ .....   1.35×    74%🕐  (  1 win /  0 lose)
w=1024       █████████████████▍     1.74×    58%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    510.72×   (510.7× faster, faster)
  allocate          1.06×   (1.1× faster, faster)
  overlap_copy      1.89×   (1.9× faster, faster)
  forder_out        1.32×   (1.3× faster, faster)
  zerodim           1.47×   (1.5× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           11.46×    6.17×    2.68×    4.60×    1.81×   vs chained 6× add
reuse            8.44×   10.45×    2.01×    1.27×    1.05×   vs rebuild each call
par8                 -    1.58×    7.20×    6.16×    6.77×   vs single-thread

biggest NumSharp wins: anyeh@1 25.91× · anyeh@1K 24.34× · anyff@1K 24.01× · anyeh@100K 23.92× · anyeh@10M 21.59×
most behind:           ravelmi@1 0.11× · flatten@1K 0.16× · astype@1K 0.21× · astype@1 0.26× · unravel@1 0.26×
```
