```
NumSharp NDIter — canonical benchmark · 2026-06-29 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (35 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: NA across selection.

HEADLINE — operation matrix: 1.20× geomean · 83%🕐 of NumPy's time · 77 win / 53 lose over 130 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ████████████▌ ......   1.26×    79%🕐  ( 17 win /  9 lose)
1K         ███████████▌ .......   1.16×    86%🕐  ( 15 win / 11 lose)
100K       ██████████▋ ........   1.07×    94%🕐  ( 12 win / 14 lose)
1M         ████████████▉ ......   1.30×    77%🕐  ( 17 win /  9 lose)
10M        ████████████▎ ......   1.23×    82%🕐  ( 16 win / 10 lose)
ALL        ███████████▉ .......   1.20×    83%🕐  ( 77 win / 53 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise████████████▊ ......   1.28×    78%🕐  ( 31 win /  9 lose)
reductions █████████████████▍     1.74×    57%🕐  ( 29 win / 11 lose)
selection  (no data)
copy/cast  ███████▎ ...........   0.73×   138%🕐  (  9 win / 16 lose)  ◄ SLOWER
index-math ███████▋ ...........   0.77×   130%🕐  (  4 win /  6 lose)  ◄ SLOWER
dtypes     ███████████▌ .......   1.16×    86%🕐  (  4 win / 11 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.65×    1.54×    1.14×    1.10×    1.08×
reductions      2.68×    2.00×    1.51×    1.43×    1.38×
selection           -        -        -        -        -
copy/cast       0.59×    0.53×    0.41×    1.40×    1.12×
index-math      0.34×    0.51×    0.99×    1.23×    1.26×
dtypes          0.70×    0.81×    1.87×    1.39×    1.44×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.61×    1.43×    0.96×    1.00×    1.02×     1.18×
  sqrt         1.67×    1.16×    1.01×    1.00×    1.01×     1.15×
  copy         1.67×    2.35×    1.62×    1.39×    1.64×     1.71×
  strided      1.65×    1.28×    0.93×    1.01×    0.99×     1.15×
  bcast        1.68×    1.32×    0.89×    0.95×    0.95×     1.12×
  reversed     1.65×    1.26×    0.93×    0.99×    0.95×     1.13×
  castbuf      1.87×    2.20×    1.63×    1.36×    1.12×     1.59×
  mixbuf       1.42×    1.72×    1.41×    1.20×    1.07×     1.35×
-- reductions
  sum          1.92×    1.85×    2.58×    1.76×    1.60×     1.92×
  sum ax0      1.71×    0.86×    1.10×    0.96×    0.96×     1.09×
  sum ax1      1.81×    0.92×    1.52×    1.79×    1.58×     1.48×
  sum dt=      1.89×    1.35×    0.48×    0.46×    0.54×     0.79×
  amin         1.69×    1.62×    0.71×    0.71×    0.76×     1.01×
  cumsum       1.35×    1.13×    1.07×    1.87×    1.65×     1.38×
  any(F)      10.04×    8.39×    2.00×    1.23×    1.00×     2.90×
  any(hit)    10.25×    8.49×    8.50×    7.88×    7.98×     8.58×
-- selection
  where           NA       NA       NA       NA       NA
  a[mask]         NA       NA       NA       NA       NA
  a[mask]=        NA       NA       NA       NA       NA
  count_nz        NA       NA       NA       NA       NA
  argwhere        NA       NA       NA       NA       NA
  a[idx]          NA       NA       NA       NA       NA
  a[idx]=         NA       NA       NA       NA       NA
-- copy/cast
  flatten      0.41×    0.33×    0.17×    2.21×    1.13×     0.56×
  astype       0.31×    0.53×    0.54×    1.94×    1.89×     0.80×
  ravel.T      0.50×    0.58×    0.53×    2.22×    1.09×     0.82×
  in-place     1.45×    0.77×    0.96×    1.03×    1.04×     1.03×
  less->b      0.81×    0.52×    0.25×    0.55×    0.75×     0.53×
-- index-math
  unravel      0.36×    0.50×    0.96×    1.00×    1.04×     0.71×
  ravel_mi     0.32×    0.53×    1.01×    1.53×    1.54×     0.83×
-- dtypes
  complex      0.71×    0.58×    0.97×    0.77×    0.93×     0.78×
  float16      0.72×    0.64×    0.58×    0.56×    0.57×     0.61×
  int8         0.66×    1.43×   11.53×    6.16×    5.66×     3.28×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          █████████▋ .........   0.97×   103%🕐  (  0 win /  1 lose)  ◄ SLOWER
3op_exl      ███████████████████▶   2.44×    41%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   2.91×    34%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   2.56×    39%🕐  (  1 win /  0 lose)
multiindex   █████████████▉ .....   1.40×    72%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   3.67×    27%🕐  (  1 win /  0 lose)
4d           █████████████████▊     1.78×    56%🕐  (  1 win /  0 lose)
8d           ██████████████████▌    1.85×    54%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▌   1.95×    51%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   2.03×    49%🕐  (  8 win /  1 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ██████▉ ............   0.69×   145%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         █████████▊ .........   0.98×   102%🕐  (  0 win /  1 lose)  ◄ PARITY
w=64         ██████████▏ ........   1.02×    98%🕐  (  1 win /  0 lose)  ◄ PARITY
w=256        ██████████████▌ ....   1.45×    69%🕐  (  1 win /  0 lose)
w=1024       █████████████ ......   1.31×    76%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    522.42×   (522.4× faster, faster)
  allocate          1.06×   (1.1× faster, faster)
  overlap_copy      1.77×   (1.8× faster, faster)
  forder_out        1.17×   (1.2× faster, faster)
  zerodim           1.55×   (1.6× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.55×    3.86×    1.44×    1.67×    2.09×   vs chained 6× add
reuse            6.06×    6.12×    1.07×    1.00×    1.00×   vs rebuild each call
par8                 -    0.67×    2.78×    3.95×    5.40×   vs single-thread

biggest NumSharp wins: i8@100K 11.53× · anyeh@1 10.25× · anyff@1 10.04× · anyeh@100K 8.50× · anyeh@1K 8.49×
most behind:           flatten@100K 0.17× · lessbool@100K 0.25× · astype@1 0.31× · ravelmi@1 0.32× · flatten@1K 0.33×
```
