```
NumSharp NpyIter — canonical benchmark · 2026-06-13 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
162 measured NumSharp-vs-NumPy pairs · best-of-rounds, Release · matched kernels/ids

HEADLINE — operation matrix: 1.24× geomean, 80 win / 52 lose over 132 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ███████████▉ .......   1.20×   ( 20 win / 13 lose)
1K         █████████████▏ .....   1.32×   ( 20 win / 13 lose)
100K       ███████████▍ .......   1.14×   ( 18 win / 15 lose)
1M         █████████████▏ .....   1.32×   ( 22 win / 11 lose)
ALL        ████████████▍ ......   1.24×   ( 80 win / 52 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise██████████████▊ ....   1.48×   ( 27 win /  5 lose)
reductions ███████████████████▶   2.03×   ( 27 win /  5 lose)
selection  █████████████▏ .....   1.32×   ( 16 win / 12 lose)
copy/cast  █████▊ .............   0.58×   (  5 win / 15 lose)   ◄ SLOWER
index-math ██████▎ ............   0.63×   (  3 win /  5 lose)   ◄ SLOWER
dtypes     ██████████▎ ........   1.03×   (  2 win / 10 lose)   ◄ PARITY

CATEGORY × TIER geomean
category       scalar       1K     100K       1M
elementwise     1.64×    2.37×    1.09×    1.14×
reductions      4.31×    2.89×    1.30×    1.04×
selection       0.90×    1.24×    1.45×    1.85×
copy/cast       0.40×    0.35×    0.57×    1.43×
index-math      0.22×    0.57×    1.18×    1.09×
dtypes          0.65×    0.64×    1.69×    1.60×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M    geomean
-- elementwise
  add          1.59×    1.32×    0.86×    1.16×     1.20×
  sqrt         1.57×    2.63×    0.99×    1.00×     1.42×
  copy         1.65×    4.15×    1.19×    0.96×     1.67×
  strided      1.60×    2.56×    0.85×    1.02×     1.37×
  bcast        1.61×    2.04×    1.04×    1.08×     1.39×
  reversed     1.62×    1.31×    0.80×    1.04×     1.15×
  castbuf      1.99×    3.32×    1.76×    1.51×     2.05×
  mixbuf       1.53×    3.00×    1.58×    1.48×     1.81×
-- reductions
  sum          1.68×    2.19×    2.79×    2.16×     2.17×
  sum ax0      1.67×    2.60×    1.12×    0.98×     1.48×
  sum ax1      4.17×    2.51×    2.50×    1.89×     2.66×
  sum dt=      2.96×    2.97×    1.14×    1.08×     1.81×
  amin         2.36×    1.55×    0.48×    0.41×     0.92×
  cumsum       2.14×    1.87×    1.99×    1.87×     1.97×
  any(F)      25.09×    6.13×    0.15×    0.07×     1.14×
  any(hit)    27.08×    6.42×    6.16×    6.09×     8.98×
-- selection
  where        0.67×    0.78×    0.85×    1.33×     0.88×
  a[mask]      0.38×    1.47×    3.72×    4.50×     1.75×
  a[mask]=     2.74×    1.09×    2.39×    2.19×     1.99×
  count_nz     3.20×    4.08×    4.30×    3.22×     3.67×
  argwhere     1.16×    2.31×    1.08×    3.07×     1.72×
  a[idx]       0.25×    0.50×    0.58×    0.75×     0.48×
  a[idx]=      0.77×    0.77×    0.66×    0.77×     0.74×
-- copy/cast
  flatten      0.39×    0.16×    0.25×    2.67×     0.45×
  astype       0.25×    0.20×    0.94×    2.45×     0.58×
  ravel.T      0.22×    0.29×    0.67×    1.72×     0.52×
  in-place     1.04×    0.86×    1.02×    0.90×     0.95×
  less->b      0.46×    0.68×    0.37×    0.59×     0.51×
-- index-math
  unravel      0.30×    0.52×    1.19×    1.00×     0.66×
  ravel_mi     0.16×    0.62×    1.18×    1.18×     0.61×
-- dtypes
  complex      0.74×    0.60×    0.91×    0.65×     0.72×
  float16      0.55×    0.49×    0.59×    0.59×     0.56×
  int8         0.67×    0.88×    8.95×   10.75×     2.74×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████████████▉   1.99×   (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.59×   (  1 win /  0 lose)
ufunc        ███████████████████▶   5.07×   (  1 win /  0 lose)
bufcast      ███████████████████▶   8.19×   (  1 win /  0 lose)
multiindex   ███████████████████▶   8.20×   (  1 win /  0 lose)
8op          ███████████████████▶  12.55×   (  1 win /  0 lose)
4d           ███████████████████▶   6.78×   (  1 win /  0 lose)
8d           ███████████████████▶   5.37×   (  1 win /  0 lose)
strided2d    ███████████████████▶   9.33×   (  1 win /  0 lose)
geomean      ███████████████████▶   6.19×   (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ███████▍ ...........   0.74×   (  0 win /  1 lose)   ◄ SLOWER
w=16         █████████▉ .........   0.99×   (  0 win /  1 lose)   ◄ PARITY
w=64         ██████████▊ ........   1.09×   (  1 win /  0 lose)
w=256        ████████████▉ ......   1.29×   (  1 win /  0 lose)
w=1024       ██████████████ .....   1.41×   (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce      0.02×   (51.4× slower, SLOWER)
  allocate          0.49×   (2.0× slower, SLOWER)
  overlap_copy      0.70×   (1.4× slower, SLOWER)
  forder_out        0.29×   (3.5× slower, SLOWER)
  zerodim           0.90×   (1.1× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M   note
fuse7           15.45×    7.71×    2.74×    4.57×   vs chained 6× add
reuse            8.19×    9.13×    1.20×    0.97×   vs rebuild each call
par8                 -    0.77×    7.43×    2.51×   vs single-thread

biggest NumSharp wins: anyeh@1 27.08× · anyff@1 25.09× · i8@1M 10.75× · i8@100K 8.95× · anyeh@1K 6.42×
most behind:           anyff@1M 0.07× · anyff@100K 0.15× · ravelmi@1 0.16× · flatten@1K 0.16× · astype@1K 0.20×
```
