```
NumSharp NpyIter — canonical benchmark · 2026-06-13 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
162 measured NumSharp-vs-NumPy pairs · best-of-rounds, Release · matched kernels/ids

HEADLINE — operation matrix: 1.19× geomean, 76 win / 56 lose over 132 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ████████████ .......   1.21×   ( 20 win / 13 lose)
1K         █████████████▏ .....   1.32×   ( 20 win / 13 lose)
100K       ██████████▎ ........   1.04×   ( 16 win / 17 lose)
1M         ████████████▏ ......   1.21×   ( 20 win / 13 lose)
ALL        ███████████▉ .......   1.19×   ( 76 win / 56 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise█████████████▍ .....   1.35×   ( 23 win /  9 lose)
reductions █████████████████▌     1.75×   ( 28 win /  4 lose)
selection  █████████████▍ .....   1.35×   ( 16 win / 12 lose)
copy/cast  █████▉ .............   0.59×   (  4 win / 16 lose)   ◄ SLOWER
index-math ██████▍ ............   0.64×   (  3 win /  5 lose)   ◄ SLOWER
dtypes     ███████████ ........   1.11×   (  2 win / 10 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M
elementwise     1.70×    2.48×    0.79×    0.98×
reductions      3.69×    2.57×    1.13×    0.87×
selection       0.94×    1.27×    1.55×    1.77×
copy/cast       0.44×    0.38×    0.53×    1.41×
index-math      0.24×    0.58×    1.15×    1.04×
dtypes          0.70×    0.64×    1.87×    1.83×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M    geomean
-- elementwise
  add          1.66×    1.48×    0.81×    0.87×     1.14×
  sqrt         1.62×    3.67×    0.43×    0.76×     1.18×
  copy         1.67×    4.34×    1.07×    1.15×     1.73×
  strided      1.65×    2.73×    0.89×    1.02×     1.42×
  bcast        1.65×    2.28×    0.46×    1.10×     1.17×
  reversed     1.66×    1.32×    0.84×    0.97×     1.16×
  castbuf      2.17×    2.95×    1.07×    1.25×     1.71×
  mixbuf       1.59×    2.50×    1.18×    0.83×     1.40×
-- reductions
  sum          1.84×    2.83×    3.08×    2.24×     2.45×
  sum ax0      1.77×    2.08×    1.18×    1.01×     1.45×
  sum ax1      3.92×    1.60×    2.58×    2.29×     2.47×
  sum dt=      2.39×    2.23×    1.16×    1.07×     1.61×
  amin         2.63×    1.29×    0.34×    0.47×     0.86×
  cumsum       1.86×    1.65×    2.04×    2.20×     1.92×
  any(F)      12.77×    7.07×    0.16×    0.03×     0.79×
  any(hit)    18.13×    5.97×    2.28×    2.20×     4.83×
-- selection
  where        0.64×    0.85×    0.79×    1.29×     0.86×
  a[mask]      0.36×    1.41×    3.74×    3.62×     1.62×
  a[mask]=     2.74×    1.26×    2.33×    2.15×     2.04×
  count_nz     3.21×    4.25×    5.32×    2.67×     3.73×
  argwhere     1.10×    2.47×    1.75×    3.03×     1.95×
  a[idx]       0.35×    0.43×    0.52×    0.78×     0.50×
  a[idx]=      0.81×    0.78×    0.65×    0.87×     0.77×
-- copy/cast
  flatten      0.36×    0.17×    0.25×    2.49×     0.44×
  astype       0.28×    0.24×    0.91×    2.13×     0.60×
  ravel.T      0.31×    0.28×    0.71×    1.72×     0.57×
  in-place     1.08×    0.97×    0.72×    0.93×     0.92×
  less->b      0.48×    0.71×    0.35×    0.66×     0.53×
-- index-math
  unravel      0.31×    0.51×    1.19×    0.89×     0.64×
  ravel_mi     0.19×    0.66×    1.11×    1.21×     0.64×
-- dtypes
  complex      0.70×    0.60×    0.85×    0.88×     0.75×
  float16      0.71×    0.48×    0.61×    0.62×     0.60×
  int8         0.68×    0.90×   12.59×   11.29×     3.05×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████████████▊   1.98×   (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.67×   (  1 win /  0 lose)
ufunc        ███████████████████▶   5.08×   (  1 win /  0 lose)
bufcast      ███████████████████▶   9.32×   (  1 win /  0 lose)
multiindex   ███████████████████▶   7.48×   (  1 win /  0 lose)
8op          ███████████████████▶  11.49×   (  1 win /  0 lose)
4d           ███████████████████▶   6.72×   (  1 win /  0 lose)
8d           ███████████████████▶   6.26×   (  1 win /  0 lose)
strided2d    ███████████████████▶   4.27×   (  1 win /  0 lose)
geomean      ███████████████████▶   5.74×   (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ███████▋ ...........   0.77×   (  0 win /  1 lose)   ◄ SLOWER
w=16         ██████████▏ ........   1.02×   (  1 win /  0 lose)   ◄ PARITY
w=64         ███████████▋ .......   1.17×   (  1 win /  0 lose)
w=256        ████████████▉ ......   1.29×   (  1 win /  0 lose)
w=1024       █████████████████▏     1.72×   (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce      0.02×   (52.0× slower, SLOWER)
  allocate          0.53×   (1.9× slower, SLOWER)
  overlap_copy      0.80×   (1.3× slower, SLOWER)
  forder_out        1.19×   (1.2× faster, faster)
  zerodim           0.69×   (1.4× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M   note
fuse7           12.70×    7.80×    2.54×    3.96×   vs chained 6× add
reuse            7.74×   11.14×    1.21×    1.00×   vs rebuild each call
par8                 -    0.90×    7.00×    3.77×   vs single-thread

biggest NumSharp wins: anyeh@1 18.13× · anyff@1 12.77× · i8@100K 12.59× · i8@1M 11.29× · anyff@1K 7.07×
most behind:           anyff@1M 0.03× · anyff@100K 0.16× · flatten@1K 0.17× · ravelmi@1 0.19× · astype@1K 0.24×
```
