```
NumSharp NDIter — canonical benchmark · 2026-08-28 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (0 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: none.

HEADLINE — operation matrix: 1.49× geomean · 67%🕐 of NumPy's time · 114 win / 51 lose over 165 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████████▏ ....   1.42×    70%🕐  ( 19 win / 14 lose)
1K         ███████████████▌ ...   1.56×    64%🕐  ( 24 win /  9 lose)
100K       ███████████████▌ ...   1.56×    64%🕐  ( 24 win /  9 lose)
1M         ████████████████▏ ..   1.62×    62%🕐  ( 23 win / 10 lose)
10M        █████████████▏ .....   1.32×    76%🕐  ( 24 win /  9 lose)
ALL        ██████████████▉ ....   1.49×    67%🕐  (114 win / 51 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise████████████▏ ......   1.22×    82%🕐  ( 35 win /  5 lose)
reductions ███████████████████▶   2.78×    36%🕐  ( 35 win /  5 lose)
selection  ███████████████▎ ...   1.53×    65%🕐  ( 20 win / 15 lose)
copy/cast  ██████████▎ ........   1.03×    97%🕐  ( 13 win / 12 lose)  ◄ PARITY
index-math ██████████▋ ........   1.06×    94%🕐  (  7 win /  3 lose)
dtypes     ██████████▋ ........   1.07×    93%🕐  (  4 win / 11 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.33×    1.51×    1.29×    1.10×    0.95×
reductions      7.14×    2.83×    2.31×    1.86×    1.92×
selection       0.64×    1.99×    2.17×    2.36×    1.30×
copy/cast       0.86×    0.80×    0.70×    1.79×    1.32×
index-math      0.87×    1.10×    1.07×    1.09×    1.21×
dtypes          0.48×    0.76×    1.99×    1.48×    1.32×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          0.99×    1.27×    1.16×    0.94×    1.01×     1.07×
  sqrt         1.01×    1.15×    1.00×    1.00×    0.23×     0.77×
  copy         1.50×    2.23×    1.25×    1.09×    1.77×     1.52×
  strided      1.56×    1.27×    1.07×    0.96×    1.02×     1.16×
  bcast        1.58×    1.39×    1.01×    1.02×    1.04×     1.18×
  reversed     1.54×    1.23×    1.55×    0.97×    1.37×     1.32×
  castbuf      1.29×    1.34×    1.89×    1.57×    1.07×     1.40×
  mixbuf       1.33×    2.84×    1.71×    1.44×    1.08×     1.59×
-- reductions
  sum          4.43×    3.53×    3.49×    2.08×    1.92×     2.94×
  sum ax0      8.84×    0.50×    1.06×    1.07×    0.97×     1.37×
  sum ax1      8.94×    1.17×    1.47×    1.59×    1.74×     2.12×
  sum dt=      4.75×    1.66×    1.10×    1.05×    1.09×     1.58×
  amin         8.83×    1.31×    0.80×    0.73×    0.92×     1.44×
  cumsum       3.11×    1.61×    1.96×    1.14×    1.60×     1.78×
  any(F)      12.15×   23.88×    3.47×    1.86×    1.44×     4.86×
  any(hit)    12.18×   23.89×   24.27×   24.25×   24.07×    21.04×
-- selection
  where        0.96×    1.25×    1.64×    1.93×    0.82×     1.25×
  a[mask]      0.26×    2.90×    4.52×    5.48×    1.85×     2.02×
  a[mask]=     0.22×    4.50×    7.78×    7.71×    3.89×     2.97×
  count_nz     0.67×    3.11×    4.05×    3.41×    1.03×     1.97×
  argwhere     3.36×    6.13×    2.09×    3.81×    2.85×     3.42×
  a[idx]       0.41×    0.76×    0.94×    0.65×    0.72×     0.67×
  a[idx]=      0.88×    0.52×    0.50×    0.59×    0.49×     0.58×
-- copy/cast
  flatten      1.11×    1.03×    0.58×    3.18×    0.86×     1.12×
  astype       0.32×    0.60×    1.60×    2.95×    1.54×     1.07×
  ravel.T      2.18×    1.65×    0.84×    1.81×    1.22×     1.46×
  in-place     0.77×    0.62×    0.87×    1.72×    1.03×     0.94×
  less->b      0.78×    0.52×    0.25×    0.63×    2.42×     0.69×
-- index-math
  unravel      0.55×    0.86×    1.03×    0.91×    1.06×     0.86×
  ravel_mi     1.37×    1.42×    1.11×    1.30×    1.39×     1.31×
-- dtypes
  complex      0.34×    0.52×    0.90×    0.84×    0.86×     0.65×
  float16      0.57×    0.61×    0.64×    0.64×    0.58×     0.61×
  int8         0.56×    1.41×   13.86×    6.11×    4.62×     3.15×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████▋ .......   1.17×    86%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.30×    23%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   3.35×    30%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.46×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.07×    48%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.31×    19%🕐  (  1 win /  0 lose)
4d           ██████████████████▊    1.89×    53%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.25×    44%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.24×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   2.74×    36%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ████▍ ..............   0.44×   228%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████ .........   1.01×    99%🕐  (  1 win /  0 lose)  ◄ PARITY
w=64         ███████████▎ .......   1.13×    88%🕐  (  1 win /  0 lose)
w=256        ████████████▎ ......   1.23×    81%🕐  (  1 win /  0 lose)
w=1024       █████████████▊ .....   1.38×    72%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    839.13×   (839.1× faster, faster)
  allocate          1.09×   (1.1× faster, faster)
  overlap_copy      1.91×   (1.9× faster, faster)
  forder_out        1.14×   (1.1× faster, faster)
  zerodim           0.62×   (1.6× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           14.84×    3.94×    1.41×    1.69×    2.02×   vs chained 6× add
reuse            5.68×    4.91×    1.04×    1.07×    1.03×   vs rebuild each call
par8                 -    0.78×    3.19×    4.32×    2.94×   vs single-thread

biggest NumSharp wins: anyeh@100K 24.27× · anyeh@1M 24.25× · anyeh@10M 24.07× · anyeh@1K 23.89× · anyff@1K 23.88×
most behind:           bassign@1 0.22× · sqrt@10M 0.23× · lessbool@100K 0.25× · bread@1 0.26× · astype@1 0.32×
```
