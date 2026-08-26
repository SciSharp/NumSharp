```
NumSharp NDIter — canonical benchmark · 2026-08-25 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (0 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: none.

HEADLINE — operation matrix: 1.45× geomean · 69%🕐 of NumPy's time · 106 win / 59 lose over 165 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ███████████████▎ ...   1.53×    65%🕐  ( 20 win / 13 lose)
1K         ███████████████▎ ...   1.53×    66%🕐  ( 21 win / 12 lose)
100K       ████████████▎ ......   1.24×    81%🕐  ( 17 win / 16 lose)
1M         ███████████████▋ ...   1.57×    64%🕐  ( 22 win / 11 lose)
10M        ██████████████ .....   1.41×    71%🕐  ( 26 win /  7 lose)
ALL        ██████████████▌ ....   1.45×    69%🕐  (106 win / 59 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise██████████████▏ ....   1.42×    70%🕐  ( 35 win /  5 lose)
reductions ███████████████████▶   2.92×    34%🕐  ( 34 win /  6 lose)
selection  ██████████████ .....   1.41×    71%🕐  ( 22 win / 13 lose)
copy/cast  ███████▏ ...........   0.72×   138%🕐  (  8 win / 17 lose)  ◄ SLOWER
index-math ████████▌ ..........   0.86×   117%🕐  (  4 win /  6 lose)  ◄ SLOWER
dtypes     ███████████▍ .......   1.14×    88%🕐  (  3 win / 12 lose)

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.67×    2.51×    1.18×    1.09×    1.10×
reductions      6.93×    4.46×    1.79×    1.93×    1.98×
selection       0.79×    1.43×    1.40×    2.04×    1.73×
copy/cast       0.72×    0.30×    0.64×    1.39×    1.04×
index-math      0.54×    0.67×    0.94×    1.16×    1.17×
dtypes          0.73×    0.70×    1.42×    2.01×    1.33×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          1.54×    1.39×    1.01×    0.94×    1.02×     1.16×
  sqrt         1.54×    1.16×    1.02×    1.02×    1.04×     1.14×
  copy         1.72×    4.92×    1.40×    1.07×    1.68×     1.84×
  strided      1.71×    2.46×    0.87×    1.02×    0.86×     1.26×
  bcast        1.68×    3.26×    1.07×    1.01×    1.05×     1.44×
  reversed     1.70×    2.57×    0.83×    0.97×    1.07×     1.30×
  castbuf      2.06×    3.36×    2.11×    1.38×    1.09×     1.86×
  mixbuf       1.44×    2.89×    1.57×    1.35×    1.13×     1.58×
-- reductions
  sum          2.81×    4.62×    3.24×    2.02×    1.98×     2.78×
  sum ax0      2.74×    1.78×    0.47×    0.97×    1.01×     1.17×
  sum ax1      8.02×    1.90×    0.72×    1.40×    1.74×     1.92×
  sum dt=      8.08×    4.12×    1.12×    1.10×    1.12×     2.15×
  amin         6.42×    2.77×    0.81×    0.76×    0.95×     1.60×
  cumsum       3.09×    1.36×    1.23×    1.95×    1.75×     1.77×
  any(F)      19.74×   26.21×    3.52×    1.92×    1.48×     5.53×
  any(hit)    27.40×   24.85×   24.72×   22.35×   24.12×    24.63×
-- selection
  where        0.96×    0.70×    0.88×    1.97×    1.06×     1.05×
  a[mask]      0.30×    1.51×    2.13×    2.90×    2.23×     1.44×
  a[mask]=     0.50×    3.23×    6.30×    6.20×    5.77×     3.25×
  count_nz     1.52×    2.95×    4.10×    3.32×    1.09×     2.32×
  argwhere     1.57×    3.56×    1.16×    3.01×    3.48×     2.32×
  a[idx]       0.32×    0.49×    0.40×    0.67×    1.02×     0.53×
  a[idx]=      1.78×    0.68×    0.47×    0.62×    0.86×     0.79×
-- copy/cast
  flatten      0.57×    0.11×    0.31×    2.27×    1.03×     0.54×
  astype       0.40×    0.25×    1.32×    2.14×    1.25×     0.81×
  ravel.T      0.79×    0.32×    0.66×    1.68×    0.93×     0.77×
  in-place     1.71×    0.41×    1.00×    0.98×    1.02×     0.93×
  less->b      0.65×    0.66×    0.38×    0.65×    0.98×     0.63×
-- index-math
  unravel      0.52×    0.44×    0.90×    0.97×    1.05×     0.73×
  ravel_mi     0.55×    1.03×    0.99×    1.37×    1.31×     1.00×
-- dtypes
  complex      0.75×    0.61×    0.74×    0.92×    0.92×     0.78×
  float16      0.74×    0.65×    0.54×    0.61×    0.60×     0.62×
  int8         0.70×    0.86×    7.23×   14.58×    4.25×     3.06×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████████████▏   1.92×    52%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.54×    22%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   5.08×    20%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.69×    27%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   6.78×    15%🕐  (  1 win /  0 lose)
8op          ███████████████████▶  13.26×     8%🕐  (  1 win /  0 lose)
4d           ███████████████████▶   5.49×    18%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   6.04×    17%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.26×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   4.89×    20%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ██████▌ ............   0.66×   152%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         █████████▏ .........   0.92×   109%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=64         ██████████▋ ........   1.07×    93%🕐  (  1 win /  0 lose)
w=256        ██████████▊ ........   1.09×    92%🕐  (  1 win /  0 lose)
w=1024       █████████████▌ .....   1.36×    74%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    640.95×   (641.0× faster, faster)
  allocate          1.08×   (1.1× faster, faster)
  overlap_copy      1.91×   (1.9× faster, faster)
  forder_out        1.62×   (1.6× faster, faster)
  zerodim           1.46×   (1.5× faster, faster)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.77×    7.45×    2.70×    4.75×    1.92×   vs chained 6× add
reuse            8.64×   13.49×    1.08×    1.93×    1.03×   vs rebuild each call
par8                 -    1.54×    6.81×    6.88×    5.19×   vs single-thread

biggest NumSharp wins: anyeh@1 27.40× · anyff@1K 26.21× · anyeh@1K 24.85× · anyeh@100K 24.72× · anyeh@10M 24.12×
most behind:           flatten@1K 0.11× · astype@1K 0.25× · bread@1 0.30× · flatten@100K 0.31× · gather@1 0.32×
```
