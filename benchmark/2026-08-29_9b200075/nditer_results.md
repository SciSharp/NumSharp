```
NumSharp NDIter — canonical benchmark · 2026-08-29 · speedup = NumPy ÷ NumSharp (>1.0× = NumSharp faster)
198 measured pairs (0 NA) · best-of-rounds, Release · matched kernels/ids
%NumPy🕐 = NumSharp ÷ NumPy × 100 = share of NumPy's time NumSharp uses (8% = takes only 8% as long; <100% = faster)

AV POLICY — a NumSharp section that crashes all retries (known intermittent
AccessViolation, an unmanaged-storage lifetime bug) is reported NA / IGNORED
and excluded from every geomean below.  THIS RUN: none.

HEADLINE — operation matrix: 1.46× geomean · 69%🕐 of NumPy's time · 108 win / 57 lose over 165 cells

OPERATIONS — BY SIZE TIER  (geomean over all families)
        slower ◄───────── 1.0 (parity) ─────────► faster
scalar     ██████████████▋ ....   1.46×    68%🕐  ( 20 win / 13 lose)
1K         ████████████████▌ ..   1.65×    60%🕐  ( 26 win /  7 lose)
100K       ██████████████▎ ....   1.43×    70%🕐  ( 19 win / 14 lose)
1M         ███████████████▍ ...   1.55×    65%🕐  ( 21 win / 12 lose)
10M        ████████████▎ ......   1.23×    82%🕐  ( 22 win / 11 lose)
ALL        ██████████████▌ ....   1.46×    69%🕐  (108 win / 57 lose)

OPERATIONS — BY CATEGORY  (geomean over its families, all sizes)
        slower ◄───────── 1.0 (parity) ─────────► faster
elementwise███████████▋ .......   1.16×    86%🕐  ( 28 win / 12 lose)
reductions ███████████████████▶   2.64×    38%🕐  ( 32 win /  8 lose)
selection  ████████████████▍ ..   1.65×    61%🕐  ( 22 win / 13 lose)
copy/cast  █████████▉ .........   0.99×   101%🕐  ( 16 win /  9 lose)  ◄ PARITY
index-math ██████████▎ ........   1.04×    97%🕐  (  6 win /  4 lose)
dtypes     █████████▋ .........   0.96×   104%🕐  (  4 win / 11 lose)  ◄ SLOWER

CATEGORY × TIER geomean
category       scalar       1K     100K       1M      10M
elementwise     1.38×    1.54×    1.28×    1.11×    0.71×
reductions      6.51×    3.42×    1.95×    1.55×    1.91×
selection       0.81×    1.86×    1.83×    2.60×    1.70×
copy/cast       0.80×    0.88×    0.70×    1.66×    1.19×
index-math      0.85×    1.12×    1.08×    1.23×    0.94×
dtypes          0.50×    0.81×    1.82×    1.18×    0.95×

PER-FAMILY × TIER  (NumPy ÷ NumSharp; >1.0 = NumSharp faster)
family        scalar       1K     100K       1M      10M    geomean
-- elementwise
  add          0.98×    1.53×    0.91×    1.13×    0.33×     0.87×
  sqrt         1.08×    1.16×    5.11×    1.00×    0.23×     1.08×
  copy         1.57×    2.35×    1.27×    1.15×    1.72×     1.56×
  strided      1.65×    1.33×    1.00×    0.97×    1.01×     1.17×
  bcast        1.62×    1.22×    0.98×    0.67×    0.92×     1.04×
  reversed     1.63×    1.26×    0.64×    0.65×    1.00×     0.97×
  castbuf      1.31×    1.34×    1.55×    2.59×    1.04×     1.49×
  mixbuf       1.37×    2.79×    1.25×    1.56×    0.50×     1.30×
-- reductions
  sum          3.15×    4.66×    3.54×    2.08×    1.93×     2.91×
  sum ax0      8.95×    1.13×    1.05×    0.95×    0.98×     1.58×
  sum ax1      9.46×    1.60×    0.86×    1.51×    1.62×     2.00×
  sum dt=      2.90×    1.96×    0.56×    1.05×    1.10×     1.30×
  amin         8.97×    1.23×    0.81×    0.73×    0.93×     1.43×
  cumsum       4.43×    1.68×    1.88×    2.50×    1.75×     2.28×
  any(F)       8.97×   23.27×    3.37×    0.68×    1.38×     3.66×
  any(hit)    11.71×   23.13×   23.18×    8.57×   23.53×    16.61×
-- selection
  where        0.94×    1.62×    1.67×    1.93×    1.55×     1.50×
  a[mask]      0.25×    2.91×    3.31×    4.82×    2.12×     1.90×
  a[mask]=     0.22×    4.53×    8.34×   16.46×    5.90×     3.82×
  count_nz     0.67×    2.64×    4.04×    3.60×    1.23×     2.00×
  argwhere     8.39×    6.28×    1.67×    3.90×    3.13×     4.04×
  a[idx]       0.65×    0.46×    0.46×    0.68×    0.89×     0.61×
  a[idx]=      1.18×    0.47×    0.48×    0.55×    0.62×     0.62×
-- copy/cast
  flatten      1.07×    1.00×    0.60×    3.24×    1.19×     1.20×
  astype       0.32×    1.14×    1.54×    3.28×    1.48×     1.22×
  ravel.T      2.16×    1.37×    0.89×    1.93×    1.25×     1.45×
  in-place     0.62×    0.70×    1.14×    1.02×    1.05×     0.88×
  less->b      0.71×    0.49×    0.18×    0.60×    1.02×     0.52×
-- index-math
  unravel      0.56×    0.86×    1.08×    1.00×    1.06×     0.89×
  ravel_mi     1.29×    1.45×    1.09×    1.51×    0.84×     1.21×
-- dtypes
  complex      0.33×    0.63×    0.91×    0.73×    0.53×     0.59×
  float16      0.61×    0.60×    0.54×    0.34×    0.34×     0.47×
  int8         0.63×    1.43×   12.16×    6.62×    4.76×     3.22×

CONSTRUCTION — iterator build+dispose vs np.nditer (size-invariant, 1K)
        slower ◄───────── 1.0 (parity) ─────────► faster
1op          ███████████▊ .......   1.18×    84%🕐  (  1 win /  0 lose)
3op_exl      ███████████████████▶   4.09×    24%🕐  (  1 win /  0 lose)
ufunc        ███████████████████▶   3.30×    30%🕐  (  1 win /  0 lose)
bufcast      ███████████████████▶   3.47×    29%🕐  (  1 win /  0 lose)
multiindex   ███████████████████▶   2.05×    49%🕐  (  1 win /  0 lose)
8op          ███████████████████▶   5.71×    17%🕐  (  1 win /  0 lose)
4d           ██████████████████▍    1.84×    54%🕐  (  1 win /  0 lose)
8d           ███████████████████▶   2.24×    45%🕐  (  1 win /  0 lose)
strided2d    ███████████████████▶   3.18×    31%🕐  (  1 win /  0 lose)
geomean      ███████████████████▶   2.73×    37%🕐  (  9 win /  0 lose)

CHUNK-WIDTH dispatch — strided rows, 2M total, inner width w (NumPy = np.positive)
        slower ◄───────── 1.0 (parity) ─────────► faster
w=4          ████▊ ..............   0.48×   207%🕐  (  0 win /  1 lose)  ◄ SLOWER
w=16         ██████████▏ ........   1.02×    98%🕐  (  1 win /  0 lose)  ◄ PARITY
w=64         ██████████ .........   1.00×   100%🕐  (  1 win /  0 lose)  ◄ PARITY
w=256        ████████████▎ ......   1.23×    81%🕐  (  1 win /  0 lose)
w=1024       █████████████▉ .....   1.40×    72%🕐  (  1 win /  0 lose)

PATHOLOGY canaries — known taxes/losses to track (NumPy ÷ NumSharp)
  bcast_reduce    936.06×   (936.1× faster, faster)
  allocate          1.04×   (1.0× faster, faster)
  overlap_copy      1.83×   (1.8× faster, faster)
  forder_out        1.24×   (1.2× faster, faster)
  zerodim           0.62×   (1.6× slower, SLOWER)

DIVIDENDS — NumSharp-only machinery (NumPy baseline = closest it can do)
                scalar       1K     100K       1M      10M   note
fuse7           12.16×    3.89×    1.44×    1.66×    1.93×   vs chained 6× add
reuse            6.16×    6.41×    1.03×    1.01×    1.02×   vs rebuild each call
par8                 -    0.74×    3.36×    6.03×    2.94×   vs single-thread

biggest NumSharp wins: anyeh@10M 23.53× · anyff@1K 23.27× · anyeh@100K 23.18× · anyeh@1K 23.13× · bassign@1M 16.46×
most behind:           lessbool@100K 0.18× · bassign@1 0.22× · sqrt@10M 0.23× · bread@1 0.25× · astype@1 0.32×
```
