# Strings — text computation over S / U / StringDType (NumSharp vs NumPy 2.4.2)

Every cell times one call on the SAME data on both sides (the NumPy twin writes it, NumSharp loads it), best of rounds after a warm-up. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. A call one side refuses for a kind (NumPy has no such loop) has no cell.

## Summary

- **442** cells, geomean **6.61×**, min **1.44×** (`argsort` on U, 20 % non-ASCII at 1K).
- Under 1.0×: **0**; under 1.5×: **4**.
- By dtype: S 7.66× (50), U 6.97× (202), T 6.01× (190).
- By size: 1K 5.98× (217), 100K 7.24× (213), 1M 8.19× (12).

| Family | Cells | Geomean | Min | Under 1.5× |
|---|---:|---:|---:|---:|
| Comparisons | 38 | 4.94× | 1.65× | 0 |
| Concatenation and repetition | 38 | 4.03× | 1.59× | 0 |
| Reductions | 12 | 5.20× | 2.16× | 0 |
| Sorting, searching, sets | 74 | 5.73× | 1.44× | 2 |
| np.strings: length and search | 68 | 3.39× | 1.47× | 1 |
| np.strings: transforms | 98 | 3.38× | 1.48× | 1 |
| np.strings: case maps and predicates | 58 | 24.16× | 3.11× | 0 |
| np.strings: translate, mod, codecs | 48 | 37.92× | 6.12× | 0 |
| Text → number | 8 | 3.78× | 2.47× | 0 |

## Comparisons

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `a == b` | U, 1–8 ASCII | ✅ 9.81× (0.00168 ms) | ✅ 12.24× (0.166 ms) | ✅ 7.21× (2.91 ms) |
| `a == b` | U, 20 % non-ASCII | ✅ 9.82× (0.00167 ms) | ✅ 12.65× (0.16 ms) |  |
| `a == b` | U, 20–40 chars | ✅ 4.24× (0.00391 ms) | ✅ 2.25× (0.996 ms) |  |
| `a == b` | S, 1–8 ASCII | ✅ 12.86× (0.00127 ms) | ✅ 33.53× (0.061 ms) |  |
| `a == b` | T, 1–8 ASCII | ✅ 2.68× (0.00445 ms) | ✅ 2.33× (0.476 ms) | ✅ 2.31× (4.91 ms) |
| `a == b` | T, 20–40 chars (arena) | ✅ 2.24× (0.00655 ms) | ✅ 2.31× (0.586 ms) |  |
| `a == b` | T, 10 % NA | ✅ 2.11× (0.00529 ms) | ✅ 1.65× (0.667 ms) |  |
| `a == b` | U, a[::2] | ✅ 9.69× (0.00171 ms) | ✅ 7.98× (0.257 ms) |  |
| `a == b` | T, a[::2] | ✅ 2.71× (0.00443 ms) | ✅ 2.36× (0.478 ms) |  |
| `a < b` | U, 1–8 ASCII | ✅ 7.56× (0.00218 ms) | ✅ 11.13× (0.188 ms) |  |
| `a < b` | U, 20 % non-ASCII | ✅ 7.50× (0.00219 ms) | ✅ 11.15× (0.187 ms) |  |
| `a < b` | U, 20–40 chars | ✅ 7.09× (0.00234 ms) | ✅ 3.17× (0.705 ms) |  |
| `a < b` | S, 1–8 ASCII | ✅ 8.88× (0.00184 ms) | ✅ 18.53× (0.115 ms) |  |
| `a < b` | T, 1–8 ASCII | ✅ 2.80× (0.00426 ms) | ✅ 3.18× (0.35 ms) |  |
| `a < b` | T, 20–40 chars (arena) | ✅ 2.74× (0.00533 ms) | ✅ 2.98× (0.453 ms) |  |
| `a < b` | T, 10 % NA | ✅ 2.12× (0.00524 ms) | ✅ 1.98× (0.557 ms) |  |
| `a < b` | U, a[::2] | ✅ 7.22× (0.0023 ms) | ✅ 8.25× (0.256 ms) |  |
| `a < b` | T, a[::2] | ✅ 2.77× (0.00432 ms) | ✅ 3.22× (0.349 ms) |  |

## Concatenation and repetition

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `np.strings.add(a, b)` | U, 1–8 ASCII | ✅ 6.88× (0.00268 ms) | ✅ 10.97× (0.369 ms) | ✅ 6.70× (6.07 ms) |
| `np.strings.add(a, b)` | U, 20 % non-ASCII | ✅ 6.29× (0.0029 ms) | ✅ 10.98× (0.364 ms) |  |
| `np.strings.add(a, b)` | U, 20–40 chars | ✅ 2.80× (0.0138 ms) | ✅ 4.03× (2.93 ms) |  |
| `np.strings.add(a, b)` | S, 1–8 ASCII | ✅ 10.69× (0.0019 ms) | ✅ 30.27× (0.123 ms) |  |
| `np.strings.add(a, b)` | T, 1–8 ASCII | ✅ 3.22× (0.012 ms) | ✅ 3.25× (1.18 ms) | ✅ 2.96× (13.1 ms) |
| `np.strings.add(a, b)` | T, 20–40 chars (arena) | ✅ 2.36× (0.0245 ms) | ✅ 3.96× (3.45 ms) |  |
| `np.strings.add(a, b)` | T, 10 % NA | ✅ 2.63× (0.0131 ms) | ✅ 2.47× (1.52 ms) |  |
| `np.strings.add(a, b)` | U, a[::2] | ✅ 6.58× (0.00285 ms) | ✅ 8.81× (0.479 ms) |  |
| `np.strings.add(a, b)` | T, a[::2] | ✅ 3.21× (0.012 ms) | ✅ 3.26× (1.18 ms) |  |
| `np.strings.multiply(a, 3)` | U, 1–8 ASCII | ✅ 3.02× (0.0138 ms) | ✅ 7.51× (0.896 ms) |  |
| `np.strings.multiply(a, 3)` | U, 20 % non-ASCII | ✅ 2.98× (0.0138 ms) | ✅ 7.16× (0.879 ms) |  |
| `np.strings.multiply(a, 3)` | U, 20–40 chars | ✅ 1.59× (0.0479 ms) | ✅ 1.74× (9.2 ms) |  |
| `np.strings.multiply(a, 3)` | S, 1–8 ASCII | ✅ 3.23× (0.012 ms) | ✅ 7.41× (0.657 ms) |  |
| `np.strings.multiply(a, 3)` | T, 1–8 ASCII | ✅ 2.84× (0.0167 ms) | ✅ 2.08× (2.35 ms) |  |
| `np.strings.multiply(a, 3)` | T, 20–40 chars (arena) | ✅ 2.05× (0.0326 ms) | ✅ 3.89× (4.61 ms) |  |
| `np.strings.multiply(a, 3)` | T, 10 % NA | ✅ 2.33× (0.019 ms) | ✅ 2.04× (2.44 ms) |  |
| `np.strings.multiply(a, 3)` | U, a[::2] | ✅ 2.91× (0.0145 ms) | ✅ 6.41× (1.02 ms) |  |
| `np.strings.multiply(a, 3)` | T, a[::2] | ✅ 2.51× (0.0188 ms) | ✅ 2.15× (2.35 ms) |  |

## Reductions

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `np.max(a)` | T, 1–8 ASCII | ✅ 2.66× (0.00627 ms) | ✅ 3.08× (0.383 ms) |  |
| `np.max(a)` | T, 20–40 chars (arena) | ✅ 3.45× (0.00657 ms) | ✅ 4.10× (0.369 ms) |  |
| `np.max(a)` | T, 10 % NA | ✅ 3.56× (0.00334 ms) | ✅ 3.78× (0.196 ms) |  |
| `np.max(a)` | T, a[::2] | ✅ 2.61× (0.00635 ms) | ✅ 3.05× (0.385 ms) |  |
| `np.sum(a) (T only; 1K: NumPy's is quadratic)` | T, 1–8 ASCII | ✅ 17.90× (0.00967 ms) |  |  |
| `np.sum(a) (T only; 1K: NumPy's is quadratic)` | T, 20–40 chars (arena) | ✅ 46.51× (0.014 ms) |  |  |
| `np.sum(a) (T only; 1K: NumPy's is quadratic)` | T, 10 % NA | ✅ 2.16× (0.00581 ms) |  |  |
| `np.sum(a) (T only; 1K: NumPy's is quadratic)` | T, a[::2] | ✅ 17.50× (0.0098 ms) |  |  |

## Sorting, searching, sets

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `np.sort(a)` | U, 1–8 ASCII | ✅ 3.62× (0.0187 ms) | ✅ 4.91× (3.47 ms) | ✅ 4.88× (41.4 ms) |
| `np.sort(a)` | U, 20 % non-ASCII | ✅ 2.30× (0.0299 ms) | ✅ 3.27× (5.02 ms) |  |
| `np.sort(a)` | U, 20–40 chars | ✅ 2.34× (0.0812 ms) | ✅ 4.28× (8.17 ms) |  |
| `np.sort(a)` | S, 1–8 ASCII | ✅ 3.93× (0.0166 ms) | ✅ 6.69× (2.35 ms) |  |
| `np.sort(a)` | T, 1–8 ASCII | ✅ 6.65× (0.0318 ms) | ✅ 10.38× (3.66 ms) | ✅ 8.17× (56.7 ms) |
| `np.sort(a)` | T, 20–40 chars (arena) | ✅ 6.07× (0.0457 ms) | ✅ 8.82× (6.27 ms) |  |
| `np.sort(a)` | T, 10 % NA | ✅ 6.75× (0.0305 ms) | ✅ 9.44× (3.84 ms) |  |
| `np.sort(a)` | U, a[::2] | ✅ 3.70× (0.0189 ms) | ✅ 4.98× (3.56 ms) |  |
| `np.sort(a)` | T, a[::2] | ✅ 7.02× (0.0314 ms) | ✅ 9.62× (3.91 ms) |  |
| `np.argsort(a)` | U, 1–8 ASCII | ✅ 2.26× (0.0135 ms) | ✅ 2.10× (7.97 ms) |  |
| `np.argsort(a)` | U, 20 % non-ASCII | ✅ 1.44× (0.0196 ms) | ✅ 1.45× (11.3 ms) |  |
| `np.argsort(a)` | U, 20–40 chars | ✅ 1.92× (0.0148 ms) | ✅ 7.01× (2.81 ms) |  |
| `np.argsort(a)` | S, 1–8 ASCII | ✅ 1.93× (0.0129 ms) | ✅ 2.03× (7.84 ms) |  |
| `np.argsort(a)` | T, 1–8 ASCII | ✅ 4.10× (0.0416 ms) | ✅ 2.89× (11.7 ms) |  |
| `np.argsort(a)` | T, 20–40 chars (arena) | ✅ 7.32× (0.0288 ms) | ✅ 11.53× (4.04 ms) |  |
| `np.argsort(a)` | T, 10 % NA | ✅ 4.29× (0.0383 ms) | ✅ 2.74× (11.6 ms) |  |
| `np.argsort(a)` | U, a[::2] | ✅ 2.79× (0.0139 ms) | ✅ 2.17× (8.4 ms) |  |
| `np.argsort(a)` | T, a[::2] | ✅ 4.71× (0.0411 ms) | ✅ 2.89× (12.1 ms) |  |
| `np.unique(a)` | U, 1–8 ASCII | ✅ 5.81× (0.0261 ms) | ✅ 9.03× (3.73 ms) |  |
| `np.unique(a)` | U, 20 % non-ASCII | ✅ 4.02× (0.0372 ms) | ✅ 6.09× (5.54 ms) |  |
| `np.unique(a)` | U, 20–40 chars | ✅ 10.19× (0.0458 ms) | ✅ 10.39× (10.4 ms) |  |
| `np.unique(a)` | S, 1–8 ASCII | ✅ 5.44× (0.0226 ms) | ✅ 9.30× (2.69 ms) |  |
| `np.unique(a)` | T, 1–8 ASCII | ✅ 6.02× (0.0461 ms) | ✅ 4.84× (9.21 ms) |  |
| `np.unique(a)` | T, 20–40 chars (arena) | ✅ 6.60× (0.0643 ms) | ✅ 9.24× (9.79 ms) |  |
| `np.unique(a)` | T, 10 % NA | ✅ 5.27× (0.05 ms) | ✅ 3.91× (9.97 ms) |  |
| `np.unique(a)` | U, a[::2] | ✅ 5.83× (0.0262 ms) | ✅ 8.87× (3.78 ms) |  |
| `np.unique(a)` | T, a[::2] | ✅ 5.68× (0.048 ms) | ✅ 4.73× (9.4 ms) |  |
| `np.isin(a, a[:100])` | U, 1–8 ASCII | ✅ 2.40× (0.0677 ms) | ✅ 3.95× (6.82 ms) |  |
| `np.isin(a, a[:100])` | U, 20 % non-ASCII | ✅ 1.79× (0.0877 ms) | ✅ 2.79× (9.51 ms) |  |
| `np.isin(a, a[:100])` | U, 20–40 chars | ✅ 2.51× (0.0974 ms) | ✅ 2.89× (17.5 ms) |  |
| `np.isin(a, a[:100])` | S, 1–8 ASCII | ✅ 2.25× (0.0601 ms) | ✅ 4.32× (5.36 ms) |  |
| `np.isin(a, a[:100])` | T, 1–8 ASCII | ✅ 83.92× (0.0184 ms) | ✅ 31.26× (3.67 ms) |  |
| `np.isin(a, a[:100])` | T, 20–40 chars (arena) | ✅ 93.86× (0.0205 ms) | ✅ 36.12× (3.93 ms) |  |
| `np.isin(a, a[:100])` | T, 10 % NA | ✅ 83.58× (0.0166 ms) | ✅ 31.57× (3.32 ms) |  |
| `np.isin(a, a[:100])` | U, a[::2] | ✅ 2.69× (0.0638 ms) | ✅ 3.78× (7.31 ms) |  |
| `np.isin(a, a[:100])` | T, a[::2] | ✅ 72.42× (0.0215 ms) | ✅ 30.00× (3.94 ms) |  |

## np.strings: length and search

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `str_len(a)` | U, 1–8 ASCII | ✅ 1.65× (0.00219 ms) | ✅ 8.64× (0.157 ms) | ✅ 9.76× (1.68 ms) |
| `str_len(a)` | U, 20 % non-ASCII | ✅ 1.64× (0.00219 ms) | ✅ 8.55× (0.157 ms) |  |
| `str_len(a)` | U, 20–40 chars | ✅ 1.75× (0.00353 ms) | ✅ 4.82× (0.463 ms) |  |
| `str_len(a)` | S, 1–8 ASCII | ✅ 2.19× (0.0016 ms) | ✅ 11.99× (0.108 ms) |  |
| `str_len(a)` | T, 1–8 ASCII | ✅ 2.70× (0.00292 ms) | ✅ 6.55× (0.225 ms) | ✅ 6.92× (2.41 ms) |
| `str_len(a)` | T, 20–40 chars (arena) | ✅ 3.63× (0.0165 ms) | ✅ 2.32× (2.62 ms) |  |
| `str_len(a)` | U, a[::2] | ✅ 1.66× (0.0022 ms) | ✅ 7.13× (0.203 ms) |  |
| `str_len(a)` | T, a[::2] | ✅ 2.66× (0.00298 ms) | ✅ 6.41× (0.228 ms) |  |
| `find(a, 'a')` | U, 1–8 ASCII | ✅ 1.80× (0.00665 ms) | ✅ 2.95× (0.723 ms) |  |
| `find(a, 'a')` | U, 20 % non-ASCII | ✅ 1.80× (0.00665 ms) | ✅ 3.15× (0.674 ms) |  |
| `find(a, 'a')` | U, 20–40 chars | ✅ 2.56× (0.0095 ms) | ✅ 3.21× (1.28 ms) |  |
| `find(a, 'a')` | S, 1–8 ASCII | ✅ 2.01× (0.0056 ms) | ✅ 3.76× (0.59 ms) |  |
| `find(a, 'a')` | T, 1–8 ASCII | ✅ 3.14× (0.011 ms) | ✅ 3.21× (1.11 ms) |  |
| `find(a, 'a')` | T, 20–40 chars (arena) | ✅ 3.75× (0.0663 ms) | ✅ 3.26× (7.73 ms) |  |
| `find(a, 'a')` | U, a[::2] | ✅ 1.87× (0.00636 ms) | ✅ 2.99× (0.725 ms) |  |
| `find(a, 'a')` | T, a[::2] | ✅ 3.38× (0.0108 ms) | ✅ 3.16× (1.12 ms) |  |
| `count(a, 'a')` | U, 1–8 ASCII | ✅ 2.14× (0.00603 ms) | ✅ 5.33× (0.474 ms) |  |
| `count(a, 'a')` | U, 20 % non-ASCII | ✅ 2.14× (0.00603 ms) | ✅ 5.27× (0.475 ms) |  |
| `count(a, 'a')` | U, 20–40 chars | ✅ 6.29× (0.00943 ms) | ✅ 8.58× (0.82 ms) |  |
| `count(a, 'a')` | S, 1–8 ASCII | ✅ 3.55× (0.00513 ms) | ✅ 7.66× (0.381 ms) |  |
| `count(a, 'a')` | T, 1–8 ASCII | ✅ 3.95× (0.0105 ms) | ✅ 5.15× (0.841 ms) |  |
| `count(a, 'a')` | T, 20–40 chars (arena) | ✅ 5.03× (0.0487 ms) | ✅ 4.61× (5.37 ms) |  |
| `count(a, 'a')` | U, a[::2] | ✅ 2.11× (0.00608 ms) | ✅ 5.31× (0.475 ms) |  |
| `count(a, 'a')` | T, a[::2] | ✅ 4.07× (0.0105 ms) | ✅ 5.18× (0.842 ms) |  |
| `startswith(a, 'a')` | U, 1–8 ASCII | ✅ 1.63× (0.00546 ms) | ✅ 3.69× (0.412 ms) |  |
| `startswith(a, 'a')` | U, 20 % non-ASCII | ✅ 1.61× (0.00544 ms) | ✅ 3.67× (0.411 ms) |  |
| `startswith(a, 'a')` | U, 20–40 chars | ✅ 2.62× (0.00558 ms) | ✅ 5.31× (0.427 ms) |  |
| `startswith(a, 'a')` | S, 1–8 ASCII | ✅ 1.47× (0.00516 ms) | ✅ 3.91× (0.382 ms) |  |
| `startswith(a, 'a')` | T, 1–8 ASCII | ✅ 2.46× (0.00951 ms) | ✅ 3.45× (0.73 ms) |  |
| `startswith(a, 'a')` | T, 20–40 chars (arena) | ✅ 2.68× (0.0265 ms) | ✅ 2.09× (3.27 ms) |  |
| `startswith(a, 'a')` | T, 10 % NA | ✅ 2.20× (0.00939 ms) | ✅ 2.95× (0.847 ms) |  |
| `startswith(a, 'a')` | U, a[::2] | ✅ 1.62× (0.00552 ms) | ✅ 3.72× (0.418 ms) |  |
| `startswith(a, 'a')` | T, a[::2] | ✅ 2.56× (0.00943 ms) | ✅ 3.37× (0.738 ms) |  |

## np.strings: transforms

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `strip(a, 'ab')` | U, 1–8 ASCII | ✅ 1.84× (0.00845 ms) | ✅ 3.81× (0.791 ms) |  |
| `strip(a, 'ab')` | U, 20 % non-ASCII | ✅ 1.83× (0.0084 ms) | ✅ 3.74× (0.786 ms) |  |
| `strip(a, 'ab')` | U, 20–40 chars | ✅ 1.48× (0.0191 ms) | ✅ 1.83× (3.54 ms) |  |
| `strip(a, 'ab')` | S, 1–8 ASCII | ✅ 2.26× (0.00655 ms) | ✅ 4.82× (0.605 ms) |  |
| `strip(a, 'ab')` | T, 1–8 ASCII | ✅ 3.36× (0.0274 ms) | ✅ 2.71× (3.45 ms) |  |
| `strip(a, 'ab')` | T, 20–40 chars (arena) | ✅ 3.40× (0.0488 ms) | ✅ 2.30× (8.72 ms) |  |
| `strip(a, 'ab')` | T, 10 % NA | ✅ 3.50× (0.0248 ms) | ✅ 2.71× (3.28 ms) |  |
| `strip(a, 'ab')` | U, a[::2] | ✅ 1.87× (0.00835 ms) | ✅ 3.83× (0.793 ms) |  |
| `strip(a, 'ab')` | T, a[::2] | ✅ 3.47× (0.0271 ms) | ✅ 2.72× (3.48 ms) |  |
| `replace(a, 'a', 'xy')` | U, 1–8 ASCII | ✅ 3.20× (0.0269 ms) | ✅ 2.87× (3.8 ms) |  |
| `replace(a, 'a', 'xy')` | U, 20 % non-ASCII | ✅ 3.35× (0.0256 ms) | ✅ 2.65× (3.78 ms) |  |
| `replace(a, 'a', 'xy')` | U, 20–40 chars | ✅ 5.69× (0.0443 ms) | ✅ 3.30× (8.79 ms) |  |
| `replace(a, 'a', 'xy')` | S, 1–8 ASCII | ✅ 4.08× (0.0209 ms) | ✅ 3.22× (3.05 ms) |  |
| `replace(a, 'a', 'xy')` | T, 1–8 ASCII | ✅ 3.07× (0.0415 ms) | ✅ 2.64× (4.54 ms) |  |
| `replace(a, 'a', 'xy')` | T, 20–40 chars (arena) | ✅ 4.10× (0.117 ms) | ✅ 3.13× (15.6 ms) |  |
| `replace(a, 'a', 'xy')` | T, 10 % NA | ✅ 3.02× (0.0396 ms) | ✅ 2.66× (4.28 ms) |  |
| `replace(a, 'a', 'xy')` | U, a[::2] | ✅ 3.39× (0.0262 ms) | ✅ 2.65× (3.81 ms) |  |
| `replace(a, 'a', 'xy')` | T, a[::2] | ✅ 3.07× (0.0423 ms) | ✅ 2.67× (4.51 ms) |  |
| `center(a, 12)` | U, 1–8 ASCII | ✅ 4.68× (0.011 ms) | ✅ 10.69× (0.698 ms) |  |
| `center(a, 12)` | U, 20 % non-ASCII | ✅ 4.70× (0.0109 ms) | ✅ 10.59× (0.697 ms) |  |
| `center(a, 12)` | U, 20–40 chars | ✅ 3.00× (0.0184 ms) | ✅ 2.49× (3.47 ms) |  |
| `center(a, 12)` | S, 1–8 ASCII | ✅ 3.24× (0.0108 ms) | ✅ 8.62× (0.482 ms) |  |
| `center(a, 12)` | T, 1–8 ASCII | ✅ 2.24× (0.0392 ms) | ✅ 1.97× (3.89 ms) |  |
| `center(a, 12)` | T, 20–40 chars (arena) | ✅ 3.53× (0.0483 ms) | ✅ 2.46× (7.95 ms) |  |
| `center(a, 12)` | T, 10 % NA | ✅ 2.29× (0.0362 ms) | ✅ 2.01× (3.68 ms) |  |
| `center(a, 12)` | U, a[::2] | ✅ 4.72× (0.011 ms) | ✅ 10.93× (0.696 ms) |  |
| `center(a, 12)` | T, a[::2] | ✅ 2.25× (0.0388 ms) | ✅ 1.99× (3.91 ms) |  |
| `zfill(a, 10)` | U, 1–8 ASCII | ✅ 4.58× (0.00764 ms) | ✅ 9.45× (0.63 ms) |  |
| `zfill(a, 10)` | U, 20 % non-ASCII | ✅ 4.58× (0.00764 ms) | ✅ 9.35× (0.637 ms) |  |
| `zfill(a, 10)` | U, 20–40 chars | ✅ 2.95× (0.0154 ms) | ✅ 2.66× (3.55 ms) |  |
| `zfill(a, 10)` | S, 1–8 ASCII | ✅ 4.74× (0.00558 ms) | ✅ 9.15× (0.431 ms) |  |
| `zfill(a, 10)` | T, 1–8 ASCII | ✅ 4.50× (0.0153 ms) | ✅ 3.17× (2.21 ms) |  |
| `zfill(a, 10)` | T, 20–40 chars (arena) | ✅ 5.34× (0.0397 ms) | ✅ 3.22× (7.69 ms) |  |
| `zfill(a, 10)` | T, 10 % NA | ✅ 4.47× (0.0144 ms) | ✅ 3.13× (2.15 ms) |  |
| `zfill(a, 10)` | U, a[::2] | ✅ 4.68× (0.00764 ms) | ✅ 9.66× (0.632 ms) |  |
| `zfill(a, 10)` | T, a[::2] | ✅ 4.66× (0.0154 ms) | ✅ 3.16× (2.24 ms) |  |
| `partition(a, 'a')` | U, 1–8 ASCII | ✅ 3.74× (0.022 ms) | ✅ 4.70× (2.18 ms) |  |
| `partition(a, 'a')` | U, 20 % non-ASCII | ✅ 4.03× (0.0209 ms) | ✅ 4.85× (2.07 ms) |  |
| `partition(a, 'a')` | U, 20–40 chars | ✅ 4.50× (0.0443 ms) | ✅ 3.21× (7.55 ms) |  |
| `partition(a, 'a')` | S, 1–8 ASCII | ✅ 4.17× (0.0193 ms) | ✅ 5.08× (1.81 ms) |  |
| `partition(a, 'a')` | U, a[::2] | ✅ 3.94× (0.0215 ms) | ✅ 4.73× (2.19 ms) |  |
| `slice(a, 1, 4)` | U, 1–8 ASCII | ✅ 2.39× (0.00763 ms) | ✅ 2.19× (1.27 ms) |  |
| `slice(a, 1, 4)` | U, 20 % non-ASCII | ✅ 2.41× (0.00767 ms) | ✅ 2.14× (1.28 ms) |  |
| `slice(a, 1, 4)` | U, 20–40 chars | ✅ 2.34× (0.0143 ms) | ✅ 4.33× (1.41 ms) |  |
| `slice(a, 1, 4)` | S, 1–8 ASCII | ✅ 2.62× (0.00744 ms) | ✅ 2.03× (1.09 ms) |  |
| `slice(a, 1, 4)` | T, 1–8 ASCII | ✅ 1.77× (0.0259 ms) | ✅ 1.78× (2.88 ms) |  |
| `slice(a, 1, 4)` | T, 20–40 chars (arena) | ✅ 4.52× (0.0395 ms) | ✅ 3.89× (4.49 ms) |  |
| `slice(a, 1, 4)` | U, a[::2] | ✅ 2.36× (0.00781 ms) | ✅ 2.19× (1.28 ms) |  |
| `slice(a, 1, 4)` | T, a[::2] | ✅ 1.83× (0.0254 ms) | ✅ 1.77× (2.9 ms) |  |

## np.strings: case maps and predicates

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `upper(a)` | U, 1–8 ASCII | ✅ 146.00× (0.00201 ms) | ✅ 147.63× (0.202 ms) | ✅ 97.75× (3.11 ms) |
| `upper(a)` | U, 20 % non-ASCII | ✅ 40.75× (0.00843 ms) | ✅ 22.07× (1.63 ms) |  |
| `upper(a)` | U, 20–40 chars | ✅ 51.74× (0.00693 ms) | ✅ 35.67× (1.08 ms) |  |
| `upper(a)` | S, 1–8 ASCII | ✅ 100.25× (0.00111 ms) | ✅ 158.85× (0.0706 ms) |  |
| `upper(a)` | T, 1–8 ASCII | ✅ 58.37× (0.00273 ms) | ✅ 70.70× (0.223 ms) | ✅ 60.63× (2.67 ms) |
| `upper(a)` | T, 20–40 chars (arena) | ✅ 9.69× (0.0229 ms) | ✅ 9.35× (2.76 ms) |  |
| `upper(a)` | U, a[::2] | ✅ 142.76× (0.00206 ms) | ✅ 116.43× (0.256 ms) |  |
| `upper(a)` | T, a[::2] | ✅ 55.84× (0.00282 ms) | ✅ 68.90× (0.228 ms) |  |
| `title(a)` | U, 1–8 ASCII | ✅ 125.59× (0.0027 ms) | ✅ 147.26× (0.234 ms) |  |
| `title(a)` | U, 20 % non-ASCII | ✅ 36.68× (0.0101 ms) | ✅ 20.24× (1.89 ms) |  |
| `title(a)` | U, 20–40 chars | ✅ 45.59× (0.0112 ms) | ✅ 33.56× (1.63 ms) |  |
| `title(a)` | S, 1–8 ASCII | ✅ 53.28× (0.00222 ms) | ✅ 64.84× (0.181 ms) |  |
| `title(a)` | T, 1–8 ASCII | ✅ 57.88× (0.00342 ms) | ✅ 68.65× (0.29 ms) |  |
| `title(a)` | T, 20–40 chars (arena) | ✅ 13.24× (0.0273 ms) | ✅ 12.65× (3.14 ms) |  |
| `title(a)` | U, a[::2] | ✅ 120.69× (0.00281 ms) | ✅ 137.91× (0.253 ms) |  |
| `title(a)` | T, a[::2] | ✅ 56.34× (0.00352 ms) | ✅ 68.23× (0.289 ms) |  |
| `isalpha(a)` | U, 1–8 ASCII | ✅ 5.47× (0.00401 ms) | ✅ 8.06× (0.338 ms) |  |
| `isalpha(a)` | U, 20 % non-ASCII | ✅ 4.50× (0.00474 ms) | ✅ 3.73× (0.752 ms) |  |
| `isalpha(a)` | U, 20–40 chars | ✅ 7.89× (0.0128 ms) | ✅ 8.31× (1.21 ms) |  |
| `isalpha(a)` | S, 1–8 ASCII | ✅ 4.92× (0.00316 ms) | ✅ 8.43× (0.263 ms) |  |
| `isalpha(a)` | T, 1–8 ASCII | ✅ 8.23× (0.00466 ms) | ✅ 9.63× (0.395 ms) |  |
| `isalpha(a)` | T, 20–40 chars (arena) | ✅ 22.35× (0.0084 ms) | ✅ 15.64× (1.2 ms) |  |
| `isalpha(a)` | T, 10 % NA | ✅ 7.82× (0.00441 ms) | ✅ 7.34× (0.471 ms) |  |
| `isalpha(a)` | U, a[::2] | ✅ 5.41× (0.00405 ms) | ✅ 8.04× (0.34 ms) |  |
| `isalpha(a)` | T, a[::2] | ✅ 8.03× (0.00473 ms) | ✅ 9.59× (0.397 ms) |  |
| `isdigit(a)` | U, digits | ✅ 3.11× (0.00779 ms) | ✅ 3.34× (0.716 ms) |  |
| `isdigit(a)` | S, digits | ✅ 6.11× (0.00321 ms) | ✅ 7.24× (0.266 ms) |  |
| `isdigit(a)` | T, digits | ✅ 10.69× (0.00429 ms) | ✅ 12.52× (0.362 ms) |  |

## np.strings: translate, mod, codecs

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `translate(a, abc→xyz)` | U, 1–8 ASCII | ✅ 140.48× (0.00581 ms) | ✅ 303.25× (0.279 ms) |  |
| `translate(a, abc→xyz)` | U, 20 % non-ASCII | ✅ 162.61× (0.00594 ms) | ✅ 355.99× (0.278 ms) |  |
| `translate(a, abc→xyz)` | U, 20–40 chars | ✅ 148.93× (0.0155 ms) | ✅ 186.47× (1.28 ms) |  |
| `translate(a, abc→xyz)` | S, 1–8 ASCII | ✅ 26.60× (0.00724 ms) | ✅ 54.98× (0.348 ms) |  |
| `translate(a, abc→xyz)` | T, 1–8 ASCII | ✅ 56.88× (0.0119 ms) | ✅ 42.25× (1.6 ms) |  |
| `translate(a, abc→xyz)` | T, 20–40 chars (arena) | ✅ 60.57× (0.0355 ms) | ✅ 55.59× (3.92 ms) |  |
| `translate(a, abc→xyz)` | U, a[::2] | ✅ 136.27× (0.00602 ms) | ✅ 299.55× (0.28 ms) |  |
| `translate(a, abc→xyz)` | T, a[::2] | ✅ 54.58× (0.0122 ms) | ✅ 41.77× (1.62 ms) |  |
| `mod('<%s>', a)` | U, 1–8 ASCII | ✅ 15.24× (0.0533 ms) | ✅ 14.48× (5.74 ms) |  |
| `mod('<%s>', a)` | U, 20 % non-ASCII | ✅ 16.68× (0.053 ms) | ✅ 16.08× (5.76 ms) |  |
| `mod('<%s>', a)` | U, 20–40 chars | ✅ 13.64× (0.067 ms) | ✅ 13.29× (7.58 ms) |  |
| `mod('<%s>', a)` | T, 1–8 ASCII | ✅ 7.10× (0.0754 ms) | ✅ 7.47× (7.23 ms) |  |
| `mod('<%s>', a)` | T, 20–40 chars (arena) | ✅ 6.86× (0.131 ms) | ✅ 6.12× (15.9 ms) |  |
| `mod('<%s>', a)` | T, 10 % NA | ✅ 6.54× (0.0827 ms) | ✅ 6.87× (8.01 ms) |  |
| `mod('<%s>', a)` | U, a[::2] | ✅ 14.97× (0.0542 ms) | ✅ 14.32× (5.93 ms) |  |
| `mod('<%s>', a)` | T, a[::2] | ✅ 7.17× (0.0752 ms) | ✅ 7.52× (7.33 ms) |  |
| `encode(a, 'utf-8')` | U, 1–8 ASCII | ✅ 66.93× (0.00933 ms) | ✅ 135.80× (0.463 ms) |  |
| `encode(a, 'utf-8')` | U, 20 % non-ASCII | ✅ 33.92× (0.0201 ms) | ✅ 23.81× (2.91 ms) |  |
| `encode(a, 'utf-8')` | U, 20–40 chars | ✅ 7.33× (0.0907 ms) | ✅ 7.82× (9 ms) |  |
| `encode(a, 'utf-8')` | T, 1–8 ASCII | ✅ 69.90× (0.00713 ms) | ✅ 248.68× (0.202 ms) |  |
| `encode(a, 'utf-8')` | T, 20–40 chars (arena) | ✅ 26.32× (0.0206 ms) | ✅ 26.93× (2.1 ms) |  |
| `encode(a, 'utf-8')` | U, a[::2] | ✅ 66.66× (0.00937 ms) | ✅ 134.43× (0.467 ms) |  |
| `encode(a, 'utf-8')` | T, a[::2] | ✅ 69.72× (0.00711 ms) | ✅ 253.88× (0.203 ms) |  |
| `decode(a, 'utf-8')` | S, 1–8 ASCII | ✅ 60.89× (0.00827 ms) | ✅ 129.55× (0.398 ms) |  |

## Text → number

| Call | Data | 1K | 100K | 1M |
|---|---|---:|---:|---:|
| `a.astype(float64)` | U, digits | ✅ 6.83× (0.046 ms) | ✅ 6.86× (4.62 ms) | ✅ 7.00× (46.4 ms) |
| `a.astype(float64)` | S, digits | ✅ 2.90× (0.0418 ms) | ✅ 2.90× (4.1 ms) |  |
| `a.astype(float64)` | T, digits | ✅ 2.47× (0.0388 ms) | ✅ 2.47× (3.85 ms) | ✅ 2.48× (39.9 ms) |

