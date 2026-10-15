# Missing NumPy API, ranked by GitHub usage

Every API the coverage inventory reports missing (generator 1.12.0, NumPy 2.4.2, journey4 `24d4a020`), with
the number of public Python files on GitHub that use it. 1,591 missing rows; 1,219 carry a
number (from 600 measures), 372 cannot be measured by code search (reasons below).
Measured 2026-10-15 with `gh api search/code`. Row-level data: `numpy-missing-api-usage.csv` beside this file.

## How the numbers were made

- **Phrase per API**, e.g. `"np.genfromtxt"`, `"np.char.upper"`, `"np.maximum.reduce"`. A class whose members are
  called on variables (`recarray`, `memmap`, `chararray`, the polynomial classes) is measured once, at class level,
  and every member row takes that number.
- **Filter:** `extension:py size:<40000 NOT site-packages in:file,path`, plus `-filename:` for the small source files
  of NumPy, SciPy, pandas, scikit-learn, Matplotlib and Numba that contain the phrase. Committed virtual
  environments and bundles carry these libraries' own tests and docs and otherwise dominate the counts.
- **Prefix matching:** GitHub's API search (the legacy engine) matches the last word of a phrase as a prefix:
  `np.test` also counts `np.testing`, `np.rec` counts `np.reciprocal`/`np.recarray`, `ufunc.reduce` counts
  `reduceat`. Every measure above 700 files was sampled (108 measures: the matched text of page 1, plus the
  deepest page above 1,000 files); its value is count x (literal, non-vendored hits / sampled hits). Measures
  dominated by a colliding word were re-counted with `NOT` terms first; `.reduce` measures carry
  `NOT reduceat`, which makes them lower bounds. Smaller measures keep their first-pass value, flagged
  `unsampled`: GitHub's rate limit (HTTP 429, waits up to 12 minutes after ~1,150 searches) ended the sampling.
- Values are GitHub's estimates (about +-10% above a few thousand files) and count files, not repositories:
  copied code (course material, cloned repositories) counts every time.

## Top 80

| # | API | Files | Missing rows | How |
|---:|---|---:|---:|---|
| 1 | `np.integer` | 203,415 | 1 | estimate 224768 x 181/200 |
| 2 | `numpy.typing (module)` | 181,274 | 1 | estimate 186880 x 194/200 |
| 3 | `np.floating` | 161,070 | 1 | estimate 166912 x 193/200 |
| 4 | `np.number` | 127,949 | 1 | estimate 130560 x 196/200 |
| 5 | `np.errstate` | 119,043 | 1 | estimate 130816 x 182/200 |
| 6 | `np.genfromtxt` | 100,614 | 1 | estimate 101120 x 199/200 |
| 7 | `np.__version__` | 67,579 | 1 | estimate 68608 x 197/200 |
| 8 | `np.datetime64` | 58,928 | 1 | estimate 81280 x 145/200 |
| 9 | `np.add.at` | 53,165 | 1 | estimate 54528 x 195/200 |
| 10 | `np.complex64` | 49,398 | 1 | estimate 68608 x 144/200 |
| 11 | `np.generic` | 45,056 | 1 | estimate 51200 x 176/200 |
| 12 | `np.memmap (class)` | 44,997 | 72 | estimate 47616 x 189/200 |
| 13 | `np.maximum.accumulate` | 41,834 | 1 | estimate 42688 x 196/200 |
| 14 | `np.timedelta64` | 41,291 | 1 | estimate 55424 x 149/200 |
| 15 | `np.fromiter` | 31,163 | 1 | estimate 33152 x 188/200 |
| 16 | `np.ndim` | 29,737 | 1 | estimate 49152 x 121/200 |
| 17 | `np.seterr` | 26,369 | 1 | estimate 27904 x 189/200 |
| 18 | `np.object_` | 22,711 | 1 | estimate 36928 x 123/200 |
| 19 | `np.str_` | 22,473 | 1 | estimate 29376 x 153/200 |
| 20 | `ndarray.ctypes (via arr.ctypes.data_as)` | 14,333 | 1 | estimate 15008 x 191/200 |
| 21 | `np.get_include` | 13,324 | 1 | estimate 15584 x 171/200 |
| 22 | `leggauss` | 11,615 | 1 | estimate 11792 x 197/200 |
| 23 | `np.bytes_` | 11,396 | 1 | estimate 15296 x 149/200 |
| 24 | `np.void` | 11,294 | 1 | estimate 17376 x 130/200 |
| 25 | `np.maximum.reduce` | 10,048 | 1 | NOT reduceat re-count 10,048 (lower bound; the original 11,680 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 26 | `np.longdouble` | 9,639 | 1 | estimate 18016 x 107/200 |
| 27 | `np.core` | 8,941 | 1 | estimate 13345 x 134/200 |
| 28 | `np.complexfloating` | 8,452 | 1 | estimate 11904 x 142/200 |
| 29 | `np.minimum.accumulate` | 7,648 | 1 | estimate 7648 x 100/100 |
| 30 | `np.logical_and.reduce` | 7,376 | 1 | NOT reduceat re-count 7,376 (lower bound; the original 7,376 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 31 | `np.typecodes` | 7,181 | 1 | estimate 14656 x 98/200 |
| 32 | `np.recarray (class)` | 6,156 | 72 | estimate 8208 x 150/200 |
| 33 | `np.fromfunction` | 5,844 | 1 | estimate 6184 x 189/200 |
| 34 | `np.minimum.reduce` | 5,520 | 1 | NOT reduceat re-count 5,520 (lower bound; the original 6,528 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 35 | `np.add.reduceat` | 5,488 | 1 | estimate 5488 x 100/100 |
| 36 | `np.logical_or.reduce` | 5,472 | 1 | NOT reduceat re-count 5,472 (lower bound; the original 5,776 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 37 | `np.True_` | 4,832 | 1 | NOT re-count; estimate 4832 x 100/100 |
| 38 | `np.maximum.at` | 4,688 | 1 | estimate 4784 x 98/100 |
| 39 | `np.subtract.outer` | 4,400 | 1 | estimate 4400 x 100/100 |
| 40 | `np.signedinteger` | 4,101 | 1 | estimate 6408 x 128/200 |
| 41 | `np.ufunc` | 3,872 | 1 | estimate 7376 x 105/200 |
| 42 | `np.rec` | 3,795 | 1 | NOT re-count; estimate 3912 x 97/100 |
| 43 | `np.add.reduce` | 3,624 | 1 | NOT reduceat re-count 3,624 (lower bound; the original 9,472 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 44 | `np.minimum.at` | 3,049 | 1 | estimate 3080 x 99/100 |
| 45 | `np.datetime_as_string` | 2,801 | 1 | estimate 3112 x 180/200 |
| 46 | `np.multiply.outer` | 2,764 | 1 | estimate 2792 x 99/100 |
| 47 | `np.char.add` | 2,570 | 1 | estimate 2720 x 189/200 |
| 48 | `np.add.outer` | 2,448 | 1 | estimate 2448 x 100/100 |
| 49 | `polynomial.Polynomial (via Polynomial.fit)` | 2,346 | 27 | estimate 2496 x 188/200 |
| 50 | `np.False_` | 2,343 | 1 | estimate 2840 x 165/200 |
| 51 | `finfo.nmant` | 2,236 | 1 | estimate 6577 x 68/200 |
| 52 | `np.logaddexp.reduce` | 2,164 | 1 | NOT reduceat re-count 2,164 (lower bound; the original 2,156 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 53 | `np.inexact` | 2,137 | 1 | estimate 4912 x 87/200 |
| 54 | `np.rec.fromarrays` | 2,098 | 1 | estimate 2512 x 167/200 |
| 55 | `np.isnat` | 2,032 | 1 | estimate 2764 x 147/200 |
| 56 | `hermgauss` | 1,952 | 1 | estimate 2088 x 187/200 |
| 57 | `np.maximum.reduceat` | 1,908 | 1 | estimate 1908 x 100/100 |
| 58 | `np.unsignedinteger` | 1,855 | 1 | estimate 2728 x 136/200 |
| 59 | `np.add.accumulate` | 1,696 | 1 | estimate 1696 x 100/100 |
| 60 | `np.info` | 1,695 | 1 | estimate 2648 x 128/200 |
| 61 | `np.multiply.reduce` | 1,648 | 1 | NOT reduceat re-count 1,648 (lower bound; the original 1,732 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 62 | `np.rec.array` | 1,578 | 1 | estimate 3712 x 85/200 |
| 63 | `np.char.lower` | 1,412 | 1 | estimate 1412 x 200/200 |
| 64 | `np.busday_count` | 1,361 | 1 | estimate 1768 x 154/200 |
| 65 | `np.exceptions` | 1,361 | 1 | estimate 1972 x 138/200 |
| 66 | `np.minimum.reduceat` | 1,307 | 1 | estimate 1334 x 98/100 |
| 67 | `np.char.decode` | 1,243 | 1 | estimate 1480 x 168/200 |
| 68 | `np.char.find` | 1,216 | 1 | estimate 1222 x 199/200 |
| 69 | `polynomial.polyfit` | 1,214 | 1 | estimate 1220 x 199/200 |
| 70 | `np.from_dlpack` | 1,197 | 1 | estimate 1260 x 190/200 |
| 71 | `np.char.startswith` | 1,188 | 1 | estimate 1188 x 200/200 |
| 72 | `np.char.replace` | 1,123 | 1 | estimate 1140 x 197/200 |
| 73 | `np.char.strip` | 1,108 | 1 | estimate 1108 x 200/200 |
| 74 | `np.show_config` | 1,085 | 1 | estimate 1276 x 170/200 |
| 75 | `np.gcd.reduce` | 1,084 | 1 | NOT reduceat re-count 1,084 (lower bound; the original 1,056 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 76 | `np.char.mod` | 1,039 | 1 | estimate 1044 x 199/200 |
| 77 | `np.character` | 1,039 | 1 | estimate 1792 x 116/200 |
| 78 | `np.bitwise_or.reduce` | 1,028 | 1 | NOT reduceat re-count 1,028 (lower bound; the original 1,092 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 79 | `np.lcm.reduce` | 990 | 1 | NOT reduceat re-count 990 (lower bound; the original 990 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 80 | `ufunc.reduce` | 973 | 1 | NOT reduceat re-count 1,158 (lower bound; the original 1,984 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); estimate 1158 x 84/100 |

## By family

- **Scalar types and the abstract dtype hierarchy:** `np.integer` 203,415, `np.floating` 161,070, `np.number` 127,949, `np.complex64` 49,398, `np.generic` 45,056, `np.object_` 22,711, `np.str_` 22,473, `np.bytes_` 11,396, `np.void` 11,294, `np.longdouble` 9,639, `np.complexfloating` 8,452, `np.typecodes` 7,181, `np.True_` 4,832, `np.signedinteger` 4,101, `np.False_` 2,343, `np.inexact` 2,137, `np.unsignedinteger` 1,855, `np.character` 1,039, `np.ScalarType` 950, `np.sctypeDict` 727, `np.flexible` 299
- **Floating-point error state:** `np.errstate` 119,043, `np.seterr` 26,369, `np.geterr` 922, `np.seterrcall` 334, `np.geterrcall` 299
- **Dates and times:** `np.datetime64` 58,928, `np.timedelta64` 41,291, `np.datetime_as_string` 2,801, `np.isnat` 2,032, `np.busday_count` 1,361, `np.is_busday` 608, `np.busday_offset` 514, `np.busdaycalendar` 173
- **File I/O and array construction:** `np.genfromtxt` 100,614, `np.memmap (class)` 44,997, `np.fromiter` 31,163, `np.fromfunction` 5,844, `np.from_dlpack` 1,197, `np.fromregex` 159
- **Record arrays:** `np.recarray (class)` 6,156, `np.rec` 3,795, `np.rec.fromarrays` 2,098, `np.rec.array` 1,578, `np.rec.fromrecords` 969, `np.record (class)` 607, `np.rec.fromstring` 457, `np.rec.fromfile` 446
- **Runtime, build and typing:** `numpy.typing (module)` 181,274, `np.__version__` 67,579, `np.ndim` 29,737, `np.get_include` 13,324, `np.core` 8,941, `np.info` 1,695, `np.exceptions` 1,361, `np.show_config` 1,085, `np.test` 17
- **numpy.polynomial** (28 measures, 21,557 files summed): `leggauss` 11,615, `polynomial.Polynomial (via Polynomial.fit)` 2,346, `hermgauss` 1,952, `polynomial.polyfit` 1,214, `chebfit` 922, `polynomial.Chebyshev (via Chebyshev.fit)` 702, `hermegauss` 556, `legfit` 346, `laggauss` 324, `polyvalfromroots` 258, `hermfit` 149, `lagfit` 146, ...
- **numpy.char** (68 measures, 21,587 files summed): `np.char.add` 2,570, `np.char.lower` 1,412, `np.char.decode` 1,243, `np.char.find` 1,216, `np.char.startswith` 1,188, `np.char.replace` 1,123, `np.char.strip` 1,108, `np.char.mod` 1,039, `np.char.array` 716, `np.char.upper` 696, `np.char.equal` 674, `np.char.encode` 665, ...
- **numpy.strings** (91 measures, 4,155 files summed): `np.strings.multiply` 542, `np.strings.find` 159, `np.strings.replace` 155, `np.strings.strip` 154, `np.strings.count` 148, `np.strings.endswith` 148, `np.strings.index` 144, `np.strings.rindex` 144, `np.strings.lstrip` 142, `np.strings.rstrip` 142, `np.strings.expandtabs` 141, `np.strings.str_len` 90, ...
- **ufunc methods** (342 measures, 186,661 files summed): `np.add.at` 53,165, `np.maximum.accumulate` 41,834, `np.maximum.reduce` 10,048, `np.minimum.accumulate` 7,648, `np.logical_and.reduce` 7,376, `np.minimum.reduce` 5,520, `np.add.reduceat` 5,488, `np.logical_or.reduce` 5,472, `np.maximum.at` 4,688, `np.subtract.outer` 4,400, `np.add.reduce` 3,624, `np.minimum.at` 3,049, ...

## Rows without a number

- 338 rows: NumPy rejects every call of this ufunc method
- 27 rows: generic member name or Python protocol: not measurable by phrase
- 3 rows: Python protocol (call / doc / name) of every ufunc
- 2 rows: not attributable: other libraries define functions with the same name
- 2 rows: generic member name (pickle.dump / json.dumps share it)
