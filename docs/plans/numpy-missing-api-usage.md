# Missing NumPy API, ranked by GitHub usage

Every API the coverage inventory reports missing (generator 1.13.0, NumPy 2.4.2, journey4 `96ff0d99`), with
the number of public Python files on GitHub that use it. 1,737 missing rows; 1,365 carry a
number (from 643 measures), 372 cannot be measured by code search (reasons below).
Measured 2026-10-15 (generator 1.12.0 rows) and 2026-10-16 (rows generator 1.13.0 added) with
`gh api search/code`. Row-level data: `numpy-missing-api-usage.csv` beside this file. APIs marked † had no
catalog row before generator 1.13.0 (see "Detected by generator 1.13.0").

## Implemented since

- **2026-10-16 (journey4): the scalar-type hierarchy and `numpy.typing`** - 20 of the rows below, 963,105 files summed
  (overlapping): `np.integer` (1st), `numpy.typing` (2nd), `numpy.typing.NDArray` (3rd), `np.floating` (4th), `np.number`
  (5th), `np.generic` (12th), `numpy.typing.ArrayLike` (16th), `np.longdouble`, `np.complexfloating`, `np.typecodes`,
  `numpy.typing.DTypeLike`, `np.True_`, `np.signedinteger`, `np.False_`, `np.inexact`, `np.unsignedinteger`,
  `np.character`, `np.sctypeDict`, `np.flexible`, `numpy.typing.NBitBase`. The abstract classes are `NDScalarType`
  objects (NumPy's class surface, `np.issubdtype` / `np.isdtype` / `np.dtype` / `np.finfo` / `np.iinfo` integration),
  and every row but `numpy.typing` and its four members is oracle-verified (the surface tier's scalar-type section,
  3,245 cases; a typing alias names a C# type and has no value to compare).
  Still missing from the family, each for a reason: `np.complex64` and its `Complex64DType` (NumSharp has one complex
  width), `np.object_`, `np.str_`, `np.bytes_`, `np.void` (no such dtypes on journey4; the strings branch has `str_` and
  `bytes_`), and `np.ScalarType` (a tuple of Python built-in classes and NumPy classes, which has no faithful C# form).

## How the numbers were made

- **Phrase per API**, e.g. `"np.genfromtxt"`, `"np.char.upper"`, `"np.maximum.reduce"`. A class whose members are
  called on variables (`recarray`, `memmap`, `chararray`, the polynomial classes, `MaskedRecords`) is measured once,
  at class level, and every member row takes that number.
- **Filter:** `extension:py size:<40000 NOT site-packages in:file,path`, plus `-filename:` for the small source files
  of NumPy, SciPy, pandas, scikit-learn, Matplotlib and Numba that contain the phrase. Committed virtual
  environments and bundles carry these libraries' own tests and docs and otherwise dominate the counts.
- **Prefix matching:** GitHub's API search (the legacy engine) matches the last word of a phrase as a prefix:
  `np.test` also counts `np.testing`, `np.rec` counts `np.reciprocal`/`np.recarray`, `ufunc.reduce` counts
  `reduceat`. Every measure above 700 files of the 1.12.0 rows, and every measure above 100 files of the rows
  1.13.0 added, was sampled (the matched text of page 1, plus the deepest page above 1,000 files); its value is
  count x (literal, non-vendored hits / sampled hits). Literal means case-sensitive: the engine ignores case and
  matches inside longer words (`"ma.core"` also returns `luma.core`). Measures dominated by a colliding word were
  re-counted with `NOT` terms first; `.reduce` measures carry `NOT reduceat`, which makes them lower bounds.
  Smaller measures keep their first-pass value, flagged `unsampled`.
- **Vendored copies:** a hit inside a published library's source tree is not use. For the 1.12.0 rows the
  queries excluded such files by name; for the 1.13.0 rows the samples classify them, including the files in
  NumPy's package root (106 of the first 151 `np.matlib.zeros` hits were copies of NumPy's own `matlib.py`).
- **Spelling:** a dotted phrase counts files that spell the API that way, not files that import the name
  (`from numpy.typing import NDArray`). Where imports dominate, a co-occurrence query counts files that contain
  the module name and the member name (`"numpy.typing" NDArray`); the larger of the two is used. A warning
  NumPy 2.0 moved to `numpy.exceptions` is counted in both spellings, `np.exceptions.X` plus the removed
  `np.X`.
- Values are GitHub's estimates (about +-10% above a few thousand files) and count files, not repositories:
  copied code (course material, cloned repositories) counts every time.

## Top 80

| # | API | Files | Missing rows | How |
|---:|---|---:|---:|---|
| 1 | `np.integer` | 203,415 | 1 | estimate 224768 x 181/200 |
| 2 | `numpy.typing (module)` | 181,274 | 1 | estimate 186880 x 194/200 |
| 3 | `numpy.typing.NDArray` † | 162,286 | 1 | max of co:numpy.typing+NDArray 162286 (estimate 173568 x 187/200); npt.NDArray 61376 (estimate 70144 x 175/200) |
| 4 | `np.floating` | 161,070 | 1 | estimate 166912 x 193/200 |
| 5 | `np.number` | 127,949 | 1 | estimate 130560 x 196/200 |
| 6 | `np.errstate` | 119,043 | 1 | estimate 130816 x 182/200 |
| 7 | `np.genfromtxt` | 100,614 | 1 | estimate 101120 x 199/200 |
| 8 | `np.__version__` | 67,579 | 1 | estimate 68608 x 197/200 |
| 9 | `np.datetime64` | 58,928 | 1 | estimate 81280 x 145/200 |
| 10 | `np.add.at` | 53,165 | 1 | estimate 54528 x 195/200 |
| 11 | `np.complex64` | 49,398 | 1 | estimate 68608 x 144/200 |
| 12 | `np.generic` | 45,056 | 1 | estimate 51200 x 176/200 |
| 13 | `np.memmap (class)` | 44,997 | 72 | estimate 47616 x 189/200 |
| 14 | `np.maximum.accumulate` | 41,834 | 1 | estimate 42688 x 196/200 |
| 15 | `np.timedelta64` | 41,291 | 1 | estimate 55424 x 149/200 |
| 16 | `numpy.typing.ArrayLike` † | 34,090 | 1 | estimate 37054 x 184/200 |
| 17 | `np.fromiter` | 31,163 | 1 | estimate 33152 x 188/200 |
| 18 | `np.ndim` | 29,737 | 1 | estimate 49152 x 121/200 |
| 19 | `np.seterr` | 26,369 | 1 | estimate 27904 x 189/200 |
| 20 | `np.object_` | 22,711 | 1 | estimate 36928 x 123/200 |
| 21 | `np.str_` | 22,473 | 1 | estimate 29376 x 153/200 |
| 22 | `numpy.ma.core (module)` † | 15,990 | 1 | estimate 29888 x 107/200 |
| 23 | `ndarray.ctypes (via arr.ctypes.data_as)` | 14,333 | 1 | estimate 15008 x 191/200 |
| 24 | `np.get_include` | 13,324 | 1 | estimate 15584 x 171/200 |
| 25 | `leggauss` | 11,615 | 1 | estimate 11792 x 197/200 |
| 26 | `np.bytes_` | 11,396 | 1 | estimate 15296 x 149/200 |
| 27 | `np.void` | 11,294 | 1 | estimate 17376 x 130/200 |
| 28 | `np.maximum.reduce` | 10,048 | 1 | NOT reduceat re-count 10,048 (lower bound; the original 11,680 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 29 | `np.longdouble` | 9,639 | 1 | estimate 18016 x 107/200 |
| 30 | `numpy.matlib (module)` † | 9,314 | 1 | estimate 10768 x 173/200 |
| 31 | `np.core` | 8,941 | 1 | estimate 13345 x 134/200 |
| 32 | `np.complexfloating` | 8,452 | 1 | estimate 11904 x 142/200 |
| 33 | `np.minimum.accumulate` | 7,648 | 1 | estimate 7648 x 100/100 |
| 34 | `np.logical_and.reduce` | 7,376 | 1 | NOT reduceat re-count 7,376 (lower bound; the original 7,376 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 35 | `np.typecodes` | 7,181 | 1 | estimate 14656 x 98/200 |
| 36 | `np.recarray (class)` | 6,156 | 72 | estimate 8208 x 150/200 |
| 37 | `np.fromfunction` | 5,844 | 1 | estimate 6184 x 189/200 |
| 38 | `np.minimum.reduce` | 5,520 | 1 | NOT reduceat re-count 5,520 (lower bound; the original 6,528 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 39 | `np.add.reduceat` | 5,488 | 1 | estimate 5488 x 100/100 |
| 40 | `np.logical_or.reduce` | 5,472 | 1 | NOT reduceat re-count 5,472 (lower bound; the original 5,776 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 41 | `numpy.typing.DTypeLike` † | 5,244 | 1 | estimate 7184 x 146/200 |
| 42 | `np.True_` | 4,832 | 1 | NOT re-count; estimate 4832 x 100/100 |
| 43 | `np.maximum.at` | 4,688 | 1 | estimate 4784 x 98/100 |
| 44 | `np.subtract.outer` | 4,400 | 1 | estimate 4400 x 100/100 |
| 45 | `numpy.version (module)` † | 4,267 | 1 | sum of np.version 2715 (estimate 4052 x 134/200); numpy.version 1552 (estimate 4032 x 77/200) |
| 46 | `np.signedinteger` | 4,101 | 1 | estimate 6408 x 128/200 |
| 47 | `np.ufunc` | 3,872 | 1 | estimate 7376 x 105/200 |
| 48 | `np.exceptions.VisibleDeprecationWarning` † | 3,831 | 1 | sum of exceptions.VisibleDeprecationWarning 480 (estimate 495 x 97/100); np.VisibleDeprecationWarning(1.x) 3351 (estimate 4296 x 156/200) |
| 49 | `np.rec` | 3,795 | 1 | NOT re-count; estimate 3912 x 97/100 |
| 50 | `np.matlib.repmat` † | 3,653 | 1 | estimate 4624 x 158/200 |
| 51 | `np.add.reduce` | 3,624 | 1 | NOT reduceat re-count 3,624 (lower bound; the original 9,472 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 52 | `np.minimum.at` | 3,049 | 1 | estimate 3080 x 99/100 |
| 53 | `np.datetime_as_string` | 2,801 | 1 | estimate 3112 x 180/200 |
| 54 | `np.multiply.outer` | 2,764 | 1 | estimate 2792 x 99/100 |
| 55 | `np.char.add` | 2,570 | 1 | estimate 2720 x 189/200 |
| 56 | `np.version.version` † | 2,538 | 1 | sum of np.version.version 1613 (estimate 1812 x 178/200); numpy.version.version 925 (estimate 934 x 99/100) |
| 57 | `np.add.outer` | 2,448 | 1 | estimate 2448 x 100/100 |
| 58 | `np.exceptions.ComplexWarning` † | 2,404 | 1 | sum of exceptions.ComplexWarning 1380 (estimate 2604 x 106/200); np.ComplexWarning(1.x) 1024 (estimate 1896 x 108/200) |
| 59 | `polynomial.Polynomial (via Polynomial.fit)` | 2,346 | 28 | estimate 2496 x 188/200 |
| 60 | `np.False_` | 2,343 | 1 | estimate 2840 x 165/200 |
| 61 | `finfo.nmant` | 2,236 | 1 | estimate 6577 x 68/200 |
| 62 | `np.logaddexp.reduce` | 2,164 | 1 | NOT reduceat re-count 2,164 (lower bound; the original 2,156 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 63 | `np.inexact` | 2,137 | 1 | estimate 4912 x 87/200 |
| 64 | `np.rec.fromarrays` | 2,098 | 1 | estimate 2512 x 167/200 |
| 65 | `np.isnat` | 2,032 | 1 | estimate 2764 x 147/200 |
| 66 | `np.exceptions.RankWarning` † | 1,969 | 1 | sum of exceptions.RankWarning 640 (estimate 970 x 66/100); np.RankWarning(1.x) 1329 (estimate 1682 x 158/200) |
| 67 | `hermgauss` | 1,952 | 1 | estimate 2088 x 187/200 |
| 68 | `np.maximum.reduceat` | 1,908 | 1 | estimate 1908 x 100/100 |
| 69 | `np.unsignedinteger` | 1,855 | 1 | estimate 2728 x 136/200 |
| 70 | `np.add.accumulate` | 1,696 | 1 | estimate 1696 x 100/100 |
| 71 | `np.info` | 1,695 | 1 | estimate 2648 x 128/200 |
| 72 | `np.multiply.reduce` | 1,648 | 1 | NOT reduceat re-count 1,648 (lower bound; the original 1,732 also counts files using reduceat, and the two differ by estimate noise when reduceat is rare); ufunc study (vendored-filtered, literal not sampled) |
| 73 | `np.rec.array` | 1,578 | 1 | estimate 3712 x 85/200 |
| 74 | `numpy.ma.extras (module)` † | 1,421 | 1 | estimate 2608 x 109/200 |
| 75 | `np.char.lower` | 1,412 | 1 | estimate 1412 x 200/200 |
| 76 | `np.busday_count` | 1,361 | 1 | estimate 1768 x 154/200 |
| 77 | `np.exceptions` | 1,361 | 1 | estimate 1972 x 138/200 |
| 78 | `np.minimum.reduceat` | 1,307 | 1 | estimate 1334 x 98/100 |
| 79 | `np.char.decode` | 1,243 | 1 | estimate 1480 x 168/200 |
| 80 | `np.char.find` | 1,216 | 1 | estimate 1222 x 199/200 |

## By family

- **Scalar types and the abstract dtype hierarchy:** `np.integer` 203,415, `np.floating` 161,070, `np.number` 127,949, `np.complex64` 49,398, `np.generic` 45,056, `np.object_` 22,711, `np.str_` 22,473, `np.bytes_` 11,396, `np.void` 11,294, `np.longdouble` 9,639, `np.complexfloating` 8,452, `np.typecodes` 7,181, `np.True_` 4,832, `np.signedinteger` 4,101, `np.False_` 2,343, `np.inexact` 2,137, `np.unsignedinteger` 1,855, `np.character` 1,039, `np.ScalarType` 950, `np.sctypeDict` 727, `np.flexible` 299
- **Dtype classes (numpy.dtypes):** `np.dtypes.StringDType` 892, `np.dtypes.ObjectDType` 330, `np.dtypes.StrDType` 92, `np.dtypes.BytesDType` 36, `np.dtypes.VoidDType` 27, `np.dtypes.Complex64DType` 14
- **Floating-point error state:** `np.errstate` 119,043, `np.seterr` 26,369, `np.geterr` 922, `np.seterrcall` 334, `np.geterrcall` 299
- **Exceptions and warnings (numpy.exceptions):** `np.exceptions.VisibleDeprecationWarning` 3,831, `np.exceptions.ComplexWarning` 2,404, `np.exceptions.RankWarning` 1,969, `np.exceptions.ModuleDeprecationWarning` 3
- **Dates and times:** `np.datetime64` 58,928, `np.timedelta64` 41,291, `np.datetime_as_string` 2,801, `np.isnat` 2,032, `np.busday_count` 1,361, `np.is_busday` 608, `np.busday_offset` 514, `np.busdaycalendar` 173
- **File I/O and array construction:** `np.genfromtxt` 100,614, `np.memmap (class)` 44,997, `np.fromiter` 31,163, `np.fromfunction` 5,844, `np.from_dlpack` 1,197, `np.fromregex` 159
- **Record arrays:** `np.recarray (class)` 6,156, `np.rec` 3,795, `np.rec.fromarrays` 2,098, `np.rec.array` 1,578, `np.rec.fromrecords` 969, `np.record (class)` 607, `np.rec.fromstring` 457, `np.rec.fromfile` 446, `np.rec.recarray` 424, `np.rec.format_parser` 73, `np.rec.record` 2
- **Masked record arrays and numpy.ma modules:** `numpy.ma.core (module)` 15,990, `numpy.ma.extras (module)` 1,421, `numpy.ma.mrecords (module; MaskedRecords and its members)` 280, `np.ma.mrecords.fromtextfile` 113, `np.ma.mvoid` 25, `np.ma.mrecords.mrecarray` 5, `np.ma.mrecords.fromarrays` 2, `np.ma.mrecords.fromrecords` 0, `np.ma.mrecords.addfield` 0
- **Matrix library (numpy.matlib, deprecated by NumPy):** `numpy.matlib (module)` 9,314, `np.matlib.repmat` 3,653, `np.matlib.identity` 572, `np.matlib.zeros` 567, `np.matlib.eye` 328, `np.matlib.ones` 296, `np.matlib.empty` 266, `np.matlib.rand` 232, `np.matlib.randn` 136
- **Runtime, build and typing:** `numpy.typing (module)` 181,274, `numpy.typing.NDArray` 162,286, `np.__version__` 67,579, `numpy.typing.ArrayLike` 34,090, `np.ndim` 29,737, `np.get_include` 13,324, `np.core` 8,941, `numpy.typing.DTypeLike` 5,244, `numpy.version (module)` 4,267, `np.version.version` 2,538, `np.info` 1,695, `np.exceptions` 1,361, `np.show_config` 1,085, `np.version.full_version` 308, `np.version.short_version` 236, `numpy.typing.NBitBase` 116, `np.version.release` 78, `np.f2py.get_include` 75, `np.f2py.run_main` 49, `np.test` 17, `np.version.git_revision` 7
- **numpy.polynomial** (28 measures, 21,557 files summed): `leggauss` 11,615, `polynomial.Polynomial (via Polynomial.fit)` 2,346, `hermgauss` 1,952, `polynomial.polyfit` 1,214, `chebfit` 922, `polynomial.Chebyshev (via Chebyshev.fit)` 702, `hermegauss` 556, `legfit` 346, `laggauss` 324, `polyvalfromroots` 258, `hermfit` 149, `lagfit` 146, ...
- **numpy.char** (68 measures, 21,587 files summed): `np.char.add` 2,570, `np.char.lower` 1,412, `np.char.decode` 1,243, `np.char.find` 1,216, `np.char.startswith` 1,188, `np.char.replace` 1,123, `np.char.strip` 1,108, `np.char.mod` 1,039, `np.char.array` 716, `np.char.upper` 696, `np.char.equal` 674, `np.char.encode` 665, ...
- **numpy.strings** (91 measures, 4,155 files summed): `np.strings.multiply` 542, `np.strings.find` 159, `np.strings.replace` 155, `np.strings.strip` 154, `np.strings.count` 148, `np.strings.endswith` 148, `np.strings.index` 144, `np.strings.rindex` 144, `np.strings.lstrip` 142, `np.strings.rstrip` 142, `np.strings.expandtabs` 141, `np.strings.str_len` 90, ...
- **ufunc objects (methods and attributes)** (335 measures, 181,184 files summed): `np.add.at` 53,165, `np.maximum.accumulate` 41,834, `np.maximum.reduce` 10,048, `np.minimum.accumulate` 7,648, `np.logical_and.reduce` 7,376, `np.minimum.reduce` 5,520, `np.add.reduceat` 5,488, `np.logical_or.reduce` 5,472, `np.maximum.at` 4,688, `np.subtract.outer` 4,400, `np.add.reduce` 3,624, `np.minimum.at` 3,049, ...

## Detected by generator 1.13.0

The 2026-10-15 ranking measured the rows the inventory had. Those counts could not reveal APIs the inventory
never catalogued, so the inventory was audited against two NumPy sources of its own (commit `96ff0d99`): NumPy's
public-module list (`numpy/tests/test_public_api.py`) and its documentation index (every `numpy.*` object of the
Sphinx inventory). 462 documented names that exist in NumPy 2.4.2 had no row: whole modules were never scanned
(`numpy.dtypes`, `numpy.exceptions`, `numpy.typing`, `numpy.version`, `numpy.ma.mrecords`, `numpy.matlib`,
`numpy.f2py`), the submodule scan kept only callables (so `numpy.ma.masked` and the polynomial domain constants had
no rows), and four public classes had no member rows. Generator 1.13.0 catalogues them, and two CI tests now hold
the catalog to both NumPy lists.

146 of the newly catalogued rows are missing in NumSharp and appear on the dashboard. 138 have measures of their own (43 measures, below); the other 8 share an existing class-level measure (listed after the table).

| Rank | API | Files | Missing rows | How |
|---:|---|---:|---:|---|
| 3 | `numpy.typing.NDArray` | 162,286 | 1 | max of co:numpy.typing+NDArray 162286 (estimate 173568 x 187/200); npt.NDArray 61376 (estimate 70144 x 175/200) |
| 16 | `numpy.typing.ArrayLike` | 34,090 | 1 | estimate 37054 x 184/200 |
| 22 | `numpy.ma.core (module)` | 15,990 | 1 | estimate 29888 x 107/200 |
| 30 | `numpy.matlib (module)` | 9,314 | 1 | estimate 10768 x 173/200 |
| 41 | `numpy.typing.DTypeLike` | 5,244 | 1 | estimate 7184 x 146/200 |
| 45 | `numpy.version (module)` | 4,267 | 1 | sum of np.version 2715 (estimate 4052 x 134/200); numpy.version 1552 (estimate 4032 x 77/200) |
| 48 | `np.exceptions.VisibleDeprecationWarning` | 3,831 | 1 | sum of exceptions.VisibleDeprecationWarning 480 (estimate 495 x 97/100); np.VisibleDeprecationWarning(1.x) 3351 (estimate 4296 x 156/200) |
| 50 | `np.matlib.repmat` | 3,653 | 1 | estimate 4624 x 158/200 |
| 56 | `np.version.version` | 2,538 | 1 | sum of np.version.version 1613 (estimate 1812 x 178/200); numpy.version.version 925 (estimate 934 x 99/100) |
| 58 | `np.exceptions.ComplexWarning` | 2,404 | 1 | sum of exceptions.ComplexWarning 1380 (estimate 2604 x 106/200); np.ComplexWarning(1.x) 1024 (estimate 1896 x 108/200) |
| 66 | `np.exceptions.RankWarning` | 1,969 | 1 | sum of exceptions.RankWarning 640 (estimate 970 x 66/100); np.RankWarning(1.x) 1329 (estimate 1682 x 158/200) |
| 74 | `numpy.ma.extras (module)` | 1,421 | 1 | estimate 2608 x 109/200 |
| 98 | `np.dtypes.StringDType` | 892 | 1 | max of dtypes.StringDType 892 (estimate 1716 x 104/200); co:numpy.dtypes+StringDType 347 (estimate 780 x 44/99) |
| 115 | `np.matlib.identity` | 572 | 1 | estimate 2288 x 50/200 |
| 116 | `np.matlib.zeros` | 567 | 1 | estimate 2412 x 47/200 |
| 131 | `np.rec.recarray` | 424 | 1 | estimate 1664 x 51/200 |
| 138 | `np.dtypes.ObjectDType` | 330 | 1 | estimate 674 x 49/100 |
| 139 | `np.matlib.eye` | 328 | 1 | estimate 2344 x 28/200 |
| 143 | `np.version.full_version` | 308 | 1 | sum of np.version.full_version 265 (estimate 270 x 98/100); numpy.version.full_version 43 (unsampled first-pass count) |
| 147 | `np.matlib.ones` | 296 | 1 | estimate 2364 x 25/200 |
| 148 | `numpy.ma.mrecords (module; MaskedRecords and its members)` | 280 | 96 | estimate 2944 x 19/200 |
| 151 | `np.matlib.empty` | 266 | 1 | estimate 2216 x 24/200 |
| 156 | `np.version.short_version` | 236 | 1 | sum of np.version.short_version 178 (estimate 742 x 24/100); numpy.version.short_version 58 (estimate 135 x 43/100) |
| 157 | `np.matlib.rand` | 232 | 1 | estimate 357 x 65/100 |
| 191 | `np.matlib.randn` | 136 | 1 | estimate 2272 x 12/200 |
| 199 | `numpy.typing.NBitBase` | 116 | 1 | estimate 3320 x 7/200 |
| 203 | `np.ma.mrecords.fromtextfile` | 113 | 1 | estimate 940 x 12/100 |
| 215 | `np.dtypes.StrDType` | 92 | 1 | unsampled first-pass count |
| 244 | `np.version.release` | 78 | 1 | sum of np.version.release 78 (estimate 970 x 8/100); numpy.version.release 0 (unsampled first-pass count) |
| 252 | `np.f2py.get_include` | 75 | 1 | estimate 680 x 11/100 |
| 254 | `np.rec.format_parser` | 73 | 1 | estimate 1216 x 12/200 |
| 267 | `np.f2py.run_main` | 49 | 1 | estimate 545 x 9/100 |
| 279 | `np.dtypes.BytesDType` | 36 | 1 | unsampled first-pass count |
| 289 | `np.dtypes.VoidDType` | 27 | 1 | unsampled first-pass count |
| 293 | `np.ma.mvoid` | 25 | 1 | estimate 496 x 5/100 |
| 315 | `np.dtypes.Complex64DType` | 14 | 1 | unsampled first-pass count |
| 341 | `np.version.git_revision` | 7 | 1 | sum of np.version.git_revision 6 (unsampled first-pass count); numpy.version.git_revision 1 (unsampled first-pass count) |
| 367 | `np.ma.mrecords.mrecarray` | 5 | 1 | unsampled first-pass count |
| 398 | `np.exceptions.ModuleDeprecationWarning` | 3 | 1 | unsampled first-pass count |
| 405 | `np.ma.mrecords.fromarrays` | 2 | 1 | unsampled first-pass count |
| 406 | `np.rec.record` | 2 | 1 | unsampled first-pass count |
| 640 | `np.ma.mrecords.addfield` | 0 | 1 | unsampled first-pass count |
| 641 | `np.ma.mrecords.fromrecords` | 0 | 1 | unsampled first-pass count |

Rows that share an existing class-level measure: `polynomial.Chebyshev (via Chebyshev.fit)`, `polynomial.Hermite (via Hermite.fit)`, `polynomial.HermiteE (via HermiteE.fit)`, `polynomial.Laguerre (via Laguerre.fit)`, `polynomial.Legendre (via Legendre.fit)`, `polynomial.Polynomial (via Polynomial.fit)`, `np.busdaycalendar` (the polynomial classes' `numpy.polynomial.<Class>` aliases and `busdaycalendar`'s members).

The same change credited 66 APIs NumSharp already implements, which had read as NumSharp-only: 27 `np.dtypes`
classes, 8 `np.ma` objects (`masked`, `nomask`, `mr_`, `masked_array`, ...), the 24 polynomial domain/zero/one/x
constants and the 7 polynomial submodules; `numpy.exceptions.AxisError`, `DTypePromotionError`, `TooHardError` and
`numpy.ma.MAError`/`MaskError` are credited to their NumSharp types.

Not catalogued, as open decisions (the CI guard exempts exactly these two classes): Python protocol dunders on
the object surfaces (operators, container and pickling hooks: `ndarray.__matmul__`, `ndarray.__getitem__`, ...;
847 over the 32 catalogued classes) and the members of NumPy's scalar types (`np.float64(x).is_integer()`, ...).
Neither is measurable by phrase: `a @ b` and `a[i]` have no searchable spelling.

### Outside the dashboard (measured for context)

The dashboard projection hides `numpy.testing`, `numpy.lib`, `numpy.ctypeslib`, the ndarray interop dunders and
`numpy.__array_namespace_info__` (Python tooling). Their rows exist and are missing in NumSharp; a few measures:

| API | Files | How |
|---|---:|---|
| `np.testing.assert_allclose` | 334,492 | testing.assert_allclose 334,492 (estimate 346624 x 193/200); co:numpy.testing+assert_allclose 36,759 (estimate 88576 x 83/200) |
| `np.testing.assert_array_equal` | 253,673 | testing.assert_array_equal 253,673 (estimate 257536 x 197/200); co:numpy.testing+assert_array_equal 21,548 (estimate 113408 x 38/200) |
| `np.testing.assert_equal` | 41,181 | testing.assert_equal 41,181 (estimate 45504 x 181/200) |
| `np.testing.assert_array_almost_equal` | 38,331 | testing.assert_array_almost_equal 38,331 (estimate 41216 x 186/200) |
| `np.testing.assert_almost_equal` | 33,792 | testing.assert_almost_equal 33,792 (estimate 38400 x 176/200) |
| `np.lib.stride_tricks.as_strided` | 19,879 | stride_tricks.as_strided 15,130 (estimate 15760 x 192/200); co:stride_tricks+as_strided 19,879 (estimate 22336 x 178/200) |
| `np.lib.stride_tricks.sliding_window_view` | 13,056 | stride_tricks.sliding_window_view 9,046 (estimate 9472 x 191/200); co:stride_tricks+sliding_window_view 13,056 (estimate 15360 x 170/200) |
| `np.ctypeslib.as_array` | 7,120 | ctypeslib.as_array 7,120 (estimate 7824 x 182/200) |
| `np.ctypeslib.ndpointer` | 4,339 | ctypeslib.ndpointer 4,339 (estimate 5016 x 173/200) |
| `np.testing.assert_raises` | 2,250 | testing.assert_raises 2,250 (estimate 2432 x 185/200) |
| `np.lib.recfunctions.structured_to_unstructured` | 662 | co:recfunctions+structured_to_unstructured 662 (estimate 1764 x 75/200) |
| `np.lib.NumpyVersion` | 635 | lib.NumpyVersion 635 (estimate 934 x 68/100) |
| `np.lib.recfunctions.append_fields` | 532 | co:recfunctions+append_fields 532 (estimate 2364 x 45/200) |
| `np.lib.mixins.NDArrayOperatorsMixin` | 408 | co:mixins+NDArrayOperatorsMixin 408 (estimate 1236 x 66/200) |

## Rows without a number

- 338 rows: NumPy rejects every call of this ufunc method
- 27 rows: generic member name or Python protocol: not measurable by phrase
- 3 rows: Python protocol (call / doc / name) of every ufunc
- 2 rows: not attributable: other libraries define functions with the same name
- 2 rows: generic member name (pickle.dump / json.dumps share it)
