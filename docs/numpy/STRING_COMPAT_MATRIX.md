# NumPy Strings vs. NumSharp — Compatibility Matrix

> **Reference (target):** NumPy 2.4.2.
> **NumSharp:** the `NPTypeCode.String` text lane with NumPy's `S`, `U` and `StringDType` descriptors, `np.strings`,
> `np.@char`, `NDCharArray` and `NDListArray`. Every row marked ✅ is replayed against NumPy 2.4.2 by the string
> oracle (`test/NumSharp.Tests.Oracle/Strings/`, one corpus file per section, every case strict).
> **Companion to:** [`NUMPY_STRING_TYPES.md`](./NUMPY_STRING_TYPES.md) (NumPy's full spec) · the user guide
> [Strings, bytes and chararrays](../website-src/docs/fundamentals/strings-and-bytes.md) · tracks
> [#592](https://github.com/SciSharp/NumSharp/issues/592).

**Legend:** ✅ present & NumPy's behavior · 🟡 present with a documented difference · ❌ missing

> Two corrections to the spec doc, found against NumPy 2.4.2 (NumSharp follows NumPy):
> 1. `np.dtype('c')` is **not** deprecated or raising: it returns `dtype('S1')`.
> 2. `S == U` does **not** raise: `==` / `!=` return element-wise `False` / `True`; only the orderings (`<`, `>`) raise.

---

## A. Type system / dtypes

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `bytes_` `'S'` | fixed-width, 1 B/char, NUL-padded | `np.dtype("S5")`: itemsize 5, kind `'S'`, the `NPTypeCode.String` lane | ✅ |
| `str_` `'U'` | fixed-width, 4 B/char (UCS-4) | `np.dtype("U5")`: itemsize 20, native UCS-4; `>U` data is byte-swapped on read (`.npy`, pythonnet) | ✅ |
| `StringDType` `'T'` | variable-width UTF-8 + NA | `np.dtypes.StringDType(na_object, coerce)`, `np.dtype("T")`: NumPy's 16-byte cell | ✅ |
| `'c'` char dtype | = `S1` | `np.dtype("c")` is `S1` | ✅ |
| `NPTypeCode.String` | n/a | the text lane shared by `S`, `U` and `T`; the descriptor carries class, width, NA object and coercion | ✅ |
| `NPTypeCode.Char` | n/a | NumSharp's 2-byte UTF-16 code unit dtype: numbers, not text (written to `.npy` as `<U1`) | 🟡 |
| `object` dtype holding strings | yes | no object dtype; `split` / `rsplit` / `splitlines` return `NDListArray` (NumPy's object array of lists); a NumPy object array of `str` imports as `StringDType` | 🟡 |

## B. Dtype parsing & properties

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `dtype('S'/'S10'/'\|S10')` | `bytes_` | the same descriptors (`'S'` is the unsized `\|S0`) | ✅ |
| `dtype('U'/'U10'/'<U10'/'>U10')` | `str_` (+ byte order) | the same; a non-native order is kept (`np.dtype(">U10").byteorder` is `'>'`) | ✅ |
| `dtype('T')`, `dtype('a')`, `dtype('a5')` | `StringDType` / the deprecated `S` alias | `StringDType()`, `\|S0`, `\|S5` | ✅ |
| `dtype('str')` / `'bytes'` / `'unicode'` | `<U0` / `\|S0` / `<U0` | the same | ✅ |
| `np.str_` / `np.bytes_` as dtypes | the unsized classes | `DType` descriptors `<U0` / `\|S0` (`<U1` / `S1` once allocated) | ✅ |
| `.char` / `.kind` / `.itemsize` / `.name` / `.str` / `.type` | `S`/`U`/`T`, exact bytes, `bytes80`/`str320`/`StringDType128`, `\|S10`/`<U10`/`StringDType()` | NumPy's values | ✅ |
| tuple form `dtype((np.str_, 5))` | yes | not provided: write `np.dtype("U5")` | ❌ |

## C. Scalars & type hierarchy

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `np.bytes_` / `np.str_` scalar objects | yes | no NumPy scalar objects: an element is a C# `string` (`U`, `T`) or `byte[]` (`S`); a 0-d array is the array scalar | 🟡 |
| `np.character` / `np.flexible` bases | yes | `NDScalarType` classes | ✅ |
| `issubdtype(d, np.character / np.flexible / np.str_ / np.bytes_)` | yes | NumPy's answers | ✅ |

## D. Array creation

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `np.array([...], dtype='S'/'U'/'T')` | auto-size, truncate, NUL-pad | the same (`np.array(string[])` is `<U{longest}`) | ✅ |
| from Python `b'...'` bytes | yes | `byte[]` values in an `object[]`, or `NDArray.Scalar(byte[])` | ✅ |
| `np.zeros` / `empty` / `full` with a text dtype | yes | the same | ✅ |
| `new NDArray(typeof(string), shape)` | n/a | a `<U1` array of empty strings | ✅ |
| a C# `string` → `NDArray` | n/a (a Python `str`) | a 0-d `<U{n}` array, as `np.asarray('abc')` | ✅ |
| `GetString` / `SetString` / `GetStringAt` | n/a | read and write `Char` vectors only; text arrays refuse them | 🟡 |

## E. Memory layout

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| Fixed-width NUL-padded buffer (`S`) | yes | the same bytes | ✅ |
| UCS-4 + byte order (`U`) | yes | the same bytes (native order in memory) | ✅ |
| 16 B cell + arena + small-string optimization + NA flags (`T`) | yes | NumPy's cells byte for byte (a port of `static_string.c`); heap strings carved from pooled pages | ✅ |

## F. Casting / conversion

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| number → text (`'1'`, `'1.5'`, `'nan'`, `'inf'`) | the scalar's `str()` (`<U21` for int64) | the same text and widths | ✅ |
| text → number (`'22'` → 22) | Python's `int()` / `float()` / `complex()` | the same parse and error texts | ✅ |
| bool → text (`'True'` / `'False'`) | yes | the same | ✅ |
| text → bool (empty is False, `'False'` is True) | yes | the same | ✅ |
| `S` ↔ `U` ↔ `T`, width truncation, casting safety | yes | NumPy's cast table and Unicode errors | ✅ |
| `datetime64` ↔ text | yes | ❌: NumSharp has no datetime storage | ❌ |

## G. Comparisons

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `==` / `!=` element-wise on text | bool array | the same | ✅ |
| `<`, `<=`, `>`, `>=` lexicographic | yes | the same (`T` with NumPy's `strncmp` order) | ✅ |
| broadcasting in comparisons | yes | the same | ✅ |
| `S` vs `U` (`==` all-False, `<` raises) | yes | the same | ✅ |
| `np.char`'s whitespace-stripping comparisons | yes | the same | ✅ |

## H. String operations — `np.strings` & `np.char`

| Family | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `add` (concatenation), `multiply` (repetition) | yes | over `S`, `U`, `T` | ✅ |
| `str_len` (code points) | yes | the same | ✅ |
| `find` / `rfind` / `index` / `rindex` / `count` / `startswith` / `endswith` | yes | the same | ✅ |
| case: `upper` / `lower` / `capitalize` / `title` / `swapcase` | yes | CPython 3.12's full case mappings | ✅ |
| classify: `isalpha` / `isdigit` / `isalnum` / `isspace` / `islower` / `isupper` / `istitle` / `isnumeric` / `isdecimal` | yes | CPython 3.12's Unicode 15.0.0 tables | ✅ |
| whitespace / pad: `strip` / `lstrip` / `rstrip` / `center` / `ljust` / `rjust` / `zfill` / `expandtabs` | yes | the same | ✅ |
| `replace`, `slice` | yes | the same | ✅ |
| `partition` / `rpartition` | three field views of one structured array | three arrays (no structured dtype; a `U0` part is `U1`) | 🟡 |
| `split` / `rsplit` / `splitlines` / `join` (object arrays) | yes | `NDListArray`, NumPy's object array of lists | 🟡 |
| `encode` / `decode` | CPython's codecs | CPython 3.12's text codecs with every error handler; the CJK multi-byte codecs, `utf-7`, `idna`, `punycode`, the escape codecs and `mbcs` raise `NotSupportedException` naming the codec | 🟡 |
| `mod`, `translate` | yes | CPython 3.12's `%` and `translate` per element | ✅ |
| `np.char.*` (53 names) + `chararray` | yes (legacy) | `np.@char` and `NDCharArray` | ✅ |

## I. Array functions with string support

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `sort` / `argsort` / `partition` (lexicographic) | yes | NumPy's order of equal strings for every `kind` | ✅ |
| `argmax` / `argmin` (lexicographic) | yes | the same | ✅ |
| `searchsorted`, `unique` (and the Array-API `unique_*`), `isin`, the set routines | yes | the same (`unique_values` sorted where NumPy keeps its hash order) | ✅ |
| `concatenate` / `stack` / `vstack` / `hstack` / `dstack` | yes | the same | ✅ |
| `where` / `take` / `nonzero` / `any` / `all` (empty is False) | yes | the same | ✅ |
| `quantile` / `percentile` / `median` | the selection methods take an element; the interpolating ones raise | the same | ✅ |
| `np.sum` / `max` / `min` / `cumsum` | `StringDType` only; `S` / `U` raise | the same | ✅ |

## J. File I/O & structured arrays

| Capability | NumPy 2.4.2 | NumSharp | |
|---|---|---|---|
| `np.save` / `np.load` of a text array | `\|S` / `<U` raw, `StringDType` as a pickle | the same bytes; `StringDType` through NumPy's own pickle (`allow_pickle`) | ✅ |
| `savez` / `.npz` | yes | the same | ✅ |
| `loadtxt` / `savetxt`, `tofile` / `fromfile` / `fromstring` text mode | yes; `fromfile` / `fromstring` refuse `StringDType` | the same, through Python's codecs: `loadtxt` / `savetxt` / text-mode `tofile` take `StringDType`, `fromfile` / `fromstring` refuse it with NumPy's errors | ✅ |
| `genfromtxt` / `fromregex` | yes | ❌ (missing for every dtype) | ❌ |
| memory-mapped `.npy` | `S` / `U` map; `StringDType` refuses | the same | ✅ |
| structured dtypes with string fields, `recarray` | yes | no structured dtypes | ❌ |

---

## Tally (≈80 capabilities)

- ✅ **~66** — the three dtypes, parsing, creation, layout, casts, comparisons, every `np.strings` / `np.char` function,
  `chararray`, the sort / search / set families, `.npy` / text I/O
- 🟡 **~8** — no object dtype (`NDListArray`), no NumPy scalar objects, three-array `partition`, a codec subset, the
  `Char` dtype
- ❌ **~5** — `dtype((np.str_, n))`, `datetime64` ↔ text, `genfromtxt` / `fromregex`, structured / record arrays

## Bottom line

NumSharp holds NumPy's three string dtypes byte for byte and computes every NumPy operation on them, with NumPy's
results and NumPy's errors. What remains belongs to families NumSharp does not model at all: object arrays (beyond the
`NDListArray` of the split functions), NumPy scalar objects, structured / record dtypes, datetime storage and
`genfromtxt`.
