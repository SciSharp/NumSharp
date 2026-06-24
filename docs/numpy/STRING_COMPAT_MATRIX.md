# NumPy Strings vs. NumSharp — Compatibility Matrix

> **Reference (target):** NumPy 2.4.2 (verified live).
> **"NumSharp — now":** current `nditer` branch (`f7be5f7e`), verified in source.
> **Companion to:** [`NUMPY_STRING_TYPES.md`](./NUMPY_STRING_TYPES.md) (full spec) · tracks [#592](https://github.com/SciSharp/NumSharp/issues/592).

**Legend:** ✅ present & compatible · 🟡 partial / hack / divergent · ❌ missing or throws

> **Two spec corrections found while ground-truthing against live NumPy 2.4.2** (the spec doc is otherwise accurate):
> 1. `np.dtype('c')` is **not** deprecated/raising — it returns `dtype('S1')`. (NumSharp currently throws on `'c'`.)
> 2. `S == U` does **not** raise — `==`/`!=` return element-wise `[False…]`; only ordering comparisons (`<`,`>`) raise.

---

## A. Type system / dtypes

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `bytes_` `'S'` (enum 18) | fixed-width, 1 B/char, null-padded | no `NPTypeCode.Bytes`; `dtype('S')` throws `NotSupportedException` | ❌ |
| `str_` `'U'` (enum 19) | fixed-width, 4 B/char (UTF-32) | no `NPTypeCode.Unicode`; `dtype('U')` throws | ❌ |
| `StringDType` `'T'` (enum 2056) | variable UTF-8 + NA | no type; `dtype('T')` throws ("cannot parse") | ❌ |
| `'c'` char dtype | = `S1` (1 byte) | `dtype('c')` throws (explicitly in `_unsupported_numpy_codes`) | ❌ |
| `NPTypeCode.Char` | n/a | exists = C# `char` (**UTF-16, 2 B**, reported size 1) | 🟡 |
| `NPTypeCode.String` (18) | n/a (NumSharp-only) | in enum, maps `typeof(string)`, `SizeOf=1`, barely wired | 🟡 |
| `object` dtype holding strings | yes | no object dtype | ❌ |

## B. Dtype parsing & properties

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `dtype('S'/'S10'/'\|S10')` | bytes_ | throws | ❌ |
| `dtype('U'/'U10'/'<U10'/'>U10')` | str_ (+byteorder) | throws | ❌ |
| `dtype('T')`, `dtype('a')` | StringDType / legacy-S | throws | ❌ |
| `dtype('string')` / `dtype('char')` | n/a | → `typeof(string)` / `typeof(char)` (NumSharp-only aliases) | 🟡 |
| tuple form `dtype((np.str_,5))` | yes | not supported | ❌ |
| `.char` / `.kind` = `S`/`U`/`T` | yes | `DType.kind` hardcodes `'S'` for Char & String; wrong for real strings | 🟡 |
| `.itemsize` (exact bytes) | `S10`→10, `U10`→40, `T`→16 | n/a for strings | ❌ |
| `.name` (`bytes80`/`str320`/`StringDType128`) | yes | `"string"` only | ❌ |
| `.str` (`\|S10`/`<U10`/`\|T16`) | yes | none | ❌ |
| `.type` (scalar class) | yes | none | ❌ |

## C. Scalars & type hierarchy

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `np.bytes_` / `np.str_` scalar types | yes | none | ❌ |
| `np.character` / `np.flexible` bases | yes | none | ❌ |
| `issubdtype(d, np.character/flexible/bytes_/str_)` | yes | not applicable (types absent) | ❌ |

## D. Array creation

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `np.array([...], dtype='S'/'U'/'T')` | auto-size, truncate, null-pad | throws (dtype) | ❌ |
| `np.array(string[])` | real `<U` array | **metadata hack**: packs to char vector `"N l1 l2:contents"` | 🟡 |
| from Python `b'...'` bytes | yes | no bytes path | ❌ |
| `np.zeros/empty/full(dtype='U10')` | yes | throws | ❌ |
| `new NDArray(typeof(string), shape)` | n/a | throws `NotSupportedException` (issue #341) | ❌ |
| single `string` → char vector (`FromString`, implicit op) | n/a | works (shape `(n,)` char) | ✅ |
| `GetString`/`SetString`/`GetStringAt` on char vectors | n/a | works | ✅ |

## E. Memory layout

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| Fixed-width null-padded buffer (S) | yes | none | ❌ |
| UTF-32/UCS-4 + byte-order (U) | yes | none | ❌ |
| 16 B header + arena + small-string-opt + NA flags (T) | yes | none | ❌ |
| Contiguous UTF-16 chars | n/a | the only thing we store | 🟡 |

## F. Casting / conversion

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| num → str (`'1'`,`'1.5'`,`'nan'`,`'inf'`) | yes (`<U21` for int64) | none | ❌ |
| str → num (`'22'`→22) | yes | none | ❌ |
| bool → str (`'True'`/`'False'`) | yes | none | ❌ |
| str → bool (**empty=False, `'False'`→True**) | yes | none | ❌ |
| `S↔U↔T`, width truncation, casting safety | yes | none | ❌ |
| `datetime64 ↔ str` | yes | none (no datetime64) | ❌ |

## G. Comparisons

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `==`/`!=` element-wise on strings | bool array | char vectors compare as **uint16 numbers**, not strings | 🟡 |
| `<`,`<=`,`>`,`>=` lexicographic | yes | numeric on char only | ❌ |
| broadcasting in comparisons | yes | n/a for strings | ❌ |
| `S` vs `U` mismatch (`==`→all-False, `<`→raises) | yes | n/a | ❌ |

## H. String operations — `np.strings` (46 ufuncs) & `np.char`

| Family | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `add` (concat), `multiply` (repeat) | yes | **entire namespace absent** | ❌ |
| `str_len` (code points) | yes | — | ❌ |
| `find/rfind/index/rindex/count/startswith/endswith` | yes | — | ❌ |
| case: `upper/lower/capitalize/title/swapcase` | yes | — | ❌ |
| classify: `isalpha/isdigit/isalnum/isspace/islower/isupper/istitle/isnumeric/isdecimal` | yes | — | ❌ |
| whitespace/pad: `strip/lstrip/rstrip/center/ljust/rjust/zfill/expandtabs` | yes | — | ❌ |
| `replace`, `slice` | yes | — | ❌ |
| `partition/rpartition` | yes | — | ❌ |
| `split/rsplit/splitlines/join` (object arrays) | yes | — | ❌ |
| `encode/decode`, `mod`, `translate` | yes | — | ❌ |
| `np.char.*` legacy + `chararray` | yes (deprecated) | — | ❌ |

## I. Array functions with string support

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `sort`/`argsort` (lexicographic) | yes | `np.sort` missing entirely; `argsort` numeric-only | ❌ |
| `argmax`/`argmin` (lexicographic) | yes | numeric-only | ❌ |
| `searchsorted`, `unique` | yes | numeric-only | ❌ |
| `concatenate`/`stack`/`vstack`/`hstack`/`dstack` | yes | n/a for strings | ❌ |
| `where`/`take`/`nonzero`/`any`/`all` (empty=False) | yes | n/a for strings | ❌ |

## J. File I/O & structured arrays

| Capability | NumPy 2.4.2 | NumSharp — now | |
|---|---|---|---|
| `np.save` string array → `.npy` | `\|S`/`<U`/`\|T` preserved | managed `string[]` only → `\|S{maxlen}`, **ASCII-lossy** `(byte)s[i]`, no `U`/`T` | 🟡 |
| `np.load` `.npy` string | dtype preserved as NDArray | reads `\|S` → managed `string[]` (null-terminated ASCII); no `U`/`T`; not a string-dtype NDArray | 🟡 |
| `savez`/`npz`, `loadtxt`/`savetxt`/`genfromtxt`, `tofile`/`fromfile`, `memmap` | yes | none for strings | ❌ |
| structured dtypes w/ string fields, `recarray` | yes | no structured dtypes | ❌ |

---

## Tally (≈70 capabilities)

- ✅ **~3** — single-string ↔ char-vector helpers
- 🟡 **~9** — char fakery, `string[]` packing hack, ASCII `.npy` round-trip, fake dtype props
- ❌ **~58** — all three real dtypes, every `np.strings`/`np.char` op, casting, comparisons, structured/text I/O

## Bottom line

NumSharp today has **no real string dtype** — only a `char` (UTF-16) vector dressed up with helper methods, one brittle `string[]→char` packing hack, and an ASCII-only `.npy` `string[]` path that bypasses the NDArray dtype system entirely. The two "🟡 file I/O" rows are the only places a multi-string array survives a round trip, and only for ASCII.

---

*Generated from NumSharp `nditer` @ `f7be5f7e` and NumPy v2.4.2 live verification.*
