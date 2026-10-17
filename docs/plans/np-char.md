# `numpy.char` to completion — implementation plan

> Plan for the whole `numpy.char` surface of NumPy 2.4.2: the module's 52 functions, the `chararray` class and its 108
> documented members — the 162 rows the coverage inventory lists under `numpy.char`. Today every row reads "missing":
> NumSharp has no `np.char` facade and no `chararray`.
>
> Two thirds of `numpy.char` IS `numpy.strings`: 36 of its 52 functions are the same objects. Their kernels belong to
> Stage D phase 2 (`docs/plans/strings-phase2.md`; P2.0 and the core of P2.1 landed in `7c817644`). This plan covers
> what `numpy.char` adds: the stripping comparisons, three wrappers with their own rules, four functions that return
> Python objects, the `chararray` class, `np.char.array` / `np.char.asarray`, and the facade itself. It takes over the
> `np.char` items of the phase-2 plan (P2.1's "rstrip comparisons with the `np.char` module", P2.6's "`np.char` extras").
>
> Source of truth: `refs/numpy/` (**2.4.2**) — `numpy/_core/defchararray.py`, `numpy/_core/strings.py`,
> `numpy/_core/src/multiarray/multiarraymodule.c` (`compare_chararrays`, `_vec_string`),
> `numpy/_core/src/umath/string_ufuncs.cpp` and `string_buffer.h` (`Buffer::rstrip`, `strcmp`). All behaviors and
> timings below were probed against `numpy==2.4.2` on CPython 3.12.12 (win-amd64, i9-13900K, two P-cores pinned with
> `start /affinity 14`) on 2026-10-09.

---

## 0. Scope and starting point

"Complete" means: each of the 162 inventory rows (the module, 52 functions, the class, 108 members) exists with NumPy's
names, parameter order and defaults, computes NumPy 2.4.2's answer for every input the string oracle can express, and
is oracle-verified in the coverage artifact.

| Part | State at `7c817644` |
|---|---|
| `np.char` facade | None. `generate_coverage.py` credits a `numpy.char` row only through a facade annotated `[ModuleName("np.char")]`. |
| Shared text kernels | Phase 2: P2.0's IL kernels (`Backends/Kernels/ILKernelGenerator.Text{,.Emit}.cs`) and P2.1's six comparisons and truth values are done; P2.2–P2.7 are open. |
| Stripping comparisons | Open (the phase-2 state log moved them to "the `np.char` module"). |
| `chararray` | None. `coverage/object_surfaces.py` lists `char.chararray` with disposition `subsystem` and no target. |
| Object results | None. NumSharp has no object dtype (`split`, `rsplit`, `splitlines` return object arrays of lists in NumPy). |

**Since 2026-10-16 (phases 1–4, see the state log at the end of §7):** `np.@char` is the module (C-D1). C1 (the
stripping comparisons, `compare_chararrays`), `add` and C4's `multiply` compute. `chararray` is `NDCharArray`: its data
rules (phase 2: the constructors, `np.char.array` / `asarray`, the finalize rule, stripped element access, views and
copies that keep the class) and its behavior (phase 3: the operators with Python's dispatch, the sort and search
members, the finalize errors of the index members, and the `np.*` functions that keep, wrap or drop the class) are done.
C3, the object path (`join`, `split`, `rsplit`, `splitlines` and their chararray members, with C-D3's `NDListArray`), is
done (phase 4). C4 is done (`partition` / `rpartition`, 2026-10-17, `ffa74606`), and C5's string methods land in
batches, one family each, as their phase-2 functions compute: batch 1 (`rpartition`, `center`, `ljust`, `rjust`,
`zfill`, `expandtabs`, with the module's re-exports of the last five), batch 2 (the search family: `count`, `find`,
`rfind`, `index`, `rindex`, `startswith`, `endswith`, with the module's re-exports and `np.@char.str_len`; `6d06fa3c`),
batch 3 (`strip`, `lstrip`, `rstrip`, `replace`, with the module's re-exports; `66b68e96`) and batch 4 (the nine
predicates `isalnum`, `isalpha`, `isdecimal`, `isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper`, with
the module's re-exports; `b1328a4f`) are done. Open: the other string methods of C5 — the five case maps compute in
phase 2 already (P2.6 milestone 2) and `translate` since P2.6 milestone 3's first half (`5bbd65c2`); `encode`, `decode`
and `%` wait for the rest of P2.6.

---

## 1. What NumPy 2.4.2 does

### 1.1 The module: 53 names, three origins

`numpy.char` is `numpy._core.defchararray`. Its `__all__` has 53 names (`np.strings.slice` is the one `np.strings` name
it lacks):

| Origin | Names | Count |
|---|---|---|
| The same objects as `np.strings` (`from numpy.strings import *`) | `add`, `capitalize`, `center`, `count`, `decode`, `encode`, `endswith`, `expandtabs`, `find`, `index`, `isalnum`, `isalpha`, `isdecimal`, `isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper`, `ljust`, `lower`, `lstrip`, `mod`, `replace`, `rfind`, `rindex`, `rjust`, `rstrip`, `startswith`, `str_len`, `strip`, `swapcase`, `title`, `translate`, `upper`, `zfill` | 36 |
| Defined in `defchararray.py` | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `multiply`, `partition`, `rpartition`, `array`, `asarray`, the class `chararray` | 12 |
| Private `np.strings` functions, public only here | `join` (`_join`), `split` (`_split`), `rsplit` (`_rsplit`), `splitlines` (`_splitlines`); `compare_chararrays` (C) | 5 |

### 1.2 The stripping comparisons (`compare_chararrays`)

`equal(x1, x2)` is `compare_chararrays(x1, x2, '==', True)`, and so on for `!=`, `<`, `<=`, `>`, `>=`. There is no
`out=`, `where=` or `dtype=`.

- `compare_chararrays(a1, a2, cmp, rstrip)` parses `cmp` first: one or two characters, `str` or `bytes`, one of
  `'=='`, `'!='`, `'<'`, `'<='`, `'>'`, `'>='`; anything else (`'='`, `'=<'`, `'<>'`, `''`) raises
  `ValueError("comparison must be '==', '!=', '<', '>', '<=', '>='")`. `rstrip` is required
  (`TypeError: compare_chararrays() missing required argument 'rstrip' (pos 4)`) and read with Python truthiness
  (`[]` False, `'x'` True).
- Both operands go through `np.asarray`. Both must be `S` or `U`, else `TypeError("comparison of non-string arrays")`
  (two numeric arrays too). An `S` operand with a `U` operand returns the `NotImplemented` singleton — and the six
  functions return it to the caller.
- The loop is P2.1's comparison with `rstrip=true` (`string_comparison_loop`): each element first drops trailing NULs
  and ASCII whitespace (`' '`, `\t`, `\n`, `\v`, `\f`, `\r`), for `U` too — `'\x1c'`, `'\xa0'`, `'\u3000'` stay.
  Embedded NULs still count (`'a\0b ' == 'a\0b'`). No Python `str` is built, so a `U` unit above `U+10FFFF`
  compares as a value.
- The iterator is `NpyIter` with `KEEPORDER` and an allocated bool output: operands broadcast, F inputs give an F
  result, a mismatch raises `ValueError("operands could not be broadcast together with shapes (2,) (3,) ")` (trailing
  space), and scalars give a 0-d bool array.

### 1.3 `multiply`, `partition`, `rpartition`

- `multiply(a, i)` is `np.strings.multiply`, with every `TypeError` turned into
  `ValueError("Can only multiply by integers")` — a float or bool `i`, and a non-text `a` too. The `OverflowError`
  `"Overflow encountered in string multiply"` passes through.
- `partition(a, sep)` / `rpartition(a, sep)` are `np.stack(np.strings.partition(a, sep), axis=-1)`: shape
  `a.shape + (3,)`, dtype the widest of the three parts, layout `np.concatenate`'s vote over the three results (an F
  input gives a result that is neither C nor F). An empty `a` raises `np.strings.partition`'s
  `ValueError("zero-size array to reduction operation maximum which has no identity")`.

### 1.4 The object path: `join`, `split`, `rsplit`, `splitlines`

These run `_vec_string(array, dtype, method, args)`: a broadcast multi-iterator over the array and every argument
(each converted with `np.asarray`), one call of the Python `str` / `bytes` method per element with NumPy scalars as
arguments, and a result array allocated in C order whatever the input layout. A NumPy scalar string has already
dropped its trailing NULs.

- `join(sep, seq)`: `sep_i.join(seq_i)`. `seq_i` is a NumPy scalar string, so its CHARACTERS are joined
  (`join('-', 'osd')` is `'o-s-d'`); a nested list `seq` is only a 2-D array of characters. The object result is
  then converted by `_to_bytes_or_str_array`: `np.asarray(result.tolist())` (the width of the longest result,
  minimum 1), an empty result `astype(seq.dtype)`. With `S` data it always fails unless the element is empty:
  iterating `bytes` yields ints, so `TypeError("sequence item 0: expected a bytes-like object, int found")`; a `U`
  separator with `S` data raises `"sequence item 0: expected str instance, int found"`.
- `split(a, sep=None, maxsplit=None)` / `rsplit`: an object array of Python lists, 0-d for a 0-d input. `sep=None`
  splits on runs of Python whitespace: for `U` the 29 code points `str.isspace` accepts (`\t`–`\r`, `\x1c`–`\x1f`,
  `' '`, `\x85`, `\xa0`, `\u1680`, `\u2000`–`\u200a`, `\u2028`, `\u2029`, `\u202f`, `\u205f`, `\u3000`), for `S`
  the 6 ASCII ones. `maxsplit` None or negative means no limit; `sep` and `maxsplit` broadcast per element. Errors:
  `''` → `ValueError("empty separator")`; a `U` array with a `bytes` separator →
  `TypeError("must be str or None, not numpy.bytes_")`; a float `maxsplit` →
  `TypeError("'numpy.float64' object cannot be interpreted as an integer")`. Quirk: an `S` array with a `str`
  separator splits on the separator's UCS-4 BYTES (a NumPy scalar exposes the buffer protocol), so
  `split([b'a b'], ' ')` is `[[b'a b']]`.
- `splitlines(a, keepends=None)`: `U` line boundaries are `\n`, `\r`, `\r\n`, `\v`, `\f`, `\x1c`, `\x1d`, `\x1e`,
  `\x85`, `\u2028`, `\u2029`; `S` boundaries are `\n`, `\r`, `\r\n`. `keepends` broadcasts per element.

Probed in phase 4 (each pinned by the oracle section `strcharobj` and `CharObjectPathTests`):
- **The method is picked by the first argument's dtype** (`bytes.*` for `S`, `str.*` for `U` and StringDType); anything
  else, a 0-d object array from `None` included, is `TypeError("string operation on non-string array")`. The other
  arguments are `np.asarray`'d and broadcast by the multi-iterator, whose error is `ValueError("shape mismatch: objects
  cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,) and arg 1 with shape (3,).")`.
- **CPython's order per element:** the NumPy scalars of the operands (a ONE-unit `U` text above U+10FFFF is CPython's
  `SystemError("invalid maximum character passed to PyUnicode_New")`: `_PyUnicode_FromUCS4`'s one-character path checks
  the code point, a longer text's maximum is capped at U+10FFFF), the descriptor check (`descriptor 'split' for 'str'
  objects doesn't apply to a 'NoneType' object` for a StringDType NA), `maxsplit`'s `__index__` (`np.bool` has none,
  `np.uint64(2**63)` is `OverflowError("Python int too large to convert to C ssize_t")`), `keepends`' `PyObject_IsTrue`
  (`0.5`, `'x'` and NaN are true), the separator's type, `"empty separator"`, then the pieces (a one-unit piece above
  U+10FFFF is the SystemError again). The first failing element in C order ends the call; an empty array raises none.
- **bytes.split takes any buffer:** a NumPy number's raw bytes are a separator (`np.char.split(s, 5)` splits on the 8
  bytes of `np.int64(5)`, `True` on `\x01`, a `str` on its UCS-4 bytes, a Python int past uint64 is `a bytes-like object
  is required, not 'int'`).
- **`[self]`:** `unicodeobject.c`'s split returns `[self]` when the separator is longer than the string or of a wider
  kind (`PyUnicode_KIND`), so a `U` element's list holds its own `np.str_` (printed `np.str_('ab')`); a StringDType
  element is an exact `str`, and bytes split has no such shortcut.
- **join's result is `_to_bytes_or_str_array(result, seq)`:** the class of `seq` (a StringDType `seq` gives the default
  `StringDType()`, its NA object dropped), as wide as the longest result (at least 1); an empty broadcast is
  `result.astype(seq.dtype)` for any dtype, an empty OBJECT array for an object `seq`. **NumPy 2.4.2 bug:** for a
  non-empty object `seq` (a split result: the idiom `np.char.join(' ', np.char.split(a))`) the cast calls
  `ObjectDType(n)`, which raises ``TypeError("currently only the no-argument instantiation is supported; use `np.dtype`
  instead.")`` after every element joined — the idiom does not work in NumPy 2.4.2.
- **Python's `None`:** `np.asarray(None)` is a 0-d object array, so `split(None)` is the class error, `join('-', None)` is
  `"can only join an iterable"`, and a list holding `None` (`[' ', None]`) is an object array whose `None` elements are
  Python's `None` (whitespace as a separator, `'NoneType' object cannot be interpreted as an integer` as a limit).

### 1.5 The re-exports on the same object path (`mod`, `encode`, `decode`, `translate`)

These are `np.strings` functions, so phase 2 (P2.6) implements them; `np.char` exposes them unchanged.

- `mod(a, values)`: `values` broadcasts as an ARRAY, so a Python tuple is not one tuple per element
  (`mod(['%s-%s'], (1, 2))` raises `TypeError("not enough arguments for format string")`). Each value reaches `%` as
  a NumPy scalar: `%s` of `np.float32(1.5)` is `'1.5'`, `%r` gives NumPy 2's repr (`'np.int64(7)'`,
  `'np.float32(1.5)'`, `"np.str_('c ')"`, `'np.True_'`, `'np.complex128(1+2j)'`). A 0-d object holding a `dict`
  feeds `'%(a)s'`. Width: the longest result.
- `encode` / `decode`: default `utf-8`; Python's codecs and error handlers; `utf-16` writes a BOM; an unknown codec
  raises `LookupError("unknown encoding: nope")`; `encode` of an `S` array raises
  `AttributeError("type object 'bytes' has no attribute 'encode'")` (and `decode` of `U` the `str` twin).
- `translate(a, table, deletechars=None)`: `U` calls `str.translate(table)` — `deletechars` is IGNORED. `S` calls
  `bytes.translate(table, deletechars)` with the table as a NumPy scalar, which drops trailing NULs: a 256-byte table
  that ends in `b'\x00'` raises `ValueError("translation table must be 256 characters long")`. The result keeps the
  INPUT dtype, so it truncates (`'abc'` → `'xyzA'` → `'xyz'` at `<U3`).

### 1.6 The `chararray` class

- **Constructor** `chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0, strides=None, order='C')`:
  `|S1` by default; `itemsize` goes through `int()`. Without a buffer it is `np.empty` (an `S` array holds whatever
  memory held, a `U` array is zeroed). A `str` buffer FILLS every element; a `bytes` buffer is a read-only view
  (OWNDATA False) at `offset` with `strides`. A negative dimension raises NumPy's `ValueError`.
- **`__array_finalize__`**: a result whose dtype is not `S`, `U` or `V` raises
  `ValueError("Can only create a chararray from string data.")`. So `arange(3).view(chararray)`, `c.astype(int)`,
  `c.astype(object)`, `c.argmax()`, `c.argmin()`, `c.argpartition(1)` and `np.argmax(c)` all raise; `c.view(np.uint32)`
  does NOT (it gives a `uint32` chararray). NumPy 2.4.2 crashes the PROCESS (access violation) on the fifth failed
  `c.argmax()` in one run.
- **`__array_wrap__`**: a ufunc result of dtype `S` or `U` becomes a chararray VIEW of the ufunc's output (OWNDATA
  False); other dtypes stay ndarray.
- **`__getitem__`**: a scalar result is the NumPy scalar's `.rstrip()` — for `U` Python's 29 whitespace code points
  (so `'\x1c'`, `'\xa0'`, `'\u3000'` strip here although the comparisons keep them), for `S` the 6 ASCII ones; trailing
  NULs are already gone. An array result is a chararray view, unstripped. `item()`, `tolist()` and `flat[i]` do not
  strip; iteration does (it calls `__getitem__`).
- **Comparisons** call the six `np.char` functions. For an `S`/`U` mix the function returns `NotImplemented`, so
  Python answers `==` with `False`, `!=` with `True`, and `<` with
  `TypeError("'<' not supported between instances of 'chararray' and 'chararray'")`. A non-text operand raises
  `TypeError("comparison of non-string arrays")`. A plain ndarray on the LEFT still strips (Python calls the
  subclass's reflected method first): `np.array(['ab']) == c` strips, and so does `'ab' in c`.
- **Arithmetic**: `+` is `np.char.add` (a ufunc → chararray view); `*` is `asarray(np.char.multiply(...))`
  (`ValueError` for a non-integer); `%` is `asarray(np.char.mod(...))`; `'%s' % c` is `str(c)` (`__rmod__` returns
  `NotImplemented`); `-c` and `c & c` raise the ufunc refusals; `c += 'z'` is `ndarray.__iadd__`: in place,
  truncated to the width.
- **The 39 string methods** follow five return rules (from the source):
  1. `asarray(...)` → chararray: `capitalize`, `center`, `expandtabs`, `ljust`, `lower`, `partition`, `rjust`,
     `rpartition`, `swapcase`, `title`, `translate`, `upper`, `zfill`. `asarray` compares the itemsize in CHARACTERS
     with the array's itemsize in BYTES, so a `U` result is always an owning copy and an `S` result a view of the
     computed array.
  2. The `np.strings` ufunc result → chararray view (OWNDATA False): `lstrip`, `rstrip`, `strip`, `replace`
     (`count=None` becomes `-1`).
  3. ndarray of text, not chararray: `decode`, `encode`, `join`.
  4. ndarray of int64 / bool: `count`, `endswith`, `find`, `index`, `rfind`, `rindex`, `startswith`, and the nine
     predicates `isalnum` … `isdecimal`.
  5. Object arrays: `split`, `rsplit`, `splitlines`.
  `argsort` is ndarray's `argsort` on the plain view (no stripping).
- **The 69 inherited ndarray members**: views and copies keep chararray (`reshape`, `ravel`, `flatten`, `T`, `mT`,
  `transpose`, `swapaxes`, `squeeze`, `copy`, `take`, `repeat`, `compress`, `diagonal` (read-only), `real`, `imag`
  (read-only empty strings), `view()`, `astype` to `S`/`U`, `flat[slice]`, `byteswap`); `all`, `any`, `nonzero` work;
  the numeric members raise ndarray's errors (`sum` / `mean` / `std` / `var` / `trace` / `cumsum`: "the resolved
  dtypes are not compatible with add.reduce…"; `max` / `min` / `prod` / `cumprod` / `clip`: UFuncTypeError; `round`,
  `conj`, `dot` (`"data type must provide an itemsize"`), `choose` (the int64 cast error)).
- **Printing**: `repr` is `"chararray(['a', 'bb'], dtype='<U2')"` with continuation lines indented 11 (the prefix
  length), a 0-d array `"chararray('ab ', dtype='<U3')"`; `str` is ndarray's.

### 1.7 `np.char.array` and `np.char.asarray`

`array(obj, itemsize=None, copy=True, unicode=None, order=None)`; `asarray(obj, itemsize=None, unicode=None, order=None)`
is `array(..., copy=False)`.

- A `str` / `bytes` object: `unicode` follows its type, `itemsize` defaults to its length, the shape is
  `len(obj) // itemsize`, then the constructor runs with `buffer=obj`: a `str` FILLS every element, so
  `np.char.array('abcdef', itemsize=2)` is `['ab', 'ab', 'ab']` (not the documented chunks); `bytes` are chunked, as a
  read-only view.
- A list or tuple goes through `np.asarray` first (a ragged list raises NumPy's inhomogeneous-shape `ValueError`;
  `['a', 1]` is `<U21`).
- An `S` / `U` ndarray: `view(chararray)`; `itemsize` defaults to the array's, divided by 4 for `U`; `order=` runs
  `np.asarray(obj, order=order)`, which DROPS the subclass (`np.char.array(x, order='F')` returns a plain ndarray);
  then `astype((dtype, itemsize))` when `copy` is set, when the itemsize differs (characters against bytes: ALWAYS for
  `U`, so `np.char.asarray` of a `U` array never shares memory, while an `S` array is shared), or when the class
  changes (`unicode=False` on `U` data: the `U` → `S` cast, `UnicodeEncodeError` for a non-ASCII character).
- Anything else: `np.array(obj, dtype=bytes_ or str_, order=order)` then `view(chararray)`: numbers become their text
  (`[1, 22]` → `[b'1', b'22']`, `[True, False]` → `|S5`), and OWNDATA is False.

### 1.8 NumPy functions over a `chararray`

- Keep the subclass: `np.sort`, `np.reshape`, `np.asanyarray`, `np.unique`, `np.flip`, `np.tile`, `np.repeat`,
  `np.atleast_2d`, `np.expand_dims`, `np.take`, and every ufunc with an `S`/`U` result (`np.add`, `np.strings.add`,
  `multiply`, `strip`, `center` …).
- Return a plain ndarray: `np.concatenate`, `np.stack`, `np.where`, `np.asarray`, `np.array`, `np.copy`,
  `np.broadcast_to`, and the object-path functions (`np.strings.upper` …).
- `np.equal(c, x)` is the plain ufunc (no stripping); `np.char.equal(c, x)` strips.

### 1.9 Result layouts

Ufunc results and the stripping comparisons follow the operands' order (an F input gives an F result). Object-path
results (`join`, `mod`, `encode`, `decode`, `upper` …, the split family) are C-ordered for any input. `partition`
follows `np.concatenate`'s vote.

---

## 2. Measurements

NumPy timed with `time.perf_counter` (best of rounds); C# prototypes are standalone `dotnet run -c Release` programs on
raw UCS-4 buffers, `DOTNET_TieredCompilation=0`, the same two pinned cores. They exclude NumSharp's fixed cost per call
(an NDArray result and an iterator: about 0.2–0.5 µs). Data: `<U8` strings of 1–5 letters with 0–3 trailing spaces;
three words of 2–5 letters joined by spaces (`<U17`).

| Operation | NumPy 1K / 100K / 1M (µs) | Prototype 1K / 100K / 1M (µs) | NPY/NS |
|---|---|---|---|
| `np.char.equal(<U8, <U8)` (stripping) | 29.7 / 5,845 / 59,693 | 4.0 / 700 / 7,545 | 7.4 / 8.3 / 7.9 |
| `np.char.less(<U8, <U8)` (stripping) | 29.1 / 5,831 / 59,626 | 6.1 / 874 / 9,040 | 4.8 / 6.7 / 6.6 |
| `np.char.split(words)` (whitespace) | 477 / 67,194 / 911,730 | 38.9 / 6,146 / 62,203 | 12 / 11 / 15 |
| `np.char.split(words, ' ')` | 646 / 87,069 / 1,022,513 | 37.1 / 6,044 / 61,536 | 17 / 14 / 17 |

NumPy's other `np.char`-only costs at 100K, per element: `rsplit(' ', 1)` 830 ns, `splitlines` 485 ns, `join` 1,107 ns,
`mod('%d')` 613 ns, `encode` 431 ns, `decode` 285 ns, `translate` (a dict) 1,464 ns, `np.char.partition` 188 ns
(`np.strings.partition` 134 ns: the stack adds 40 %), `multiply` 84 ns, `chararray.upper()` 472 ns, `chararray + 'x'`
47 ns, `[c[i] for i in range(n)]` 585 ns, `np.char.array(list)` 234 ns. For reference, the plain `np.equal` costs 22 ns
per element: NumPy's stripping loop is 2.6× slower than its plain loop.

Findings:
- The stripping comparison needs no new memory pass: one vector compare per trailing-unit class and a leading-zero
  count give each element's stripped length; after the strip the last unit is never NUL, so when the common prefix
  ties, the longer stripped element is greater. The prototype matched a scalar port of `Buffer::rstrip` + `strcmp`
  on every element of the three sizes.
- NumPy's object path costs 300–1,500 ns per element (a Python call and Python objects per element). A C# kernel over
  code units removes all of it; the split prototype stores each piece at the input width (an upper bound, one pass)
  and still writes 3 × 68 bytes per element.

---

## 3. Technique decisions

**C-D1. The facade is `np.@char` — owner decision.** `char` is a C# keyword, so NumPy's `np.char.upper(a)` is spelled
`np.@char.upper(a)` (as `np.@long`, `np.@decimal`). But `np.@char` is today the `DType` field of NumSharp's Char dtype
(`APIs/np.cs`), used 64 times (58 in `Logic/np.find_common_type.cs`, 6 in tests), all in DType contexts.
- Recommended: `np.@char` becomes the module object (`CharModule`, `[ModuleName("np.char")]`, the `np.fft` /
  `np.ma` shape), with an implicit conversion to the Char `DType`. Every current use compiles and keeps its meaning
  (method arguments, `==`, array initializers, tuple dictionary keys, `.Should().Be(np.@char)`). A standalone C#
  check of these contexts compiled and gave the old answers: a tuple key `(int16, np.@char)`, an array initializer
  typed `DType[]`, a `DType` argument, `==` in both orders, generic inference (`AreEqual<T>`) picking `DType`. What
  breaks: binary compatibility (the field's type changes), `ReferenceEquals(x, np.@char)`, boxing to `object`, and
  member access such as `np.@char.itemsize` (none in the repository).
- Alternatives: keep the `DType` and give the module another C# name (ported code must be rewritten; the coverage
  credit still works through `[ModuleName("np.char")]`), or rename the Char alias.

**C-D2. `chararray` is `NDCharArray : NDArray`.** The `NDArray<T>` precedent applies: `NDArray`'s
`protected internal (UnmanagedStorage, Shape)` constructors, and `new`-hidden members for static typing. A wrapper
like `NDMaskedArray` was rejected: a chararray IS an ndarray (every `np.*` function accepts it, `np.asanyarray`
keeps it), and a wrapper would need an explicit forward for each of the 108 members. NumPy's dynamic hooks become
four explicit points:
1. **`__array_finalize__` → one virtual hook on `NDArray`**, called at the return of the members and `np.*` functions
   that keep the subtype in NumPy (Appendix B and §1.8 list them). `NDArray`'s implementation is the identity (one
   virtual call); `NDCharArray`'s re-wraps an `S`/`U` result as an `NDCharArray` over the same storage (O(1)) and
   raises the `ValueError` for any other dtype — which is how `argmax`, `argmin`, `argpartition` and `astype(int)`
   reproduce NumPy.
2. **`__array_wrap__` → the text ufunc drivers** (`NDStringComparison` and the phase-2 drivers) wrap an `S`/`U` result
   as an `NDCharArray` view when an input is one.
3. **Python's operator dispatch → `NDArray`'s operators** test `is NDCharArray` on either operand, because in Python
   the subclass's reflected method wins: `np.array(['ab']) == c` strips as in NumPy.
4. **Static typing → `new`-hidden members** on `NDCharArray`, so `c.reshape(4)` is typed `NDCharArray`; the hook
   gives the same runtime type through an `NDArray`-typed variable.

**As built (phases 2 and 3, owner decision "static + boundary", 2026-10-16).** Points 1 and 3 were not built: no member
of `NDArray` became virtual, and `NDArray`'s operators do not test for a chararray. The class follows `NDMatrix`'s rule
instead. Its members and operators are re-declared on `NDCharArray` and bind at COMPILE time, so a chararray behind an
`NDArray` reference uses `NDArray`'s plain meaning. The `np.*` functions find a chararray at RUN time through
`NDCharArrayBoundary` (point 2 and §1.8), which shares `NDMatrixBoundary`'s depth. See the phase-2 and phase-3 entries of
the state log.

Element access: an index that selects one element returns a 0-d COPY that holds the stripped value (Python's `str`),
width = the stripped length (minimum 1, as `np.array('')` is `<U1`). A view would make `c[0] + "x"` give `'ab  x'`
where NumPy gives `'abx'`. Iteration yields the same copies; `item()`, `tolist()` and the flat iterator keep
`NDArray`'s unstripped answers. Printing takes the repr prefix from the runtime type (`"chararray("`).
`NDArray.view` gains NumPy's `type` parameter (`a.view(type: typeof(NDCharArray))`).

**C-D3. Object results are an `NDListArray` — owner decision.** NumSharp has no object dtype.
- Recommended: a new `NDListArray`: one flat 1-D pieces array (the input's class and width) plus int64 `start` and
  `end` arrays of the outer shape. `this[...]` gives a 1-D view of one element's pieces; a slice gives an
  `NDListArray` view (only `start` / `end` are sliced); `tolist()` gives NumPy's nested lists; `np.@char.join(sep,
  NDListArray)` joins each element's pieces, which serves NumPy's idiom `np.char.join(' ', np.char.split(a))`.
  Measured 11–17× NumPy (§2).
- Alternatives: (b) NumPy's `result.tolist()` as nested `List<object>`: no new type, one .NET string per piece
  (estimated 4–6× NumPy), no shape object; (c) refuse with `NotSupportedException` (the current answer for object
  dtype): 3 functions and 3 members stay unimplemented.

**As built (phase 4, owner decision "NDListArray", 2026-10-16).** Three points differ from the recommendation:
1. **The pieces are stored exact, not at one width.** `NDListArray` keeps every piece's code units back to back
   (`Units`: uint8 for bytes pieces, uint32 for str pieces) with an int64 `Offsets` array, plus the element ranges
   (`Starts` / `Ends`, the outer shape) and the `[self]` flags. A one-width pieces array would cost pieces × longest
   piece (one long piece among a million short ones makes it gigabytes) and would drop a piece's trailing NUL.
2. **`this[i, j]` is a COPY**, a new 1-D array of the element's pieces (`S` / `U` as wide as the longest piece, or
   StringDType), not a view: exact pieces have no common width to view. `item(...)` and `tolist()` return the exact
   pieces (NumPy's Python list); slice notation gives an `NDListArray` view (only the ranges are sliced).
3. **`np.@char.join(sep, NDListArray)` raises NumPy 2.4.2's TypeError** (§1.4): NumPy itself cannot finish the idiom,
   so NumSharp reproduces the failure after every element joined; an empty broadcast returns an empty `NDListArray`.

**C-D4. The stripping comparisons are a flavor of P2.1's comparison kernels.** Per element: the stripped length (a
vector compare against NUL and the ASCII whitespace classes, then a leading-zero count), then the first differing unit
within the shorter stripped length, else the lengths. The driver is P2.1's (`NDStringComparison`: NumPy's output
layout, the trivial loop, NDIter with the broadcast-pattern stages). NumPy's argument checks run first, in NumPy's
order (§1.2). Measured 4.8–8.3× NumPy.

**C-D5. `np.char`'s own object-path functions run on code units.** `join` and the split family are two passes (sizes,
then fill), with the D2 primitives of the phase-2 plan. The whitespace and line-boundary sets are fixed lists (29
whitespace code points for `U`, 6 for `S`; 10 line-boundary code points plus `\r\n` for `U`, `\n`, `\r`, `\r\n` for
`S`), so these functions need no generated Unicode table. Results are
C-ordered (NumPy's `_vec_string`); text results take NumPy's width rule (the longest, minimum 1; an empty result takes
`seq`'s dtype).

**C-D6. Arguments and Python sentinels.** `np.char` functions take array_likes as `object` (the convention of the
`numpy.polynomial` package) and convert them with a TEXT rule: a C# `string` is NumPy's `str`, a `byte[]` NumPy's
`bytes`, `string[]` / `byte[][]` / `IList` the matching arrays. This is phase 2's D10 for text parameters, and it is
non-breaking because the module is new. Python's `NotImplemented` maps to `null` (owner decision): the six comparison
functions return `null` for an `S`/`U` mix. `NDCharArray`'s operators answer as Python's fallback does: `==` a 0-d
`False`, `!=` a 0-d `True`, ordering a `TypeError` with Python's text (the two type names).

**C-D7. NumPy's quirks are reproduced and pinned.** NumPy is the source of truth, so each probed quirk is
implemented, documented in the member's remarks and pinned by a unit test: `asarray` always copies a `U` array;
`array('abcdef', itemsize=2)` fills; `order=` returns a plain array; `argmax` / `argmin` / `argpartition` /
`astype(non-string)` raise the finalize `ValueError` and `view(uint32)` does not; an `S` split with a `str` separator
splits on UCS-4 bytes; a translate table that ends in NUL fails; `S` join raises. NumPy's crash is not reproduced:
NumSharp raises the `ValueError` every time.

**C-D8. NDExpr fusion is not used**, for phase 2's reason (D8 there): the text kernels are most of each call.

---

## 4. Subphases

Each subphase ships its kernels, its entry points with NumPy's parameters, unit tests from NumPy output, a string-oracle
section with zero excuses over the relayout kinds (C, F, permuted, stepped, reversed, offset views, broadcasts, 0-d,
one-element, empty, empty views; both classes; widths past the kernel cap; embedded and trailing NULs, bytes ≥ 0x80, `U`
units above `U+10FFFF`), benchmark rows at 1K / 100K / 1M across layouts, and its documentation.

### C0 The facade and its conventions

- `np.@char` (C-D1) with all 52 functions (NumPy's names, parameter order, defaults) and the `chararray(...)` factory
  (the `np.random` class-factory shape). The 36 shared names call the internal entry points that `np.strings` uses
  (phase 2 creates them); a function whose kernel has not landed throws the existing deferral.
- The text argument converter and the `null`-for-`NotImplemented` rule (C-D6); a `LookupError` house exception
  (`encode` / `decode`).
- Gate: a signature-parity test against NumPy's signatures (as `LinAlgSignatureParityTests`); the coverage generator
  then credits the facade members.
- Depends on: owner decision C-D1. Size: small.

### C1 The stripping comparisons

- `compare_chararrays`, `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` (C-D4).
- Gate: unit tests per whitespace class, embedded NULs, `S`/`U` mixes, layouts and broadcasting; the oracle section.
- Target: ≥ 4.8× NumPy (the measured floor). Depends on: P2.0 / P2.1 (done). Size: small.

### C2 The `chararray` class

- `NDCharArray` (C-D2): the factory and constructors, `np.@char.array` / `asarray` with every rule of §1.7, the
  finalize hook on `NDArray` and its call sites, the wrap rule in the text ufunc drivers, the operators (including the
  runtime test in `NDArray`'s operators), element access and iteration (stripped copies), printing, `NDArray.view`'s
  `type` parameter.
- Gate: an oracle object section (a construction recipe, then one operation; the case records the result's Python
  type, dtype, shape, values and C / F / OWNDATA / WRITEABLE) and unit tests per quirk.
- Target: element access and iteration ≥ 1.5× NumPy's `c[i]` (585 ns per element in a Python loop; each NumSharp
  access allocates a 0-d copy, so measure it).
- Depends on: C1; P2.4 for non-text inputs to `array` / `asarray` and for number stores (`c[0] = 5`); P2.3 for `+`
  and `*`; P2.6 for `%`. The parts with no phase-2 dependency can start now. Size: large.

### C3 `np.char`'s object-path functions

- `join` (also over an `NDListArray`), `split`, `rsplit`, `splitlines` (C-D3, C-D5), with every error of §1.4 in
  NumPy's order.
- Target: ≥ 10× NumPy (split measured 11–17×). Depends on: C0; owner decision C-D3. Size: medium.

### C4 `np.char`'s wrappers of phase-2 functions

- `multiply` (the `ValueError` mapping) and `partition` / `rpartition` (the stacked result). Compose `np.stack` as
  NumPy does (layout for free); fuse into one stacked-layout kernel only if the stack pass keeps a cell under 1.5×.
- Depends on: P2.3 (`multiply`), P2.5 (`partition`). Size: small.
- Done (2026-10-17, `ffa74606`): `partition` / `rpartition` compose `np.stack` as NumPy does. No cell is under 1.5× (1.89–4.14×
  NumPy), so there is no fused stacked-layout kernel.

### C5 The `chararray` members

- The 39 string methods with NumPy's five return rules (§1.6) and the 69 inherited members (Appendix B); `argsort` on
  the plain view.
- Each member lands with its function (C1–C4, or the phase-2 subphase). Size: medium (many small members).
- In batches of one family each, with the module's re-exports of the same names: batch 1 (`rpartition`, `center`,
  `ljust`, `rjust`, `zfill`, `expandtabs`) is done (2026-10-17, `ffa74606`), batch 2 (`count`, `find`, `rfind`, `index`,
  `rindex`, `startswith`, `endswith`, and the module's `str_len`) too (2026-10-17, `6d06fa3c`), batch 3 (`strip`,
  `lstrip`, `rstrip`, `replace`; 2026-10-17, `66b68e96`), and batch 4 (the nine predicates; 2026-10-18, `b1328a4f`).

### C6 Completion

- The whole surface strict in the string oracle; `coverage/oracle_map.json` (`char.*` op keys, `chararray.*` member
  keys) and `coverage/object_surfaces.py` (`char.chararray` → `NumSharp.NDCharArray`); the docs
  (`docs/website-src/docs/fundamentals/strings-and-bytes.md`, a `numpy.char` section; `.claude/CLAUDE.md`'s API list);
  the benchmark rows (`benchmark/strings`); the release notes.
- Size: small.
- The object-surface mapping `char.chararray` → `NumSharp.NDCharArray` is done (2026-10-17, `b4a3b78f`, with the
  coverage generator's join of the API diff gap map): 104 of the 193 `numpy.char.chararray` rows read available (90
  exact, 14 alias) and their signatures are compared. The 24 named members still missing are C5's open batches
  (`capitalize`, `lower`, `upper`, `swapcase`, `title`, the `is*` predicates, `strip` / `lstrip` / `rstrip`, `replace`,
  `translate`, `encode`, `decode`) and three ndarray members NumSharp has nowhere (`ctypes`, `dump`, `dumps`); the rest
  are Python protocols. Batches 3 and 4 have since added `strip` / `lstrip` / `rstrip` / `replace` and the nine
  predicates (`66b68e96`, `b1328a4f`), which leaves 11 of the 24 by that count. The `chararray.*` member keys of
  `oracle_map.json` are still open: the credited rows read oracle-unverified.

### Order

`C0 → C1 → C2 → C3`, then C4 and C5 as their phase-2 dependencies land, then C6. With phase 2 at `7c817644`
(P2.0 and the core of P2.1), C0, C1, C3 and most of C2 have no open phase-2 dependency; C0 waits only for the owner's
C-D1 decision and C3 for C-D3.

| Phase-2 subphase | Unlocks in `numpy.char` |
|---|---|
| P2.3 (concatenation, repetition) | `add`; C4's `multiply`; `chararray`'s `+` and `*` |
| P2.4 (casts) | `np.@char.array` / `asarray` over numbers; number stores into a `chararray` |
| P2.5 (length, search, strip, pad, replace, partition) | 17 re-exports; C4's `partition` / `rpartition`; their members |
| P2.6 (Unicode properties, case, `mod`, codecs, `translate`) | 18 re-exports; `chararray`'s `%`; their members |

---

## 5. Gates

- **Oracle.** `test/oracle/gen_strings_oracle.py` gains `gen_strchar()` (the module functions over the relayout kinds,
  both classes, the argument kinds of C-D6, every error in NumPy's order; result kinds: an array, `NotImplemented`,
  object lists compared through `tolist()`) and `gen_chararray()` (the object surface of C2). The replay adds
  `StringOracleOps.Char.cs` and `StringOracleOps.Chararray.cs`. A case that makes `chararray`'s finalize hook raise runs
  in its own Python process (NumPy's crash, §1.6).
- **Coverage.** The facade credits the 53 module rows; the object-surface target credits the 108 members;
  `oracle_map.json` joins each row to its contracts.
- **Benchmarks.** Every `np.char`-only function at 1K / 100K / 1M over C, F, strided and broadcast layouts; the bar is
  ≥ 1.5× NumPy on every cell, with the targets of §4.

---

## 6. Out of scope and divergences

- **Object arrays as inputs** (`np.char.array` of an object array, `mod` with one tuple per element or a `dict`):
  NumSharp has no object dtype. Proposed extension (owner decision): `np.@char.mod(a, params object[] values)` with
  two or more operands gives each element the tuple of the broadcast operands' elements (NumPy needs an object array of
  tuples); a C# `IDictionary<string, object>` as a 0-d mapping value.
- **`c += x`**: C# compound assignment rebinds `c` to `c + x`; NumPy's `__iadd__` writes in place and truncates to the
  width. `[Misaligned]`.
- **`ctypes`, `dump`, `dumps`**: `chararray` inherits `NDArray`'s answer (NumSharp has neither ctypes nor pickle).
- **Python's `NotImplemented` and `bool` results** from an operator typed `NDArray`: the C-D6 mapping. `[Misaligned]`.
- **NumPy's crash** on repeated failed `argmax`: not reproduced (§1.6).
- **Phase 3's divergences** (pinned `[Misaligned]` in `CharArrayBehaviorTests`; the oracle names each excused case):
  - **The static type decides** (owner decision): a chararray behind an `NDArray` reference binds `NDArray`'s operators
    and members, which compute plain (`==` does not strip, `+` returns a plain array, `argmax()` returns the index, an
    element keeps its whitespace). NumPy dispatches on the object. The `np.*` functions find a chararray at run time.
  - **`c == null`** keeps `NDArray`'s reference test (a 0-d `False`; `!=` a 0-d `True`), as C# code expects of
    `x == null`. NumPy's `c == None` raises `TypeError: comparison of non-string arrays`.
  - **`np.squeeze`** of a chararray with no length-1 axis returns a new chararray view; NumPy returns the operand itself.
  - **A view of an element stand-in is read-only** (`np.atleast_2d(c[0])`); NumPy converts the Python `str` into a fresh
    writeable array.
  - **`chararray + StringDType`** where NumPy's trivial loop does not copy the chararray: NumPy returns a PLAIN
    StringDType array, which the operator's `NDCharArray` type cannot hold, so the operator raises
    `NotSupportedException` naming `np.add` (which computes NumPy's answer).
  - **numpy.ma** does not model a chararray `_baseclass`: a masked array over a chararray holds plain data, as it does
    for a matrix.
  - **Inherited plain-array gaps** (the array-object oracle's ids): VF12 (`np.array` copies a permuted or broadcast input
    in C order, NumPy in K order), VF26 (`np.tile` with unit reps of a 0-d input owns its copy), VF01 (`np.choose` of a
    0-d index returns a writeable 0-d array, NumPy a scalar), VF17 (`np.broadcast_arrays` makes a read-only operand's
    result writeable).
- **Phase 4's divergences** (pinned in `CharObjectPathTests`):
  - **The object result is an `NDListArray`**, not an ndarray: it has the array attributes, basic indexing, `item`,
    `tolist`, iteration and NumPy's repr / str, but none of the other ndarray members (`reshape`, `T`, `flatten`, ...).
  - **Element access is a 1-D piece array** (C-D3 "As built", point 2; `[Misaligned]`): NumPy returns the Python list,
    which `item()` gives.
  - **Python's `None` is `null` here** (the object path models NumPy's object arrays), where the rest of the module
    raises `ArgumentNullException` for a null array argument (C-D6).
  - **The broadcast error** is `IncorrectShapeException` (the house convention for shape errors) with NumPy's text.

---

## 7. Risks and open questions

1. **Owner decisions:** C-D1 (`np.@char` becomes the module), C-D3 (`NDListArray`), C-D6 (`null` for
   `NotImplemented`), the `mod` extension of §6.
2. **The finalize hook touches every subtype-keeping member.** A member that misses the call loses `chararray`
   semantics without an error. The oracle's type facet catches it: every member of Appendix B and every function of
   §1.8 gets a case.
3. **Coordination with the phase-2 session.** This plan takes the `np.char` items out of `strings-phase2.md` (P2.1's
   stripping comparisons, P2.6's `join`, stacked `partition` and `multiply`); the phase-2 plan keeps the shared kernels
   and the 36 re-exported functions. Both write into `gen_strings_oracle.py`: add sections, never reorder them.
4. **The fixed whitespace lists** (C-D5) assume Unicode 15.0.0's `White_Space` (CPython 3.12). A pin test compares
   them with `unicodedata` output committed in the corpus.

### State log

- **2026-10-15 — phase 1: the module functions `numpy.strings` computes** (`4c313632`; the owner's scope: "the `np.@char` facade,
  `compare_chararrays` and the six comparisons that strip trailing whitespace, over S/U (T refused), `np.char.add` and
  `np.char.multiply` (NumPy's ValueError rule), oracle section `strchar`"). Owner decisions taken: C-D1 (the module),
  C-D6's `null` for `NotImplemented`.
  - **Facade** (`APIs/np.char.cs`): `np.@char` is `CharModule` (`[ModuleName("np.char")]`, the `np.fft` shape), with
    an implicit conversion to the `Char` `DType`; every former use compiles unchanged (58 in `np.find_common_type.cs`,
    the tests). Array arguments are `object` read by C-D6's text rule (`TextOperand`: an `NDArray` itself, a `byte[]` a
    0-d `S`, a sequence of `byte[]` an `S` array, else `NDArray.FromPythonOperand`); the members are `[NDScoped]`, so a
    converted argument is reclaimed and the operands are rooted. `compare_chararrays` takes `cmp` as `string` (UTF-8,
    NumPy's `s#`: a lone surrogate is CPython's UnicodeEncodeError) or `byte[]`; `null` is NumPy's `None` text.
  - **The rstrip kernel** (`ILKernelGenerator.Text.Strip.cs`): one flag of P2.1's comparison key, so the 1-D kernel,
    its broadcast stages, the rows and masked forms all exist for it. For at most 64 code units a bit-mask body: per
    chunk, one load per operand feeds the equal bits and both strip-unit masks (`v == 0 | v == ' ' | min(v - 9, 4) ==
    v - 9`); kept length `64 - lzcnt(~S & full)`; `d = tzcnt(~E)`; the units at `d` decide when `d` is inside both kept
    strings, else the kept lengths (the last kept unit is nonzero). A scalar chunk (an element under 16 bytes) is
    tested as the low lanes of a zeroed `Vector128`, its padding lanes masked off. Wider elements call
    `TextKeptBytes` (a 32-byte backward scan) and `TextCompareKept` (`CommonPrefixLength`). A broadcast operand's kept
    length and chunks are hoisted.
  - **Driver** (`NDStringComparison.CompareChararrays`): NumPy's order — fixed-width text check (`PyArray_ISSTRING`:
    a `StringDType` array is `comparison of non-string arrays`), the class check (`NotImplemented`, before the
    broadcast), the broadcast, then P2.1's routes with `rstrip`.
  - **`np.char.multiply`**: `np.strings.multiply` with `TypeError` → `ValueError("Can only multiply by integers")`.
    `np.strings.multiply` gained NumPy's StringDType branch (`a * i` after the integer check: deferred) and dropped the
    zero fill of its result (`NDArray.ForOverwrite`: the repetition writes every byte; 24.8 ms of eager zeroing at 1M
    `<U24`). P2.3's narrow repetition kernel gained a DIRECT form (zero the element, store each copy into the output)
    when the last copy fits, replacing the stack row whose read-back follows overlapping stores (8.3 → 5.8 ns per
    `<U8 × 3` element) and the per-element helper call of a one-unit partial fill.
  - **Gates.** Oracle section `strchar` (2,790 cases, 215 NumPy errors, 3 deferred StringDType computations, zero
    excuses): the six comparisons over every layout kind and the cross pairs at 11 width pairs on both sides of the
    64-unit limit, every strip unit and the whitespace NumPy keeps, units past U+10FFFF, first differences in a unit's
    higher byte, the arguments as Python values with NumPy's refusals in order, `compare_chararrays` with 30 `cmp`
    spellings × both `rstrip` values, `add`, `multiply`. Unit tests `Backends/Strings/CharModuleTests.cs` (12: every
    kernel form against a literal port of `Buffer::rstrip` + `strcmp`, the facade's literals). 14 planted bugs, each
    killed by the unit tests and by the oracle (M5, a wide `U` diff read unaligned, needed the high-byte cases). Leak
    catalogue entries for the ten members.
  - **Speed** (NPY/NS, two pinned P-cores, 1K / 100K / 1M, 57 cells): comparisons 2.6–10.5× (U8 4.8–6.2×, scalar
    comparand 2.6–3.2×, U3 2.7–3.0×, the 64-unit limit 3.8–4.8×, wide U70 2.7–3.8×, wide S300 6.2–8.3×, strided /
    F / column-broadcast 3.6–9.5×); `np.char.add` 4.9–6.9×; `np.char.multiply` 2.42× / 4.95× / 1.52× (the 1M cell
    is a fresh 96 MB result: both sides pay the page faults; before the kernel and allocation changes 1.30× / 1.85× /
    0.94×).
  - **Open:** the other 43 functions (C3–C5), `chararray` (C2, the user's phase 2), C-D6's `None` (an
    `ArgumentNullException` here; NumPy's object array).
- **2026-10-16 — phase 2: `chararray` as data** (`012ac869`; C2's data half; the owner's scope: "`NDCharArray : NDArray`. The
  constructor, `np.char.array` and `np.char.asarray` with NumPy's quirks (numbers and T become text). The finalize rule. The
  subclass is kept on views and copies. Element access and iteration strip whitespace. `item()` and `tolist()` do not. The
  `chararray(...)` repr and `view(type=)`. Oracle section `strchararray`").
  - **Class** (`Char/NDCharArray{,.Strip,.Indexing,.Methods}.cs`): `NDCharArray : NDArray`, constructed over a storage.
    **Deviation from C-D2:** no virtual finalize hook on `NDArray`. The class follows `NDMatrix`'s static rule: every member
    NumPy answers with the subclass is re-declared with `new` (`astype`, `copy`, `view`, `byteswap`, `getfield`,
    `to_device`, `T`, `mT`, `transpose`, `swapaxes`, `reshape` ×6, `ravel`, `flatten`, `squeeze`, `diagonal`, `take`,
    `compress`, `repeat`, `real`, `imag`, `Clone`) and wraps the plain result (`From` / `TakeOver`: O(1), the storage
    object taken over or aliased), so an `NDArray`-typed call site gets the plain `NDArray` member — NumPy's dynamic
    dispatch through a static type is phase 3's question, with the operators and the `np.*` functions (§1.8). The finalize
    rule (`dtype.char in 'VSUbc'`, else `ValueError("Can only create a chararray from string data.")`) runs BEFORE each
    allocation, after NumPy's own earlier checks (astype: the casting rule; take: the index); `view(dtype)` reads the
    dtype the array HAS, so `c.view(np.uint32)` is a uint32 chararray whose own views raise. NumPy answers `self` from
    `real`, `squeeze()` without a length-1 axis, `astype` of the same dtype under `copy=False` (when the order is
    satisfied) and `to_device`, and so does NumSharp.
  - **Element access** (`NDCharArray.Indexing.cs`): one indexer per spelling (`Slice[]`, slice notation, `object[]`,
    `NDArray<bool>`, `NDArray<int>[]`, `__getitem__`). A full integer index (one integer per axis, 0-d integer arrays
    included: mapping.c's HAS_INTEGER) is the element STRIPPED as Python's `rstrip()` strips it (`NDCharArray.Strip.cs`:
    29 code points for `U` — .NET's `char.IsWhiteSpace` plus U+001C–U+001F —, 6 bytes for `S`, trailing NULs first, a
    NUL stops the strip), returned as a NEW read-only 0-d stand-in (`ScalarOf`: width = the kept length, at least 1; a
    `U` unit above U+10FFFF is CPython's SystemError). Every other index returns a chararray: a view (basic), a view of the
    gathered array (an index array, a mask, a scalar bool), an owning copy (HAS_SCALAR_ARRAY). `item()`, `tolist()`,
    `take(i)` (a 0-d result is the UNSTRIPPED stand-in) and the flat iterator keep NumPy's unstripped values.
  - **Iteration**: a 1-d chararray yields its stripped elements as .NET `string` / `byte[]` (exact, a kept NUL included),
    an N-d one chararray row views; a 0-d one raises `TypeError: iteration over a 0-d array`.
  - **Constructors** (`APIs/np.char.array.cs`): `chararray(shape, itemsize, unicode, buffer, offset, strides, order)` — a
    `string` buffer FILLS (`self[...] = filler`), `ReadOnlyMemory<byte>` is Python's read-only `bytes`, `byte[]` a
    writable `bytearray`, an `NDArray` is borrowed, everything else `np.ndarray`'s buffer rules; `array` / `asarray` —
    str / bytes (`itemsize = len`, shape `len // itemsize`, Python's ZeroDivisionError as `DivideByZeroException`), lists
    and tuples (`np.asarray` first), S / U arrays (a view, `itemsize // 4` for `U`, a cast when the width, the class or
    `copy` asks; an `S` array under `unicode=True, copy=False` stays `S`), any other array through NumPy's `tolist()` route
    (value-based widths: float32 `0.10000000149011612`; with an itemsize a cast at that width, 0 the unsized string),
    Python scalars (`np.array(obj, dtype)`).
  - **Printing**: `ReprClassName` gives `chararray(` and its continuation indent; `ArrayFormatter` wraps the element
    formatter in `StrippedTextReprFormat` for an N-d chararray (NumPy's `_formatArray` indexes through `__getitem__`); a
    0-d chararray prints unstripped. `NDArray.view(Type)` / `view(DType, Type)` are NumPy's `view(type=)` for
    `NDArray` / `NDCharArray` / `NDMatrix` (`Type must be a sub-type of ndarray type` otherwise).
  - **Divergences** (pinned `[Misaligned]`, `CharArrayTests`): `np.char.array(..., order=)` of string data returns a
    chararray (NumPy's `np.asarray(obj, order=order)` drops the class: a plain ndarray; the oracle's one class excuse,
    2,981 cases); `chararray(itemsize=0)` refused (NumPy's S0 / U0); a stripped value that ends in a NUL loses it in the
    stand-in (a fixed-width element drops trailing NULs; the enumerator's value keeps it); `squeeze()` of a chararray over
    numbers with a length-1 axis raises where NumPy hands back a plain array; `np.*` functions return plain arrays
    (phase 3). Open with them (phase 3): `c.argmax()` / `argmin()` / `argpartition()` and `np.argmax(c)` return the
    index (the text arg-reductions compute since `a9ec801a`), where NumPy's finalize raises `ValueError("Can only create
    a chararray from string data.")` unless `out=` is given (probed: `c.argmax(out=plain)` computes, `c.argsort()` is a
    plain array; NumPy 2.4.2 can crash on the second failed finalize in one process).
  - **Base fixes the section's replay found** (each NumPy's answer for every array, with unit tests and the oracle cells
    that return to the gates): a scalar `bool` index is a COPY (`a[True]` was an `expand_dims` view, `a[False]` the
    dims-less sentinel, the string indexer counted the bool as an axis); an out-of-bounds integer in slice notation on a
    strided view is NumPy's `IndexError` text (`UnmanagedStorage.GetView` checks every integer against its source axis
    before the newaxis expansion), and index 0 of an empty axis raises; `getfield` of an empty array is a view (VF26's
    part); `np.imag` of a real F array is F-ordered (`PyArray_ISFORTRAN`; VF31, MX03); `np.asanyarray(object[,])` is
    NumPy's nested list of rows. The replay of the carved gaps also found MS15 matching (fixed by `cb3a4186`): 15 + 9 + 12
    carved cases return to the matrix / view-flags / ma-surface tiers.
  - **Gates.** Oracle section `strchararray` (11,505 cases, 1,448 NumPy errors, 0 deferred, 182 cases with the layout of
    an advanced index of a non-C source excused (OpenBugs.FancyIndex's class), 2,981 with the order quirk's class): the
    constructor's buffers, offsets, strides and errors; `array` / `asarray` over every branch × itemsize / copy / unicode /
    order; element access over every whitespace code point and byte, every layout kind and index kind in three C#
    spellings; iteration; 46 member calls over 14 layouts, the allocating finalize errors recorded in isolated NumPy processes
    (NumPy 2.4.2 corrupts its heap there); `view(type)`; repr / str. Unit tests `Backends/Strings/CharArrayTests.cs` (17)
    plus the base fixes' tests. Leak gates: `NDCharArray` is an object owner of the leak surface (`LeakCatalogue.Char.cs`,
    property-read targets).
  - **Performance** (NPY/NS, `start /affinity 14`, the benchmark's 55 cells: element access, iteration, the constructors,
    `np.char.array` / `asarray` over every branch, the members that keep the class, the gathers, `view(type)` and repr at
    1K / 100K / 1M): geomean 3.61× with tiered compilation off, 4.32× warmed with it on; every cell at least 1.63× (was
    geomean 2.53×, 16 cells under 1.5×, the slowest 0.15×). What it took, cell by cell:
    - element access 1.65–2.80× (was 0.44–0.77×): the element's address comes straight from the strides (no view of the
      indexed array), the descriptors of the stand-in widths are cached, and the stand-in's text is cut from a per-thread
      2 KB arena block (`Char/NDCharArray.StandIn.cs`): the pieces share the block's Disposer, so a stand-in pays no pool
      round trip and no finalizable Disposer; it still owns an exclusive copy, and the block goes back to the pool with
      its last stand-in. A lean element slice and storage (`StringArraySlice`'s trusted constructor,
      `UnmanagedStorage.ReadOnlyStringScalar`) and an `int` / `long` indexer finish it;
    - `c[mask]` 4.0–5.4×, `c[idx]` 2.7–3.2×, `take` 1.9–3.1× (were 0.76–1.03× at 1K): a fixed-width string array gathers
      directly — the boolean mask through a new IL kernel (`DirectILKernelGenerator.MaskGather.cs`: a vector compare and
      movemask per 32 mask bytes, the set bits walked with TrailingZeroCount), an index array through the fancy-index
      kernel route — instead of through its unit view; the flat gather / scatter kernels now bake ANY slab width up to 256
      bytes (`FixedCopyKindFor`: a constant-size `cpblk` the JIT unrolls; a `<U8` element was a memory-move call per
      index), which numeric row gathers use too, and `np.take`'s result skips the `U` zero fill it overwrites;
    - `c[1:]` 1.8–2.7× (was 0.32–0.42× in slice notation): `Slice.ParseSlices` keeps a bounded parse cache (~1.3 µs → 35 ns,
      fresh slices per call), and `GetView` / `Shape.Slice` drop their per-call allocations (the index array of the
      all-integer case, a LINQ closure, two growing lists): any basic slice view of any dtype went from ~570 to ~320 ns;
    - `c.T` 2.2–3.3× (was 1.29× at 1K) and `c.copy()` 1.6–5.0× (was 1.34×): built as the subclass from the start (the
      engine's `TransposedShape` over a storage alias; a fresh buffer filled by the dense block copy), and a copy of a
      source already in fresh C layout reuses its shape (`NDIter.CopyAsShape`: `AsClean` instead of a rebuild). The 1K
      copy is memcpy-bound on both sides: 32 KB moves in ~850 ns, L2 bandwidth;
    - `np.char.array(float64)` 2.2–4.9× (was 1.15–1.20×): the float64 → text cast at the fixed width, the longest text and
      a narrowing copy (`FloatsAsText`) — one repr per element where the `tolist()` route formatted twice. NumPy's
      float64 cast text equals Python's `repr` (probed on 200,000 random doubles);
    - `chararray(n)` 1.8–3.8× (was 0.17× at 100K): a zero-filled string buffer takes `np.zeros`' allocation (calloc, or the
      OS's demand-zero pages from 128 KB), and such a VirtualAlloc'd block reports its memory through the pool's coarse
      accounting (`AccountForeignAllocation`) instead of a `GC.AddMemoryPressure` per block, which ran a gen-2 collection
      about every 150 allocations of a short-lived 3.2 MB array — also for numeric `np.zeros`.
    Unit tests `Backends/Strings/CharArrayFastPathTests.cs` (15) hold each fast path to the general route it replaces; 21
    planted bugs (arena, lean objects, offsets, the copy widths, the mask kernel's lanes and tail, the parse cache, the
    copy shape, the transpose, the float text, the view trims, the zero fill) are killed by them.
- **2026-10-16 — phase 3: `chararray` behavior** (`d343a5bc`; C2's behavior half and the finalize and sorting rows of C5; the owner's
  scope: "Comparison operators that strip trailing whitespace, also when a plain array is on the left. + and * that
  return a chararray. The sort and search members (sort in place; argsort and searchsorted return plain arrays). NumPy's
  finalize ValueError for argmax, argmin and argpartition. The np.* functions that keep the class (np.sort, np.reshape,
  np.unique, np.take, text ufuncs) and those that drop it (np.concatenate, np.stack, np.where). A new oracle section.").
  Owner decision taken: **the static rule plus the boundary** (`NDMatrix`'s rule; C-D2's "As built" note). No new kernel.
  - **Operators** (`Char/NDCharArray.Operators.cs`): `==`, `!=`, `<`, `<=`, `>`, `>=`, `+` and `*`, each in seven forms
    (a chararray against a chararray, an `NDArray`, an `object` or a `string`, in both orders), so an expression with a
    TYPED chararray on either side binds here and never meets an ambiguity with `NDArray`'s forms. Python's dispatch is
    replayed. The chararray's method runs first. When a plain array is on the left, the chararray's REFLECTED method runs
    first, because a subclass's reflected method wins (so `u == c` strips). For an `S` / `U` mix (`NotImplemented`) the
    other operand answers: a plain array compares unstripped (`c == s` is all False, `c < s` the ufunc's no-loop
    TypeError); with nothing to answer, `==` is a 0-d False, `!=` a 0-d True, and an ordering raises
    `TypeError("'<' not supported between instances of 'chararray' and 'bytes'")`. An element stand-in (`c[0]`, NumPy's
    Python `str`) is not a chararray operand. `+` is `np.char.add` in the written order: a chararray VIEW (OWNDATA False).
    `*` is `np.char.asarray(np.char.multiply(...))`: a `U` result is a copy, an `S` result a view, and a count that is not
    an integer is `ValueError("Can only multiply by integers")`. The C# `null` keeps `NDArray`'s reference test for `==` and
    `!=`. For `+`, `null` is NumPy's `None`: np.add's trivial loop copies a 0-d or short 1-d chararray into object as a
    chararray (the finalize error), and a longer one reaches CPython's `can only concatenate str (not "NoneType") to str`.
  - **Members** (`Char/NDCharArray.Arg.cs`): `argmax`, `argmin` and `argpartition` create their int64 result as the
    subclass in NumPy (`PyArray_NewFromDescr(Py_TYPE(ap), ...)`). They run the plain computation for NumPy's earlier errors
    (the axis, `attempt to get argmax of an empty sequence`, the kind, the kth bounds) and then raise `ValueError("Can only
    create a chararray from string data.")`; with `out=` they compute into it. `partition(sep)` is the STRING partition
    (deferred to P2.5: 22 oracle cases); its sort forms with `axis=` raise NumPy's `TypeError("chararray.partition() got
    an unexpected keyword argument 'axis'")`. `sort` (in place, unstripped order), `argsort` and `searchsorted` (plain
    arrays) are the inherited members, already NumPy's answers.
  - **The boundary** (`Char/NDCharArrayBoundary.cs`): a public function that sees a chararray at the outermost call packs
    its arguments in a tuple and calls itself again through a rule (a `static` lambda: no allocation). The rule raises the
    depth it shares with `NDMatrixBoundary`, so nested public calls compute plain for both subclasses, and
    `NDMatrixBoundary.PlainFor` opens a region for a chararray receiver too. So a function decides its chararray rule
    BEFORE any `PlainFor` (np.quantile's tuple-axis overloads missed chararrays until the flag moved above it). The rules
    follow NumPy's mechanism per function:
    - **Kept** (NumPy calls the method or works on `np.asanyarray(a)`; the plain result becomes a chararray over the same
      storage, owning exactly where it owns, and the operand itself where NumPy returns it): `np.reshape`, `np.ravel`,
      `np.transpose`, `np.permute_dims`, `np.matrix_transpose`, `np.swapaxes`, `np.moveaxis`, `np.rollaxis`, `np.squeeze`,
      `np.expand_dims`, `np.atleast_1d` / `2d` / `3d`, `np.fliplr`, `np.flipud`, `np.rot90`, `np.roll`, `np.tile`,
      `np.repeat`, `np.sort`, `np.compress`, `np.extract`, `np.diagonal`, `np.diag` of a 2-D chararray, `np.trim_zeros`,
      `np.require`, `np.asanyarray`, `np.copy(subok=True)` (the new `subok` parameter), `np.empty_like` / `zeros_like` /
      `ones_like` / `full_like` (a numeric `dtype=` is the finalize error before anything is filled), `np.resize` of an
      empty source or result (`zeros_like`; any other resize is plain). KeptAll: `np.split`, `np.array_split`,
      `np.hsplit`, `np.vsplit`, `np.unstack`. KeptTake: `np.take` (a 0-d result is the UNSTRIPPED NumPy scalar).
      KeptIndexed: `np.flip` (`m[indexer]`: a 0-d result is the element, stripped). KeptPerOperand: `np.meshgrid` (each
      result keeps ITS input's class).
    - **Wrapped** (`chararray.__array_wrap__`: a text result becomes a chararray VIEW, OWNDATA False; a chararray `out=`
      comes back as a new view of it): `np.add`, `np.multiply`, `np.strings.add` / `multiply`, `np.char.add` / `multiply`,
      `np.insert`, `np.delete`, `np.diagflat`, `np.kron`, `np.take_along_axis`, `np.setdiff1d`, `np.unique_values`, and
      np.unique's values (WrappedFirst; the index arrays stay plain). With a return flag NumPy's `_unique1d` finds the runs
      with the chararray's STRIPPING `!=`, so `['b ', 'a', 'b']` gives `['a', 'b']` with counts `[1, 2]`; without a flag
      it compares the plain data.
    - **Plain** (NumPy converts with `np.asarray`): `np.asarray`, `np.ascontiguousarray`, `np.asfortranarray`,
      `np.asarray_chkfinite`, the stack family (`np.stack`, `np.vstack`, `np.hstack`, `np.dstack`, `np.column_stack`),
      `np.append`, `np.argwhere`, `np.lexsort`, `np.sort_complex`, `np.fft.fftshift` / `ifftshift` / `hfft`,
      `np.linalg.norm` / `vector_norm` / `matrix_norm` / `cross` / `tensorsolve`, `np.cross`, `np.einsum`,
      `np.tensordot`, `np.vdot`, the `np.emath` functions, `np.unwrap`, `np.polyval` with a chararray `p`. No rule is needed
      where the computation already returns a fresh plain array (`np.concatenate`, `np.where`, the set routines but
      `setdiff1d`, `np.nonzero`, the comparison ufuncs' bool results).
    - **IndexResult / IndexScalar**: `np.argmax`, `np.argmin`, `np.nanargmax`, `np.nanargmin`, `np.argpartition` — the
      plain computation's checks, then the finalize error; with `out=` the call computes.
    - **The string partition**: `np.partition`, `np.median`, `np.quantile`, `np.percentile` raise
      `chararray.partition() got an unexpected keyword argument 'axis'`, after the checks NumPy makes first (the axes,
      keepdims' `out`, an unknown method's ValueError). `np.median(overwrite_input=True)` without an axis reproduces NumPy's
      real string partition around the kth list: its broadcast error for an even length other than 2, else the closing
      mean's refusal of text.
    - **NumPy's trivial-loop copy** (`check_for_trivial_loop`, `TrivialLoopCopyError`): without a mask, a ufunc input whose
      loop dtype differs from its own is COPIED into the loop dtype when it is 0-d or 1-d with at most `np.getbufsize()`
      elements, and the copy keeps the subclass. So `np.logical_and(c, c)` (a cast to bool) and `np.add(c, t)` with a
      `StringDType` `t` raise the finalize error for a short chararray and compute plain for a longer one. Only the Python
      object `True` is no mask (`_wheremask_converter`): the C# literal `true` is marked `WasPythonBool`, and `false` or
      `np.array(true)` select the masked loop.
    - **Other NumPy behaviors reproduced**: `np.polyval` with a chararray `x` raises `ValueError("Can only multiply by
      integers")` (an empty `p`: the chararray zeros); `np.flip` of a 0-d chararray is its stripped element;
      `np.unstack` of a 1-d chararray iterates it (stripped elements, `StrippedElements`); `np.choose` reports a broadcast
      failure in NumPy's multi-iterator text (the choices first, the index last).
  - **Library fixes found** (each NumPy's answer for every array, with unit tests):
    - `Shape.ResolveReturnShape((1,), ())` returned `()`: a one-element 1-d operand against a 0-d one now keeps its axis.
    - `np.unstack` of an empty array returns views, as for any other array (it returned owning copies).
    - `np.choose`'s broadcast error names the shapes in NumPy's order.
    - The logical ufuncs over text allocate their result in NpyIter's layout over the ORIGINAL operands: a broadcast
      operand abstains on its stride-0 axis (its dense truth array voted for F order).
    - The failed `S == U` comparison (an all-False result) is laid out as a ufunc output over the two operands (an F pair
      gives an F result).
    - The chararray comparison operators released the driver's untyped result wrapper only in the finalizer (the leak
      sweep found one stranded buffer per comparison); the typed result now shares the storage (`AsGeneric`), so it owns
      its data as NumPy's does.
    - numpy.ma: a masked array over a chararray holds a plain view of its data (`NDCharArrayBoundary.PlainViewOf`), as it
      does for a matrix; the masked computations no longer meet chararray rules.
    - Test harness: the ownership gate quiesces collections before each entry's ledger snapshot (a retained view of an
      earlier entry, finalized during the next one, read as that entry's over-disposal); the native-allocation chokepoint
      allowlist covers P2.T's static tables and in-place temporaries (`NDVStringArithmetic.cs` 4, `NDVStringCells.cs` 3).
  - **Divergences** (pinned `[Misaligned]`, `CharArrayBehaviorTests`; §6): the static type decides; `c == null` is a
    reference test; `np.squeeze` without a length-1 axis returns a new view; a view of an element stand-in is read-only;
    `chararray + StringDType` (NumPy's plain StringDType result) is refused by the operator; numpy.ma's chararray base
    class is not modelled.
  - **Gates.** Oracle section `strcharops` (3,266 cases, 1,392 NumPy errors, 22 deferred: `c.partition(sep)`, 8 excused
    refusals: `chararray + StringDType`): the comparisons with every partner kind in both orders (words with and without
    whitespace, plain arrays, chararrays, the other class, numbers, bools, `StringDType`, lists, empties, element
    stand-ins), `+` and `*` with their errors, a chararray past the buffer size, the members over 11 layouts × both
    classes. Oracle section `strcharfn` (2,633 cases over 99 functions, 709 NumPy errors): the class, values, OWNDATA,
    identity and errors of every rule over 12 layouts; 21 cases excused by layout (the plain gaps VF12 `np.array` of a
    permuted / broadcast input, VF26 `np.tile` of a 0-d input, VF01 `np.choose`'s 0-d result, VF17 `np.broadcast_arrays`
    of a read-only operand, and a view of an element stand-in) and 15 by identity (`np.squeeze`). The boundary audit
    (`AuditInternalCallers`: a stack walk at every rule) replays both sections and, in
    `CharArrayOperand_LibraryNeverReachesACharArrayRule`, all 74,606 string-oracle cases with their text operands made
    chararrays (floor 30,000): no library method reaches a chararray rule. Unit tests `Backends/Strings/
    CharArrayBehaviorTests.cs` (16, five `[Misaligned]`). Leak catalogue entries for the 14 new members (the comparison
    operators through every route, `+`, `*`, the index members, `partition`).
  - **Performance** (NPY/NS, `start /affinity 14`, two runs of each side, the better of each; 91 cells at 1K / 100K / 1M):
    geomean 3.54×, min 1.03×. Comparison operators 2.8–6.1× (geomean 4.4×), `+` and `*` 2.4–14× (5.3×), members 1.0–4.3×
    (2.0×), `np.*` that keep or wrap the class 1.5–9.0× (3.3×), `np.*` that drop it 1.3–4.8× (3.0×). The cells under 1.5×
    cost the same on the plain path, so the chararray layer adds nothing measurable there:
    - `c.searchsorted(keys)` 1.27× / 1.03× / 1.12× (64 keys): the plain `np.searchsorted` of P2.2 milestone B is
      1.07–1.17×. Both libraries run a binary search per key at 6–8 ns per step, a dependent chain of loads and compares.
      The lever is the search kernel's: interleave several keys' searches to overlap their memory latencies.
    - `c.argmax(out=o)` 1.27× at 1K and 1.39× at 1M (1.62× at 100K): the plain argmax kernel. At 1K its fixed cost is
      ~0.5 µs of 1.45 µs (NumPy's method skips `_wrapfunc`, 1.9 µs where `np.argmax` costs 3.0 µs); at 1M the 32 MB scan
      runs at ~75 % of one core's DRAM bandwidth.
    - `np.concatenate((c, c))` 1.34× at 1K: the plain concatenate of 2 × 32 KB (allocation and copy fixed cost).
    Two Kept-rule costs were measured: a rule adds ~255 ns (one storage alias and one `NDCharArray`), and the plain
    `reshape(-1, 8)` spends ~210 ns more than an explicit shape (`_inferMissingDimension` builds an intermediate `Shape`);
    `np.reshape(c, (-1, 8))` is 2.4–3.1×.
  - **Open:** `c.partition(sep)` (P2.5), C3 (phase 4, C-D3's `NDListArray`), the string methods of C5 (P2.5 / P2.6), the
    coverage mapping `char.chararray` → `NumSharp.NDCharArray` (C6).
- **2026-10-16 — phase 4: the object path** (the owner's scope: "join, split, rsplit and splitlines, with NumPy's errors in
  NumPy's order"; owner decision C-D3: `NDListArray`).
  - **Built.** `np.@char.split` / `rsplit` / `splitlines` / `join` (`APIs/np.char.split.cs`), the chararray members
    `split` / `rsplit` / `splitlines` / `join` (`Char/NDCharArray.Split.cs`), the result type `NDListArray`
    (`Char/NDListArray.cs`) and the engine `NDCharObjectPath` (`Backends/Strings/NDCharObjectPath.cs`): NumPy's
    `_vec_string` as one NDIter walk in C order over the broadcast operands (each with its own dtype, no casting, no
    buffer), CPython 3.12's `split_whitespace` / `split_char` / `split` / their `rsplit` twins / `splitlines` / `join`
    statement for statement over 1-byte or 4-byte units, every argument read as the Python object NumPy's
    `PyArray_ToScalar` makes (text, NumPy numbers with their buffer bytes, StringDType NA objects, object lists holding
    `None`, Python ints past uint64). `ArrayFormatter.ObjectArrayStr` / `ObjectArrayRepr` print object arrays through the
    existing layout engine.
  - **A .NET runtime trap found on the way:** under `DOTNET_TieredCompilation=0` the ReadyToRun bodies of CoreLib are
    never re-optimized, and inside the split loop its `memmove` and span searches ran 3–4× slower than the same loop with
    `DOTNET_ReadyToRun=0` (U `split(' ')` 324 against 61 µs per 1,000 elements; `str.split(None)` 198 against 52 µs). The
    pieces are now copied inline (two overlapping moves up to 32 bytes, 16-byte blocks up to 512) and the separators found
    by the engine's own vector searches (`FindUnit` / `FindUnitLast`, and a candidate scan for short texts in
    `FindNeedle` / `FindNeedleLast`); both modes are fast now.
  - **Gates.** Oracle section `strcharobj` (3,625 cases, 294 NumPy errors, zero excuses): S / U / StringDType over the 11
    core layouts × 8 separators × 5 limits for split / rsplit, 11 keepends forms, join over every class pair and layout,
    the argument kinds of 13 dtypes and of Python values (big ints, complex, `None`, object lists), StringDType NA
    objects, broadcasting and its error, NumPy's error order, the one-unit U+10FFFF rule, the whitespace and
    line-boundary sets exhaustively (every code point below U+3010 and every byte), long texts, the chararray members,
    the result's views and printing, join over the object result. An object result is compared piece by piece with its
    `[self]` pieces (a new expectation kind `lists`). 8 planted bugs (rsplit order, `[self]`, a vector search, the
    `\x85` whitespace, CRLF, maxsplit 0, the lone-unit check, the piece copy) each turned 2–1,707 cases red. Unit tests
    `Backends/Strings/CharObjectPathTests.cs` (16, one `[Misaligned]`). Leak catalogue entries for the 8 new members.
  - **Performance** (NPY/NS, `start /affinity 14`, the default tiered runtime; 99 cells: split / rsplit / splitlines /
    join × S / U / StringDType × 1K / 100K / 1M, plus strided, a separator array and a two-unit separator): geomean
    12.6×, min 4.6× (`rsplit(None, 1)` of StringDType at 1K). split / rsplit: U 10–25×, StringDType 4.6–48×, S 5.5–11×
    (NumPy's bytes methods cost half its str methods); splitlines 6.7–36×; join 10–18×. Under `TieredCompilation=0`:
    geomean 11.1×, min 5.5×.
  - **Open:** the coverage mapping `char.chararray` → `NumSharp.NDCharArray` (C6); `NDListArray`'s other ndarray
    members, if wanted.
- **2026-10-17 — C4 and C5's first batch: `partition` / `rpartition` and the padding functions** (`ffa74606`; the owner's scope:
  "C4, the wrappers … partition and rpartition … And do around 6 methods at a time, preferably similar/family/near/group
  of methods … C5, the members").
  - **Built.** `np.@char.partition` / `rpartition` (`APIs/np.char.transform.cs`): NumPy's `np.stack(np.strings.partition(a,
    sep), axis=-1)`, one PLAIN array with a last axis of 3 (np.stack keeps no subclass; an F-ordered input gives NumPy's
    stack strides, (24, 48, 8) for `<U2`), the separator cast to the text's dtype first (`c.partition(1)` splits on `'1'`,
    `np.char.partition(5, ' ')` is NumPy's `invalid literal for int() with base 10: np.str_(' ')`), a `StringDType` text
    needing a `StringDType` separator. The re-exports `center` / `ljust` / `rjust` / `zfill` / `expandtabs` ARE
    `np.strings`' functions: each converts its array_likes with `TextOperand` (a `byte[]` is Python bytes) and a null
    optional argument is NumPy's default (`OptionalOperand`). The chararray members `rpartition` / `center` / `ljust` /
    `rjust` / `zfill` / `expandtabs` (`Char/NDCharArray.Transform.cs`, `NDCharArray.Arg.cs`) are NumPy's
    `asarray(f(self, …))`: they read the STORED text (`np.char.array(['ab  ']).center(6, '*')` is `'*ab  *'`), and
    `AsArrayOfFresh` gives the computed array NumPy's final form directly — a `U` result is taken over instead of copied
    (the computed array is private to the call, so its owning chararray is the same object to every observer), an `S`
    result is viewed. `chararray.partition` takes the same route (it copied a `U` result through `np.char.asarray`).
  - **Size pass.** `np.strings`' padding (`NDStringMethods.PadDescriptor`, `S` / `U` only) skips its length scan when the
    widest width reaches the element width: no string is longer than its element, so the width is the maximum.
  - **Gates.** Oracle section `strcharmeth` (2,811 cases, 938 NumPy errors, zero excuses, with the boundary audit): the
    members over both classes × 14 layouts, every separator / width / fill / tab-size form, zfill's signs and
    expandtabs' columns, the module functions over plain and chararray operands of every layout, a chararray in a
    non-text position, `StringDType`, Python values and broadcasting; each case records the result's class, OWNDATA and
    strides. zfill receivers hold no empty element (NumPy 2.4.2's zfill of an empty element reads uninitialized memory for
    its sign; the `{"nonempty": …}` recipe). Unit tests `Backends/Strings/CharStringMethodsTests.cs` (12). 6 planted bugs
    (a `U` result viewed, axis 0 stacking, rpartition running partition, the size-pass skip one unit early, the boundary
    region removed, expandtabs' default 4) each turned the section red (7–621 cases, or the audit).
  - **Performance** (NPY/NS, two pinned P-cores, the default tiered runtime; 117 cells: the 7 functions over `U` / `S` /
    `StringDType` and the 7 members over `U` / `S` at 1K / 100K / 1M, plus reversed and F-ordered views): geomean 3.70×,
    1.75× minimum in the full run; members 2.53–12.2× (a `U` member result taken over: 1M `U` `center` 91.9 → 39.8 ms),
    `partition` / `rpartition` 1.89–4.14×. The lowest cell is `expandtabs` over `U` at 1M: NumPy sizes a `U` result in
    bytes (`<U80` for 20-character texts, 320 MB), and the page faults of that fresh result cost both libraries the same
    46 ms; measured alone it is 1.30× (NumPy 139 ms, NumSharp 105 ms). The lever is a vector path for `U` elements wider
    than 64 bytes in P2.5's kernel.
  - **Open:** the other C5 string members, in batches (§0).
- **2026-10-17 — C5's second batch: the search family** (`6d06fa3c`; the same scope, one family per batch).
  - **Built.** `np.@char.str_len` / `count` / `find` / `rfind` / `index` / `rindex` / `startswith` / `endswith`
    (`APIs/np.char.search.cs`: `np.strings`' functions, `str_len` with the ufunc's `out=` / `where=` / `dtype=`) and the
    chararray members of the seven search names (`Char/NDCharArray.Search.cs`). NumPy's members are the function's int64 /
    bool result as it is (`chararray.__array_wrap__` keeps the class only for text), so they forward like phase 4's
    `split`: a PLAIN result, a 0-d array for a 0-d chararray (the wrap ignores `return_scalar`), over the stored text
    (`np.char.array(['ab  ']).count(' ')` is 2).
  - **Found and fixed in `np.strings`.** (1) A one-input text ufunc (`str_len`, the nine predicates, the whitespace
    strips) named a C# literal by NEP 50's weak class in its no-loop refusal; NumPy's `convert_ufunc_arguments` stops at
    the first input when `nin == 1`, before it marks a Python scalar weak, so `np.strings.str_len(5)` names `Int64DType`
    (5.0 `Float64DType`, 1j `Complex128DType`, 2**63 `UInt64DType`; `NDStringUfuncs.UnaryDispatchClass`). (2)
    `startswith` over `U` ran 1.01–1.58× NumPy: the lane path built the needle's match mask at every offset of the element
    to read one bit, after a length scan. `NDTextElements.TailMatchWholeRange` serves a range from a non-negative start to
    the element's end (NumPy's default `end=None`): `startswith` compares once at `start` and reads no length (a non-empty
    needle's last unit is not NUL, so a match proves the element long enough), `endswith` reads the length and compares
    once. (3) The 1-D text kernels prefetch their text operand 4 KB ahead for elements of 32 bytes or more (`str_len` of
    1M `<U24`: 6.6–7.4 → 5.2 ms; narrower elements lost 3–7 % to the prefetch and emit none).
  - **Gates.** Oracle section `strcharmeth` +5,388 cases (8,199 in all, 2,796 NumPy errors, zero excuses, with the
    boundary audit): the members over both classes × 14 layouts with every needle and range form, long texts past the
    lane cap, the functions over plain and chararray operands, `str_len`'s ufunc keywords, `StringDType`, Python values
    (an int and a float among them: the unary refusal), broadcasting. Unit tests: `CharStringMethodsTests` (16) and
    `StringMethodsLanesTests.TailMatchWholeRange_EqualsTheSpanBody_EveryWidth`. 6 planted bugs each turned the oracle red
    (6–523 cases).
  - **Performance** (141 cells: the 8 functions over `U` / `S` / `StringDType` and the 7 members over `U` / `S` at 1K /
    100K / 1M, plus a range, a long needle, reversed and F-ordered views): geomean 3.18×, min 1.66×; `startswith` over `U`
    2.26–2.89×, `str_len` over `U` at 1M 1.93×. The C# twin needs a warm-up pass over every case: its first cell otherwise
    runs before the background tier-1 compiler finishes the shared call path (`str_len` over `U` at 1K read 4.3 µs,
    2.55 µs warm).
  - **Open:** the strips and `replace` (batch 3), the predicates, the case maps.
- **2026-10-17 — C5's third batch: the strips and `replace`** (`66b68e96`; the same scope, one family per batch).
  - **Built.** `np.@char.strip` / `lstrip` / `rstrip` / `replace` (`APIs/np.char.transform.cs`: `np.strings`' functions)
    and the chararray members of the four names (`Char/NDCharArray.Transform.cs`), NumPy's rule 2: the member returns the
    function's result itself, the chararray view `chararray.__array_wrap__` makes (OWNDATA false), over the STORED text
    (`np.char.array([' xa ']).rstrip('x')` keeps `' xa '`: its spaces end it). `replace`'s `old` / `new` without a dtype —
    a C# string, a `byte[]` (Python bytes, `NDArray.WasPythonBytes`), a number, a chararray ELEMENT stand-in (`c[i]` is
    `__getitem__`'s Python str / bytes, so `getattr(old, 'dtype', None)` is None) — is cast to the text's dtype.
  - **Found and fixed: NumPy's trivial-loop copy over a chararray, on the whole P2.5 surface.** A `StringDType` argument
    makes the function's loop StringDType, and NumPy's `check_for_trivial_loop` copies each input the loop casts — 0-d,
    or 1-d with at most `np.getbufsize()` elements — with `PyArray_CastToType`, which keeps the class: a chararray copied
    into `StringDType()` is refused by `__array_finalize__` (`Can only create a chararray from string data.`). The inputs
    are checked in order and the first larger one stops every later copy, a non-text one too (a 2-d int32 `width` before
    a chararray fill). Phase 3 had the rule for the operators and the comparison / logical ufuncs; batch 3 gives it a
    third input mark (`NDCharArrayBoundary.WrappedUfunc`, `PlainUfunc` for a chararray only in `old` / `new` /
    `fillchar` — NumPy's result stays plain there — and `InsideUfunc` for the pad members) and runs it in every P2.5
    function with a second text input: the search family, the strips with a set, `replace`, the pads with a fill
    (after the input casts, before the broadcast). Past the copy NumPy computes a PLAIN result: a StringDType array for
    the strips and `replace`, which the rule-2 members' `NDCharArray` type cannot hold (`NotSupportedException` naming
    `np.char.<name>`, the `+` operator's precedent, an excused refusal); a pad member converts it with
    `np.char.asarray`, an `S` chararray of its ASCII text, as NumPy's does.
  - **Found and fixed in `np.strings`.** The `chars` strips over `U` elements wider than 64 bytes searched the set once per
    tested unit (1K `<U27` with a three-unit set: 1.20× NumPy). `NDTextElements.StripCharsWide` tests 32-byte chunks from
    each end against every set unit at once (stripped runs are short: one chunk from each end usually decides), then one
    constant-width zero fill and one copy: 2.32×. Whole-element masks measured 1.5× SLOWER for the whitespace strip (they
    test every unit against 29 code points where the unit loops stop at the first kept one), which keeps its loops.
  - **Gates.** Oracle section `strcharmeth` +2,306 cases (10,505 in all, 3,323 NumPy errors, with the boundary audit): the
    members over both classes × 14 layouts with every set / old / new / count form, the functions over plain and chararray
    operands, StringDType, Python values, broadcasting, wide `<U44` elements whose stripped runs cross the chunks, and the
    copy rule (234 cases: 10 layouts, the chararray in every text position, an earlier input that refuses the trivial
    loop, an `S` chararray). Excused: 20 rule-2 member results NumPy returns plain, and 18 strips of a 0-d text NumPy
    returns as a NumPy scalar (VF01). Unit tests: `CharStringMethodsTests` (22), `StringMethodsLanesTests`
    (`StripCharsWide` against the scalar body over every wide width and over fenced pages). 10 planted bugs: 9 killed by
    both gates (two only after the gaps they found were closed: a bytes element stand-in in a unit test, wide elements in
    the oracle); the tenth is equivalent.
  - **Performance** (117 cells: the 4 functions over `U` / `S` / `StringDType` and the members over `U` / `S` at 1K / 100K
    / 1M, whitespace and a set, `replace` growing, shrinking and with a count, reversed and F-ordered views): 106 cells
    geomean 2.70×, min 1.56×. The 11 1M `U` strip cells are page-fault bound: the fresh 103 MiB `<U27` result costs both
    libraries the same faults (a copy of the input: NumPy 27.9 ms, NumSharp 28.6 ms), and NumPy's whitespace strip is
    1.43–1.51× its own copy, so even a free kernel stays near 1.5×; measured alone and alternating (min of 3) the
    whitespace strips read 1.07–1.32×, the set strips 1.40–1.48×. The levers left are non-temporal stores through an L1
    staging block and a pool that keeps blocks past its 64 MiB cap.
  - **Open:** the predicates (batch 4), the case maps (batch 5).
- **2026-10-18 — C5's fourth batch: the nine predicates** (`b1328a4f`; the same scope, one family per batch).
  - **Built.** `np.@char.isalnum` / `isalpha` / `isdecimal` / `isdigit` / `islower` / `isnumeric` / `isspace` /
    `istitle` / `isupper` (`APIs/np.char.predicates.cs`) and the chararray members of the nine names
    (`Char/NDCharArray.Predicates.cs`). NumPy's module functions ARE the ufunc objects
    (`np.char.isalpha is np.strings.isalpha`), so the re-exports keep the ufunc's `out=` / `where=` / `dtype=` (a
    positional `out=`) and every answer is `np.strings`': a number's no-loop TypeError names its default DType
    (`Int64DType`, `Float64DType`), a non-bool `dtype=` has no loop, and `isdecimal` / `isnumeric` have no bytes loop.
    The members are NumPy's rule 4, `isalpha(self)`: a PLAIN bool result that owns its data (a 0-d array for a 0-d
    chararray: the wrap ignores `return_scalar`) over the STORED text — `np.char.array(['ab  '])` is not alpha but is
    lower, though its element reads `'ab'` — and a chararray `out=` comes back as a new chararray view of it holding
    `'True'` / `'Fals'` (`<U4`).
  - **No `np.strings` change.** P2.6 milestone 1 (`8bbc0f79`) computes the predicates, and np.strings' predicates apply
    the chararray boundary themselves, so the re-exports forward and the replay passed at once (no new excuse).
  - **Gates.** Oracle section `strcharmeth` +1,035 cases (11,540 in all, 3,467 NumPy errors, with the boundary audit;
    the earlier 10,505 cases byte-identical): the members over both classes × 14 layouts and over 24 `U` / 16 `S`
    predicate texts (cases, titles, ASCII and Unicode digits — decimal U+0661, digit-only U+00B2, numeric-only U+00BD /
    U+2155 —, whitespace with U+3000 / U+1680 / U+2028, an embedded NUL, the titlecase U+01C5, uncased letters, trailing
    whitespace) as a row and as a column; the functions over plain and chararray operands; the ufunc keywords (`out=`
    bool / int32, keyword and positional, a chararray out, `where=`, `dtype=` bool and int64, an out of another shape);
    `StringDType`; Python values (a str, bytes, lists, an int, a float). Unit tests: `CharStringMethodsTests` (26: +4).
    The leak catalogue: 18 entries (686 in all). 9 planted bugs, all killed: the oracle killed 7 (2–54 cases each), the
    leak catalogue 8, the unit tests 5; a lost `[NDScoped]` is caught by the leak catalogue alone, and a `byte[]`
    converted as numbers by the unit tests and the leak catalogue (the oracle hands a Python bytes value over as the
    `NDCharArray` stand-in).
  - **Performance** (198 cells: the 9 functions over `U` / `StringDType` and the 7 with a bytes loop over `S`, the
    members over `U` / `S`, the functions over a `U` chararray, at 1K / 100K / 1M, long letter texts that `isalpha` /
    `islower` / `isupper` read to the end, reversed and F-ordered views, `out=`): geomean 4.71×, min 2.52× (`isspace`
    over `StringDType` at 1M), none under 1.5×.
  - **Open:** the case maps and `translate` (batch 5); `encode`, `decode` and `%` wait for the rest of P2.6.

---

## Appendix A — the 53 module rows

| Subphase | Rows |
|---|---|
| C1 | `compare_chararrays`, `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` |
| C2 | `array`, `asarray`, `chararray` |
| C3 | `join`, `split`, `rsplit`, `splitlines` (phase 4, done) |
| C4 | `multiply` (phase 1, done), `partition`, `rpartition` (done, `ffa74606`) |
| P2.3 | `add` |
| P2.5 | `str_len`, `count`, `find`, `rfind`, `index`, `rindex`, `startswith`, `endswith` (done, `6d06fa3c`), `lstrip`, `rstrip`, `strip`, `replace` (done, `66b68e96`), `expandtabs`, `center`, `ljust`, `rjust`, `zfill` (the last five: done, `ffa74606`) |
| P2.6 | `capitalize`, `swapcase`, `title`, `upper`, `lower`, `isalnum`, `isalpha`, `isdigit`, `islower`, `isspace`, `istitle`, `isupper`, `isnumeric`, `isdecimal` (the nine predicates: done, `b1328a4f`), `mod`, `encode`, `decode`, `translate` |

The facade (C0) declares all 53; a row is complete when its computation lands.

## Appendix B — the 108 `chararray` members

| Group | Members | NumPy's result | Lands with |
|---|---|---|---|
| String methods, rule 1 (`asarray`) | `capitalize`, `center`, `expandtabs`, `ljust`, `lower`, `partition`, `rjust`, `rpartition`, `swapcase`, `title`, `translate`, `upper`, `zfill` | chararray; `U` owning copy, `S` view | P2.5 / P2.6 / C4 (`center`, `expandtabs`, `ljust`, `partition`, `rjust`, `rpartition`, `zfill`: done, `ffa74606`) |
| String methods, rule 2 (ufunc wrap) | `lstrip`, `rstrip`, `strip`, `replace` | chararray view | P2.5 (done, `66b68e96`) |
| String methods, rule 3 | `decode`, `encode`, `join` | ndarray of text | P2.6 / C3 (`join`: phase 4, done) |
| String methods, rule 4 | `count`, `endswith`, `find`, `index`, `rfind`, `rindex`, `startswith`, `isalnum`, `isalpha`, `isdecimal`, `isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper` | ndarray of int64 / bool | P2.5 / P2.6 (`count`, `endswith`, `find`, `index`, `rfind`, `rindex`, `startswith`: done, `6d06fa3c`; the nine predicates: done, `b1328a4f`) |
| String methods, rule 5 | `split`, `rsplit`, `splitlines` | object array of lists (`NDListArray`) | C3 (phase 4, done) |
| Views and copies (the finalize hook) | `T`, `mT`, `transpose`, `swapaxes`, `reshape`, `ravel`, `flatten`, `squeeze`, `copy`, `take`, `repeat`, `compress`, `diagonal`, `real`, `imag`, `view`, `astype`, `byteswap`, `flat` | chararray (`diagonal`, `imag` read-only); `astype` / `view` to a numeric dtype per §1.6 | C2 |
| Finalize errors | `argmax`, `argmin`, `argpartition` | `ValueError("Can only create a chararray from string data.")`; with `out=`, the index | phase 3 (done) |
| Sorting | `argsort` (plain view), `sort` (in place), `searchsorted` | ndarray's answer | phase 3 (done; the kernels are P2.2's) |
| Truth | `all`, `any`, `nonzero` | ndarray's answer | P2.1 (done) |
| Numeric refusals | `max`, `min`, `sum`, `mean`, `prod`, `std`, `var`, `cumsum`, `cumprod`, `trace`, `clip`, `round`, `conj`, `conjugate`, `dot`, `choose` | ndarray's errors | phase 1 (done) |
| Mutation and fields | `fill`, `put`, `resize`, `setfield`, `getfield`, `setflags` | ndarray's answer (in place for the mutators) | C2 (number stores: P2.4) |
| Attributes and export | `base`, `data`, `device`, `dtype`, `flags`, `item`, `itemsize`, `nbytes`, `ndim`, `shape`, `size`, `strides`, `to_device`, `tobytes`, `tofile`, `tolist` | ndarray's answer (`item`, `tolist` unstripped) | C2 |
| Not in NumSharp | `ctypes`, `dump`, `dumps` | ndarray's disposition | — |
