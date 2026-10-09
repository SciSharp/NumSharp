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

### C5 The `chararray` members

- The 39 string methods with NumPy's five return rules (§1.6) and the 69 inherited members (Appendix B); `argsort` on
  the plain view.
- Each member lands with its function (C1–C4, or the phase-2 subphase). Size: medium (many small members).

### C6 Completion

- The whole surface strict in the string oracle; `coverage/oracle_map.json` (`char.*` op keys, `chararray.*` member
  keys) and `coverage/object_surfaces.py` (`char.chararray` → `NumSharp.NDCharArray`); the docs
  (`docs/website-src/docs/fundamentals/strings-and-bytes.md`, a `numpy.char` section; `.claude/CLAUDE.md`'s API list);
  the benchmark rows (`benchmark/strings`); the release notes.
- Size: small.

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

---

## Appendix A — the 53 module rows

| Subphase | Rows |
|---|---|
| C1 | `compare_chararrays`, `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` |
| C2 | `array`, `asarray`, `chararray` |
| C3 | `join`, `split`, `rsplit`, `splitlines` |
| C4 | `multiply` (after P2.3), `partition`, `rpartition` (after P2.5) |
| P2.3 | `add` |
| P2.5 | `str_len`, `count`, `find`, `rfind`, `index`, `rindex`, `startswith`, `endswith`, `lstrip`, `rstrip`, `strip`, `replace`, `expandtabs`, `center`, `ljust`, `rjust`, `zfill` |
| P2.6 | `capitalize`, `swapcase`, `title`, `upper`, `lower`, `isalnum`, `isalpha`, `isdigit`, `islower`, `isspace`, `istitle`, `isupper`, `isnumeric`, `isdecimal`, `mod`, `encode`, `decode`, `translate` |

The facade (C0) declares all 53; a row is complete when its computation lands.

## Appendix B — the 108 `chararray` members

| Group | Members | NumPy's result | Lands with |
|---|---|---|---|
| String methods, rule 1 (`asarray`) | `capitalize`, `center`, `expandtabs`, `ljust`, `lower`, `partition`, `rjust`, `rpartition`, `swapcase`, `title`, `translate`, `upper`, `zfill` | chararray; `U` owning copy, `S` view | P2.5 / P2.6 / C4 |
| String methods, rule 2 (ufunc wrap) | `lstrip`, `rstrip`, `strip`, `replace` | chararray view | P2.5 |
| String methods, rule 3 | `decode`, `encode`, `join` | ndarray of text | P2.6 / C3 |
| String methods, rule 4 | `count`, `endswith`, `find`, `index`, `rfind`, `rindex`, `startswith`, `isalnum`, `isalpha`, `isdecimal`, `isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper` | ndarray of int64 / bool | P2.5 / P2.6 |
| String methods, rule 5 | `split`, `rsplit`, `splitlines` | object array of lists (`NDListArray`) | C3 |
| Views and copies (the finalize hook) | `T`, `mT`, `transpose`, `swapaxes`, `reshape`, `ravel`, `flatten`, `squeeze`, `copy`, `take`, `repeat`, `compress`, `diagonal`, `real`, `imag`, `view`, `astype`, `byteswap`, `flat` | chararray (`diagonal`, `imag` read-only); `astype` / `view` to a numeric dtype per §1.6 | C2 |
| Finalize errors | `argmax`, `argmin`, `argpartition` | `ValueError("Can only create a chararray from string data.")` | C2 |
| Sorting | `argsort` (plain view), `sort` (in place), `searchsorted` | ndarray's answer | P2.2 |
| Truth | `all`, `any`, `nonzero` | ndarray's answer | P2.1 (done) |
| Numeric refusals | `max`, `min`, `sum`, `mean`, `prod`, `std`, `var`, `cumsum`, `cumprod`, `trace`, `clip`, `round`, `conj`, `conjugate`, `dot`, `choose` | ndarray's errors | phase 1 (done) |
| Mutation and fields | `fill`, `put`, `resize`, `setfield`, `getfield`, `setflags` | ndarray's answer (in place for the mutators) | C2 (number stores: P2.4) |
| Attributes and export | `base`, `data`, `device`, `dtype`, `flags`, `item`, `itemsize`, `nbytes`, `ndim`, `shape`, `size`, `strides`, `to_device`, `tobytes`, `tofile`, `tolist` | ndarray's answer (`item`, `tolist` unstripped) | C2 |
| Not in NumSharp | `ctypes`, `dump`, `dumps` | ndarray's disposition | — |
