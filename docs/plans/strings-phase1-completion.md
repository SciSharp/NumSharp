# Strings phase 1 completion — text as data for `S`, `U` and `StringDType` (`T`)

> Plan to finish Stage D phase 1 of `docs/plans/dtype-system.md`. Phase 1 must hold text as DATA end to end for all
> three NumPy 2.4.2 string dtypes: create, store, move, convert, print, save, load and cross the bridges. Only
> computation on text (comparisons and arithmetic on `T`, sorting, searching, the `np.strings` functions) stays in
> phase 2 (`docs/plans/strings-phase2.md`).
>
> Owner direction (2026-10-10): "we are supposed to complete the phase 1 fully, we are not even supporting variable
> strings … then we'll move to more complex functions." This overrules the earlier split, which scheduled NEP 55
> `StringDType` as "phase 3".
>
> Source of truth: `refs/numpy/` (2.4.2): `numpy/_core/src/multiarray/stringdtype/{dtype.c, static_string.c,
> casts.cpp, utf8_utils.c}`, `numpy/_core/src/multiarray/{arraytypes.c.src, scalartypes.c.src}`. Every fact below was
> probed against `numpy==2.4.2` on CPython 3.12.12 (win-amd64) on 2026-10-10.

---

## 0. State before this plan (branch `strings`, rebased onto journey4 `f6086479`)

| Part | State |
|---|---|
| `S` / `U` storage, descriptors, movement, string ↔ string casts, `.npy`, `loadtxt` / `savetxt` / `tofile` / `fromfile`, printing, `np.nditer`, numpy.ma, NumPy's refusals, shape-decided answers | Done (phase 1, then phase 2 P2.0 / P2.1 / P2.3). The string oracle has 38,150 cases. |
| `StringDType` (`T`) | Absent. `np.dtype("T")` raises `NotSupportedException: data type 'T' not understood`. |
| Number ↔ text | Deferred everywhere with `StringComputationNotSupportedException`: `u[0] = 5`, `np.ones(2, "U")`, `np.full(2, 7, "U")`, `np.pad(u, 1)`, `np.array(new object[] { "x", 1 })`, every `astype` between text and numbers. 43 code sites (29 `TextConversion`, 6 `NumberToText`, 2 `MixedKinds`, 6 direct throws, counted at `40f2d20f`); about 444 of the 970 deferred oracle cases. |
| pythonnet bridge | Imports `<U1` as `Char`, refuses wider `U` and every `S`; exports no text dtype. |
| D10 | `NDArray x = "[1, 2, 3]"` parses numbers; NumPy gives a 0-d `<U9`. |

## 1. Decisions

| # | Decision | Owner's answer (2026-10-12) |
|---|---|---|
| D1 | Move number ↔ text conversion (phase-2 P2.4) into phase 1. | **Yes.** `T` cannot store a number without it: its default `coerce=True` write is `str(value)`. |
| D2 | Storage layout for `T`. | **Port NumPy's 16-byte packed cell and arena exactly.** `itemsize`, `nbytes` and strides match NumPy, an all-zero cell reads `''` (so `np.zeros` / `np.empty` cost nothing), and `tobytes()` of a freshly built array can match NumPy byte for byte. |
| D3 | `.npy` / `.npz` for `T` (NumPy writes a pickle with descr `'|O'`). | Open (belongs to P1.7). |
| D4 | Storage lane for `T`. | **Use `NPTypeCode.String` and rely on the dtype information.** `T` shares the lane of `S` / `U`; the descriptor's meta class tells them apart (see §3). |
| D5 | `na_object` in C#. | **Yes:** `null` = Python `None`, a float NaN = `nan`, a `string` = a string NA, any other object = an "other" NA. Two factory overloads, `StringDType(bool coerce = true)` and `StringDType(object na_object, bool coerce = true)`, keep "not given" different from `null`. |
| D6 | D10: the numeric parse in the implicit `string → NDArray` conversion. | **Remove it.** A C# string is NumPy's `np.array(str)`: a 0-d `U`. |
| D7 | pythonnet crossing for `T` (NumPy has no buffer protocol for `T`). | **NumPy's C API** (2026-10-13): an exact copy in both directions through the `NpyString_*` functions of the `_ARRAY_API` capsule. With it, the owner decided: NumPy's `<U1` imports as `U1`, a dtype of its own (a C# `char` is 2 bytes; `Char` keeps exporting as `<u2`); a NumPy object array is refused, except one whose every element is a `str`, which is copied (it cannot be viewed); a missing-data object with no C# form (`pd.NA`, an arbitrary object) is refused. |
| D8 | ML.NET text columns and Parquet string columns as `T`. | Open (later). |
| D9 | Branch base. | Done: the owner rebased `strings` onto journey4 `f6086479`. |

## 2. NumPy 2.4.2 facts for `StringDType`

**Descriptor.** `np.dtype('T')` and `np.dtype('<T')` are `StringDType()`; `'T8'` and `'StringDType'` are not
understood; `newbyteorder` raises `TypeError: Cannot use DescrNewByteOrder for this new style DTypes.`. `char` and
`kind` `'T'`, `num` 2056, `itemsize` 16, `alignment` 8, `name` `'StringDType128'`, `str` `'StringDType()'`, `descr`
`[('', 'StringDType()')]`, `isbuiltin` 0, `isnative` True, `hasobject` True, `flags` 107
(`USE_SETITEM | USE_GETITEM | NEEDS_INIT | LIST_PICKLE | ITEM_REFCOUNT`), `byteorder` `'|'`, `type` `str`.
Parameters: `na_object` (the attribute does not exist when it is not set) and `coerce` (default True). `size=` is
`TypeError: 'size' is an invalid keyword argument for StringDType()`. Equality and hash use both parameters
(`StringDType(na_object=nan) == StringDType(na_object=nan)`). repr: `StringDType()`, `StringDType(na_object=nan)`,
`StringDType(na_object=None)`, `StringDType(na_object='NA')`, `StringDType(coerce=False)`. `np.dtype(StringDType)`
(the class) is `dtype('O')`. `np.isdtype(T, …)` raises `TypeError: dtype argument must be a NumPy dtype, but it is a
<class 'numpy.dtypes.StringDType'>.`. `np.issubdtype(T, np.character)` and `np.issubdtype(T, np.flexible)` are False.

**Storage** (`static_string.c`). One 16-byte cell per element (`npy_packed_static_string`). The last byte holds the
flags (`MISSING 0x80`, `INITIALIZED 0x40`, `OUTSIDE_ARENA 0x20`, `LONG 0x10`). Up to 15 bytes are stored in the cell
(flags `INITIALIZED | OUTSIDE_ARENA`, size in the low nibble). Longer text lives in a per-array ARENA, as
`(offset, size | flags << 56)`: a medium string (≤ 255 bytes) has a 1-byte size prefix in the arena, a long one an
8-byte prefix. A string that a later write makes longer than its old allocation moves to the HEAP (`OUTSIDE_ARENA`,
the field holds a pointer). An all-zero cell is an uninitialized empty string. Views share the arena; copies duplicate
each string. `tobytes()` returns the cells (`['ab']` is `b'ab' + 13 * b'\x00' + b'b'`; two 20-byte strings are offsets
1 and 22).

**Writes.** With `coerce=True`, `setitem` stores `str(value)` for any value: numbers (`3.25` → `'3.25'`), `None` →
`'None'` and `nan` → `'nan'` (unless they are the `na_object`), even arrays (`np.array(['zz', 'y'])` → `"['zz' 'y']"`).
`bytes` decode as UTF-8 (`b'\x80'` raises `UnicodeDecodeError`). A lone surrogate raises `UnicodeEncodeError: 'utf-8'
codec can't encode character '\ud800' in position 0: surrogates not allowed`. With `coerce=False`, any value that is
not a `str` raises `ValueError: StringDType only allows string data when string coercion is disabled.`.

**Creation.** `np.array([...], dtype='T')`; `np.empty` / `np.zeros` give `''`; `np.ones(3, 'T')` gives `'1'`;
`np.full(2, 7, 'T')` gives `'7'`; `np.pad(t, 1)` pads with `'0'`; `np.zeros_like` keeps `T`. A ragged list raises the
inhomogeneous-shape `ValueError`. Scalars read back as Python `str` (not `np.str_`), or as the `na_object`.

**Promotion.** `T` + `U` → `T`. `T` + `S`, `T` + numbers, `T` + bool raise `DTypePromotionError: The DTypes <class
'numpy.dtypes.StringDType'> and <class 'numpy.dtypes.BytesDType'> do not have a common DType. …`. The common instance
of two `T` descriptors: different `na_object`s raise `TypeError: Cannot find common instance for incompatible dtypes
'StringDType(na_object=None)' and 'StringDType(na_object=nan)'`; otherwise the FIRST operand's descriptor (its
`coerce` too).

**Cast safety** (`can_cast`). Numbers and bool → `T`: safe. `T` → numbers: unsafe only. `T` → bool: same_kind.
`T` ↔ `U` and `T` ↔ `S`: same_kind.

**Casts.** `U` → `T` and `S` → `T` (`S` DECODES UTF-8, so `b'\x80'` raises `UnicodeDecodeError: 'utf-8' codec can't
decode byte 0x80 in position 0: invalid start byte`). `T` → `U` / `S` need a size: unsized raises `TypeError: cannot
cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>.` (also `astype(str)` / `astype(bytes)`); sized casts cut
to the width; `T` → `S` ENCODES ASCII (`UnicodeEncodeError: 'ascii' codec can't encode character …`). An NA value
becomes `str(na_object)` in `U` / `S` (`'None'`, `'nan'`, `'NA'`). `T` → `T` with another `na_object` converts NA
values. `T` → bool: `''` False; a NaN-like NA True; a `None`-like or string NA False. `T` → numbers parse (`'12'` →
12, `'nan'` → nan); an NA value raises `ValueError: Arrays with missing data cannot be converted to a non-nullable
type`. Numbers → `T` format as the NumPy scalar's `str()` (`1e+16`, `1e-05`, `(1+2j)`, `True`).

**Behavior that differs from `U`.** `view(dtype)` raises `TypeError: Cannot change data-type for array of
references.`; `memoryview` raises `ValueError: cannot include dtype 'numpy.dtypes.StringDType' in a buffer`;
`np.frombuffer(..., dtype='T')` raises `ValueError: cannot create an OBJECT array from memory buffer`;
`np.ndarray((1,), 'T', buffer=bytearray(16))` works (it reads the bytes as cells). `np.nditer` needs `refs_ok`
(`TypeError: Iterator operand or requested dtype holds references, but the NPY_ITER_REFS_OK flag was not enabled`).
`np.dot` raises `TypeError: This function currently only supports native NumPy dtypes and old-style user dtypes, but
the dtype was StringDType().`. `np.sum`, `np.max`, `np.cumsum` COMPUTE on `T` (NumPy registers `add`, `maximum`,
`minimum` loops for it): these are phase-2 deferrals for `T`, not refusals.

**NumPy bug.** NumPy 2.4.2 crashes the process (segmentation fault) on `np.place` over a `T` array. The oracle
generator runs that case in a child process.

**Speed** (µs, 1K / 100K / 1M, two pinned P-cores; random lowercase strings of 1–8 characters):

| Operation | NumPy `T` | NumPy `<U8` |
|---|---|---|
| copy | 15 / 1,631 / 16,421 | 1.9 / 754 / 8,400 |
| `a[::2].copy()` | 8.9 / 695 / 8,345 | 1.9 / 552 / 5,256 |
| `np.take(a, idx)` | 115 / 12,109 / 169,495 | 4.4 / 1,079 / 20,643 |
| `a[mask]` | 35 / 3,200 / 34,355 | 3.9 / 1,007 / 9,830 |
| `np.concatenate([a, a])` | 30 / 3,373 / 33,687 | 4.6 / 1,589 / 16,627 |
| `np.where(mask, a, a[::-1])` | 112 / 11,447 / 113,312 | 8.8 / 1,930 / 16,543 |
| `np.repeat(a, 2)` | 226 / 23,535 / 235,336 | 7.0 / 1,805 / 17,211 |
| `np.array(list, dtype)` | 77 / 7,957 / 81,159 | 56 / 6,219 / 62,450 |
| `U` → `T` / `T` → `U` | 59 / 6,292 / 62,723 · 28 / 3,840 / 38,329 | — |

With 20–40-character strings (arena strings) NumPy's `T` copy costs 43 / 8,603 / 85,821 µs. Inline cells refer to no
arena memory, so a copy can move them as raw 16-byte blocks; ≥ 1.5× NumPy is the floor, ≥ 3× a fair target.

## 3. Design consequences of D4 (one `NPTypeCode.String` lane)

`T` shares `NPTypeCode.String` with `S` / `U`, so the storage type code no longer says how an element is laid out.
Three rules keep that safe:

1. **The meta class decides.** `S` / `U` descriptors have `StringDTypeMeta`; `T` descriptors have their own
   `VStringDTypeMeta` (not a subclass). Every route that treats an element as an opaque `itemsize`-byte blob (the unit
   views, the whole-element copy kernels, `WhereBytes`, the fancy / mask kernel route, the take / put / place /
   putmask kernels, concatenate's slab copies, the codec) is gated on the FIXED-WIDTH meta, not on the type code.
2. **`T` never takes a raw byte path.** A `T` cell holds an arena offset or a heap pointer that is valid only for its
   own buffer's allocator. A raw copy into another array would read garbage. So `T` has its own slice type
   (`VStringArraySlice`, not a `StringArraySlice`), every `T` copy goes through one reference-aware copy core, and
   every entry that is not yet ported refuses a `T` array loudly (NumPy's text where NumPy refuses, else a clear
   `NotSupportedException`). A silent raw copy is the one outcome that is not allowed.
3. **The allocator belongs to the buffer.** The arena and the heap strings of a `T` buffer are owned by its memory
   block: the block's release action (the `External` disposer) runs NumPy's clear loop (frees heap strings), frees the
   arena, then returns the cells to the pool. Views and slices share the block, so they share the allocator; a copy
   gets a new block and a new allocator. One lock per allocator, taken once per operation (not per element) — NumPy's
   allocator mutex; the single-threaded rule forbids parallel kernels, not this lock.

## 4. Work packages

| ID | Work | Depends on | Size | This round |
|---|---|---|---|---|
| P1.1 | Number ↔ text for `S` / `U`: the formatter (the NumPy scalar's `str()` for bool, integers, float16 / 32 / 64, complex; NumSharp's Decimal and Char), the parser (CPython `int()` / `float()` / `complex()` with NumPy's errors), and the 43 sites plus the parse-fed numerics. | — | L | yes |
| P1.2 | `T` descriptor: `np.dtypes.StringDType(...)`, `np.dtype("T")`, the descriptor surface, equality and hash, repr, promotion, the common instance, the cast-safety table, `can_cast` / `promote_types` / `result_type`, `isdtype` / `issubdtype`. | — | M | yes |
| P1.3 | `T` storage: `VStringArraySlice` with NumPy's cell layout, the allocator (arena, heap, clear loop, lock) owned by the memory block, the UTF-8 codec, the NA kinds, allocation / alias / slice / clone. | P1.2 | L | yes |
| P1.4 | `T` movement: the reference-aware copy core (inline cells move as blocks; arena and heap strings are duplicated), behind every movement entry, and the loud refusal on every raw path. | P1.3 | L | yes |
| P1.5 | `T` creation, element access and printing: `np.array` / `asarray` / `empty` / `zeros` / `ones` / `full` and the `_like` forms, the `setitem` coercion (uses P1.1), `item` / `tolist` / `foreach` / `AsStringArray` / `(Array)`, NA objects, `repr` / `str` / `array2string`. | P1.1, P1.3 | M | yes |
| P1.6 | `T` casts: `T` ↔ `T` (NA rules), `T` ↔ `U` / `S`, `T` ↔ bool, `T` ↔ numbers (uses P1.1). | P1.1, P1.4 | M | yes |
| P1.7 | `T` in the rest of the library: `np.nditer` (`refs_ok`, buffered casts), numpy.ma (fill `'N/A'`), `loadtxt` / `savetxt`, `.npy` (D3), the whole-surface sweep with `T` (NumPy's refusals and shape-decided answers in NumPy's order; computation deferred to phase 2). | P1.4–P1.6 | L | later |
| P1.8 | Bridges and leftovers: pythonnet `S` / `U` zero-copy both ways and `T` as a copy (D7); D10 (D6). | — | M | done (D10 with P1.1, the bridge 2026-10-13) |
| P1.9 | Gate, speed, docs: oracle sections for the new behavior with zero excuses; unit tests that cover all new phase-1 code; lifetime tests and runs under `FencedMemory` / guard pages; a mutation pass; a benchmark sheet (≥ 1.5× NumPy at 1K / 100K / 1M); docs. | with every package | M | yes |

Order: P1.1 and P1.2 → P1.3 → P1.4 → P1.5 and P1.6 → P1.9's closing pass. Each package ends in its own commit with its
tests and its oracle section.

After this plan, phase 2 loses P2.4 and gains the `T` forms of P2.1–P2.6 plus the loops only `T` has (`np.sum`,
`np.max` / `np.min`, `np.cumsum`).

## 5. State log

- 2026-10-10 — Investigation and plan (in chat). The owner answered D1, D2, D4, D5, D6 on 2026-10-12 and rebased
  `strings` onto journey4 (D9).
- 2026-10-12 — This document.
- 2026-10-12 — **P1.1 done.** Number ↔ text for `S` / `U` through every route that converts. Engine:
  `Backends/Strings/NDTextFormat.cs` (the NumPy scalar's `str()` per type, `REQUIRED_STR_LEN`), `NDTextParse.cs` (CPython
  3.12 `int()` / `float()` / `complex()`, NumPy's error texts in both quoting styles, the LP64 overflow texts, the
  4300-digit limit, strict UTF-8 for `S → complex`), `NDTextConversion.cs` (NumPy's raw assignment order, blocks of 128
  committed whole, masked runs), `Utilities/PyUnicode.cs` (CPython 3.12's decimal and space tables). Routes: `astype`,
  setitem / `fill` / `full` / `ones` / `zeros_like` / `pad` / `copyto` / `np.array`, the iterator's buffered casts, numpy.ma
  fills, `np.add` under a string `dtype=` over numbers (`AddUnderStringRequest`), comparisons and logical ufuncs into a
  text `out=`, every numeric ufunc / clip / matmul / modf / frexp into a text `out=` (`IntoTextOut`: with `where=` the out's
  text is parsed into the loop dtype first, as NumPy's READWRITE buffer does), numeric-`dtype=` reductions over text
  (`ParseForNumericReduce`: prod / cumprod / nancumprod / max / min), `np.cov` / `np.corrcoef` with a numeric `dtype=`,
  `np.linalg.norm`, `np.i0`, `np.roots`, `np.poly`, `matrix_power(text, 0)`. D6 done: a C# string is a 0-d `U` array.
  Found on the way and fixed: numpy.ma's `astype` converts the stored fill BEFORE the data (NumPy's `__array_finalize__`
  runs at allocation); a masked fill value prints with its mask in the conversion error; `np.power`'s integer loop raised
  for a masked-off negative exponent (NumPy raises inside the loop); `frexp`'s cast errors name "output 1" / "output 2".
  Gate: oracle section `strconv` (4,487 cases, strict), the conversion cases of the other sections flipped to phase 1,
  per-section deferral ceilings (`StringOracleTests.DeferralCeilings`: 537 deferrals left, all sorting / searching / set
  routines / histograms / quantiles; were 970), `TextNumberConversionTests` (20). Speed (NPY/NS, two pinned P-cores,
  `DOTNET_TC_CallCountingDelayMs=0`): i8→U 5.7–7.9×, f8→U 2.6–3.0×, U→i8 5.4–5.6×, U→f8 3.2–3.7×, S→f8 2.1–2.5×, U→c16
  3.5–4.0×, copyto f8→U32 2.5–2.9×, `np.add(f8, f8, dtype='U')` 2.2–3.0× at 1K / 100K / 1M. Benchmark trap: without
  `DOTNET_TC_CallCountingDelayMs=0` the first row of a run is measured at tier 0 even after 120 warm-up calls (U→f8 at 1K
  read 1.36× there, 3.7× with it).
- 2026-10-12 — **P1.2 done.** The `StringDType` descriptor, without arrays. Class `DTypes/VStringDTypeMeta.cs` (the
  DType class `numpy.dtypes.StringDType`: num 2056, kind / char `'T'`, a 16-byte cell aligned to 8, flags 107,
  parametric, no singleton — every descriptor is fresh — and the `NPTypeCode.String` lane shared with `S` / `U` per D4:
  `OwnsTypeCode` / `OwnsScalarType` are false, so `FromTypeCode(String)` and `FromScalarType(string)` stay `StrDType`).
  Parameters `DTypes/VStringParameters.cs` (`na_object`, `coerce`, and NumPy's derived `has_nan_na`, `has_string_na`,
  `na_name`, `default_string`, computed once). Factories `np.dtypes.StringDType(bool coerce = true)` and
  `np.dtypes.StringDType(object na_object, bool coerce = true)` (D5; a bool sentinel goes BY NAME, a positional bool is
  `coerce`); `np.dtype("T")` with any byte-order prefix. Descriptor surface (`str` = `repr` = `StringDType(...)` with
  Python's repr of the sentinel, name `StringDType128`, `isbuiltin` 0, `byteorder` `'|'`, `na_object` / `coerce` with
  NumPy's AttributeError when absent), equality = NumPy's `_eq_comparison` + `na_eq_cmp`, hash consistent with it,
  `newbyteorder` refuses (TypeError after the order is parsed), `issubdtype` through Python's `str` hierarchy (`U` is a
  subtype of `T`, not the reverse; `T` under no category), `isdtype` refuses `T` as dtype or kind, a `T` INSTANCE as a
  ufunc `dtype=` is NumPy's new-style refusal, promotion (`T` with `U` is `T`, no other class; the common instance takes
  the present sentinel and the FIRST descriptor's coerce; NumPy's sequence reduction and its texts), the cast table of
  `casts.cpp` (`DTypes/VStringCastingImpls.cs`: `T → T` spec unsafe / resolved no-cast, sentinel → no sentinel unsafe;
  `U` / `S` → `T` same_kind; `T` → `U` / `S` same_kind with NumPy's unsized-destination TypeError; bool → `T` safe,
  `T` → bool same_kind (resolved unsafe); numbers and datetimes → `T` safe, `T` → numbers / datetimes unsafe; decimal
  casts as float64, char as uint16). The CPython protocols the descriptor runs: `Backends/Strings/NDPyObject.cs`
  (`str()`, `repr()`, `==` and a hash over the C# boundary map; NumPy's scalar rules where an `np.float16` / decimal takes
  part — `np.float16(2048) == 2049` is True — and Python's container `==`) and `Backends/Strings/NDUtf8.cs` (the strict
  UTF-8 codec with CPython 3.12's error texts; validation and transcoding on .NET's vectorized routines, CPython's
  diagnosis only on failure). Allocating a `T` array is refused loudly until P1.3 (`UnmanagedStorage.ThrowIfVStringStorage`).
  Found and fixed on the way: a string sentinel with a lone surrogate is CPython's `UnicodeEncodeError` at construction
  (NumPy keeps `default_string` and `na_name` as UTF-8); `np.dtype` raised "nested array dtypes" for any string with a
  parenthesis, where NumPy's grammar says "not understood" (`'(2)i4'`, `'T()'`) or reaches the datetime parser
  (`'M8(s)'`); CPython compares an int and a float exactly even when the int's double overflows (`2**1100 != inf`).
  Validation outside the corpus: `NDUtf8` against CPython over 373,249 inputs (every 1- and 2-byte sequence, 3- and 4-byte
  sweeps, 60,000 random byte strings, ~36,000 strings with lone surrogates), `NDPyObject` over 83 objects × `str` /
  `repr` and a 57 × 57 `==` matrix of Python numbers, `np.float16`, `np.float64` and containers — all identical. Gate:
  oracle section `tdtype` (1,188 cases, strict), `DTypes/StringDTypeDescriptorTests` (25),
  `Backends/Strings/StringDTypeProtocolTests` (10). Test trap: a lone surrogate cannot travel in `[DataRow]` (attribute
  arguments are UTF-8 in the metadata) nor in `[DynamicData]` (the test platform serializes rows as XML and refuses a
  high surrogate followed by another): keep such strings in the method body.
- 2026-10-12 — **P1.3–P1.6 done** (commit `ea04c19e`). `StringDType` arrays. Storage
  (`Backends/Strings/VStringAllocator.cs`, a statement-for-statement port of `static_string.c`: the packed cell, short
  strings in the cell, medium / long strings in the per-buffer arena behind a 1- / 8-byte prefix, heap strings for a
  longer rewrite, NA cells that keep their slot, the clear loop; `Backends/Unmanaged/VStringArraySlice.cs`: the cells,
  the allocator and the descriptor instance; the cells' memory block owns the allocator, so views keep it alive). The
  safety gate: `DTypeSize` / `dtypesize` THROW for a `T` storage, so every raw byte route fails loudly; byte arithmetic
  reads `ItemBytes` (16). Movement (`NDVStringCopy.cs`: `PyArray_AssignArray` + `string_to_string`, NumPy's overlap
  rules; `NDVStringMove.cs`: gathers run the numeric route on the CELL view and re-home each cell into a fresh buffer,
  scatters write int64 marks and then duplicate the marked values) behind every movement entry (take, fancy / mask get
  and set, where, repeat, resize, choose, select, compress, extract, take_along_axis, put_along_axis, put, place,
  putmask, fill_diagonal, diag / diagflat, tril / triu, copyto, and the compositions over them). Creation, element access
  and printing (`stringdtype_setitem`'s coercion and NA rules; `repr` with Python's quoting and the NA object as itself).
  Casts (`NDVStringCasting.cs`, the `casts.cpp` loops `T` ↔ `T` / `U` / `S` / bool / numbers, NumPy's error texts).
- 2026-10-12 — **P1.9 part 1** (commit `ecb27efb`). The `StringDType` oracle: sections `tstore` (324), `tcreate`
  (3,402), `tmove` (1,990), `tcast` (2,608) — 8,324 strict cases. A `T` value travels as values, never as cells (an
  arena offset or a heap pointer of NumPy's allocator means nothing to NumSharp); `tobytes()` cells are compared with
  the heap pointers masked. The replay found 12 divergences, all fixed: `tobytes()` of a non-contiguous array ravels it
  first; the copy core walks NumPy's destination-stride order (arena offsets follow the walk); `np.resize` keeps its view
  of a fresh buffer; `np.where` takes NumPy's output layout; `.data` and `np.frombuffer` refuse `T` with NumPy's texts;
  `None` / bool / str literals store by setitem; the store functions convert their values as `PyArray_FromAny(values,
  dtype)`; `NDArray.Scalar(value, T)` is setitem; np.nditer's per-step views share the operand's allocator.
- 2026-10-13 — **P1.9 done** (commit `8357fa2f`). The closing pass over all phase-1 code.
  - Coverage: 99.37 % of the executable lines phase 1 added to `src/NumSharp.Core` (3,600 / 3,623; a cobertura run of the
    unit + oracle suites joined with `git diff -U0 0cf77427`). The 23 left cannot run: a big-endian host, texts past 2 GB,
    allocation-failure returns the fault hook does not reach, guards behind conditions the callers exclude.
  - Unit tests: `Backends/Strings/StringDType{Storage,Movement,Cast,Creation,Fault,Fence}Tests`, the P1.1 residuals in
    `TextNumberConversionTests`, the docs page's `StringDType_*` tests; the oracle's
    `StringDType_NoAllocatorOutlivesTheReplay` (lifetime). Fault injection: `VStringAllocator.FailAllocationAfter(n)` fails
    the (n+1)-th arena or heap allocation of the thread, so every `MemoryError` path is tested with NumPy's text and the
    disposal of the half-built result. `FencedMemory`: cells between two inaccessible pages. Guard pages: the unit suite
    (only the 4 known pool-counting failures, no crash) and the oracle string sections.
  - Mutation pass: 27 planted bugs, 26 killed, 1 equivalent (an element view's offset, always 0). Two fast-path mutants
    survived their first round (a heap destination overwritten without its free, an empty source copied as a short
    cell); two tests probed against NumPy now kill them.
  - Fixes the tests found: an index of integers alone stores a Python object by setitem (`array_assign_item`, setitem's
    texts); `fill` and slice assignment run the copy loop (free, then allocate, the copy loop's texts); a `T` → number
    cast reports a corrupt cell with `load_non_nullable_string`'s text; np.nditer allocates an operand with a `T`
    `op_dtype`; `np.fill_diagonal(t, null)` is Python's `None`; a cell copied onto itself keeps its text (NumPy 2.4.2
    makes it `''`, a bug).
  - Speed (NPY/NS, two pinned P-cores, best of rounds): geomean 5.35×, minimum 1.80×, over copy, arena copy, stepped copy,
    `np.take`, a mask, `np.concatenate`, `np.where`, `np.repeat`, `np.array(list)`, `U` → `T`, `T` → `U8` at 1K / 100K / 1M
    (the table is in the commit). Levers: the copy core moves a short non-empty cell into a cell without arena / heap text
    as two 8-byte words; a bulk copy reserves the arena's exact total first, counting only the cells that
    `heap_or_arena_allocate` puts in the arena (capacity only, never an offset); allocator memory goes through the pool's
    coarse accounting instead of a `GC.AddMemoryPressure` per growth; `np.array(list, T)` writes short ASCII texts straight
    into their fresh cells under one lock; `T` → `U` decodes the ASCII prefix without the UTF-8 decoder.
- 2026-10-13 — **This round (P1.1–P1.6 and P1.9) is complete.** Later rounds: P1.7 (`T` in np.nditer's buffering and
  casting, numpy.ma, `loadtxt` / `savetxt`, `.npy`, the whole-surface sweep) and P1.8 (pythonnet `S` / `U` zero-copy
  both ways, `T` as a copy). Then phase 2 (computation on text).
- 2026-10-13 — **P1.8 done** (the pythonnet bridge; D7 answered with three companion decisions, §1).
  - `S` / `U` are ordinary buffers: `ToNumpy` / `AsNDArray` share them both ways in every layout (the dtype maps and
    `WrapExternal` carry the sized descriptor); `<U1`, `array('w')` and a 4-byte `wchar_t` buffer view as `U1`, a 2-byte one
    as `Char`; big-endian `>U<n>` copies byte-swapped; `np.str_` / `np.bytes_` are 0-d `U` / `S`, an empty one copies to
    `''`. `FromNumpyDtypeStr` / `FromBufferFormat` return `DType`.
  - `T` copies through `NumpyStringApi` (slots 313 load, 314 pack, 315 pack_null, 316 acquire, 318 release, 160
    `PyArray_GetPtr`) with `NpyString_dup`'s rules. Import: one pass settles NA / empty / short cells and sums the arena,
    one reservation, a second pass duplicates the long cells; the cells equal numpy's own C-order copy's. Export: one pass
    into a fresh `np.empty(shape, StringDType(...))`, locking the RESULT's descriptor allocator; the cells equal
    `np.array(list, dtype)`'s. `na_object` and `coerce` cross; an NA object with no C# / Python form is refused.
  - An object array of `str` (a pandas text column) copies as `StringDType()` through numpy's own
    `astype(StringDType(coerce=False))`; any other object array is refused, naming the first non-`str` element.
  - The codec: Auto decode copies `T` and object-of-`str`; Auto encode copies `T` (View declines it). `ToTorch` refuses text.
  - Fixed on the way: every import paid two .NET exceptions for a failed `memoryview` (~28 µs; now a Python-side `try`,
    `PythonRuntimeInterop.TryMemoryview`), and every import paid the built-in Torch / Pandas adapters' MRO walk (5.3 µs;
    `PythonArrayAdapterRegistry.IsPlainBufferExporter` answers for ndarray / bytes / bytearray / memoryview / array.array
    from the type's name). A 1,000-element float64 `ToNDArray` went 9.2 → 3.7 µs.
  - Gates (live NumPy 2.4.2): `TextFixedWidthCrossingTests` (10), `TextStringDTypeCrossingTests` (9: every NA kind × both
    `coerce` × every layout, both directions, cells compared, fault injection, lifetime), the docs page's census (55
    varieties: 50 view, 4 copy, 1 rejected) and troubleshooting rows. The interop suite: 669 / 669 on net10.0 and net8.0
    (numpy only), the ecosystem run 94 passed, 1 skipped (no GPU) on both.
  - Speed (µs, minimum of three pinned runs of best-of-9; NumPy's `a.copy()` / the Python `tolist()` route / ToNDArray /
    ToNumpyCopy): short 1K 17.0 / 160 / 9.5 / 11.7, 100K 1,874 / 17,112 / 826 / 1,216, 1M 18,353 / 171,185 / 7,875 /
    11,840; long 1K 43.4 / 188 / 36.0 / 49.5, 100K 7,749 / 25,023 / 4,285 / 8,284, 1M 79,156 / 258,126 / 41,217 / 85,795;
    mixed with 10 % NA 1K 30.0 / 190 / 22.8 / 28.9, 100K 3,590 / 21,359 / 2,503 / 3,422, 1M 46,093 / 222,544 / 23,958 /
    44,835. Against the Python route every cell is 3–22×. Against NumPy's own copy the import is 1.2–2.3× and the
    short-string export 1.45–1.55×; the long-string export is 0.88–1.05×: numpy's allocator stores each long string
    through `NpyString_pack`, the same work its own copy does, and the C API has no way to reserve numpy's arena.
    Fixed-width views are flat: `ToNumpy` 5.7–6.6 µs, `AsNDArray` 4.0–4.1 µs from 1K to 1M elements.
- 2026-10-13 — **Conversions and NEP 50 across types** (owner's request: "'Converts' and other classes that interop and
  NEP50 across types, fix them all").
  - `Converts` text is NumPy's (`Utilities/Converts.NumPyText.cs`): number → text is the NumPy scalar's `str()` through
    `NDTextFormat` (`str(np.float16(1))` is `'1.0'`, a complex `'(1+2j)'`, char its uint16 number, decimal its float64);
    text → number is NumPy's store of a Python str (`NDTextParse.ParseObject`: CPython `int()` / `float()` / `complex()`
    and NumPy's packing check, errors quoting `'1.5'`); a bool target takes the str's truth. Both culture-invariant: the
    `IFormatProvider` overloads ignore the provider for dtype values. Every route reaches it: the typed and object
    `To*` / `ToString` overloads, `ChangeType` (generic, `NPTypeCode`, .NET `TypeCode`, with provider),
    `FindConverter` / `Converts<T>` (a string source no longer truncates through `IConvertible.ToDouble`),
    `NonGenericConvert`, `TypelessConvert`, `ArrayConvert` (its `sbyte[]` / `Half[]` / `Complex[]` → `string[]` used .NET
    text; `ToComplex(string[])` is `complex()`: `''` raises as in NumPy, null stays zero). A char target follows astype
    for numbers (a float past int32 through the int32 intermediate: `4294967295.0` → 0, was 0xFFFF) and the uint16 parse
    for text (`"70000"` raises).
  - Other interop classes: `py.complex` is CPython's `complex(str)` (was a culture-dependent regex that refused `'1e3'`,
    `'inf'`, `'(1+2j)'` and accepted `'1+'`); `Converts.ToDateTime64(string)` reads the invariant culture;
    `np.loadtxt`'s converter results are stored by `PyArray_Pack` (`ReductionInitial.TryPack`: a float truncates — the
    `Convert` route rounded 1.7 to 2 —, a str parses, a `Half` wraps, None is NumPy's positive NaN / `nan+nanj` / False);
    `np.asanyarray` of mixed object items converts through `Converts` (`Convert.ToDouble` threw for a `Half` item).
  - Promotion engines: `NDIterCasting.PromoteTypes` is the table (its own rules differed on 30 of 225 pairs: int32 +
    float32 gave float32), `NDIterCasting.FindCommonDtype` and the iterator's VIRTUAL operand take NumPy's
    order-independent `result_type`, `NDExprTypeRules.PromoteStrong` keeps char beside bool / uint8 (was uint16), and the
    set promotion behind `np._FindCommonType` (`_can_coerce_all`) is `PyArray_PromoteDTypeSequence` (the old loop kept
    only the last pair of four or more types; three types folded in list order: 10 of NumPy's 2,197 triples differed).
  - NEP 50 with text: a C# number literal is a WEAK Python scalar wherever a promotion reads the mark
    (`NDArray.Nep50Operand`, `NDArray.FromPythonOperand` for the `object` overloads), so `np.where(c, u, 1)`,
    `np.choose(idx, [u, 1])`, `np.select([c], [u], default=1)` and `np.r_[u, 1]` raise NumPy's `DTypePromotionError`
    (they promoted the literal's int32 to `<U11`). `np.select` makes NumPy 2.4.2's ONE `result_type` call over the choices
    and the default (one error text). `np.choose` converts each choice to the common dtype under 'safe' (a U choice beside
    a `StringDType` one raises NumPy's TypeError). `np.r_` / `np.c_` promote text by descriptor (`np.r_[i, u]` is `<U21`).
    A marked literal inside an `NDArray[]` / NDArray default is weak in the numeric routes of `np.choose`, `np.select` and
    `np.r_` too (`np.choose(idx, [int8, 1000])` is int8 `[1, -24]`, was int32). `np.put` / `np.insert` of a str into a
    numeric array quote the str (`'1.5'`, was `np.str_('1.5')`).
  - Open, for the owner (next phase): the ufunc family and `np.where`'s numeric route read EVERY 0-d operand as weak (the
    house rule, `[Misaligned]`, registry branch (1)); NumPy 2.x reads a 0-d array or a NumPy scalar as strong. With the
    marks a library can tell them apart, but ~373 library sites build their scalar operands with `NDArray.Scalar` and rely
    on the weak reading.
  - Gates: `Casting/ConvertsNumPyTextTests` (16), `DTypes/PromotionEngineParityTests` (6),
    `APIs/Nep50LiteralPromotionTests` (11), `ConvertsBattleTests` (6 expectations moved from .NET text to NumPy's). Probes
    against NumPy 2.4.2: Converts text 16,048 route × value × culture checks, 0 differ except the 280 LP64 "Python int too
    large to convert to C long" texts NumSharp pins (the probe ran Windows NumPy); scalar casts vs astype 0 of 7,875 (were
    7); engines 0 of 225 pairs and 0 of 5,197 sets; element APIs 0 of 815; text `where` / `choose` / `select` / `r_` 3 of 172
    (NumPy reads a bytes scalar in `r_` as a dtype spec; C# has no bytes scalar); numeric 74 of 420, all the house 0-d rule
    (28), complex64 (10) or probe artifacts (36).
- 2026-10-13 — **P1.8 validated** (owner's request: "Validate and confirm the implementation, add integration tests,
  covering edge cases and more"; `0409db83`).
  - Two fixes. An EMPTY read-only source imported as a writeable view: the three view routes returned a fresh empty array
    before they read the read-only flag. It is now refused by default and non-writeable on opt-in, like a non-empty one,
    and an empty NumSharp broadcast exports as a read-only numpy array. A `StringDType` whose `na_object` is an `np.str_`
    failed to import: pythonnet's `As<string>()` compares the type with `str` exactly, so the value is read through `str()`.
  - Gate: `TextCrossingEdgeCaseTests` (14, live numpy, cells compared with numpy's own copy): cell histories (heap
    rewrites, NA / `''` over arena text, a short text over an arena cell past 64 KiB keeping the old offset's bytes), UTF-8
    sizes on every cell and prefix boundary, more NA objects both ways and NumSharp-born ones (int, short, byte, ulong,
    float, Half, char), ranks 5–6, empty / read-only / broadcast sources, a 100,000-element arena, 240 seeded random arrays
    and views, the codec pipeline, eight threads, and the non-text import routes the P1.8 routing change touched. Four
    planted defects each turn at least one of them red.
  - A scratch differential fuzzer ran 10,000 `T` and 10,000 fixed-width random cases (0–6-D, 19 NA kinds, histories,
    random views, both `U` byte orders, random read-only): 0 failures after the empty fix, every live counter at
    baseline. The interop suite: 683 / 683 on net10.0 and net8.0; the ecosystem run 94 passed, 1 skipped (no GPU).
  - Unchanged: the long-string export stays 0.88–1.05× numpy's own copy (`NpyString_pack`, the work numpy's copy does).
