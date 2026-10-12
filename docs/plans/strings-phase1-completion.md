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
| D7 | pythonnet crossing for `T` (NumPy has no buffer protocol for `T`). | Open (belongs to P1.8). |
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
| P1.8 | Bridges and leftovers: pythonnet `S` / `U` zero-copy both ways and `T` as a copy (D7); D10 (D6). | — | M | D10 only, with P1.1 |
| P1.9 | Gate, speed, docs: oracle sections for the new behavior with zero excuses; unit tests that cover all new phase-1 code; lifetime tests and runs under `FencedMemory` / guard pages; a mutation pass; a benchmark sheet (≥ 1.5× NumPy at 1K / 100K / 1M); docs. | with every package | M | yes |

Order: P1.1 and P1.2 → P1.3 → P1.4 → P1.5 and P1.6 → P1.9's closing pass. Each package ends in its own commit with its
tests and its oracle section.

After this plan, phase 2 loses P2.4 and gains the `T` forms of P2.1–P2.6 plus the loops only `T` has (`np.sum`,
`np.max` / `np.min`, `np.cumsum`).

## 5. State log

- 2026-10-10 — Investigation and plan (in chat). The owner answered D1, D2, D4, D5, D6 on 2026-10-12 and rebased
  `strings` onto journey4 (D9).
- 2026-10-12 — This document.
