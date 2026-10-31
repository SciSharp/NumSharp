---
paths:
  - "src/NumSharp.Core/Ma/**"
  - "test/NumSharp.Tests/Ma/**"
  - "test/oracle/{gen_ma_oracle,gen_ma_surface_oracle,ma_setops_reference}.py"
---

# `numpy.ma`: `NDMaskedArray`

`np.ma.*` and `NDMaskedArray` (`src/NumSharp.Core/Ma/`) run numpy.ma's statements in numpy.ma's order, for every dtype
the text classes included. The masked reductions' statistics (`ma.mean` / `ma.var`): `reductions.md`.

## Semantics, by result kind

- Inheritance of the fill value and the hard mask: the `_arraymethod` views inherit, `take` does not, the `*_like`
  functions copy the mask, `_fromnxfunction` results always carry a mask array, `concatenate` shrinks an all-False mask.
  `ma.array` is soft by default, `masked_array` inherits the hardness, and a caller's bool mask is ADOPTED
  (`getmask(x) is m`).
- `getdata`, `filled`, `compressed` and `compress_nd` return views; `shrink_mask` works in place. `.data` / `.mask` return
  a NEW view per access, as NumPy's do: read the `_data` / `_mask` FIELDS for identity or disposal (an adapter that
  disposed `x.data` disposed a fresh view instead).
- `argsort`'s default axis flattens from 2-D; `alltrue` / `sometrue` are logical reductions along axis 0; `count` without
  a mask is a writeable full array; `flatnotmasked_edges` follows NumPy's three cases. `trace` casts the diagonal first
  when there is a mask; `minimum` / `maximum` take two operands (`where(less(a, b), a, b)`: a NaN does not propagate);
  `allclose` runs statement for statement; `average` raises the 0-d `AttributeError`; `setdiff1d` keeps the first
  operand's dtype; `unique` keeps the first masked element's data.
- The masked ufuncs take the first masked operand's fill value (RAW) and hard mask; `ma.power` takes its base's only and
  fills invalid slots with it; the shifts fill masked slots with 0 and inherit nothing. `-x`, `~x`, `%`, `&`, `|` and `^`
  run ndarray's ufunc and `__array_wrap__` (no fill-back, `%`'s domain filled with 1, a weak C# number); the ordering
  operators run `_comparison` (no fill-back, a shrunk COPY of the mask, the fill value cast to bool, `getdata` making a C#
  number strong). `copy`, `astype`, `view`, `diagonal`, `clip`, `round`, `sort`, `anom` and `diff` inherit; `real`, `imag`
  and `mT` reset. Masks are shared or copied as NumPy shares or copies them (a binary ufunc that built its mask on an
  operand's own mask object masked the operand when the result was masked).
- `ma.sort` of a hard-masked array writes back through the hard `__setitem__`; a float masked slot sorts by NaN. `ma.dot`
  always carries a mask array (an empty contraction masks every slot); `ma.diff` broadcasts a 0-d prepend;
  `masked_where(copy=False)` masks the input in place. `byteswap` / `argpartition` / `ma.cumsum` / `ma.cumprod` give the
  result a COPY of the source's mask (and `byteswap` / `argpartition` keep the hardness). `view(dtype)` of an array with a
  mask and another itemsize raises NumPy's `cannot reshape array of size 4 into shape (16,)` (the house
  `IncorrectShapeException`, NumPy's ValueError). `ma.apply_along_axis` stores the default fill at its result's dtype
  (NumPy's closing `result.fill_value = ma.default_fill_value(result)`: an int8 / uint8 / int16 / uint16 result raises
  `Cannot convert fill_value 999999 to dtype uint8`). `x + 1` on an int8 / uint8 / int16 / uint16 / bool masked array is
  numpy.ma's strong int64 (the literal reaches `operator +(NDMaskedArray, NDArray)` as a marked int32 0-d array, which
  `BareScalarStrongDtype` reads as the Python int). The masked print renders float16 / float32 elements as the Python
  floats `astype(object)` makes of them (`1.732421875`, not `1.732`).
- The indexer parses a lone slice string once for the data and the mask (`GetView` rewrites its slice ARRAY, so the mask
  indexes a copy of it).
- Open: a 0-d result NumPy returns as a plain scalar stays a 0-d masked array. `NDArray.view()` of a DISPOSED array
  returns a live view over the freed buffer (`InitializeArc` ignores `TryAddRef`'s failure).
- Gate: `Ma/NDMaskedArrayInheritanceTests.cs` (probed member by member against NumPy).

## The fill value

Stored as NumPy stores it, for every dtype (`Ma/NDMaskedArray.cs`; `_fill_value` is an NDArray, null = NumPy's None):
- Conversion (`MaskedArrayModule.CheckFillValue`, `_check_fill_value`): a given fill becomes a 0-d array of the DATA's
  dtype through `NDArray.AsStoreValue` (`PyArray_FromAny`): a C# number is a Python scalar, packed with the range checks
  (`5.7` on int8 is 5; `300` raises `TypeError: Cannot convert fill_value 300 to dtype int8`, the OverflowError /
  ValueError rewrapped, the value in Python's `%s`); a complex into a real dtype raises Python's own `int()` / `float()`
  TypeError; a `Half` / `char` / `decimal` (NumPy scalars) or any array is cast (`np.float16(300)` on int8 is 44).
- The default is materialized on first read (`DefaultFill`, `_check_fill_value(None)`) at the DEFAULT's own dtype: int64
  999999 for every signed integer dtype, uint64 for unsigned (char too), float64 1e20 for every float (decimal too),
  complex128, bool. So `fill_value` of int8 data is a `long`, the repr prints `fill_value=np.int64(999999)`, and `filled()`
  casts it under `same_kind` (63 in int8). The repr reads the getter, so printing materializes too.
  `ma.default_fill_value(obj)` takes an array, a scalar or a list as well as a dtype (NumPy's `_get_dtype_of`; None is
  `'?'`).
- The setter writes INTO a stored array (`_fill_value[()] = target`, unsafe): `fill_value = 5` on a fresh int8 array reads
  back `(sbyte)5`, after a read of the default `5L`; `fill_value = null` after `fill_value = 5` writes 999999 into the int8
  fill (63). An array with dimensions into a stored 0-d fill is NumPy's `ValueError: setting an array element with a
  sequence.` (complex: `only 0-dimensional arrays can be converted to Python scalars`).
- Inheritance: `_update_from` (the `_arraymethod` family, views, the masked ufuncs, the operators) SHARES the stored
  object, so once a fill was set or read, a set on either array reaches both (`x.copy()` included);
  `__array_finalize__` (constructors, `astype`, `view`, `byteswap`, `argpartition`, `masked_where`, the `*_like`
  functions) stores a converted COPY (`astype(int64)` of a float64 array whose default 1e20 was READ gives
  `long.MinValue`, MSVC's out-of-range cast; of an unread one 999999). The masked singleton refuses both getter and setter
  (`AttributeError: attributes of masked are not writeable`).
- Lifetime: a NEW array's fill (`StoreFill`) is tracked by the ambient scope like its data and leaves with it (`Yield`);
  a fill stored into an EXISTING array (the getter's default, the setter) is detached (`Own`).
- Divergence: `filled()` whose stored fill cannot reach the data under `same_kind` (a fill of another kind shared with a
  derived array: `ma.absolute(complex_with_fill).filled()`) is NumPy's OBJECT-array fallback; NumSharp raises
  `NotSupportedException` (no object dtype).
- Gates: the zero-excuse, type-strict oracle tier `ma_fill.jsonl` (call-history scripts: `gen_ma_oracle.py fill`) and
  `Ma/np.ma.fill_value.NumPyPort.Test.cs`.

## The set routines

numpy.ma's arraysetops are NumPy's statements (`Ma/NDMaskedArray.SetOps.cs`): `unique` (with NumPy's `return_index` /
`return_inverse`, a `MaskedArrayModule.UniqueResult` tuple), `in1d`, `isin`, `intersect1d`, `setxor1d`, `union1d`,
`setdiff1d` replay `numpy/ma/extras.py` over `_unique1d`, `ma.concatenate` (with `axis=None`), the masked sort / argsort
(masked slots keyed by NaN for a floating dtype, the dtype maximum otherwise) and the masked `==` / `!=` (two masked slots
equal, a masked slot unequal to every value, every comparison that reads a masked slot masked).
- So a flag whose sorted neighbour is masked is masked (`ma.in1d([3, 5], ma.array([3, 9], mask=[0, 1]))` is
  `[True, --]`), NaNs collapse only behind an UNMASKED last NaN, a masked bool slot sorts among the unmasked Trues, a hard
  mask ORs into `ma.unique`'s in-place sort, and setxor1d compares FILLED values (999999 equals a masked int slot).
- One inherent divergence: NumPy's quicksort is unstable (x86-simd-sort networks for 32/64-bit dtypes, introsort past 16
  elements) and NumSharp's sort is stable, so the two differ only where an unstable sort meets equal keys that are not
  interchangeable (a masked key tied with an equal unmasked value, ±0, NaN payloads); the generator replays a stable copy
  of NumPy's statements (`test/oracle/ma_setops_reference.py`) and drops exactly those cases.
- Speed: NumSharp's int64 masked argsort sorts the unmasked values alone when no unmasked value equals the fill key, and
  in1d's `order.argsort()` of a permutation is its inverse permutation, one scatter.
- Gates: the zero-excuse oracle tier `ma_setops.jsonl` (`gen_ma_oracle.py setops`) and `Ma/np.ma.setops.NumPyPort.Test.cs`.

## `ma.put`

NumPy's `MaskedArray.put` / `ma.put`, statement for statement (`Ma/NDMaskedArray.cs`).
- Under a HARD mask (a mask array present, even an all-False one) NumPy runs `mask = self._mask[indices]; values =
  narray(values, subok=True); values.resize(indices.shape); indices = indices[~mask]; values = values[~mask]` before
  `self._data.put`, and so does NumSharp, with every consequence: the mask lookup indexes the FIRST axis of an N-D mask, so
  every hard-masked N-D put raises IndexError before it writes (`too many indices for array: array is 1-dimensional, but
  2 were indexed`, or the lookup's own out-of-bounds text); short values are ZERO-PADDED, not cycled (`put([0, 2, 3], 5)`
  writes 5, 0, 0; a C# number or array is a fresh `np.array`, which resizes); masked values (the masked constant included)
  raise `A masked array does not own its data and therefore cannot be resized.`; an `NDArray` values of another size
  raises `ndarray.resize`'s errors (the caller holds it, so NumPy's reference check refuses; single-segment and OWNDATA
  first: the house `IncorrectShapeException`); the lookup ignores `mode`, the write does not (under `clip` a negative index
  writes, and unmasks, index 0); a literal is converted before the cast, so it wraps where the soft branch raises the
  OverflowError.
- The soft branch writes a masked values' DATA (the masked constant's 0.0, cast), a plain target goes through
  `ndarray.put`, and the mask is written in place and shrunk to nomask when all-False.
- One divergence (`[Misaligned]`): for an `NDArray` values of the indices' size but another shape NumPy reshapes the
  caller's array in place; NumSharp relabels an alias.
- Gates: the zero-excuse, type-strict oracle tier `ma_put.jsonl` (`gen_ma_oracle.py put`) and
  `Ma/np.ma.put.NumPyPort.Test.cs`.

## Text and `StringDType`

- Over fixed-width text, where the ufunc behind a member has no string loop, the member refuses with NumPy's text: the
  masked reductions, products, extrema, `clip`, `allclose`, `median`, `trace` and `masked_values` (`add.reduce`'s
  resolved-dtypes TypeError, the comparison no-loop texts that name `StrDType` / `Float64DType` classes, `isinf`'s
  safe-casting text, the str / `_PyFloatDType` DTypePromotionError, `data type must provide an itemsize`); where the member
  reads the text (a number formatted into a filled slot, truth values, a sort with masked text) it computes. The text fill
  value is MATERIALIZED at its own width (`'N/A'`, `<U3` / `|S3`) when it is first read; a later assignment writes INTO
  that stored array (`'hello'` → `'hel'`), and only `__array_finalize__` converts it to a new dtype (`<U1` → `'N'`).
- Over `StringDType` the fill value is `'N/A'` (`default_filler['T']`, stored as a `<U3`, read back as a string);
  movement, printing (an NA element prints as its NA object), fills, masks, casts, comparisons, `add`, the logical
  ufuncs and the sums compute, and `sort` / `argsort` / `median` / `unique` compute while no masked slot needs a fill
  value. Every member that orders or extremizes through numpy.ma's fill tables (`min` / `max` / `ptp` / `argmin` /
  `argmax`, `sort` and `median` with a mask, `unique` with a masked slot, the set routines) raises `Unsuitable type
  StringDType() for calculating minimum.` BEFORE it reads the mask: the fill tables are keyed by the scalar type, and a
  `StringDType`'s is Python's `str`. The
  ufunc members raise the dispatcher's refusal, which names DType CLASSES (`NDStringUfuncs.VString.cs`: `(<class
  'numpy.dtypes.StringDType'>, <class 'numpy.dtypes._PyFloatDType'>) -> None`; a C# number literal is NEP 50's abstract
  class through its literal mark).
- The set routines reach NumPy's refusal WITHOUT the uniques before it (`VStringSetOpDryRun`, stand-ins with the
  operand's dtype, masked-ness and emptiness). The legacy products (`dot`, `inner`, `convolve`, `correlate`): a
  `StringDType` first operand (or a `U` one beside it) is the legacy-only refusal (`This function currently only supports
  native NumPy dtypes ...`); beside any other class NumPy multiplies Python objects and its FIRST product decides; a
  product that does not raise is an object array, a `NotSupportedException` (`VStringProductAnswer` /
  `VStringSlidingAnswer`). `np.unique` over `T` answers at most one element and refuses an axis over two or more
  dimensions with NumPy's two texts; `np.vander` raises `promote_types`'s `DTypePromotionError`.
- Stores: a HARD mask's single-element store raises `'str' object has no attribute 'size'` (the element is a Python
  object); a C# number stored into a masked array is a Python number (literal mark): `coerce=False` refuses it and an
  integer out of range raises `OverflowError`, as in NumPy.
- Gates: string-oracle sections `strma` and `tma`; `Backends/Strings/StringMaskedArrayTests.cs`,
  `Ma/NDMaskedArrayNumPyStatementsTests.cs`, `Ma/np.ma.StringDType.Test.cs`.
