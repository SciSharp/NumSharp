---
paths:
  - "src/NumSharp.Core/Selection/**"
  - "src/NumSharp.Core/Indexing/**"
  - "src/NumSharp.Core/Creation/np.tri.cs"
  - "src/NumSharp.Core/Backends/NDArray.{PythonLiteral,ShapeBase,TwoDimBase}.cs"
  - "src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.{Mapping,ItemSelection}.cs"
  - "src/NumSharp.Core/Backends/Default/Indexing/Default.{RavelMultiIndex,UnravelIndex,BooleanMask}.cs"
  - "src/NumSharp.Core/Backends/Kernels/Direct/*{Take,Put,Place,PutMask,Select,GatherFlat,DiagWrite,TakeAlongAxis,PutAlongAxis,TakeLeading}*.cs"
  - "test/NumSharp.Tests/{Indexing,Selection}/**"
---

# Selection and indexing

## Index forms

- ONE indexer per array class, NumPy's `__getitem__(key)` / `__setitem__(key, value)`: `NDArray` has only
  `public virtual NDArray this[params object[] key]`, one `__getitem__(object key)` and one `__setitem__(object key,
  object value)` (an `object[]` / `Slice[]` / C# tuple key is the index tuple; a typed `int[]` / `bool[]` key is ONE
  entry, NumPy's list: an index array or a mask). Gate `Selection/NDArray.Indexer.OneSignature.Test.cs`.
  - Integer and slice-string indexing (`nd[0]`, `nd["1:3"]`, `nd["::-1"]`); boolean masks (`nd[mask]`; `nd[mask] = v` is
    `TensorEngine.BooleanMaskSet`, NumPy's `array_assign_boolean_subscript`, below); fancy indexing (`nd[indices]`).
  - A raw `int[]`/`long[]` as the ONLY index is fancy (NumPy parity): `nd[new int[]{0,2}]` selects rows 0 and 2 (shape
    `(2,…)`), not the element at `(0,2)`. `nd[0,2]` (separate scalar ints) is a coordinate. Coordinate access by array is
    `nd.GetData(int[]/long[])` (`Backends/NDArray.cs`), and internal coordinate callers must use it.
  - The dispatch: a `Slice[]` passed whole, ONE string (slice notation, comma-separated), or entries that are each a
    `Slice`, a string (one slice) or an integer of a type C# converts to `int`/`long` (`sbyte` … `uint`, `long`, `char`,
    `nint`) take the view body (`TryBasicIndex`: `GetView` + `AssignThroughView`); a lone `NDArray<bool>` takes the mask
    body; everything else the general one. Trap: the general path has no `case long` / `case short` for a single entry,
    so without that pre-dispatch `a[5L]` would raise "only integers…"; a `ulong` keeps the general path (no implicit
    conversion to `long`).
  - `NDArray<T>` overrides the GETTER only, returning `NDArray<T>` (C# lets an override narrow its type only without a
    setter, CS1715); the setter is `NDArray`'s, so through an `NDArray<T>` variable the value must be an `NDArray<T>` or a
    `T` (`implicit operator NDArray<T>(T)`: `typed[0] = 3.14`, `mask[m] = false`); assign an untyped value through an
    `NDArray` reference or `__setitem__`. There is no scalar indexer: an element is a 0-d view, read with a cast
    (`(double)typed[0]`) or, in a hot loop, `GetValue<T>(…)` (the view costs about 0.4 µs). Virtual dispatch reaches the
    override through an `NDArray` reference too, so a typed array's selection is an `NDArray<T>` at run time. Trap: an
    advanced result is wrapped over its OWN storage object (`new NDArray<T>(result.Storage)`, then the untyped wrapper
    disposed), never `MakeGeneric`: an alias reads as a view, so the gather would lose `flags.owndata`.
  - `NDCharArray` and `NDMatrix` HIDE the indexer (`new`), they do not override it: their rules (a stripped element; a
    matrix row or column) apply through their static type, while library code that receives such an object as `NDArray`
    indexes it as a plain array (`NDMatrixBoundary` hands the matrix object itself to computations that run with
    plain-array semantics: `np.piecewise`'s `x[cond]` would otherwise gather a `(1, K)` matrix). `NDMatrix` also keeps
    `NDArray this[long row, long column]`. `NDListArray` (not an `NDArray`) has one `NDListArray this[params object[]
    key]`: a full integer index is the list's 0-d stand-in, an index array or mask gathers the ids (the lists are shared,
    as NumPy's object pointers), the pieces come from `item()` / `tolist()`.
- An index is read as NumPy's `prepare_index` reads it (`Selection/NDArray.Indexing.PrepareIndex.cs`), and a store
  follows `array_assign_subscript` (gates `Selection/*.NumPyPort.Test.cs`):
  - Entries: a double / float16 / decimal is refused; a C# bool is NumPy's 0-d bool array (alone the 0-d mask, beside
    other entries an advanced entry); a 0-d integer array with no index array beside it is a VIEW index
    (`HAS_SCALAR_ARRAY`: the getter copies the view in K order, the setter assigns through it); `Slice.Index(n)` and the
    notation `"n"` beside an advanced entry are NumPy's integer (`NormalizeIndexInputs` turns them into longs).
  - Errors in NumPy's order: the boolean indices' shapes (a mask with more dimensions than the array is `too many indices
    for array: array is 1-dimensional, but 2 were indexed`, checked first; then the first mask axis of non-zero length
    that differs, `boolean index did not match indexed array along axis 1; size of axis is 3 but size of corresponding
    boolean axis is 4`; a zero-length mask axis matches any length), an index array neither integer nor bool (empty and
    0-d included: `arrays used as indices must be of integer (or boolean) type`, before the too-many check), the
    integers' bounds, the index arrays' broadcast (`... with shapes (2,) (3,) `, a space after every shape), then the index
    arrays' values.
  - Placement: the advanced block goes in FRONT when a slice, a new axis or the ellipsis separates two advanced entries,
    an ellipsis that spans no axis included (`AdvancedEntriesAreConsecutive` reads the tuple as given, not the
    ellipsis-expanded list).
  - Stores: the index is validated, the value converted with the array's own descriptor (a list item by item through its
    setitem; a list into a view limited to the view's rank; an array into a view is `PyArray_AssignArray`, leading unit
    dimensions stripped), then the index arrays' values are checked. The caller's key array is never written
    (`NormalizeIndexInputs` copies it at the first conversion).
  - The typed fetch and store are `UnmanagedStorage.MapIterGet` / `MapIterSet`
    (`Backends/Unmanaged/UnmanagedStorage.Mapping.cs`); NDArray's protected `FetchIndices` / `SetIndices` / `...ND` /
    `...NDNonLinear` forward to them. The boolean-mask getter's result is not zero-filled: both of its gathers write every
    element, so `DefaultEngine.BooleanMask` allocates with `NDArray.ForOverwrite`.
- `NDArray.PrepareAssignedValue` (`Selection/NDArray.Indexing.AssignedValue.cs`) is NumPy's MapIter value rule for every
  index-array assignment: a value with more axes than the selection loses its leading axes (a `(1, N)` matrix row fits an
  `(N,)` selection), and a value that does not fit raises `shape mismatch: value array of shape ... could not be
  broadcast to indexing result of shape ...` before anything is written. A boolean mask of the array's own shape follows
  `array_assign_boolean_subscript` (a value with two or more axes is TypeError; a wrong length is `NumPy boolean array
  indexing assignment cannot assign 3 input values to the 6 output values where the mask is true`).
- The fancy-index operator runs the take/put kernels: `a[idx]`, `a[idx] = v`, `m[ridx]`, `m[ridx] = row` (ONE integer
  index array over a C-contiguous source of any rank) go through `Selection/FancyIndexKernelRoute.cs`: a flat
  gather/scatter (`DirectILKernelGenerator.GatherFlat.cs`: a compile-time element width, running cursors, NumPy's
  `check_and_adjust_index` as one add and one unsigned compare) for primitive-width slabs, and the general take/put
  kernels (`cpblk`) for whole-sub-array slabs. The setter pre-validates every index with a `Vector<T>` bounds scan before
  its first store (NumPy's `mapiter_trivial_set`), so a bad index leaves the array untouched. The IndexError text, the two
  value-broadcast ValueError texts, last-write-wins on duplicates, and view offsets on source and index array
  (`Storage.Address + Shape.offset × itemsize` serves contiguous and strided slices) are pinned by `Indexing.KernelRoute.Tests`.
  Two index arrays, a slice plus an index array, F-order source rows and a strided index view run the slower delegate
  route.
- int32 indices are read in place (`idx32` kernel variants; `np.array(new int[]{…})` is int32), and `np.take`/`np.put`
  read them in place too. NumSharp indexes with int64 everywhere; the int32 load width exists because widening an int32
  index array into an int64 temp costs more than reading it in place, while an int32 index costs the same as an int64 one.
  All index arithmetic stays int64 (the int32 is sign-extended on load). Drop the variants if that stops being true.
- Fancy indexing does not follow NumPy's result-layout rules (`a[:, idx]` is C-ordered and owning, where NumPy returns
  its transposed MapIter view), and an N-D index array that is not C-contiguous beside a slice / ellipsis / integer is
  read in memory order: pinned in `OpenBugs.FancyIndex.cs`.
- A subshaped fancy SET (fewer axes than ndim: `arr[[0,2]] = v` on a 2-D+ array writes a whole trailing sub-array per
  index) into a non-contiguous destination (transposed, F-contiguous, row/column-strided, negative-stride) goes through
  `SetIndicesNDNonLinear<T>`, an odometer over the trailing `subShape` axes that advances the destination offset through
  the view's strides (from the C-contiguous value buffer broadcast to `retShape`); a C-contiguous destination takes the
  block-copy `SetIndicesND<T>`. Gates: the index oracle's `order='K'` setter cases (`gen_index_oracle` copies view bases
  with `order='K'`, so the SET runs into a non-contiguous destination) and `AuditV2` `T1_15a`/`T1_15b`.

## Storing a C# literal

Every store entry point packs a C# literal as NumPy's `PyArray_Pack` of a Python scalar (gates section S of the
`weak_scalar` tier, zero excuses, and `Indexing/np.store.PythonLiteral.Test.cs`).
- The marks. An `NDArray` parameter receives the literal `1000` and an explicit `np.array(1000)` as the same 0-d array,
  and NumPy stores them differently (`a[0] = 1000` on int8 raises `OverflowError: Python integer 1000 out of bounds for
  int8`; `a[0] = np.array(1000)` casts to -24). So the implicit conversion operators MARK the 0-d array
  (`Backends/NDArray.PythonLiteral.cs`): an integer primitive / `float` / `double` / `Complex` is `WasPythonLiteral`
  (NumPy's own `NPY_ARRAY_WAS_PYTHON_LITERAL` flag), a `Half` / `char` is `WasNumPyScalar` (`np.float16` / `np.uint16`);
  `bool` and `decimal` are unmarked. The mark belongs to that one object (a view, copy, reshape or result is unmarked);
  `NDArray.Scalar(x)` / `np.array(x)` build an unmarked, strong 0-d. ONLY the store paths read the marks; the ufuncs and
  comparisons read every 0-d operand as weak (np.clip's rule), because library code builds its scalar operands with
  `NDArray.Scalar`.
- The pack. `NDArray.PackForStore(value, dtype, pyArrayPackRoute)` packs a marked value with `ReductionInitial.TryPack`
  (the complete `PyArray_Pack` port: a Python int range-checked with the LP64 texts, a float truncated then range-checked,
  NaN / inf / complex refused with Python's texts, an int past 2^53 rounded TWICE into float16 / float32, a bool dtype
  takes the truth). A NumPy scalar follows NumPy's route split: the `PyArray_Pack` routes (an integer index, a slice or
  ellipsis view, `fill`, `flat[i]`, `np.pad`) int()-convert it with the range check into a SIGNED integer
  (`a[0] = (Half)300` on int8 raises); the `PyArray_FromAny` routes (fancy, mask, `np.put`, `putmask`, `place`,
  `put_along_axis`, `insert`, `full`, `asarray`, flat slices) CAST it (44).
- The entry points: the indexer setters (`UnmanagedStorage.SetData` after an integer index's bounds check; the fancy core
  after the too-many-indices check but BEFORE the index bounds, as NumPy; the mask setter after the mask's shape check;
  an empty selection packs too), `fill` (`CoerceFillValue`, one `TryPack`), `np.full` / `full_like` (NumPy's
  `copyto(casting='unsafe')`: a Python int range-checked, a float or complex cast unsafely,
  `UnmanagedStorage.TryCopytoUnsafeScalarElement`), `np.put`, `putmask`, `place`, `put_along_axis`, `insert`,
  `fill_diagonal` (it packs before its early returns, so an empty array still refuses), the flat iterator (`flat[i]`
  reports any pack failure as NumPy's `ValueError: Error setting single item of array.`) and `np.asarray(literal, dtype)`.
  `np.pad` stores one or two constants as NumPy scalars (`ReductionInitial.PackNumPyScalar`: `_as_pairs` runs
  `np.array(constant_values)`, so int8 refuses 1000 where uint8 pads with 232) and a larger table as the Python scalars
  of its `tolist()` (`TryPack`: `-1` into uint8 raises).
- `None` is a value in every store route: a C# `null` where NumPy passes `None` stores NumPy's setitem of `None`: float
  NaN as NumPy's `NPY_NAN` (the POSITIVE quiet NaN, `0x7ff8…` / `0x7fc00000` / `0x7e00`; .NET's `double.NaN` is
  negative), complex `nan+nanj`, bool `False`, `'None'` truncated to an `S` / `U` width, a `StringDType`'s NA object or
  `'None'`; every integer dtype and `char` raise `int(None)`'s TypeError, `decimal` Python's `decimal.Decimal(None)`
  TypeError. ONE primitive: `ReductionInitial.PackNone` / `PackStore` / `BoxNone` (store paths never call `TryPack` with a
  null: there null means a reduction's `initial=None`) and `NDArray.NoneForStore(descr)` (a 0-d value of any
  descriptor). Each route converts at NumPy's point, so the errors keep NumPy's ORDER: the indexers after an integer
  index's bounds but before a fancy index's, and for an empty slice; `fill` even when empty; `np.put` / `putmask` /
  `place` after the target, index-cast and mask-size checks; `np.insert` after a scalar position's check, and for an
  EMPTY position list too; the flat iterator's slice / list setters convert the value BEFORE resolving positions. Three
  routes see `None` as NumPy's 0-d OBJECT array, cast per element: `np.full` / `np.full_like` (an empty result converts
  nothing) and `np.copyto(dst, (NDArray)null)` (`'unsafe'` only, converted once up front for a destination of more than
  one element, not for `StringDType`, whose descriptor NumPy's `REFCHK` exempts). `np.full(shape, null)` /
  `np.asarray((NDArray)null)` / `np.asanyarray(null)` WITHOUT a dtype are NumPy's object array, refused
  (`NDArray.NoneWithoutDtype`); with one they build the 0-d value (`np.asanyarray(null, dtype)` binds the `MemoryView`
  overload, which therefore treats null as None). Divergences: `np.pad(constant_values: null)` is the omitted default (C#
  cannot tell an explicit null). Gates: the string oracle's `storenone` section, `Backends/NoneStoreTests.cs`.
- Kept as casts: `SetValue` / `itemset` (NumSharp's low-level typed setters; `itemset` is gone from NumPy 2.0), an
  unmarked 0-d array everywhere, and a 0-d view from integer indexing (`a[i] = b[j]`: NumPy's `b[j]` is a NumPy scalar,
  which a SIGNED integer destination would range-check: `[Misaligned]`). Text into a numeric store is parsed like NumPy's
  setitem of a Python `str` (`float64.fill("5.5")` is 5.5, `int32[0] = "hello"` raises `invalid literal for int() with
  base 10: 'hello'`).

## `take` / `put` / `place` / `putmask`

- `np.put` / `np.take` are NumPy's `PyArray_PutTo` / `PyArray_TakeFrom`, statement for statement
  (`Indexing/np.{put,take}.cs`, `Indexing/ClipModes.cs`; gates `Indexing/np.put.take.NumPyPort.Test.cs`, the `ma_put`
  tier and the aliasing tier's `put_selfvals` / `put_selfidx`, where the values or indices are a view of the target):
  - The mode is `ClipmodeConverter` (NumPy's `PyArray_ClipmodeConverter` + `clipmode_parser`), shared by `put`, `take`,
    `choose` and `ravel_multi_index`, and parsed FIRST: the first character selects, the exact lowercase spelling is
    required (`"Raise"` is `ValueError: Use one of 'clip', 'raise', or 'wrap' for clip mode`), an empty or unknown string
    is `ValueError: clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')`, `null` is NumPy's None (raise).
    `ravel_multi_index`'s `modes` is `ConvertClipmodeSequence` (`list of clipmodes has wrong length (1 instead of 2)`),
    converted before the coordinates are counted.
  - put's order: mode → writeability → the index cast (safe; an EMPTY float index array is the TypeError too) →
    IndexError `cannot replace elements of an empty array` → the value packing (a literal is range-checked even when
    nothing is written) → the no-op of empty values → the kernel (out of bounds: IndexError, the indices before it
    written into a contiguous target, as NumPy's loop writes). NumPy writes into a copy of the target when it may share
    memory with the converted values or indices (`arrays_overlap`): `np.put(a, [1, 2, 3], a[0:3])` is `[0, 0, 1, 2, 4]`.
  - take's order: AxisError (a 0-d source is raveled first, so only axis 0 / -1) → the index cast → the `out` checks
    (ValueError `output array does not match result of ndarray.take`, the safe cast, `WRITEBACKIFCOPY base is
    read-only`) → IndexError `cannot do a non-empty take from an empty axes.` only when the RESULT is non-empty → the
    gather, whose raise loop checks every index of every outer position even when the slabs are empty
    (`np.take(np.zeros((3, 0, 0)), [0], axis=1)` raises; an empty outer block checks nothing). Out of bounds is NumPy's
    IndexError.
  - C-order casts: put's values, the take / put index cast (`CastIndicesToInt64`), place's values and mask and putmask's
    mask are converted with `order: 'C'`, because the kernels read them linearly (a `'K'` cast of an F-order operand
    would be read in memory order); unravel_index's index cast is laid out in NumPy's iteration order with its result
    (below).

- Negative indices under `mode='raise'` follow NumPy's `check_and_adjust_index`: normalized once (`idx += n`) before the
  bounds test, so `take(a, [-1])` / `put(a, -1, v)` address the last element, and an index still out of range raises with
  the original value (`SelectionTests` `Take/Put_Raise_NegativeIndices_Normalize`; the `groupa` tier's negative,
  `wrap` and `clip` cases).
- Kernels (`DirectILKernelGenerator.{Take,Put,Place}.cs`): the per-element copy is a typed `Ldind`/`Stind` move for the
  dtype width (`CopyKindFor`/`EmitElementCopy`; two 8-byte moves for 16-byte Complex/Decimal), keyed into the kernel
  cache; `cpblk` only for multi-element `take` axis slabs. The `take` gather prefetches (`Sse.Prefetch0` of the slab
  `PrefetchDistance` = 32 indices ahead), gated at emit time on SSE and at the call site on a gathered footprint above
  `IndexPrefetchThresholdBytes` = 2 MiB (below it, prefetching is overhead), with a separate `(copyKind, prefetch)` kernel
  cached. `put`/`place` keep a wrapping values cursor instead of an `i % valuesCount` division per element.
- Index dtype validation: `CastIndicesToInt64` rejects a non-castable index dtype with NumPy's `TypeError("Cannot cast
  array data from dtype('…') to dtype('int64') according to the rule '…'")`, with NumPy's split: `take` converts under
  `same_kind` (a `uint64` index is allowed), `put` under `safe` (`uint64` rejected); each names its own rule. Integer and
  bool indices pass, float and complex are refused (`SelectionTests` `Take_FloatIndices_Throws_SameKind`,
  `Put_FloatIndices_Throws_Safe`, `Take_UInt64Indices_Allowed_But_Put_Rejects`).
- `np.putmask(a, mask, values)` (NumPy's `PyArray_PutMask` / `npy_fastputmask`; `Indexing/np.putmask.cs` over
  `UnmanagedStorage.PutMaskCore`, kernel `DirectILKernelGenerator.PutMask.cs`) writes `a.flat[i] = values.flat[i %
  values.size]` wherever `mask` is True, walking both in C order. It shares `place`'s structure (the writeable check
  first, the same-`size` mask contract, a non-bool mask cast to `!= 0` with NaN/inf → True, the `ascontiguousarray` +
  `copyto` writeback for non-contiguous targets, and the `arrays_overlap` → ENSURECOPY guard:
  `NDMemOverlap.SolveMayShareMemory(maxWork: 0)`, NumPy's `NPY_MAY_SHARE_BOUNDS`, gives a fresh copy of `a` when
  `values`/`mask` may alias it). It differs from `place` in two ways: the values cursor advances by POSITION (every
  element, in lockstep with `i`, wrapping at `nv`), not per True, and an empty `values` is a silent no-op (`place`
  raises ValueError). The kernel has a cursor-free `nv == 1` broadcast path (NumPy's `if (nv == 1)`). Errors: read-only
  → `"putmask: output array is read-only"` (first, before the mask size and nv), a wrong size → `"putmask: mask and data
  must be the same size"`. `values` is converted as NumPy's `PyArray_FromAny(values, a.dtype)` converts it, after the
  mask's size check (`PutMaskCore`, shared with place's `PlaceCore`): an array of values must cast under the `'safe'`
  rule (NumPy's TypeError text), a 0-d array (a NumPy scalar) converts without it, a C# literal is packed with Python's
  range checks, and a C# array is NumPy's Python LIST, discovered into `a`'s dtype item by item (`ListValuesLikeNumPy`:
  `[1000, 2]` into int8 raises `OverflowError`, `[1, 2.5]` into `<U3` stores `['1', '2.5']`, a NaN into an integer array
  raises `int()`'s `ValueError`). A typed array is cast in one pass only where the cast stores exactly those values (a
  widening; numbers into bool; doubles into a float or complex class; numbers into an integer class when the values' min
  and max, taken by np.amin / np.amax's IL kernels on the buffer, lie in the class). Gates: `Indexing/SelectionTests.cs`
  (`PutMask_*`) and the `putmask` tier.

## `take_along_axis` / `put_along_axis` / `select`

- `np.take_along_axis(arr, indices, axis=-1)` (`numpy/lib/_shape_base_impl.py`; its statements are
  `NDArray.take_along_axis` in `Backends/NDArray.ShapeBase.cs`) is the per-slice gather that inverts
  `argsort`/`argmax(keepdims=True)` (`take_along_axis(a, a.argsort(axis), axis)` ≡ `sort(a, axis)`). One whole-array
  strided-odometer IL gather (`DirectILKernelGenerator.TakeAlongAxis.cs`: an outer odometer around a tight loop over the
  innermost axis, dtype-agnostic through a byte-width-keyed copy). `indices` must be an INTEGER array
  (`IndexError "\`indices\` must be an integer array"` for bool/float/complex) with `arr`'s ndim (`ValueError "\`indices\`
  and \`arr\` must have the same number of dimensions"`); the axis dimension of `indices` supplies `J` (need not equal
  `arr.shape[axis]`), and every other dimension broadcasts both (a stride-0 source view is read fine; a non-axis conflict
  raises NumPy's fancy-index `IndexError "shape mismatch: indexing arrays could not be broadcast together with shapes …"`).
  `axis=None` flattens `arr` in logical C order and needs 1-D `indices` (`ValueError "when axis=None, \`indices\` must
  have a single dimension."`). A negative index normalizes once (`idx += M`); anything still out of range raises
  `IndexError "index {orig} is out of bounds for axis {axis} with size {M}"` with the original value. The result is a
  fresh writeable C-contiguous copy of `arr`'s dtype; any source layout is read through its own strides.
  `[Misaligned]`: when a non-contiguous `indices` holds several out-of-range values, NumSharp reports the first in the
  RESULT's C order while NumPy reports whichever its `PyArray_MapIterCheckIndices` visits first (only the offending value
  differs; the type, axis and size match, and `argsort`/`argmax` output is always exact). There is no SIMD gather: a
  hardware `VPGATHERQD/QQ` gather is barely faster than the scalar loop on AVX2 before the bounds validation a raise-mode
  gather still needs. Gates: `Indexing/TakeAlongAxisTests.cs` and the `groupa` tier.
- `np.put_along_axis(arr, indices, values, axis)` is the setter twin (`NDArray.put_along_axis`), in place (returns
  `void`); `axis` is required.
  A whole-array strided odometer (`DirectILKernelGenerator.PutAlongAxis.cs`) sharing take's iteration shape and
  non-axis broadcast, plus three setter rules: (1) `values` is BROADCAST, not cycled, to the indexing result shape
  (right-aligned, extra leading size-1 dims stripped); a mismatch raises `shape mismatch: value array of shape … could
  not be broadcast to indexing result of shape …`; it is cast to `arr`'s dtype (floats truncate toward zero, overflow
  wraps); where several positions collapse onto one element, the last write in C order wins. (2) Atomicity: every index
  is validated against the axis bound before the first store (NumPy's `PyArray_MapIterCheckIndices`), so an out-of-bounds
  index leaves `arr` untouched: two kernels, `PutAlongAxisValidate` then a bounds-check-free `PutAlongAxisScatter`.
  (3) `axis=None` treats `arr` as `np.array(arr.flat)`: a C-contiguous `arr` is written back through the flat view, a
  non-contiguous one raises read-only (NumPy's flat copy is read-only there; the dtype check fires first). A
  COPY_IF_OVERLAP guard snapshots `values` when it may alias `arr` (`put_along_axis(a, reversing_idx, a)` reverses
  through a copy). Validation order: axis → dtype → ndim → writeable → non-axis broadcast → value broadcast → per-index
  bounds. The take divergence above applies (the offending value follows C order). Gates:
  `Indexing/PutAlongAxisTests.cs` and the `groupa` tier.
- `np.select(condlist, choicelist, default=0)` (`numpy/lib/_function_base_impl.py`) draws each element from the choice
  whose condition is true, the FIRST true condition winning, else `default`. C# shapes: `condlist` is `NDArray[]`,
  `choicelist` is `object[]` (an `NDArray[]` binds through array covariance; scalar choices need `new object[]{…}`),
  `default` is `object` (null is NumPy's `0`). Weak scalars, select's own rule: every C# integer primitive, `float`,
  `double` and `Complex` literal is weak (it adopts the other operands' dtype); `bool`, `char`, `Half`, `decimal`, arrays
  and every `NDArray` are strong (NumPy's `type(x) in (int,float,complex) else asarray(x)`), so a weak-int default that
  overflows the resolved dtype WRAPS (`int8` choice + `1000` default → -24); the result dtype is
  `result_type(*choices, default)` through `NDExprTypeRules.PromoteStrong`. The copies receive what NumPy's
  `np.broadcast_arrays` makes of a weak literal (`np.asarray` of the Python scalar: int64 / float64 / complex128, of the
  choice group's shape), so a C# `byte` beside a uint8 array is refused as NumPy's Python int is (`Cannot cast array data
  from dtype('int64') to dtype('uint8') according to the rule 'same_kind'`). Conditions and choices broadcast in two
  separate groups (NumPy's `np.broadcast_arrays`, `Shape.BroadcastShape`: a mismatch names the two operands of its
  group); a non-bool condition is `TypeError("invalid entry {i} in condlist: should be boolean ndarray")`; a length
  mismatch and an empty condlist are `ValueError`. `Indexing/np.select.cs` is the facade, NumPy's statements are
  `NDArray.select` (`Manipulation/NDArray.FunctionBase.cs`). Contiguous, no-cast, full-size array choices run one fused
  IL kernel (`DirectILKernelGenerator.Select.cs`:
  a reverse `Vector.ConditionalSelect` chain seeded with the default, choices overlaid n-1 → 0 so the first true
  condition wins, 4× unrolled with four accumulators, the scalar default hoisted with `CreateBroadcast`, `np.where`'s
  `EmitInlineMaskCreation` for the bool → lane expansion). Every other shape (broadcast, strided, cast, scalar choices)
  takes NumPy's structure: allocate the default-filled result, then `np.copyto(where=cond)` each choice in reverse.
  Gates: `Indexing/np.select.Test.cs`, `Indexing/np.select.NumPyOrder.Test.cs` and the `select` op of the `groupa` tier.

## `unravel_index` / `ravel_multi_index`

- `np.unravel_index` is NumPy's statements (`TensorEngine.UnravelIndex`,
  `Backends/Default/Indexing/Default.UnravelIndex.cs`): the order letter (both cases), the dims overflow, the index
  dtype's TypeError, then `only 'C' or 'F' order is permitted`; an out-of-range index is NumPy's ValueError, the FIRST
  one in the iteration order; `shape=()` is legal. The result is the columns of ONE interleaved `indices.shape +
  (len(dims),)` buffer (element stride `len(dims)`), laid out along the index array's iteration order
  (`NpyIter_CreateCompatibleStrides`, KEEPORDER with `DONT_NEGATE_STRIDES`, is `Shape.NpyIterOutputShape` over the index
  array: an F index array gives F-strided coordinates, a broadcast or reversed one C). Every coordinate array reports
  OWNDATA false, and a 0-d index array gives NumPy's scalars (fresh read-only 0-d arrays): code that reads their raw
  buffers must use the strides. Positive dimensions take a Granlund–Montgomery reciprocal IL kernel (multiply-high and
  shifts: RyuJIT emits TWO hardware divides for `val % m` and `val / m`); a zero or negative dimension keeps the divmod
  kernel.
- `np.ravel_multi_index` is `arr_ravel_multi_index` (`TensorEngine.RavelMultiIndex`,
  `Backends/Default/Indexing/Default.RavelMultiIndex.cs`, with NumPy's `astype_anyint`, which the unravel member shares
  and which asks the descriptor's kind): the result is its iterator's ALLOCATE output (KEEPORDER over the coordinate
  arrays: an F pair gives F, a mixed pair C), the checks run in NumPy's order (order letter, modes, ravel strides, count,
  dtypes, broadcast with the iterator's text), `dims=()` gives 0, a zero dimension raises `cannot unravel if shape has
  zero entries (is empty).` in every mode, and the errors are ValueError. Its `mode` is NumPy's mode object (a string for
  every axis, a sequence of strings, None: `ClipModes.ConvertClipmodeSequence(object, int)`), and a null coordinate array
  is NumPy's None (`TypeError: only int indices permitted`).
- Gates: the layout-parity oracle's `unravel` / `ravelmi` families, the string oracle (no layout excuse) and
  `Indexing/IndexSearchLayout.Test.cs`.

## The diagonal and triangular family

NumPy's `_twodim_base_impl.py` + `_index_tricks_impl.py`; tests `Indexing/np.{diag,tri,diag_indices}.Test.cs`, oracle
tiers `manip`/`matmul`; facades `Creation/np.tri.cs`, `Indexing/np.{diag,tril,diag_indices,tril_indices,fill_diagonal}.cs`.
The statements are the array object's: `NDArray.tri` / `tril` / `triu` / `tril_indices` / `triu_indices` and the _from
forms (`Backends/NDArray.TwoDimBase.cs`, with the write-once `NDArray.ZeroBytes`), `NDArray.diag_indices` /
`diag_indices_from` and `fill_diagonal` (`Indexing/NDArray.IndexTricks.cs`, sharing `RequireEqualDimensions`).
`np.mask_indices` keeps its statements: its `mask_func`'s result goes through np.not_equal and np.nonzero, whose class
rules apply to whatever class the callback returns.
- `diag(v, k=0)`: a 1-D input CONSTRUCTS a fresh writeable C-contiguous `(n, n)` matrix (`n = v.size + |k|`) with `v` on
  the k-th diagonal, scattered down the stride-`(n+1)` diagonal by the IL strided-store kernel
  (`DirectILKernelGenerator.DiagWrite`: a byte-width-keyed MOV loop for itemsize 1/2/4/8/16; the aliased-view + `SetData`
  route when dynamic codegen is unavailable, as under AOT). A 2-D input EXTRACTS through `diagonal(v, k)`: a read-only
  view sharing storage. Anything else raises `ValueError("Input must be 1- or 2-d.")`.
- `diagflat(v, k=0)` is `diag`'s 1-D branch for any rank: the input is raveled in logical C order first; 0-d → `(1,1)`.
  The statements of both and their diagonal scatter are `NDArray.diag` / `diagflat` (`Backends/NDArray.TwoDimBase.cs`,
  beside `NDArray.eye`); the `StringDType` route asks `HasReferences`.
- `tri(N, M=None, k=0, dtype=float)`: ones at and below the k-th diagonal, written as a prefix of one pre-built ones row
  per row; negative `N`/`M` give an empty axis.
- `tril(m, k=0)` / `triu(m, k=0)`: always a fresh writeable C-contiguous copy, the dtype kept, applied to the last two
  axes. A 1-D input yields a 2-D result (`tril([1,2,3])` → `[[1,0,0],[1,2,0],[1,2,3]]`), and a 0-d input raises NumPy's
  leaked `tri() missing 1 required positional argument: 'N'`. Rows are filled directly (copy the kept run, zero the rest);
  sources whose last axis is not unit-stride use `where(tri(...), m, 0)`.
- `fill_diagonal(a, val, wrap=False)`: in place, returns void. Values tile cyclically and truncate, they do not broadcast
  (`[1,2,3,4]` into a 6×6 gives `[1,2,3,4,1,2]`; an empty one is a no-op); `wrap` only affects tall matrices. Errors:
  `array must be at least 2-d`, `All dimensions of input must be of equal length`, `underlying array is read-only`. The
  targets resolve to real strides and are written by `DiagWrite` (one stride-`Σstrides` run, or `ceil(count/cols)`
  equally strided blocks when wrapping), so it writes correctly through transposed, sliced, F-order and negative-stride
  views. `val` is `object` (scalars, `NDArray`, C# arrays (a rectangular 2-D one ravels), `IEnumerable`), converted the
  way NumPy's flat assignment converts it, `PyArray_FromAny(val, a.dtype, FORCECAST)`, BEFORE the diagonal is walked,
  so an empty array still raises (`NDArray.FlatStoreValue`, `Indexing/NDArray.IndexTricks.cs`): a scalar or str by the
  dtype's setitem (a Python int range-checked, a str parsed by `int()` / `float()` / `complex()` and quoted as the str,
  its `str()` into `S` / `U`, a `BigInteger` included), an `object[]` or collection item by item the same way
  (`new object[] { 300, 2 }` into int8 raises), an `NDArray` or typed C# array by an unsafe cast. A null `val` is Python's
  `None`: NaN for a float array, `False` for bool, `'None'` for text, `int(None)`'s TypeError for an integer array,
  converted after the read-only check and before the empty-diagonal return, as NumPy's `a.flat[:end:step] = None` does.
- `diag_indices(n, ndim=2)` / `diag_indices_from(arr)` / `tril_indices(n, k=0, m=None)` / `triu_indices` /
  `tril_indices_from(arr, k=0)` / `triu_indices_from` / `mask_indices(n, mask_func, k=0)` return `NDArray<long>[]`,
  the first six OWNING arrays as NumPy's fresh ones are, `mask_indices` the views `np.nonzero` returns. `diag_indices`
  returns the same array instance in every slot (NumPy's `(idx,) * ndim` aliasing); `ndim <= 0` gives an
  empty array. `mask_indices`' arity follows the mask function's rank (`diag` yields one array). `tril_indices` /
  `triu_indices` emit the triangle's row/column runs directly in O(output), with no N×M bool mask.
- The triangular fills pick their allocation by size: below 64 MiB (`SizeBucketedBufferPool.WriteOnceMaxBytes`) a pooled
  buffer comes back dirty, so they allocate uninitialised and write every byte once; above it `np.zeros` rides free OS
  zero pages and only the kept run is written.
- `diag` / `diagflat` construction allocates a fresh `n×n` buffer and faults about n scattered pages on the sparse
  diagonal write; NumPy's benchmark loop reuses a warm block, which NumSharp's allocator does not cache.
