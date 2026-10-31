---
paths:
  - "src/NumSharp.Core/Selection/**"
  - "src/NumSharp.Core/Indexing/**"
  - "src/NumSharp.Core/Creation/np.{r_,c_,ogrid,mgrid,meshgrid,tri}.cs"
  - "src/NumSharp.Core/Backends/Kernels/Direct/*{Take,Put,Place,PutMask,Select,GatherFlat,DiagWrite,TakeAlongAxis,PutAlongAxis,TakeLeading}*.cs"
  - "test/NumSharp.Tests/{Indexing,Selection}/**"
---

# Selection and indexing

## Index forms

- Integer and slice-string indexing (`nd[0]`, `nd["1:3"]`, `nd["::-1"]`); boolean masks (`nd[mask]`; `nd[mask] = v` is
  `TensorEngine.BooleanMaskSet`, NumPy's `array_assign_boolean_subscript`, below); fancy indexing (`nd[indices]`).
- A raw `int[]`/`long[]` as the ONLY index is fancy (NumPy parity): `nd[new int[]{0,2}]` selects rows 0 and 2 (shape
  `(2,…)`), not the element at `(0,2)`. `nd[0,2]` (separate scalar ints) is a coordinate. Coordinate access by array is
  `nd.GetData(int[]/long[])` (`Backends/NDArray.cs`), and internal coordinate callers must use it. The typed
  `NDArray<T>.this[int[]]` returns a scalar `T` (always a coordinate).
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

## `take` / `put` / `place` / `putmask`

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
- `np.putmask(a, mask, values)` (NumPy's `PyArray_PutMask` / `npy_fastputmask`; `Indexing/np.putmask.cs`, kernel
  `DirectILKernelGenerator.PutMask.cs`) writes `a.flat[i] = values.flat[i % values.size]` wherever `mask` is True,
  walking both in C order. It shares `place`'s structure (the writeable check first, the same-`size` mask contract, a
  non-bool mask cast to `!= 0` with NaN/inf → True, the `ascontiguousarray` + `copyto` writeback for non-contiguous
  targets, and the `arrays_overlap` → ENSURECOPY guard: `NDMemOverlap.SolveMayShareMemory(maxWork: 0)`, NumPy's
  `NPY_MAY_SHARE_BOUNDS`, gives a fresh copy of `a` when `values`/`mask` may alias it). It differs from `place` in two
  ways: the values cursor advances by POSITION (every element, in lockstep with `i`, wrapping at `nv`), not per True, and
  an empty `values` is a silent no-op (`place` raises ValueError). The kernel has a cursor-free `nv == 1` broadcast path
  (NumPy's `if (nv == 1)`). Errors: read-only → `"putmask: output array is read-only"` (first, before the mask size and
  nv), a wrong size → `"putmask: mask and data must be the same size"`. Like `place`/`put`, `values` is cast to `a`'s
  dtype with `astype`. Gates: `Indexing/SelectionTests.cs` (`PutMask_*`) and the `putmask` tier.

## `take_along_axis` / `put_along_axis` / `select`

- `np.take_along_axis(arr, indices, axis=-1)` (`numpy/lib/_shape_base_impl.py`) is the per-slice gather that inverts
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
- `np.put_along_axis(arr, indices, values, axis)` is the setter twin, in place (returns `void`); `axis` is required.
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
  `default` is `object` (null is NumPy's `0`). Weak scalars, select's own rule: only `int`/`float`/`double`/`Complex`
  literals are weak (they adopt the other operands' dtype); `bool`, `char`, `Half`, `decimal`, arrays and every `NDArray`
  are strong (NumPy's `type(x) in (int,float,complex) else asarray(x)`), so a weak-int default that overflows the
  resolved dtype WRAPS (`int8` choice + `1000` default → -24); the result dtype is `result_type(*choices, default)`
  through `NDExprTypeRules.PromoteStrong`. Conditions and choices broadcast in two separate groups; a non-bool condition
  is `TypeError("invalid entry {i} in condlist: should be boolean ndarray")`; a length mismatch and an empty condlist are
  `ValueError`. Contiguous, no-cast, full-size array choices run one fused IL kernel (`DirectILKernelGenerator.Select.cs`:
  a reverse `Vector.ConditionalSelect` chain seeded with the default, choices overlaid n-1 → 0 so the first true
  condition wins, 4× unrolled with four accumulators, the scalar default hoisted with `CreateBroadcast`, `np.where`'s
  `EmitInlineMaskCreation` for the bool → lane expansion). Every other shape (broadcast, strided, cast, scalar choices)
  takes NumPy's structure: allocate the default-filled result, then `np.copyto(where=cond)` each choice in reverse.
  Gates: `Indexing/np.select.Test.cs` and the `select` op of the `groupa` tier.

## The grid / slice-expression DSL: `r_`, `c_`, `ix_`, `s_`, `index_exp`

NumPy's `_index_tricks_impl.py`; tests `Creation/np.r_.Test.cs`, `Indexing/np.ix_.Test.cs`; oracle cases `r_`/`c_`/`ix_`
in the `manip` tier. Sources `Creation/np.{r_,c_}.cs`, `Indexing/np.{ix_,s_}.cs`.
- A Python slice literal is a STRING in C#: `np.r_[0:5]` is `np.r_["0:5"]`. A string containing `':'` is a slice
  expression; a string without one is a NumPy special directive (`"r"`, `"c"`, `"-1"`, `"0,2"`, `"1,2,0"`). One string
  may hold several comma-separated slices (`np.r_["1:3, 5:8"]`), and a `Slice` / `Slice[]` object works as an entry, so
  `np.s_[…]` composes into `np.r_[…]`.
- `np.r_[…]` / `np.c_[…]` are static properties returning `RClass`/`CClass` (`AxisConcatenator` subclasses with a
  `this[params object[] key]` indexer, the port of NumPy's `__getitem__`). `r_` concatenates along axis 0; `c_` is
  `r_["-1,2,0", …]` (entries upgraded to ≥ 2-D with the 1s post-pended, a 1-D entry becomes a column, stacked on the last
  axis). Slice entries become `arange(start, stop, step)` or, for an imaginary step (`"-1:1:6j"`), `linspace(start, stop,
  N)` with the stop inclusive. Integer-ness follows the LITERAL (`"0:5"` is int64, `"0.0:5"` float64). A missing stop
  re-reads start as stop (`np.r_["2:"]` is `[0, 1]`, `np.r_["5::2"]` is `[0, 2, 4]`). Directives: axis, `axis,ndmin`,
  `axis,ndmin,trans1d` (NumPy's two upgrade paths: the array branch's `defaxes` permutation and the slice branch's
  `swapaxes(-1, trans1d)`), a silently ignored fourth field, and `"r"`/`"c"` matrix coercion (the result is an
  `NDMatrix` typed `NDArray`). Errors: `special directives must be the first entry.`, `unknown special directive` /
  `unknown special directive '0,q'` (the comma form quotes).
- `ndmin` is uncapped (NumSharp has no 64-dimension ceiling; NumPy raises `ndmin must be <= ndmax (64)`), and cheap:
  `np.AtLeastNdView` prepends every axis in one `Storage.Alias` over `Shape.PrependDimensions` (O(ndim)); a prepended
  axis takes stride `dimensions[0] * strides[0]` (0 for a 0-d source), so strides, flags and the writeable bit equal a
  per-axis `expand_dims` loop. An `ndmin` too large to allocate a shape for raises `OutOfMemoryException`; a non-positive
  `ndmin` is a no-op.
- `np.ix_(params object[])` → `NDArray[]` (an open mesh). The dtype is preserved (no integer validation; a float or int8
  sequence fails only at the later indexing call), except that an EMPTY non-ndarray input becomes int64; a bool sequence
  is a mask (`nonzero`). Results are views that write through, built with `expand_dims` (not `reshape`) so a
  stride-0/read-only operand keeps its strides and its read-only flag. `ValueError("Cross index must be 1 dimensional")`,
  which a null sequence also raises.
- `np.s_[…]` / `np.index_exp[…]` capture an index expression; `arr[X]` ≡ `arr[np.s_[X]]` for every X. Three indexers:
  `this[string]` and `this[params Slice[]]` → `Slice[]` for basic expressions (`int` and `string` convert to `Slice`), and
  `this[params object[]]` → `object[]` as soon as an entry cannot be a `Slice` (an `NDArray` index or mask,
  `int[]`/`long[]`/`bool[]`, an `IList`, a basic/advanced mix). Both feed `arr[…]` directly. `s_` and `index_exp` are
  identical in C# (NumPy's differ only by `maketuple`, which C# cannot observe).
- Weak scalars by C# type: `bool`, the eight integer primitives, `float`, `double` and `Complex` are weak (the C# literal
  is the Python literal); `char`, `Half`, `decimal` and every `NDArray` (0-d included) are strong
  (`np.r_[i8, NDArray.Scalar(1L)]` is int64, `np.r_[i8, 1L]` int8). A weak integer that does not fit the adopted dtype
  raises `OverflowException("Python integer 1000 out of bounds for int8")` (`NDExprTypeRules.CheckIntLiteralFits`); a
  weak float saturates to ±inf. An all-literal key whose integer does not fit int64 lifts the default to uint64. One
  complex width: `result_type(float32, 1j)` is complex128.
- Matched, no need to re-prove: `-0.0` keeps its sign; NaN/±inf/subnormals round-trip; the weak-integer boundaries are
  exact (`i8+127` fits, `+128` raises); `9e18:9e18+2` collapses to empty; huge slice expressions raise
  `OutOfMemoryException`; NumPy's error order holds. `np.r_`/`np.c_` are stateless singletons (a directive is copied to
  locals per call).
- Divergences: a `null` entry throws (NumPy builds an object array); `np.ix_("abc")` is accepted as a char sequence;
  a sub-1e-15 slice step reports `np.arange`'s "step can't be 0". No `bmat` branch: NumPy routes a lone bare string key
  into `matrixlib.bmat`, which resolves words against the caller's Python frame (and raises `NameError` for every
  literal), so a lone string is read like any other entry.
- `np.ogrid[...]` (`Creation/np.ogrid.cs`, `OGridClass`): the open mesh, with the same slice grammar
  (`AxisConcatenator.ParseSliceToken`/`SliceSpec`) and no directives. One slice returns a 1-D array (arange, or linspace
  for an imaginary step); several return N arrays each of extent 1 on every axis but its own. The whole mesh shares one
  dtype (int64 iff every field of every slice is an integer literal, else float64; an imaginary step floats the mesh).
  A single slice may omit its stop; a multi-slice may not (`ValueError`), and an imaginary step always needs a stop.
  `OGridResult` converts implicitly to `NDArray` (one slice) or `NDArray[]` and deconstructs
  (`var (y, x) = np.ogrid["0:3", "0:5"];`). Gate: `Creation/np.ogrid.Test.cs`.
- `np.mgrid[...]` (`Creation/np.mgrid.cs`, a property): one slice returns a 1-D array; several return ONE stacked array
  `(N, size_0, …, size_{N-1})`, built as `np.indices(sizes, typ)` with each non-trivial axis (start ≠ 0, step ≠ 1,
  imaginary) overwritten by its 1-D line broadcast across the grid (`np.copyto`), so `np.mgrid[0:a, 0:b]` is one fused
  `indices` fill. NumPy sizes each axis by `ceil((stop-start)/step)`, which may be negative: `mgrid[5:0, 0:3]` raises
  `"negative dimensions are not allowed"` where `ogrid[5:0, 0:3]` gives an empty axis, and a zero step divides by zero.
  `MGridResult` converts to `NDArray`, deconstructs (`var (x, y) = np.mgrid["0:5", "0:3"];`) and indexes the per-axis
  grids. Gate: `Creation/np.mgrid.Test.cs`.
- `np.meshgrid(x1, x2[, x3], string indexing="xy", bool sparse=false, bool copy=true)` plus `meshgrid(NDArray[] xi, …)`
  (`Creation/np.meshgrid.cs`): `"xy"` swaps the first two axes ((N, M) for 2-D) against `"ij"` ((M, N)); `sparse` keeps
  the open-mesh `(1,…,Ni,…,1)` shapes; `copy=false` returns broadcast views. Each grid keeps its input's dtype. A bad
  `indexing` raises `ValueError "Valid values for \`indexing\` are 'xy' and 'ij'."`. `MeshgridResult` converts to
  `NDArray[]`, deconstructs and indexes. Gate: `Creation/np.meshgrid.Test.cs`.

## The diagonal and triangular family

NumPy's `_twodim_base_impl.py` + `_index_tricks_impl.py`; tests `Indexing/np.{diag,tri,diag_indices}.Test.cs`, oracle
tiers `manip`/`matmul`; sources `Creation/np.tri.cs`, `Indexing/np.{diag,tril,diag_indices,tril_indices,fill_diagonal}.cs`.
- `diag(v, k=0)`: a 1-D input CONSTRUCTS a fresh writeable C-contiguous `(n, n)` matrix (`n = v.size + |k|`) with `v` on
  the k-th diagonal, scattered down the stride-`(n+1)` diagonal by the IL strided-store kernel
  (`DirectILKernelGenerator.DiagWrite`: a byte-width-keyed MOV loop for itemsize 1/2/4/8/16; the aliased-view + `SetData`
  route when dynamic codegen is unavailable, as under AOT). A 2-D input EXTRACTS through `diagonal(v, k)`: a read-only
  view sharing storage. Anything else raises `ValueError("Input must be 1- or 2-d.")`.
- `diagflat(v, k=0)` is `diag`'s 1-D branch for any rank: the input is raveled in logical C order first; 0-d → `(1,1)`.
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
  views. `val` is `object` (scalars, `NDArray`, C# arrays (a rectangular 2-D one ravels), `IEnumerable`); only genuine
  scalars take the `NDArray.Scalar` path; `Half`/`Complex`/arrays route through `asanyarray`. A null `val` throws
  `ArgumentNullException` (NumPy writes nan for a float array: a null in C# is a caller bug).
- `diag_indices(n, ndim=2)` / `diag_indices_from(arr)` / `tril_indices(n, k=0, m=None)` / `triu_indices` /
  `tril_indices_from(arr, k=0)` / `triu_indices_from` / `mask_indices(n, mask_func, k=0)` return `NDArray<long>[]`.
  `diag_indices` returns the same array instance in every slot (NumPy's `(idx,) * ndim` aliasing); `ndim <= 0` gives an
  empty array. `mask_indices`' arity follows the mask function's rank (`diag` yields one array). `tril_indices` /
  `triu_indices` emit the triangle's row/column runs directly in O(output), with no N×M bool mask.
- The triangular fills pick their allocation by size: below 64 MiB (`WriteOnceMaxBytes`) a pooled buffer comes back
  dirty, so they allocate uninitialised and write every byte once; above it `np.zeros` rides free OS zero pages and only
  the kept run is written.
- `diag` / `diagflat` construction allocates a fresh `n×n` buffer and faults about n scattered pages on the sparse
  diagonal write; NumPy's benchmark loop reuses a warm block, which NumSharp's allocator does not cache.
