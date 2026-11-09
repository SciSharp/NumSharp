---
paths:
  - "src/NumSharp.Core/Sorting_Searching_Counting/**"
  - "src/NumSharp.Core/Backends/Default/Sorting/**"
  - "src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Arg*.cs"
  - "src/NumSharp.Core/Backends/Default/Indexing/Default.{Argwhere,FlatNonZero}.cs"
  - "src/NumSharp.Core/Utilities/QuickSelect.cs"
  - "test/NumSharp.Tests/{Sorting,Sorting_Searching_Counting}/**"
---

# Sorting and searching

The engine members are DefaultEngine's `TensorEngine.Sort` / `SortInPlace` / `ArgSort` / `Partition` /
`PartitionInPlace` / `ArgPartition` (`Backends/Default/Sorting/Default.Sort.cs`) and `LexSort` (`Default.LexSort.cs`),
over the AxisSort / AxisPartition kernels; `Sorting_Searching_Counting/np.*.cs` are the facades. Text (`S` / `U` /
`StringDType`) sorts, searches and set routines: `strings-compute.md`.

## The float value sort: `±0` in input order

`np.sort` / `ndarray.sort` of float32 / float64 radix-sort the raw-bit keys (`AxisSort.SortLineF32` / `SortLineF64`),
which order `-0.0` below `+0.0`. NumPy's comparators call the zeros equal, so its stable sort keeps them in input order
(`np.sort([0.0, -0.0, 1.0, -0.0, 0.0], kind='stable')` is `[0., -0., -0., 0., 1.]`), while its default kind (x86-simd-sort)
orders them by CPU (on an AVX2 host not even as a permutation of the signs). A line holding a `-0.0` (detected while its
keys are built: one compare per element for every other line) goes to `SortLineF32SignedZeros` / `SortLineF64SignedZeros`
for EVERY kind: the zeros stay out of the radix, their bits wait in the key buffer's top slots in input order beside the
stable kind's NaN bits, and the write-back is the negatives, the zeros, the positives, then the NaN tail (canonical NaN
for the default kind, each NaN's bits for the stable one). That equals a stable sort on the folded key, since two equal
non-zero floats have the same bits. Gate: `Sorting/np.sort.SignedZero.Test.cs`.

## `np.argsort`: one stable radix order

`np.argsort` is one stable LSD radix order for every numeric dtype but decimal (`Backends/Default/Sorting/AxisSort.cs`
over `RadixSort.ArgSortU32/U64`). The kind is ignored: every kind gets NumPy's stable (mergesort) order. An argsort only
orders indices, so it can fold what NumPy's comparators call equal, which the VALUE sorts of float16 and complex128
cannot (they rebuild values from keys and need distinct keys for ±0 and the NaN payloads, so they keep the comparison
sort):
- float16 (`ArgLineF16`): `HArgKey16` maps the sign-magnitude bits to an unsigned order with both zeros on one key, the
  non-NaN indices go through the 4-byte core with two key bytes, and the NaN indices follow in input order (`HALF_LT`
  with ties by index).
- complex128 (`ArgLineC128`): `CDOUBLE_LT` is lexicographic on (class, first key, second key), so the line is
  partitioned stably by class (no NaN part; `R+nanj`; `nan+Rj`; `nan+nanj`), the first class radix-sorted by its
  imaginary key and then stably by its real key, the second and third by their one key, the fourth kept in input order.
- np.unique's argsort-and-gather routes run on it (`unique.md`). Gate `Sorting/np.argsort.HalfComplex.Test.cs` (NumPy's
  stable indices under every kind, past the insertion thresholds).

## `np.nonzero` of a 0-d array

`np.nonzero` / `ndarray.nonzero` / single-argument `np.where` of a 0-d array raise NumPy 2.4.2's `ValueError: Calling
nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.` for every dtype, before reading the
element (`DefaultEngine.NonZero`; parity QA-02, breaking: the numeric dtypes answered like `atleast_1d(a).nonzero()`).
A BOOL operand gets NumPy's longer text, the same words followed by ``If the context of this error is of the form
`arr[nonzero(cond)]`, just use `arr[cond]`.`` (`PyArray_Nonzero`; `DefaultEngine.NonZero0dMessage` picks it, and
numpy.ma's nonzero inherits it). `np.argwhere` keeps NumPy's own atleast_1d branch, and a 0-d boolean INDEX
(`a[np.array(True)]`) is not a nonzero call. Gates: `Indexing/NonzeroIlRefactorTests.cs` (`Refactor_0d_*`), the nine
`dtype_text` cases `nonzero_all/scalar_0d/*` (no excuse) and the `ma_surface` tier's ma.nonzero cases.

## `np.argwhere` / `np.flatnonzero`

Both return views of one buffer, as NumPy's do (`Backends/Default/Indexing/Default.{Argwhere,FlatNonZero}.cs`):
`np.flatnonzero` is the column of nonzero's `(k, 1)` buffer; `np.argwhere` the TRANSPOSE of the `(ndim, k)` stack
`np.asarray(np.nonzero(a))` makes, F-contiguous with byte strides `(8, 8k)` (a 0-d input: `argwhere(atleast_1d(a))[:,
:0]`). Both report OWNDATA false; code that reads their raw buffers must use the strides. Gates: the layout-parity
oracle's argwhere / flatnonzero families, the string oracle (no layout excuse) and `Indexing/IndexSearchLayout.Test.cs`.

## Axis `argmax` / `argmin`

`Backends/Default/Math/Reduction/Default.Reduction.ArgAxis.Fast.cs`, `TryExecuteAxisArgFast`, hooked into
`ExecuteAxisArgReduction` (one hook serves both). The non-axis dims coalesce, and the walk is picked from the strides:
- Zero: a broadcast axis (every stride 0; nothing is read).
- Rows: the axis contiguous → the flat SIMD row kernel per output (inline under 32 elements).
- Slab: a non-axis run at least as tight → up to 1024 lanes folded together row by row, the running best in L1 scratch,
  a `Vector256` compare per row for a ±1 run (-1 flipped), an overlapping final vector instead of a scalar tail, the
  index written only for improved lanes.
- AxisWalk: the axis is the tightest non-contiguous stride.

NumPy's rule in logical axis order: the first occurrence (-0 == +0), the first NaN wins argmax AND argmin, integers
strict, bool the first True/False (0 when none; bool slabs stop once every lane is decided). Bool, integers, Char (as
u16), f32 and f64; Half/Decimal/Complex and rank > 64 decline (hooks `DefaultEngine.DisableFastAxisArg` /
`FastAxisArgRuns`, thread-static).
- The 64-bit flat row kernel is NumPy's single-pass tournament (`ArgMaxInt64Tournament` / `ArgMinInt64Tournament` in
  `DirectILKernelGenerator.Reduction.Arg.cs`, the port of `simd_argmax_{s64,u64}`): `Vector256.Max<long>/<ulong>` has no
  AVX2 instruction. A 32-bit tournament is slower in cache, so 8-, 16- and 32-bit lanes keep the two-pass kernel; the
  routing is gated on `Vector256.IsHardwareAccelerated`.
- An out-of-range axis is NumPy's AxisError (`np.argmax(a2d, -3)`; a 0-d input is raveled first, `PyArray_CheckAxis`:
  `axis-arguments.md`). `out=` is `TensorEngine.ReduceArgMax` / `ReduceArgMin`'s contract (`Default.ArgOut.cs`: the out's
  shape, its safe cast to intp, then the unsafe copy; a text operand's out is checked before its comparison).
- Gate: `Backends/Kernels/AxisArgFastPathTests.cs`.

## `partition`, `argpartition`, `lexsort`, `nanargmax`, `nanargmin`, `sort_complex`

Compositions and wrappers over introselect (`Utilities/QuickSelect.cs`, the median/percentile primitive), the stable
radix argsort (`AxisSort`) and the engine's `ArgMax`/`ArgMin`. Gates: `Sorting/np.{partition,lexsort,nanargmax,
sort_complex}.Test.cs` and the `sort`/`nanreduce` tiers. Sources:
`Sorting_Searching_Counting/np.{partition,argpartition,lexsort,nanargmax,sort_complex}.cs` (the facades),
`Backends/Default/Sorting/Default.NanArgMax.cs` (`TensorEngine.NanArgMax` / `NanArgMin`: the axis check, the text route,
`_replace_nan` and the arg reduction) and `Default.{Sort,LexSort}.cs`.
- `partition(a, kth, axis=-1, kind='introselect', order=null)` and the in-place `ndarray.partition`: the kth element(s)
  land in their sorted position, the sides unordered. `Backends/Default/Sorting/AxisPartition.cs` drives one introselect
  line kernel per 1-D lane through `AxisSort.DriveAllButAxis` (the NDIter IterAllButAxis loop the sorts use).
  - Validation in NumPy's order with its texts: kind (`select kind must be 'introselect' (got '…')`) → order (`Cannot
    specify order when the array has no fields.`) → axis → writeable (`partition array is read-only`) → kth
    (`kth(=N) out of bounds (M)` with the post-wrap value; bounds are skipped for size-0 arrays; the kth list is sorted so
    partitions compose, and multi-kth segments confine each introselect).
  - Floats move NaNs to the tail with their ORIGINAL bit patterns (unlike np.sort's canonicalizing radix); Complex uses
    NumPy's CDOUBLE_LT comparator, whose NaN order is positional ((1,NaN) before (NaN,1)) and is not compacted.
  - An empty `int[]` kth is NumPy's `np.array([], intp)`: a valid no-op copy (Python's bare `[]` is float64 → TypeError;
    a typed array carries no such ambiguity). The result is a fresh C-contiguous copy (NumPy's `copy(order='K')` keeps F
    order for an F input; the values are identical).
  - The NDArray-kth overload (NumPy's array form, on np.partition/np.argpartition and the ndarray methods) makes NumPy's
    kth-dtype rejections reachable: a bool kth `Booleans unacceptable as partition index`, a non-integer (Decimal
    included) `Partition index must be integer` (TypeError), more than 1-D `object too deep for desired array`, with
    NumPy's staging: kind → order → too-deep (before the axis: `partition(a, [[1]], axis=9)` reports too-deep) → axis →
    bool/integer dtype → wrap/bounds. Integer kth values cast to intp MODULARLY (uint64 2^63 wraps negative and its
    bounds error reports the wrapped value; 2^64−1 wraps to a legal −1); Char rides the integer route; 0-d and any layout
    are accepted.
- `argpartition`: the same driver returning int64 indices; the source is only read (broadcast views are legal). Built on
  the index-tracking QuickSelect overloads (`PartitionAtMany(T*, long*, …)`: values and their original indices swap in
  lockstep, the `RadixSort.ArgSortU32/U64` pattern). A 0-d input ravels to `(1,)` FIRST (NumPy's arg-side
  `PyArray_CheckAxis`: axis/kth errors then report dimension/size 1, the result is `[0]`), while `np.partition(0-d)`
  raises AxisError. Float NaN indices go to the tail in encounter order (argsort's policy). The tier pins the DERIVED kth
  values (`take(partition(a, ks), ks)` / the take_along_axis gather at ks): the arrangement between kth anchors is
  implementation-specific on both sides, so the two-sided ≤/≥ invariant is unit-test-pinned instead.
- The edge selection (`Utilities/QuickSelect.EdgeSelect.cs`, called first by the contiguous-line kernels of both
  functions): a single kth with m ≤ 64 positions on its side (the top-k idiom `argpartition(x, -k)`, or a kth near 0) of
  a contiguous line of at least max(256, 32·m, m²) elements, for the ten dtypes with `Vector<T>` lanes. One read keeps
  the m best sorted and rejects whole vectors against the worst kept; argpartition then writes every other index
  ascending (a vector iota with holes) and the kept ones from kth outwards, partition moves only the kept values that
  were outside the kth's block. Among equal values it keeps the lower index. It declines, leaving the line untouched for
  the quickselect: any NaN (the compaction's tail policy), more than m·(log2 n + 8) admissions (a line sorted the wrong
  way for its side), and the lines NumPy's own linear shortcuts serve with NumPy's tie choice (kth < 3, a float
  kth = n − 1). `NS_EDGE_SELECT_MAX` (0..64; 0 turns it off) is read once. 51,200 float32, kth = n − 5: argpartition
  206 → 23 µs (NumPy 91–96), partition 170 → 17 µs. Gate: `Sorting/np.partition.EdgeSelect.Test.cs`.
- `lexsort(keys, axis=-1)` (`TensorEngine.LexSort`): an indirect stable sort, the LAST key primary, reproducing
  `PyArray_LexSort` pass for pass:
  successive stable argsorts first → last key, each re-sorting the running permutation through `take_along_axis`
  (identical indices to NumPy's mergesort passes, ties included, because the radix argsort is stable). Errors: `need
  sequence of keys with len > 0 in lexsort` (TypeError), `all keys need to be the same shape` (ValueError); 0-d keys let
  axis 0/-1 slip (NumPy's back-compat), and size ≤ 1 returns a 0-filled int64 array. The single-NDArray overload reads
  the first axis as the key list (a 1-D input degenerates to N scalar keys → the 0-d `0`, as in NumPy).
- `nanargmax`/`nanargmin(a, axis=null, out=null, keepdims=false)`: NumPy's `_replace_nan(∓inf)` plus the all-NaN-slice
  guard (`ValueError("All-NaN slice encountered")`), then `argmax`/`argmin`; int/bool/char/decimal pass through (NumPy's
  `mask=None` branch). The flat form returns `long`; the `int?`-axis overload returns the int64 NDArray and handles
  `keepdims` with `axis=null` (shape `(1,)*ndim`).
  - `out=` follows NumPy's argmax contract: any bool/int dtype safe-castable to intp is accepted (bool through uint32;
    uint64 and floats raise `Cannot cast array data from dtype('…') to dtype('int64') according to the rule 'safe'`),
    the shape must EQUAL the keepdims-adjusted result (`output array does not match result of np.argmax.`; nanargmin's
    text says "argmin"), the int64 indices cast UNSAFELY into it (an int8 out wraps index 200 to −56, a bool out reads
    index ≠ 0), the same instance returns, a strided out writes through its base, and the all-NaN guard fires before out
    is touched. The third positional is `out` (NumPy's order), so pass `keepdims:` by name (the implicit bool → NDArray
    conversion would otherwise bind a positional bool to `out`).
  - Axis validation runs first for every dtype path, with NumPy's rule that a 0-d array accepts axis 0/-1. NumPy's
    NaN+Inf tie caveats fall out of the composition identically.
- Flat `argmax`/`argmin` engine facts: Decimal takes the `ExecuteReducing` fallback family (`DecimalArg{Max,Min}Kernel`;
  the IL arg kernel's `Bgt`/`Blt` cannot compare a 16-byte struct); Char rides the zero-copy uint16 reinterpret its amax
  uses; the strided/scalar IL step compares floats with ORDERED `Bgt`/`Blt` (the unordered `_Un` branches also fire when
  the ACCUMULATOR holds NaN, so every later finite value would steal the index). The reduce tier generates the flat
  `axis=None` argmax/argmin cells (keepdims=False flat cells replay through `NDArray.Scalar(np.argmax(a))`), and the
  decimal oracle has flat argmax cases.
- `sort_complex(a)`: copy and sort along the LAST axis in the input's own dtype (an int64 past 2^53 orders by its exact
  value), then up-cast. NumSharp's single complex type absorbs NumPy's complex64 cells (int8/16, uint8/16 inputs: the
  values are identical, the width differs; those four dtypes are unit-test-pinned and left out of the tier). A 0-d input
  raises `ndarray.sort()`'s `axis -1 is out of bounds for array of dimension 0`, as NumPy leaks it.
