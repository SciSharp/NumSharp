---
paths:
  - "src/NumSharp.Core/Backends/Iterators/**"
  - "src/NumSharp.Core/APIs/np.{nditer,nested_iters,flatiter}*.cs"
  - "src/NumSharp.Core/APIs/NDArrayNetInterop.cs"
  - "src/NumSharp.Core/Indexing/np.{ndindex,ndenumerate}.cs"
  - "src/NumSharp.Core/Backends/NDArray.Unsafe*.cs"
  - "test/NumSharp.Tests/APIs/**"
  - "test/NumSharp.Tests/Backends/Iterators/NDIter*.cs"
---

# Iteration

## NDIter

The NumPy-aligned multi-operand iterator (`Backends/Iterators/NDIter.cs`): C/F/A/K order, broadcasting, external loops,
buffering, casting, masks, reductions, and synchronized traversal for copy and elementwise kernels.
- A heap `NDIterState` is ONE native block: the header plus a 1536-byte inline arena that `AllocateDimArrays` carves the
  dimension/operand arrays from (`NDIterRef.AllocateStateBlock` / `ReleaseStateBlock`, `NDIter.State.cs`), recycled
  through a bounded `[ThreadStatic]` 4-slot cache. `ResetForRecycle` re-zeroes the header and only the carved bytes:
  every array the iterator hands out must start zeroed (`BaseOffsets` is accumulated with `+=` by
  `FlipNegativeStrides`). Arena overflow (about ndim × nop beyond 8 × 8) falls back to separate blocks; stack states
  (`CreateCopyState`) have no arena.
- The ufunc routes identify their Tier-3B kernel by the packed `InnerLoopKernelKey` (a front cache over the string-keyed
  `_innerLoopCache`; `ToCacheKey()` reproduces the `npy_binop_{op}_{l}_{r}_{res}` string on a miss), never by
  interpolating a string per call.
- The unbuffered EXTERNAL_LOOP drivers (`ForEach`/`ExecuteGenericMulti`/`ExecuteReducing`) call `ExternalLoopNext`
  directly, inlined with the state's array pointers hoisted into locals, not through the cached advance delegate. A
  narrow inner loop is a kernel-selection matter (the 2-D block kernel, `kernels.md`), not iterator overhead.
- The three `*UfuncInto` routes skip `Broadcast` + `ResolveUfuncIterationShape` when every operand has identical dims,
  and `ResolveWritebacks` disposes the COPY_IF_OVERLAP temp eagerly and reverts the operand slot to the original.
- The `where=` masked driver scans mask runs with SIMD (`NDIter.Execution.cs` `InvokeInner`): one movemask per 32-byte
  block, with the runs inside walked by trailing-zero counts (a per-boundary vector probe would slow run-length-1 masks).
- Gates: `NDIterStateBlockTests`, `InnerLoopKernelKeyTests`, `NDIterComparisonOutRouteTests`, the `out_where` tier.
- Benchmarks of the iterator run with `DOTNET_TC_CallCountingDelayMs=0` (the orchestrator sets it): tiered compilation's
  100 ms quiet-window rule keeps the first rows of each section at tier 0 otherwise.

## The NumPy iteration objects

`np.ndindex`, `np.ndenumerate` and `np.nditer` follow the `np.broadcast` house shape (a lowercase factory returning a
PascalCase nested class) and are their own iterators (NumPy's `iter(x) is x`), so a second enumeration RESUMES. Indices
are `long` (intp), and every step yields a fresh index array.
- `np.ndindex(...)` → `np.NDIndex`: a C-order odometer over a shape's index space, the last axis fastest. `params long[]`
  takes both spellings (`np.ndindex(3, 2)`, `np.ndindex(arr.shape)`); an `int[]` overload exists because array covariance
  will not widen it (not `params`, or `np.ndindex()` would be ambiguous). `ndindex()` and an empty shape yield ONE empty
  index; a zero-length dimension yields nothing; a negative dimension raises `ArgumentException "negative dimensions are
  not allowed"` at construction. Not NDIter-backed (NumPy's ndindex is `itertools.product`). `.AsSpans()` →
  `np.NDIndexSpans` is the allocation-free form: a `ref struct` enumerator yielding the index as a `ReadOnlySpan<long>`
  over a reused buffer (valid until the next step); it restarts on each `foreach`. `Indexing/np.ndindex.cs`.
- `np.ndenumerate(arr)` → `np.NDEnumerate` yields `(index, value)` in LOGICAL C order whatever the layout (NumPy's
  `asarray(arr).flat` + `flatiter.coords`; F-contiguous, transposed, reversed, sliced and broadcast views read through
  their strides with `Shape.TransformOffset`); a 0-d array yields one pair with an empty index, an empty array nothing.
  The coordinates come from an odometer advanced with the flat position. `np.ndenumerate<T>(arr)` yields unboxed `T`.
  `np.ndenumerate<T>(arr).AsRef(writeable=false)` → `np.NDEnumerateRef<T>` is allocation-free and by `ref T`: each step
  yields an `Entry` ref struct with `Index` (a `ReadOnlySpan<long>` over a reused buffer) and `Value` (a `ref T`, write-
  through), composing the C-order value ref of `NDRefIter<T>.Enumerator` with a lockstep coordinate odometer. `T` must be
  the exact dtype; `writeable: true` refuses a read-only broadcast; it restarts on each `foreach`.
  `Indexing/np.ndenumerate.cs`.
- `np.nditer(...)` → `np.NDIterator` is the managed face of `NDIterRef` (the port of `nditer_pywrap.c`): the full
  signature (`op, flags, op_flags, op_dtypes, order='K', casting="safe", op_axes, itershape, buffersize`), all 13 global
  and 15 per-op flag strings (including `grow_inner`/`growinner`), NumPy's error texts (`Unexpected iterator global flag
  "…"`, `Unexpected per-op iterator flag "…"`, `order must be one of 'C', 'F', 'A', or 'K' (got 'Q')`, `casting must be
  one of …`, `Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked`, `Iteration of
  zero-sized operands is not enabled`, `Must provide at least one operand`). Properties `nop`, `ndim`, `shape`,
  `itersize`, `operands`, `dtypes`, `value`, `it[i]`, `index`, `multi_index`, `iterindex`, `iterrange`, `itviews`,
  `finished`, `has_index`, `has_multi_index`, `has_delayed_bufalloc`, `iterationneedsapi`; methods `iternext`, `reset`,
  `close`, `copy`, `remove_axis`, `remove_multi_index`, `enable_external_loop`, `debug_print`. `APIs/np.nditer.cs`.
  - An ALLOCATE operand with no `op_dtypes` entry infers its dtype by promoting the operands that have one
    (`np.nditer([a, null])` allocates int64 for an int64 input; `np.nditer([int_a, float_b, null])` float64); the
    inference lives in the wrapper, where NumPy puts it.
  - The per-element `it[0]` loop costs one `NDArray` view per element (about 0.6 µs); the iteration engine itself is
    fast. The 0-d hot path takes the direct slice constructor (marking the storage `ExternalBase`, so the view reports
    NumPy's `owndata` False); the strided `external_loop` path (once per chunk) keeps the storage route. Prefer the typed
    iterators, `np.evaluate` or vectorized operations.
  - Construction carries NumPy's descriptors (`NDIterRef.AdvancedNew(..., opDescrs)`,
    `Backends/Iterators/NDIter.Descriptors.cs`): `npyiter_check_casting`'s read and write-back checks, the copy routes of
    `npyiter_allocate_arrays` (a read-only 0-d cast operand, `copy`, `updateifcopy` written back at `close()`), the
    "required copying or buffering" TypeError, allocated outputs laid out in the settled order, and the string classes
    (0-d `S` / `U` views, buffered / copy-route conversions between widths, between `S` and `U` and between text and
    numbers). The wrapper is `nditer_pywrap.c`: the started / finished flags, every getter's and setter's checks and texts
    ("Iterator is past the end", "Iterator is invalid", the delayed-allocation text), `it[i] = v` storing like `a[...] = v`
    (NumPy's `PyArray_CopyObject`: a C# literal packed as a Python scalar with NumPy's range check, a 0-d array cast, null
    as Python's None), NumPy's write flags on the views. The views are the iterator's (`NDIterRef.CurrentView`,
    `Backends/Iterators/NDIter.OperandViews.cs`: the storage lane, a string view's width and class, a `StringDType` view's
    allocator); like NumPy's they do not own their data, and a write flag cleared on one cannot be set again (the view's
    storage is `WriteProtected`; a sub-view of a 0-d view refuses too, where NumPy's may be set again).
    `np.nested_iters` splits flags and dtypes per level (a buffered cast lands in the innermost level).
  - Engine rules a port keeps: the flat index rides its own strides (`NDIterState.IndexStrides`, NumPy's index "operand",
    permuted, flipped and merged with the axes, so an index-tracking iterator coalesces when the index allows and
    `remove_multi_index` cannot break it); `shape` without a multi-index lists the iterator's axes innermost first;
    `remove_multi_index` only coalesces (a forced C / F order keeps walking that order). On the descriptor path the K-order
    sort is NumPy's `npyiter_find_best_axis_ordering` (`NDIterCoalescing.FindBestAxisOrderingNumPy`: a zero stride is
    ambiguous, C order wins a conflict); the internal callers keep the min-stride sort. The op_axes reduction checks are
    `npyiter_check_reduce_ok_and_set_flags` (innermost axis first; an explicit reduction axis needs no `reduce_ok`). A
    written operand's reduction refusal is NumPy's SHAPE test (`NDIterRef.IsBroadcastAxis`: the operand's own dimension 1
    or missing under a longer iteration axis, not any zero stride: a zero-size array has zero strides without a
    broadcast), with NumPy's axis number counted from the innermost axis.
  - The construction checks run in NumPy's order: the wrapper converts `flags`, `order` and `casting` before the operand
    count; `npyiter_init`'s `itershape` rule; `npyiter_check_op_axes` BEFORE the flag checks (a duplicate axis "contained
    duplicate value 0", an axis past 64 dimensions "contained invalid values"); each operand's
    `npyiter_prepare_one_operand` checks with the reference check in its place (a `StringDType` operand without `refs_ok`
    is NumPy's TypeError even when its `op_axes` is wrong too; an output the iterator allocates is never checked:
    `NDIterRef.ThrowIfReferenceOperandUnsupported`); then `npyiter_fill_axisdata`'s shape pass (`FillAxisDataShapePass`,
    NumPy's broadcast texts with the remapped and requested shapes, an op_axes entry printed REVERSED as NumPy prints it, a
    NEGATIVE `itershape` entry deduced from the operands). The stride pass's refusals are ONE loop,
    `NDIterRef.ThrowStridePassRefusals` (iteration axes innermost first, operands in order: `no_broadcast`'s
    `operand_different_than_broadcast` ValueError, an explicit reduction axis of extent other than 1, the reduction rule);
    the wrapper hands its shape to the engine (`AdvancedNew(..., shapeValidated: true)`), which then skips its own
    broadcast computation.
  - Buffered iteration is a statement-for-statement port of NumPy 2.4.2's buffering
    (`Backends/Iterators/NDIterBufferManager.Buffering.cs`), for `np.nditer` and every internal buffered iterator (the
    ufunc `out=` / `where=` routes, the clip / power iterator routes, buffered casts): `npyiter_find_buffering_setup` (the
    core / outer split by the cost model `(1 + buffered operands) / elements per fill`, per-operand BUFNEVER /
    single-stride / reduce decisions, the cap in whole cores and its `grow_inner` exemption), `npyiter_copy_to_buffers` /
    `npyiter_copy_from_buffers` (a fill never crosses the outer block, the reduce double loop's outer size, the buffer
    reuse rule, the masked write-back), `npyiter_buffered_iternext` / `npyiter_buffered_reduce_iternext` and
    `npyiter_goto_iterindex`'s core offset; `NPY_ITER_CONTIG` makes a non-contiguous operand a CAST operand. Mapping:
    NumPy numbers axes from the innermost (idim 0 = NumSharp axis `NDim - 1`); its USERPTRS are `DataPtrs` and its
    NIT_DATAPTRS `ArrayWritebackPtrs`; `np.nditer` passes a buffer size of 8192 when given 0 (NumPy's NPY_BUFSIZE:
    `np.setbufsize` reaches only the engine's ufunc iterators, as in NumPy). NumPy's own edges are kept where they are
    memory-safe (a fill started inside a core writes nothing back; a reset that finds a buffer starting at the range start
    keeps it, so `NDIterRef.Iternext`'s range-end pre-check spares an iterator that still holds a buffer;
    `remove_multi_index` keeps the setup where it still walks the right elements, so an external loop after a merge of
    every axis hands out the old core's chunks: a (3, 4) float64 cast with a buffer of 5 gives `[4, 4, 4]` where a fresh
    iterator gives `[5, 5, 2]`). Not kept, all
    NumPy memory or value errors: iternext before a delayed allocation, a ranged reduce fill of one core and a part, and
    any other partial `remove_multi_index` merge (the port reruns the setup, `RebuildBufferingSetup`); `[Misaligned]`
    pins in `APIs/np.nditer.BufferedReduce.Test.cs`.
  - A `StringDType` operand: each buffer of a `T` side owns a `VStringAllocator` (NumPy's allocator belongs to the
    descriptor, NumSharp's to the cells' memory block): `NDIterVStringContext` (`Backends/Iterators/NDIter.VString.cs`),
    held by the state as a GCHandle and freed in `FreeDimArrays`; transfers are `NDVStringCasting.TransferRun` (duplicate,
    or the cast of the pair); `copy()` duplicates the buffered strings. Trap: a contiguous operand without a cast is
    iterated IN PLACE even with `buffered` (BUFNEVER): test buffers with a transpose walked in C order or a cast.
  - A failed iteration is not written back: `NDIterRef.Dispose` is NumPy's `NpyIter_Deallocate`, so when a driver threw
    (`ForEach`, `ForEachRows`, `ExecuteGeneric`, `ExecuteReducing`, `BufferedReduce`, the `Execute*` drivers and the 2-D
    block kernels) the held buffers are cleared (`NDIterBufferManager.ClearBuffers`, `npyiter_clear_buffers`) and the
    COPY_IF_OVERLAP temporaries discarded; the fills completed before the error stay written. The flag is
    `NDIterState.Failed`, set by the drivers' exception filter `catch (Exception) when (MarkFailed()) { throw; }`, which
    never catches: a new driver must wrap its loop the same way, or a kernel that throws mid-fill writes the half-filled
    buffer into the operands. A loop a caller writes AROUND the iterator (np.nditer's element walk, a hand-written
    `Iternext` loop) is not a failure: its buffers are written back, as NumPy's `with` block writes them back. Gate:
    `Backends/Iterators/NDIterFailedIterationTests.cs`.
- Traps:
  - `NDIterRef` is a `ref struct` and cannot be a class field. The bridge is `NDIterRef.Detach`/`Borrow`
    (`Backends/Iterators/NDIter.Detach.cs`), which hands the heap `NDIterState*`, the operands AND the pending
    COPY_IF_OVERLAP write-backs to the managed owner and re-borrows a non-owning `NDIterRef` per call
    (`ReleaseState`/`FreeState` alone would strand `_writebackOriginals`). `np.nditer` must be disposed (`close()` /
    `using`), like NumPy's `with np.nditer(...) as it:`.
  - `GetEnumerator()` must not return `this`: `foreach` disposes the enumerator it obtains, and this class's `Dispose`
    frees unmanaged state. A thin wrapper with a no-op `Dispose` shares the same live cursor.
  - `MoveNext` is NumPy's `__next__` (`npyiter_next`): it advances on entry once its `started` flag is set and then
    publishes, so after one `next()` the cursor still reads the element it returned (`iterindex` 0), a `copy()` taken
    there yields from the second element, and a BUFFERED readwrite loop's write to the last element of each buffer (or
    external-loop chunk) reaches the operand. Under buffered `external_loop`, a chunk view held past the next step
    aliases a refilled buffer (hence NumPy's `[x.copy() for x in it]` idiom).
  - Do not fix `it[0]` by re-seating a cached view: `UnmanagedStorage` keeps three synchronized address caches (the
    public `byte* Address`, the `IArraySlice InternalArray`, and a per-dtype `ArraySlice<T> _arrayXxx` field). Re-seating
    `Address` alone makes `GetAtIndex`/`ToString`/casts correct while `np.sum(it[0])` and `it[0] * 2` silently return 0;
    re-seating `Address` and `InternalArray` still leaves the arithmetic operators and comparisons stale. It is safe only
    behind one `UnmanagedStorage` method that updates every cache.
- Gates: unit tests from probed NumPy 2.4.2 output (`Indexing/np.ndindex.Test.cs`, `Indexing/np.ndenumerate.Test.cs`,
  `APIs/np.nditer.Test.cs`, `APIs/np.nditer.NumPyParity.Test.cs`, `APIs/np.nditer.{Construction,Protocol,Strings}.Test.cs`,
  `APIs/np.nditer.BufferedReduce.Test.cs`) and the oracle tiers that materialize the protocols into comparable results:
  `iter.jsonl` (value streams, multi_index, c_index / f_index and external-loop chunk lengths), `nditer_construct.jsonl`
  and `nditer_protocol.jsonl` (state traces as text; the buffered sections record the reduce double loop, the chunking
  with and without `grow_inner`, CONTIG and WRITEMASKED), and the string oracle's `striter` / `titer` sections. The unit
  suites hold what a trace does not reach: the setters, copies, closed-state errors, buffered write-back and the
  `no_broadcast` rules.

## `np.nested_iters`, `flatiter`

- `np.nested_iters(op, axes, flags=null, op_flags=null, op_dtypes=null, order='K', casting="safe", buffersize=0)` →
  `NDIterator[]` (NumPy's `NpyIter_NestedIters`): one `np.NDIterator` per entry of `axes` (outermost first), all iterating
  the same operand buffer over disjoint axis subsets; advancing an outer iterator re-bases every inner one
  (`NDIterRef.ResetBasePointers`), so nested `foreach` loops walk nested loops. Each level is an `NDIterator` with
  `op_axes = [axes[level]]` sharing iter0's operands. The nested levels use NumPy's advance-on-entry protocol (the
  `started` flag; `_nestedMode` guards it, so plain nditer keeps its own). Validation (`ValueError`): `axes` needs at
  least 2 entries ("axes must have at least 2 entries for nested iteration"), no axis reused ("An axis is used more than
  once"), no negative axis ("An axis is out of bounds"); an axis past the operand's ndim is `IncorrectShapeException`
  where NumPy raises `ValueError`. The `multi_index` path and the bare value stream match NumPy (the op_axes contiguity
  check, `NDIter.CheckAllOperandsContiguous`, judges the iterator's remapped layout, so a non-contiguous axis subset keeps
  NumPy's memory order). `APIs/np.nested_iters.cs`; gate `APIs/np.nested_iters.Test.cs`.
- `NDArray.flatiter` / `np.flat(a)` → `np.FlatIterator`: a write-through C-order flat iterator, NumPy's `flatiter`.
  `NDArray.flat` returns a raveled `NDArray` (embedded in Core and not changed), which copies for a non-contiguous layout,
  so `a.T.flat[i] = v` is lost; `FlatIterator` reads and writes through for every layout (each element maps a flat index
  through the base's strides with `Shape.TransformOffset`, via `GetAtIndex`/`SetAtIndex`, no per-element allocation).
  Surface: `this[long]` get/set (negative wraps; out of bounds → `IndexError` "index N is out of bounds for size M"),
  fancy (`int[]`/`long[]`/`NDArray`) and slice-string (`"1:4"`, `"::2"`) get/set, the `index`/`coords` cursor (past the
  end `coords` leaves the slowest axis unwrapped: index == size on `(2,3)` → `(2,0)`), `Base`, `size`, `copy()`, and
  C-order iteration sharing the cursor (a second pass resumes). Scalar assignment into an integer dtype: an in-range
  value truncates toward zero, an out-of-range WEAK scalar (a C# primitive) raises `OverflowException` ("Python integer
  300 out of bounds for int8"; floats truncate first, NaN/inf raise; a bool target never raises), a STRONG scalar (an
  `NDArray`, the fancy/slice setters) wraps. Into float/complex/decimal it uses `Convert`.
- `np.flat<T>(a, writeable=false)` / `a.flatiter.AsTyped<T>(writeable=false)` → `np.FlatRefIter<T>` yields `ref T` in
  logical C order, write-through, for every layout: exactly `np.nditer<T>(a, order: 'C')` (a `readonly struct` whose
  `GetEnumerator()` reuses `NDRefIter<T>.Enumerator` pinned to `NPY_CORDER`). `T` must be the exact dtype (no coercion);
  `writeable: true` refuses a read-only broadcast; 0-d yields its element; empty iterates zero times; re-enumeration
  restarts. The instance spelling is `a.flatiter.AsTyped<T>()`: a generic method cannot share the `flatiter` property's
  name (CS0102). `APIs/np.flatiter.cs`; gate `APIs/np.flatiter.Test.cs`.

## Typed iteration: `np.nditer<T>` / `np.nditer_chunks<T>`

The unboxed counterparts of `np.nditer` over the same engine (NumPy has no equivalent; gate
`APIs/np.nditer.Typed.Test.cs`, against the boxed `np.nditer` and `ndenumerate` across every layout and all 15 dtypes):

```csharp
foreach (ref double x in np.nditer<double>(a, writeable: true)) x *= 2;              // element-wise
foreach (Span<double> c in np.nditer_chunks<double>(a, writeable: true))             // chunk-wise
    TensorPrimitives.Multiply(c, 2.0, c);
```

- `GetEnumerator()` must not return `this`: these own unmanaged state, so the factory returns a `readonly struct`
  enumerable holding no unmanaged state, whose `GetEnumerator()` builds a fresh `NDIterRef`. Re-enumeration therefore
  restarts (unlike the class-based `np.nditer`, which resumes).
- `foreach` disposes a `ref struct` enumerator through the same pattern (no `IDisposable` needed); hand-driven
  `MoveNext` loops must dispose it themselves.
- The walk is chunk-driven: EXTERNAL_LOOP is always on, and `MoveNext` walks the inner loop with pointer arithmetic,
  touching the iterator only at a chunk boundary (a contiguous array is one chunk).
- Never buffered: a `ref`/`Span` into a buffer would dangle at the next fill; unbuffered pointers are absolute into the
  operand, which is also why the enumerator can advance the iterator before handing out the captured chunk.
- Order is `'K'` (memory order), as `np.nditer`: `a[:, ::-1]` of `arange(6).reshape(2,3)` yields `0 1 2 3 4 5` in both
  libraries and `2 1 0 5 4 3` under `order='C'`. Pass `order: 'C'` for `ndenumerate`'s logical order.
- Divergences (pinned): a dtype mismatch throws (a `ref` cannot convert); an empty array iterates zero times where the
  boxed form demands `zerosize_ok`; `nditer_chunks<T>` rejects a non-unit-stride inner loop (`a[":, ::2"]`) up front,
  because a `Span<T>` is contiguous.
- `NDIterRef` defects the typed iterators route around (`TypedIterHelpers.ReadInnerLoop`): `GetInnerLoopSizePtr()` reads
  `Shape[NDim - 1]` and so access-violates on a 0-d operand (the internal `ResolveInnerLoopCount` guards it);
  `GetInnerStrideArray()` returns ELEMENT strides while the kernel contract takes byte strides from the private
  `GetInnerLoopByteStrides()` (no public byte-stride accessor); `ResetToIterIndexRange` + EXTERNAL_LOOP does not clamp
  the inner count to the range, so it reads past the array end (NumPy requires `NPY_ITER_RANGED` at construction), which
  blocks parallel range-splitting (splitting by array slice works).

## .NET interop: `nd.Unsafe` views and safe bridges

Both cover all 15 dtypes (generic `<T>` plus a `typecode == InfoOf<T>.NPTypeCode` guard). Gate
`APIs/NDArrayUnsafeInterop.Test.cs`; docs `docs/website-src/docs/iterating-and-enumerating.md` → ".NET interop".
- `nd.Unsafe` (`Backends/NDArray.Unsafe.Bcl.cs`, a partial of the `NDArray._Unsafe` struct, backed by
  `Backends/Unmanaged/UnmanagedMemoryManager.cs`): zero-copy BCL views that ALIAS the unmanaged buffer and do NOT root
  it (no ARC reference): if the NDArray is disposed or collected while a view is live, the view dangles, so keep the
  NDArray alive alongside. Surface: `Span<T>()` / `ReadOnlySpan<T>()`, `Memory<T>()` / `ReadOnlyMemory<T>()`
  (manager-backed: the `Memory` roots the small manager, never the NDArray), `Bytes()` / `ReadOnlyBytes()`
  (`size*itemsize`, dtype-agnostic), `Pointer<T>()` (logical element 0, any layout), `TryGetSpan<T>` / `TryGetMemory<T>`.
  The method names are the return types (the existing `AsSpan<T>()` returns NumSharp's long-indexable
  `UnmanagedSpan<T>`). Span/Memory/Bytes require C-contiguity (an offset slice `a[2:5]` qualifies; F, transposed,
  strided and broadcast throw `InvalidOperationException`, TryGet* return false), the exact dtype (`ArgumentException`)
  and at most `int.MaxValue` elements (larger: `AsSpan<T>()` / `Pointer<T>()`). The logical start pointer is
  `(T*)Storage.Address + Shape.offset`. `Pointer<T>()` of a non-contiguous array is element 0; walk it with `nd.strides`.
- Safe bridges (`APIs/NDArrayNetInterop.cs`, extension methods) copy or enumerate into caller-owned storage:
  `nd.AsEnumerable<T>()` → `IEnumerable<T>` in logical C order for every layout (unboxed; the dtype check fires eagerly),
  `nd.ToArray<T>()` (a C-order copy), and `.ToArray()` / `.CopyTo(Span<T>)` on the typed iterators (`np.flat<T>`,
  `np.nditer<T>`, `np.nditer_chunks<T>`; `CopyTo` returns the count and throws when the span is short). The boxed
  iteration objects implement `IEnumerable`, so LINQ works on them directly.
