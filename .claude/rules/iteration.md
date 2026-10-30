---
paths:
  - "src/NumSharp.Core/Backends/Iterators/**"
  - "src/NumSharp.Core/APIs/np.{nditer,nested_iters,flatiter}*.cs"
  - "src/NumSharp.Core/APIs/NDArrayNetInterop.cs"
  - "src/NumSharp.Core/Indexing/np.{ndindex,ndenumerate}.cs"
  - "src/NumSharp.Core/Backends/NDArray.Unsafe*.cs"
  - "test/NumSharp.Tests/APIs/**"
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
    fast. The 0-d hot path takes the direct slice constructor; the strided `external_loop` path (once per chunk) keeps
    the storage route. Prefer the typed iterators, `np.evaluate` or vectorized operations.
- Traps:
  - `NDIterRef` is a `ref struct` and cannot be a class field. The bridge is `NDIterRef.Detach`/`Borrow`
    (`Backends/Iterators/NDIter.Detach.cs`), which hands the heap `NDIterState*`, the operands AND the pending
    COPY_IF_OVERLAP write-backs to the managed owner and re-borrows a non-owning `NDIterRef` per call
    (`ReleaseState`/`FreeState` alone would strand `_writebackOriginals`). `np.nditer` must be disposed (`close()` /
    `using`), like NumPy's `with np.nditer(...) as it:`.
  - `GetEnumerator()` must not return `this`: `foreach` disposes the enumerator it obtains, and this class's `Dispose`
    frees unmanaged state. A thin wrapper with a no-op `Dispose` shares the same live cursor.
  - `MoveNext` publishes the element and then advances; NumPy's `__next__` (`npyiter_next`) advances on entry once its
    `started` flag is set and then returns `value`, so after one `next()` NumPy's cursor still reads the element it
    returned (`iterindex` 0) and a `copy()` taken there yields from the second element. NumSharp's `copy()` output
    agrees, but after k steps `iterindex` / `value` / `multi_index` read one element ahead, and in a BUFFERED readwrite
    loop (`foreach (var x in it) x[...] = ...`) the advance flushes the buffer before the loop body writes the last
    element of each buffer (or external-loop chunk), so that write never reaches the operand. Unbuffered loops are
    unaffected. Open gap SU42 (`surface_gaps.jsonl`, `OpenBugs.Surface.cs`). Under buffered `external_loop`, a chunk view
    held past the next step aliases a refilled buffer (hence NumPy's `[x.copy() for x in it]` idiom).
  - Do not fix `it[0]` by re-seating a cached view: `UnmanagedStorage` keeps three synchronized address caches (the
    public `byte* Address`, the `IArraySlice InternalArray`, and a per-dtype `ArraySlice<T> _arrayXxx` field). Re-seating
    `Address` alone makes `GetAtIndex`/`ToString`/casts correct while `np.sum(it[0])` and `it[0] * 2` silently return 0;
    re-seating `Address` and `InternalArray` still leaves the arithmetic operators and comparisons stale. It is safe only
    behind one `UnmanagedStorage` method that updates every cache.
- Gate: unit tests (`Indexing/np.ndindex.Test.cs`, `Indexing/np.ndenumerate.Test.cs`, `APIs/np.nditer.Test.cs`, from
  NumPy output). These are iteration protocols with no `(dtype, shape, bytes)` result, so they are not in the
  differential corpus.

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
