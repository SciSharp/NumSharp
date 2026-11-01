---
paths:
  - "src/NumSharp.Core/View/**"
  - "src/NumSharp.Core/Backends/Unmanaged/**"
  - "src/NumSharp.Core/Backends/NDArray*.cs"
  - "src/NumSharp.Core/Generics/**"
  - "test/NumSharp.Tests/View/**"
  - "test/NumSharp.Tests/Backends/Unmanaged/**"
---

# Shape, views and memory

## Shape

`Shape` is a `readonly struct` whose `ArrayFlags` are computed once, at construction:

```csharp
public readonly partial struct Shape
{
    internal readonly int _flags;         // cached ArrayFlags bitmask
    internal readonly int _hashCode;      // precomputed hash code
    internal readonly long size;          // total element count
    internal readonly long[] dimensions;  // dimension sizes
    internal readonly long[] strides;     // stride values (0 = broadcast dimension)
    internal readonly long bufferSize;    // size of the underlying buffer
    internal readonly long offset;        // base offset into storage
}
```

`ArrayFlags` (NumPy's `ndarraytypes.h`):

| Flag | Value | Meaning |
|------|-------|---------|
| `C_CONTIGUOUS` | 0x0001 | row-major contiguous |
| `F_CONTIGUOUS` | 0x0002 | column-major contiguous |
| `OWNDATA` | 0x0004 | the array owns its data buffer |
| `ALIGNED` | 0x0100 | shape bit (cleared by `setflags(align=False)`); `flags.aligned` also needs `UnmanagedStorage.IsDataAligned`, false over a caller's buffer at a misaligning byte offset |
| `WRITEABLE` | 0x0400 | false for broadcast views |
| `BROADCASTED` | 0x1000 | has a stride of 0 with a dimension > 1 |

O(1) properties: `IsContiguous` (`C_CONTIGUOUS`), `IsFContiguous`, `IsBroadcasted`, `IsWriteable` (false for broadcast
views), `IsSliced` (offset != 0, a different size, or non-contiguous), `IsSimpleSlice` (`IsSliced && !IsBroadcasted`, the
fast offset path). `GetOffset(coords)` and `GetCoordinates(offset)` translate between coordinates and a storage offset.

Shapes and arrays share dimension and stride arrays (an allocation keeps its caller's shape, an elementwise result keeps
its input's), which is safe only because nothing public writes into them: `Shape` has no setter (its indexer included);
`a.shape`, `Shape.Dimensions` and `Shape.Strides` return copies (NumPy's tuples); every public constructor, conversion
and method that takes a caller's `long[]` copies it (NumPy's `PyArray_IntpConverter`). Inside Core:
- read the internal fields `x.Shape.dimensions` / `x.Shape.strides`, never `x.shape`, `x.Shape.Dimensions` or
  `x.Shape.Strides`, each of which allocates on every read;
- build a shape over an array Core created with `Shape.Adopt(arr)` (also `Adopt(arr, order)`, `Adopt(dims, strides)`, and
  `x.reshape(Shape.Adopt(arr))`), never `new Shape(arr)` or `(Shape)arr`, which copy;
- inline values (`new Shape(3, 4)`, `a.reshape(2, 3)`, `a.resize(2, 3)`) bind `params ReadOnlySpan<long>` overloads and
  allocate once.

## Views

Slicing returns a view that shares memory: `original["2:5"][0] = 999` writes `original[2]`; `.copy()` makes a copy.
A view keeps its source's engine (`UnmanagedStorage.Engine`: `Alias`, the new-storage slice paths of `GetViewInternal` /
`GetData`, broadcast and buffer views set it), and an engine binds what it computes to itself (`TensorEngine.NewArray` /
`NewScalar`), so a new view creator copies `Engine` too.

- `Slice` (`View/Slice.cs`) parses Python notation (`"1:5:2"`) into `Start` / `Stop` / `Step`; `SliceDef` holds the
  absolute indices, and `SliceDef.Merge()` composes a slice of a slice. Special instances: `Slice.All` (`:`),
  `Slice.Ellipsis` (`...`, fills the missing dimensions), `Slice.NewAxis` (inserts a dimension), `Slice.Index(n)` (one
  element; the dimension is dropped).
- Broadcasting aligns shapes from the right; each dimension pair must be equal or contain a 1, which stretches.
  `DefaultEngine.Broadcast` (`Backends/Default/ArrayManipulation/Default.Broadcasting.cs`) resolves the shapes.
- The operators live in `Operations/Elementwise/`: `NDArray.Primitive.cs` (`+ - * / %` and unary `-`, delegating to
  `TensorEngine.Add` / `Subtract` / …; a scalar operand is wrapped by `NDArray.Scalar()`) and `NDArray.Equals.cs` /
  `NDArray.Greater.cs` / `NDArray.Lower.cs` (comparisons returning `NDArray<bool>`). Indexing is
  `Selection/NDArray.Indexing*.cs` (`selection-indexing.md`).

A view keeps its owner's buffer alive, and so does a byte-reinterpreting alias:
- Every NDArray holds one counted reference on its storage's slice (`TryAddRef` at construction, `Release` on `Dispose`,
  `Abandon` from the finalizer), and an ordinary view shares the owner's slice.
- A view that reinterprets the bytes as another element type (a complex array's real/imaginary lane through
  `np.real`/`np.imag`/`.real`/`.imag`, `view(dtype)`, `getfield`) gets a slice of its own from `ArraySlice.WrapShared`,
  with an `AllocationType.Forward` disposer that forwards `TryAddRef`/`Release`/`Abandon`/`IsReleased`/
  `IsUniquelyReferenced` to the owner's slice (an alias of an alias reaches the owning block).
- NumPy's consequences follow: the owner's refcount counts each live alias, so `ndarray.resize(refcheck: true)` refuses
  while one lives, and the buffer returns to the pool when the last of owner and aliases is released. Gate:
  `Backends/Unmanaged/ArcLifecycleTests.cs` (`ReinterpretingAlias_*`).
- A method that reads an operand's raw pointer and runs a kernel with no later use of the operand needs that operand
  rooted, or optimized code lets a temporary operand's block be finalized under the kernel: an `[NDScoped]` method gets
  its operands rooted by the weaver (`ScopeWeaver.InjectParameterAnchors`), an unscoped entry point carries
  `[NDAnchored]`; never a hand-placed `GC.KeepAlive` (the lifetime gate: `oracle.md`).
- An internal alias must be disposed, or created inside an `[NDScoped]` member: an undisposed one holds the owner's
  buffer out of the pool until the finalizer runs. The `real`/`imag` setters and `setfield` are `[NDScoped]`;
  `view<T>()` and `getfield<T>()` construct the typed array directly (an untyped NDArray followed by `AsGeneric<T>()`
  would build a second NDArray).

Allocating from a view's Shape is safe. A `Shape` carries a view's strides, offset and buffer size next to its
dimensions, so `np.empty(x.Shape)` must not hand the allocator the view's addressing:
- Every allocating entry point (`UnmanagedStorage.Allocate(Shape, …)` behind `new NDArray(dtype, shape)`, `np.empty` and
  `np.zeros`, plus the `np.ones` and `np.full` cores) routes the shape through `Shape.ForFreshAllocation()`
  (`View/Shape.Allocation.cs`). A layout that already describes a fresh buffer (offset 0, buffer size == element count,
  dense C/F or a KEEPORDER permutation) is kept: the `copy(order='K')`, `NpyIterOutputShape` and `ConcatOutputShape`
  callers allocate exactly such shapes. Any view layout (a step, an offset, a negative stride, a broadcast, a contiguous
  slice reporting its parent's buffer) is laid out again in C order, NumPy's answer for a dims-only request. The result
  is always writeable.
- A shape used only for its dimensions (NumPy's `size` tuple, every `np.random` sampler) goes through the stricter
  `Shape.AsClean()`, which also re-lays out a dense F or permuted view (a sampler fills in memory order, so `x.T.Shape`
  would transpose the draws) and returns its input unchanged when it already is the C layout.
- The wrap constructors `new NDArray(Array | IArraySlice, Shape, order)` (and the `NDArray<T>` twins) are NumPy's
  `np.ndarray(shape, buffer=…, order=…)` and read only the shape's dimensions (`Shape.ForWrappedBuffer`): C strides, or F
  for `order='F'` at any rank, offset 0, writeable. `'A'`/`'K'` mean C and an invalid code is NumPy's `ValueError`
  (`OrderResolver.ResolveBufferOrder`). NumPy's size rule: a smaller buffer is `TypeError: buffer is too small for
  requested array`, a larger one is read from its start (a prefix sharing the caller's memory), and the lone `-1` takes
  every element.
- Internal code that must install a specific layout over a fresh buffer (`np.ones`'s KEEPORDER result) uses
  `new UnmanagedStorage(IArraySlice, Shape)`, which trusts its layout like NumPy's C API.
- An `order` argument is read by NumPy's `PyArray_OrderConverter` texts: `OrderResolver.Resolve` (A / K against a source
  array; without one, a creation function's `only 'C' or 'F' order is permitted`), `ResolveBufferOrder` (a caller's
  buffer: every order but F is C) and `Convert` (the parse-time conversion, for a function that decides later what each
  order means) raise the ValueError `order must be one of 'C', 'F', 'A', or 'K' (got 'Q')`, and accept every letter in
  either case (np.nditer included). `memoryview.tobytes` keeps CPython's rule (upper case only, None is C).
- Gates: `View/Shape.ForFreshAllocation.Tests.cs`, `View/Shape.Immutability.Tests.cs`, `View/Shape.CallerArrays.Tests.cs`,
  `Creation/FreshAllocation.ViewShape.Tests.cs`, `Creation/WrapConstructor.ViewShape.Tests.cs`,
  `RandomSampling/RandomSizeViewShape.Test.cs`, and the oracle replay `FuzzCorpusTests.RandomApiSizeLayouts` (every
  random overload that takes a `Shape`, its size handed over as a view's layout of the same dimensions).

`NDArray<T>.reshape(Shape)` / `reshape(ref Shape)` read only the dimensions of their argument, like the untyped
`NDArray.reshape(Shape)` (`ReshapeCore`): they pass `newShape.AsClean()` on, so a plain dimension list is kept as it is
and any other layout is laid out again in C order. `reshape_unsafe(Shape)` (typed and untyped) installs the Shape it is
given: it is the unchecked relabel broadcasting uses. The typed overloads copy a non-contiguous source where the untyped
route returns NumPy's strided view (`_attempt_nocopy_reshape`), a view/copy difference rather than an addressing error.
Gate: `Manipulation/TypedReshapeViewShapeTests.cs`.

## The buffer pool

- Buffers of one page or more (`SizeBucketedBufferPool.AlignedMinBytes` = 4096) start on a 64-byte boundary
  (`BufferAlignment`), whether fresh, zeroed, reused from a bucket or outside the pooled window. The C allocator aligns to
  16 bytes, which makes every second 32-byte vector load cross a cache line in half of all buffers; a kernel can align
  one stream with a prefix (`AlignmentPeel`), only the allocator aligns all of them.
- The pool allocates a block 80 bytes larger, hands out the first boundary a header past its start, and keeps the
  block's own pointer and the requested size in that header. A header rather than `NativeMemory.AlignedAlloc`: the zeroed
  path must stay `calloc` (lazily zeroed pages for a large `np.zeros`), and with the header both paths end in the same
  `NativeMemory.Free` on every operating system. Every Native block comes from the pool's `Take` / `TakeZeroed` (two
  sites in `UnmanagedMemoryBlock`).
- `Return(ptr, bytes)` must get the size the buffer was taken with: the size selects the bucket and tells whether the
  buffer has a header (a `Debug.Assert` compares it with the header's). Never free a pool pointer directly.
- A cache line, not a page: a page boundary would cost up to 4 KB a buffer and put every stream of a many-operand kernel
  (`np.evaluate`, `np.where`) in the same cache set at each step; the distance between operands' page offsets does not
  change kernel speed.
- Gates: `Backends/Unmanaged/PoolBufferAlignmentTests.cs`, and the unit suite in a Debug build (where the `Debug.Assert`
  on the returned size is live).
