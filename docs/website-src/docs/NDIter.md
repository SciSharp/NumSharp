# NDIter — kerneling your NDArray with IL generation

NumPy's `NpyIter` is one of the core schedulers behind ufuncs, reductions, broadcasting, and advanced iteration. It decides which axes to walk, which axes can coalesce, whether operands need buffers, and which pointer/stride window a typed inner loop receives. The iterator is C; many of the specialized loops it feeds are generated from NumPy's `.src` templates.

NumSharp reaches the same boundary with a stack-only `NDIterRef`, heap-allocated unmanaged state, and two kinds of generated kernel: **per-chunk** `NDInnerLoopFunc` kernels driven by the iterator, and **whole-array** `DirectILKernelGenerator` kernels that consume the iterator's post-coalesce shape/strides directly. Those are different contracts and use different caches. This page maps both, together with the managed `np.nditer` wrapper, typed C# extensions, buffering/casting/masking, production call sites, and the known parity gaps.

The implementation claims below were rechecked against the local NumPy source clone, live NumPy 2.4.3 probes, the NumSharp tests, and the current benchmark corpus on 2026-08-28. Read this page end-to-end if you're writing a new `np.*` function, porting a ufunc, or changing iterator scheduling.

## Table of Contents

- [Overview](#overview)
- [Public Iteration Surface](#public-iteration-surface)
  - [Typed iteration — `np.nditer<T>` / `np.nditer_chunks<T>`](#typed-iteration--unboxed-elements-and-span-chunks)
  - [Managed `np.nditer` contract](#managed-npnditer-contract)
- [What NDIter Is](#what-nditer-is)
  - [Subsystem map and evidence](#subsystem-map-and-evidence)
- [Divergences from NumPy](#divergences-from-numpy)
  - [Current known gaps](#current-known-gaps)
- [Iterator State](#iterator-state)
- [Construction](#construction)
- [Coalescing, Reordering, and Flipping](#coalescing-reordering-and-flipping)
- [Memory Overlap and COPY_IF_OVERLAP](#memory-overlap-and-copy_if_overlap)
- [Iteration Mechanics](#iteration-mechanics)
- [Buffering](#buffering)
- [Masked writes](#masked-writes)
- [Buffered Reduction: The Double Loop](#buffered-reduction-the-double-loop)
- [Kernel Integration Layer](#kernel-integration-layer)
  - [Quick reference](#quick-reference)
  - [Decision tree](#decision-tree)
  - [Measured behavior](#measured-behavior)
  - [Cache state — separate families and lifetimes](#cache-state--separate-families-and-lifetimes)
  - [Layer 1 — Canonical Inner-Loop API](#layer-1--canonical-inner-loop-api)
  - [Layer 2 — Struct-Generic Dispatch](#layer-2--struct-generic-dispatch)
  - [Layer 3 — Typed ufunc Dispatch](#layer-3--typed-ufunc-dispatch)
  - [Custom Operations (Tier 3A / 3B / 3C)](#custom-operations-tier-3a--3b--3c)
    - [Tier 3A — Raw IL](#tier-3a--raw-il)
    - [Tier 3B — Templated Inner Loop](#tier-3b--templated-inner-loop)
    - [Tier 3C — Expression DSL](#tier-3c--expression-dsl)
      - [Node catalog](#node-catalog)
      - [Operator overloads](#operator-overloads)
      - [Call — invoke supported .NET numeric methods](#call--invoke-supported-net-numeric-methods)
      - [Type discipline](#type-discipline)
      - [SIMD coverage rules](#simd-coverage-rules)
      - [Caching and auto-keys](#caching-and-auto-keys)
      - [Memory model and lifetime](#memory-model-and-lifetime)
      - [Validation and errors](#validation-and-errors)
      - [Behavioral notes](#behavioral-notes)
      - [Debugging compiled kernels](#debugging-compiled-kernels)
      - [When to use Tier 3C](#when-to-use-tier-3c)
- [Path Detection](#path-detection)
- [Production consumers and legacy surfaces](#production-consumers-and-legacy-surfaces)
- [Fused kernels in production](#fused-kernels-in-production)
  - [General expression fusion — `np.evaluate`](#general-expression-fusion--npevaluate)
  - [Measured fused versus unfused performance](#measured-fused-versus-unfused-performance)
  - [Specialized production fusions](#specialized-production-fusions)
- [Worked Examples](#worked-examples)
- [Performance](#performance)
  - [JIT warmup and measurement](#jit-warmup-and-measurement)
  - [Implementation Notes](#implementation-notes)
  - [When Does Each Layer Pay Off?](#when-does-each-layer-pay-off)
  - [Allocations](#allocations)
- [Summary](#summary)

---

## Overview

### What Is An Iterator?

An array is just a pointer plus a shape plus strides. Iterating "through" it means producing, one element (or chunk of elements) at a time, the byte offset into the buffer. For a contiguous row-major 3×4 array this is trivial — walk from 0 to 11 with stride 1. For a transposed view, a sliced view, a broadcasted view, or two arrays with mismatched strides, it is not.

`NDIterRef` takes that tangle and produces a deterministic schedule of pointer advances. A chunk driver can then use the canonical loop — `do { kernel(dataptrs, byteStrides, count); } while (iternext);` — while a whole-array helper can pass the post-coalesce `Shape` plus element-stride arrays to a DirectIL kernel.

### Why Build Our Own?

NumPy's iterator is a large C subsystem split across `nditer_constr.c`, `nditer_api.c`, `nditer_templ.c.src`, and the Python wrapper. NumSharp cannot copy that ABI, but it ports the same scheduling ideas: coalescing, reordering, negative-stride flipping, `ALLOCATE`, `COPY_IF_OVERLAP`, buffered casting, buffered reduction, C/F/A/K ordering, per-operand flags, and `op_axes` reduction encoding.

NumSharp.Core remains managed C#, but the iterator owns explicitly allocated unmanaged state (`NativeMemory.AllocZeroed`/`AlignedAlloc`) because its hot contract is raw pointers. `NDIter.Execution.cs` connects that schedule to chunk kernels and selected DirectIL whole-array helpers; it is not the only consumer of the iterator.

---

## Public Iteration Surface

`NDIterRef` is the kernel-author engine — the rest of this page is about it. Most application code never constructs one directly. The public, user-facing ways to walk an array are:

| You want | Use | Notes |
|----------|-----|-------|
| Every element of one array, C-order | `foreach (var x in ndarray)` or `ndarray.GetAtIndex(i)` | Resolves slices, strides, offset, and stride-0 broadcast; no intermediate copy. |
| Every element **unboxed / by reference** | `foreach (ref T x in np.nditer<T>(a))` | The fast path. See [Typed iteration](#typed-iteration--unboxed-elements-and-span-chunks). |
| A **`Span<T>` per inner loop**, to vectorize | `foreach (Span<T> c in np.nditer_chunks<T>(a))` | Hands the chunk straight to `TensorPrimitives` / `Vector<T>`. A contiguous array is one chunk. |
| NumPy's `nditer` object, with its full flag surface | `using var it = np.nditer(a, flags: …)` | `multi_index`, `external_loop`, `buffered`, `op_axes`, `itviews`, `copy()`, … — the parity surface. |
| Nested loops over disjoint axis groups | `var levels = np.nested_iters(a, axes)` | Returns linked `NDIterator` objects; advancing an outer level re-bases and resets its child. |
| `(index, value)` pairs in logical C-order | `foreach (var (idx, v) in np.ndenumerate(a))` | `np.ndenumerate<T>(a)` yields unboxed `T`. |
| Just the index space of a shape | `foreach (var idx in np.ndindex(3, 2))` | Pure odometer; no operands. |
| Each operand of a broadcast, flattened | `np.broadcast(a, b, …).iters[i]` — a `NDFlatIterator` | One flat C-order stream per operand, each stretched to the broadcast shape (e.g. `np.broadcast([1,2,3], [[10],[20]]).iters[0]` yields `1,2,3,1,2,3`). |
| The broadcast itself, as tuples | `foreach (object[] vals in np.broadcast(a, b, …))` | NumPy's `np.broadcast` object: one per-operand value tuple per step, with a live `.index` cursor, `.numiter`, `.size`, and `.reset()`. |

All of these use the same `Shape`/stride semantics, but they do **not** all use the same iterator implementation:

- `np.nditer`, `np.nditer<T>`, `np.nditer_chunks<T>`, and `np.nested_iters` are driven by `NDIterRef`.
- `np.ndindex` is a pure C-order odometer, matching modern NumPy's `itertools.product` implementation rather than routing through `nditer`.
- `np.ndenumerate` is a C-order coordinate walker plus `GetAtIndex`; `np.broadcast` and `NDFlatIterator` similarly use broadcast views plus logical flat access.

That separation is intentional: index-only and boxed convenience APIs should not pay for unmanaged multi-operand iterator state.

Like `NDIterRef`, `np.broadcast(...)` has **no explicit operand-count cap** — NumPy caps the multi-iterator at 64 (`NPY_MAXARGS`); NumSharp sizes per-operand structures dynamically (see [Divergences from NumPy](#divergences-from-numpy)).

### Typed iteration — unboxed elements and Span chunks

A NumSharp extension with no NumPy counterpart: Python has no unboxed generics, so `np.nditer`'s `it[0]` must hand back an array object. NumSharp's can hand back a **reference**.

```csharp
// element-wise, by reference — reads and writes go straight to the array
foreach (ref double x in np.nditer<double>(a, writeable: true))
    x *= 2;

// chunk-wise — one Span per inner loop, ready to vectorize
foreach (Span<double> c in np.nditer_chunks<double>(a, writeable: true))
    TensorPrimitives.Multiply(c, 2.0, c);
```

Both build an unbuffered, single-operand `EXTERNAL_LOOP` `NDIterRef`, so contiguous, F-order, transposed, reversed, broadcast, 0-d, and empty schedules come from the same engine. `np.nditer<T>` can consume any inner stride. `np.nditer_chunks<T>` is narrower: a `Span<T>` can describe only a unit-stride inner loop, so stepped inner axes are rejected before iteration.

The performance reason for these extensions is structural rather than a fixed multiplier: they remove the per-element `NDArray` alias view and boxing. For current numbers use the canonical [NDIter benchmark report](https://scisharp.github.io/NumSharp/docs/reports/nditer_results.html), not an isolated timing embedded in this page.

**Rules of the road:**

- **The dtype must match exactly.** A `ref` cannot convert, so `np.nditer<double>` over an `int32` array **throws** rather than reinterpreting the bytes. Cast first with `astype`.
- **Order is `'K'` (memory order)**, matching `np.nditer` exactly — on a reversed view `a[:, ::-1]` of `arange(6).reshape(2,3)` both libraries yield `0 1 2 3 4 5`, and both yield `2 1 0 5 4 3` under `order: 'C'`. `'K'` is also what lets reversed / F-contiguous / transposed views coalesce into a single chunk. Pass `order: 'C'` for the logical order `np.ndenumerate` uses.
- **The returned type is always mutable.** `writeable: true` asks the iterator to validate/open the operand as `READWRITE`, and it correctly rejects a broadcast view. It is **not** a C# read-only guarantee: both overloads still return `ref T` / `Span<T>` when `writeable` is false, and assigning through either currently mutates the array. Treat `writeable: false` as a caller contract, not enforcement; never assign through that reference (especially for stride-0 broadcast views).
- **`nditer_chunks<T>` needs a unit-stride inner loop.** A `Span<T>` is contiguous by definition, so a stepped view such as `a[":, ::2"]` is rejected up front — at `GetEnumerator`, never mid-loop. Use `np.nditer<T>`, which handles any stride, or iterate a `.copy()`.
- **An empty array iterates zero times**, where `np.nditer` raises `Iteration of zero-sized operands is not enabled` unless given NumPy's `zerosize_ok`. Deliberate: forcing `if (a.size > 0)` around every `foreach` is not how C# collections behave.
- **Re-enumeration restarts.** The value returned by `np.nditer<T>(…)` holds no unmanaged state; each `foreach` builds a fresh iterator. This is deliberately unlike the class-based `np.nditer`, which is its own iterator (NumPy's `iter(x) is x`) and therefore *resumes*.
- **Disposal is automatic under `foreach`** — C# disposes a `ref struct` enumerator through the same pattern that drives it, and that is what frees the iterator state. If you drive `MoveNext()` by hand, dispose it yourself.

Being `ref struct`s, the enumerators cannot escape to a field, a lambda or an `async` frame — the compiler enforces the lifetime the `ref`/`Span` needs. For fusing several operations into one pass instead of walking elements yourself, see [Tier 3C — Expression DSL](#tier-3c--expression-dsl) and `np.evaluate`.

### Managed `np.nditer` contract

The class returned by the non-generic `np.nditer(...)` overloads is `np.NDIterator`. It owns a detached `NDIterState*`, re-borrows a non-owning `NDIterRef` for each call, and must be closed or disposed. Prefer `using`; disposal flushes a pending buffer window, resolves `COPY_IF_OVERLAP`/`UPDATEIFCOPY` write-backs, and frees state.

| Behavior | NumSharp contract |
|----------|-------------------|
| Enumeration result | Always `NDArray[]`. A one-operand iterator yields `vals[0]`; NumPy yields a bare 0-d array in that case. |
| Cursor | One shared cursor. A second `foreach` resumes; call `reset()` to rewind. Typed iterators instead create a fresh cursor per enumeration. |
| Current values | 0-d alias views, or 1-d alias views under `external_loop`. Unbuffered aliases keep pointing at the published element; buffered aliases may be overwritten on the next refill. Copy a value you need to retain. |
| Null output operand | Defaults to `writeonly, allocate`; dtype is inferred from the non-null operands when omitted. |
| `copy()` | Copies the iterator at its current cursor and duplicates active buffers; the copy advances independently. |
| `remove_multi_index()` | Clears index tracking, reorders/coalesces, and resets the cursor to the start. |
| `iterrange` | Requires `ranged`; setting `[start,end)` repositions the iterator to `start` and stops at `end`. |
| `itviews` | Exposes the internal post-reorder/coalesce operand views only when unbuffered. See the scalar gap below. |
| delayed buffers | With `delay_bufalloc`, call `reset()` before reading `it[i]` or enumerating the managed wrapper. Kernel entry points on `NDIterRef` auto-materialize delayed buffers, but the managed indexer currently does not. |

`np.nested_iters` links two or more of these managed iterators. Outer levels drop `external_loop`/`buffered`; the innermost keeps them and receives `buffersize`. Each outer advance calls `ResetBasePointers` down the child chain, then rewinds the children. Dispose **every** returned level.

---

## What NDIter Is

The stack-only engine is `NDIterRef`, a `ref partial struct` in `NumSharp.Backends.Iteration`. `NDIter` is a different type: a static compatibility/helper class that owns the production copy/cast core. Keep the names distinct:

| Name | Lifetime and boundary |
|------|-----------------------|
| `np.NDIterator` | Managed public wrapper returned by `np.nditer`; owns detached unmanaged state and has a finalizer safety net. |
| `NDIterRef` | Stack-only low-level scheduler and kernel-author API; owns or borrows `NDIterState*`. |
| `NDIterState` | Blittable unmanaged state and dynamically sized pointer arrays. |
| `NDIter` | Static copy/cast/reduction helpers (`Copy`, `CopyAs`, `TryCopySameType`, legacy state factories). It is not an iterator instance. |
| `NDIterExecution` / `NDIterPathSelector` | Unused legacy scaffolding in `NDIterKernels.cs`; the current bridge uses `NDIterRef.ForEach`/`Execute*` and `DetectExecutionPath` instead. Do not build new code on it. |

The active low-level shape is:

```
NDIterRef (ref partial struct)                ← stack-only handle (5 partial files)
    ├── _state: NDIterState*                  ← heap-allocated unmanaged state
    ├── _operands: NDArray[]                   ← kept alive by GC root
    └── _cachedIterNext: NDIterNextFunc?      ← memoized iterate-advance delegate

NDIterState (unmanaged struct)                ← unmanaged state + dynamic blocks
    ├── Scalars: NDim, NOp, IterSize, IterIndex, ItFlags, ...
    ├── Dim arrays (size = NDim): Shape*, Coords*, Strides*, Perm*
    ├── Op arrays (size = NOp):   DataPtrs*, ResetDataPtrs*, BufStrides*,
    │                              InnerStrides*, BaseOffsets*, OpDTypes*, ...
    └── Reduction arrays:         ReduceOuterStrides*, ReduceOuterPtrs*,
                                  ArrayWritebackPtrs*, CoreSize, CorePos, ...
```

The handle is cheap to pass, but construction performs unmanaged allocation: one state object, one dimension-dependent block when `ndim > 0`, and one per-operand block. Buffering may add one aligned block per operand. `Dispose`/`FreeState` must release the matching ownership path.

### The Files

| File | What lives there |
|------|------------------|
| `NDIter.cs` | `NDIterRef` construction/configuration/advance/indexing/lifecycle plus the static `NDIter` copy/cast core (~4,600 lines) |
| `NDIter.State.cs` | `NDIterState`, the two dynamic blocks, accessors, `Advance`, `Reset`, `GotoIterIndex`, `BufferedReduceAdvance` |
| `NDIter.Detach.cs` | Ownership transfer/borrow bridge used by the managed `np.NDIterator` wrapper |
| `NDIter.Execution.cs` | Chunk drivers plus the separate whole-array DirectIL helper family (~970 lines) |
| `NDIterFlags.cs` | `NDIterFlags`, `NDIterOpFlags`, `NDIterGlobalFlags`, `NDIterPerOpFlags`, casting/order enums |
| `NDIterCoalescing.cs` | `CoalesceAxes`, `ReorderAxesForCoalescing`, `FlipNegativeStrides` |
| `NDIterCasting.cs` | Safe/same-kind/unsafe cast rules, `ConvertValue`, `FindCommonDtype` |
| `NDIterBufferManager.cs` | Aligned buffer allocation, copy-in/copy-out, `GROWINNER`, `BUF_REUSABLE` |
| `NDIter.Reduce.cs` | `NewReduce` — builds the reduction iterator (stride-ordered `op_axes`, stride-0 output axis) |
| `NDIter.Execution.Custom.cs` | Custom-op tiers: `ExecuteRawIL` (3A), `ExecuteElementWise` (3B), `ExecuteExpression` (3C) |
| `NDIterKernels.cs` | Legacy `INDIterKernel`/`NDIterExecutionPath` experiment; no current production or test call sites |
| `NDMemOverlap.cs` | Diophantine memory-overlap solver for `COPY_IF_OVERLAP` (port of NumPy's `mem_overlap.c`) |
| `NDExpr.cs`, `NDExpr.Typing.cs`, `NDExpr.Evaluate.cs` | Tier 3C / `np.evaluate` expression tree, per-node NEP50 typing, lowering to one pass |
| `NDScalarReductionKernels.cs`, `NDNanReductionKernels.cs` | Half/Complex + NaN-aware scalar (axis=None) reduction kernel structs |
| `NDAxisIter.cs`, `NDAxisIter.State.cs` | Single-axis reduction iterator for `var` / `std` / `cumsum` / `cumprod` |
| `NDLogicalReductionKernels.cs` | `All` / `Any` reduction kernel structs |

### Subsystem map and evidence

These stable area ids are useful when discussing iterator changes. Status labels mean: **operational** = implemented and a representative path was exercised; **partial** = meaningful implementation with a known boundary; **risk** = a reproduced correctness or lifetime hazard.

| Area | Status | Boundary | Primary evidence |
|------|--------|----------|------------------|
| `A1` Public surfaces | operational | `np.nditer`, typed iterators, `np.nested_iters`, plus the independent `ndindex`/`ndenumerate`/`broadcast` conveniences | `APIs/np.nditer*.cs`, `APIs/np.nested_iters.cs`, `Indexing/np.nd*.cs`, `Creation/np.broadcast.cs`; focused iterator run: 670/670 tests |
| `A2` Schedule and state | operational | broadcast shape, `op_axes`, order, permutation, flipping, coalescing, index/range tracking, ownership | `NDIter.cs`, `NDIter.State.cs`, `NDIterCoalescing.cs`, `NDIter.Detach.cs`; battle/parity/state tests |
| `A3` Transfer safety | partial / risk | casting, window buffers, masks, overlap copies, write-back | `NDIterCasting.cs`, `NDIterBufferManager.cs`, `NDMemOverlap.cs`; buffer/overlap/masked-write tests; known gaps below |
| `A4` Kernel execution | operational | per-chunk drivers, struct-generic reductions, custom compiled inner loops, whole-array DirectIL helpers | `NDIter.Execution*.cs`, `DirectILKernelGenerator.InnerLoop.cs`; custom-op/reduction/scan tests |
| `A5` Production consumers | operational | engine routing for elementwise, reductions, selection, fusion, sorting, byteswap, copy/fill/pad | `DefaultEngine.{BinaryOp,CompareOp,UnaryOp,Evaluate,ReductionOp}.cs`, `np.where.cs`, `np.diff.cs`, `NDArray.byteswap.cs`, `np.copyto.cs` |

The focused test run covers construction, ordering, indices, buffering, overlap, masks, custom kernels, public wrappers, reductions, scans, and nested iteration. The separate AuditV2 iterator class intentionally still exposes current gaps; one of its failures (`T1.2`) is a stale reflection check for an old method name—the live 20,005-element buffered probes completed as 20,005 scalar steps and three external-loop windows.

---

## Divergences from NumPy

NumPy's `nditer` has two hard-coded limits that NumSharp drops:

| Limit | NumPy | NumSharp |
|-------|-------|----------|
| `NPY_MAXDIMS` | 64 | no explicit construction cap; practical memory/stack limits remain |
| `NPY_MAXARGS` | 64 | no explicit construction cap; per-operand state is dynamic |

NumPy uses fixed-size workspaces and caps these public dimensions. NumSharp allocates its dimension and operand blocks from the actual `(ndim, nop)`, so construction has no explicit `NPY_MAXDIMS`/`NPY_MAXARGS` check. This is not literally infinite: memory and stack remain practical limits, and selected whole-array helpers still `stackalloc` stride arrays proportional to `ndim`.

Other deliberate differences:

- **Flag bit layout.** NumSharp reserves low bits 0-7 for the compatibility flags (`SourceBroadcast`, `SourceContiguous`, `DestinationContiguous`). NumPy-parity flags (`IDENTPERM`, `HASINDEX`, `REDUCE`, ...) sit at bits 8-15. Transfer flags pack into the top byte at shift 24. Semantics match NumPy; positions do not.
- **Two stride units.** NumPy's axis data is byte-strided. NumSharp keeps source-array axis strides in **elements** (`state.Strides`) and scales them with `SrcElementSizes`; `BufStrides` and the `NDInnerLoopFunc` contract are **bytes**. Whole-array DirectIL helpers receive element strides. Mixing these units is a correctness bug, not a performance detail.
- **No Python object support.** `REFS_OK`, garbage collection hooks, and `NDIter_GetBufferNeedsAPI` are no-ops. All cast routines are written assuming the data is plain unmanaged bytes.
- **Int64 indexing.** Iteration counters are `long`. This matches 64-bit NumPy's `npy_intp`; the difference appears on 32-bit hosts, where NumPy's index type is pointer-sized.
- **NumSharp-only internal flag.** `PARALLEL_SAFE` records a construction-time range-splitting judgment. It is metadata for planned/selected parallel drivers, not automatic parallel execution by `ForEach`.
- **Typed empty iteration.** `np.nditer<T>` and `np.nditer_chunks<T>` treat an empty array as an empty C# collection. The managed parity wrapper follows NumPy and requires `zerosize_ok`.
- **Delayed-buffer execution entry.** `NDIterRef.ForEach`/`ExecuteGeneric`/`ExecuteReducing` auto-materialize `DELAY_BUFALLOC`; NumPy requires an explicit reset. The managed wrapper does not share that guard—see the gap below.

### Current known gaps

These are observed current-state boundaries, not target behavior:

| Surface | NumPy 2.4.x | Current NumSharp | Safe practice |
|---------|-------------|------------------|---------------|
| Scalar `itviews` / `GetIterView(0)` | Returns a 0-d view | Indexes `original.flat[0]` and throws `IndexError` | Use `it[0]`/`value[0]`; do not request `itviews` for 0-d until fixed |
| `enable_external_loop()` after index tracking | Raises `ValueError` | Sets `EXLOOP`, leaving an illegal `HASINDEX`/`HASMULTIINDEX` combination | Choose `external_loop` at construction and never combine it with index flags |
| Managed `delay_bufalloc` read before `reset()` | Raises `ValueError` | `it[i]` can reinterpret source bytes as the requested buffer dtype (garbage) | Always call `reset()` before managed access; low-level kernel entry points auto-ensure |
| Typed `writeable: false` | No NumPy equivalent | Still returns mutable `ref T`/`Span<T>` and writes succeed | Treat it as read-only by convention; set `writeable: true` for intentional writes |
| `np.nested_iters(..., order: 'A')` without `multi_index` | Resolves each axis-group traversal from Any-order rules | Six of 48 probed 3-D axis partitions produced a different value stream; C/F/K matched in that matrix | Prefer explicit `C`, `F`, or `K`, or track `multi_index` when coordinates define the contract |
| `BUFFERED + REDUCE + WRITEMASKED` write-back | Supported for valid masks | Construction succeeds, write-back throws `NotSupportedException` | Use unbuffered masked reduction or separate the mask/reduction pass |
| Low-level `GetInnerLoopSizePtr()` on 0-d | C API callers use the scalar protocol | Direct call addresses `Shape[-1]`; managed/typed/`ForEach` paths guard it | For `NDim == 0`, use `count = 1` and stride `0` |
| `NDExpr.Call` signature dtypes | Python callables are dynamically typed | Supports 12 CLR numeric signatures; rejects `SByte`, `Half`, and `Complex` parameters/returns | Compose built-in nodes or use Tier 3B/3A for those types |

---

## Iterator State

A couple of fields deserve a closer look because every later section refers to them.

### Shape, Coords, Strides

```csharp
public long* Shape;             // [NDim]        — post-coalesce dimension sizes
public long* Coords;            // [NDim]        — current position, 0..Shape[d]
public long* Strides;           // [NOp * NDim]  — element stride per (op, axis)
public sbyte* Perm;             // [NDim]        — Perm[internal] = original_axis
                                //                 negative means axis was flipped
```

After coalescing, `NDim` can shrink. `StridesNDim` captures the stride allocation width so `GetStride(axis, op) = Strides[op * StridesNDim + axis]` still works.

`Perm[internal_axis] = original_axis` records how internal axes relate to the axes the caller passed in. If `FlipNegativeStrides` rewrote an axis, `Perm[d] = -1 - original_axis` encodes the flip. `GetMultiIndex` uses Perm to translate internal coords back into caller-space.

### DataPtrs vs ResetDataPtrs vs BaseOffsets

```csharp
public long* ResetDataPtrs;     // base pointer per operand; start of iteration
public long* BaseOffsets;       // byte accumulator from FlipNegativeStrides
public long* DataPtrs;          // live pointer; moves every Advance()
```

`Reset()` copies `ResetDataPtrs` into `DataPtrs`. When the iterator flips an axis it walks the data pointer to the end-of-axis first (since we'll iterate backwards in original memory, forwards in flipped-coord space) and records the byte delta in `BaseOffsets`. `ResetBasePointers(newPtrs)` lets the caller swap the array out while keeping the iteration schedule: new reset = new base + stored offset.

### Buffering Fields

```csharp
public long  BufferSize;        // elements per operand buffer (default 8192)
public long  BufIterEnd;        // how far into the buffer we're iterating
public long* Buffers;           // aligned-64 buffer pointer per operand (0 = no buffer)
public long* BufStrides;        // inner-loop stride per operand in bytes
                                //   == ElementSizes[op] for buffered operands
```

When buffering is active, an operand's `DataPtrs[op]` points into `Buffers[op]`, not into the original NDArray. The kernel sees a contiguous buffer at the buffer dtype; `NDIterBufferManager` handles the strided copy-in and copy-out.

### Reduction Fields (double-loop)

```csharp
public int  OuterDim;           // which internal axis is the reduce axis
public long CoreSize;           // elements per output slot (inner-loop length)
public long CorePos;            // position within core, 0..CoreSize
public long ReduceOuterSize;    // number of output slots in current buffer
public long ReducePos;          // position within outer loop

public long* ReduceOuterStrides;  // stride per op, advances to next output slot
public long* ReduceOuterPtrs;     // saved pointer at start of current output slot
public long* ArrayWritebackPtrs;  // array-space pointer for flushing output buffer
```

These only come into play when the iterator has both `BUFFER` and `REDUCE` flags. They're explained in detail in [Buffered Reduction: The Double Loop](#buffered-reduction-the-double-loop).

---

## Construction

Creating an iterator looks like this:

```csharp
using var iter = NDIterRef.MultiNew(
    nop: 3,
    op: new[] { a, b, out },
    flags: NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.BUFFERED,
    order: NPY_ORDER.NPY_KEEPORDER,
    casting: NPY_CASTING.NPY_SAFE_CASTING,
    opFlags: new[] {
        NDIterPerOpFlags.READONLY,
        NDIterPerOpFlags.READONLY,
        NDIterPerOpFlags.WRITEONLY },
    opDtypes: new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double });
```

Behind the scenes:

```
1. Pre-check WRITEMASKED/ARRAYMASK pairing    (state-free validation)
2. Resolve broadcast shape                    (ResolveReturnShape; respects op_axes)
3. If COPY_IF_OVERLAP: solve read/write overlap, replace hazardous writers
   with forced-copy temporaries, register write-back originals
4. Allocate null ALLOCATE/VIRTUAL operands with resolved dtype and shape
5. state.AllocateDimArrays(ndim, nop)         (dimension block + operand block)
6. Set MaskOp from ARRAYMASK flag
7. Find common dtype if COMMON_DTYPE
8. For each operand:
      - SetOpSrcDType (array dtype)
      - SetOpDType (buffer dtype; equals array dtype when not casting)
      - Translate NDIterPerOpFlags → NDIterOpFlags
      - Mark CAST / FORCECOPY / HAS_WRITEBACK as needed
      - Compute strides (respecting op_axes or broadcast)
      - Set data pointer = arr.Address + offset * elemSize
      - Mark SourceBroadcast or validate/mark a stretched WRITE as REDUCE
9. Require BUFFERED when iteration dtype differs; validate casting rule
10. Apply op_axes reduction flags (implicit + REDUCTION_AXIS-encoded axes)
11. Validate WRITEMASKED reduction strides against ARRAYMASK
12. FlipNegativeStrides (K-order only; skipped for C/F/A)
13. Reorder; coalesce only when index tracking and requested order allow it;
    otherwise remove unit axes where safe
14. Set EXLOOP, GROWINNER, HASMULTIINDEX, HASINDEX; initialize flat index
15. Refresh inner-stride and contiguity caches
16. If BUFFERED: allocate/prime a window, or set DELAYBUF
17. If BUFFERED + REDUCE: allocate and configure the legacy double loop
18. If IterSize <= 1: set ONEITERATION
19. Derive PARALLEL_SAFE metadata from reduction/write-overlap state
```

The result is a state machine ready to produce pointers.

### The Flag Families

There are four mostly-disjoint flag enums. A quick reference:

**`NDIterGlobalFlags` — passed at construction, affect the whole iterator.**

| Flag | Meaning |
|------|---------|
| `C_INDEX`, `F_INDEX` | Track a flat index in C or F order |
| `MULTI_INDEX` | Track per-dim coords (needed for `GetMultiIndex`) |
| `EXTERNAL_LOOP` | Caller handles inner dim — iterator returns inner-dim-sized chunks |
| `COMMON_DTYPE` | Find common dtype across all operands and cast to it |
| `REDUCE_OK` | Allow reduction operands (needed for axis reductions) |
| `BUFFERED` | Enable operand buffering (required with cross-type casting) |
| `GROWINNER` | Make inner loop as large as possible within buffer |
| `DELAY_BUFALLOC` | Defer allocation. Call `reset()` before managed `NDIterator` access; low-level kernel drivers auto-ensure. |
| `DONT_NEGATE_STRIDES` | Suppress `FlipNegativeStrides` |
| `COPY_IF_OVERLAP` | Copy operand if it overlaps another in memory |
| `RANGED` | Iterator covers a sub-range |

**`NDIterPerOpFlags` — passed per operand, affect just that operand.**

| Flag | Meaning |
|------|---------|
| `READONLY`, `WRITEONLY`, `READWRITE` | Direction |
| `COPY`, `UPDATEIFCOPY` | Force copy / update on dealloc |
| `ALLOCATE` | `op[i]` is null — iterator allocates using `opDtypes[i]` |
| `CONTIG` | Require contiguous view (may force buffering) |
| `NO_BROADCAST` | Error if this operand would need to broadcast |
| `WRITEMASKED`, `ARRAYMASK` | Writemask pair for masked writes |
| `VIRTUAL` | Null temporary operand; currently materialized as a real array, matching NumPy's observable behavior |
| `OVERLAP_ASSUME_ELEMENTWISE_PER_OP` | Lets exact same-layout aliases avoid a `COPY_IF_OVERLAP` temporary when elementwise access is guaranteed |

**`NDIterFlags` — internal state, set/cleared during iteration.** (`IDENTPERM`, `NEGPERM`, `HASINDEX`, `BUFFER`, `REDUCE`, `ONEITERATION`, etc.) These flow from construction decisions.

**`NDIterOpFlags` — per-operand internal state.** (`READ`, `WRITE`, `CAST`, `REDUCE`, `VIRTUAL`, `WRITEMASKED`, `BUF_REUSABLE`, `CONTIG`.)

---

## Coalescing, Reordering, and Flipping

The single biggest performance lever the iterator has is **reducing NDim**. A 3-D contiguous array should iterate in one flat loop, not in three nested ones.

### Coalescing Rule

Two adjacent axes `d` and `d+1` can merge if, for **every** operand:

```
stride[op][d] * shape[d] == stride[op][d+1]
```

...or either axis is size 1 with stride 0 (broadcast pass-through). When that holds, the pair is collapsed: the new shape is `shape[d] * shape[d+1]`, the new stride is `stride[op][d]` (the inner one).

A contiguous 2×3×4 float32 array has strides `[12, 4, 1]` in elements. The coalescing check succeeds at both boundaries, and `CoalesceAxes` reduces NDim from 3 to 1 with shape 24 and stride 1. One flat SIMD loop, exactly.

### Reordering

Coalescing only works if adjacent axes are *already* stride-ordered. `ReorderAxesForCoalescing` sorts axes by minimum absolute stride (smallest innermost) when the requested order allows it:

```
C-order:  last axis innermost (no reorder — identity perm)
F-order:  first axis innermost (reverse axes)
K-order:  smallest stride innermost (insertion sort by stride)
A-order:  behaves like K-order
```

For K-order on a non-contiguous broadcast array, stride-based sorting produces the wrong iteration order, so the iterator falls back to C-order. This guard rail lives in the construction logic around `effectiveOrder`.

### Negative-Stride Flipping

`FlipNegativeStrides` only runs under K-order (not C/F/A — those are "forced orders" that preserve logical iteration direction). For each axis where *all* operands have zero or negative strides, the iterator:

1. Negates the stride.
2. Accumulates `(shape[d] - 1) * old_stride * elem_size` into `BaseOffsets[op]`.
3. Marks the axis flipped via `Perm[d] = (sbyte)(-1 - Perm[d])`.

The effect: a reversed slice still iterates contiguous memory in ascending order, which the SIMD kernels can chew on. Later, `GetMultiIndex` decodes the flip so the caller sees original coordinates.

### Interaction with MULTI_INDEX and HASINDEX

If `MULTI_INDEX` is set we **reorder but don't coalesce** — coalescing would lose the mapping from internal to original axes. Same for `C_INDEX`/`F_INDEX`, which need original axis structure to compute the flat index.

---

## Memory Overlap and COPY_IF_OVERLAP

When an output operand shares memory with an input — `np.add(a, a, out=a)`, `b[1:] += b[:-1]`, a transposed view written back over its source — a naive iterator can clobber input elements before they're read, producing undefined results. NumPy guards this with `NPY_ITER_COPY_IF_OVERLAP`; NumSharp ports the guard, solver and all.

### The flag's contract

Pass `NDIterGlobalFlags.COPY_IF_OVERLAP` and, for each **write** operand, the constructor checks it against every **read** operand. If they may share memory, the write operand is replaced with a fresh C-order temporary; the kernel writes into the temp, and the temp is copied back over the original when the iterator is disposed. Read operands are never copied — only the writers that would corrupt a reader. This runs in `Initialize` *before* the ALLOCATE step, so an iterator-allocated output can never be made to overlap an input.

### The overlap solver (`NDMemOverlap.cs`)

"May these two views share memory?" is not the bounding-box question it looks like. `a[::2]` and `a[1::2]` occupy the *same* address range yet never touch the same byte. The exact test is a bounded Diophantine equation — the two arrays overlap iff

```
Σ strideₐ·xₐ − Σ strideᵦ·xᵦ = baseᵦ − baseₐ      (0 ≤ x[i] < shape[i])
```

has an integer solution. `NDMemOverlap` is a faithful port of NumPy's `numpy/_core/src/common/mem_overlap.c` (Pauli Virtanen's solver), with `System.Int128` standing in for NumPy's `npy_extint128.h`:

1. **Bounds quick-check.** `GetMemoryExtents` computes each array's half-open `[start, end)` byte range. Disjoint ranges → `No` overlap, no further work (this is NumPy's `MAY_SHARE_BOUNDS`, i.e. `maxWork == 0`).
2. **Diophantine solve.** Overlapping bounds fall through to `SolveDiophantine` (an extended-Euclid GCD chain plus a depth-first bounded search), which decides whether a *real* shared element exists. The search is **work-capped** by `maxWork`: if the cap is hit the result is `TooHard`, which callers conservatively treat as "may share."

`SolveMayShareMemory(a, b, maxWork)` answers the two-array question; `SolveMayHaveInternalOverlap(a, maxWork)` answers "does one array alias itself?" (a stride-0 dim over a non-unit axis). The result enum is `{ No, Yes, TooHard, Overflow }` — anything but `No` means "copy to be safe."

### Resolution and writeback

`ComputeCopyIfOverlap` runs the write-vs-read matrix with `maxWork = 1` — exactly NumPy's budget. Operands flagged for copy go through `MaterializeForcedCopies`: each is swapped for a C-order temp (copied in first if it's also read), tagged `FORCECOPY | HAS_WRITEBACK`, and the original recorded in `_writebackOriginals`. On `Dispose()`, `ResolveWritebacks` copies each temp back over its original — but only *after* `FlushBufferWindow`, so buffered writes reach the temp before the temp lands in the user's array. Get that ordering wrong and a buffered in-place op silently drops its last window.

### The elementwise short-circuit

Forcing a copy for *every* in-place `out=` would be wasteful — `np.add(a, a, out=a)` is perfectly safe because the inner loop reads each element before it writes the same slot. The per-operand flag `OVERLAP_ASSUME_ELEMENTWISE` encodes that promise: when both operands carry it, are the same dtype, and have identical base/shape/strides with no internal self-overlap, `AssumeElementwiseExactAlias` skips the copy. Every ufunc `out=` path sets this flag, so the common in-place case pays nothing.

This is also the link to `PARALLEL_SAFE`: the same forced-copy machinery that makes a single in-place writer correct is what certifies that writer for parallel range-splitting (see [Divergences from NumPy](#divergences-from-numpy)).

---

## Iteration Mechanics

`GetIterNext()` returns the advance function for the current flag set. For unbuffered iteration there are three:

| Flavor | Picked when | Behavior |
|--------|-------------|----------|
| `SingleIterationNext` | `ONEITERATION` | One shot, done |
| `ExternalLoopNext` | `EXLOOP` | Advance *outer* coords only; inner dim is the caller's problem |
| `StandardNext` | otherwise | Full ripple-carry advance, one element at a time |

Buffered iteration adds two more: `BufferedNext` (windowed, with `EXTERNAL_LOOP`) jumps a whole buffer fill per call, and `BufferedElementNext` (without `EXTERNAL_LOOP`) walks the buffer one element at a time, refilling at the window edge. A buffered reduction uses the double-loop advance instead — see [Buffering](#buffering) and [Buffered Reduction](#buffered-reduction-the-double-loop).

`state.Advance()` is the ripple-carry primitive. For each axis from innermost to outermost:

```
for axis in (NDim-1 ... 0):
    coord[axis]++
    if coord[axis] < shape[axis]:
        dataptrs[op] += stride[op][axis] * elem_size[op]   for every op
        return
    // carry: reset this axis
    coord[axis] = 0
    dataptrs[op] -= stride[op][axis] * (shape[axis] - 1) * elem_size[op]
// fell through: iteration complete
```

Straightforward, but note the rewind on carry: when axis 2 wraps, we subtract `stride*(shape-1)*size` so the pointer lands back at the axis-2 start, then axis 1 will add one stride. The net effect is identical to `dataptr = base + sum(coord[d] * stride[d][op]) * size`, but computed incrementally.

### GetInnerLoopSizePtr()

Ideally the inner loop processes many elements per `iternext` call. The iterator exposes this via:

```csharp
long* size = iter.GetInnerLoopSizePtr();
```

- Windowed buffered (`BUFFER`, no `REDUCE`): returns `&state.BufTransferSize` — the element count of the current buffer fill (NumPy's `NBF_SIZE`).
- Buffered reduction (`BUFFER + REDUCE`): returns `&state.BufIterEnd`, which the double-loop uses as the inner-loop count.
- Otherwise: returns `&state.Shape[NDim-1]` (the innermost dimension size).

With `EXTERNAL_LOOP` set and the array coalesced to 1-D, one `iternext` call returns the entire array size — a single kernel invocation processes everything.

Two low-level traps matter to kernel authors:

- `GetInnerLoopSizePtr()` is not scalar-safe by itself; `NDim == 0` would address `Shape[-1]`. `ForEach`, the managed indexer, and both typed iterators special-case the scalar as `(count=1, stride=0)`.
- `GetInnerStrideArray()` exposes internal **element** strides. The `NDInnerLoopFunc` contract is **byte** strides. Let `ForEach`/`ExecuteGeneric` do the conversion, or multiply by the correct source element size yourself. `BufStrides` are already bytes.

---

## Buffering

Buffering solves two problems:

1. **Casting.** If the caller wants to see doubles but the NDArray is int32, the iterator gathers/converts into a double buffer. Writable operands convert/scatter back at window flush or disposal.
2. **A non-linear iteration walk.** When an operand cannot be represented as one arithmetic progression through the whole post-coalesce iteration space, a tight window simplifies the kernel contract.

`NDIterBufferManager.AllocateBuffers` allocates 64-byte-aligned blocks (AVX-512-friendly) per operand that needs buffering. Default buffer size is 8192 elements; this can be tuned per call.

`BUFFERED` does **not** imply that every operand gets a buffer. An operand stays direct (`BUFNEVER`-style) when it needs no cast/`CONTIG` requirement and its positions form one linear progression: contiguous arrays, constant-stride 1-D views, and fully broadcast scalars qualify. Its `BufStrides[op]` records the true inner **byte** stride. Only cast/`CONTIG`/reduction/non-linear operands get an aligned block and a tight element-size `BufStride`.

```
CONTIG-requested strided array          aligned 64-byte buffer (size ≤ 8192)
┌─────┬─────┬─────┬─────┐               ┌──┬──┬──┬──┬──┬──┬──┐
│ a[0]│  ?  │  ?  │  ?  │  CopyToBuffer │a0│a5│a10│...         │
│  ?  │  ?  │  ?  │ a[5]│    ────────▶  └──┴──┴──┴──┴──┴──┴──┘
│  ?  │  ?  │a[10]│  ?  │                   ^
│     ...            │                      DataPtrs[op] points here
└─────────────────────┘                     BufStrides[op] = sizeof(T)
```

Once a buffered operand is filled, its `DataPtrs[op]` points into the tight block; an unbuffered operand's pointer remains in its array. At a window boundary `FlushBufferWindow` scatters writable buffers back through the original strides (and mask, when present), `GotoIterIndex` repositions array pointers, and `FillBufferWindow` gathers the next window. `Reset`, final `iternext`, `ForEach`, and disposal all contain flush paths so a last partial window is not lost.

### GROWINNER

`GROWINNER` requests that the inner loop grow to consume as much of the buffer as possible per kernel call — more work per invocation, less loop overhead. In NumSharp the inner-loop length is set by axis coalescing together with the buffer-window size: a contiguous array coalesces to a single axis (a 5×6 contiguous array becomes one length-30 axis), so the inner loop already spans the whole contiguous run, and the buffer window is sized identically with or without the flag. The flag is carried for NumPy API parity.

### BUF_REUSABLE

For reductions, the same input block may be read multiple times (e.g. `mean` when accumulator type differs). The `BUF_REUSABLE` flag tells the iterator "the buffer contents are still valid, skip the copy." `CopyToBufferIfNeeded` honors it.

---

## Masked writes

NumPy's ufunc `where=` machinery is represented by one `ARRAYMASK` operand and one or more `WRITEMASKED` outputs. Construction enforces the pairing: exactly one mask, at least one write-masked operand, the mask is read-only, and the mask cannot itself be write-masked. For a reduction, the mask stride must not vary along an axis where the output is pinned.

The execution path depends on buffering:

1. `ForEach`/Tier 3B recognizes a trailing bool/`Byte` mask and decomposes every chunk into mask-true runs. It calls the ordinary unmasked kernel only for those runs, so dense masks keep SIMD-sized stretches.
2. Buffered non-reduction output uses the same run decomposition in `CopyWindowFromBufferMasked`; false slots are never scattered back.
3. A stride-0 mask gates the entire chunk/window from one byte.
4. Buffered masks must be `Boolean` or `Byte`. Other mask dtypes keep the plain low-level contract, where the custom kernel is responsible for masking.
5. Buffered **reduction** write-back is the remaining gap: it throws rather than silently corrupt false-mask slots.

---

## Buffered Reduction: The Double Loop

When you do `np.sum(a, axis=0)` on a 2-D array, the output has one fewer axis than the input. The iterator must visit every input but accumulate into a fixed output position while the reduction axis is scanned. The efficient way to do this with buffering is NumPy's **double loop**:

```
CoreSize    = length of reduce axis              ("how many inputs per output")
ReduceOuterSize = other-axes length fitted into buffer   ("how many output slots")

For each buffer fill:
    for outer in 0..ReduceOuterSize:              ← advance ReduceOuterPtrs by ReduceOuterStrides
        for core in 0..CoreSize:                  ← advance DataPtrs by BufStrides
            kernel(dataptrs, bufstrides, 1)       ← accumulate into output
        // reset inner, move outer pointer to next output slot
```

The trick: reduce operands have `BufStrides[op] = 0`, so inside the core loop their pointer stays pinned. The kernel keeps adding into the same output slot until the reduce axis is exhausted; the outer loop then moves to the next output slot.

`NDIterState.BufferedReduceAdvance()` returns:
- `1` — more elements in current buffer (inner or outer)
- `0` — buffer exhausted, caller must refill
- `-1` — iteration complete, caller must flush

The bridge's `BufferedReduce` method drives this explicitly.

### IsFirstVisit

Reduction kernels must initialize the output before accumulating. `iter.IsFirstVisit(op)` returns `true` only when every reduction-axis coordinate is zero *and* `CorePos == 0` in buffered mode. Kernels check this once at each output slot to emit identity-write semantics:

```csharp
if (iter.IsFirstVisit(reduceOp)) *(double*)ptrs[reduceOp] = 0.0;
*(double*)ptrs[reduceOp] += *(double*)ptrs[inputOp];
```

---

## Kernel Integration Layer

Everything up to this point describes scheduling. `NDIter.Execution.cs` exposes **three execution shapes** over that state. They are peers, not one seven-level call stack:

```
                         NDIterState
             (DataPtrs, Shape, strides, buffers)
                    /              |              \
                   /               |               \
  chunk delegate driver     struct-generic driver    whole-array DirectIL helpers
  ForEach(NDInnerLoopFunc)   ExecuteGeneric/Reducing  ExecuteBinary/Unary/...
           ▲                         |                        |
           |                         |                        |
  Tier 3A / 3B / 3C          JIT-specialized struct     family-specific cache
  compile NDInnerLoopFunc     no delegate indirection   direct delegate invocation
```

- **Chunk delegate driver.** `ForEach` owns delayed-buffer materialization, window advancement, byte-stride conversion, and bool/byte mask-run decomposition. Custom Tier 3A/B/C kernels compile to `NDInnerLoopFunc` and enter here.
- **Struct-generic driver.** `ExecuteGeneric` and `ExecuteReducing` repeat the scheduling logic without a delegate. The JIT specializes the concrete kernel struct; reducing kernels can early-exit.
- **Whole-array helpers.** `ExecuteBinary`, `ExecuteUnary`, `ExecuteReduction`, `ExecuteComparison`, `ExecuteScan`, and `ExecuteCopy` prepare post-coalesce element-stride arrays and call family-specific `DirectILKernelGenerator` delegates directly. Except for the buffered binary/copy cases, these helpers require a C-contiguous output because their legacy kernels ignore output strides. They do **not** funnel through `ForEach`.

### Quick reference

| Entry point | Contract | Use it when |
|-------------|----------|-------------|
| `ForEach(NDInnerLoopFunc)` | Chunk/window driver; byte strides; optional mask wrapper | You already have a reusable inner-loop delegate or need `auxdata`/a closure |
| `ExecuteGeneric<TKernel>` / `ExecuteReducing<TKernel,TAccum>` | Same schedule, struct call; early exit on reducing form | A hot custom kernel should avoid delegate dispatch |
| `ExecuteElementWiseUnary/Binary/Ternary` | Tier 3B compiler → cached chunk delegate → `ForEach` | Production elementwise routing over arbitrary inner strides; this is what `DefaultEngine` commonly uses |
| `ExecuteExpression(NDExpr)` | Tier 3C expression → Tier 3B compiler → `ForEach` | A fused expression fits the DSL and avoids temporaries |
| `ExecuteRawIL` | Tier 3A caller-authored `NDInnerLoopFunc` IL → `ForEach` | The loop body/auxdata cannot be expressed by Tier 3B/C |
| `ExecuteBinary` / `Unary` / `Comparison` / `Scan` / `Copy` | Whole-array DirectIL helper | The exact legacy helper contract fits, especially a contiguous output; do not assume it is the universal ufunc route |
| `ExecuteReduction<TResult>` | Whole-array typed reduction helper | A one-operand scalar reduction; this has a current production call site |
| `BufferedReduce<K,T>` | Legacy `BUFFER+REDUCE` double-loop driver | Specialized internal experiments only; no current production call site |

### Decision tree

```
Does an existing DefaultEngine/np.* route already implement this operation?
  yes → extend that route and its kernel family; preserve its layout/dtype gates.
  no ↓

Is it a one-operand scalar reduction handled by ExecuteReduction<TResult>?
  yes → use that whole-array helper and its family-specific cache.
  no ↓

Can I express it as a tree of DSL nodes (Add, Sqrt, Where, Exp, …)?
  yes → Tier 3C → cached inner-loop delegate → ForEach.
  no ↓

Is the missing piece a supported CLR method signature?
  yes → Tier 3C + Call (scalar-only; not SByte/Half/Complex signatures).
  no ↓

Is the computation rectangular and element-wise?
  yes → Tier 3B. Supply scalar IL and optional vector IL; the factory wraps strides/tails.
  no ↓

Need a custom emitted inner loop or auxdata?
  yes → Tier 3A.
  no ↓

Do I need an early-exit reduction (Any / All / find-first)?
  yes → ExecuteReducing. Return false from the kernel to bail out.
  no ↓

Already have a chunk delegate / closure?
       → ForEach.
```

### Measured behavior

The authoritative source is the generated [NDIter benchmark report](https://scisharp.github.io/NumSharp/docs/reports/nditer_results.html). Its 2026-08-28 snapshot measures matched NumSharp/NumPy kernels across scalar, 1K, 100K, 1M, and 10M sizes:

| Signal | Current snapshot | Interpretation |
|--------|------------------|----------------|
| 165-cell operation matrix | 1.49× NumPy÷NumSharp geomean; 114 wins / 51 losses | Overall useful, not a universal win |
| Iterator construction (9 configs) | 2.74× geomean; 9/9 wins | Dynamic state setup is competitive despite unmanaged allocation |
| Strided-row chunk width 4 | 0.44× | Per-chunk dispatch dominates very narrow loops |
| Strided-row chunk width 16 | 1.01× | Break-even is workload/host dependent |
| Early-hit `any` | about 24× at 100K–10M | Struct reducing early exit avoids the full scan |
| 10M `sqrt` | 0.23× | A prominent size-specific loss; never quote only the geomean |

The harness uses Release builds, warm pilots, time-bound best-of rounds, matched case ids, and correctness checks. Re-run it when a kernel, buffer policy, or route threshold changes; do not preserve isolated microprobe numbers in prose.

### Cache state — separate families and lifetimes

There is no single cache shared by every entry point:

- Tier 3A/B/C share `DirectILKernelGenerator._innerLoopCache`, keyed by string. `GeneratedDelegates.InnerLoopCount` observes only that cache.
- Whole-array helpers use family-specific caches such as `_mixedTypeCache`, `_unaryCache`, `_comparisonCache`, `_elementReductionCache`, `_scanCache`, and `_copyKernelCache`.
- `ExecuteGeneric`/`ExecuteReducing` rely on normal JIT generic specialization, not a DynamicMethod dictionary.
- A caller-supplied `ForEach` delegate is not inserted into a generator cache by `ForEach`.
- `DelegateSlots` separately roots bound targets/captured delegates used by `NDExpr.Call`; those registrations live for the process.

The public observability for the custom inner-loop/Call lifetimes is:

```csharp
int customInnerLoops = GeneratedDelegates.InnerLoopCount;
int callSlots        = DelegateSlots.RegisteredCount;

GeneratedDelegates.ClearInnerLoop();  // internal, test-only
DelegateSlots.Clear();                 // test-only; clear with the kernel cache
```

Every new captured delegate instance gets a new slot even if an explicit kernel key reuses compiled IL. Keep delegates/targets in stable fields and construct reusable expression trees once; see [Memory model and lifetime](#memory-model-and-lifetime).

### Layer 1 — Canonical Inner-Loop API

This is the NumPy-in-C pattern. You hand the iterator a function pointer (a delegate in C#), and it runs the canonical loop:

```csharp
public void ForEach(NDInnerLoopFunc kernel, void* auxdata = null);

public unsafe delegate void NDInnerLoopFunc(
    void** dataptrs, long* strides, long count, void* auxdata);
```

One call per *inner loop*, not per element. The iterator decides what "inner loop" means:

| Scenario | Call count | Count per call |
|----------|-----------|----------------|
| Fully coalesced + contiguous, with `EXTERNAL_LOOP` | 1 | `IterSize` |
| Non-coalesced with `EXTERNAL_LOOP` | outer product | `Shape[NDim-1]` |
| Buffered non-reduction | `ceil(IterSize / BufferSize)` | current `BufTransferSize` |
| Buffered reduction | double-loop dependent | legacy `BufIterEnd` contract |
| Neither `EXTERNAL_LOOP` nor `BUFFERED` | `IterSize` | 1 |

The strides passed to the kernel are always in **bytes** — the bridge converts from element strides for the non-buffered path. This matches NumPy's convention and makes the kernel body identical whether or not the iterator is buffering.

**Performance note.** Post-tier-1 the JIT autovectorizes both byte-pointer and typed-pointer loops into Vector256. To get there faster and to keep the fast path as simple as possible, branch on stride at the top and drop to typed pointers:

```csharp
using var iter = NDIterRef.MultiNew(3, new[] { a, b, c },
    NDIterGlobalFlags.EXTERNAL_LOOP,
    NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_NO_CASTING,
    new[] { NDIterPerOpFlags.READONLY,
            NDIterPerOpFlags.READONLY,
            NDIterPerOpFlags.WRITEONLY });

iter.ForEach((ptrs, strides, count, _) => {
    // Fast branch: contiguous, element stride == sizeof(float).
    // The JIT autovectorizes this to Vector256 sqrt.
    if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
        float* a = (float*)ptrs[0], b = (float*)ptrs[1], c = (float*)ptrs[2];
        for (long i = 0; i < count; i++)
            c[i] = MathF.Sqrt(a[i] * a[i] + b[i] * b[i]);
        return;
    }
    // Slow branch: strided / broadcast. Correct but scalar.
    long sA = strides[0], sB = strides[1], sC = strides[2];
    byte* pA = (byte*)ptrs[0]; byte* pB = (byte*)ptrs[1]; byte* pC = (byte*)ptrs[2];
    for (long i = 0; i < count; i++) {
        float av = *(float*)(pA + i * sA);
        float bv = *(float*)(pB + i * sB);
        *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
    }
});
```

Use this when you already own a reusable inner-loop delegate, need `auxdata`, or want closure ergonomics while exploring. `ForEach` itself does not allocate a delegate; a capturing lambda at the call site does.

### Layer 2 — Struct-Generic Dispatch

Delegates have an indirect call. For hot inner loops, that hurts. Layer 2 trades a delegate for a struct type parameter:

```csharp
public interface INDInnerLoop
{
    void Execute(void** dataptrs, long* strides, long count);
}

public interface INDReducingInnerLoop<TAccum> where TAccum : unmanaged
{
    bool Execute(void** dataptrs, long* strides, long count, ref TAccum accumulator);
}

public void ExecuteGeneric<TKernel>(TKernel kernel)
    where TKernel : struct, INDInnerLoop;

public TAccum ExecuteReducing<TKernel, TAccum>(TKernel kernel, TAccum init)
    where TKernel : struct, INDReducingInnerLoop<TAccum>
    where TAccum : unmanaged;
```

Because `TKernel` is constrained to `struct`, the JIT specializes one copy of `ExecuteGeneric` per struct type at codegen time and inlines `kernel.Execute(...)` at the call site. No vtable, no delegate, no boxing. It's the closest managed C# gets to C++ templates.

The bridge splits `ExecuteGeneric` internally so the single-inner-loop case (the common case: coalesced contig + `EXTERNAL_LOOP`, `ONEITERATION`, or buffered-fits-in-one-fill) goes through `ExecuteGenericSingle` — a tiny `[AggressiveInlining]` method with one `kernel.Execute` call and no `do/while`. That's what lets the JIT autovectorize the kernel's body. The multi-loop path keeps the canonical `do { kernel.Execute(...); } while (iternext); ` driver.

```csharp
readonly unsafe struct HypotKernel : INDInnerLoop
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Execute(void** p, long* s, long n)
    {
        // Fast branch — typed pointers so the JIT autovectorizes.
        if (s[0] == 4 && s[1] == 4 && s[2] == 4) {
            float* a = (float*)p[0], b = (float*)p[1], c = (float*)p[2];
            for (long i = 0; i < n; i++)
                c[i] = MathF.Sqrt(a[i] * a[i] + b[i] * b[i]);
            return;
        }
        // Slow branch — any stride, scalar.
        long sA = s[0], sB = s[1], sC = s[2];
        byte* pA = (byte*)p[0]; byte* pB = (byte*)p[1]; byte* pC = (byte*)p[2];
        for (long i = 0; i < n; i++) {
            float av = *(float*)(pA + i * sA);
            float bv = *(float*)(pB + i * sB);
            *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
        }
    }
}

iter.ExecuteGeneric(default(HypotKernel));  // zero-alloc, inlined
```

For early-exit reductions, the kernel returns `false` to abort:

```csharp
readonly unsafe struct AnyNonZero : INDReducingInnerLoop<bool>
{
    public bool Execute(void** p, long* s, long n, ref bool acc)
    {
        long st = s[0]; byte* pt = (byte*)p[0];
        for (long i = 0; i < n; i++)
            if (*(int*)(pt + i * st) != 0) { acc = true; return false; }  // stop
        return true;
    }
}

bool found = iter.ExecuteReducing<AnyNonZero, bool>(default, false);
```

On a 1M-element array with a non-zero near the start, this returns after one kernel call.

### Layer 3 — Typed ufunc Dispatch

This section describes the bridge's **whole-array helper family**. It is useful, but it is not the universal production ufunc layer: current `DefaultEngine` elementwise routes commonly build `EXTERNAL_LOOP` iterators and call Tier 3B `ExecuteElementWise*`. Of the helpers below, `ExecuteReduction<TResult>` has a direct current engine call site; several others are available low-level APIs with test coverage but no production callers.

```csharp
public void ExecuteBinary(BinaryOp op);       // [in0, in1, out]
public void ExecuteUnary(UnaryOp op);          // [in, out]
public void ExecuteComparison(ComparisonOp);   // [in0, in1, bool out]
public TResult ExecuteReduction<TResult>(ReductionOp op);  // [in] → T
public void ExecuteScan(ReductionOp op);       // [in, out]
public void ExecuteCopy();                     // [src, dst]
public void BufferedReduce<K, T>(K kernel);    // explicit BUFFER+REDUCE double-loop
```

Under the hood the non-buffered whole-array helpers do four things:

1. **Validate.** Throw if operand count or flags are wrong.
2. **Detect path.** Scan operand strides, pick `SimdFull` / `SimdScalarRight` / `SimdScalarLeft` / `SimdChunk` / `General`.
3. **Prepare args.** `stackalloc` one stride array per operand, fill with element strides, grab `_state->Shape` and data pointers.
4. **Invoke.** A family-specific `DirectILKernelGenerator.Get*Kernel(key)` — cache hit returns its delegate, cache miss emits IL and stores it in that family's dictionary.

The helpers bridge to legacy whole-array delegates whose output is implicitly dense. `RequireContiguousOutput` therefore rejects strided outputs for unary/comparison/scan and non-buffered binary. Use Tier 3B/`ForEach` when every operand stride must be honored. Buffered `ExecuteBinary` is the exception: `RunBufferedBinary` consumes windows and lets buffer write-back honor the real destination strides.

### Custom Operations (Tier 3A / 3B / 3C)

The enum-driven whole-array helpers cover a closed set of binary, unary, comparison, reduction, scan, and copy shapes. Production routes also need masking, arbitrary output strides, mixed-type emitters, and fused expressions; those are reasons to use the custom inner-loop family even for an operation whose enum already exists.

The Custom Operations extension lets the bridge **IL-generate a stride-aware rectangular elementwise kernel** while preserving the Tier 3B 4×-unrolled SIMD shell when the supplied dtype/vector body permits it. Three tiers trade control for convenience:

```
              ┌─────────────────── You provide ────────────────────┐
 Tier 3A       │  the entire inner-loop IL body                     │  Maximum control
 Tier 3B       │  per-element scalar + (optional) vector IL body    │  Shared unroll shell
 Tier 3C       │  an expression tree (NDExpr)                      │  No IL required
              └────────────────────────────────────────────────────┘
                           │
                           ▼
               ILKernelGenerator.CompileInnerLoop  (new partial)
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Contig SIMD path      Scalar strided path
          (4× unroll + V256      (per-element, stride-aware
           + 1-vec remainder      pointer walk)
           + scalar tail)
                 └─────────┬─────────┘
                           ▼
                  NDInnerLoopFunc delegate (cached)
                           │
                           ▼
                 NDIterRef.ForEach → do { kernel(...); } while (iternext)
```

All three tiers produce the same delegate shape (`NDInnerLoopFunc`) and funnel through `ForEach`. The factory emits a runtime contig check at the top of the kernel: if every operand's byte stride equals its element size, take the SIMD path; otherwise fall into the scalar-strided loop. Cache keys are user-supplied strings; Tier 3C derives a structural signature automatically if you don't provide one.

| Method on `NDIterRef` | Tier | What you supply |
|------------------------|------|------------------|
| `ExecuteRawIL(emit, key, aux)` | A | `Action<ILGenerator>` — the entire method, including `ret` |
| `ExecuteElementWise(operandTypes, scalarBody, vectorBody, key)` | B | Two `Action<ILGenerator>` — per-element scalar and vector |
| `ExecuteElementWiseUnary/Binary/Ternary(...)` | B | Typed convenience overloads |
| `ExecuteExpression(expr, inputTypes, outputType, key?)` | C | An `NDExpr` tree |

#### Tier 3A — Raw IL

You emit everything. Arguments are the canonical inner-loop shape: `arg0 = void** dataptrs`, `arg1 = long* byteStrides`, `arg2 = long count`, `arg3 = void* auxdata`. Your body must emit its own `ret`. Cached by the string key you pass — same key returns the same compiled delegate.

```csharp
iter.ExecuteRawIL(il =>
{
    // Pull out pointers and strides once.
    var p0 = il.DeclareLocal(typeof(byte*));
    var p1 = il.DeclareLocal(typeof(byte*));
    var p2 = il.DeclareLocal(typeof(byte*));
    // ... load dataptrs[0..2], strides[0..2] ...

    // for (i = 0; i < count; i++) *p2 = *p0 + *p1
    var i = il.DeclareLocal(typeof(long));
    il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);

    var top = il.DefineLabel(); var end = il.DefineLabel();
    il.MarkLabel(top);
    il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, end);
    // compute p2[i*s2] = p0[i*s0] + p1[i*s1]
    // ...
    il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
    il.Emit(OpCodes.Br, top);
    il.MarkLabel(end);
    il.Emit(OpCodes.Ret);
}, cacheKey: "my_int32_add");
```

Use when: you need a loop shape the templated shell can't express (gather, scatter, cross-element dependencies, non-rectangular write patterns).

#### Tier 3B — Templated Inner Loop

Supply only the per-element work; the factory wraps it in the standard 4×-unrolled SIMD + 1-vector remainder + scalar tail + scalar-strided fallback. The two `Action<ILGenerator>` callbacks are stack-based:

- **`scalarBody`** — on entry, stack holds N input scalars in order (operand 0 deepest, operand N-1 on top); on exit, stack must hold one value of the output dtype.
- **`vectorBody`** — same contract but with `Vector{W}<T>` values. Optional — pass `null` for scalar-only. If non-null **and** all operand dtypes are identical **and** the type is SIMD-capable, the factory emits the fast path.

```csharp
// out = a*b + 1 on 16 float32s, fused in one pass.
iter.ExecuteElementWiseBinary(
    NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
    scalarBody: il =>
    {
        // Stack: [a, b] -> [a*b + 1]
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Ldc_R4, 1.0f);
        il.Emit(OpCodes.Add);
    },
    vectorBody: il =>
    {
        // Stack: [va, vb] -> [va*vb + 1]
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);
        il.Emit(OpCodes.Ldc_R4, 1.0f);
        ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
    },
    cacheKey: "fma_f32_c1");
```

The `ILKernelGenerator.Emit*` helpers (`EmitVectorOperation`, `EmitVectorCreate`, `EmitVectorLoad`, `EmitVectorStore`, `EmitScalarOperation`, `EmitConvertTo`, `EmitLoadIndirect`, `EmitStoreIndirect`, `EmitUnaryScalarOperation`, `EmitUnaryVectorOperation`) are exposed as `internal` so you can compose primitives without reinventing IL emission. The same helpers power the baked `ExecuteBinary`/`ExecuteUnary` kernels.

Convenience overloads exist for common arities:

```csharp
iter.ExecuteElementWiseUnary(inType, outType, scalarBody, vectorBody, key);
iter.ExecuteElementWiseBinary(lhs, rhs, outType, scalarBody, vectorBody, key);
iter.ExecuteElementWiseTernary(a, b, c, outType, scalarBody, vectorBody, key);
```

For arity > 3 or variable operand counts, use the array form `ExecuteElementWise(NPTypeCode[], ...)`.

**When SIMD is skipped.** The factory emits the vector path only when `CanSimdAllOperands(operandTypes)` returns true — every operand's dtype must be identical and SIMD-capable (i.e. not `Boolean`, `Char`, or `Decimal`). If either condition fails, only the scalar path is emitted. Mixed-type ufuncs (e.g. `int32 + float32 → float32`) use the scalar path with the user's `EmitConvertTo` inside the body.

**Contig runtime check.** The kernel's first act is to compare each operand's stride with its element size. If any differ, control jumps to the scalar-strided loop — inner-axis iteration that advances pointers by their declared byte strides. This means a single kernel handles both contiguous and sliced inputs without recompiling.

Use when: you want SIMD + 4× unrolling for a fused or non-standard op but don't want to hand-roll the whole loop.

#### Tier 3C — Expression DSL

The expression DSL lets you compose ops with C# operator syntax, and `Compile()` emits the IL for you. No `ILGenerator` exposure in your code.

```csharp
// out = sqrt(a² + b²)
var expr = NDExpr.Sqrt(NDExpr.Square(NDExpr.Input(0)) +
                        NDExpr.Square(NDExpr.Input(1)));

iter.ExecuteExpression(expr,
    inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
    outputType: NPTypeCode.Single);
```

##### Node catalog

**Leaves.**

| Factory | Semantics | NumPy |
|---------|-----------|-------|
| `NDExpr.Input(i)` | Reference operand `i` (0-based input index). Auto-converts to output dtype on load. | — |
| `NDExpr.Const(value)` | Literal — `int / long / float / double` overloads. Emitted at the output dtype. | — |

**Binary arithmetic.**

| Factory | Operator | SIMD | NumPy equivalent | Notes |
|---------|----------|:----:|------------------|-------|
| `Add(a,b)` | `a + b` | ✓ | `np.add` | |
| `Subtract(a,b)` | `a - b` | ✓ | `np.subtract` | |
| `Multiply(a,b)` | `a * b` | ✓ | `np.multiply` | |
| `Divide(a,b)` | `a / b` | ✓ | `np.divide` | True-division for floats; integer division for ints. |
| `Mod(a,b)` | `a % b` | — | `np.mod` | Floored modulo — result sign follows divisor (like Python `%`), unlike C# `%` which truncates toward zero. |
| `Power(a,b)` | — | — | `np.power` | Routed through `Math.Pow(double, double)`; integer operands are promoted to double and the result converted back. |
| `FloorDivide(a,b)` | — | — | `np.floor_divide` | Floor toward negative infinity. For signed int operands, correctly returns `-4` (not `-3`) for `-10 // 3`. |
| `ATan2(y,x)` | — | — | `np.arctan2` | Four-quadrant arctan via `Math.Atan2`. |

**Binary bitwise.** Integer types only; floating-point operands are a compile-time IL emission error.

| Factory | Operator | SIMD | NumPy equivalent |
|---------|----------|:----:|------------------|
| `BitwiseAnd(a,b)` | `a & b` | ✓ | `np.bitwise_and` |
| `BitwiseOr(a,b)` | `a \| b` | ✓ | `np.bitwise_or` |
| `BitwiseXor(a,b)` | `a ^ b` | ✓ | `np.bitwise_xor` |

**Scalar-branchy combinators** (scalar path only).

| Factory | Semantics | NumPy equivalent |
|---------|-----------|------------------|
| `Min(a,b)` | Delegates to `Math.Min` — NaN-propagating per IEEE 754. | `np.minimum` (**not** `np.fmin`) |
| `Max(a,b)` | Delegates to `Math.Max` — NaN-propagating per IEEE 754. | `np.maximum` (**not** `np.fmax`) |
| `Clamp(x,lo,hi)` | `Min(Max(x,lo),hi)` — sugar, shares the compiled kernel structure with the underlying pair. | `np.clip` |
| `Where(cond,a,b)` | Branchy ternary select: if `cond != 0` return `a` else `b`. `cond` is evaluated in the output dtype, so floats, integers, and decimals all work uniformly. | `np.where` (with eager eval of both branches) |

> `Where`'s branches are **both emitted** into the kernel but only the taken one runs per element — the `brfalse` branches past the untaken side. If one side is much more expensive (e.g. `Exp`), the cost is only paid on elements where it's selected, making `Where` a real optimization over `cond * a + (1-cond) * b` for expensive alternatives.

**Unary — arithmetic.**

| Factory | Operator | SIMD | NumPy equivalent |
|---------|----------|:----:|------------------|
| `Negate(x)` | unary `-x` | ✓ | `np.negative` |
| `Abs(x)` | — | ✓ | `np.abs` / `np.absolute` |
| `Sqrt(x)` | — | ✓ | `np.sqrt` |
| `Square(x)` | — | ✓ | `np.square` |
| `Reciprocal(x)` | — | ✓ | `np.reciprocal` |
| `Cbrt(x)` | — | — | `np.cbrt` |
| `Sign(x)` | — | — | `np.sign` |

**Unary — exp / log.** All route through `Math.Exp / Log / ...` (or `MathF` for `Single`); integer inputs are auto-promoted to double around the call and cast back at the end.

| Factory | Semantics | SIMD | NumPy equivalent |
|---------|-----------|:----:|------------------|
| `Exp(x)` | eˣ | — | `np.exp` |
| `Exp2(x)` | 2ˣ | — | `np.exp2` |
| `Expm1(x)` | eˣ − 1 (accurate for small x) | — | `np.expm1` |
| `Log(x)` | ln x | — | `np.log` |
| `Log2(x)` | log₂ x | — | `np.log2` |
| `Log10(x)` | log₁₀ x | — | `np.log10` |
| `Log1p(x)` | ln(1 + x) (accurate for small x) | — | `np.log1p` |

**Unary — trigonometric.**

| Factory | Semantics | SIMD | NumPy equivalent |
|---------|-----------|:----:|------------------|
| `Sin(x)`, `Cos(x)`, `Tan(x)` | Standard trig | — | `np.sin / cos / tan` |
| `Sinh(x)`, `Cosh(x)`, `Tanh(x)` | Hyperbolic | — | `np.sinh / cosh / tanh` |
| `ASin(x)`, `ACos(x)`, `ATan(x)` | Inverse | — | `np.arcsin / arccos / arctan` |
| `Deg2Rad(x)` | x · π/180 | ✓ | `np.deg2rad` / `np.radians` |
| `Rad2Deg(x)` | x · 180/π | ✓ | `np.rad2deg` / `np.degrees` |

**Unary — rounding.**

| Factory | Semantics | SIMD | NumPy equivalent |
|---------|-----------|:----:|------------------|
| `Floor(x)` | ⌊x⌋ | ✓ | `np.floor` |
| `Ceil(x)` | ⌈x⌉ | ✓ | `np.ceil` |
| `Round(x)` | Banker's rounding (half-to-even) | — | `np.rint` (matches NumPy's half-to-even default) |
| `Truncate(x)` | Toward zero | — | `np.trunc` |

> `Round` and `Truncate` have a working SIMD path on .NET 9+, but NumSharp's library targets .NET 8 as well, where `Vector256.Round/Truncate` don't exist. NDExpr gates them to the scalar path unconditionally so the compiled kernel works on both frameworks. Other contiguous rounding ops autovectorize after tier-1 JIT promotion.

**Unary — bitwise / logical / predicates.**

| Factory | Operator | SIMD | NumPy equivalent | Notes |
|---------|----------|:----:|------------------|-------|
| `BitwiseNot(x)` | `~x` | ✓ | `np.invert` / `np.bitwise_not` | Integer types only. |
| `LogicalNot(x)` | `!x` | — | `np.logical_not` | Returns 1 if `x == 0` else 0. Routes through `EmitComparisonOperation(Equal, outType)` — correct for all dtypes including Int64, Single, Double, Decimal (see [Behavioral notes](#behavioral-notes)). |
| `IsNaN(x)` | — | — | `np.isnan` | Returns 0/1 at output dtype. For integer types: always 0. |
| `IsFinite(x)` | — | — | `np.isfinite` | Returns 0/1 at output dtype. For integer types: always 1. |
| `IsInf(x)` | — | — | `np.isinf` | Returns 0/1 at output dtype. For integer types: always 0. |

**Comparisons** (produce numeric 0 or 1 at output dtype; scalar path only).

| Factory | Semantics | NumPy equivalent |
|---------|-----------|------------------|
| `Equal(a,b)` | `a == b` | `np.equal` |
| `NotEqual(a,b)` | `a != b` | `np.not_equal` |
| `Less(a,b)` | `a < b` | `np.less` |
| `LessEqual(a,b)` | `a <= b` | `np.less_equal` |
| `Greater(a,b)` | `a > b` | `np.greater` |
| `GreaterEqual(a,b)` | `a >= b` | `np.greater_equal` |

Unlike NumPy's comparison ufuncs (which return `bool` arrays), Tier 3C's single-output-dtype model collapses comparisons to `0 or 1` at the output dtype. This composes cleanly with arithmetic — e.g. ReLU becomes `(x > 0) * x`.

NaN semantics match IEEE 754: any comparison involving NaN produces 0 (false). `NaN == NaN → 0`, `NaN < 5 → 0`, `NaN >= 5 → 0`. To test for NaN, use `IsNaN(x)`.

**Call — invoke a supported .NET numeric method.** The escape hatch for math not in the node catalog. Scalar path only; signatures are limited to the 12 CLR dtypes listed below.

| Factory | Semantics |
|---------|-----------|
| `Call<T1…Tn, TR>(Func<T1…Tn, TR> f, NDExpr a1, …)` | Typed generic overloads for arity 0–4. Accept method groups without cast (`NDExpr.Call(Math.Sqrt, x)`, `NDExpr.Call(Math.Pow, x, y)`). |
| `Call(Delegate func, params NDExpr[] args)` | Catch-all for pre-constructed delegates. Use when the arity exceeds 4 or when the typed overload is ambiguous. |
| `Call(MethodInfo staticMethod, params NDExpr[] args)` | Invoke a reflection-obtained static method. |
| `Call(MethodInfo instanceMethod, object target, params NDExpr[] args)` | Invoke a reflection-obtained instance method against `target`. |

See [Call — invoke supported .NET numeric methods](#call--invoke-supported-net-numeric-methods) below for dispatch paths, auto-conversion rules, supported signatures, performance envelope, and overload-disambiguation guidance.

##### Operator overloads

An expression tree reads like ordinary C#:

```csharp
// (a + b) * c + 1
var linear = (NDExpr.Input(0) + NDExpr.Input(1)) * NDExpr.Input(2) + NDExpr.Const(1.0f);

// ReLU via comparison × input
var relu = NDExpr.Greater(NDExpr.Input(0), NDExpr.Const(0.0f)) * NDExpr.Input(0);

// Clamp with no named method call
var clamped = NDExpr.Min(NDExpr.Max(NDExpr.Input(0), NDExpr.Const(0f)), NDExpr.Const(1f));
```

Overloads: `+ - * /` (arithmetic), `%` (NumPy mod), `& | ^` (bitwise), unary `-` (negate), `~` (bitwise not), `!` (logical not). No overloads for `<`, `>`, `==`, `!=` (those need to return `bool` in C#, which would collide with `object.Equals` and similar) — use the factory methods (`Less`, `Greater`, `Equal`, `NotEqual`, `LessEqual`, `GreaterEqual`) for comparisons.

##### Call — invoke supported .NET numeric methods

The DSL's built-in catalog covers most element-wise math. `Call` is the escape hatch for supported numeric user-defined activations, BCL helpers without a dedicated node (for example `Math.BitDecrement`/`Math.CopySign`), plugin methods discovered through reflection, and captured-state logic. It trades SIMD for flexibility, but it is not arbitrary managed interop.

Supported parameter and return types are `Boolean`, `Byte`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Char`, `Single`, `Double`, and `Decimal`. `SByte`, `Half`, and `Complex` are currently rejected when the `CallNode` is constructed, as are `void`, reference types, by-ref parameters, and unrelated structs. Built-in `NDExpr` nodes cover more dtypes than `Call`; use Tier 3B/3A when the missing signature matters.

**One node, four factory shapes, three dispatch paths.** All four factories construct the same `CallNode`; the node inspects its input and picks the cheapest dispatch at construction:

```
                      ┌─────────────────────────┐
  NDExpr.Call(...)   │       CallNode          │
  ─────────────▶      │  _kind ∈ {              │
                      │    StaticMethod,        │ ← call <methodinfo>
                      │    BoundTarget,         │ ← load target, callvirt
                      │    Delegate             │ ← load delegate, Invoke
                      │  }                      │
                      └─────────────────────────┘
```

**Path A — static methods (zero indirection).**

```csharp
// Func<T...> overload: compiler infers delegate signature, no cast needed
// for non-overloaded methods.
NDExpr.Call(Math.Sqrt,   NDExpr.Input(0));
NDExpr.Call(Math.Pow,    NDExpr.Input(0), NDExpr.Input(1));
NDExpr.Call(MathF.Tanh,  NDExpr.Input(0));

// MethodInfo overload: useful when reflecting.
var mi = typeof(Math).GetMethod("BitIncrement", new[] { typeof(double) });
NDExpr.Call(mi, NDExpr.Input(0));
```

Emit: one `call <methodinfo>` opcode after the arguments are pushed. The JIT may inline the target when it's small and visible. No DelegateSlots entry, no runtime lookup. This is the fast path and is what you get automatically whenever the delegate has no captured state.

**Path B — bound instance methods (one indirection).**

```csharp
class Activations
{
    public double Temperature { get; set; }
    public double Softmax(double x) => Math.Exp(x / Temperature);
}

var inst = new Activations { Temperature = 1.5 };
var mi = typeof(Activations).GetMethod("Softmax");

NDExpr.Call(mi, inst, NDExpr.Input(0));
```

Emit: the kernel first loads the target object from a process-wide `DelegateSlots` registry by integer ID, casts it to the method's declaring type, pushes the arguments, then `callvirt <methodinfo>`. Cost is one dictionary lookup (~5 ns) plus a virtual call. The target object's state is live — mutations are visible to subsequent kernel invocations.

**Path C — captured delegates (one indirection).**

```csharp
// Works uniformly for lambdas with captures, instance-method-bound delegates,
// or any pre-constructed Delegate instance.
Func<double, double> swish = x => x / (1.0 + Math.Exp(-x));
NDExpr.Call(swish, NDExpr.Input(0));

// Pre-constructed delegate with explicit type (no method-group cast needed here).
Delegate d = swish;
NDExpr.Call(d, NDExpr.Input(0));
```

Emit: the kernel loads the delegate from `DelegateSlots`, casts it to its concrete runtime type (e.g. `Func<double, double>`), pushes arguments, then `callvirt Invoke`. Same ~5-10 ns overhead as Path B, plus the `Delegate.Invoke` dispatch stub (single virtual call).

**Auto-conversion at the call boundary.**

The node respects the DSL's single-output-dtype invariant:

```
        ctx.OutputType        param dtype       return dtype      ctx.OutputType
  ┌──────────────────┐  ┌─────────────┐   ┌───────────────┐   ┌──────────────────┐
  │ args evaluated   │─▶│ convert via │──▶│ method runs   │──▶│ convert via      │
  │ in outputType    │  │ EmitConvertTo│   │               │   │ EmitConvertTo    │
  └──────────────────┘  └─────────────┘   └───────────────┘   └──────────────────┘
```

So `NDExpr.Call(Math.Sqrt, Input(0))` with an `Int32` input and a `Double` output works end-to-end: the int gets loaded, converted to double at `InputNode`, arrives at the call as double (no further conversion needed for a `Double` param), `Math.Sqrt` runs, the double return flows out to the `Double` output slot. Flip the output dtype to `Single` and you'd get an extra `Conv_R4` after the call.

**Overload disambiguation.**

Non-overloaded static methods bind to the typed `Func<...>` overload via method-group conversion — no cast needed:

```csharp
NDExpr.Call(Math.Sqrt, x);        // ✓ Func<double,double>
NDExpr.Call(Math.Cbrt, x);        // ✓ same
NDExpr.Call(MathF.Tanh, x);       // ✓ Func<float,float>
NDExpr.Call(Math.Pow, x, y);      // ✓ Func<double,double,double>
```

Methods with multiple overloads (same name, different signatures) need a cast to disambiguate which one you want:

```csharp
// ERROR: 'Math.Abs' has 9 overloads.
// NDExpr.Call(Math.Abs, x);
//                ^^^^^^^^
// CS0121: The call is ambiguous between ...

// Cast to the concrete Func<...> you want:
NDExpr.Call((Func<double, double>)Math.Abs, x);       // ✓ picks Math.Abs(double)
NDExpr.Call((Func<float, float>)MathF.Abs,  x);       // ✓ picks MathF.Abs(float)
NDExpr.Call((Func<long, long>)Math.Abs,     x);       // ✓ picks Math.Abs(long)
```

Alternatively, use the `MethodInfo` overload to pick by signature explicitly:

```csharp
var mi = typeof(Math).GetMethod(nameof(Math.Abs), new[] { typeof(double) });
NDExpr.Call(mi, x);  // unambiguous — the MethodInfo is already picked
```

**Thread safety.**

`DelegateSlots` registration uses `Interlocked.Increment` for ID generation and `ConcurrentDictionary` for storage, so concurrent `Call` construction publishes safe slot ids. The inner-loop cache also publishes safely with `ConcurrentDictionary.GetOrAdd`, but its value factory is allowed to run more than once under contention; multiple equivalent DynamicMethods may be compiled transiently and only one is retained. Once published, kernels are re-entrant because invocation reads schedule pointers plus immutable slot references.

**Performance envelope.**

Per-element cost of the three paths, measured against a built-in DSL op on a post-warmup 1M-element double array:

| Path | Relative to built-in Sqrt | Notes |
|------|--------------------------|-------|
| Static method (Path A) | ~1.5× slower | One managed call per element; JIT may inline small targets |
| Bound instance (Path B) | ~2-3× slower | Dict lookup + castclass + virtual call |
| Captured delegate (Path C) | ~2-4× slower | Same lookup + castclass + `Delegate.Invoke` stub |

These ratios assume the user's method does comparable arithmetic to `Math.Sqrt`. If your target does substantially more work (e.g. three `Math.Exp` calls), the ratio collapses toward 1 — the call overhead becomes negligible compared to the math.

##### Type discipline

Every intermediate value flows through the output dtype: `Input(i)` loads the i-th operand's dtype and auto-converts (via `EmitConvertTo`) to the output dtype; constants are emitted directly in the output dtype. This **single-type intermediate invariant** keeps the DSL simple — you don't need to reason about mixed-type arithmetic inside the tree.

**Concrete example — integer to float promotion.**

```csharp
// Input is int32, output is float64. The DSL handles the promotion automatically.
var a = np.array(new int[] { 1, 4, 9, 16, 25 });
var r = np.empty(new Shape(5), np.float64);

using var iter = NDIterRef.MultiNew(2, new[] { a, r }, ...);
iter.ExecuteExpression(NDExpr.Sqrt(NDExpr.Input(0)),
    inputTypes: new[] { NPTypeCode.Int32 }, outputType: NPTypeCode.Double);
// r = [1.0, 2.0, 3.0, 4.0, 5.0]
```

What the emitted IL does per element: load `int32`, `Conv_R8` (promote to double), call `Math.Sqrt(double)`, store `double`. The conversion is emitted at the `Input` node, not at the `Sqrt` node — all subsequent operations see the output-dtype value.

**SIMD gate.** The vector path is enabled only when **every** input dtype equals the output dtype (so a single `Vector<T>` instantiation covers the whole tree) **and every node in the tree has a SIMD emit**. If any node lacks a SIMD path, the whole compilation falls back to scalar — correctness preserved, but no 4× unroll. For mixed-dtype work you're in the scalar-strided fallback regardless.

##### SIMD coverage rules

A node's `SupportsSimd` determines whether Tier 3C emits the vector body:

- **Yes:** `Input`, `Const`, the four arithmetic binary ops (`+ - * /`), the three bitwise binary ops (`& | ^`), and the unary ops `Negate`, `Abs`, `Sqrt`, `Floor`, `Ceil`, `Square`, `Reciprocal`, `Deg2Rad`, `Rad2Deg`, `BitwiseNot`.
- **No:** `Mod`, `Power`, `FloorDivide`, `ATan2`, `Min`/`Max`/`Clamp`/`Where`, all comparisons, `Round`, `Truncate` (no net8 SIMD method), all trig (except `Deg2Rad`/`Rad2Deg`), all log/exp, `Sign`, `Cbrt`, `LogicalNot`, predicates (`IsNaN`/`IsFinite`/`IsInf`), `Call` (user methods are always scalar — there is no vectorization path for arbitrary managed calls).

**Predicate / LogicalNot result handling.** Predicates (`IsNaN`/`IsFinite`/`IsInf`) and `LogicalNot` emit an I4 0/1 on the stack, not a value of the output dtype. `UnaryNode` detects these ops and inserts a trailing `EmitConvertTo(Int32, outType)` so the factory's final `Stind` matches. `LogicalNot` in particular routes through `EmitComparisonOperation(Equal, outType)` with an output-dtype zero literal: the `Ldc_I4_0 + Ceq` sequence is correct only for I4-width values (bool/byte/int16/int32), so the typed-zero comparison is what makes the result correct for Int64, Single, Double, and Decimal too.

A tree's `SupportsSimd` is true only if **every** node in it does. One unsupported node demotes the whole tree to scalar-only — which is usually still autovectorized by the JIT after tier-1 promotion, just without the 4× unroll.

##### Caching and auto-keys

Pass `cacheKey` to share the compiled delegate across iterators; omit it and the compiler auto-derives one from the tree's structural signature plus input/output dtypes. Actual examples (verified against `NDExpr.AppendSignature`):

```
NDExpr:Add(Multiply(In[0],Const[2]),Const[3]):in=Double:out=Double
NDExpr:Sqrt(Add(Square(In[0]),Square(In[1]))):in=Single,Single:out=Single
NDExpr:Where(CmpGreater(In[0],Const[0]),In[0],Multiply(Const[0.1],In[0])):in=Double:out=Double
NDExpr:Min(In[0],In[1]):in=Int32,Int32:out=Int32
NDExpr:IsNan(In[0]):in=Double:out=Double
NDExpr:LogicalNot(In[0]):in=Double:out=Double
NDExpr:BitwiseNot(In[0]):in=Int32:out=Int32
NDExpr:Mod(In[0],Const[3]):in=Double:out=Double
NDExpr:Sqrt(In[0]):in=Int32:out=Double           ← int input, double output
NDExpr:Call[System.Math.Sqrt#100663308@<guid>](In[0]):in=Double:out=Double
NDExpr:Call[MyApp.Activations.Swish#167772171@<guid>,target#7](In[0]):in=Double:out=Double
```

Enum names appear verbatim (e.g. `Multiply`, not `Mul`; `IsNan`, not `IsNaN` — the enum is spelled `IsNan`).

Two trees with identical structure and types get the same auto-derived key and share a cached kernel. Each node class contributes a distinct signature prefix:

| Node class | Signature fragment |
|------------|--------------------|
| `InputNode` | `In[i]` |
| `ConstNode` | `Const[value]` (integer form if constructed from int/long; decimal form for float/double) |
| `BinaryNode` | `<BinaryOp>(L,R)` (e.g. `Add(...)`, `Mod(...)`, `ATan2(...)`) |
| `UnaryNode` | `<UnaryOp>(C)` (e.g. `Sqrt(...)`, `IsNan(...)`, `BitwiseNot(...)`) |
| `ComparisonNode` | `Cmp<Op>(L,R)` (e.g. `CmpEqual(...)`, `CmpGreater(...)`) |
| `MinMaxNode` | `Min(L,R)` or `Max(L,R)` |
| `WhereNode` | `Where(C,A,B)` |
| `CallNode` | `Call[<declaringType>.<name>#<metadataToken>@<moduleGuid>](args)` — for instance methods, additionally `,target#<slotId>` |

> **Constant value sensitivity.** Two trees that differ only in a constant value (e.g. `x + 1` vs `x + 2`) generate distinct keys — the constant is part of the signature, because it's baked into the emitted IL. If you need many kernels parameterized by a scalar, consider passing the scalar as a second input operand (as a 0-d `NDArray` or a broadcast view) rather than a compile-time constant.
>
> **Integer/float const collision.** `NDExpr.Const(1)` and `NDExpr.Const(1.0)` both serialize to `Const[1]` when the `double` value is whole. With the same output dtype they produce identical IL, so sharing a cache entry is correct. If you need to distinguish — say, to force a specific integer vs float constant interpretation — construct both trees separately and supply an explicit `cacheKey`.

##### Memory model and lifetime

Three things outlive a single Tier 3C call. Knowing what they are and how long they live explains the steady-state memory profile of a Tier 3C-heavy program.

**1. Compiled kernels (`_innerLoopCache`).**

Every unique `(structural signature, inputTypes, outputType)` triple retains one `DynamicMethod` in a process-wide `ConcurrentDictionary<string, NDInnerLoopFunc>` keyed by the cache-key string. Under first-use contention more than one equivalent method may be compiled, but only one value is published. The cache is append-only within the process lifetime. GC can collect the old tree nodes once compilation completes, while the retained delegate keeps its `DynamicMethod` handle alive.

Typical memory profile:
- Each compiled kernel is ~2-5 KB of native code + its metadata in the runtime's dynamic-method table.
- Typical application: a few dozen unique expressions → ~100-200 KB of steady-state cache.
- Worst case: a hot loop constructing new-per-call trees → linear growth. Reuse expression objects or pass explicit cache keys.

To inspect or reset during tests:
```csharp
GeneratedDelegates.InnerLoopCount;        // count of compiled kernels (public)
GeneratedDelegates.ClearInnerLoop();  // internal — wipe for fresh-start testing
```

Both are `internal`, so scripts need the `AssemblyName=NumSharp.DotNetRunScript` override.

**2. Registered delegates and bound targets (`DelegateSlots`).**

Paths B and C of `Call` stash a managed reference in a static `ConcurrentDictionary<int, Delegate>` or `ConcurrentDictionary<int, object>` so the emitted IL can look it up at runtime. The reference is **strong** — entries live for the process lifetime. This is necessary: if the reference were weak, the GC could collect the delegate while a compiled kernel still holds its slot ID, and the next lookup would throw.

The cost is small per registration (~16-32 bytes for the dictionary entry plus whatever the delegate captures), but unbounded across registrations. Registering one delegate per kernel is fine; registering one delegate per iteration of a loop is a leak.

| Pattern | Registrations | Memory impact |
|--------|---------------|---------------|
| Static method (Path A) | zero | none |
| Cached delegate reused every iter | one | negligible |
| Per-call lambda | one per call | linear in call count |

Test hook:
```csharp
DelegateSlots.RegisteredCount;  // strong-ref count across both dicts
DelegateSlots.Clear();          // wipe for testing (invalidates kernels that reference it!)
```

> Calling `DelegateSlots.Clear()` while a kernel that references a slot is still compiled makes the next call throw `KeyNotFoundException` from inside the generated IL. Use it only in test setup/teardown, paired with clearing the inner-loop cache.

**3. NDArrays referenced by the iterator.**

Orthogonal to Tier 3C, but worth mentioning in the same section for completeness: `NDIterRef` holds a managed `NDArray[]` field so the operands' backing memory isn't collected mid-iteration. The field is released when you `Dispose()` the ref — the `using var iter = ...` pattern handles this automatically. Forgetting to dispose keeps the NDArrays alive for however long the iterator lives.

**Registration-once pattern.**

For `Call`-based activations or user kernels used in hot loops, the idiomatic pattern is:

```csharp
static class MyActivations
{
    // One delegate instance, registered once when the static class is first touched.
    public static readonly Func<double, double> Swish =
        x => x / (1.0 + Math.Exp(-x));

    public static readonly Func<double, double> GELU =
        x => 0.5 * x * (1.0 + Math.Tanh(
            Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
}

// Usage — reuses the same slot + cached kernel every time:
var swished = NDExpr.Call(MyActivations.Swish, NDExpr.Input(0));
var gelud   = NDExpr.Call(MyActivations.GELU,  NDExpr.Input(0));
```

##### Validation and errors

The DSL fails fast at tree-construction time for structural errors and at compile time for type-mismatch or arity errors:

| Error condition | Where | Exception |
|----------------|-------|-----------|
| `NDExpr.Input(-1)` | Factory | `ArgumentOutOfRangeException` |
| `NDExpr.Sqrt(null)` | Node constructor | `ArgumentNullException` |
| `NDExpr.Add(null, x)` / `Add(x, null)` | Node constructor | `ArgumentNullException` |
| `ExecuteExpression(expr, null, outType)` | Bridge entry | `ArgumentNullException` |
| `ExecuteExpression(expr, inputTypes, outType)` with too-few inputs vs operand count | Bridge entry | `ArgumentException` |
| `Input(5)` when tree compiled with 2 inputs | Compile-time IL emission | `InvalidOperationException` — message: `"Input(5) out of range; compile provided 2 inputs."` |
| Tree calls a vector-only path on a non-SIMD type (shouldn't happen via public API) | Compile-time | `NotSupportedException` |

Runtime errors depend on the op and dtype:

- `Divide` / `Mod` / `FloorDivide` with a zero integer divisor → `DivideByZeroException` from the CLI. Float division by zero produces `±Infinity` / `NaN` per IEEE 754, no exception.
- `Power(neg, fractional)` → `NaN` via `Math.Pow`, no exception.
- Overflow during `Conv_*` from a float that's outside the target integer range → silently wraps or saturates per the CLI's conv opcode semantics (matches `unchecked {}` casts in C#). Use `Conv_Ovf_*` if you need checked behavior — not exposed through the DSL.

##### Behavioral notes

Non-obvious but permanent behaviors of the DSL:

- **NaN propagation in `Min`/`Max` matches `np.minimum`/`np.maximum`, not `np.fmin`/`np.fmax`.** If you need NaN-skipping min/max, compose with `IsNaN` and `Where`:
  ```csharp
  // fmin(a, b): return non-NaN if one is NaN, else min
  var fmin = NDExpr.Where(NDExpr.IsNaN(a),
      b,
      NDExpr.Where(NDExpr.IsNaN(b), a, NDExpr.Min(a, b)));
  ```

- **`Mod` doesn't match C# `%` for negative operands.** C# truncates toward zero (`-10 % 3 == -1`); NumPy (and `NDExpr.Mod`) floor toward negative infinity (`-10 mod 3 == 2`). This matches Python `%`.

- **Integer division by zero throws.** `Divide(int_arr, int_arr_with_zero)` raises `DivideByZeroException` at runtime. Float division is silent (produces `±Infinity`/`NaN`).

- **Constants widen to the output dtype.** `NDExpr.Const(1_000_000_000) + NDExpr.Input(0)` where the output is `Byte` will emit `Ldc_I4 1000000000` followed by `Conv_U1` — the billion wraps to a small byte. The DSL won't check that the constant fits; you get silent truncation.

- **Bitwise ops require integer output dtype.** `NDExpr.Input(0) & NDExpr.Input(1)` with `outputType = Double` is a malformed tree — `EmitScalarOperation(BitwiseAnd, Double)` doesn't emit `And` for floats. You'll get an `InvalidOperationException` or unverifiable IL at compile time. Use an integer output dtype, or convert through `BitwiseNot`/`BitwiseAnd` in integer land and cast to float separately.

- **`LogicalNot` is `x == 0`, not `x != 0`.** It returns 1 when the input is zero and 0 otherwise. Same as Python's `not` applied to a numeric value. If you want "non-zero as 1", use `NDExpr.NotEqual(x, NDExpr.Const(0))`.

- **Input dtype mismatch is silent.** If your `inputTypes[]` says `Int32` but the actual NDArray operand is `Int16`, the kernel reads 4 bytes starting at the int16 pointer — garbage. The iterator's buffer/cast machinery only kicks in with `BUFFERED | NPY_*_CASTING`. For ad-hoc Tier 3C use, make sure `inputTypes[i]` matches the actual NDArray dtype, or run the iterator with casting flags.

- **Comparisons in non-float arithmetic can be off-by-one.** For integer-output trees, `NDExpr.Greater(x, Const(0.5))` with `x` as `Int32` will compare two integers — `Const(0.5)` gets emitted as `Ldc_I4 0`, because `ConstNode.EmitLoadTyped` converts the literal to the output dtype's CLI type. `Greater(int_x, 0)` is rarely the intended comparison. Use an explicit `Const(1)` with the correct integer threshold, or change the output dtype to a float.

- **`Where` duplicates both branches in IL.** The true-branch IL and false-branch IL are emitted sequentially with a `br` skipping the false side when cond is true. Deeply-nested `Where`s quadruple IL size (1 → 2 → 4 → 8 branches). For more than ~10 levels of nesting, consider flattening with a lookup table via Tier 3B.

- **`Call` delegates are held forever.** `CallNode` stashes captured delegates and bound instance targets in a process-wide `DelegateSlots` dictionary so the emitted IL can look them up. There is no eviction. If you call `NDExpr.Call(x => x * scale, in0)` inside a hot loop (creating a new closure each iteration), the dictionary grows without bound. Register delegates once at startup — a `static readonly Func<double, double>` field or a DI singleton — and reuse them.

- **`Call` method-group ambiguity.** `NDExpr.Call(Math.Abs, x)` fails to compile because `Math.Abs` has nine overloads (`double`, `float`, `int`, `long`, etc.) and the compiler can't pick one. Cast to the specific `Func<...>` you want: `NDExpr.Call((Func<double, double>)Math.Abs, x)`. Single-overload methods like `Math.Sqrt`, `Math.Cbrt`, `Math.Log` bind without cast.

- **`Call` runs at scalar speed.** A managed method call per element forfeits SIMD. For a sustained throughput-critical op, it's ~30-50% slower than the equivalent built-in DSL node because the call itself has overhead beyond just computing the result. Use `Call` for math the DSL doesn't expose (user-defined activations, `MathNet.Numerics` routines, lookup tables via a method), not for things like `Sqrt` where `NDExpr.Sqrt(x)` is the right answer.

- **`Call` return type widening is lossy for NaN.** If a delegate returns `int` and the tree output is `double`, `Math.Floor(NaN) = NaN` gets cast to `int` (yielding `0` or some CPU-dependent value), which widens back to the float representation of that integer. NaN information is lost across integer-returning calls. Match return dtype to output dtype when NaN correctness matters.

##### Debugging compiled kernels

Tier 3C kernels are `DynamicMethod` delegates — you can't step into their IL with a debugger as-is. What you *can* do:

- **Inspect the kernel cache.** `GeneratedDelegates.InnerLoopCount` (public) gives you a count. `GeneratedDelegates.ClearInnerLoop()` (internal; needs `[InternalsVisibleTo]` or a `dotnet_run` script with `AssemblyName=NumSharp.DotNetRunScript`) lets you force recompilation in a test.
- **Inspect the delegate slot registry** (only relevant when `Call` is in play). `DelegateSlots.RegisteredCount` (internal) returns the sum of registered delegates + registered instance targets. Growing unboundedly means a per-call lambda or target allocation somewhere — find it by comparing counts before and after your suspected hot path. `DelegateSlots.Clear()` wipes the registry; always pair with `GeneratedDelegates.ClearInnerLoop()` because cleared-but-cached kernels will throw `KeyNotFoundException` on their next invocation.
- **Print the auto-derived cache key.** Construct the tree, call `new StringBuilder().Also(e => node.AppendSignature(sb))` (`AppendSignature` is internal). The printed signature is exactly what goes into the cache key — useful for diagnosing "why aren't these two trees sharing a kernel?". For `Call` nodes in particular, the signature includes `MetadataToken` and `ModuleVersionId` — if those differ across two calls of what you thought was the same method, the compiler loaded the method from different assemblies or modules.
- **Reduce to a minimal tree.** If a compiled kernel misbehaves, isolate the failing subtree by compiling just that fragment against a tiny input (1-3 elements). `ExecuteExpression` on a 3-element array still exercises the scalar path; crashes become reproducible in a few lines.
- **Watch the output dtype.** `ExecuteExpression` expects `outputType` to match the output NDArray's dtype. If they disagree, the kernel reads/writes wrong byte counts. Double-check both.
- **Diagnose "method group ambiguous" errors.** If you see `CS0121: The call is ambiguous between the following methods` when writing `NDExpr.Call(Math.X, ...)`, the method has multiple overloads (e.g. `Math.Abs` has 9). Cast to the specific `Func<...>` you want, or use the `MethodInfo` overload with an explicit parameter-types array to `GetMethod`.
- **Diagnose "Method X returns void"** errors — you passed a method with no return value to `Call`. Tier 3C requires every node to contribute a value to the output dtype.
- **Diagnose "Target is X, method declares Y"** errors — your instance `MethodInfo` call received a target that isn't an instance of the method's declaring type. Confirm both the method and the target came from the same type, especially if you're reflecting across a plugin boundary.
- **Enable IL dumps** by emitting into a persistent assembly instead of `DynamicMethod` — not a supported build configuration, but `ILKernelGenerator.InnerLoop.cs` is a single partial file you can modify in a workspace-only diff if you need to dump bytes during development.

##### When to use Tier 3C

Reach for Tier 3C when a fused/custom rectangular elementwise operation fits the DSL. The catalog covers arithmetic, bitwise, rounding, transcendentals, predicates, comparisons, `Min`/`Max`/`Clamp`/`Where`, and common compositions without caller-authored IL. SIMD is an all-or-nothing property of the tree, so measure the compiled route; drop to Tier 3B when you need a vector emitter the DSL lacks or need to support one of `Call`'s excluded signature dtypes.

**Decision tree: which tier do I need?**

```
Does an existing production route already implement the op?
  yes → extend that route; do not replace its layout/dtype gates with a helper by name alone.
  no ↓

Can I express it as a tree of DSL nodes (Add, Sqrt, Where, Exp, etc.)?
  yes → Tier 3C. Fused, SIMD-or-scalar automatic, no IL.
  no ↓

Is the missing piece a supported BCL/user numeric method signature?
  yes → Tier 3C with Call. Scalar but fused.
  no ↓

Do I need V256/V512 intrinsics the DSL doesn't wrap (Fma, Shuffle, ...)?
  yes → Tier 3B. Hand-write the vector body; factory wraps the shell.
  no ↓

Is the loop shape non-rectangular (gather/scatter, cross-element deps)?
  yes → Tier 3A. Emit the whole inner-loop IL yourself.
```

**Caching is shared across the three custom tiers only.** Tier 3A/B/C write into the same `_innerLoopCache` inside `DirectILKernelGenerator.InnerLoop.cs`; the whole-array helper families use other dictionaries. The first custom compile for a key emits/JITs, and later calls with that key return the cached `NDInnerLoopFunc`. `GeneratedDelegates.InnerLoopCount` exposes this custom-inner-loop cache size.

---

## Path Detection

`DetectExecutionPath()` serves the whole-array mixed-type/comparison helper family. It looks at the iterator *after* coalescing and negative-stride flipping, and picks:

```csharp
if (CONTIGUOUS flag set)                                return SimdFull;
if (NDim == 0)                                          return SimdFull;
if (op1 is scalar AND op0 is contiguous)                return SimdScalarRight;
if (op0 is scalar AND op1 is contiguous)                return SimdScalarLeft;
if (every operand's innermost stride ∈ {0, 1})          return SimdChunk;
otherwise                                               return General;
```

"Scalar" here means every stride is 0 across every dimension — the operand is a 0-d array or a fully broadcasted view. "Contiguous" uses the standard backward stride check.

The resulting `ExecutionPath` is baked into the `MixedTypeKernelKey`:

```csharp
var key = new MixedTypeKernelKey(LhsType, RhsType, ResultType, Op, Path);
```

Different paths get different whole-array IL. `SimdFull` emits a flat 4×-unrolled SIMD loop. `SimdScalarRight`/`Left` splat a stride-0 operand. `SimdChunk` processes a unit/zero-stride inner dimension inside an outer coordinate loop. `General` performs coordinate-based iteration. This is distinct from Tier 3B's runtime byte-stride check inside a cached `NDInnerLoopFunc`.

Do not confuse this with `NDIterPathSelector.SelectPath` in `NDIterKernels.cs`. That older selector returns a different `NDIterExecutionPath` enum, its buffered executor still contains TODO transfer hooks, and it has no call sites. `NDIterRef.DetectExecutionPath` is the active helper selector.

---

## Production consumers and legacy surfaces

The iterator is broader than `np.nditer`, and production code does not enter through one universal method:

| Consumer family | Representative files | Entry shape |
|-----------------|----------------------|-------------|
| Elementwise binary/compare/unary/out/shift | `DefaultEngine.BinaryOp.cs`, `CompareOp.cs`, `UnaryOp.cs`, `UfuncOut.cs`, `Default.Shift.cs` | Usually `NDIterRef.MultiNew(EXTERNAL_LOOP, ...)` + Tier 3B `ExecuteElementWise*`; direct contiguous bypasses stay outside NDIter |
| Fused evaluation | `DefaultEngine.Evaluate.cs` | `AdvancedNew`/`MultiNew` + compiled expression/inner-loop kernels |
| Reductions and scans | `DefaultEngine.ReductionOp.cs`, `Default.Reduction.{Add,CumAdd,Nan}.cs`, `Default.All.cs`, `Default.Any.cs` | `ExecuteReduction`, `ExecuteReducing`, `ForEach`, `NewReduce`, or dedicated axis iterators depending dtype/layout |
| Selection/indexing/sorting | `np.where.cs`, `Default.BooleanMask.cs`, `Default.NonZero.cs`, `AxisSort.cs` | Chunk `ForEach`/reducing kernels where a direct whole-array kernel does not fit |
| Copy/cast/fill/pad/concatenate | static `NDIter.Copy`/`CopyAs`, called by `np.copyto`, `NDArray.fill`, `np.pad`, `np.concatenate`, storage cloning/casting | Separate static copy-state core with contiguous, strided-IL, scalar-cast, broadcast, and overlap-temporary tiers |
| Other focused consumers | `np.diff.cs`, `NDArray.byteswap.cs`, `np.average.cs`, `np.nan{mean,var,std}.cs` | Lean Tier 3B or chunk/reducing routes tailored to operation semantics |

Three older surfaces remain for compatibility or experiments:

- Static `NDIter.CreateCopyState`/`CreateReductionState` and helpers are used by the copy core; they are a smaller state model than full `NDIterRef.AdvancedNew`.
- `NDIterInnerLoopFunc` (declared in `NDIter.cs`) and `NDInnerLoopFunc` (declared in `NDIter.Execution.cs`) have the same signature but are distinct delegate types. New chunk code uses `NDInnerLoopFunc`.
- `INDIterKernel`, `NDIterPathSelector`, and `NDIterExecution` in `NDIterKernels.cs` are unused legacy scaffolding, not an alternative production backend.

---

## Fused kernels in production

Here, **fused** means that one kernel evaluates multiple array operations before it stores a result, or evaluates an elementwise expression while accumulating its reduction. It does **not** merely mean that the CPU happened to use a fused multiply-add instruction. The useful property is fewer full-array passes and no intermediate `NDArray` for values such as `a * b` or `a - b`.

There are two production shapes:

1. **General expression fusion** through `np.evaluate`: an `NDExpr` tree is typed with NumPy rules and compiled to one cached `NDInnerLoopFunc`, then `NDIterRef` supplies its broadcast/layout-aware chunks.
2. **Specialized fusion** inside an API such as weighted `np.average`, contiguous `np.diff`, or `np.asarray_chkfinite`: a purpose-built kernel combines the exact passes that operation needs. Some of these use `NDIterRef`; others are whole-array kernels whose own layout gate is cheaper.

That distinction matters when reading benchmark results. The dedicated fusion benchmark compares the same NumSharp expression **with and without** fusion. The unified API benchmark compares NumSharp with NumPy and may include other differences besides fusion.

### General expression fusion — `np.evaluate`

`np.evaluate` is a NumSharp extension (the closest NumPy-ecosystem analogue is `numexpr.evaluate`). The public API binds embedded `NDArray` leaves, `DefaultEngine.Evaluate` resolves each expression node's NumPy/NEP50 dtype, and `NDExpr.CompileNumPy` emits the cached inner loop. The elementwise route then executes it as:

```text
NDExpr tree
  → bind each repeated NDArray once
  → infer every node's NumPy dtype
  → compile/cache one NDInnerLoopFunc
  → NDIterRef.MultiNew(EXTERNAL_LOOP | COPY_IF_OVERLAP, ...)
  → iter.ForEach(kernel)
```

These are real expressions from `benchmark/fusion/evaluate_bench.cs`:

```csharp
// Multiply and add without materializing a*b.
NDArray mulAdd = np.evaluate((NDExpr)a * b + c);

// Four arithmetic nodes, but still one output pass and no a-b / a+b temporaries.
NDArray normalized = np.evaluate(
    (NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));

// The multiply and scalar reduction share one pass; a*b is never allocated.
NDArray dotLike = np.evaluate(NDExpr.Sum((NDExpr)a * b));

// Per-node typing preserves the equivalent NumPy promotion: int32 * weak int,
// then addition with float64, all inside the compiled kernel.
NDArray mixed = np.evaluate((NDExpr)ai * 2 + c);

// A caller-owned result removes the fused path's result allocation as well.
NDArray output = np.empty_like(c);
np.evaluate((NDExpr)a * b + c, @out: output);
```

Elementwise fusion supports broadcasting, strided operands, overlap-safe `out=`, and buffered write-back when `out` needs a cast. A root `Sum`, `Prod`, `Min`, `Max`, or `Mean` fuses element evaluation with the reduction; reductions nested inside a larger elementwise expression are intentionally rejected and require two calls.

### Measured fused versus unfused performance

The current [fusion benchmark report](https://scisharp.github.io/NumSharp/docs/reports/fusion_results.html) uses 4 million elements and warmed best-window measurements. In this table, **speedup is NumSharp unfused ÷ NumSharp fused**; it is not a NumPy/NumSharp ratio.

| Real expression | Fused work | Fused | Unfused | Fusion speedup |
|-----------------|------------|------:|--------:|---------------:|
| `a*b+c` (`float64`) | multiply + add | 3.38 ms | 5.76 ms | **1.70×** |
| `(a-b)/(a+b)` (`float64`) | two subtract/add branches + divide | 2.83 ms | 12.00 ms | **4.24×** |
| `sum(a*b)` (`float64`) | multiply + scalar sum | 2.23 ms | 3.64 ms | **1.63×** |
| `sum(af*bf)` (`float32`) | multiply + scalar sum | 1.20 ms | 1.45 ms | **1.21×** |
| `i4*2+f8` | mixed-dtype multiply + add | 2.68 ms | 3.79 ms | **1.41×** |

The allocating comparisons above are the fair fusion ratios. The separate `a*b+c, out=` case is a one-pass write into a preallocated result and measures **3.26 ms**; the report deliberately gives it no unfused ratio because an `out=` spelling of the unfused chain is still two passes.

Fusion is not automatically faster for every layout. The same benchmark applies `a*b+c` to three operands with the same 2-D layout:

| Operand layout | Fused | Unfused | Fusion speedup |
|----------------|------:|--------:|---------------:|
| C-contiguous | 3.38 ms | 5.75 ms | **1.70×** |
| F-contiguous | 11.74 ms | 17.66 ms | **1.50×** |
| Transposed | 11.71 ms | 17.48 ms | **1.49×** |
| Strided (`[:, ::2]`) | 11.64 ms | 4.76 ms | **0.41×** |
| Broadcast (stride 0) | 2.02 ms | 7.72 ms | **3.81×** |

The broadcast case is the largest win because the fused kernel repeatedly consumes a stride-0 value without writing intermediates. The strided case is the important counterexample: the current fused scalar-strided route is about **2.44× slower** than the unfused chain (`1 / 0.41`). Treat fusion as a memory-traffic opportunity, not a guarantee; benchmark the exact dtype and layout, and keep this regression visible when changing the general stride path.

The canonical NDIter benchmark also contains `fuse7`, a lower-level eight-operand inner loop that adds seven `float64` arrays in one pass and compares it with six chained `np.add(..., out=...)` calls. Its current NumPy÷NumSharp dividend is **1.41× at 100K, 1.69× at 1M, and 2.02× at 10M**. That case validates the iterator/kernel mechanism directly; the `np.evaluate` table above is the user-facing production evidence.

### Specialized production fusions

Several APIs use focused kernels because their loop shape is more specific than a general `NDExpr` tree:

| API / implementation | What is fused | Active boundary | Current benchmark evidence |
|----------------------|---------------|-----------------|----------------------------|
| Weighted `np.average` / `DirectILKernelGenerator.WeightedSum.cs` | Accumulates `sum(a * weights)` **and** `sum(weights)` into two outputs in one four-operand pass | Primitive numeric dtypes; a contiguous reduce-all case calls the kernel directly, while axis reductions use `NDIterRef` with pinned or scatter outputs. `Bool`, `Char`, `Half`, `Complex`, and `Decimal` fall back to multiply then sum. | The unified `np.average(a)` rows use `weights == null` and therefore dispatch to `mean`; they do **not** measure this fused kernel. An isolated weighted fusion benchmark is still needed. |
| `np.diff` / `DirectILKernelGenerator.AdjacentDiff.cs` | Computes `out[i] = a[i+1] - a[i]` from overlapping loads in one whole-array pass, avoiding two slice views and generic binary iteration | `n == 1`, no prepend/append, C-contiguous non-broadcast input, innermost axis, non-Boolean dtype | The unified 100K `float64` case is **0.04747 ms** and **0.490× NumPy**; the 1K case is **0.00306 ms** and **1.343× NumPy**. It is a real fused route, but the larger current case remains a performance gap. |
| `np.asarray_chkfinite` / `FiniteScan.cs` | Fuses `isfinite(a)` and `all(...)` into one NaN-poison scan with no Boolean temporary | Float-family dtypes; contiguous SIMD plus strided/transposed/negative-stride/broadcast handling | The unified 100K `float64` case is **0.01176 ms**, **1.814× faster than NumPy**. |

The last column uses the [unified benchmark report](https://scisharp.github.io/NumSharp/docs/reports/benchmark-report.html), where the ratio is **NumPy ÷ NumSharp**. Do not compare those ratios directly with the dedicated fusion table's unfused-NumSharp baseline.

---

## Worked Examples

Nineteen worked examples grouped by execution shape.

**Chunk drivers and whole-array helpers:**
1. [Three-operand binary over a 3-D contiguous array](#1-three-operand-binary-over-a-3-d-contiguous-array)
2. [Array × scalar with broadcast detection](#2-array--scalar-with-broadcast-detection)
3. [Sliced view — non-contiguous input](#3-sliced-view--non-contiguous-input)
4. [Fused hypot via Layer 1](#4-fused-hypot-via-layer-1)
5. [Early-exit Any over 1M elements](#5-early-exit-any-over-1m-elements)

**Custom compiled inner loops:**

6. [Fused hypot via Tier 3C expression](#6-fused-hypot-via-tier-3c-expression)
7. [Fused linear transform via Tier 3B with vector body](#7-fused-linear-transform-via-tier-3b-with-vector-body)

**More Tier 3C expression DSL examples:**

8. [ReLU via Tier 3C comparison-multiply](#8-relu-via-tier-3c-comparison-multiply)
9. [Clamp with Min/Max](#9-clamp-with-minmax)
10. [Softmax-ish: exp then divide-by-sum](#10-softmax-ish-exp-then-divide-by-sum)
11. [Sigmoid via Where for numerical stability](#11-sigmoid-via-where-for-numerical-stability)
12. [NaN-replacement using IsNaN + Where](#12-nan-replacement-using-isnan--where)
13. [Leaky ReLU via piecewise Where](#13-leaky-relu-via-piecewise-where)
14. [Manual abs via comparison + Where](#14-manual-abs-via-comparison--where)
15. [Heaviside step function](#15-heaviside-step-function)
16. [Polynomial evaluation via Horner's method](#16-polynomial-evaluation-via-horners-method)
17. [Piecewise: absolute value of sine (abs(sin(x)))](#17-piecewise-absolute-value-of-sine-abssinx)
18. [User-defined activation via NDExpr.Call](#18-user-defined-activation-via-ndexprcall)
19. [Reflected MethodInfo with an instance method](#19-reflected-methodinfo-with-an-instance-method)

### 1. Three-operand binary over a 3-D contiguous array

```csharp
var a = np.arange(24, dtype: np.float32).reshape(2, 3, 4);
var b = (np.arange(24, dtype: np.float32).reshape(2, 3, 4) * 2f).astype(np.float32);
var c = np.zeros(new Shape(2, 3, 4), np.float32);

using var iter = NDIterRef.MultiNew(
    nop: 3, op: new[] { a, b, c },
    flags: NDIterGlobalFlags.None,
    order: NPY_ORDER.NPY_KEEPORDER,
    casting: NPY_CASTING.NPY_NO_CASTING,
    opFlags: new[] { NDIterPerOpFlags.READONLY,
                     NDIterPerOpFlags.READONLY,
                     NDIterPerOpFlags.WRITEONLY });

iter.ExecuteBinary(BinaryOp.Add);
// NDim = 1 after coalesce, Path = SimdFull
// ILKernelGenerator emits a 4×-unrolled V256 add loop
// c[1,2,3] = 69
```

One call. 3-D → 1-D coalesce → one SIMD kernel runs over 24 elements. The generated IL is the same regardless of whether `a` and `b` started as 3-D, 4-D, or flat — as long as they're contiguous.

### 2. Array × scalar with broadcast detection

```csharp
var vec = np.arange(8, dtype: np.float32);
var sc  = np.full(new Shape(), 100f, NPTypeCode.Single);   // 0-d scalar
var res = np.zeros(new Shape(8), np.float32);

using var iter = NDIterRef.MultiNew(3, new[] { vec, sc, res }, ...);

Console.WriteLine(iter.DetectExecutionPath());  // SimdScalarRight
iter.ExecuteBinary(BinaryOp.Multiply);
// res = vec * 100
```

The 0-d scalar comes through with all strides equal to 0, so `DetectExecutionPath` picks `SimdScalarRight`. The kernel loads the scalar once, splats it into a V256 register, and multiplies the whole LHS against it.

### 3. Sliced view — non-contiguous input

```csharp
var big   = np.arange(20, dtype: np.float32).reshape(4, 5);
var slice = big[":, 1:4"];     // 4×3 view, strides = [5, 1]
var dst   = np.zeros(new Shape(4, 3), np.float32);

using var iter = NDIterRef.MultiNew(2, new[] { slice, dst }, ...);
iter.ExecuteUnary(UnaryOp.Sqrt);
// dst[3,2] = sqrt(big[3,3]) = sqrt(18) ≈ 4.243
```

The slice can't coalesce (stride 5 on outer axis, stride 1 on inner) so NDim stays at 2 and `IsContiguous` is false. Layer 3 picks the strided `UnaryKernel`, which computes `offset = sum(coord[d] * stride[d])` at each element.

### 4. Fused hypot via Layer 1

```csharp
using var iter = NDIterRef.MultiNew(3, new[] { a, b, result },
    NDIterGlobalFlags.EXTERNAL_LOOP, ...);

iter.ForEach((ptrs, strides, count, _) => {
    if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
        float* pa = (float*)ptrs[0], pb = (float*)ptrs[1], pc = (float*)ptrs[2];
        for (long i = 0; i < count; i++)
            pc[i] = MathF.Sqrt(pa[i] * pa[i] + pb[i] * pb[i]);   // JIT → V256
    } else {
        byte* pA = (byte*)ptrs[0], pB = (byte*)ptrs[1], pC = (byte*)ptrs[2];
        long sA = strides[0], sB = strides[1], sC = strides[2];
        for (long i = 0; i < count; i++) {
            float av = *(float*)(pA + i * sA);
            float bv = *(float*)(pB + i * sB);
            *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
        }
    }
});
```

Without Layer 1 this operation would be `sqrt(a * a + b * b)` — three Layer 3 calls and three temporary arrays. Fused into one kernel, it runs in a single pass with zero intermediates. The stride branch is the idiom that lets the JIT autovectorize the tight case while the outer shape keeps the kernel correct for strided inputs.

### 5. Early-exit Any over 1M elements

```csharp
var data = np.zeros(new Shape(1_000_000), NPTypeCode.Int32);
data[500] = 1;

using var iter = NDIterRef.New(data, flags: NDIterGlobalFlags.EXTERNAL_LOOP);
bool found = iter.ExecuteReducing<AnyNonZero, bool>(default, false);
// found = true, after exactly one ForEach call (SIMD early exit inside kernel).
```

### 6. Fused hypot via Tier 3C expression

The same hypot operation written as an expression tree — no IL, no hand-written stride branch. The factory emits a 4×-unrolled V256 kernel on the contiguous path and a scalar-strided fallback on non-contiguous input.

```csharp
using var iter = NDIterRef.MultiNew(3, new[] { a, b, result },
    NDIterGlobalFlags.EXTERNAL_LOOP, ...);

var expr = NDExpr.Sqrt(NDExpr.Square(NDExpr.Input(0)) +
                        NDExpr.Square(NDExpr.Input(1)));

iter.ExecuteExpression(expr,
    inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
    outputType: NPTypeCode.Single);
// result[i] = sqrt(a[i]² + b[i]²), fused in one pass, SIMD-vectorized
```

Compare with example 4 — same output, same performance envelope, no IL emission visible in your code. The tree's structural signature `"Sqrt(Add(Square(In[0]),Square(In[1])))"` becomes the cache key, so every iterator that runs the same expression reuses the same compiled delegate.

### 7. Fused linear transform via Tier 3B with vector body

When you want the Tier 3C ergonomics but also want the vector body under your control (e.g. to insert a Vector256 intrinsic the DSL doesn't expose):

```csharp
iter.ExecuteElementWiseBinary(
    NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
    scalarBody: il =>
    {
        // Stack: [a, b] → [a*2 + b*3]
        il.Emit(OpCodes.Ldc_R4, 2.0f); il.Emit(OpCodes.Mul);  // a*2
        var tmp = il.DeclareLocal(typeof(float));
        il.Emit(OpCodes.Stloc, tmp);                           // stash a*2
        il.Emit(OpCodes.Ldc_R4, 3.0f); il.Emit(OpCodes.Mul);  // b*3
        il.Emit(OpCodes.Ldloc, tmp); il.Emit(OpCodes.Add);    // a*2 + b*3
    },
    vectorBody: il =>
    {
        // Stack: [va, vb]
        il.Emit(OpCodes.Ldc_R4, 2.0f); ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);  // va*2
        var tmp = il.DeclareLocal(ILKernelGenerator.GetVectorType(typeof(float)));
        il.Emit(OpCodes.Stloc, tmp);
        il.Emit(OpCodes.Ldc_R4, 3.0f); ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);  // vb*3
        il.Emit(OpCodes.Ldloc, tmp);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
    },
    cacheKey: "linear_2a_3b_f32");
```

Single pass, no temporaries, SIMD-unrolled. Conceptually the same as `2*a + 3*b` written via Tier 3C, but lets you drop in `Vector256.Fma` or similar intrinsics if you ever need them.

### 8. ReLU via Tier 3C comparison-multiply

ReLU in one fused kernel, leveraging Tier 3C's "comparison returns 0/1 at output dtype" semantics:

```csharp
using var iter = NDIterRef.MultiNew(2, new[] { input, output },
    NDIterGlobalFlags.EXTERNAL_LOOP, ...);

var relu = NDExpr.Greater(NDExpr.Input(0), NDExpr.Const(0.0f)) * NDExpr.Input(0);
iter.ExecuteExpression(relu,
    new[] { NPTypeCode.Single }, NPTypeCode.Single);
// output[i] = max(input[i], 0) for every i
```

No branch, no intermediate array. The comparison node emits an I4 0/1, gets converted to float, and the multiply folds it into the final value. Scalar path only (comparisons don't SIMD), but the JIT autovectorizes the resulting tight loop post-tier-1.

### 9. Clamp with Min/Max

```csharp
var clamped = NDExpr.Clamp(NDExpr.Input(0), NDExpr.Const(-1.0), NDExpr.Const(1.0));
iter.ExecuteExpression(clamped,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
// output[i] = min(max(input[i], -1), 1)
```

`Clamp` is just sugar for `Min(Max(x, lo), hi)` — both map to branchy scalar selects that propagate NaN (matching `np.minimum` / `np.maximum` rather than `np.fmin` / `np.fmax`).

### 10. Softmax-ish: exp then divide-by-sum

Tier 3C is element-wise; reductions (like summing all elements) aren't expressible directly. But the element-wise half of softmax is:

```csharp
// out = exp(x - max_x) / sum_exp   — where max_x and sum_exp are precomputed scalars.
var shifted = NDExpr.Subtract(NDExpr.Input(0), NDExpr.Const(maxX));
var numerator = NDExpr.Exp(shifted);
var result = numerator / NDExpr.Const(sumExp);
iter.ExecuteExpression(result,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Scalar path only (Exp isn't in the vector emit set), but the tree fuses three operations into one kernel — versus three Layer 3 calls with two temporary arrays.

### 11. Sigmoid via Where for numerical stability

The naive `1 / (1 + exp(-x))` overflows for very negative `x` (exp of a large positive number). A numerically stable form uses two branches:

```csharp
//           { 1 / (1 + exp(-x))   if x >= 0
// sigmoid = { exp(x) / (1 + exp(x)) if x < 0
var x = NDExpr.Input(0);
var pos = NDExpr.Const(1.0) / (NDExpr.Const(1.0) + NDExpr.Exp(-x));
var neg = NDExpr.Exp(x) / (NDExpr.Const(1.0) + NDExpr.Exp(x));
var stable = NDExpr.Where(NDExpr.GreaterEqual(x, NDExpr.Const(0.0)), pos, neg);

iter.ExecuteExpression(stable,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Every branch computes three `Exp` calls in the worst case, but only the taken branch's values are materialized — `Where` emits actual `brfalse` + jump IL, not a branchless blend. For large arrays, branch prediction handles a sign-bit pattern well. If your input is already known to be mostly positive or mostly negative, this is noticeably cheaper than the naive `1/(1+exp(-x))` kernel.

### 12. NaN-replacement using `IsNaN` + `Where`

```csharp
// replace NaN with 0
var x = NDExpr.Input(0);
var clean = NDExpr.Where(NDExpr.IsNaN(x), NDExpr.Const(0.0), x);
iter.ExecuteExpression(clean,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

`IsNaN(x)` emits a `double.IsNaN` call that leaves an I4 0/1 on the stack, and `UnaryNode` inserts an implicit `EmitConvertTo(Int32, Double)` so `Where`'s condition-normalizer gets the right dtype. The whole tree is scalar-only but fuses NaN-detection and replacement into a single pass.

### 13. Leaky ReLU via piecewise Where

```csharp
// leaky_relu(x, alpha=0.1) = x if x > 0 else alpha * x
var x = NDExpr.Input(0);
var leaky = NDExpr.Where(
    NDExpr.Greater(x, NDExpr.Const(0.0)),
    x,
    NDExpr.Const(0.1) * x);
iter.ExecuteExpression(leaky,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Contrast with the "branchless" ReLU (`(x > 0) * x`): that works for plain ReLU because the false branch is zero, but doesn't handle Leaky ReLU's non-zero negative side. `Where` is the general escape hatch.

### 14. Manual abs via comparison + Where

A worked example of combining comparisons with `Where` for pedagogical purposes (the DSL's `Abs` is faster — it has a SIMD path):

```csharp
var x = NDExpr.Input(0);
var manualAbs = NDExpr.Where(
    NDExpr.Less(x, NDExpr.Const(0.0)),
    -x,           // operator overload for Negate
    x);
iter.ExecuteExpression(manualAbs,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

This is ~10% slower than `NDExpr.Abs(x)` because it runs the scalar-only `Where` instead of the SIMD-vectorized `Abs`. Use the built-in where possible; `Where` is the generalization when no built-in fits.

### 15. Heaviside step function

```csharp
// heaviside(x, h0) = 0 if x < 0, h0 if x == 0, 1 if x > 0
// NumPy's np.heaviside(x, 0.5) is the default "midpoint" convention.
var x = NDExpr.Input(0);
var step = NDExpr.Where(
    NDExpr.Less(x, NDExpr.Const(0.0)),
    NDExpr.Const(0.0),
    NDExpr.Where(
        NDExpr.Greater(x, NDExpr.Const(0.0)),
        NDExpr.Const(1.0),
        NDExpr.Const(0.5)));   // h0 value at x == 0

iter.ExecuteExpression(step,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Three-way nested `Where` flattens to linear IL — two `brfalse` branches at runtime. The auto-derived cache key becomes `Where(CmpLess(In[0],Const[0]),Const[0],Where(CmpGreater(In[0],Const[0]),Const[1],Const[0.5]))`. Reused automatically across iterators.

### 16. Polynomial evaluation via Horner's method

Evaluate `p(x) = 1·x⁴ + 2·x³ + 3·x² + 4·x + 5` with optimal multiplications:

```csharp
// ((((1·x + 2)·x + 3)·x + 4)·x + 5
var x = NDExpr.Input(0);
var poly = (((NDExpr.Const(1.0) * x + NDExpr.Const(2.0)) * x
             + NDExpr.Const(3.0)) * x
             + NDExpr.Const(4.0)) * x
             + NDExpr.Const(5.0);
iter.ExecuteExpression(poly,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Four `Multiply`s, four `Add`s — all SIMD-capable. Whole tree emits the 4×-unrolled V256 path. For a degree-N polynomial this stays in registers end-to-end, with no intermediate array allocations. Compare with the naïve `1*x*x*x*x + 2*x*x*x + 3*x*x + 4*x + 5` — ten multiplications, same IL size after constant folding by the JIT, but less readable.

### 17. Piecewise: absolute value of sine (abs(sin(x)))

Combine the two unary SIMD-capable ops for the pattern `|sin x|`:

```csharp
var expr = NDExpr.Abs(NDExpr.Sin(NDExpr.Input(0)));
iter.ExecuteExpression(expr,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

`Sin` is scalar-only, so the whole tree runs scalar (no 4× unroll). But both ops fuse into one pass — a single `Math.Sin` call + `Math.Abs` per element. The alternative — two Layer 3 calls on three arrays — would allocate a `sin(x)` temporary.

### 18. User-defined activation via `NDExpr.Call`

Say you want **Swish** (`x * sigmoid(x)`, used in EfficientNet and family) but Tier 3C doesn't have a `Sigmoid` node. Drop to `Call`:

```csharp
// Registered once at startup — static readonly field, not a per-call lambda.
static readonly Func<double, double> SwishActivation =
    x => x / (1.0 + Math.Exp(-x));

// Tree: out = Swish(x) + bias  (bias is a broadcast-scalar Input, not a Const)
var expr = NDExpr.Call(SwishActivation, NDExpr.Input(0)) + NDExpr.Input(1);
iter.ExecuteExpression(expr,
    new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double);
```

The `SwishActivation` delegate is registered exactly once into `DelegateSlots`; every subsequent iter reuses the same slot ID and the same compiled kernel (auto-derived cache key is stable because it's keyed by `MethodInfo.MetadataToken`, not delegate identity). Runtime overhead is ~5 ns per element for the slot lookup + one `Delegate.Invoke` call per element — still single-pass, still zero intermediates.

For maximum speed, if your activation is hot enough to matter, compose it out of DSL primitives:
```csharp
var x = NDExpr.Input(0);
var swish = x / (NDExpr.Const(1.0) + NDExpr.Exp(-x));   // same op, no Call overhead
```

### 19. Reflected MethodInfo with an instance method

Sometimes you're calling a method you discovered via reflection (e.g. an op registered through a plugin system). Use the `MethodInfo + target` overload:

```csharp
var provider = new PluginActivations { Temperature = 1.5 };
var method = provider.GetType().GetMethod("ApplyTempered")!;
// ApplyTempered(double x) => Math.Pow(x, 1.0 / Temperature);

var expr = NDExpr.Call(method, provider, NDExpr.Input(0));
iter.ExecuteExpression(expr,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

The `provider` object's state (`Temperature`) is captured into the compiled kernel via a `DelegateSlots` slot ID. Mutating `provider.Temperature` between calls is visible to subsequent invocations — the slot holds the reference, not a snapshot.

---

## Performance

Performance is a property of the **selected route, layout, dtype, size, chunk width, and warm state**, not of the `NDIter` name. The current canonical results are generated into [the NDIter benchmark report](https://scisharp.github.io/NumSharp/docs/reports/nditer_results.html); the snapshot above deliberately includes both wins and losses.

The dominant costs to reason about are:

- **Construction:** state + dimension/operand blocks, broadcast/order analysis, overlap solving, and optional aligned buffers.
- **Dispatch width:** one delegate call over a long coalesced chunk is cheap; millions of width-1/4 chunks are not. The canonical width-4 canary is slower than NumPy while width 16 is near parity.
- **Transfer:** buffering can unlock tight SIMD but adds gather/scatter and possibly casting. Linear strided operands are intentionally left direct.
- **Kernel body:** Tier 3B vector IL gets the shared 4×-unrolled shell only when every operand dtype is identical/SIMD-capable and a vector body exists. Otherwise it uses the scalar-strided body.
- **Fusion:** Tier 3C/Tier 3B can avoid intermediate arrays and memory traffic, but unsupported nodes demote the entire expression to scalar.
- **Early exit:** `ExecuteReducing` can stop on the first decisive chunk; this is why the benchmark's early-hit `any` can be dramatically faster than a full scan.
- **`Call`:** static methods emit a direct call; captured delegates/bound targets add a process-wide slot lookup and virtual dispatch per element. Use it for meaningful scalar work, not as a replacement for a built-in SIMD-capable node.

### JIT warmup and measurement

.NET uses tiered compilation: a struct-generic or caller-authored scalar loop may first run as tier-0 code and later be recompiled/optimized. DynamicMethod compilation and family-cache population also add first-use costs. A short ad-hoc benchmark can therefore measure a different program from the steady-state route.

Under-warmed measurement shows up as:
- `ExecuteGeneric` and `ForEach` swap apparent order between runs;
- a first custom compile dominates the operation;
- iterator reuse looks implausibly better because only one call site reached optimized code;
- a tiny chunk-width route measures dispatch rather than arithmetic.

Use the canonical harness's warm pilot, adaptive batching, best-of rounds, matched NumPy ids, and correctness checks. If you create a focused benchmark, record runtime, CPU, vector support, exact layout/dtype/size, construction inclusion, warmup policy, and cache state.

### Implementation Notes

The chunk bridge is tuned for the JIT in two ways:

1. **Fast-path split.** `ExecuteGeneric` dispatches to `ExecuteGenericSingle` (1 call, inlineable) or `ExecuteGenericMulti` (do/while driver). Small single-call bodies are what the autovectorizer needs to do its job — a do/while with a delegate inside prevents tier-1 SIMD promotion.

2. **`AggressiveInlining + AggressiveOptimization`.** Both attributes sit on the fast path so the JIT doesn't punt on inlining due to method size and immediately promotes to tier-1 once discovered hot.

Tier 3B/C instead use an emitted runtime stride check: dense byte strides enter the optional SIMD/unroll body; any mismatch enters the scalar-strided body. Whole-array helpers have different code shapes and caches, so do not infer their behavior from the struct-generic driver.

### When Does Each Layer Pay Off?

| Route | Good for | Boundary |
|-------|----------|----------|
| `ForEach` | Existing/capturing chunk delegate, `auxdata`, masked run wrapper | Delegate call per chunk; caller controls whether delegate allocation occurs |
| `ExecuteGeneric` | Hot reusable struct kernel; JIT specialization | Kernel implements its own scalar/SIMD body and stride branches |
| `ExecuteReducing` | Early-exit `All`/`Any`/find-first and scalar accumulators | Not an axis-output scheduler by itself |
| Tier 3B `ExecuteElementWise*` | Production rectangular elementwise ops over arbitrary inner strides | Caller emits valid scalar IL; SIMD requires compatible identical dtypes and vector IL |
| Tier 3C `ExecuteExpression` | Fused DSL expressions and no temporary arrays | One unsupported node demotes SIMD; `EXTERNAL_LOOP` is required unless one iteration |
| Whole-array `Execute*` helpers | Exact closed helper shape, especially scalar reduction/contiguous output | Separate family contracts/caches; most are not the current general elementwise route |
| `BufferedReduce` | Legacy explicit `BUFFER+REDUCE` schedule | No production callers; masked write-back remains unsupported |

For a struct-generic kernel, keep a typed-pointer fast branch and a byte-stride fallback. Add explicit vector/unroll code only after the canonical benchmark shows the route matters:

```csharp
public void Execute(void** p, long* s, long n) {
    if (s[0] == sizeof(float) && s[1] == sizeof(float)) {
        float* src = (float*)p[0]; float* dst = (float*)p[1];
        for (long i = 0; i < n; i++) dst[i] = MathF.Sqrt(src[i]);  // JIT → V256
    } else {
        byte* p0 = (byte*)p[0]; byte* p1 = (byte*)p[1];
        long s0 = s[0], s1 = s[1];
        for (long i = 0; i < n; i++)
            *(float*)(p1 + i * s1) = MathF.Sqrt(*(float*)(p0 + i * s0));
    }
}
```

The emitted Tier 3B shell already owns its vector remainder and scalar tail; do not duplicate that shell inside a Tier 3B scalar/vector body.

### Allocations

Separate iterator-state allocation from execution-call allocation:

- `NDIterRef.AdvancedNew` allocates unmanaged state plus dimension/operand blocks; buffering adds aligned blocks. `np.NDIterator` detaches and owns the same state.
- Whole-array helpers use `stackalloc` stride arrays; that is stack space, not a managed heap allocation. First use can allocate/compile a cached `DynamicMethod`.
- `ExecuteGeneric`/`ExecuteReducing` do not allocate a delegate. The iterator state already exists.
- `ForEach` does not manufacture a delegate; a capturing lambda supplied by the caller allocates, while a cached/static delegate does not.

**Custom-op tiers:**

| Tier | Per-call allocation | One-time allocation |
|------|--------------------|--------------------|
| Tier 3A (`ExecuteRawIL`) | Caller-provided emit callback; `ForEach` schedule | Compiled `DynamicMethod` cached by key for process lifetime |
| Tier 3B (`ExecuteElementWise`) | Operand-type array and caller emit callbacks (typically static/cached) | Compiled `NDInnerLoopFunc` cached by key |
| Tier 3C (`ExecuteExpression`) | Expression tree is allocated by the caller; auto-key walks the tree | Compiled inner loop cached by structural/explicit key |
| Tier 3C with `Call` | same as Tier 3C, plus one `DelegateSlots` entry per unique captured delegate / bound target | registered references live for process lifetime; see [Memory model and lifetime](#memory-model-and-lifetime) |

The unbounded-growth anti-pattern is constructing a new `Call` delegate per iteration: each reference gets a new slot id, and an auto-derived key includes that id, so it also grows the custom kernel cache. An explicit reused key would instead bind to the first compiled slot, not the new delegate—another reason to register and reuse stable delegates.

---

## Summary

`NDIterRef` turns operands with different shapes, strides, offsets, orders, dtypes, masks, and overlap relationships into a deterministic pointer/window schedule. The public `np.NDIterator` owns that schedule across managed calls; typed iterators borrow it for allocation-free C# traversal; `np.nested_iters` links several schedules by re-basing child pointers.

**Three execution shapes.** `ForEach` drives a chunk delegate, `ExecuteGeneric`/`ExecuteReducing` drive a JIT-specialized struct, and the closed `ExecuteBinary/Unary/...` helpers call whole-array DirectIL delegates. Only custom Tier 3A/B/C kernels universally funnel through `ForEach`; caches are split by family.

**Custom ops.** `ExecuteRawIL` emits the whole `NDInnerLoopFunc`; `ExecuteElementWise` supplies scalar and optional vector value bodies inside the shared shell; `ExecuteExpression` lowers a typed DSL tree to that Tier 3B compiler. They preserve iterator chunk/window/mask semantics and share the custom inner-loop cache.

**Coalesce first.** A 3-D contiguous array should run as one flat SIMD loop, not a triple-nested loop. The iterator does this for you — as long as you don't set flags that disable it (`MULTI_INDEX`, `C_INDEX`, `F_INDEX`).

**Buffer selectively.** Cast/`CONTIG`/reduction/non-linear operands get aligned windows; contiguous, constant-stride 1-D, and scalar-broadcast operands can remain direct with true byte strides.

**Struct-generic is a template substitute.** Constraining a type parameter to `struct` lets the JIT specialize the driver for a concrete kernel type and removes delegate dispatch. Inlining and vectorization remain runtime- and body-dependent, so verify the exact hot path instead of assuming a particular tier threshold.

**Autovectorization is not a contract.** A simple typed-pointer or constant-byte-stride loop may vectorize after optimization; branch shape, runtime version, CPU, and caller context can change the result. Tier 3B's explicit vector body is the controlled SIMD route.

Everything else—flag parsing, `op_axes`, negative-stride flipping, index tracking, overlap solving, masked runs, and buffered write-back—protects correctness around the hot loop. The current gaps are explicit above; do not describe the subsystem as full parity until those boundaries are closed.

## See Also

- [IL Generation](il-generation.md) — the kernel side of the bridge
- [Broadcasting](broadcasting.md) — stride-0 iteration
- [Buffering & Memory](buffering.md) — buffer allocation and lifetime
