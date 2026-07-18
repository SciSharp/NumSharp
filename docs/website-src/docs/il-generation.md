# IL Kernel Generation in NumSharp

Most numerical libraries face the same fork in the road: write generic managed
loops that are portable but slow, or bind to a native backend that is fast but
opaque. NumSharp takes a third path. It ships a small compiler inside the
library — a code generator that, the first time you ask for `a + b` on two
`float64` arrays, emits a `DynamicMethod` specialized for exactly that operation,
dtype pair, memory layout, and SIMD width, hands the raw IL to the .NET JIT to
lower into native vector code, and caches the resulting delegate for the rest of
the process. Every later `a + b` of that shape is a dictionary lookup and a
direct call into machine code.

That layer is the performance backbone behind elementwise ufuncs, reductions,
scans, casts, selection and indexing, `np.where`, `np.average`, `np.evaluate`,
and the NDIter custom-operation bridge. This page is its specification: what the
two generators are, how a request becomes a kernel, how layout and SIMD width are
chosen, and — the part that makes it more than an optimizer — how NumPy's dtype
and edge-case semantics survive the trip into hand-written machine code.

## At a Glance

The kernel layer is large, and every figure below is measured from the
checked-in source (see [Refreshing the Inventory](#refreshing-the-inventory)),
never estimated:

| The system in numbers | |
| --- | ---: |
| Kernel source files | 84 |
| &nbsp;&nbsp;— whole-array `DirectILKernelGenerator` partials | 64 |
| &nbsp;&nbsp;— NDIter `ILKernelGenerator` partials | 5 |
| &nbsp;&nbsp;— shared infrastructure | 15 |
| `DynamicMethod` kernel factories | 57 |
| Distinct generated-kernel caches | 45 |
| Binary / unary / reduction / comparison ops | 17 / 35 / 20 / 6 |
| Dtypes every kernel must handle | 15 |
| Binary memory-layout paths | 5 |
| Total kernel source | ~48,000 lines |
| Raw IL instructions emitted by hand | ~7,000 |

Read the last two rows together: roughly 48,000 lines of C# exist to hand-emit
about 7,000 raw IL instructions across 57 factories. The ratio is the whole
story — most of the code is not the loops themselves but the decisions *around*
the loops: which dtype promotion applies, which of the five layout paths the
strides demand, whether a SIMD path is legal on this hardware, and what NumPy
does at the edges. The emitted IL is small and hot; the C# that decides which IL
to emit is where the intelligence lives.

The operation surface those kernels cover:

| Surface | Count | Examples |
| --- | ---: | --- |
| Binary operations | 17 | `add`, `subtract`, `multiply`, `divide`, bitwise, `power`, `floor_divide`, `arctan2`, `maximum`, `fmax` |
| Unary operations | 35 | trig, exp/log family, rounding, `abs`, `sign`, `reciprocal`, predicates, bitwise/logical not |
| Reductions | 20 | `sum`, `prod`, `min`, `max`, `arg*`, `mean`, `std`, `var`, and NaN-aware variants |
| Comparisons | 6 | `equal`, `not_equal`, `<`, `<=`, `>`, `>=` |
| Dtypes | 15 | bool, integer widths, char, half, single, double, decimal, complex |
| Binary layout paths | 5 | full SIMD, scalar-left/right broadcast, chunked SIMD, general strided |

## Two Generators, Two Contracts

There are two physically separate generator classes, and the reason is not
historical accident or arbitrary partitioning. It is a single question: **who
drives the loop?**

### `DirectILKernelGenerator` — the kernel owns the array

*Location: `src/NumSharp.Core/Backends/Kernels/Direct/`*

A Direct kernel is handed the whole array and told to get on with it. One
delegate call processes every element; the generated method owns the stride
walk, the coordinate odometer, the SIMD main loop, the scalar tail, and the
fallback logic. The array's shape and strides are just arguments.

```csharp
unsafe delegate void MixedTypeKernel(
    void* lhs,
    void* rhs,
    void* result,
    long* lhsStrides,   // element strides
    long* rhsStrides,
    long* shape,
    int ndim,
    long totalSize);
```

This is the right model when the operation genuinely owns its traversal: casts,
same-type and mixed-type binary ops, whole-array unary loops, axis reductions,
`take`/`put`/`place`/`search`, `trace`, `matmul`, `repeat`, `quantile`, and the
rest of the mature, directly-dispatched surface. It carries the bulk of the
system — 64 partials, ~42,500 lines, 52 of the 57 factories.

### `ILKernelGenerator` — the iterator owns the loop

*Location: `src/NumSharp.Core/Backends/Kernels/`*

An NDIter kernel is the opposite bargain. It processes **one inner-loop chunk**
per call and knows nothing about axes or coordinates; `NDIterRef` drives the
outer loop and advances the operand pointers between calls. This is a direct
port of NumPy's `PyUFuncGenericFunction` contract.

```csharp
unsafe delegate void NDInnerLoopFunc(
    void** dataptrs,    // current operand pointers
    long*  strides,     // per-operand BYTE stride for this chunk
    long   count,       // elements in this chunk
    void*  auxdata);    // op-specific extras (e.g. axis index)
```

Note the strides are **byte** strides, matching NumPy's C convention — a small
but load-bearing difference from the Direct contract's element strides. This is
the migration target for new iterator-driven ufunc work. Today it backs the
chunked `np.where` inner loop, the per-chunk reductions (including the
NumPy-faithful pairwise sum), and cumsum. The [NDIter](NDIter.md) page describes
the scheduling layer that feeds these kernels.

### The exception that proves the split

The tidy story — "Direct owns arrays, NDIter owns chunks" — has exactly two
deliberate exceptions, and knowing them saves an hour of confusion. Two *Direct*
partials emit the per-chunk `NDInnerLoopFunc` contract rather than a whole-array
kernel:

- **`DirectILKernelGenerator.InnerLoop.cs`** — the reusable inner-loop factory
  behind `np.evaluate` (the `NDExpr` DSL) and custom NDIter operations.
- **`DirectILKernelGenerator.WeightedSum.cs`** — the kernel behind `np.average`.

So the class a kernel lives in tells you its *file neighborhood*, but the
delegate signature tells you its *driving contract*. When in doubt, look at the
delegate.

### Relationship

```text
np.* / NDArray operator / DefaultEngine
        │
        ├─ resolves: shape, dtype promotion, out/where, layout
        │
        ├─ DirectILKernelGenerator ─────────── one call owns the whole array
        │      shape + strides + ndim are kernel arguments
        │
        └─ NDIterRef + ILKernelGenerator ───── iterator owns traversal
               kernel owns one inner-loop chunk
```

Both generators share one namespace, one set of operation enums, the same
kernel-key structs, the same SIMD and scalar reflection caches, and the same
type utilities. Only the loop-driving model differs.

### Which one to touch?

| You are changing | Start here | Why |
| --- | --- | --- |
| A mature whole-array op with existing direct dispatch | `DirectILKernelGenerator.*.cs` | The caller already passes shape/strides/ndim to a whole-array delegate. |
| A new ufunc-style inner loop | `ILKernelGenerator.<Op>.cs` | Let the iterator own broadcasting, coalescing, buffering, and chunk advancement. |
| A fused expression or custom operation | `DirectILKernelGenerator.InnerLoop.cs` + `NDExpr` | The inner-loop factory supplies the SIMD shell and the scalar-strided fallback. |
| Reflection or intrinsic lookup behavior | `VectorMethodCache.cs` / `ScalarMethodCache.cs` | Eligibility checks and emitters must agree on which runtime methods exist. |
| Cache observability or test reset | `GeneratedDelegates.cs` | Public live counts and internal clear hooks are centralized there. |


## Source Inventory

This is the precise, refreshable audit that the [At a Glance](#at-a-glance)
figures round from.

### Generator totals

| Area | Files | Lines | `new DynamicMethod` | `CreateDelegate` | Gen-kernel caches | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct whole-array generator | 64 | 42,522 | 52 | 52 | 41 | Primary high-performance kernel set |
| NDIter inner-loop generator | 5 | 1,987 | 5 | 4 | 4 | NumPy-style per-chunk kernels |
| Shared root infrastructure | 15 | 3,561 | 0 | 0 | 0 | Keys, delegates, SIMD/scalar reflection, stride detection |

Across the tree: **7,012** raw `Emit(...)` calls, **390** `EmitCall(...)` sites,
**484** local slots, and **524** labels are declared by hand.

### Direct generator by category

The 64 Direct partials group cleanly by what they do. The line counts are less
interesting than the silhouette they form — casts and axis reductions alone are a
third of the generator, because that is where NumPy hides most of its corner
cases.

| Category | Files | Lines | What lives there |
| --- | ---: | ---: | --- |
| Axis reductions | 7 | 7,154 | axis sum/prod/min/max/mean/std/var/arg/NaN and the widening path |
| Casts | 15 | 7,020 | contiguous, strided, masked, half, complex, bool, float→int, subword |
| Selection and indexing | 12 | 5,521 | `where`, `nonzero`, `argwhere`, `take`, `put`, `place`, ravel/unravel, search, indices, filter |
| Specialized kernels | 9 | 4,944 | clip, modf, quantile, weighted sum, repeat, the inner-loop factory, matmul, trace, scalar delegates |
| Reductions (flat) | 4 | 3,568 | flat, arg, boolean, and NaN reductions |
| Binary operations | 3 | 3,557 | same-type binary, mixed-type binary, shifts |
| Unary operations | 6 | 2,875 | unary dispatch, math, decimal, predicate, vector, strided |
| Scan | 1 | 2,520 | cumsum/cumprod and axis scans |
| Core emit helpers | 1 | 2,038 | type mapping, vector width, scalar/vector emit primitives |
| Masking | 3 | 1,683 | boolean masks, NaN masks, var/std mask helpers |
| Comparison | 1 | 1,317 | comparison kernels and scalar comparison delegates |
| Copy and aliasing | 2 | 325 | copy kernels and storage field-alias helpers |

### NDIter generator partials

| File | Lines | Responsibility |
| --- | ---: | --- |
| `ILKernelGenerator.cs` | 55 | Contract documentation and the partial root |
| `ILKernelGenerator.Reduction.cs` | 825 | Per-chunk reductions |
| `ILKernelGenerator.Reduction.Pairwise.cs` | 609 | NumPy-faithful SIMD pairwise sum |
| `ILKernelGenerator.Where.cs` | 333 | Per-chunk `np.where` with SIMD conditional-select |
| `ILKernelGenerator.Scan.cs` | 165 | Per-chunk cumsum inner loop |

### Shared infrastructure

| File | Responsibility |
| --- | --- |
| `KernelOp.cs` | The `BinaryOp` / `UnaryOp` / `ReductionOp` / `ComparisonOp` / `ExecutionPath` enums |
| `BinaryKernel.cs`, `ReductionKernel.cs`, `ScalarKernel.cs`, `CopyKernel.cs` | Cache-key structs and delegate contracts |
| `VectorMethodCache.cs` | Cached closed `MethodInfo` for `Vector128/256/512` and x86 intrinsics |
| `ScalarMethodCache.cs` | Cached scalar `Math`, `MathF`, decimal, and helper lookups |
| `StrideDetector.cs` | Contiguous / scalar-broadcast / chunked-SIMD / general path classification |
| `GeneratedDelegates.cs` | Cache visibility and reset hooks for tests |
| `SimdMatMul*.cs`, `SimdDot.cs` | Hand-written SIMD primitives for linear algebra |
| `FiniteScan.cs` | Fused NaN/inf-poison SIMD scan behind `np.asarray_chkfinite` |
| `IndexCollector.cs` | Growable unmanaged index buffer for selection kernels |

### Refreshing the inventory

The headline figures drift every time a kernel is added — this document has gone
stale before. Re-derive them after large kernel changes; these snippets are the
source of truth for the numbers above.

```powershell
$direct = Get-ChildItem src\NumSharp.Core\Backends\Kernels\Direct -Filter 'DirectILKernelGenerator*.cs'
$inner  = Get-ChildItem src\NumSharp.Core\Backends\Kernels -Filter 'ILKernelGenerator*.cs'
$shared = Get-ChildItem src\NumSharp.Core\Backends\Kernels -Filter '*.cs' |
    Where-Object { $_.Name -notlike 'ILKernelGenerator*' }

@($direct + $inner + $shared) |
    ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } |
    Measure-Object -Sum
```

```powershell
$files = @($direct + $inner + $shared)
$text  = ($files | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
[regex]::Matches($text, 'new\s+DynamicMethod\s*\(').Count   # DynamicMethod sites -> 57
[regex]::Matches($text, 'CreateDelegate\s*(<|\()').Count    # CreateDelegate sites -> 56
[regex]::Matches($text, '\.Emit\(').Count                   # raw Emit calls -> 7,012
[regex]::Matches($text, '\.EmitCall\(').Count               # EmitCall sites -> 390
[regex]::Matches($text, 'DeclareLocal').Count               # local slots -> 484
[regex]::Matches($text, 'DefineLabel').Count                # labels -> 524
[regex]::Matches($text, '(?m)^\s*(public |internal |private |protected |static |unsafe |sealed )*delegate\s').Count  # delegate types -> 40
[regex]::Matches($text, '(?m)^\s*(public |internal |private |protected |static |sealed )*unsafe delegate\s').Count   # unsafe delegates -> 39
```

The Direct-generator cache-field count uses a narrower scope. The regex is
anchored to declaration sites so the two `ConcurrentDictionary<...>` mentions
that appear only in comments are not counted — it yields all 43 declared fields,
of which 41 hold generated kernels and 2 hold reflection results:

```powershell
# Direct-generator cache fields (declaration sites only) -> 43
$directText = ($direct | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
[regex]::Matches($directText, '(?m)^\s*(private|internal|public|protected).*ConcurrentDictionary<').Count
```

## The Life of a Kernel

Every generated kernel — Direct or NDIter — travels the same road from a user
call to native code:

```text
1. Caller enters DefaultEngine / np.* / an NDArray operator
2. NumSharp resolves dtype promotion, output dtype, out/where, and shape
3. Layout is classified from strides and shape
4. A structural cache key is built
5. The cache is checked
6. On a miss, a DynamicMethod is emitted
7. CreateDelegate materializes the callable kernel
8. The delegate is cached and invoked with raw storage pointers
```

On a warm path only steps 5 and 8 run: a dictionary lookup, then a direct call
into JIT-compiled native code. The first call of a given shape pays the emit-and-JIT
cost — microseconds — and every call after it is free of that overhead.

**Correctness never depends on the fast path existing.** Kernel entry points
follow a `TryGet*Kernel` discipline: if IL generation is disabled, the dtype is
unsupported, or reflection/emit throws for any reason, the method returns `null`
and the caller drops to a scalar reference loop. This catch-all-return-`null`
pattern appears at 14 sites across the Direct partials. It is why the generator
can be aggressive — a path that only ever makes things faster, and quietly steps
aside when it cannot, is a path you can trust.

## Choosing the Loop

Layout classification is where the generator earns its keep, and the two
generators make the *same* decision in two revealingly different ways.

### The Direct model: bake the path into the key

For a whole-array binary op, `StrideDetector.Classify<T>()` inspects the operand
strides and shape and returns one `ExecutionPath`, in priority order:

| Path | Trigger | Kernel shape |
| --- | --- | --- |
| `SimdFull` | both operands fully C-contiguous | flat vector loop + scalar tail |
| `SimdScalarRight` | RHS strides all zero | load RHS once, splat into a vector |
| `SimdScalarLeft` | LHS strides all zero | load LHS once, splat into a vector |
| `SimdChunk` | inner dimension contiguous or broadcast | outer coordinate loop + inner SIMD chunk |
| `General` | arbitrary strides | coordinate-based scalar loop |

The chosen path is then **part of the cache key**:
`MixedTypeKernelKey(lhsType, rhsType, resultType, op, path)`. Each combination
gets its own branch-free specialized kernel — up to 12 × 12 × 5 × 5 = 3,600
possible binary kernels, and 12 × 12 × 6 × 5 = 4,320 comparison kernels, though
in practice only the handful your program actually exercises are ever emitted.
Because the path is decided before the kernel is even looked up, the hot loop
contains no layout branching at all.

One optimization is worth calling out because it turns a slow path into a fast
one for free. Before classifying, `DefaultEngine` coalesces adjacent dimensions
whose strides are compatible across both operands and the result. An
F-contiguous N-D array collapses to a single contiguous 1-D run, so the
classifier promotes it from `General` (the ~13× slower coordinate loop) straight
to `SimdFull`. The layout the user *wrote* and the layout the kernel *sees* need
not be the same shape.

### The NDIter model: dispatch at runtime

The inner-loop factory (`DirectILKernelGenerator.InnerLoop.cs` — a Direct partial
that emits the per-chunk contract, per [the exception above](#the-exception-that-proves-the-split))
cannot bake the path into the key, because a single compiled kernel is reused
across chunks whose strides differ from one call to the next. So it emits the dispatch *into the kernel*: cheap integer compares at the
top of the body check each operand's byte stride against its element size and
branch to the matching loop.

| Tier | Entry point | Who emits what |
| --- | --- | --- |
| Raw IL | `CompileRawInnerLoop(body, key)` | caller emits the entire `NDInnerLoopFunc` body |
| Templated elementwise | `CompileInnerLoop(operandTypes, scalarBody, vectorBody, key)` | factory emits the loop shell; caller supplies scalar/vector element bodies |
| Expression DSL | `NDExpr` compilation via `np.evaluate` | the expression tree emits scalar/vector bodies into the templated shell |

Inside that templated shell the factory can emit as many as five runtime-selected
loops from one body pair: an all-contiguous SIMD loop (4× unrolled, plus a
one-vector remainder and a scalar tail), a broadcast-binary SIMD loop that
pre-broadcasts the scalar operand once via `Vector.Create` outside the loop, an
AVX2 hardware-gather loop for genuinely strided inputs, a mixed-dtype scalar
contiguous loop, and — always present, the floor everything falls to — a
scalar-strided loop. Two dispatch philosophies, one classification; the split
mirrors the loop-ownership split exactly.

## SIMD Strategy

The vector layer adapts to the hardware it finds at startup.
`DirectILKernelGenerator.VectorBits` is set once to 512, 256, 128, or **0** — and
that 0 matters: it forces the scalar path everywhere, so the whole system
degrades cleanly on a machine with no SIMD at all.

`VectorMethodCache` is the pivot. Reflection over `Vector128/256/512` is
expensive and easy to get subtly wrong, so the cache resolves each `MethodInfo`
once, already closed over dtype and width, and hands back the same handle to
every emitter that asks — loads, stores, operators, comparisons, the various
`Create` overloads, narrows, widens, `As<TFrom,TTo>`, `Zero`, and the x86
intrinsic entry points.

That last category is a genuine performance lever, not a portability footnote.
The portable `Vector256.*` helpers JIT-emit roughly **1.8–2× slower** code than
the platform-specific `System.Runtime.Intrinsics.X86.Avx/Avx2.*` methods on the
same AVX2 host — same IL `call` instruction, different code-gen path. So when the
host supports them (`UseX86_256/128/512`), the cache routes load, store, add,
sub, mul, div, min, max, sqrt, and the bitwise ops through the x86 intrinsic
`MethodInfo`. The routing table also encodes where the hardware simply has no
instruction — there is no integer SIMD divide, and AVX2 lacks int64 min/max and
multiply (those need AVX-512) — and returns `null` so the caller falls back
rather than emitting invalid IL.

On top of that routing, the emitted loops use the classic throughput repertoire:

- a vector main loop plus a scalar tail
- 4×-unrolled vector bodies where the factory can express them
- scalar values pre-broadcast into a vector once, outside the loop
- horizontal vector reductions for final sums / min / max
- x86 gather for selected strided float and double paths
- `ConditionalSelect` for mask-driven selection
- typed widen / narrow and bit reinterpretation instead of boxing or virtual calls

## NumPy Semantics in the Fast Path

Here is the claim that separates this generator from a generic SIMD loop
compiler: **it carries NumPy's behavior — including the parts NumPy itself is
slightly embarrassed by — into specialized machine code.** Speed with the wrong
answer is a bug; the fast path is only allowed to exist because it is also the
correct path.

### NEP50 promotion

Kernel keys carry input, accumulator, and result dtypes separately, so a kernel
can compute at a different width than it stores. Reductions and scans honor
NumPy's widened-accumulator rules — integer `sum` and `prod` accumulate into
64-bit, integer `mean` accumulates in `double` — and the widened axis path in
`DirectILKernelGenerator.Reduction.Axis.Widening.cs` does the sign/zero extension
and float conversion in AVX2. It streams input rows through an 8192-element
output-slab block that stays L2-resident, mirroring NumPy's own buffer-size
pattern, so a leading-axis reduction reads memory sequentially instead of
gathering a strided column per output element.

### Pairwise floating sum

`ILKernelGenerator.Reduction.Pairwise.cs` is the sharpest example of "same answer
*and* fast." NumPy sums floats with pairwise summation for O(lg n) rounding
error; a naive C# port matched NumPy's summation order bit-for-bit but was
scalar, because the .NET JIT will not auto-vectorize an eight-accumulator loop
the way GCC does. The emitter's trick is to map NumPy's eight accumulators onto
SIMD lanes so accumulator `r[k]` still gathers elements `{k, k+8, k+16, …}` — the
result is *independent of vector width*, so V128, V256, and V512 all produce
identical bits, bit-for-bit equal to `np.add.reduce`. The file records the
measurement: for `float64`, `axis=1`, on a 1000×1000 array on an AVX2 host,
scalar pairwise ran at 0.267 ms and the emitted SIMD pairwise at 0.123 ms — a
2.18× self-speedup that also beats NumPy 2.4.2's 0.207 ms by 1.69×.

### NaN, predicate, and edge semantics

NaN-aware reductions, `fmax`/`fmin` versus `maximum`/`minimum`, and
`isnan`/`isfinite`/`isinf` are first-class generated paths, not scalar
afterthoughts, and they distinguish NaN-propagating from NaN-ignoring behavior
exactly where NumPy does. The transcendental and edge-case corners route through
NumSharp helpers rather than the BCL because the BCL diverges from NumPy at the
C99 boundaries: complex functions go through `NDComplexMath` (Annex-G non-finite
tables, branch-cut signs, signed zeros), integer `power` through
`NDIntegerPower` (dtype-native wrapping the double round-trip would lose),
floored division and remainder through `NDDivision`, and `float`→integer casts
through the `Converts` table (truncate-toward-zero with a type-min sentinel on
NaN/overflow, where the IL `conv` opcodes would saturate and yield 0).

A representative small trap: a `bool` array is *logically* only `{0, 1}`, but a
`np.frombuffer` view or foreign interop buffer can legally hold a byte like 255.
So `EmitConvertTo` normalizes `!= 0 → 1` **before** widening a bool to any
numeric type — otherwise `sum` over such a buffer would add 255 instead of
counting a True. Even `NegateHalf` earns a bespoke helper: the BCL's
`Half.op_UnaryNegation` round-trips through `float`, which benchmarked 7.3×
slower and made half-precision negate the single worst cell in the elementwise
matrix (~0.14× of NumPy), so the emitter flips the sign bit directly instead.

### View and broadcast semantics

Generated kernels receive base pointers that already fold in `Shape.offset`, then
use element or byte strides to honor sliced, reversed, transposed, and broadcast
views. Broadcast reads use stride zero; broadcast *writes* stay blocked at the
ndarray/view layer, because a broadcast view is read-only by construction.

## Technique Highlights

### Cast kernels — where the corner cases live

The cast subsystem is the densest part of the generator — 15 files, ~7,000 lines
— for the same reason casting is where NumPy keeps its strangest behavior. The
notable pieces:

- `float`/`double` → signed integer through truncating paths that match NumPy's
  NaN/overflow sentinel, not C#'s saturating conversion.
- `float`/`double` → `uint32`/`uint64` with NumPy's modular-wrap behavior rather
  than a checked conversion.
- `Half` widen/narrow through bit-level (Giesen) conversion helpers.
- complex → int and complex → bool paths that deinterleave real and imaginary
  lanes as needed.
- subword same-size copies, narrows, and widens for 1- and 2-byte dtypes.
- strided kernels that detect an inner unit stride at runtime and still take a
  vector inner loop even when the outer dimensions are sliced.
- unsupported-pair caches, so a dtype combination that failed to emit once is
  never retried.

### Axis reductions — not one loop but ten

Reducing along an axis is not a single fallback. The Direct generator carries
separate paths for same-dtype SIMD reductions, widening reductions, arg
reductions, boolean `all`/`any`, NaN-aware reductions, the var/std two-pass
algorithm, leading-axis C-contiguous streaming, innermost-axis contiguous row
reductions, sliced axis-0 layouts with contiguous inner slabs, and a scalar
general fallback. The leading-axis case is the one that pays off most: reducing
`axis=0` of a C-contiguous array streams whole rows sequentially while the output
slab stays hot, instead of chasing a strided column for every output element.

### `np.where` — read the bool, don't cast it

The NDIter `np.where` inner loop is deliberately not a reuse of `NDExpr.Where`.
It reads the condition operand as a raw bool byte and selects directly: on an
inner-contiguous chunk it emits SIMD mask expansion plus `ConditionalSelect`;
otherwise it runs scalar-strided IL. The point is to skip a per-element
bool-to-output-dtype cast on the common path.

### Custom inner loops — the factory behind `np.evaluate`

`DirectILKernelGenerator.InnerLoop.cs` is a reusable inner-loop factory, shared by
NDIter custom operations and `np.evaluate`. Its templated tier wraps
user-supplied scalar and vector IL bodies in a ready-made loop shell:

```text
load dataptrs and byte strides
if every operand is inner-contiguous:
    run 4×-unrolled SIMD loop → one-vector remainder → scalar tail
else if AVX2 gather applies (strided, 32/64-bit, V256):
    gather inputs, run vector body, contig-store or per-lane scatter
else:
    run the scalar-strided fallback
return
```

That gives a custom operation the same generated loop shape as a built-in ufunc
without hand-writing any pointer arithmetic. The gather path is worth a note: its
lane-index budget is `GatherStrideLimit = int.MaxValue / 8` (the largest lane
offset is 7× the stride), and it applies only to 32- and 64-bit dtypes at 256-bit
width — precisely the set NumPy's own `npyv_loadn` gathers.

### Reflection caches — one source of truth

Runtime IL emission needs a `MethodInfo` for every intrinsic it calls.
`VectorMethodCache` and `ScalarMethodCache` keep those lookups centralized and
closed over dtype and width. The payoff is not just avoiding dozens of scattered
reflection searches; it is that the *eligibility check* and the *emitter* consult
the same cache, so they can never disagree about whether the running BCL actually
exposes a given method at a given width.

## Cache Families

The Direct generator surfaces 41 generated-kernel caches and the NDIter generator
4, for the 45 that `GeneratedDelegates` exposes. (The Direct generator declares 43
`ConcurrentDictionary` fields in all; the other two cache reflection results, not
kernels, and so are not counted here.) They group by family:

| Family | Cache examples |
| --- | --- |
| Elementwise | `_contiguousKernelCache`, `_mixedTypeCache`, `_unaryCache`, `_stridedUnaryCache`, `_comparisonCache` |
| Scalar delegates | `_unaryScalarCache`, `_binaryScalarCache`, `_comparisonScalarCache` |
| Casts | `_castCache`, `_stridedCastCache`, `_maskedCastCache`, `_innerCastCache`, plus unsupported-pair caches |
| Reductions | `_elementReductionCache`, `_nanElementReductionCache`, `_axisReductionCache`, `_nanAxisReductionCache`, `_boolAxisReductionCache` |
| Scans | `_scanCache`, `_axisScanCache` |
| Selection / indexing | `_whereKernelCache`, the scalar `where` caches, `_argwhereCount`, `_argwhereFlat`, `_searchCache`, `_filterAxis` |
| Other helpers | `_copyKernelCache`, `_matmulKernelCache`, `_quantileKernelCache`, the repeat caches, `_trace`, `_weightedSumCache`, the storage-alias cache |
| Custom inner loops | `_innerLoopCache`, surfaced through `GeneratedDelegates` for tests |

Every key is structural: it encodes exactly enough to make one emitted body valid
— operation, dtype tuple, accumulator/result dtype, execution path, copy mode,
quantile method, or a caller-supplied custom key. Two requests that would emit
identical IL share a key; two that would not, don't.

`GeneratedDelegates` exposes a public live count per cache (and a `TotalCount`)
plus internal clear hooks for tests. The reflection caches are deliberately
excluded from that registry — they hold `MethodInfo` lookups, not generated
executable kernels, and counting them would blur the one number tests actually
care about: how many kernels this process has compiled.

## Working on Kernels

### Adding or changing one

1. **Probe NumPy first.** Capture dtype, shape, stride, NaN, empty, scalar,
   broadcast, and output-dtype behavior from actual NumPy 2.x. The generator's
   job is to reproduce that, so it has to be observed before it can be emitted.
2. **Choose the driving model.** `ILKernelGenerator` for NDIter chunk loops and
   new ufunc-style work; `DirectILKernelGenerator` when the op owns a whole-array
   traversal or is part of the existing direct surface.
3. **Define the cache key before the emitter.** If a decision changes the emitted
   IL, it belongs in the key. A missing field silently reuses the wrong body.
4. **Keep the generated signature narrow.** Raw pointers, strides, shape, axis,
   sizes — no managed allocations in the hot body.
5. **Separate path selection from loop emission.** The house style is a
   dispatcher, then path-specific `Generate*` methods, then `Emit*Loop` helpers.
6. **Handle all 15 dtypes,** or document the unsupported ones and give them a
   clear fallback.
7. **Preserve view semantics.** Offset is already in the base pointer; strides
   must handle non-contiguous, negative, and broadcast cases.
8. **Add NumPy-derived tests** — contiguous, strided, broadcast, empty/scalar,
   NaN, and promotion cases, scaled to the operation's risk.
9. **Benchmark in Release only.** Debug JIT and intrinsic behavior differ enough
   to invalidate kernel timings. Ratios follow the project convention
   `NumPy_ms / NumSharp_ms`, so a value above `1.0` means NumSharp is faster —
   see the [Benchmarks dashboard](benchmarks-dashboard.md).

### Debugging one

When a generated path misbehaves:

- Confirm `DirectILKernelGenerator.Enabled` and `VectorBits`.
- Inspect the cache key — a missing field is the classic "wrong IL" bug.
- Check which layout path was chosen *before* the generator ran.
- For NDIter kernels, inspect the current chunk's byte strides; a natural byte
  stride is required before the SIMD chunk path fires.
- Use `GeneratedDelegates.InnerLoopCount` and the reset hooks to prove a custom
  inner loop is cached (or regenerated).
- If a `TryGet*Kernel` returns `null`, read the fallback path before assuming the
  bug is in emitted IL — the fast path may simply have declined.
- Remember `DynamicMethod` IL is not step-through friendly. Log around path
  selection, or temporarily emit into a debuggable assembly, rather than trying
  to single-step the delegate.

## What Makes This System Unusual

Most managed numerical libraries pick a lane: generic managed loops, or a native
backend. NumSharp sits between them on purpose — it generates managed IL at
runtime, lets the .NET JIT lower that IL to native machine code, and keeps NumPy
layout and dtype semantics *inside* the generated body. No single ingredient is
novel; the combination is:

- NumPy-style dtype and accumulator rules, encoded in the kernel key.
- Unmanaged ndarray storage and pointer-level kernels.
- Runtime specialization by dtype tuple, operation, and layout.
- Width-adaptive SIMD across V128, V256, and V512.
- x86 intrinsic routing when it out-generates the portable vector APIs.
- Whole-array kernels for the mature direct paths.
- NDIter inner-loop kernels for iterator-style execution and fusion.
- Pairwise floating reductions that preserve NumPy's summation order while
  recovering SIMD throughput.
- Cast kernels that encode NumPy's odd corners — modular unsigned float casts,
  half conversion, complex deinterleaving, subword lanes, masked output.

That is why the IL generator is not merely an optimization layer bolted onto a
correct-but-slow core. It is the layer where NumSharp turns NumPy's compatibility
rules into specialized machine code — and the reason a managed array library can
answer exactly like NumPy while running at native speed.
