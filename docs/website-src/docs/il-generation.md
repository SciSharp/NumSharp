# IL Kernel Generation in NumSharp

NumSharp's kernel layer is a runtime compiler. It emits `DynamicMethod` bodies
specialized for the exact NumPy operation, dtype combination, memory layout, and
SIMD width seen at the call site, then caches the resulting delegate for the
rest of the process.

This is the performance backbone behind elementwise ufuncs, reductions, scans,
casts, selection/indexing helpers, `np.where`, `np.average`, `np.evaluate`, and
the NDIter custom-operation bridge.

## Two Generators

There are two physically separate generators because there are two different
kernel-driving contracts.

### `DirectILKernelGenerator`

Location: `src/NumSharp.Core/Backends/Kernels/Direct/`

`DirectILKernelGenerator` emits whole-array kernels. The caller invokes one
delegate for the entire array operation, and the generated method owns the
stride walk, coordinate odometer, SIMD loop, scalar tail, and fallback logic.

Typical shape:

```csharp
unsafe delegate void MixedTypeKernel(
    void* lhs,
    void* rhs,
    void* result,
    long* lhsStrides,
    long* rhsStrides,
    long* shape,
    int ndim,
    long totalSize);
```

Use this model when the kernel needs to own the whole array traversal: casts,
same-type binary fast paths, mixed-type binary operations, whole-array unary
loops, axis reductions, selection helpers, `take`, `put`, `place`, `search`,
`trace`, `matmul`, `repeat`, `quantile`, and similar operations.

### `ILKernelGenerator`

Location: `src/NumSharp.Core/Backends/Kernels/`

`ILKernelGenerator` emits NumPy-style inner loops driven by `NDIterRef`.
The iterator advances operand pointers between chunks; the emitted kernel only
processes one inner-loop chunk.

Contract:

```csharp
unsafe delegate void NDInnerLoopFunc(
    void** dataptrs,
    long* strides,
    long count,
    void* auxdata);
```

The `strides` array uses byte strides, matching NumPy's ufunc convention. This
model is the migration target for new iterator-driven ufunc work. `ILKernelGenerator`
itself emits the chunked kernels behind `np.where`, the per-chunk reductions
(including NumPy-faithful pairwise sum), and cumsum. The same `NDInnerLoopFunc`
contract is also emitted by two `DirectILKernelGenerator` partials that own an
inner chunk rather than a whole array: `InnerLoop.cs` (the factory behind
`np.evaluate` and custom NDIter operations) and `WeightedSum.cs` (behind
`np.average`). See [NDIter](NDIter.md) for the iterator scheduling layer that
feeds them.

### Relationship

```text
np.* / NDArray operator / DefaultEngine
        |
        +-- shape, dtype, promotion, out/where, layout decisions
        |
        +-- DirectILKernelGenerator
        |      one call owns the whole array traversal
        |      shape + strides + ndim are kernel arguments
        |
        +-- NDIterRef + ILKernelGenerator
               iterator owns traversal
               kernel owns one inner-loop chunk
```

Both generators share the same kernel namespace, operation enums, key structs,
SIMD reflection cache, scalar reflection cache, and type utilities. The split is
about who drives the loop.

### Which One To Touch?

| You are changing | Start here | Why |
| --- | --- | --- |
| A mature whole-array operation with existing direct dispatch | `DirectILKernelGenerator.*.cs` | The caller already passes shape/strides/ndim to a whole-array delegate. |
| A new ufunc-style inner loop | `ILKernelGenerator.<Op>.cs` | The iterator should own broadcasting, coalescing, buffering, and chunk advancement. |
| A fused expression or custom operation | `DirectILKernelGenerator.InnerLoop.cs` and `NDExpr` | The inner-loop factory supplies the SIMD shell and scalar-strided fallback. |
| Reflection or intrinsic lookup behavior | `VectorMethodCache.cs` / `ScalarMethodCache.cs` | Eligibility checks and emitters must agree on available runtime methods. |
| Cache observability or test reset behavior | `GeneratedDelegates.cs` | Public counts and internal clear hooks are centralized there. |


## Source Inventory

### At A Glance

| Measurement | Current source snapshot |
| --- | ---: |
| Kernel source files | 83 |
| Kernel source lines | 42,822 |
| Whole-array `DirectILKernelGenerator` partials | 64 files / 38,004 lines |
| NDIter `ILKernelGenerator` partials | 5 files / 1,833 lines |
| Shared kernel infrastructure files | 14 files / 2,985 lines |
| `new DynamicMethod(...)` constructor sites | 57 |
| `CreateDelegate(...)` sites | 56 |
| Raw IL `Emit(...)` calls in the kernel tree | 7,012 |
| `EmitCall(...)` sites | 390 |
| Local slots declared by emitters | 484 |
| Labels declared by emitters | 524 |
| Kernel delegate type declarations | 40 |
| `unsafe delegate` contracts | 39 |
| Direct generator cache declarations | 43 `ConcurrentDictionary` fields |
| Test files touching kernel, NDIter, or SIMD terminology | 136 |

The kernel surface is broad:

| Surface | Count | Examples |
| --- | ---: | --- |
| Binary operations | 17 | `add`, `subtract`, `multiply`, `divide`, bitwise ops, `power`, `floor_divide`, `arctan2`, `maximum`, `fmax` |
| Unary operations | 35 | trig, exp/log family, rounding, `abs`, `sign`, `reciprocal`, predicates, bitwise/logical not |
| Reductions | 20 | `sum`, `prod`, `min`, `max`, `arg*`, `mean`, `std`, `var`, NaN-aware variants |
| Comparisons | 6 | `equal`, `not_equal`, `<`, `<=`, `>`, `>=` |
| Dtypes | 15 | bool, integer widths, char, half, single, double, decimal, complex |
| Binary layout paths | 5 | full SIMD, scalar-left/right broadcast, chunked SIMD, general strided |

These numbers come from the checked-in source, not from handwritten estimates.

### Generator Totals

| Area | Files | Lines | `new DynamicMethod` sites | `CreateDelegate` sites | Cache declarations | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct whole-array generator | 64 | 38,004 | 52 | 52 | 43 | Primary high-performance kernel set |
| NDIter inner-loop generator | 5 | 1,833 | 5 | 4 | 4 | NumPy-style per-chunk kernels |
| Shared root infrastructure | 14 | 2,985 | 0 | 0 | 6 | Kernel keys, delegates, SIMD reflection, stride detection |

### Direct Generator By Category

| Category | Files | Lines | What lives there |
| --- | ---: | ---: | --- |
| Axis reductions | 7 | 6,581 | axis sum/prod/min/max/mean/std/var/arg/NaN/widening paths |
| Casts | 15 | 6,429 | contiguous, strided, masked, half, complex, bool, float-to-int, subword kernels |
| Selection and indexing | 12 | 4,818 | `where`, `nonzero`, `argwhere`, `take`, `put`, `place`, ravel/unravel, search, indices, filter |
| Specialized kernels | 9 | 4,380 | clip, modf, quantile, weighted sum, repeat, inner-loop factory, matmul, trace, scalar delegates |
| Reductions | 4 | 3,102 | flat reductions, arg reductions, boolean reductions, NaN reductions |
| Binary operations | 3 | 3,073 | same-type binary, mixed-type binary, shifts |
| Unary operations | 6 | 2,534 | unary dispatch, math, decimal, predicate, vector, strided unary |
| Scan | 1 | 2,308 | cumsum/cumprod and axis scans |
| Core emit helpers | 1 | 1,886 | type mapping, vector width, scalar/vector emit primitives |
| Masking | 3 | 1,462 | boolean masks, NaN masks, var/std mask helpers |
| Comparison | 1 | 1,137 | comparison kernels and scalar comparison delegates |
| Copy and aliasing | 2 | 294 | copy kernels and storage field alias helpers |

### NDIter Generator Partials

| File | Lines | Responsibility |
| --- | ---: | --- |
| `ILKernelGenerator.cs` | 54 | Contract documentation and partial root |
| `ILKernelGenerator.Reduction.cs` | 775 | Per-chunk reductions |
| `ILKernelGenerator.Reduction.Pairwise.cs` | 562 | NumPy-faithful SIMD pairwise sum |
| `ILKernelGenerator.Where.cs` | 290 | Per-chunk `np.where` with SIMD conditional-select |
| `ILKernelGenerator.Scan.cs` | 152 | Per-chunk cumsum inner loop |

### Shared Infrastructure

| File | Responsibility |
| --- | --- |
| `KernelOp.cs` | `BinaryOp`, `UnaryOp`, `ReductionOp`, `ComparisonOp`, `ExecutionPath` |
| `BinaryKernel.cs`, `ReductionKernel.cs`, `ScalarKernel.cs`, `CopyKernel.cs` | cache keys and delegate contracts |
| `VectorMethodCache.cs` | cached closed `MethodInfo` lookups for `Vector128/256/512` and x86 intrinsics |
| `ScalarMethodCache.cs` | cached scalar `Math`, `MathF`, decimal, and helper method lookups |
| `StrideDetector.cs` | contiguous, scalar-broadcast, chunked-SIMD, and general path classification |
| `GeneratedDelegates.cs` | cache visibility and reset hooks for tests |
| `SimdMatMul*.cs`, `SimdDot.cs` | hand-written SIMD primitives used by linear algebra paths |
| `IndexCollector.cs` | growable unmanaged index collection for selection kernels |

### Refreshing The Inventory

The headline numbers above should be refreshed after large kernel changes. These
PowerShell snippets are the source of truth used for this page:

```powershell
$direct = Get-ChildItem src\NumSharp.Core\Backends\Kernels\Direct -Filter 'DirectILKernelGenerator*.cs'
$inner = Get-ChildItem src\NumSharp.Core\Backends\Kernels -Filter 'ILKernelGenerator*.cs'
$shared = Get-ChildItem src\NumSharp.Core\Backends\Kernels -Filter '*.cs' |
    Where-Object { $_.Name -notlike 'ILKernelGenerator*' }

@($direct + $inner + $shared) |
    ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } |
    Measure-Object -Sum
```

```powershell
$files = @($direct + $inner + $shared)
$text = ($files | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
[regex]::Matches($text, 'new\s+DynamicMethod\s*\(').Count   # DynamicMethod sites -> 57
[regex]::Matches($text, 'CreateDelegate\s*(<|\()').Count    # CreateDelegate sites -> 56
[regex]::Matches($text, '\.Emit\(').Count                   # raw Emit calls -> 7,012
[regex]::Matches($text, '\.EmitCall\(').Count               # EmitCall sites -> 390
[regex]::Matches($text, 'DeclareLocal').Count               # local slots -> 484
[regex]::Matches($text, 'DefineLabel').Count                # labels -> 524
[regex]::Matches($text, '(?m)^\s*(public |internal |private |protected |static |unsafe |sealed )*delegate\s').Count  # delegate types -> 40
[regex]::Matches($text, '(?m)^\s*(public |internal |private |protected |static |sealed )*unsafe delegate\s').Count   # unsafe delegates -> 39
```

The Direct-generator cache-field count and the test-file tally use narrower
scopes. The cache regex is anchored to declaration sites so the two
`ConcurrentDictionary<...>` mentions that appear only in comments are not counted:

```powershell
# Direct-generator cache fields (declaration sites only) -> 43
$directText = ($direct | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
[regex]::Matches($directText, '(?m)^\s*(private|internal|public|protected).*ConcurrentDictionary<').Count

# Test files mentioning kernel / NDIter / SIMD terminology -> 136
(Get-ChildItem test -Recurse -Filter *.cs | Select-String -Pattern 'kernel|NDIter|SIMD' -List).Count
```

## Kernel Lifecycle

Every generated kernel follows the same high-level lifecycle:

```text
1. Caller enters DefaultEngine / np.* / NDArray operator
2. NumSharp resolves dtype promotion, output dtype, out/where, and shape
3. Layout is classified from strides and shape
4. A structural key is built
5. The cache is checked
6. On miss, a DynamicMethod is emitted
7. CreateDelegate materializes the callable kernel
8. Delegate is cached and invoked with raw storage pointers
```

On a hot path, steps 5 and 8 dominate: a dictionary lookup followed by a direct
delegate call into JIT-compiled native code. The first call pays the emit and JIT
cost; subsequent calls reuse the delegate.

`TryGet*Kernel` methods intentionally degrade gracefully. If IL emission is
disabled, unsupported, or fails during reflection/emit, the method returns
`null` and the caller falls back to another path. This keeps correctness first
while still giving fast paths room to be aggressive.

## Execution Paths

### Direct Binary Paths

`StrideDetector.Classify<T>()` chooses the binary path from element strides and
shape:

| Path | Trigger | Kernel shape |
| --- | --- | --- |
| `SimdFull` | both operands are fully C-contiguous | flat vector loop plus scalar tail |
| `SimdScalarRight` | RHS strides are all zero | load RHS once, splat into a vector |
| `SimdScalarLeft` | LHS strides are all zero | load LHS once, splat into a vector |
| `SimdChunk` | inner dimension is contiguous or broadcast | outer coordinate loop plus inner SIMD chunk |
| `General` | arbitrary strides | coordinate-based scalar loop |

The same dtype can take different generated IL depending on layout. Contiguous
arrays get pointer increments and vector loads. Broadcast inputs get stride-zero
splat paths. Sliced and transposed views keep NumPy view semantics through
coordinate-derived offsets.

### NDIter Inner-Loop Paths

The inner-loop factory (`DirectILKernelGenerator.InnerLoop.cs`) has a different
split:

| Tier | Entry point | Who emits what |
| --- | --- | --- |
| Raw IL | `CompileRawInnerLoop(body, key)` | caller emits the entire `NDInnerLoopFunc` body |
| Templated elementwise | `CompileInnerLoop(operandTypes, scalarBody, vectorBody, key)` | factory emits the loop shell; caller emits scalar/vector element body |
| Expression DSL | `NDExpr` compilation through `np.evaluate` | expression tree emits scalar/vector bodies into the templated shell |

The templated shell checks the runtime byte strides for the current chunk. If
every operand has a natural contiguous inner stride, it takes the SIMD path. If
any operand is strided or broadcast in a way the vector path cannot express, it
falls back to scalar-strided IL for that chunk.

## SIMD Strategy

The SIMD layer is width-adaptive:

- `DirectILKernelGenerator.VectorBits` detects 128, 256, or 512-bit support (or
  0, which forces the scalar path, when no SIMD hardware is present).
- `VectorMethodCache` resolves `Vector128`, `Vector256`, or `Vector512` methods.
- On x86, the cache routes many operations to `Sse/Sse2`, `Avx/Avx2`, or
  `Avx512F` methods when available.
- Unsupported methods are treated as a normal fallback condition, not a failure.

`VectorMethodCache` is central because reflection is expensive and easy to get
wrong. It caches already-closed `MethodInfo` values for loads, stores,
operators, comparisons, creates, narrows, widens, conversions, `As<TFrom,TTo>`,
`Zero`, and x86 intrinsic entry points. The source notes that routing through
x86-specific intrinsic methods can generate roughly 1.8-2x tighter code than
the cross-platform `Vector256.*` helpers on an AVX2 host.

The emitted loops also use classic throughput patterns:

- vector main loop plus scalar tail
- 4x unrolled vector bodies where the factory can express them
- scalar pre-broadcast into a vector once per loop
- horizontal vector reductions for final sums/min/max
- x86 gather for selected strided float/double paths
- `ConditionalSelect` for mask-driven selection
- typed widen/narrow and bit reinterpretation rather than boxing or virtual calls

## NumPy Semantics In The Fast Path

The generator is not just fast C#. It carries NumPy behavior into specialized
machine code.

### NEP50 Promotion

Kernel keys include input, accumulator, and result dtypes. Reductions and scans
honor NumPy's widened accumulator rules, such as integer `sum` and `prod`
accumulating into 64-bit results, and integer `mean` accumulating in `double`.

`DirectILKernelGenerator.Reduction.Axis.Widening.cs` handles widened axis
reductions with AVX2 sign/zero extension and float conversion. It uses
8192-element output-slab blocks, matching NumPy's buffer-size pattern: the block
stays hot while input rows stream through memory.

### Pairwise Floating Sum

`ILKernelGenerator.Reduction.Pairwise.cs` emits NumPy-style pairwise summation.
The important property is not just lower error. The emitter reproduces NumPy's
recursive split and accumulator order while mapping the eight accumulators onto
SIMD lanes. The file documents a measured AVX2 case where emitted SIMD pairwise
sum for a 1000x1000 `double` axis-1 reduction improved 0.267 ms to 0.123 ms
(2.18x) and beat NumPy 2.4.2 (0.207 ms) by 1.69x on the same scenario.

### NaN And Predicate Semantics

NaN-aware reductions, `fmax/fmin`, `maximum/minimum`, `isnan`, `isfinite`, and
`isinf` live in generated or SIMD-assisted paths rather than being scalar-only
afterthoughts. The code distinguishes NaN-propagating and NaN-ignoring behavior
where NumPy does.

### View And Broadcast Semantics

Generated kernels receive base pointers that already include `Shape.offset`,
then use element or byte strides to preserve sliced, reversed, transposed, and
broadcast views. Broadcast reads use stride zero. Broadcast writes remain
blocked at the ndarray/view layer because broadcast views are read-only.

## Technique Highlights

### Cast Kernels

The cast subsystem is one of the densest parts of the generator: 15 files and
6,429 lines. It covers contiguous casts, arbitrary strided casts, masked casts,
scalar inner-cast loops, and dtype-specific fast paths.

Notable techniques:

- `float`/`double` to signed integers through truncating conversion paths.
- `float`/`double` to `uint32` and `uint64` with NumPy-compatible modular wrap
  behavior rather than C# checked-conversion behavior.
- `Half` widening/narrowing through bit-level conversion helpers.
- complex-to-int and complex-to-bool paths that deinterleave real and imaginary
  lanes as needed.
- subword same-size copies, narrows, and widens for 1-byte and 2-byte dtypes.
- strided kernels that detect inner-unit stride at runtime and use a vector
  inner loop even when outer dimensions are sliced.
- unsupported-pair caches so failed IL attempts are not repeated.

### Axis Reductions

Axis reductions are not a single fallback loop. The direct generator has separate
paths for:

- same-dtype SIMD reductions
- widening reductions
- arg reductions
- boolean `all`/`any`
- NaN-aware reductions
- var/std two-pass algorithms
- leading-axis C-contiguous streaming
- innermost-axis contiguous row reductions
- sliced axis-0 layouts with contiguous inner slabs
- scalar general fallback

The leading-axis path is especially important. For C-contiguous arrays, reducing
`axis=0` can stream input rows sequentially while keeping the output slab hot,
instead of gathering a strided column for each output element.

### `np.where`

The NDIter `np.where` inner loop is deliberately not just an `NDExpr.Where`
reuse. It reads the condition operand as a raw bool byte and branches/selects
directly. When the chunk is inner-contiguous, it emits SIMD mask expansion and
`ConditionalSelect`; otherwise it runs scalar-strided IL. This avoids per-element
bool-to-output-dtype casts in the common path.

### Custom Inner Loops

`DirectILKernelGenerator.InnerLoop.cs` exposes a reusable inner-loop factory used
by NDIter custom operations and `np.evaluate`.

The templated path wraps user-provided scalar and vector IL bodies in:

```text
load dataptrs and byte strides
check whether all operands are inner-contiguous
run 4x-unrolled SIMD loop when legal
finish vector tail
run scalar-strided fallback for the rest
return
```

That gives custom operations access to the same generated-loop shape as built-in
ufuncs without requiring every caller to hand-write pointer arithmetic.

### Reflection Caches

Runtime IL emission needs `MethodInfo` handles for every intrinsic call it emits.
`VectorMethodCache` and `ScalarMethodCache` keep those lookups centralized and
closed over dtype and width. This avoids dozens of local reflection searches and
prevents emitter/eligibility checks from disagreeing about what the runtime BCL
actually exposes.

## Cache Families

The direct generator has cache fields for these families:

| Family | Cache examples |
| --- | --- |
| Elementwise | `_contiguousKernelCache`, `_mixedTypeCache`, `_unaryCache`, `_stridedUnaryCache`, `_comparisonCache` |
| Scalar delegates | `_unaryScalarCache`, `_binaryScalarCache`, `_comparisonScalarCache` |
| Casts | `_castCache`, `_stridedCastCache`, `_maskedCastCache`, `_innerCastCache`, plus unsupported-pair caches |
| Reductions | `_elementReductionCache`, `_nanElementReductionCache`, `_axisReductionCache`, `_nanAxisReductionCache`, `_boolAxisReductionCache` |
| Scans | `_scanCache`, `_axisScanCache` |
| Selection/indexing | `_whereKernelCache`, scalar `where` caches, `_argwhereCount`, `_argwhereFlat`, `_searchCache`, `_filterAxis` |
| Other generated helpers | `_copyKernelCache`, `_matmulKernelCache`, `_quantileKernelCache`, repeat caches, `_trace`, `_weightedSumCache`, storage alias cache |
| Custom inner loops | `_innerLoopCache`, surfaced through `GeneratedDelegates` for tests |

Keys are structural. They encode enough information to make a generated body
valid: operation, dtype tuple, accumulator/result dtype, execution path, copy
mode, quantile method, or custom user key.

`GeneratedDelegates` exposes public live counts for generated-kernel caches and
internal clear hooks for tests. Reflection caches are intentionally excluded
because they store `MethodInfo` lookup results, not generated executable kernels.

## Adding Or Changing A Kernel

Use this workflow for new work:

1. Probe NumPy first. Capture dtype, shape, stride, NaN, empty, scalar, broadcast,
   and output dtype behavior from actual NumPy 2.x.
2. Choose the driving model. Use `ILKernelGenerator` for NDIter chunk loops and
   new ufunc-style work. Use `DirectILKernelGenerator` when the operation owns a
   whole-array traversal or is still part of the direct migration surface.
3. Define the cache key before writing the emitter. If a choice changes emitted
   IL, it belongs in the key.
4. Keep the generated signature narrow. Pass raw pointers, strides, shape, axis,
   and sizes; avoid managed allocations in the hot body.
5. Separate path selection from loop emission. The existing code usually has a
   dispatcher, then path-specific `Generate*` methods, then `Emit*Loop` helpers.
6. Handle all 15 NumSharp dtypes or document the unsupported cases and provide a
   clear fallback.
7. Preserve view semantics. Offset is already in the base pointer; strides must
   handle non-contiguous, negative, and broadcast cases.
8. Add NumPy-derived tests. Include contiguous, strided, broadcast, empty/scalar,
   NaN, and dtype-promotion cases according to the operation's risk.
9. Benchmark in Release. Kernel timings are invalid in Debug because JIT and
   intrinsic behavior differ materially. Use the project benchmark convention:
   NumSharp-vs-NumPy ratios are `NumPy_ms / NumSharp_ms`, so values above `1.0`
   mean NumSharp is faster. See the [Benchmarks dashboard](benchmarks-dashboard.md).

## Debugging

Useful checks when a generated path behaves strangely:

- Confirm `DirectILKernelGenerator.Enabled` and `VectorBits`.
- Inspect the cache key. A missing field in the key can reuse the wrong IL body.
- Check which layout path was selected before entering the generator.
- For NDIter kernels, inspect byte strides for the current inner loop. Natural
  byte stride is required before the SIMD chunk path fires.
- Use `GeneratedDelegates.InnerLoopCount` and reset hooks to prove a custom
  inner loop is cached or regenerated.
- If a `TryGet*Kernel` returns `null`, inspect the debug output and the scalar
  fallback path before assuming a correctness bug is in emitted IL.
- Remember that `DynamicMethod` IL is not directly step-through friendly. When
  needed, temporarily add logging around path selection or switch a local
  workspace diff to emit into a debuggable assembly.

## What Makes This System Unusual

Most managed numerical libraries choose between generic managed loops and a
native backend. NumSharp sits in a different place: it generates managed IL at
runtime, lets the .NET JIT lower that IL to native machine code, and keeps NumPy
layout and dtype semantics in the generated body.

The interesting pieces are the combination:

- NumPy-style dtype and accumulator rules.
- Unmanaged ndarray storage and pointer-level kernels.
- Runtime specialization by dtype tuple, operation, and layout.
- Width-adaptive SIMD with V128, V256, and V512 support.
- x86 intrinsic routing when it produces tighter code than portable vector APIs.
- Whole-array kernels for mature direct paths.
- NDIter inner-loop kernels for NumPy-like iterator execution and fusion.
- Pairwise floating reductions that preserve NumPy's summation order while
  recovering SIMD throughput.
- Cast kernels that encode NumPy's odd corners, including modular unsigned
  float casts, half conversion, complex deinterleaving, subword lanes, and
  masked output.

This is why the IL generator is not just an optimization layer. It is where
NumSharp turns NumPy compatibility rules into specialized machine code.
