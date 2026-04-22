# Unified Iterator Design (v5 — current state)

> **Status:** implemented. The plan in v1-v4 (build a new `NDIterator` class with
> three tiers of kernels) was superseded by porting NumPy's `nditer` directly —
> now `NpyIterRef`. The three "tiers" morphed into seven layered integration
> points, all sharing one IL-emitted-kernel cache. This document captures the
> final shape and how we got here.
>
> **Production docs:** `docs/website-src/docs/NDIter.md` has the full user-facing
> reference (~1900 lines). This file is the design rationale and migration
> crib-sheet for contributors porting old patterns.

---

## Design principles (unchanged from v4)

1. **No backwards compatibility** — old iterators/incrementors deleted (done; see Migration below)
2. **Direct IL control** — users can inject their own IL at every layer
3. **Zero allocation** — struct-based state, unmanaged memory, no closures on hot paths
4. **Layered, not flat** — seven entry points on an ergonomics-vs-control axis

## What changed since v4

| v4 plan | v5 reality | Why |
|---------|-----------|-----|
| Build new `NDIterator` class from scratch | Port NumPy's `nditer` as `NpyIterRef` | Every ufunc, reduction, and broadcast in NumPy already goes through it; reinventing the scheduler would re-discover the same design choices (coalescing, buffered reduction, op_axes). Porting preserves 1-to-1 behavioral parity. |
| 3 tiers (interface / IL / Func) | 7 entry points (Layer 1/2/3 + Tier 3A/3B/3C + Call) | Three layers conflated "how does the kernel dispatch?" with "what kernel shape am I authoring?". Splitting gives us baked ufuncs *and* custom-op escape hatches without mode-switching. |
| `IUnaryKernel<TIn,TOut>` static abstracts | `NpyInnerLoopFunc` delegate + struct-generic `INpyInnerLoop` | Static-abstract generics don't inline reliably across assemblies on net8; struct-generic dispatch is cleaner and the `NpyInnerLoopFunc` delegate matches NumPy's C-API loop signature 1-to-1. |
| `IKernelEmitter` interface for IL injection | `Action<ILGenerator>` per-element + factory-wrapped shell | A full `IKernelEmitter` interface was overkill for the common "I just want SIMD with a custom op" case. The factory handles the unroll shell; users write only the per-element body. Raw-IL power-users use `ExecuteRawIL(Action<ILGenerator>)`. |
| `Func<T,TOut>` delegates as Tier 3 | `ForEach(NpyInnerLoopFunc)` + `NpyExpr.Call(Delegate)` | The `Func<>` path morphed into two: Layer 1 `ForEach` for whole-loop delegates, and `NpyExpr.Call` for per-element managed methods embedded inside a DSL tree. |

## The seven techniques

```
           ergonomics                                                     control
              ▲                                                              ▲
              │                                                              │
  Layer 3     │  ExecuteBinary / Unary / Reduction / Comparison / Scan      │  90% case
              │  "one call, NumPy-style — one line per op"                   │
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier 3C     │  ExecuteExpression(NpyExpr)                                  │  compose
              │  "build a tree with operators; no IL in caller"              │  with DSL
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier 3C     │  NpyExpr.Call(Math.X / Func / MethodInfo, args)              │  inject any
    + Call    │  "invoke arbitrary managed method per element"               │  BCL / user op
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier 3B     │  ExecuteElementWiseBinary(scalarBody, vectorBody)            │  hand-tune
              │  "write per-element IL; factory wraps the unroll shell"      │  the vector body
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier 3A     │  ExecuteRawIL(emit, key, aux)                                │  emit
              │  "emit the whole inner-loop body including ret"              │  everything
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Layer 2     │  ExecuteGeneric<TKernel> / ExecuteReducing<TKernel, TAccum>  │  struct-
              │  "zero-alloc; JIT specializes per struct; early-exit reduce" │  generic
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Layer 1     │  ForEach(NpyInnerLoopFunc kernel, void* aux)                 │  delegate,
              │  "closest to NumPy's C API; closures welcome"                │  anything goes
              │                                                              │
              ▼                                                              ▼
           NpyIter state (Shape, Strides, DataPtrs, Buffers, ...)
                                  │
                                  ▼
              ILKernelGenerator (DynamicMethod + V128/V256/V512)
```

All seven share:
- one `ConcurrentDictionary<string, NpyInnerLoopFunc>` inner-loop cache
- one `ForEach` driver at the bottom (`do { kernel(dataptrs, strides, count, aux); } while (iternext);`)
- the same SIMD machinery in `ILKernelGenerator` (V128 / V256 / V512 selection at startup)

---

## Layer 3 — Baked ufuncs (the 90% case)

```csharp
using var iter = NpyIterRef.MultiNew(3, new[] { a, b, c },
    NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_NO_CASTING,
    new[] { NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.WRITEONLY });
iter.ExecuteBinary(BinaryOp.Add);
```

`ExecuteBinary / Unary / Reduction / Comparison / Scan / Copy` resolve to a cached `MixedTypeKernelKey` lookup in `ILKernelGenerator`. First call JIT-compiles; every subsequent call with matching types/path returns the cached delegate.

Benchmark: 1M float32 `a + b` = **0.58 ms/run** (4×-unrolled V256, post-warmup).

---

## Tier 3C — Expression DSL (`NpyExpr`)

45+ node types compose with operators:

```csharp
var x = NpyExpr.Input(0);
var pos = NpyExpr.Const(1.0) / (NpyExpr.Const(1.0) + NpyExpr.Exp(-x));
var neg = NpyExpr.Exp(x) / (NpyExpr.Const(1.0) + NpyExpr.Exp(x));
var stable = NpyExpr.Where(
    NpyExpr.GreaterEqual(x, NpyExpr.Const(0.0)), pos, neg);

iter.ExecuteExpression(stable,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Covers arithmetic, bitwise, rounding, transcendentals (exp/log/trig/hyperbolic/inverse-trig), predicates, comparisons, Min/Max/Clamp/Where. Auto-derives a cache key from the tree's structural signature (e.g. `NpyExpr:Sqrt(Add(Square(In[0]),Square(In[1]))):in=Single,Single:out=Single`).

Benchmark: stable sigmoid on 1M f64 = **13.6 ms/run** (3 × `Math.Exp` per element dominates).

## Tier 3C + Call — Inject any .NET method

```csharp
// Typed Func overloads — method groups bind without cast
NpyExpr.Call(Math.Sqrt,   NpyExpr.Input(0));
NpyExpr.Call(Math.Pow,    NpyExpr.Input(0), NpyExpr.Input(1));

// Cast to disambiguate overloaded methods
NpyExpr.Call((Func<double, double>)Math.Abs, NpyExpr.Input(0));

// Pre-constructed delegate with captures
static readonly Func<double, double> GELU = x =>
    0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) *
                               (x + 0.044715 * x * x * x)));
NpyExpr.Call(GELU, NpyExpr.Input(0));

// MethodInfo — static
var mi = typeof(Math).GetMethod("BitIncrement", new[] { typeof(double) });
NpyExpr.Call(mi, NpyExpr.Input(0));

// MethodInfo + instance target
NpyExpr.Call(instanceMethod, targetObject, NpyExpr.Input(0));
```

Three dispatch paths, selected automatically at node construction:

| Condition | Emitted IL | Per-element cost |
|-----------|------------|------------------|
| Static method, no captures | `call <methodinfo>` | Direct call; JIT may inline |
| Instance `MethodInfo` with explicit `target` | `ldc.i4 slotId` → `DelegateSlots.LookupTarget` → `castclass T` → `callvirt <methodinfo>` | ~5 ns + virtual call |
| Any other Delegate | `ldc.i4 slotId` → `DelegateSlots.LookupDelegate` → `castclass Func<...>` → `callvirt Invoke` | ~5-10 ns + `Delegate.Invoke` |

Strong-ref `DelegateSlots` registry keeps captured delegates alive for the process lifetime — user must register once at startup (static field) to avoid unbounded growth.

Benchmark: GELU via captured lambda on 1M f64 = **8.08 ms/run**.

---

## Tier 3B — Templated element-wise, hand-written vector body

Factory emits the 4×-unrolled SIMD + 1-vec remainder + scalar-tail + scalar-strided fallback shell. User provides only the per-element scalar and (optional) vector body:

```csharp
iter.ExecuteElementWiseBinary(
    NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
    scalarBody: il =>
    {
        // Stack: [a, b] → [2a + 3b]
        il.Emit(OpCodes.Ldc_R4, 2f); il.Emit(OpCodes.Mul);
        var tmp = il.DeclareLocal(typeof(float)); il.Emit(OpCodes.Stloc, tmp);
        il.Emit(OpCodes.Ldc_R4, 3f); il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Ldloc, tmp); il.Emit(OpCodes.Add);
    },
    vectorBody: il =>
    {
        // Vector256<float> ops — all via ILKernelGenerator primitives
        il.Emit(OpCodes.Ldc_R4, 2f);
        ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);
        // … symmetric for 3b, then add …
    },
    cacheKey: "linear_2a_3b_f32");
```

**When SIMD is skipped.** Vector body is emitted only when `CanSimdAllOperands(operandTypes)` is true (all operand dtypes identical *and* SIMD-capable). Mixed-type ufuncs (int32 + float32 → float32) run the scalar body with `EmitConvertTo` inside.

**Runtime contig check.** Factory emits a stride-vs-elemSize comparison at kernel entry. Any stride mismatch falls into the scalar-strided loop — one kernel handles both contiguous and sliced inputs without recompile.

Benchmark: `2a + 3b` on 1M f32 = **0.61 ms/run** — within ~7% of baked Layer 3 Add.

---

## Tier 3A — Raw IL escape hatch

User emits the entire inner-loop body against the NumPy ufunc signature
`void(void** dataptrs, long* byteStrides, long count, void* aux)`:

```csharp
iter.ExecuteRawIL(il =>
{
    // c[i] = |a[i] - b[i]| for int32 operands, fused in one kernel.
    var p0 = il.DeclareLocal(typeof(byte*));
    var p1 = il.DeclareLocal(typeof(byte*));
    var p2 = il.DeclareLocal(typeof(byte*));
    var s0 = il.DeclareLocal(typeof(long));
    // ... 60 lines of il.Emit(OpCodes.*) ...
    il.Emit(OpCodes.Ret);
}, cacheKey: "abs_diff_i32");
```

Use when the loop shape is non-rectangular (gather/scatter, cross-element dependencies, branch-on-auxdata). Otherwise prefer Tier 3B which gets you the SIMD shell for free.

Benchmark: `abs(a - b)` on 1M i32 = **1.27 ms/run** (scalar loop, JIT autovectorizes post tier-1).

---

## Layer 2 — Struct-generic dispatch (zero-alloc)

The JIT specializes `ExecuteGeneric<TKernel>` per struct type at codegen time. No delegate indirection, no boxing. **Only path with early-exit reductions.**

```csharp
readonly unsafe struct HypotKernel : INpyInnerLoop
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Execute(void** p, long* s, long n)
    {
        if (s[0] == 4 && s[1] == 4 && s[2] == 4)
        {
            float* pa = (float*)p[0], pb = (float*)p[1], pc = (float*)p[2];
            for (long i = 0; i < n; i++)
                pc[i] = MathF.Sqrt(pa[i] * pa[i] + pb[i] * pb[i]);
        }
        // … strided fallback …
    }
}

readonly unsafe struct AnyNonZero : INpyReducingInnerLoop<bool>
{
    public bool Execute(void** p, long* s, long n, ref bool acc)
    {
        byte* pt = (byte*)p[0]; long st = s[0];
        for (long i = 0; i < n; i++)
            if (*(int*)(pt + i * st) != 0) { acc = true; return false; } // STOP
        return true;
    }
}

iter.ExecuteGeneric(default(HypotKernel));
bool found = iter.ExecuteReducing<AnyNonZero, bool>(default, false);
```

Benchmark: `AnyNonZero` early-exit over 1M int32 with hit at idx 500 = **0.001 ms/run** — the kernel returns `false`, the bridge bails out of the do/while after one call.

---

## Layer 1 — ForEach delegate (NumPy-C-API parity)

```csharp
iter.ForEach((ptrs, strides, count, aux) => {
    if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
        float* pa = (float*)ptrs[0], pb = (float*)ptrs[1], pc = (float*)ptrs[2];
        for (long i = 0; i < count; i++)
            pc[i] = MathF.Sqrt(pa[i] * pa[i] + pb[i] * pb[i]);
    } else {
        // … strided scalar fallback …
    }
});
```

Classic NumPy-C-API shape. One delegate closure per call. Most flexible for one-offs, fused kernels with captures, or mid-execution experimentation.

---

## Decision tree

```
Is the op a standard NumPy ufunc already in ExecuteBinary/Unary/Reduction?
  yes → Layer 3. Fastest, zero work. Done.
  no ↓

Can I express it as a tree of DSL nodes (Add, Sqrt, Where, Exp, …)?
  yes → Tier 3C. Fused, SIMD-or-scalar automatic, no IL.
  no ↓

Is the missing piece a BCL method (Math.X, user activation, reflected plugin)?
  yes → Tier 3C + Call. Scalar-only but fused. Done.
  no ↓

Do I need V256/V512 intrinsics the DSL doesn't wrap (Fma, Shuffle, Gather, …)?
  yes → Tier 3B. Hand-write the vector body; factory wraps the shell.
  no ↓

Is the loop shape non-rectangular (gather/scatter, cross-element deps)?
  yes → Tier 3A. Emit the whole inner-loop IL yourself.
  no ↓

Do I need an early-exit reduction (Any / All / find-first)?
  yes → Layer 2 ExecuteReducing. Returns false from the kernel to bail out.
  no ↓

Just exploring or writing a one-off?
       → Layer 1 ForEach. Delegate per call; flexible.
```

---

## Performance summary (1M elements, post-warmup)

| Technique | Operation | Time / run | Notes |
|-----------|-----------|-----------:|-------|
| Layer 3 | `a + b` (f32) | 0.58 ms | baked, 4×-unrolled V256, cache hit |
| Tier 3B | `2a + 3b` hand V256 (f32) | 0.61 ms | within ~7% of baked |
| Layer 2 reduction | `AnyNonZero` early-exit (hit @ 500) | 0.001 ms | returns `false` from kernel |
| Tier 3A | `abs(a - b)` raw IL (i32) | 1.27 ms | scalar, JIT autovectorizes |
| Tier 3C + Call | `GELU` via captured lambda (f64) | 8.08 ms | `Math.Tanh` dominates |
| Tier 3C | stable sigmoid via `Where` (f64) | 13.6 ms | 3 × `Math.Exp` per element |

Tier-0 JIT caveat applies to Layer 1/2 element-wise kernels in ephemeral hosts (dotnet_run, cold-start scripts) — they can look 30-50× slower than production until tier-1 promotion kicks in (~100 hot-loop iterations).

---

## NpyIter state (unified, post-port)

Replaces the v4 `IteratorState` struct. Heap-allocated via `NativeMemory.AllocZeroed` (not stack-allocated with `fixed int[16]`) because NumSharp drops NumPy's `NPY_MAXDIMS=64` ceiling — state is sized to the actual `(ndim, nop)`.

```csharp
public unsafe struct NpyIterState
{
    // Scalars
    public int NDim, NOp;
    public long IterSize, IterIndex;
    public NpyIterFlags ItFlags;

    // Dim arrays (size = NDim)
    public long* Shape;
    public long* Coords;
    public long* Strides;           // element strides per (op, axis)
    public sbyte* Perm;             // negative = axis was flipped

    // Op arrays (size = NOp)
    public long* DataPtrs, ResetDataPtrs, BufStrides, InnerStrides, BaseOffsets;
    public NPTypeCode* OpDTypes;

    // Reduction arrays
    public long* ReduceOuterStrides, ReduceOuterPtrs, ArrayWritebackPtrs;
    public long CoreSize, CorePos, ReduceOuterSize, ReducePos;

    // Buffer
    public long BufferSize, BufIterEnd;
    public long* Buffers;
}
```

See `src/NumSharp.Core/Backends/Iterators/NpyIter.State.cs` for the full definition and `NDIter.md` for the field-by-field walkthrough.

---

## Migration: old patterns → NpyIter

### Pattern 1: element-wise loop via `NDIterator`

**Old:**
```csharp
var iter = new NDIterator<double>(source, false);
while (iter.HasNext())
{
    var val = iter.MoveNext();
    sum += val * val;
}
```

**New (Layer 2 struct-generic):**
```csharp
readonly unsafe struct SumOfSquares : INpyReducingInnerLoop<double>
{
    public bool Execute(void** p, long* s, long n, ref double acc)
    {
        byte* pt = (byte*)p[0]; long st = s[0];
        for (long i = 0; i < n; i++)
        {
            double v = *(double*)(pt + i * st);
            acc += v * v;
        }
        return true;
    }
}

using var iter = NpyIterRef.MultiNew(1, new[] { source },
    NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_NO_CASTING, new[] { NpyIterPerOpFlags.READONLY });
double sum = iter.ExecuteReducing<SumOfSquares, double>(default, 0.0);
```

**New (Tier 3C DSL):**
```csharp
// If you also want the *array* of x² (not just the reduction):
var expr = NpyExpr.Square(NpyExpr.Input(0));
iter.ExecuteExpression(expr, new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

### Pattern 2: axis-wise iteration via `NDCoordinatesAxisIncrementor`

**Old:**
```csharp
var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
var slices = iterAxis.Slices;
do
{
    var slice = arr[slices];
    ret[slices] = ProcessSlice(slice);
} while (iterAxis.Next() != null);
```

**New (axis-reducing iterator with op_axes):**
```csharp
// Use the axis-reduction construction path; NpyIter handles the double-loop
// buffered reduction internally via REDUCE_OK + ExecuteReduction.
using var iter = NpyIterRef.AdvancedNew(2, new[] { input, output },
    NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.REDUCE_OK
        | NpyIterGlobalFlags.BUFFERED,
    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
    new[] { NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.ALLOCATE },
    opAxes: new[][] { null, outputAxes });
iter.ExecuteReduction<double>(ReductionOp.Sum);
```

### Pattern 3: broadcast paired iteration via `MultiIterator`

**Old:**
```csharp
var (lIter, rIter) = MultiIterator.GetIterators(lhs, rhs, broadcast: true);
while (lIter.HasNext())
    lIter.MoveNextReference() = rIter.MoveNext();
```

**New (Layer 3 Copy):**
```csharp
using var iter = NpyIterRef.MultiNew(2, new[] { rhs, lhs },
    NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_SAFE_CASTING,
    new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
iter.ExecuteCopy();
```

### Pattern 4: coordinate access via `ValueCoordinatesIncrementor`

**Old:**
```csharp
var incr = new ValueCoordinatesIncrementor(ref shape);
int[] coords = incr.Index;
do
{
    var offset = shape.GetOffset(coords);
    Process(data + offset);
} while (incr.Next() != null);
```

**New (Layer 1):**
```csharp
using var iter = NpyIterRef.MultiNew(1, new[] { arr },
    NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_NO_CASTING, new[] { NpyIterPerOpFlags.READONLY });
iter.ForEach((ptrs, strides, count, aux) => {
    byte* p = (byte*)ptrs[0]; long s = strides[0];
    for (long i = 0; i < count; i++)
        Process((double*)(p + i * s));
});
```

---

## Files — current state

**Core (production):**

```
src/NumSharp.Core/Backends/Iterators/
├── NpyIter.cs                    construction wrappers, MultiNew/AdvancedNew
├── NpyIter.State.cs              NpyIterState struct, Advance/Reset/GotoIterIndex
├── NpyIter.Execution.cs          Layer 1/2/3 — ForEach, ExecuteGeneric, Execute*
├── NpyIter.Execution.Custom.cs   Tier 3A/3B/3C — ExecuteRawIL, ExecuteElementWise, ExecuteExpression
├── NpyExpr.cs                    Tier 3C DSL — 45+ nodes + Call factory + DelegateSlots
├── NpyIterFlags.cs               flag enums (Global / PerOp / internal)
├── NpyIterCoalescing.cs          CoalesceAxes, ReorderAxesForCoalescing, FlipNegativeStrides
├── NpyIterCasting.cs             safe/same-kind/unsafe cast rules
├── NpyIterBufferManager.cs       aligned buffer alloc, copy-in/copy-out
├── NpyIterKernels.cs             INpyInnerLoop, INpyReducingInnerLoop interfaces
├── NpyAxisIter.cs, NpyAxisIter.State.cs   axis-reduction iterator
└── NpyLogicalReductionKernels.cs generic boolean/numeric axis reduction structs

src/NumSharp.Core/Backends/Kernels/
└── ILKernelGenerator.InnerLoop.cs  CompileRawInnerLoop, CompileInnerLoop, factory shell
```

**Deleted (v4 → v5 migration, completed):**

```
src/NumSharp.Core/Backends/Iterators/
├── INDIterator.cs                    [deleted]
├── IteratorType.cs                   [deleted]
├── MultiIterator.cs                  [deleted]
├── NDIterator.cs                     [deleted]
├── NDIterator.template.cs            [deleted]
├── NDIteratorExtensions.cs           [deleted]
└── NDIteratorCasts/NDIterator.Cast.*.cs (×12)  [deleted]

src/NumSharp.Core/Utilities/Incrementors/
├── NDCoordinatesAxisIncrementor.cs   [deleted]
├── NDCoordinatesIncrementor.cs       [deleted]
├── ValueCoordinatesIncrementor.cs    [deleted]
└── ValueOffsetIncrementor.cs         [deleted]
```

---

## Scope limitations (unchanged)

1. **Multi-output operations** (e.g., `modf` returning two arrays) — use `ILKernelGenerator.Modf` directly, not via the seven-tier bridge
2. **Type promotion** — caller's responsibility via `np._FindCommonType` / NPTypeCode utilities
3. **Memory allocation** — caller provides output NDArray (or uses `NpyIterPerOpFlags.ALLOCATE`)

Broadcasting is **not** a scope limitation anymore — NpyIter handles it inherently via stride=0 dimensions.

---

## Known bugs (post-port)

The bridge works around two bugs in the ported `NpyIter` that should be fixed in-place eventually:

- **Bug A:** `NpyIterRef.Iternext()` unconditionally calls `state.Advance()`, ignoring `EXLOOP`. Bridge sidesteps by calling `GetIterNext()` directly.
- **Bug B:** Buffered + Cast path computes wrong byte deltas because `state.Strides[op]` holds element strides but `ElementSizes[op]` is buffer-dtype size. Bridge routes buffered paths through `RunBuffered*` methods using `BufStrides` instead.

Eight additional bugs surfaced during development (C through H, covering Where, Decimal size, predicate I4 leak, LogicalNot type mismatch, Vector256.Round availability, MinMax NaN propagation) were **fixed**. See `NDIter.md § Known Bugs and Workarounds` for full writeups.

---

## References

- **Production reference docs:** `docs/website-src/docs/NDIter.md` — complete user-facing documentation (~1900 lines)
- **NumPy port source:** NumPy's `numpy/_core/src/multiarray/nditer_*.c`
- **Test coverage:** 264 tests across
  `NpyIterCustomOpTests.cs` (14 basic),
  `NpyIterCustomOpEdgeCaseTests.cs` (76 edge cases),
  `NpyExprExtensiveTests.cs` (136 DSL ops),
  `NpyExprCallTests.cs` (38 Call variants) —
  all passing on net8.0 and net10.0.
