# Handover — Remaining NumSharp Performance / Pathing Gaps

Status date: 2026-06-23 · Branch: `nditer`

This handover covers the performance gaps found by probing the dispatch paths against
NumPy 2.4.2. Two **pure order-of-pathing** bugs were already fixed (see below); what
remains needs new kernels or a semantics decision — **none is a free reorder**. Each
gap has a verbatim reproduction, the confirmed root cause (file:line), what is still
unknown, fix options with trade-offs, and a definition-of-done.

---

## 0. Already fixed (context — do not redo)

| ID | Bug | Commit | One-line cause |
|----|-----|--------|----------------|
| F1 | `np.evaluate` 12–16× slower on F-contiguous / transposed operands | `16afbac9` | `EvaluateCore` forced `NPY_ORDER.NPY_CORDER` + C-contig output; now F-aware (`AreAllInputsStrictFContig`) |
| F2 | `a * 2` (python-literal) ~2.25× slower than `a * 2f` | `d1932b00` | `ExecuteBinaryOp` same-dtype gate (`BinaryOp.cs:452`) shadowed the scalar-broadcast SIMD path; now promotes the weak 0-d scalar to the loop dtype first |

Both re-verified on current HEAD: parity restored, **3,437 tests green** (net8.0 + net10.0)
across binary/bitwise/comparison/logic/evaluate/math/reduction/statistics/ufunc.

These two were the *only* pure order-of-pathing bugs. The gaps below are different in
kind (missing SIMD kernel, layout-specific kernel behaviour, promotion semantics).

---

## 1. How to reproduce anything here (shared harness)

**Always build Release first** — Debug taints hand-written kernels ~2× and invalidates timing.

```bash
cd /k/source/NumSharp
dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
```

**Micro-probe template** — write to a file, run `dotnet run -c Release -p:WarningLevel=0 - < file.cs`:

```csharp
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System.Diagnostics; using NumSharp;            // + using NumSharp.Backends.Iteration; for NDExpr
var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core"); return; }
double Best(Action f,int it,int wm,int rd){for(int i=0;i<wm;i++)f();double b=1e9;for(int r=0;r<rd;r++){var sw=Stopwatch.StartNew();for(int i=0;i<it;i++)f();b=Math.Min(b,sw.Elapsed.TotalMilliseconds/it);}return b;}
// AssemblyName=NumSharp.DotNetRunScript grants access to internal members (Shape.strides, _flags, etc.)
```

**Benchmark subsystems** (committed result sheets land in `benchmark/*/*_results.md`):

```bash
python benchmark/run_benchmark.py                       # full op-matrix + nditer + layout + operand + cast + fusion (~3h)
python benchmark/layout/layout_sheet.py  --skip-build   # reductions/copy/elementwise × layout × dtype
python benchmark/cast/cast_sheet.py      --skip-build   # astype src→dst × layout × dtype
python benchmark/fusion/fusion_sheet.py  --skip-build   # np.evaluate
```

**Methodology that found every gap:** pit two variants that *should* take the same fast
path (scalar-vs-literal, F-copy-vs-transpose, int-vs-float mean). A gap = a misrouted or
missing fast path. Keep using it.

**No-harm bar (definition of done for any fix):** the parity probe must close, values must
be bit-identical to the prior path, and these must stay green on **both** frameworks:

```bash
cd test/NumSharp.UnitTest && dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&(ClassName~BinaryOp|ClassName~Arithmetic|ClassName~Promotion|ClassName~Scalar|ClassName~Broadcast|ClassName~Bitwise|ClassName~Comparison|ClassName~Logic|ClassName~Reduction|ClassName~Statistic|ClassName~Math|ClassName~Ufunc|ClassName~Evaluate|ClassName~NDExpr|ClassName~Cast)"
# plus the FuzzMatrix differential gate for anything touching kernels:
dotnet test --no-build --filter "TestCategory=FuzzMatrix"
```

---

## 2. Gap catalog (priority order)

### G1 — F-contiguous axis reductions take the slow kernel  **[HIGH · diagnostic-first]**

**Symptom**
```
sum(F-copy) axis0 1.574 ms   vs   sum(T-view) axis0 0.124 ms   → 12× slower
                                  (both (1000,1000) f64, both column-major (8,8000))
benchmark/layout reductions geomean: strided 0.48 · bcast 0.51 · negcol 0.57 · F-heavy rows weak
```

**Reproduce**
```csharp
var C=((np.arange(1_000_000)%17)+1).astype(np.float64).reshape(1000,1000);
var F=C.copy(order:'F'); var T=C.T;
Console.WriteLine($"F {Best(()=>{{var _=np.sum(F,0);}},40,8,3):F4}  T {Best(()=>{{var _=np.sum(T,0);}},40,8,3):F4}  C {Best(()=>{{var _=np.sum(C,0);}},40,8,3):F4}");
```

**Confirmed root cause** — `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Add.cs:82`
```csharp
var key = new AxisReductionKernelKey(inputType, outputType, op, shape.IsContiguous && axis == arr.ndim - 1);
```
The fast Direct axis-reduce variant is gated on **C-contiguity** (`shape.IsContiguous`) AND
reducing the last axis. An F-contiguous array has `IsContiguous == false`, so it can never
select the fast kernel even when reducing its memory-contiguous axis (axis 0 for column-major).
`UseNDIterReduce` only diverts Complex dtypes, so f64/f32/int all land on this Direct kernel.

**Still UNKNOWN — run this first** (it was blocked by a broken build last session). Both F and T
have `flag=false` → the *same* general kernel + (on paper) identical `(8,8000)` strides, yet
T is 12× faster. Find the property that actually diverges:
```csharp
void Dump(string n, NDArray a){var s=a.Shape;
  Console.WriteLine($"{n} strides=[{string.Join(",",s.strides)}] C={s.IsContiguous} F={s.IsFContiguous} sliced={s.IsSliced} off={s.offset} buf={s.bufferSize} flags=0x{s._flags:X}");}
Dump("F",C.copy(order:'F')); Dump("T",C.T);
```
Hypotheses to confirm/kill: (a) `copy(order:'F')` produces different strides/flags than `.T`;
(b) the general axis kernel's loop nesting makes F's buffer access cache-hostile while T (a view
over a C-contig buffer) stays sequential; (c) an upstream rewrite turns `sum(C.T,0)` into
`sum(C,1)` on the C-contig base but doesn't fire for a genuine F array.

**Fix options**
1. If the general kernel is the problem (hypothesis b): give the axis reducer a column-major
   fast kernel and broaden the gate to `(IsContiguous && axis==ndim-1) || (IsFContiguous && axis==0)`.
   *Only do this once the diagnostic confirms the existing kernel doesn't already work for F.*
2. If it's an upstream rewrite gap (hypothesis c): apply the same transpose→base rewrite to
   F-contiguous inputs.

**DoD:** `sum(F)` within ~1.2× of `sum(T)`; `FuzzMatrix` + reduction/statistics suites green;
re-run `benchmark/layout/layout_sheet.py` and confirm the F/strided/negcol reduction columns rise.

---

### G2 — integer `mean` / `var` / `std` axis 7–16× slower than integer `sum`  **[HIGH]**

**Symptom**
```
[i64] sum axis0 1.00 ms   mean axis0 7.65 ms        [f64] sum 1.09 ≈ mean 1.10 (float is fine)
[u64] sum axis0 0.97 ms   mean axis0 8.99 ms
op-matrix: np.mean axis=0 (uint64) @10M  NumPy 7.339 ms  NS 117.565 ms  → 0.062× (16× slower)
```

**Reproduce**
```csharp
var a=((np.arange(4_000_000)%17)+1).astype(np.int64).reshape(2000,2000);
Console.WriteLine($"sum {Best(()=>{{var _=np.sum(a,0);}},20,4,3):F3}  mean {Best(()=>{{var _=np.mean(a,0);}},20,4,3):F3}");
```

**Confirmed root cause** — `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Mean.cs:78`
```csharp
var outputType2 = needsCast ? NPTypeCode.Double : (typeCode ?? inputTc.GetComputingType());
```
For float input, `GetComputingType()` keeps `InputType == AccumulatorType` → fast same-type SIMD
axis kernel. For **integer** input it returns `Double` (NumPy-faithful: int mean accumulates in
float64 to avoid overflow), which forces the **widening** axis kernel (i64→f64) — and that kernel
is not SIMD-vectorised. So it is a *correct path with a missing SIMD kernel*, not a reorder.

**Fix options**
1. SIMD-ify the widening axis reduction (`DirectILKernelGenerator.Reduction.Axis.Widening.cs`):
   widen i64→f64 in vector lanes and accumulate in f64 vectors. Cleanest; also lifts `var`/`std`.
2. For integer mean, reuse the fast same-type i64 sum axis kernel into an i64 accumulator, then
   divide+cast at the end — **but** verify overflow behaviour matches NumPy (NumPy accumulates in
   f64 precisely to avoid int overflow; an i64 accumulator can wrap on large/long axes). Option 1
   is safer.

**DoD:** int `mean`/`var`/`std` axis within ~1.5× of int `sum` axis; values bit-identical to the
current widening path (this is where overflow edge-cases bite — fuzz it); `FuzzMatrix` +
statistics suites green. Note `Default.Reduction.{Var,Std}.cs` share this widening path.

---

### G3 — mixed-dtype scalar comparison loses SIMD (`f32_array < int_literal`)  **[MED · needs a decision]**

**Symptom**
```
less(f32arr, int32-scalar) 0.768 ms   vs   less(f32arr, f32-scalar) 0.401 ms   → 1.9× (values identical)
```
Only the **f32-array** case is affected. `f64`/`i64` arrays vs an int scalar are already at parity
(the array already is the comparison common type, so only the one-element scalar is converted):
```
less(f64,i32) 0.400 ≈ less(f64,f64) 0.396     less(i64,i32) 0.233 ≈ less(i64,i64) 0.234
```

**Reproduce**
```csharp
var a=((np.arange(1_000_000)%17)+1).astype(np.float32);
var i32=NDArray.Scalar(2,NPTypeCode.Int32); var f32=NDArray.Scalar(2f,NPTypeCode.Single);
Console.WriteLine($"i32 {Best(()=>{{var _=np.less(a,i32);}},200,40,5):F5}  f32 {Best(()=>{{var _=np.less(a,f32);}},200,40,5):F5}");
```

**Confirmed root cause** — `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.CompareOp.cs`.
`TryTrivialContiguousComparisonOp` *does* take the `SimdScalarRight` arm for the mixed scalar
(no same-dtype gate, unlike binary) — but the per-element body (lines 436-451) emits the SIMD
vector path **only when `lhsType==rhsType==cmpType`**; mixed dtypes drop to a scalar convert loop.
The comparison common type is strong: `np._FindCommonScalarType(Single, Int32) = Double`, so
`f32 < int` resolves to **f64** and the f32 **array** must widen f32→f64 *per element* — there is
no cheap same-type path to route to.

**Why this is NOT pure ordering:** comparing at result_type (f64 here) is NumSharp's documented,
test-pinned behaviour (`greater(i8 2^53+1, f8 2^53) → False`, `NDEvaluateTests`/comparison tests).
Treating the literal as a *weak* scalar (→ f32) would match NumPy and unlock the fast path but
**changes comparison semantics** and risks that pinned test.

**Fix options (pick one — this is the decision)**
1. **Weak-scalar comparison promotion** (semantics change): for a genuine 0-d/size-1 scalar operand,
   resolve the comparison type with the weak-aware `_FindCommonType` (as binary ops do) instead of
   `_FindCommonScalarType`. Pros: matches NumPy weak semantics, unlocks SIMD for free. Cons: must
   re-validate every "compare at result_type" probe in `MisalignedRegistry`/comparison tests; the
   weak-vs-strong 0-d ambiguity (`np.int32(2)` vs C# `int`) is unresolvable in NumSharp.
2. **Mixed-dtype SIMD comparison kernel** (new kernel): emit a vector body that widens the array to
   `cmpType` in lanes (e.g. f32→f64 `Vector256.Widen`) then compares. No semantics change. More work.

**DoD:** `less(f32,int)` within ~1.3× of `less(f32,f32)`; values bit-identical; comparison/logic +
`FuzzMatrix` green; if option 1, explicitly re-confirm the `greater(i8 2^53+1, f8 2^53)` probe.

---

### G4 — non-C-layout reductions broadly 0.48–0.72×  **[MED · likely subsumed by G1]**

`benchmark/layout` reductions: `strided 0.48 · bcast 0.51 · negcol 0.57 · negrow 0.62 · sliced 0.68`;
by op `min 0.59 · max 0.57`. Same gate as G1 — only C-contig/last-axis has a fast kernel; everything
else uses the general strided kernel. Fix G1 first, then re-measure; the remainder is general-kernel
strided-walk efficiency. Reproduce: `python benchmark/layout/layout_sheet.py --skip-build`.

### G5 — same-type / small-N cast overhead  **[LOW]**

`flatten@100K 0.17× · astype@scalar 0.31×`; same-type 1-byte `astype(copy)` e.g. `u8|C|u8 0.20× ·
bool|T|bool 0.16×`. Correct path (`Default.Cast.cs:71` same-type branch → `Clone()`/`CastCrossType`),
but `Clone()`/alloc overhead dominates a 1 MB byte copy. Fix: route same-type contiguous copies through
a raw `Buffer.MemoryCopy`/`cpblk` with minimal allocation. Reproduce: `python benchmark/cast/cast_sheet.py --skip-build`.

### G6 — scalar-path math & index-math  **[LOW]**

`np.exp2(f32)@100K 0.10× · isnan(f32)@100K 0.087× · unravel@scalar 0.37× · ravel_multi_index@1 0.35× ·
right_shift(int64)@1K 0.041×` (array shift = scalar loop). Each is a scalar/overhead path needing SIMD
or per-call overhead reduction. Surfaced by the op-matrix worst tables (`benchmark/run_benchmark.py`).

### G7 — float16 elementwise ~0.59×  **[WONTFIX]**

No `Vector<Half>` in the .NET BCL → scalar path. Inherent/documented. Leave as-is unless the BCL adds
half-precision vector arithmetic.

---

## 3. Benchmark-integrity bugs (fix the harness, not the engine)

| ID | What | Where | Fix |
|----|------|-------|-----|
| B1 | `bcast_reduce` canary swung `0.02×` → `516.85×` between runs and contradicts `layout`'s `i32\|bcast\|sum 0.03×` | `benchmark/nditer/` (PATHOLOGY canary) | Pin the canary's shape/op, assert a correctness check before timing; a 516× "win" is a degenerate/early-return artifact |
| B2 | `dec` reductions show `0.01–0.08×` but that's Decimal-vs-float64 | `benchmark/layout/reduce_layout_bench.py:17` → `("dec", np.float64)  # dec modelled as f64` | Drop `dec` from the twin (like the `cast` subsystem's `—`) or mark non-comparable so it's excluded from ratios |
| B3 | op-matrix `np.sum f64 @100K` reads `0.089×` but a direct probe is `0.0043 ms` (4× *faster* than NumPy) | `benchmark/NumSharp.Benchmark.CSharp` Reduction suite | Per-iteration overhead inflates small-N cells; audit the BDN reduction benchmark setup, cross-check small-N against direct probes |

Verify B3 quickly: `var a=((np.arange(100_000)%17)+1).astype(np.float64); Best(()=>{var _=np.sum(a);},400,80,6)` → ~0.0043 ms.

---

## 4. Priorities & sequencing

1. **G1 diagnostic** (run the `Dump` probe) — cheap, unblocks both G1 and G4, explains the most
   surprising number (12× between identical layouts).
2. **G2** — biggest absolute regression (117 ms vs 7 ms at 10M) and shared by `var`/`std`.
3. **G3** — most *frequent* user pattern (`array < threshold`); needs the semantics decision first.
4. B1/B2/B3 — fix before trusting those benchmark cells in any report.
5. G4 (re-measure after G1), G5, G6 as capacity allows. G7 won't-fix.

All of G1–G3 are non-trivial (new kernel or a semantics call) — none is the free reorder that F1/F2 were.
Use the no-harm bar in §1 as the gate, and add a `FuzzMatrix` corpus case for any kernel you touch.
