# NDIter 2-D block kernel for narrow strided rows

**Commit:** `af25a746` (branch `journey3`, 2026-09-03)
**Lever:** `docs/NDITER_PERF_DISCOVERY.md` §7 angle 2 — *"A 2-D kernel contract for narrow rows"* — now LANDED.
**Headline:** `np.positive`/`sqrt`/`add`/… on a `(rows, w)` strided view went from **0.82× NumPy** (w=4)
to **1.4–2.5×** across 2-D *and* N-D trailing-narrow views, with a **1.59×** bonus on contiguous
`positive`. Bit-identical to the previous path; FuzzMatrix 98/98, main suite 14446/0.

Ratios follow the house convention **NPY/NS = NumPy_ms ÷ NumSharp_ms, higher = NumSharp faster**. All
numbers measured on the same host (Windows 11, i9-13900K class, AVX2, .NET 10, NumPy 2.4.2), fresh and
idle, `dotnet run -c Release` with `DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1`.

---

## 1. The problem

A view that is **contiguous on the inner axis and strided on the outer axis** — the classic `m[:, :w]`
column slice, where each row is `w` contiguous elements separated by a gap — cannot coalesce to 1-D. The
production ufunc route (`DefaultEngine.UfuncOut.cs` → `NDIterRef.ExecuteElementWiseUnary`/`Binary` →
`ForEach`) drives it under `EXTERNAL_LOOP`, calling the per-chunk kernel **once per row**. For a narrow
`w`, that call pays two size-invariant taxes on **every** row, then does a handful of elements of real
work:

1. **The driver (~4.5 ns/row)** — a delegate dispatch of the kernel plus `ExternalLoopNext`
   (`NDIter.cs`), the odometer that bumps every operand pointer and carries coordinates.
2. **The kernel's own prologue (~a few ns/row)** — the emitted per-chunk body
   (`DirectILKernelGenerator.InnerLoop.cs`, `CompileInnerLoop`) snapshots pointers/strides into locals
   and then runs a **runtime SIMD-viability dispatch** (a `stride == elemSize` compare per operand) to
   choose SIMD-contig / broadcast / gather / scalar — the answer is identical for every row, but it is
   recomputed per row.

The cost model (`docs/NDITER_PERF_DISCOVERY.md` §2) is `time ≈ FIXED + rows × PER_CHUNK + elements ×
PER_ELEMENT`. For narrow rows `PER_CHUNK` dominates, and both the odometer and the prologue live inside it.

---

## 2. Investigation

### 2.1 Baseline (this host, fresh)

| 2M f64 `(rows, w)`, `out=` | NumSharp | NumPy | NPY/NS |
|---|---:|---:|---:|
| `positive` w=4 | 3.637 ms | 2.994 | **0.82×** |
| `positive` w=16 | 2.468 | 2.230 | 0.90× |
| `sqrt` w=4 | 3.660 | 3.348 | 0.91× |

`np.sqrt` was already near parity at w≥16 (its per-element compute amortizes the overhead); the pure
memory-bound `positive` at small `w` was the clear loss.

### 2.2 The floor — the decisive measurement

A hand-written 2-D loop (one prologue, an inner SIMD run over `w`, a per-row pointer bump) establishes the
ceiling a perfect kernel could reach:

| 2M f64 `(rows, w)` | copy floor | production (before) | NumPy | floor vs NumPy |
|---|---:|---:|---:|---:|
| w=4 | **1.705 ms** | 3.629 | 2.994 | **1.76×** |
| w=16 | 1.228 | 2.456 | 2.230 | 1.82× |
| w=64 | 1.120 | 1.792 | 2.005 | 1.79× |

The floor is **1.7–2.1× faster than NumPy**, and production was **~2× off its own floor**. Conclusion: the
gap is *per-row overhead*, not memory bandwidth — the win is real and large, not a parity chase.

### 2.3 Why buffering doesn't help (measured, rejected)

Gathering the strided rows into a contiguous window (one kernel call per 8192 elements, then scatter back)
was measured **slower** than the per-row route (w=4: 4.45 ms vs 3.95). The gather/scatter costs as much as
it saves, because the inner runs are already SIMD-able *in place*. This matches the discovery doc's
rejection.

### 2.4 Routing trace

`np.positive(sv, out)` / `np.sqrt(sv, out)` and their no-`out` forms all funnel through the three
packed-key entry points in `NDIter.Execution.Custom.cs` (`ExecuteElementWise`,
`ExecuteElementWiseUnary`, `ExecuteElementWiseBinary`), which build the kernel and call `ForEach`.
Comparisons take a different route (`TryExecuteComparison`, a whole-array Direct kernel that already walks
strides), so they were never part of this problem.

---

## 3. Angles reviewed

| Angle | Verdict |
|---|---|
| **2-D block kernel** — loop the outer axis inside the kernel (prologue once) | ✅ implemented — the lever |
| **Buffered windows** | ❌ rejected (measured slower) |
| **`positive` never engaged** the SIMD path | ✅ root-caused: it had *no vector body* (scalar even contiguously); added an identity vector body |
| **N-D (`x[:, :, :w]`)** | ✅ NumSharp does not coalesce outer axes (stays NDim=3); added an outer-axis *flatten* so N-D reuses the same kernel |
| **Deterministic 3-D crash** during testing | 🔎 confirmed **pre-existing** (clean HEAD crashes identically); root cause is `np.empty(view.Shape)` fabricating strides |
| **inner-broadcast (`add(A, col)`) / non-flattenable outer (`x[::2, :, :w]`)** | left on the per-chunk route — correct, unaccelerated |

---

## 4. The fix

### 4.1 The 2-D kernel contract

`ND2DElementwiseKernel` (`src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop2D.cs`):

```csharp
void(void** dataptrs, long innerCount, long* outerByteStrides, long outerCount)
```

The emitted body runs the SIMD-viability dispatch and the loop-bound setup **once**, then loops
`outerCount` rows: an inner 4×-unrolled SIMD + 1-vector-remainder + scalar-tail loop over `innerCount`
contiguous elements, followed by a per-operand pointer bump `ptr[op] += outerByteStrides[op]`. Neither the
odometer nor the prologue runs per row.

The enabling observation: the existing `EmitSimdContigLoop` addresses each operand as `ptr + i*elemSize`
and **never mutates the base pointer**, so the same inner body, wrapped in an outer pointer-bump loop, *is*
the 2-D kernel. `EmitSimd2DContigLoop` reuses the low-level emit helpers (`EmitAddrIPlusOffset`,
`EmitVectorLoad`/`Store`, `EmitScalarElement`) and hoists the `unrollEnd`/`vectorEnd` computation above the
outer loop (they depend only on `innerCount`).

Kernels are cached in a dedicated `_innerLoop2DCache` keyed by the same `InnerLoopKernelKey` the per-chunk
route uses (a `(op, dtypes)` identity can appear in both caches — same ufunc served either per-chunk or as
one 2-D block).

### 4.2 Dispatch + gate

`NDIterRef.Is2DElementwiseShape()` + `TryExecute2DElementwise()` (`NDIter.Execution.Custom.cs`), wired into
the three packed-key `ExecuteElementWise*` entry points **before** `ForEach`, so it serves both the `out=`
and the allocating unary/binary routes. The gate (all cheap, allocation-free — the common contiguous/1-D
case fails the `NDim >= 2` / inner-contiguous test immediately, so the operand-type array is only built for
genuinely strided 2-D calls):

- unbuffered (a buffered cast/promote keeps the windowed path)
- no `where=` mask (`MaskOp < 0` → the masked ForEach driver)
- `EXTERNAL_LOOP`, not `ONEITERATION`
- inner (last) axis element-contiguous for **every** operand
- all operands the same **SIMD-capable** dtype (`CanSimdAllOperands`)

Because it reuses the **same** `scalarBody`/`vectorBody` emit delegates as the per-chunk kernel, and
element-wise ops carry no cross-element state, the result is **byte-identical** to the ForEach path — only
the loop structure differs. A broadcast-row operand (outer stride 0) is handled naturally (it re-reads the
same row).

### 4.3 N-D flatten

NumSharp's NDIter does **not** coalesce outer axes — introspection confirmed a 3-D `x[:, :, :w]` stays
`NDim=3` with strides `[8192, 16, 1]`. So the gate additionally flattens **mutually contiguous** outer axes
(`stride[d] == stride[d+1] * shape[d+1]`, true for any trailing-narrow N-D slice) into one
`(outerCount = Π outer dims, outerStride = innermost-outer stride)` and reuses the same 2-D kernel. This
covers 3-D/4-D trailing-narrow views **and** routes them off a pre-existing crash in the ForEach 3-D
odometer path (§6). Non-flattenable outer axes (a doubly-strided `x[::2, :, :w]`) stay on the per-chunk
route.

### 4.4 The `positive` fix (rode along)

`positive` is NumPy's identity ufunc. Its scalar body already emitted nothing (identity), but
`EmitUnaryVectorOperation` had no `Positive` case and `CanUseUnarySimd(Positive)` returned false — so
`positive` had a **null vector body** and was scalar *even when contiguous*, and could not engage the 2-D
path. Added an identity vector body (`case UnaryOp.Positive: return;` — load→store = a SIMD copy) and
`CanUseUnarySimd(Positive) = CanUseSimd(input)` (all SIMD dtypes, i64/u64 included — a copy needs no
abs/negate emulation). Bonus: contiguous `positive` is now a SIMD copy (**1.59×** NumPy, was scalar).

### 4.5 Files changed

| File | Change |
|---|---|
| `Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop2D.cs` | **new** — `ND2DElementwiseKernel`, cache, `Compile2DElementwiseKernel`, `EmitSimd2DContigLoop` |
| `Backends/Iterators/NDIter.Execution.Custom.cs` | `Is2DElementwiseShape` + `TryExecute2DElementwise`, wired into 3 packed-key entry points |
| `Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop.cs` | `CanSimdAllOperands` → `internal` |
| `Backends/Kernels/Direct/DirectILKernelGenerator.Unary.Vector.cs` | `positive` identity vector body |
| `Backends/Kernels/Direct/DirectILKernelGenerator.Unary.cs` | `CanUseUnarySimd(Positive)` |
| `Backends/Kernels/GeneratedDelegates.cs` | `_innerLoop2DCache` in `ClearInnerLoop` + `InnerLoop2DCount`/`TotalCount` |
| `docs/NDITER_PERF_DISCOVERY.md`, `.claude/CLAUDE.md` | mark landed |

---

## 5. Results (NPY/NS, higher = NumSharp faster)

### 5.1 Narrow strided rows, 2M f64, `out=`

| op | w=4 | w=8 | w=16 | w=32 | w=64 | w=256 |
|---|---|---|---|---|---|---|
| positive | 2.06× | 1.81× | 1.72× | 1.29× | 1.75× | 1.46× |
| negative | 2.01× | 1.79× | 1.73× | 1.40× | 0.98× | 1.42× |
| sqrt | 2.19× | 2.01× | 2.02× | 1.39× | 1.72× | 1.72× |
| abs | 2.01× | 1.41× | 1.35× | 1.60× | 1.17× | 1.75× |
| add | 2.11× | 1.71× | 1.64× | 1.39× | 1.30× | 1.50× |
| multiply | 2.14× | 1.75× | 1.62× | 1.52× | 1.15× | 1.54× |

### 5.2 2-D and N-D, 2M f64, no-`out` (includes fresh allocation)

| | w=4 | w=16 | w=64 |
|---|---|---|---|
| **2-D** positive | 1.95× | 2.13× | 2.18× |
| **2-D** sqrt | 2.41× | 2.52× | 2.11× |
| **2-D** add | 2.29× | 2.02× | 1.82× |
| **3-D** positive | 1.66× | 1.85× | 2.07× |
| **3-D** sqrt | 2.26× | 2.22× | 1.51× |
| **3-D** add | 2.17× | 2.10× | 1.37× |

Every measured cell is a win.

### 5.3 Contiguous regression check, 2M f64

| op | NumSharp | NumPy | NPY/NS |
|---|---:|---:|---:|
| positive | 0.649 ms | 1.034 | **1.59×** (was scalar) |
| negative | 0.653 | 0.666 | 1.02× (unchanged) |
| sqrt | 1.150 | 1.161 | 1.01× (unchanged) |
| add | 1.289 | 1.251 | 0.97× (unchanged) |

Only `positive` moves (the SIMD-copy bonus); everything else is unchanged — no regression.

---

## 6. The pre-existing crash (found, not caused)

While testing the 3-D path, `np.sqrt` on a `(512,512,8)` trailing-narrow view crashed with an
`AccessViolationException` in the **existing** per-chunk kernel. Stashing all changes and rebuilding clean
HEAD reproduced the crash **identically** — it is pre-existing.

Root cause: the test allocated its output with `np.empty(view.Shape)`, and `np.empty(Shape)` **inherits the
view's strides** — it produced a size-`N` buffer with strides that address beyond `N` (e.g. a 2,097,152-
element buffer with strides `[8192, 16, 1]` addressing up to element 4,194,296). Any kernel that honors
those strides writes out of bounds. This is a footgun in `np.empty(Shape)`, unrelated to this lever, and it
crashes both the old ForEach path and the new kernel equally (a malformed array). Real usage — no-`out`
(the production path `.Clean()`s to contiguous) or a properly-backed strided `out` — is correct.

Side benefit: the N-D flatten routes flattenable narrow-inner N-D views through the new kernel instead of
the ForEach 3-D odometer, so it sidesteps this crash for that case.

---

## 7. Correctness / gates

| Gate | Result |
|---|---|
| FuzzMatrix (all 64 corpus tiers, incl. `out_where` across every strided layout) | **98/98** |
| Main suite (`TestCategory!=OpenBugs&!=HighMemory`) | **14446 passed, 0 failed, 11 skipped** |
| Iterator / NDIter / NDExpr / Evaluate filter | 907 passed |
| 2-D self-consistency probe (strided vs contiguous-copy, ops × dtypes × widths × layouts) | **700/700 bit-identical** |
| N-D self-consistency probe (3-D + 4-D trailing-narrow) | **123/123 bit-identical** |
| Both target frameworks build | net8.0 ✅, net10.0 ✅ |

The self-consistency probes compare the 2-D/N-D strided result against the same op on a contiguous copy of
the input (which takes the proven 1-D path) — the tightest possible check that the new kernel matches the
old one bit-for-bit.

---

## 8. What's left (correct, unaccelerated)

- **inner-broadcast 2-D** — `add(A, col)` where `col` broadcasts over the inner axis (stride-0 inner):
  the gate requires a contiguous inner for all operands, so this keeps the per-chunk route (which SIMDs the
  broadcast per row). Extending the 2-D kernel with a broadcast-inner variant is possible future work.
- **non-flattenable outer** — a doubly-strided `x[::2, :, :w]`: outer axes are not mutually contiguous, so
  they cannot collapse to a single stride; keeps the per-chunk route. A full N-D outer odometer in the
  kernel would cover it but re-introduces per-row odometer cost.
- **non-SIMD dtypes** (Half / Decimal / Complex) and **comparisons** (mixed dtype → bool) are excluded by
  design; comparisons already have the whole-array Direct strided kernel.
- **`np.empty(Shape)` inheriting view strides** — a separate pre-existing footgun (§6).

---

## 9. Reproduction

```bash
# baseline / after — narrow strided rows, both sides
DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
  dotnet run -c Release benchmark/nditer/probes/angles_probe.cs         # §1 = narrow rows
OPENBLAS_NUM_THREADS=1 python benchmark/nditer/probes/numpy_twins.py angles

# the gates
(cd test/NumSharp.Tests.Oracle && dotnet test --no-build -c Release -f net10.0 --filter "TestCategory=FuzzMatrix")
(cd test/NumSharp.Tests && dotnet test --no-build -c Release -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory")
```

Measurement discipline (see `docs/NDITER_PERF_DISCOVERY.md` §3): Release only, `DOTNET_TC_CallCountingDelayMs=0`,
`OPENBLAS_NUM_THREADS=1`, `#:property PublishAot=false`, a **fresh filename per rebuild** (a stale `#:project`
DLL cache silently runs the old Core — the tell here is the 2-D cache count not growing), interleaved and
repeated on an idle host.

---

## 10. See also

- `docs/NDITER_PERF_DISCOVERY.md` — the cost model, the measuring traps, the probes, and the full ranked
  next-wins list (this is §7 angle 2).
- `.claude/CLAUDE.md` → the NDIter section — the concise landed-lever summary.
