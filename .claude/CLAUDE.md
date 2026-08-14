# NumSharp Project Instructions

NumSharp is a .NET port of Python's NumPy library targeting **1-to-1 API and behavioral compatibility with NumPy 2.x**.

## NumPy Reference Source

A full clone of the NumPy repository is available at `src/numpy/`. Use this as the authoritative reference for API behavior, edge cases, and implementation details when implementing or verifying NumSharp functions.

## Core Principles

1. **Match NumPy Exactly**: Run actual Python/NumPy code first, observe behavior, replicate in C#
2. **Match NumPy Implementation Patterns**: Don't just match behavior - match NumPy's implementation structure. If NumPy has a clean approach and NumSharp has spaghetti code, refactor to match NumPy's design
3. **Edge Cases Matter**: NaN handling, empty arrays, type promotion, broadcasting, negative axis
4. **Breaking Changes OK**: Breaking changes are acceptable to match NumPy
5. **Test From NumPy Output**: Tests should be based on running actual NumPy code

**When fixing bugs:** Don't just patch symptoms. Check `src/numpy/` for how NumPy implements the same functionality, then refactor NumSharp to match NumPy's structure.

## Definition of Done (DOD) - Operations

Every np.* function and DefaultEngine operation MUST satisfy these criteria:

### Memory Layout Support
- **Contiguous arrays**: Works correctly with C-contiguous memory (SIMD fast path)
- **Non-contiguous arrays**: Works correctly with sliced/strided/transposed views
- **Broadcast arrays**: Works correctly with stride=0 dimensions (read-only)
- **Sliced views**: Correctly handles Shape.offset for base address calculation

### Dtype Support
All 15 NumSharp types must be handled (or explicitly documented as unsupported):
Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Half, Single, Double, Decimal, Complex

### NumPy API Parity
- Function signature matches NumPy (parameter names, order, defaults)
- Type promotion matches NumPy 2.x (NEP50)
- Edge cases match NumPy (empty arrays, scalars, NaN handling, broadcasting)
- Return dtype matches NumPy exactly

### Testing
- Unit tests based on actual NumPy output
- Edge case tests (empty, scalar, broadcast, strided)
- Dtype coverage tests

## Supported Types (15)

| NPTypeCode | C# Type | NPTypeCode | C# Type |
|------------|---------|------------|---------|
| Boolean | bool | UInt64 | ulong |
| Byte | byte | Char | char |
| SByte | sbyte | Half | System.Half |
| Int16 | short | Single | float |
| UInt16 | ushort | Double | double |
| Int32 | int | Decimal | decimal |
| UInt32 | uint | Complex | System.Numerics.Complex |
| Int64 | long |  |  |

All operations must handle all 15 types via type switch pattern.

**Perf notes:**
- SByte / Byte / Int*/UInt* / Single / Double — full SIMD via the mixed-type kernel's `SimdFull` execution path (V128/V256/V512 detected at startup).
- Half — scalar path (no `Vector<Half>` arithmetic in .NET BCL). Routes through `Half→double→Math.Pow→Half` for `np.power`; ~2× slower than NumPy.
- Complex — scalar path via `System.Numerics.Complex` operators / `Complex.Pow`. ~2× slower than NumPy.
- Decimal — scalar path via `DecimalMath.Pow`. Highest precision, slowest.

## Architecture

```
NDArray           Main class (like numpy.ndarray)
├── Storage       UnmanagedStorage (raw pointers, not managed arrays)
├── Shape         Dimensions, strides, offset calculation
└── TensorEngine  Computation backend (DefaultEngine = pure C#)

np                Static API class (like `import numpy as np`)
├── np.random     NumPyRandom (1-to-1 seed/state with NumPy)
└── np.*          Functions in Creation/, Math/, Statistics/, Logic/, etc.
```

### NumSharp.Core is 100 % managed C# — optional packages are the only native path

`NumSharp.Core` has **no native dependency and no P/Invoke**: every kernel is its own managed code,
and that is the default a user gets. A native backend arrives ONLY as a separate NuGet package, and
Core's whole knowledge of it is ONE abstraction plus ONE settable property on the engine:

```csharp
public interface IBlasBackend                                    // Backends/IBlasBackend.cs
{
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;   // optional (DIM)

    // + 15 more DIMs in IBlasBackend.LinearAlgebra.cs, all defaulting to false:
    //   products      TryInner TryVdot TryVecdot TryMatvec TryVecmat
    //   factorisations TryCholesky TryDet TrySlogdet TryEig TryEigh TryInv
    //                  TryLstsq TryQr TrySolve TrySvd
}

public abstract partial class TensorEngine
{
    public IBlasBackend Blas { get; set; }   // null (the default) = NumSharp's own managed kernels
}
```

The package implements the interface and assigns the property from a `[ModuleInitializer]`, so
**referencing the package is the whole opt-in**; assigning null takes it back out. Every member is
`Try`-shaped: a backend answers only for the operands it implements and returns false for the rest,
so an optional package can change *which* implementation computes a product, never *whether* one can
be computed. The engine is **not** subclassed and not replaced — deliberately: an `NDArray` binds
its engine at CONSTRUCTION and `np.dot(a,b)` dispatches on `a.TensorEngine`, so swapping the engine
would only reach arrays created afterwards, while a property on the cached singleton takes effect
for arrays that already exist (probed: 38/38 construction/operation/IO paths reach the one
`EngineCache<DefaultEngine>` instance).

Three rules the review pinned, all with a reproduction (`docs/GEMM_PARITY.md` §9): read `Blas` into
a **local** before calling it (test-then-call on a settable property was ~2 % NREs under concurrent
`Disable()`); a failed `OpenBlasEngine.Enable` is a **no-op** (it used to unload the working library first,
leaving `Enabled` reporting true while products silently reverted to managed kernels); and a backend
needs **no** `InternalsVisibleTo` — read operands as `(T*)a.GetData().Address + a.Shape.Offset` with
`a.Shape.Strides` (all public; `a.GetData<T>().Address` densifies non-contiguous views and must not
be paired with strides). `TryMatMulBatched` is a default interface method taking a whole stacked
product, so per-product setup is hoisted the way NumPy hoists it out of its gufunc's outer loop —
bit-identical to the per-element route (25/25 A/B), 0.80× → **6.47×** on 2000 stacked 8×8 f32.

| Package | Implements | Serves | Why |
|---------|-----------|--------|-----|
| `NumSharp.Interop.OpenBLAS` (`src/NumSharp.Interop.OpenBLAS/`) | `OpenBlasBackend : IBlasBackend` | `np.dot`, `np.matmul` for float32/float64 | matrix products through OpenBLAS — faster on large matrices, and **byte-identical to NumPy**. **Bundles the binaries NumPy itself pins** (scipy-openblas 0.3.31.22.0, byte-identical to numpy 2.4.2's) as per-RID runtime assets, so parity needs no Python installed. See below and `docs/GEMM_PARITY.md`. |

**`NumSharp.Interop.OpenBLAS`** exists because no portable algorithm can match NumPy's float matrix
products: for f32/f64 mat@mat NumPy **always** calls cblas (since gh-23588 it copies non-blasable
operands into a temp rather than take its own portable loop), and scipy-openblas' `sgemm` uses an
arch-specific **multi-accumulator** scheme matching neither a sequential mul+add chain nor a
sequential FMA chain — NumSharp's managed GEMM differed on **94.5 %** of a `(128,784)@(784,128)` f32
product (max ~976 K ULP) and **45 %** at K=10, which compounds until 50 Adam steps spread it to 73 %
of a weight matrix. The backend is a route-for-route port of **both** NumPy dispatchers, because
they are different C code and **disagree** when an operand is not blasable (a stride-2 matrix @ vector:
`np.dot` takes gemv-on-a-copy, `np.matmul` the portable loop — 278/300 elements differ): `Dot` →
`cblas_matrixproduct` + the N-D `dotfunc` tail (which is NOT gemm), `MultiplyMatrix` →
`@TYPE@_matmul`'s five routes incl. the `a @ a.T` **syrk** shortcut and NumPy's copy/transpose
rules — which is why `IBlasBackend` has both `TryDot` and `TryMatMul2D` rather than one
matrix-product method. Strides are ported in ELEMENTS (NumPy's are bytes — same logic with
`itemsize == 1`). Scope is Single/Double; every other dtype falls through to the managed kernel
(integer products are bit-exact by construction — modular addition is associative).

**The package BUNDLES the binary** (2026-07-26). It ships the prebuilt OpenBLAS artifacts NumPy
itself pins — `scipy-openblas64==0.3.31.22.0` from numpy's `requirements/ci_requirements.txt`,
**verified byte-identical** to numpy 2.4.2's copy (both sha256 `74a40872…`; NumPy's mangled filename
IS that hash) — as `runtimes/<rid>/native/` assets for 8 RIDs (62 MB packed). This is not a reversal
of the old "ships no native binaries" rule but its resolution: the objection was to bundling *an*
OpenBLAS, and this is *the* one. **win-arm64 is scipy-openblas32**, because NumPy's own build selects
LP64 there — it exports `scipy_cblas_sgemm`, a symbol scheme `Bind` lacked (added; binding a
win-arm64 numpy's BLAS would have failed outright). Binaries are gitignored;
`tools/openblas-manifest.json` is the checked-in pin and `tools/fetch_openblas.py` verifies **two**
hashes per RID. Discovery order is fixed (`docs/OPENBLAS_DELIVERY_DESIGN.md` §3; runtime side
implemented, gate `OpenBlasDeliveryTests`): explicit path/`NUMSHARP_OPENBLAS_PARITY` (binding) →
override path(s) (`NUMSHARP_OPENBLAS_PATH` env, then a build-recorded `OpenBlasPath` marker;
non-binding) → build-staged **version override** (`openblas.source.json` marker, `mode:"version"` —
**hard-required**: a miss throws `OpenBlasRequiredOverrideException`, never falls through; sha-verified
when pinned) → explicit OpenBLAS root (`OPENBLAS_HOME`/`OPENBLAS_ROOT`, promoted ABOVE the bundle
2026-08-14 — a deliberate opt-in that trades parity-by-default for the named build) → **bundled**
(the parity default; §10.1 resolved: it stays ABOVE *ambient* machine tooling) →
machine-wide OpenBLAS (apt/brew/MacPorts/conda/vcpkg/source install dirs; honours the ambient
`VCPKG_ROOT`/`CONDA_PREFIX`; **not** parity) → bare names (64-bit **and**
32-bit spellings — both scipy-openblas64/ILP64 and scipy-openblas32/LP64 bind, and 32 is the only
build for 32-bit x86) → PATH sweep (last resort, non-standard names); `NUMSHARP_OPENBLAS_BUNDLED=0` drops
the bundled entry. `numpy.libs` is NEVER scanned (tier deleted — we never grab OpenBLAS out of a
numpy installation; the bundle already IS that binary at the pin). **Build-time override** (design
§5–§7, shipped in `buildTransitive/{props,targets}` + an inline `RoslynCodeTaskFactory` task —
netstandard2.0 surface only, own `MiniJson`, since the factory can't resolve simple-name References):
`<OpenBlasVersion>` on ANY `PackageReference` (or `NUMSHARP_OPENBLAS_VERSION` — env wins) downloads
that scipy-openblas from PyPI at build, double-hash-verifies, caches the EXTRACTED lib per-user
(`%LOCALAPPDATA%/NumSharp/openblas`, wheel deleted, offline-capable once primed), stages it over the
bundle and writes the marker; failure = BUILD ERROR (hard requirement). `<OpenBlasPath>` = soft
read-in-place dir (also the download-to dir when combined with a version);
`<OpenBlasDelivery>package</OpenBlasDelivery>` bakes ALL 8 RIDs + markers into a dependent's own
nupkg. Two MSBuild traps encoded there: a target's `Condition` evaluates BEFORE `DependsOnTargets`
run (the has-override gate must sit on the TASK, not the target), and `dotnet pack` defaults to
Release on .NET 8+ while this repo's incremental build can declare stale Release outputs up-to-date
— always `-t:Rebuild` before packing. The cache is SELF-HEALING: every hit re-verifies the entry's
content hash (the dir name IS the claim) and a truncated/tampered entry is discarded and
re-downloaded; writes are temp+rename so killed/racing builds can't strand garbage under the hash
name. Gate: `tools/verify_build_override.sh` (15-step scripted nupkg-flow run incl. tamper,
mirror-needs-sha, invalid-knob, delivery=none, two-ref-conflict and poisoned-cache steps; network
once, then cache) + `OpenBlasDeliveryTests` (20 — priority pins and hostile markers).

**Four load-bearing details:** the result bits depend on the BLAS **thread count** (1/2/4/24 threads
give four different answers); they ALSO depend on the **DYNAMIC_ARCH kernel** the CPU dispatches, so
bundling pins the build but *not* the answer — `OpenBlasEngine.Enable(coreType: OpenBlasEngine.CoreType)` pins that
too (probed: Haswell/Nehalem/Sandybridge/Katmai all differ; it is opt-in because the default must
match the local NumPy, which also dispatches by CPU; it must be set BEFORE load, and forcing an ISA
above the CPU **kills the process with SIGILL**, so `Enable` refuses what it can recognise). A
**named library is binding** (`OpenBlasEngine.Enable(path)` / `NUMSHARP_OPENBLAS_PARITY` is never silently
substituted, not even by the bundled copy). And the BLAS path is *faster* than the managed GEMM
anyway (1.7–13×; 2048³ f64 293 ms vs 3860 ms).

**Trap:** `Environment.SetEnvironmentVariable` does NOT reach a native `getenv` — .NET keeps its own
table, so setting `OPENBLAS_CORETYPE` that way reads back correctly and changes nothing. Go through
the CRT (`ucrtbase!_putenv_s` / `libc!setenv`); Windows' `SetEnvironmentVariableW` does not work
either.

API: `OpenBlasEngine.Enable(library, threads, coreType)` / `OpenBlasEngine.Disable()` / `OpenBlasEngine.Enabled` / `OpenBlasEngine.Info` /
`OpenBlasEngine.LibraryPath` / `OpenBlasEngine.CoreName` / `OpenBlasEngine.IsBundledLibrary` (marker-aware: a staged override in
the same folder layout reports false) / `OpenBlasEngine.CoreType`; `NUMSHARP_OPENBLAS_PATH` adds
highest-priority (non-binding) discovery locations tried before bundled;
`NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0` opts out of the module-load install (`OpenBlasEngine.BundleAutoinstall`;
the pre-rename `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL` and `NUMSHARP_BLAS_AUTOINSTALL` spellings are RETIRED
and ignored — removed before any
release shipped them, pinned by `RetiredAutoinstallAlias_NoLongerSuppresses` — and a
required-override miss at module load is reported to stderr, backend left uninstalled — never
substituted). Gate: the **host-pinned**
`matmul_parity` corpus tier (342 cases — 342 bit-exact with the backend, 294 divergent without it)
which goes `Inconclusive`, never red, on a host that cannot load the pinned library. The pin matches
the library by **content hash**, not filename — bundling forced that and it is strictly stronger
(a name is mangled by delvewheel on one side and plain on the other, so name-matching skipped a
genuinely bit-identical host and would have accepted a differently-built same-named library).
Packaging contract: `MatmulParityBackendTests` (30 tests).

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Unmanaged memory | Benchmarked fastest; optimized for performance |
| Order-aware layout | Row-major (C-order) remains the default. `Shape` also tracks F-contiguity, and APIs with an `order` parameter resolve NumPy `C`/`F`/`A`/`K` modes through `OrderResolver`. |
| TensorEngine abstract | Pluggable computation backend behind one contract (`DefaultEngine` = pure C#) |
| View semantics | Slicing returns views (shared memory), not copies |
| Shape readonly struct | Immutable after construction (NumPy-aligned). Contains `ArrayFlags` for cached O(1) property access |
| Broadcast write protection | Broadcast views are read-only (`IsWriteable = false`), matching NumPy behavior |
| ILKernelGenerator + DirectILKernelGenerator | Runtime IL emission with SIMD V128/V256/V512 (~36K lines). Two classes split by kernel-driving contract: `ILKernelGenerator` (per-chunk kernels driven by NDIter — `np.where`, flat/pairwise reductions) and `DirectILKernelGenerator` (whole-array kernels, 63 partials in `Direct/`). |

## ILKernelGenerator + DirectILKernelGenerator

Runtime IL generation via `System.Reflection.Emit.DynamicMethod` for high-performance kernels. **Two physically distinct classes**, each encoding its kernel-driving contract in the type name:

### `DirectILKernelGenerator` — whole-array kernels (63 partials)

**Location:** `src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.*.cs`

**Contract:** A single kernel call processes the **whole array**. The kernel itself walks dimensions/strides; the iterator (if any) is only a setup helper. Signature shape varies per kernel family but typically:

```csharp
delegate void DirectKernel(
    void* input, void* output,
    long* strides, long* shape, int ndim,
    long iterSize);
```

Carries the bulk of NumSharp's elementwise, reduction, scan, cast, and selection kernels.

**Partial files in `Direct/` (63):**
| Category | Files |
|----------|-------|
| Core | `DirectILKernelGenerator.cs` (type mapping, SIMD detection), `.Scalar.cs` |
| Binary | `.Binary.cs`, `.MixedType.cs`, `.Shift.cs` |
| Unary | `.Unary.cs`, `.Unary.Math.cs`, `.Unary.Decimal.cs`, `.Unary.Vector.cs`, `.Unary.Predicate.cs`, `.Unary.Strided.cs` |
| Comparison | `.Comparison.cs` |
| Reduction (flat) | `.Reduction.cs`, `.Reduction.Arg.cs`, `.Reduction.Boolean.cs`, `.Reduction.NaN.cs` |
| Reduction (axis) | `.Reduction.Axis.cs`, `.Reduction.Axis.Arg.cs`, `.Reduction.Axis.Boolean.cs`, `.Reduction.Axis.Simd.cs`, `.Reduction.Axis.NaN.cs`, `.Reduction.Axis.VarStd.cs`, `.Reduction.Axis.Widening.cs` |
| Scan | `.Scan.cs` (CumSum, CumProd) |
| Masking | `.Masking.Boolean.cs`, `.Masking.NaN.cs`, `.Masking.VarStd.cs` |
| Cast & Copy | `.Cast.cs`, `.Cast.Masked.cs`, `.Cast.Scalar.cs`, `.Cast.Half.cs`, `.Cast.ToHalf.cs`, `.Cast.FloatNarrow.cs`, `.Cast.FloatWideInt.cs`, `.Cast.FloatToUInt.cs`, `.Cast.IntNarrow.cs`, `.Cast.ShortNarrow.cs`, `.Cast.SubwordCopy.cs`, `.Cast.SubwordNarrow.cs`, `.Cast.SubwordWiden.cs`, `.Cast.ToBool.cs`, `.Cast.Complex.cs`, `.Copy.cs` |
| Selection | `.Where.cs`, `.Where.Scalar.cs`, `.Place.cs`, `.Put.cs`, `.Take.cs`, `.NonZero.cs`, `.Argwhere.cs`, `.Indices.cs`, `.Filter.cs`, `.Search.cs` |
| Linear algebra | `.MatMul.cs`, `.Trace.cs` |
| Other | `.Clip.cs`, `.Modf.cs`, `.Repeat.cs`, `.Quantile.cs`, `.WeightedSum.cs`, `.RavelMultiIndex.cs`, `.UnravelIndex.cs`, `.InnerLoop.cs`, `.StorageAlias.cs` |

### `ILKernelGenerator` — NDIter-driven per-chunk kernels

**Location:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator*.cs` (root, not in `Direct/`)

**Contract:** Kernel processes **one inner-loop chunk** per call. The iterator (`NDIterRef`) drives the loop and advances pointers between calls. Matches NumPy's `PyUFuncGenericFunction` model:

```csharp
unsafe delegate void NDInnerLoopFunc(
    void** dataptrs,   // [nop] current operand pointers
    long*  strides,    // [nop] per-operand byte stride for inner loop
    long   count,      // number of elements this chunk
    void*  auxdata);   // op-specific extras (e.g. axis index)
```

Emits the per-chunk kernels run as an NDIter inner loop — `np.where` and the flat/pairwise reductions (partials `ILKernelGenerator.{Where,Reduction,Reduction.Pairwise}.cs`). Driven via `NDIterRef.Execute(kernelKey)`.

### Shared infrastructure

Both classes use these helpers at `src/NumSharp.Core/Backends/Kernels/` (root, outside `Direct/`):
- `VectorMethodCache.cs`, `ScalarMethodCache.cs` — reflection caches (Vector{128,256,512}, Math/MathF)
- `KernelOp.cs` — `BinaryOp`, `UnaryOp`, `ReductionOp`, `ComparisonOp`, `ExecutionPath` enums
- `BinaryKernel.cs`, `CopyKernel.cs`, `ReductionKernel.cs`, `ScalarKernel.cs` — kernel key structs and delegate types
- `StrideDetector.cs` — layout classification (Contiguous / Strided / Broadcast)
- `SimdMatMul.*.cs` — matmul SIMD primitives

**Execution Paths (DirectILKernelGenerator):**
1. **SimdFull** - Both operands contiguous, SIMD-capable dtype → Vector loop + scalar tail
2. **ScalarFull** - Both contiguous, non-SIMD dtype (Decimal) → Scalar loop
3. **General** - Strided/broadcast → Coordinate-based iteration

**NEP50 Dtype Alignment (NumPy 2.x):**
| Operation | Returns |
|-----------|---------|
| `sum(int32)` | `int64` |
| `prod(int32)` | `int64` |
| `cumsum(int32)` | `int64` |
| `abs(int32)` | `int32` (preserves) |
| `sign(int32)` | `int32` (preserves) |
| `power(int32, float)` | `float64` |

**ILKernel Coverage:**
| Category | Operations |
|----------|------------|
| Binary | Add, Sub, Mul, Div, Power, FloorDivide, BitwiseAnd/Or/Xor |
| Shift | LeftShift, RightShift (SIMD for scalar, scalar loop for array) |
| Unary | Negate, Abs, Sign, Sqrt, Cbrt, Square, Reciprocal, Floor, Ceil, Truncate, Trig, Exp, Log, BitwiseNot |
| Reduction | Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any, Std, Var |
| Scan | CumSum, CumProd (element-wise SIMD + axis support) |
| Comparison | Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual |
| Clip/Modf | Clip, Modf (SIMD helpers) |
| Axis reductions | Sum, Prod, Min, Max, Mean, Std, Var (iterator path) |
| Cast (SIMD, contig+strided) | float→int32 (cvtt); float/c128→u32, →u64 (AVX2 mod-2³² reduce + hi/lo split, bit-exact incl. NumPy modular wrap — but note the NaN/±inf/out-of-range inputs are a **C-undefined conversion**: NumSharp reproduces the MSVC NumPy answer, glibc/gcc's differs, and so do one build's vectorized vs scalar loops, so that cell is a NumSharp regression pin rather than portable NumPy parity; see `Fuzz/README.md` → "Host-dependent values"; strided →u64 negcol `[:, ::-1]` via contiguous load + VPERMQ reverse, `::2` via 2-load deinterleave (b loaded at `2i+3` so no over-read), c128 negcol via UnpackLow+VPERMQ-0x72 deinterleave-reverse — reals for →u64, reals **and** imags for →bool); float→{i8,u8,i16,u16,char} (cvtt+Narrow); int→narrower-int (Narrow); Half↔ via Giesen bit-fiddle: f16→f32 widen (NaN-payload exact), f16→i64, f16→{i32,u32,u64,narrow}; X→Half (Giesen narrow, sNaN-preserving) incl. i64/u64→f16 (AVX2: clamp |v|≥65520→±70000 sentinel → low-32 PermuteVar8x32 pack → cvtdq2ps → Giesen); c128→{int,bool} (real deinterleave); *→bool (!=0; f16→bool strided via SubwordNarrow-style deinterleave (`::2`)/reverse (`::-1`) + NumPy half-truthiness `(bits & 0x7fff)!=0` so ±0→False, NaN/inf→True). Strided rows gather (4/8-byte) or stage gather→buffer→convert. Sub-word (1B/2B) strided/negcol (non-gatherable) use SIMD lane shuffles (`.Cast.SubwordCopy.cs`/`.Cast.SubwordNarrow.cs`): same-type & same-size-bit-reinterpret copies via VPACKUS deinterleave (`[:, ::2]`) / VPSHUFB reverse (`[:, ::-1]`); 2B-int→{1B,bool} via deinterleave/reverse + narrow; 1B-int→2B via deinterleave/reverse + sign/zero `Vector256.Widen` (`.Cast.SubwordWiden.cs`). The inner SIMD loop is inlined in the odometer. |

## Shape Architecture (NumPy-Aligned)

Shape is a `readonly struct` with cached `ArrayFlags` computed at construction:

```csharp
public readonly partial struct Shape
{
    internal readonly int _flags;         // Cached ArrayFlags bitmask
    internal readonly int _hashCode;      // Precomputed hash code
    internal readonly long size;          // Total element count
    internal readonly long[] dimensions;  // Dimension sizes
    internal readonly long[] strides;     // Stride values (0 = broadcast dimension)
    internal readonly long bufferSize;    // Size of underlying buffer
    internal readonly long offset;        // Base offset into storage
}
```

**ArrayFlags enum** (matches NumPy's `ndarraytypes.h`):
| Flag | Value | Meaning |
|------|-------|---------|
| `C_CONTIGUOUS` | 0x0001 | Data is row-major contiguous |
| `F_CONTIGUOUS` | 0x0002 | Data is column-major contiguous |
| `OWNDATA` | 0x0004 | Array owns its data buffer |
| `ALIGNED` | 0x0100 | Always true for managed allocations |
| `WRITEABLE` | 0x0400 | False for broadcast views |
| `BROADCASTED` | 0x1000 | Has stride=0 with dim > 1 |

**Key Shape properties:**
- `IsContiguous` — O(1) check via `C_CONTIGUOUS` flag
- `IsFContiguous` — O(1) check via `F_CONTIGUOUS` flag
- `IsBroadcasted` — O(1) check via `BROADCASTED` flag
- `IsWriteable` — False for broadcast views (prevents corruption)
- `IsSliced` — True if offset != 0, different size, or non-contiguous
- `IsSimpleSlice` — IsSliced && !IsBroadcasted (fast offset path)

## Critical: View Semantics

**Slicing returns views, not copies!** Memory is shared:
```csharp
var view = original["2:5"];  // Shares memory with original
view[0] = 999;               // Modifies original[2]!
var copy = original["2:5"].copy();  // Explicit copy
```

## Slicing Syntax

```csharp
nd[":"]           // All elements
nd["1:5"]         // Elements 1-4 (stop exclusive)
nd["::2"]         // Every 2nd element
nd["-1"]          // Last element (reduces dimension)
nd["::-1"]        // Reversed
nd[":, 0"]        // All rows, first column
nd["..., -1"]     // Ellipsis fills dimensions
```

---

## Supported np.* APIs

Tested against NumPy 2.x.

### Array Creation
`arange`, `array`, `asanyarray`, `asarray`, `asarray_chkfinite`, `ascontiguousarray`, `asfortranarray`, `asmatrix`, `copy`, `empty`, `empty_like`, `eye`, `frombuffer`, `full`, `full_like`, `identity`, `linspace`, `meshgrid`, `mgrid`, `ogrid`, `ones`, `ones_like`, `require`, `tri`, `zeros`, `zeros_like`

The `as*` conversion family mirrors NumPy: `asarray_chkfinite(a, dtype=None, order='K')` = `asarray` then raise `ValueError("array must not contain infs or NaNs")` if a **float-family** dtype (Half/Single/Double/Complex — NumPy's `typecodes['AllFloat']`; Decimal/int/bool skip the check) holds any inf/NaN, via a **fused single-pass NaN-poison SIMD reduction** (`Backends/Kernels/FiniteScan.cs`: `acc += v - v` — +0 for finite, absorbing-NaN for non-finite; AVX2 gather + reversed-contiguous fast path for strided/negative-stride views; ~2–27× NumPy contiguous, ≥1× strided). `require(a, dtype=None, requirements=None)` parses C/F/A/W/O/E flags (+aliases; single-string requirements iterate by char like NumPy, so `"F_CONTIGUOUS"` as one string raises), resolves an order and copies only if a remaining ALIGNED/WRITEABLE/OWNDATA flag is unsatisfied (ALIGNED is always true in NumSharp, so only broadcast-non-writeable and views force a copy). `asmatrix(data, dtype=None)` returns a **2-D view** (NumSharp has no `matrix` subclass — the deprecated NumPy one; no `*`-as-matmul/`.H`/`.I`): 0-D→(1,1), 1-D→(1,N), 2-D unchanged, >2-D drops length-1 axes and must land on 2-D else `ValueError("shape too large to be a matrix.")`; also parses matrix strings (`"1 2; 3 4"`). See `Creation/np.{asarray_chkfinite,require,asmatrix}.cs`.

### Shape Manipulation
`append`, `array_split`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `block`, `c_`, `column_stack`, `concat`, `concatenate`, `delete`, `dsplit`, `dstack`, `expand_dims`, `flatten`, `flip`, `fliplr`, `flipud`, `hsplit`, `hstack`, `insert`, `intersect1d`, `matrix_transpose`, `moveaxis`, `pad`, `permute_dims`, `r_`, `ravel`, `repeat`, `reshape`, `resize`, `roll`, `rollaxis`, `rot90`, `setdiff1d`, `setxor1d`, `split`, `squeeze`, `stack`, `swapaxes`, `tile`, `transpose`, `trim_zeros`, `union1d`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`, `unstack`, `vsplit`, `vstack`

**Set routines** `intersect1d`, `union1d`, `setxor1d`, `setdiff1d` (NumPy `_arraysetops_impl.py`, sorted/unique set
operations over flattened inputs; all probed against 2.4.2; gate `Manipulation/np.setops.Test.cs` (28) + the
`groupa` fuzz tier) are pure compositions of the existing `unique`/`sort`/`argsort`/`concatenate` machinery, ported
line-for-line from NumPy so their promotion, NaN handling and edge cases match by construction. `union1d(ar1, ar2)`
= `unique(concatenate(ravel(ar1), ravel(ar2)))` — unique sorted union in the promoted dtype. `intersect1d` returns
the sorted common values through a **3-overload split** that reproduces NumPy's bare-array-vs-tuple return
polymorphism: `intersect1d(ar1, ar2)` and `intersect1d(ar1, ar2, assume_unique)` are the bare-return forms
(→`NDArray`), while `intersect1d(ar1, ar2, assume_unique=false, return_indices=false)`→`NDArray[]{values, comm1,
comm2}` carries BOTH keywords optional so `np.intersect1d(a, b, return_indices: true)` (assume_unique omitted) ports
verbatim — the bare overloads have no `return_indices` parameter so they can't capture it, and C# betterness resolves
`intersect1d(a,b[,au])` to the bare `NDArray` form (the multi overload's extra optional loses), so the split is
unambiguous (empirically verified). The tuple path uses a **stable** argsort so the first occurrence in each input
wins (NumSharp's argsort is stable, verified). `setxor1d` returns
values in exactly one input; `setdiff1d(ar1, ar2, assume_unique=false)` returns `ar1`'s values absent from `ar2`
(built on `isin(..., invert=true)`), preserving `ar1`'s dtype and — under `assume_unique` — its duplicates. Empty
inputs collapse to NumPy's float64 empty (from the empty concatenate). NaN never matches itself, so a shared NaN is
excluded from `intersect1d` but each NaN survives `setxor1d`/`union1d`/`setdiff1d` as distinct (matching NumPy). A
surviving **float32/float64** NaN is **canonicalized to NumPy's positive quiet NaN** (`0x7ff8000000000000` /
`0x7fc00000`): NumPy's SIMD float sort (which `unique`/`sort` drive) rewrites EVERY NaN — negative-signed,
signalling, or payload-bearing — to that single value, while **float16 and complex128 NaNs are left exactly as-is**
(probed 2.4.2). NumSharp's own `sort` yields .NET's negative NaN and `unique` preserves payloads, so
`union1d`/`setxor1d`/`setdiff1d` run a guarded NaN-rewrite pass on a float32/float64 result — `np.where(np.isnan(r),
canonical, r)`, entered only when `np.any(np.isnan(r))` (≈0.3 ms on a 490 K result, skipped when no NaN survives) —
to stay bit-identical (`CanonicalizeSetOpNaN` in `np.setops.cs`; verified across neg/payload/multi-NaN × f16/f32/f64
+ complex against NumPy 2.4.2). `isin` (bool result) and `intersect1d` (NaN never shared) need no pass.
**Perf (NPY/NS, Release, best-of-21, warm):** **2.2–13.5×** — NumSharp's `unique`/`sort` outrun NumPy's at every
measured size (the wins widen at 1M, where NumPy's set-ops cost 60–400 ms); the NaN pass is O(result) and does not
move these numbers. See `Manipulation/np.{setops,union1d,intersect1d,setxor1d,setdiff1d}.cs`.

**`np.static unique` returns `np.UniqueResult`** (the shared core the set routines reuse) — NumPy's bare-array-OR-tuple
return expressed as ONE type, because C# cannot pick a return type from a runtime flag. **ONE** static overload
`unique(ar, bool return_index=false, bool return_inverse=false, bool return_counts=false, int? axis=null, bool
equal_nan=true, bool sorted=true)`→`UniqueResult` makes **every** NumPy call shape port verbatim — including the two
that used to need a C# idiom: `np.unique(ar, return_counts: true)` and `np.unique(ar, return_inverse: true)`
(**sole keyword, no `return_index`**), plus `np.unique(ar)`, `np.unique(ar, axis: 0)`, `np.unique(ar, equal_nan:
false)`. `UniqueResult` is a `readonly struct` (nested `np.UniqueResult`) that stands in for both NumPy shapes: it
converts **implicitly to `NDArray`** (the bare form — yields `values`, so `NDArray u = np.unique(ar)` and every
argument-passing/assignment site is unchanged) and **implicitly to `NDArray[]`** (the tuple form — the present
outputs in NumPy field order `[values, index?, inverse?, counts?]`); it indexes (`[k]` = k-th present output, matching
the old `NDArray[]` return), `Deconstruct`s (`var (values, counts) = np.unique(ar, return_counts: true)`), and exposes
named `values`/`indices`/`inverse_indices`/`counts` (**case-identical with NumPy's namedtuple fields** —
NumSharp's whole NumPy surface is lowercase, e.g. `nd.shape`/`nd.size`/`nd.dtype`; the last three are `null`
when not requested). **BREAKING (accepted):**
`np.unique(ar)` now returns `UniqueResult`, not a bare `NDArray` — so `np.unique(ar)[k]` selects the k-th OUTPUT (not
the k-th unique value → use `.values[k]`). To keep the ~230 existing call-sites working unchanged, the struct
**forwards the common `NDArray` surface to `values`** (`size`/`shape`/`Shape`/`ndim`/`dtype`/`typecode`/`array_equal`/
`GetAtIndex`/`GetAtIndex<T>`/`GetDouble`/`GetSingle`/`GetBoolean`/`GetInt32`) — the value the caller means in the
no-flags case — and the test `Should()` extension gained a `UniqueResult` overload (`FluentExtension.cs`). The
**instance** `NDArray.unique(int? axis=null, …)`→`NDArray` and `NDArray.unique(bool return_index, …)`→`NDArray[]`
are UNCHANGED (all internal consumers — `NDArray.unique.Hash.cs`, `np.unique_values.cs` — stay on them); only the
static `np.unique` was lifted to `UniqueResult`, which wraps the instance `NDArray[]`. The other three sibling
result structs (`UniqueAllResult`/`UniqueCountsResult`/`UniqueInverseResult`) are always tuples, so they carry no
`values`-forwarding surface — the asymmetry is deliberate (only `np.unique` has a bare-array case). Gate:
`Manipulation/np.unique.AxisEdgeCases.Test.cs` (`UniqueResult_*` + `BareReturn_*` pins); the whole test suite compiles
and passes against the new return type (differential-verified: `np.unique(x, return_counts=True)` bit-identical to
NumPy 2.4.2). See `Manipulation/np.unique.cs`.

**Array-API unique family** `unique_values`, `unique_counts`, `unique_inverse`, `unique_all` (NumPy 2.x
`_arraysetops_impl.py`; probed against 2.4.2; gate `Manipulation/np.unique_values.Test.cs` (19) + a 102-case
in-process differential across all 13 NumPy dtypes × layouts × NaN/empty/0-d — 102/102 bit-exact). All four are thin
wrappers over the existing `unique` machinery with **`equal_nan=False`** (each NaN is a DISTINCT value — the whole
point of these variants vs `np.unique`'s `equal_nan=True` default) and `axis=None` (flattened). NumPy returns
namedtuples; C# stands in with named-field `readonly struct`s whose fields are **case-identical with NumPy's**
(`UniqueAllResult{values,indices,inverse_indices,counts}`, `UniqueCountsResult{values,counts}`,
`UniqueInverseResult{values,inverse_indices}` — lowercase/snake_case, matching NumPy 2.4.2's `_fields`), each with implicit `NDArray[]`
conversion + `Deconstruct` + indexer (the `np.meshgrid` house shape). `unique_values` returns a bare `NDArray`.
`values` preserves the input dtype; `indices`/`inverse_indices`/`counts` are int64 (NumPy intp). `inverse_indices` is
reshaped to the ORIGINAL input shape (NumPy 2.0: reconstruct via `np.take(values, inverse_indices)`) — the sort path
already yields that for ndim ≥ 1, and `ReshapeInverseToInput` restores the `()` scalar shape a **0-d** input needs
(the flat path returns `(1,)` there — a latent `np.unique` 0-d bug the wrappers route around). **ONE deliberate
divergence, `unique_values` only (`[Misaligned]`):** NumPy's `_unique_hash` (`unique.cpp`) dedups **integer and
complex** dtypes through a `std::unordered_set` under `sorted=False`, returning them in platform/compiler-specific
hash-bucket order (neither sorted nor first-occurrence — e.g. `[0,8,16,24,1,9,17]` → `[24,16,8,0,17,9,1]` on
win-amd64, not reproducible in C#); NumSharp returns them SORTED (identical AS A SET, deterministic, portable, and
consistent with the `values` field of the other three). **Plain float32/64 are NOT in NumPy's hash map**, so they
fall to NumPy's own sort path and are already sorted — NumSharp's float `unique_values` is thus bit-exact with NumPy,
not divergent. **Hash fast path** (`NDArray.unique.Hash.cs`): the 10 integer-family dtypes (bool/byte/sbyte/int16/
uint16/int32/uint32/int64/uint64/char) take a purpose-built open-addressing table (byte-size-keyed generic hash,
load-factor-0.5 growth, **splitmix64 finalizer**) then a sort of the small unique set — mirroring NumPy's own
algorithm SELECTION (NumPy hashes int + complex, both in its `_unique_hash` map, and sorts floats). **Complex** joins
the hash path (NumPy's scalar lexicographic complex sort is slow, so it is a clean win, no all-unique regression);
**float/half/double/decimal keep the sort path**. Float hashing was measured end-to-end and REJECTED: it helps
duplicated data but regresses ~2× on all-unique input (insert N + scalar-sort N), and a cheap cardinality sample
provably cannot distinguish "100K-unique-in-1M" (hash helps) from all-unique (hash hurts). The **splitmix64
finalizer is load-bearing** — a single Fibonacci multiply then a low-bits table index clustered integer-valued
double bit patterns (`1.0/2.0/4.0…`, discriminating bits high, low bits zero) into one bucket at ~500 probes/element;
splitmix avalanches every bit down and is unchanged for integers. **Perf (NPY/NS, Release, best-of-rounds, warm):**
`unique_all` **2.3–7.6×** at every measured dtype/size/cardinality (the flagship "give me everything" call, where
NumPy pays for a mergesort argsort + cumsum + nonzero + diff); integer `unique_values` **1.3–5.4×**, integer
`unique_counts` **1.0–1.5×** at ≤1% cardinality; **complex `unique_values`/`unique_counts` 3.0–6.7×**; `unique_inverse`
**0.7–1.4×**. The remaining sub-1.5× cells — high-cardinality `unique_counts`, `unique_inverse`, and ALL float
`unique_values`/`unique_counts` (0.1–0.3×) — are bounded by NumPy's SIMD `vqsort` (`x86-simd-sort`), which NumSharp's
scalar sort core lacks; closing them is a shared-sort-core change, not a property of these wrappers. Full design +
the measured "why the axis comparator stays scalar / why floats stay sorted / why splitmix" decisions:
**`docs/UNIQUE_DESIGN.md`**. See `Manipulation/np.unique_values.cs` + `Manipulation/NDArray.unique.Hash.cs`.

**`np.unique` full parameter parity** — `unique(ar, return_index=false, return_inverse=false,
return_counts=false, axis=null, equal_nan=true, sorted=true)`, all six probed against 2.4.2 (gates
`Manipulation/np.unique.AxisEdgeCases.Test.cs` (9) + `NDArray.unique.Test.cs`, backed by a 450-case
in-process axis differential — 13 dtypes × axis{None,0,1,2,-1} × every return-flag combo × layouts
{C,F,transposed,reversed,sliced} × NaN/-0.0/complex/empty, 448 bit-exact + 2 excused f16). The **`sorted`**
parameter (NumPy 2.3) is accepted for API parity but is a **no-op — NumSharp always returns sorted output**:
NumPy's `sorted=False` hash-iteration order (integer/complex, values-only) is platform-specific and not
reproducible in C# (same divergence class as `unique_values`; identical as a set). An out-of-range axis raises
the house **`AxisError`** ("axis {a} is out of bounds for array of dimension {n}", NumPy-verbatim), reporting
the ORIGINAL axis. **The axis path compares each sub-array as a NumPy structured scalar** (`_arraysetops_impl`
consolidates the axis into a `('f0',…)` void dtype), which has two consequences ported exactly and easy to get
wrong: **`equal_nan` has NO EFFECT on the axis path** (the void dtype's kind isn't in `_unique1d`'s `"cfmM"`
guard, so its NaN-collapsing branch is skipped) — every NaN sub-array is DISTINCT (`nan != nan`), and a
**signed-zero sub-array collapses** (`-0.0 == +0.0`, the surviving zero's sign following input order). The slab
equality therefore uses IEEE `==` (not `.Equals`, which had `nan.Equals(nan)==true` — collapsing NaN rows — and
`(-0.0).Equals(0.0)==false` — splitting signed zeros, both wrong); the slab ORDER uses IEEE `<`/`>` (not
`CompareTo`, so `-0.0`/`+0.0` compare equal) with a stable tie-break by original index (reproducing NumPy's
mergesort argsort, so identical NaN slabs keep input order). **One deliberate divergence (`[Misaligned]`):** for
**float16 + axis + NaN**, NumPy 2.4.2's structured/void sort mis-orders NaN to the FRONT — inconsistent with its
OWN 1-D float16 sort and its float32/64 structured sorts (all NaN-last); NumSharp keeps the consistent, correct
NaN-to-end ordering across every float width (the unique-row set and counts are identical, only NaN position
differs). See `Manipulation/NDArray.unique.{cs,Kwargs.cs}`.

`flip`/`fliplr`/`flipud` return **O(ndim) views** (stride negation + base-offset shift via `Storage.Alias`,
the Transpose pattern — no slice resolution, no data movement), bit-identical to the `m[..., ::-1, ...]`
slice path for every layout incl. F-contiguous/transposed/stepped/broadcast (flip of a read-only broadcast
stays read-only). `flip(m, axis: null)` flips ALL axes; `flip(m, int[])` is the tuple form (empty array
= flip nothing; the axes are normalized NumPy-style: full `AxisError` pass — reporting the original
negative axis — THEN `ValueError("repeated axis")`, so `(0,0,5)` raises AxisError). 0-d: `flip(m)` returns
the scalar (NumPy's `m[()]`), `flip(m, 0)` raises AxisError. `fliplr` requires ndim≥2 / `flipud` ndim≥1
(`ValueError("Input must be >= 2-d." / ">= 1-d.")` verbatim). See `Manipulation/np.flip.cs`.

`rot90(m, k=1, axes=(0,1))` rotates in the plane of two axes by `90°·k` (counterclockwise from the first axis toward the second). Like NumPy it is a **pure view composition** of axis-flips + a transpose — no data moves, so the result shares memory with `m` (read-only when the source is, e.g. a broadcast view). `k` is taken mod 4 (Python-style, so negatives wrap); `k∈{0}` returns a full view, `k∈{2}` flips both axes, `k∈{1,3}` flip-then-transpose. Validation mirrors NumPy verbatim (`ArgumentException`): `len(axes)!=2` → "len(axes) must be 2.", `axes[0]==axes[1] || |axes[0]-axes[1]|==ndim` → "Axes must be different." (this fires *before* the range check, so e.g. `axes=(0,2)` on 2-D reports "different", not out-of-range), else out-of-range. See `Manipulation/np.rot90.cs`.

`np.trim_zeros(filt, trim="fb", axis=null)` crops leading/trailing all-zero hyperplanes (NumPy 2.2.0+ N-dimensional bounding box; probed against 2.4.2). `trim` is case-insensitive ('f'=front, 'b'=back, "fb"/"bf"=both) and validated (`ArgumentException "unexpected character(s) in \`trim\`: '…'"`). `axis=null` trims the smallest bounding box over all axes; an explicit axis/axis-list (int overload + `int[]` overload) trims only those (negative + repeat/out-of-range checks → `ArgumentException "repeated axis in \`axis\` argument"` / `AxisOutOfRangeException`); an **empty** axis list or a **0-d** input returns the input unmodified. Always returns a **view** (dtype & ndim preserved); all-zero/empty inputs collapse the trimmed axes to length 0 (trim spec ignored, matching NumPy). Implementation is NOT NumPy's `argwhere.min/max` (which materializes every non-zero coordinate) — the bounding box is separable, so it takes the first/last non-zero index from `np.nonzero` (1-D) or `np.nonzero(np.any(filt, other_axes))` (N-D), touching only the trimmed axes. Measured **3.4×–361× faster than NumPy 2.4.2** (min on tiny 1-D, up to 361× on 3-D single-axis). Differential-fuzzed 400/400 bit-exact across 1–3D × {int/uint/float/bool} × sparsity × trim × axis. See `Manipulation/np.trim_zeros.cs`.

`permute_dims` and `matrix_transpose` are the NumPy 2.x / Array-API transpose aliases (both pure O(1) views). `np.permute_dims(a, axes=null)` is a straight alias of `transpose` (delegates to `TensorEngine.Transpose`). `np.matrix_transpose(x)` swaps the two innermost axes (`(..., M, N)→(..., N, M)`, i.e. `swapaxes(x, -1, -2)`) and requires `ndim ≥ 2`, else `ArgumentException "Input array must be at least 2-dimensional, but it is {ndim}"`; the companion property `ndarray.mT` does the same but raises the NumPy-verbatim `"matrix transpose with ndim < 2 is undefined"` (distinct message, both probed against 2.4.2). Read-only/broadcast inputs stay non-writeable through the alias. See `Manipulation/np.permute_dims.cs`, `Manipulation/np.matrix_transpose.cs`, `Backends/NDArray.cs` (`mT`).

`Transpose` axis-handling is NumPy-exact (probed against 2.4.2, shared by `transpose`/`permute_dims`/`swapaxes`/`moveaxis`/`rollaxis`): only `axes=null` reverses — an explicit **length-0** `axes` (`new int[0]`, NumPy's `()`/`[]`) is a length-0 permutation that matches **only a 0-d array** and raises `ArgumentException "axes don't match array"` for any `ndim ≥ 1` (NumPy's `ValueError`); wrong-length and repeated/negative-duplicate axes raise `ArgumentException` ("axes don't match array" / "repeated axis in transpose", NumPy `ValueError`); out-of-range axes raise `AxisOutOfRangeException` (NumPy `AxisError`). See `Backends/Default/ArrayManipulation/Default.Transpose.cs`.

**Stacking/joining details** (all probed against NumPy 2.4.2; tests in `Creation/np.{block,column_stack,concat,unstack}.Test.cs`):
- `concat` — NumPy 2.0 Array-API alias of `concatenate`; mirrors every overload (array form with `axis`/`out`/`dtype`/`casting` + tuple-arity forms). `Creation/np.concat.cs`.
- `unstack(x, axis=0)` (NumPy 2.1) — returns `NDArray[]` of **views** (shared memory; write-through), each `x.shape` minus the axis entry. Built by dropping the axis from dims/strides and advancing offset — no moveaxis call, no iterator. 1-D input → 0-d views; 0-d input → `ValueError("Input array must be at least 1-d.")` verbatim. `Creation/np.unstack.cs`.
- `column_stack(tup)` — sub-2-D inputs become columns via a single-alias `(N,1)` view (NumPy's `array(a, ndmin=2).T`), then `concatenate(axis=1)`; ndim≥2 pass through untouched (3-D+ works, like NumPy). `Creation/np.column_stack.cs`.
- `block(arrays)` — full port of `numpy/_core/shape_base.py`: depth-check walk (verbatim ValueError texts for depth mismatch / empty lists; `TypeError` for `ITuple`s), `_atleast_nd` leading-1 view padding, and BOTH algorithms behind NumPy's exact `list_ndim * final_size > 2*512*512` threshold — repeated-concatenate (intermediates disposed eagerly) vs single-allocation `_block_slicing` (per-leaf `NDIter.Copy`, `order='F' if allF and not allC`). C# nesting map: `object[]`/`NDArray[]`/1-D primitive arrays/`IList` = Python list; `NDArray`/scalars/rank≥2 rectangular arrays = leaves; `np.block(x)` depth-0 returns a **copy**. C#-scalar dtype note: leaves keep their C# type (`10` → int32), unlike Python ints (int64). `Creation/np.block.cs`.
- `concatenate` layout vote fixed to NumPy's ambiguity resolution: **F output only when all inputs are F-contiguous AND not all are also C-contiguous** — both-C&F inputs (e.g. `(N,1)` columns) now produce C order, matching `PyArray_CreateMultiSortedStridePerm`. Also added the uniform-tiny-slab interleave fast path (constant-size loads/stores when every per-outer slab is the same 1/2/4/8/16 bytes — the `(N,1)` column pattern; ~10-20× over per-slab `Buffer.MemoryCopy`).
- `UnmanagedStorage.GetData<T>()` now narrows to the shape's `[offset, offset+size)` window before returning/casting (zero-copy, still writes through) — previously a **contiguous view at a non-zero offset** (np.split / np.unstack children) returned the WHOLE backing buffer from `Data<T>()`.

`resize` ships as both `np.resize(a, new_shape)` (function — fills the enlarged output with **repeated copies** of `a` in C-order via an exact-sized doubling byte-tile; empty source / zero new-size → `zeros`; always C-contiguous; any input layout is raveled first) and `ndarray.resize(new_shape, refcheck=true)` (**in-place** — grows with **zeros**, shrinks by truncation, operates on the raw contiguous buffer so an F-contiguous resize relabels memory column-major). The method mirrors NumPy's guards verbatim (`IncorrectShapeException`): single-segment only, and when the byte size changes it must own its data (not a view) and — under `refcheck` — not be shared (`IArraySlice.IsUniquelyReferenced`, backed by the ARC block refcount); `refcheck:false` bypasses. Same-size resize is a pure in-place reshape (no ownership/reference check). See `Manipulation/np.resize.cs`, `Manipulation/NDArray.resize.cs`.

### Allocation & reshape guards (error parity)

The `size × itemsize` overflow class and the reshape rejection family, both ported from NumPy's own
C rather than hand-rolled. Probed against 2.4.2; gates: `Backends/AllocationGuardTests.cs` (13),
`Manipulation/ReshapeErrorParityTests.cs` (19), `Manipulation/ManipulationErrorParityTests.cs` (11).

- **`AllocationGuard`** (`Exceptions/AllocationGuard.cs`) is the dimension-check block of
  `PyArray_NewFromDescr_int`. **A size computed by multiplication cannot be trusted until its
  overflow is checked, and the failure is silent by construction** — a wrapped byte count looks
  like a small allocation, not an error. `np.zeros((2^31, 2^31))` of float64 used to *succeed*:
  element count 2^62 is representable but 2^62×8 = 2^65 wraps to **exactly 0**, so the allocator
  was asked for zero bytes and obliged — `[0,0]` read 0 and the last element read out of bounds;
  `np.ones` on the same shape wrote out of bounds. Two behaviours here are load-bearing and easy
  to get wrong: the running product starts at **itemSize** (the quantity checked is BYTES — which
  is why `np.zeros(2^60)` f8 is rejected while the same shape of int8 is merely `OutOfMemoryException`
  ≡ NumPy's `MemoryError`), and **a zero dimension does NOT short-circuit the scan** (NumPy keeps
  multiplying "as if" it were 1, so `np.zeros((0, 2^62))` f8 still reports "array is too big").
  Dims are scanned left to right so the error follows NumPy's order: `(-1, 2^62)` is
  `negative dimensions are not allowed`, `(2^62, -1)` is `array is too big; ...`.
  There are **two distinct wrap points** and a guard that sees only one misses the other:
  `(2^31, 2^31)` wraps the BYTE count, `(2^62, 2^62)` wraps the ELEMENT COUNT itself to 0 — which
  is why the guard keys off DIMENSIONS, never `shape.size`. `np.full` / `np.full_like` / `np.ones`
  allocate from `shape.size` directly instead of via `UnmanagedStorage.Allocate`, so they carry the
  call explicitly; `UnmanagedMemoryBlock<T>`'s two allocating ctors hold a byte-count backstop for
  any path that reaches an allocator without passing a `Shape`.
- **`ndarray.resize` is NOT the same loop** — `PyArray_Resize_int` differs from the creation guard
  on all three axes, all observable: it **breaks** at the first zero dim (so `a.resize((0,-1))` is
  legal and yields shape `(0,-1)`, while `np.zeros((0,-1))` raises), its text omits the "are"
  (`negative dimensions not allowed`), and an overflowing product is a **MemoryError**, not a
  ValueError. It therefore allocates by element COUNT and relabels — as NumPy does, mallocing
  `newnbytes` and only then writing the dimensions — instead of routing the requested dims through
  the creation guard.
- **`Shape.Reshape`** is now `_fix_unknown_dimension` + `convert_shape_to_string`. **ANY negative
  dim is the unknown one**, not just `-1` (probed: `np.zeros(6).reshape(-3)` is `(6,)`,
  `reshape(3,-5)` is `(3,2)`) — matching only `-1` let a SECOND negative ride through as a literal
  dimension, so `reshape(-1,-2)` silently produced the shape `(-3,-2)`, a negative-extent array.
  The zero product is tested **before** the division (`s_known == 0 || size % s_known != 0`), which
  is why `np.zeros((0,3)).reshape(-1,0)` no longer raises `DivideByZeroException`. Every rejection
  except the multiple-unknown one carries NumPy's single verbatim text,
  `cannot reshape array of size {n} into shape {s}`, where `{s}` follows `convert_shape_to_string`'s
  three quirks: LEADING unknowns are dropped (`reshape(-1,0)` → `(0)`), later ones read `newaxis`
  (`reshape(0,-1)` → `(0,newaxis)`), and a one-element shape closes with `,)` (`reshape(0)` → `(0,)`)
  — so `(0)` and `(0,)` both occur and mean different things. Type stays the long-standing
  `IncorrectShapeException` (house convention); the multiple-unknown case is `ValueError`
  ("can only specify one unknown dimension"), previously a bare `ArgumentException`.
- **`np.expand_dims(a, int)`** validates the axis against the OUTPUT ndim (`a.ndim + 1`) and reports
  the axis **as given**. `Shape.ExpandDimension` guards only the negative side and does it against
  the *normalized* value, so a positive overshoot fell through to `Arrays.Insert` and surfaced as
  `ArgumentOutOfRangeException ... (Parameter 'index')` — a parameter `expand_dims` does not have —
  a negative overshoot reported the normalized axis, and a **0-d input skipped the check entirely**
  (`expand_dims(scalar, 5)` silently returned shape `(1)`). The guard lives at the `np.*` boundary,
  not in `Shape.ExpandDimension`, because that method has internal callers (keepdims reductions,
  fancy-index shaping, `asmatrix`) not bound by NumPy's axis contract.

**Matched, do not re-prove:** the manipulation family's *values* are clean across the degenerate
inputs (`split`/`array_split`/`hsplit`/`vsplit` × {0, negative, empty}, `repeat`/`tile`/`roll` ×
{0, negative, empty axis}, `squeeze`, `moveaxis`, `swapaxes`). **Known remaining divergences**, all
typed-and-texted on both sides (text only, no raw framework leak): `np.split(a, 0)` raises
"number sections must be larger than 0." where NumPy leaks `ZeroDivisionError: integer modulo by
zero` (NumSharp is stricter, deliberately); `split(..., axis=oob)` is `ArgumentOutOfRangeException`
vs NumPy's leaked `IndexError`; `squeeze`, `repeat(-1)`, `tile(-1)` and the `AxisOutOfRangeException`
texts differ in wording; `np.broadcast_to(a, (2^62, 6))` builds the view where NumPy refuses with
`iterator is too large`.

### Broadcasting
`are_broadcastable`, `broadcast`, `broadcast_arrays`, `broadcast_to`

### Math — Arithmetic
`abs`, `absolute`, `add`, `arccos`, `arcsin`, `arctan`, `arctan2`, `cbrt`, `ceil`, `clip`, `convolve`, `cos`, `cosh`, `deg2rad`, `degrees`, `divide`, `exp`, `exp2`, `expm1`, `floor`, `floor_divide`, `log`, `log10`, `log1p`, `log2`, `mod`, `modf`, `multiply`, `negative`, `positive`, `power`, `rad2deg`, `radians`, `reciprocal`, `rint`, `sign`, `sin`, `sinh`, `sqrt`, `square`, `subtract`, `tan`, `tanh`, `true_divide`, `trunc`

**ufunc `out=` / `where=` parameters** are supported on the elementwise core (NumPy semantics, probed against 2.4.2): binary `add`/`subtract`/`multiply`/`divide`/`true_divide`/`mod`/`power`/`floor_divide`/`arctan2`/`bitwise_and`/`bitwise_or`/`bitwise_xor`, unary `sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/`abs`/`absolute`/`negative`/`square`/`log2`/`log10`/`log1p`/`exp2`/`expm1`/`cbrt`/`sign`/`floor`/`ceil`/`trunc`/`reciprocal`/`sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/`arctan`/`deg2rad`(`radians`)/`rad2deg`(`degrees`)/`invert`(`bitwise_not`)/`rint`. `round_`/`around` take `out=` ONLY (np.round is a function, not a ufunc — no where/dtype; decimals≠0 cast errors name ufunc 'multiply' per NumPy's composition). `rint` is the TRUE ufunc form of round-half-to-even: unlike `round_`/`around` (which preserve integer dtype) it is float-tier (bool/i8/u8→f16, i16/u16→f32, i32+→f64, floats/complex preserved) and reuses `UnaryOp.Round`'s kernel (complex rounds real+imag; `dtype=<int>`→no-loop). floor/ceil/trunc have IDENTITY loops on every bool/int dtype (dtype preserved; np.round's int path is an identity copy); the loop dtype comes from the input tier (`sinh(i1, out=f8)` stores float16-precision values); reciprocal int 1/0 → signed MinValue (NumPy 2.4.2); sign/positive reject bool with the verbatim no-loop UFuncTypeError; bitwise/invert raise the no-loop TypeError for float inputs (probed order: bad where → no-loop → out-cast → shape). `out` joins the broadcast but is never stretched, requires a same_kind cast from the loop dtype (resolved from inputs), returns the same instance, and may alias an input (overlap-safe via COPY_IF_OVERLAP). `where` must be bool, broadcasts and joins the output shape; masked-off `out` slots keep prior contents. Engine plumbing: `Backends/Default/Math/DefaultEngine.UfuncOut.cs`.

**Each of those ufuncs exposes ONE NumPy-shaped overload** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` mirroring NumPy's `f(x1[, x2], /, out=None, *, where=True, dtype=None)`: `out` is the second/third positional slot exactly like NumPy, `where`/`dtype` are reachable by name without `out`. `dtype=` selects the LOOP (NumPy loop-signature semantics, probed): computation runs at that precision even with `out=` (`sqrt([2.], out=f64, dtype=f32)` stores the f32-rounded value; `add(0.1, 0.2, out=f64, dtype=f32)` stores `0.30000001…`), `power(2, -1, dtype: f64) = 0.5` (the negative-int-exponent ValueError applies only to integer loops), `power(10, 11, dtype: f64) = 1e11` exactly (no compute-then-cast), `add(True, True, dtype: i32) = 2` (the bool→logical-OR remap keys off the FINAL loop dtype), `negative(bool, dtype: f64)` is legal, and inputs must reach the loop via same_kind casts (verbatim `Cannot cast ufunc '<name>' input [N] from ...` errors; binary errors are indexed, unary are not). Loop-existence gates raise `No loop matching ... ufunc <name>`: float-only ufuncs (sqrt/exp/log/trig + `divide`/`true_divide`) reject int/bool dtype; bitwise rejects float/complex/decimal dtype. `positive` is a full ufunc (identity loops at every dtype EXCEPT bool: plain `positive(bool)` and `dtype: bool` raise the verbatim `did not contain a loop with signature matching types <class 'numpy.dtypes.XDType'> -> ...` texts; `positive(bool, dtype: f64)` works). `round_`/`around` follow NumPy's non-ufunc shape `round(a, int decimals = 0, NDArray out = null)` (2nd positional is decimals, NOT out). Positional-dtype overloads (`np.sqrt(x, NPTypeCode.Single)`, `(x, Type)`) also exist for source compat as non-NumPy call forms (NumPy's 2nd positional is `out`). Tests: `Math/UfuncDtypeOverloadTests.cs`.

**Five float32 unary loops are BIT-EXACT with NumPy**, not merely close — `exp`, `log`, `sin`, `cos`, `tanh` —
because `Utilities/NDFloatMath.cs` ports the kernels NumPy 2.4.2 actually runs rather than calling the platform
libm: `simd_exp_FLOAT` and `simd_log_FLOAT` (`numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`, the
`SIMD_AVX2_FMA3` instantiation — on Windows the AVX-512 variant is `!defined(_MSC_VER)`-gated and never compiled)
and `simd_sincos_f32` (`loops_trigonometric.dispatch.cpp`, `NPY_SIMD_FMA3`). `rad2deg`/`deg2rad` join them by
forming their constant at the OPERAND's precision, as NumPy's `RAD2DEG`/`DEG2RAD` macros do. None of these kernels
is correctly rounded — NumPy documents 2.52 ULP for exp, 3.83 for log, 0.647 for the sine polynomial — so
`MathF.Exp`/`Log`/`Sin`/`Cos` (~correctly rounded) differed from NumPy on **35% / 4.7% / 6.9% / 7.8%** of float32
inputs, and the one-ULP-low `rad2deg` factor on **79%**; the blanket `unary ~ULP` fuzz excuse was absorbing all of
it silently. Each is now verified **exhaustively: all 2³² float32 bit patterns**, through both the SIMD kernel and
the scalar entry point (chunked-checksum sweep). The corpus tier `numpy_f32_kernels.jsonl` (140 cases) pins the
discriminating inputs in CI and `MisalignedRegistry`'s unary-ULP branch EXCLUDES these ops at a float32 result, so
a 1-ULP regression turns the gate red.

**`tanh` is the only one of the five that is also a FLOAT64 port**, and the only one that needed to be: NumPy ships
its own table-driven kernel at BOTH widths (`simd_tanh_f32`/`simd_tanh_f64` in
`numpy/_core/src/umath/loops_hyperbolic.dispatch.cpp.src`, converted by NumPy from Intel SVML), where exp/log/sin/cos
fall through to the platform's scalar `npy_*` at f8 and already agreed. So `MathF.Tanh`/`Math.Tanh` differed from
NumPy on **9.7% / 8.1%** of inputs (≤2 ULP / 1 ULP). tanh is odd, so the kernel works on `|x|` and ORs the sign back:
`[0, SATURATION_THRESHOLD)` splits into **32 subintervals at f32 / 16 at f64**, the exponent field of `|x|` indexes a
table of per-subinterval minimax coefficients plus a recentring offset `b`, and the answer is a degree-**6 / 16**
Horner evaluation of `P(|x| - b)`. The top subintervals hold `1 + 0·y + 0·y² …`, which is how the saturated range
returns exactly ±1 through the same arithmetic. **Unlike exp/log/sincos the FMA here is not a host pin** — that kernel
spells its Horner steps as `hn::MulAdd`, an *explicit* fused multiply-add, so `FusedMultiplyAdd` is a literal
transcription rather than a bet on MSVC's contraction. Verified over **all 2³² float32 inputs** (SIMD + scalar, and a
negative control confirms the checksums detect the pre-port kernel in 34 of 512 chunks) and, at float64, over
**4.83 billion** values — 2³² covering every sign × exponent × 2²⁰ mantissa prefix, plus 5.4×10⁸ full-width
splitmix64 patterns. Gate: `numpy_f32_kernels.jsonl` + the new **`numpy_f64_kernels.jsonl`** (24 cases), with tanh
carved out of the unary-ULP excuse at BOTH widths (`NumPyPortedFloat32Kernels` / `NumPyPortedFloat64Kernels`).

**One deliberate departure from NumPy's own AVX2 shape, and it is what makes tanh fast:** the coefficient fetch is a
**transpose, not a gather**. A subinterval's 8 f32 coefficients are 8 *consecutive* floats — exactly one `Vector256` —
so the whole table read is 8 contiguous row loads plus an 8×8 transpose, where a gather-per-coefficient issues 8
`vgatherdps` (64 element loads for the same 8 rows). Both were implemented and measured, bit-identical against the
same table: transpose **0.071 ms** vs gather 0.124 ms at 100K and **7.41** vs 12.87 ms at 10M — 1.7×, the difference
between losing to NumPy and beating it. float64 stays **scalar on purpose** (its lookup would cost 18 gathers per 4
lanes, and the scalar kernel already outruns NumPy); `Vector512` is composed from two verified `Vector256` halves
rather than an AVX-512 gather that cannot be tested on this host.

**Perf (NPY/NS, Release, best-of-21 warm, `out=`, 100K / 10M):** `tanh` **1.78× / 1.82×** at float32 and
**1.90× / 1.92×** at float64 — up from 0.18×/0.19× and 0.70×/0.68×, i.e. **9.9×/9.7×** and **2.7×/2.8×** faster than
the `MathF.Tanh`/`Math.Tanh` calls it replaces.

Specials (probed, and mostly consequences of NumPy blanking NaN lanes before its range compares): any NaN — quiet,
signalling, either sign, any payload — returns the canonical `0x7fc00000` (note .NET's own `float.NaN` is
`0xffc00000`); `exp` saturates at `x ≥ 88.72283935546875` → `+inf` and `x ≤ -103.97208404541015625` → `+0`, with
the band between producing subnormals through `fma_scalef_ps`'s exponent split; `log` returns `-inf` at ±0 and a
**negative** NaN for a negative argument (but a positive one for a NaN argument); `sin`/`cos` hand arguments past
their Cody-Waite limits (**different per function** — 117435.992 and 71476.0625) to the platform libm exactly as
NumPy does, so those inputs are libm-dependent by construction. NumPy also raises FP status flags
(`RuntimeWarning`) that NumSharp does not model — pre-existing, not introduced here.

**Host pin:** each kernel's quadrant is computed with a FUSED multiply-add because MSVC 19.44 (the numpy==2.4.2
win-amd64 wheel's compiler) contracts NumPy's separate multiply and add. It is observable at `x = 0xc26d0e6c`,
where `x·log2 e` lands on the exact tie `-85.5` and the two spellings round `k` to -85 vs -86 — a 1-ULP
difference. Same pin class as the MSVC-pinned cast kernels (`Fuzz/README.md` → "Host-dependent values").

**Perf (NPY/NS, Release, best-of-21 warm, `out=`, 100K / 10M):** `exp` **1.11× / 1.11×**, `log` **1.28× / 1.27×**,
`sin` **1.07× / 1.04×**, `cos` **1.05× / 1.06×**, `rad2deg` **11.7× / 2.42×** — NumSharp is now FASTER than NumPy
on every one, having been 0.33×/0.41×/0.20×/0.21× before. The ports are 1.4–5.2× faster than the `MathF.*` calls
they replace because they vectorize: `Utilities/NDFloatMath.Simd.cs` carries `Vector{128,256,512}<float>`
overloads (each algorithm is purely elementwise, so lane `i` is bit-identical to the scalar call — which is why
NumPy generates its own kernels from one template for `__m256` and `__m512`), gated by
`DirectILKernelGenerator.NumPyFloatKernelSimdAvailable` on hardware FMA exactly as NumPy gates its own; without
FMA the scalar entry point runs, same bits, one lane at a time. Two measured details worth keeping: the
rarely-taken subnormal branches are kept as BRANCHES rather than folded into branchless selects (folding exp's
denormal split cost ~0.17 ns/element and log's ~0.15 — a fifth of each kernel), and the vector overloads carry
`AggressiveOptimization` because without it tier-0 JIT made the first ~40 calls 27× slower than steady state.

**Still the platform libm at float32/float64** (measured, documented, NOT ported): `exp2` (0.04% / 0.02% of
inputs differ, 1 ULP), and `expm1`/`log1p` (31.1% / 30.7% and 33.6% / 15.1%), which NumSharp additionally computes as
`Exp(x)-1` / `Log(1+x)` — catastrophic for small |x| (`expm1(1e-8)` returns **0** where NumPy returns 1e-8) and a
signed-zero bug (`expm1(-0.0)` → `+0.0`). **None of these three is portable** — NumPy calls the CRT for all of them,
so bit-parity is not reachable the way it was for exp/log/sin/cos/tanh; the expm1/log1p *accuracy* bug is a separate,
genuine defect worth fixing on its own terms (note .NET's `float.ExpM1`/`double.ExpM1`/`LogP1` are themselves just
`Exp(x)-1`/`Log(1+x)` and do NOT help). float16 loops (17 differing values out of all 65,536) and float64
`exp`/`log`/`sin`/`cos` already agree with NumPy bit-for-bit on this platform and need no port.
### Math — Reductions
`all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `average`, `average_returned`, `count_nonzero`, `cumprod`, `cumsum`, `diff`, `ediff1d`, `max`, `mean`, `median`, `min`, `percentile`, `prod`, `ptp`, `quantile`, `std`, `sum`, `var`

### Math — NaN-Aware
`nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanpercentile`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar`

### Bitwise
`bitwise_and`, `bitwise_or`, `bitwise_xor` (ufunc `out=`/`where=`/`dtype=` supported; float/complex/decimal INPUTS raise NumPy's coercion TypeError while a float/complex/decimal `dtype=` raises the no-loop text — distinct messages, both probed; probed order: bad `where` → no-loop → out-cast → shape), `invert`, `left_shift`, `right_shift`

### Comparison & Logic
`all`, `allclose`, `any`, `array_equal`, `equal`, `fmax`, `fmin`, `greater`, `greater_equal`, `isclose`, `iscomplex`, `iscomplexobj`, `isfinite`, `isin`, `isinf`, `isnan`, `isreal`, `isrealobj`, `isscalar`, `iterable`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `maximum`, `minimum`, `not_equal`

`np.isin(element, test_elements, assume_unique=false, invert=false, kind=null)` is the element-wise membership
test — a bool array of `element`'s shape, True where `element[i]` is a value in the (flattened) `test_elements`
set (NumPy `_arraysetops_impl.py`). Comparison happens in `result_type(element, test)`, matching NumPy's
brute/sort/table methods (which all agree on the RESULT — `kind` only trades speed for memory). The engine picks
its own method by the test set's value range, always returning the correct membership: an integer/bool pair with a
range ≤ `6·(|element|+|test|)` takes a **counting-table gather** (build a bool lookup over `[min,max]`, then one
`take(mode='clip')` per element with a range mask — no widening of the element array, staying in its own dtype),
otherwise a **sort + `searchsorted` + equality probe** (which also covers float/complex/decimal/char — complex
sorts lexicographically like NumPy). NaN is never a member (NaN ≠ NaN falls out of the equality probe). The one
exact-integer trap is handled explicitly: **uint64 paired with a signed integer** promotes to float64 under NEP50,
which would call two distinct integers equal past 2^53, so that pair is routed through the uint64 domain with a
sign filter (`isin(uint64[2^63+1], int64[2^63-1])` is correctly `False`, matching NumPy's table method — pinned).
`kind` is validated verbatim: `Invalid kind: '…'. Please use None, 'sort' or 'table'.` (ValueError), and for
`kind='table'` a non-integer input raises `The 'table' method is only supported for boolean or integer arrays…`
(ValueError) and a signed range exceeding the dtype max raises NumPy's RuntimeError text (as
`InvalidOperationException`). Non-contiguous `element` (transposed/strided/negative-stride/broadcast) is read in
logical C-order and the bool result is reshaped to a C-contiguous copy of `element`'s dimensions. **Perf (NPY/NS,
Release, best-of-21, warm):** integer isin **1.9–3.0×**, float64 **2.4–3.4×**, large-range int **3.3×** — faster
than NumPy at every measured cell. **Two deliberate divergences** (`[Misaligned]`): with `assume_unique=true` on
NON-unique inputs NumPy documents "incorrect results" and its output depends on the internal method it picks;
NumSharp always returns the correct membership. And `kind='table'` with a large-but-range-safe integer range
computes via `searchsorted` (no table allocation), so it does not raise `MemoryError` where NumPy's literal table
allocation would. See `Manipulation/np.isin.cs`; gate: `Manipulation/np.isin.Test.cs` (37) + the `isin` cases in the
`groupa` differential-fuzz tier (22, across int32/float64/uint8/complex128 × invert/assume_unique/kind/layouts).

`np.iterable(y)` is NumPy's pure iterability predicate — `try: iter(y); return True; except TypeError: return False` (`numpy/lib/_function_base_impl.py`). It is NOT an iteration op: it never touches element data, so it uses **no kernel/NDIter/loop** (O(1) rank/type check in `Logic/np.is.cs`, mirroring `isscalar`). C# mapping, each matching NumPy 2.4.2's `iter()` outcome: `null`→false; **`NDArray`→`ndim != 0`** (the one surprise NumPy documents — a 0-d array is the *only* non-iterable array: `iter()` on it raises `TypeError`, exactly as `NDArray.GetEnumerator()` does, while every rank≥1 array incl. empties is iterable); `string`→true (Python strings are iterable); any `IEnumerable` (C# arrays/lists/dicts/sets)→true; every scalar value type (int/double/bool/char/Half/decimal/Complex/…)→false. The `NDArray` case precedes the `IEnumerable` catch because `NDArray` implements `IEnumerable`. Being a bool predicate over an arbitrary object rather than a value-producing kernel over array data, it is unit-test-only (`Logic/np.iterable.Test.cs`, 33 cases) and deliberately absent from the differential-fuzz corpus — same rationale as `isscalar`/`nditer`/`ndindex`. **Deliberate C# divergences** (adversarially probed, pinned `[Misaligned]`): NumSharp maps Python `iter()`-ability onto C# `foreach`-ability (`IEnumerable`), so four inputs iterable in Python but NOT `foreach`-able in C# return false where NumPy returns true — a bare `IEnumerator`/`IEnumerator<T>` cursor (no `GetEnumerator`) and `ValueTuple`/`Tuple`/`ITuple` (no `IEnumerable`). Real ported code passes the collection (array/`List`/`NDArray`), which is foreach-able and matches. Construction traps are clean: NumSharp's `np.array(5)`, `NDArray x = 5`, and `a[0,0]` all build genuine 0-d arrays (→false), and a bare scalar literal boxes to its own type rather than firing an `int→NDArray` implicit conversion into the `object` overload.

The six comparisons and `isnan`/`isfinite`/`isinf` expose **ONE NumPy-shaped overload each** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` (no bare/out split). It returns plain `NDArray` — NumPy's `np.less(a, b, out=f64)` returns the f64 out itself; `True→1` at any numeric out dtype since bool casts same_kind to all of them. A plain call still returns an `NDArray<bool>` *instance* (TensorEngine contract), so the typed wrapper is one zero-alloc cast away and the C# comparison operators (`==`, `<`, …) keep the `NDArray<bool>` static type via `AsGeneric<bool>()`. `dtype=` is validate-only (probed 2.4.2): bool loops only — `dtype: Boolean` is a no-op, anything else raises `No loop matching the specified signature and casting was found for ufunc <name>`. Comparisons compare at `result_type(lhs, rhs)` inside the kernel (probed: `greater(i8 2^53+1, f8 2^53)` → False, `equal` → True). Engine members follow the house order `(inputs, typeCode, out, where)`.

### Type Promotion & Dtype
`can_cast`, `common_type`, `find_common_type`, `finfo`, `iinfo`, `issubdtype`, `min_scalar_type`, `mintypecode`, `promote_types`, `result_type`

### Selection
`compress`, `extract`, `index_exp`, `indices`, `ix_`, `place`, `put`, `ravel_multi_index`, `s_`, `select`, `take`, `take_along_axis`, `unravel_index`, `where`

`np.take_along_axis(arr, indices, axis=-1)` (NumPy `numpy/lib/_shape_base_impl.py`) is the per-slice
gather: it matches 1-D index and data slices oriented along `axis` and looks each output element up
in `arr` — the inverse operation to `argsort`/`argmax(keepdims=True)` (`take_along_axis(a, a.argsort(axis),
axis)` ≡ `sort(a, axis)`). NumPy implements it as advanced indexing (`arr[_make_along_axis_idx(...)]`);
NumSharp reproduces the identical semantics as ONE whole-array strided-odometer IL gather kernel
(`DirectILKernelGenerator.TakeAlongAxis.cs`, an outer odometer wrapped around a tight inner loop over
the innermost axis so the multi-level carry runs per-slice, not per-element), dtype-agnostic via a
byte-width-keyed (1/2/4/8/16) element copy. `indices` must be an INTEGER array (NumPy's
`issubdtype(dtype, integer)` — bool/float/complex raise `IndexError "\`indices\` must be an integer
array"`) whose ndim equals `arr`'s (else `ValueError "\`indices\` and \`arr\` must have the same number
of dimensions"`); the axis dimension of `indices` supplies `J` (need NOT equal `M = arr.shape[axis]`)
while every OTHER dimension broadcasts BOTH `arr` and `indices` (a stride-0 broadcast source view is
read fine; a non-axis conflict raises NumPy's fancy-index `IndexError "shape mismatch: indexing arrays
could not be broadcast together with shapes …"` listing each `arange`-grid shape). `axis=None` flattens
`arr` in C (logical) order and requires 1-D `indices` (else `ValueError "when axis=None, \`indices\`
must have a single dimension."`); a negative index normalizes ONCE (`idx += M`, advanced-indexing
semantics) and anything still out of range raises `IndexError "index {orig} is out of bounds for axis
{axis} with size {M}"` reporting the ORIGINAL value. The result is a fresh writeable C-contiguous copy
of `arr`'s dtype (any source layout — C/F/strided/reversed/sliced/transposed/broadcast — is read through
its own strides, no materialisation). **≥1.5× faster than NumPy on every measured variation** (geomean
~3.5–3.7× NPY/NS; the tightest cell is a random 16M axis-0 gather at ~1.55×, latency-bound). Differentially
fuzzed **24,000 value + 18,000 adversarial** cases bit-exact vs NumPy 2.4.2 (ranks 1–6, all axes, wrap/OOB,
all element widths, non-contiguous SOURCES × non-contiguous INDICES: F/reversed/strided/swapped/broadcast/
offset) — values, shapes, dtypes and error taxonomy 100% bit-exact.

**One deliberate divergence (documented, `[Misaligned]`):** when a *non-contiguous* `indices` array holds
*multiple* out-of-range values, NumSharp reports the first in the RESULT's C-order while NumPy reports
whichever its fancy-index `NpyIter` (`PyArray_MapIterCheckIndices`) visits first — index-memory/axis order
with negative-stride flipping, an internal artifact that is neither pure min-address (fails on 1-D negative
stride) nor pure logical C-order (fails on F-layout). Only the offending index VALUE differs; the error
TYPE, axis and size always match, and it never arises from `argsort`/`argmax` output (C-contiguous → exact).

**No SIMD gather (measured):** a hardware `VPGATHERQD/QQ` gather was benchmarked against the scalar loop
for the hot 4/8-byte contiguous case at **only ~1.16×** on this host (AVX2; no AVX-512), before the SIMD
bounds-validation + scalar OOB-fallback a raise-mode gather still needs — not worth a per-width gather
kernel or its CPU-dependence. The wins are structural: the outer-odometer/inner-loop split (per-slice
carry) and a branch-light unsigned-bounds resolve.

Gates: `Indexing/TakeAlongAxisTests.cs` (37) + 44 `take_along_axis` cases in the `groupa` differential-fuzz
tier. See `Indexing/np.take_along_axis.cs`.

`np.select(condlist, choicelist, default=0)` (NumPy `numpy/lib/_function_base_impl.py`) draws each
output element from the choice whose condition is true, FIRST matching condition winning; positions
where every condition is false take `default`. C# shapes: `condlist` is `NDArray[]` (boolean arrays),
`choicelist` is `object[]` (an `NDArray[]` binds via array covariance; scalar choices need
`new object[]{…}`), `default` is `object` (`null` ≙ NumPy's `0`). **NEP50 weak scalars, select's own
rule:** only `int`/`float`/`double`/`Complex` literals are weak (adopt the other operands' dtype) —
`bool`, `char`, `Half`, `decimal`, arrays and every `NDArray` are strong (NumPy's
`type(x) in (int,float,complex) else asarray(x)`), so a weak-int default that overflows the resolved
dtype WRAPS (`int8` choice + `1000` default → int8, 1000→−24) rather than raising; the result dtype
is `result_type(*choices, default)` folded via `NDExprTypeRules.PromoteStrong`. Conditions and choices
broadcast in two SEPARATE groups; a non-bool condition is `TypeError("invalid entry {i} in condlist:
should be boolean ndarray")`, length mismatch / empty condlist are `ValueError` (verbatim). Two code
paths, both NumPy-exact in VALUE, split by layout: the **contiguous / no-cast / full-size-array-choice**
case (the one NumPy leaves memory-bound) runs a **fused single-pass IL kernel**
(`DirectILKernelGenerator.Select.cs`) — a reverse `Vector.ConditionalSelect` chain (seed = default;
overlay choices n−1→0 so the first true condition wins), 4×-unrolled with four independent
accumulators, the scalar default hoisted once via `CreateBroadcast`, reusing `np.where`'s exact
`EmitInlineMaskCreation` bool→lane expansion — so each condition and choice is read once and the result
is written once. Every OTHER shape (broadcast / strided / cast / **scalar-choice**) declines the fused
gate (`TrySelectFused`) and takes NumPy's OWN structure — allocate the default-filled result, then
`np.copyto(where=cond)` each choice in REVERSE — where each masked write rides NumSharp's SIMD
masked-cast kernel. The composition alone was already 1.2–8.5× NumPy everywhere EXCEPT the 8-byte
(int64/float64) all-array-choice large cases, which sat at 1.22–1.54× because its `(n+1)` masked
copytos each re-read AND re-write the whole 80 MB result; the fused kernel cuts that redundant result
traffic and lifts those cells to **1.6–2.4×** (min 1.60× at int64 n=1 10M with an ARRAY default — the
irreducible 4-array-traffic memory-bound floor). **Perf (NPY/NS, Release, best-of-rounds):** array
choices **1K 3.3–5.0× · 100K 8–18× · 10M 1.6–5.8×**; scalar choices (composition) **3.3–3.8×** — no
measured cell below 1.5×. See `Indexing/np.select.cs`; gates: `Indexing/np.select.Test.cs` (21 — incl.
the fused SIMD-body/unroll/remainder/offset-slice pins) + the `select` op in the `groupa`
differential-fuzz tier (11 cases, dtype × precedence × broadcast × transposed).

**`np.take` / `np.put` negative indices under `mode='raise'`** now match NumPy's
`check_and_adjust_index`: a negative index is normalized once (`idx += n`) before the bounds test, so
`take(a, [-1])` / `put(a, -1, v)` address the last element and an index still out of range after the
shift raises (the original value survives for the diagnostic). The IL kernels
(`DirectILKernelGenerator.{Take,Put}.cs`) previously rejected EVERY negative index in raise mode — a
real bug, pinned now by `SelectionTests` (`Take/Put_Raise_NegativeIndices_Normalize`) and negative
`take`/`put` + `wrap`/`clip` cases in the `groupa` fuzz tier.

**`take`/`put`/`place` kernel performance (three levers, all in `DirectILKernelGenerator.{Take,Put,Place}.cs`).**
The gather/scatter/masked-copy kernels moved off the **per-element `cpblk`-of-a-runtime-tiny-size**
pathology (a memcpy call where a `mov` belongs — measured ~2× slower; it made `take` *lose* to NumPy at
0.68×). The per-element copy is now specialized to the dtype width via `CopyKindFor`/`EmitElementCopy` —
a typed `Ldind`/`Stind` MOV (two 8-byte MOVs for 16-byte Complex/Decimal), keyed into the kernel cache;
`cpblk` survives only for multi-element `take` axis slabs. Two more levers on top: (1) **software
prefetch** in the `take` gather (`Sse.Prefetch0` of the slab `PrefetchDistance=32` indices ahead, gated
at emit-time on SSE **and** at the call site on the gathered footprint exceeding
`IndexPrefetchThresholdBytes = 2 MiB` — below that it is pure overhead, ~1.6× slower at 100K, so a
separate `(copyKind, prefetch)` kernel is cached and the small path stays lean); NumPy does not prefetch
in take, so this is a clean win at scale. (2) **A wrapping values cursor** in `put`/`place` replacing the
per-element **`i % valuesCount` integer division** (NumPy's iterator model) — the dominant cost for
cyclic values, and the reason `place` was stuck at ~1.0× (it idiv'd once per True). **Perf (NPY/NS,
Release, best-of-rounds):** `take` flat **100K 1.0–1.3× / 10M 1.7–2.5×** (prefetch), axis-slab
**4-byte 2.4–2.6× / 8-byte ~1.1×** (memcpy-bandwidth-bound); `put` **100K 2.0–3.1× / 10M 1.2–1.4×**
(random scatter, memory-latency-bound); `place` **100K 2.0–4.5× / 10M 4-byte 2.5× / 8-byte ~1.3×**
(bandwidth-bound). The sub-1.5× cells are all memory-latency / bandwidth / allocation-overhead bound —
the same walls NumPy hits — not algorithmic.

**Index dtype validation (correctness fix).** `take`/`put` used to `astype(int64)` any index array,
**silently truncating a float index** where NumPy raises. `CastIndicesToInt64` now rejects a
non-castable index dtype with NumPy's verbatim `TypeError("Cannot cast array data from dtype('…') to
dtype('int64') according to the rule '…'")` — and reproduces NumPy's SUBTLE rule split: `take` converts
under **`same_kind`** (so a `uint64` index is allowed) while `put` uses **`safe`** (which rejects
`uint64` too); each leaks its own rule name in the message. Integer/bool indices pass; float/complex are
refused. Pinned by `SelectionTests` (`Take_FloatIndices_Throws_SameKind`, `Put_FloatIndices_Throws_Safe`,
`Take_UInt64Indices_Allowed_But_Put_Rejects`) + the 16-byte-fidelity and cyclic-wrap tests.

**Subshaped fancy-SET into a NON-contiguous destination (correctness fix — silent corruption).** A
fancy assignment consuming FEWER axes than `ndim` (`arr[[0,2]] = v` on a 2-D+ array — each selected
index writes a whole trailing SUB-array) used to route unconditionally through `SetIndicesND<T>`, whose
block-`MemoryCopy` copies one CONTIGUOUS `subShape` per selected offset. That assumption holds ONLY for a
C-contiguous destination; into a **transposed / F-contig / row- or col-strided / negative-stride** view
the sub-arrays are strided in the buffer, so the copy landed each row at the wrong physical offsets —
silently corrupting the view in Release (`f = arr.reshape(3,4).T; f[[0,2]] = 99` set `f[2,0..2]`'s buffer
neighbours instead of `f`'s column-strided row 2). The dedicated `SetIndicesNDNonLinear<T>` — long a
`throw new NotImplementedException` stub behind a commented-out `//TODO:` dispatch — is now implemented
as the scatter mirror of the getter's `FetchIndicesNDNonLinear`: it walks each sub-array as an odometer
over the trailing `subShape` axes, advancing the destination offset **incrementally** through the view's
strides (section D "Incremental coord advance") from the C-contiguous (already broadcast to `retShape`)
value buffer, and the setter dispatches to it whenever `!source.Shape.IsContiguous`. The C-contiguous
block-copy fast path is untouched. Bit-exact with NumPy 2.4.2 across all 15 dtypes × {F-contig,
transposed 3-D, row/col-strided, negative-stride} × {scalar, sub-row broadcast, full grid} values
(264-case differential + Char/Decimal self-consistency). The scatter is memory-latency bound (a strided
write is a cache miss per element; NumPy is ~1.3× faster on the dense case and up to ~3× on the
cache-friendly col-strided one — 1.5× is not reachable on a memory-bound scatter, and the previously
"fast" path was returning WRONG data). Gate: the index oracle's new `order='K'` setter cases
(`gen_index_oracle` copies view bases with `order='K'` so the SET runs into a genuinely non-contiguous
destination — 11 appended `set:fancy*` cases over `AT`/`BT`/strided layouts, 7 of which diverge on the
pre-fix code) + `AuditV2` `T1_15a`/`T1_15b`. See `Selection/NDArray.Indexing.Selection.Setter.cs`.

### Grid / slice-expression DSL (`r_`, `c_`, `ix_`, `s_`, `index_exp`)

NumPy's `_index_tricks_impl.py` index-expression family. All probed against 2.4.2; tests in
`Creation/np.r_.Test.cs` + `Indexing/np.ix_.Test.cs`; oracle jobs in the `manip` tier
(`gen_oracle.gen_index_tricks` ↔ `OpRegistry` cases `r_`/`c_`/`ix_`, 2,262 cases).

**How a Python slice literal is spelled in C#.** C# has no `a[1:5:2]` syntax, so a slice is written
as a **string**: `np.r_[0:5]` → `np.r_["0:5"]`. Strings do double duty here exactly as they do in
NumPy, and are told apart by a colon — **a string containing `':'` is a slice expression, a string
without one is a NumPy special directive** (`"r"`, `"c"`, `"-1"`, `"0,2"`, `"1,2,0"`). The two
grammars are disjoint upstream too (a directive never contains a colon, because Python's own syntax
supplies the slices), so every NumPy expression transcribes verbatim. One string may hold several
comma-separated slices (`np.r_["1:3, 5:8"]` ≙ `np.r_[1:3, 5:8]`), and a `Slice` / `Slice[]` object
works as an entry too, so `np.s_[…]` composes into `np.r_[…]`.

- **`np.r_[…]` / `np.c_[…]`** — static properties returning `RClass`/`CClass`, both `AxisConcatenator`
  subclasses with a `this[params object[] key]` indexer (the port of NumPy's `__getitem__`). `r_`
  concatenates along axis 0; `c_` is `r_["-1,2,0", …]`, i.e. entries are upgraded to ≥2-D with the 1s
  POST-pended (a 1-D entry becomes a **column**) and stacked on the last axis. Slice entries become
  `arange(start, stop, step)`, or — for an **imaginary step** (`"−1:1:6j"`) — `linspace(start, stop, N)`
  with the stop **inclusive**. Integer-ness follows the LITERAL, not the value: `"0:5"` is int64 while
  `"0.0:5"` is float64. A **missing stop re-reads start as stop** (NumPy's `arange(start, None, step)`
  IS `arange(0, start, step)`), so `np.r_["2:"]` is `[0, 1]` and `np.r_["5::2"]` is `[0, 2, 4]`.
  Directives are reproduced whole: axis, `axis,ndmin`, `axis,ndmin,trans1d` (with NumPy's two DIFFERENT
  upgrade paths — the array branch's `defaxes` permutation vs the slice branch's `swapaxes(-1, trans1d)`),
  a silently-ignored 4th field, and `"r"`/`"c"` matrix coercion (routed through `np.asmatrix`, since
  NumSharp has no `matrix` subclass). Verbatim errors: `special directives must be the first entry.`,
  `unknown special directive` / `unknown special directive '0,q'` (the comma form quotes, the bare form
  does not). **`ndmin` is deliberately UNCAPPED** — NumPy validates it against `NPY_MAXDIMS` and raises
  `ndmin must be <= ndmax (64)`, but NumSharp has no 64-dimension ceiling anywhere, so importing one
  here would make the DSL refuse ranks the rest of the library accepts; `np.r_["0,100000", x]` builds
  the 100000-dim array. It has to be CHEAP rather than merely bounded, because `ndmin` comes from a
  user-typed directive: `np.AtLeastNdView` prepends every axis in ONE `Storage.Alias` over
  `Shape.PrependDimensions` (O(ndim), 1.4 ms at `ndmin=100000`) where the per-axis `expand_dims` loop
  it replaced cloned dims+strides each step — quadratic, 27.6 s at 100000 and unbounded at 2³¹-1. An
  `ndmin` too large to allocate a shape for (`"0,2147483647"`) surfaces as `OutOfMemoryException` in
  ~1 ms, the same answer any other oversized NumSharp request gives. A non-positive `ndmin` stays a
  no-op, as upstream. The closed form is exact, not approximate: a prepended axis takes stride
  `dimensions[0] * strides[0]` (0 for a 0-d source), so strides, cached flags and the writeable bit
  are bit-identical to the loop — verified across 16 layouts × 6 ranks, broadcast views included.
- **`np.ix_(params object[])` → `NDArray[]`** — open mesh. The dtype is **PRESERVED**, not forced to
  `intp`: `ix_` does no integer validation, so a float or int8 sequence rides through and only fails at
  the later indexing call; the one exception is NumPy's own, casting an **empty non-ndarray** input to
  int64 so it does not default to float64. A bool sequence is a mask (`nonzero`). Results are **views**
  that write through, built with `expand_dims` rather than `reshape` so a stride-0/read-only operand
  keeps its strides AND its non-writeable flag (reshape materializes those and would hand back a
  writeable copy, which NumPy does not do). `ValueError("Cross index must be 1 dimensional")` — which
  a **null** sequence also raises, reproducing the outcome NumPy reaches via `asarray(None)`'s 0-d
  object array rather than leaking a bare `ArgumentNullException` from the conversion helper.
- **`np.s_[…]` / `np.index_exp[…]`** — capture an index expression instead of applying it. The
  **indexer surface mirrors `NDArray`'s**, because NumPy's `s_` passes ANY index object straight
  through (`np.s_[[1,2]]` is the list, `np.s_[..., mask]` the tuple) — so the contract is
  **`arr[X]` ≡ `arr[np.s_[X]]` for every X**, verified across the whole vocabulary. Three overloads
  cover what `NDArray`'s three public indexers do, each returning the array type that indexer
  consumes: `this[string]` and `this[params Slice[]]` → **`Slice[]`** for BASIC expressions (slices,
  slice strings, integers — `int` and `string` convert implicitly to `Slice`, so `np.s_[0]` and
  `np.s_["1:3", "::2"]` stay here), and `this[params object[]]` → **`object[]`** the moment an entry
  cannot be a `Slice` — an `NDArray` fancy index or boolean mask, `int[]`/`long[]`/`bool[]`, an
  `IList` sequence, or a basic/advanced mix like `np.s_[Slice.All, np.array(new[]{0,2})]`. Both feed
  `arr[…]` directly, including an `NDArray<T>` (its own indexers do not hide the inherited
  `this[params object[]]`). **C# divergence:** `s_` and `index_exp` are identical here — NumPy's pair
  differs only in `maketuple` (`np.s_[2::2]` is a bare slice, `np.index_exp[2::2]` a 1-tuple), but a
  lone `Slice` and a one-element `Slice[]` are literally the same call in C#. Both are kept so NumPy
  code ports verbatim; `maketuple` is exposed but has no observable effect.

**NEP50 weak scalars.** NumPy distinguishes a Python literal (`5`, weak — adopts the other operand's
dtype) from a NumPy scalar (`np.int64(5)`, strong). C# has no such split, so the mapping is by type:
`bool`, the eight integer primitives, `float`, `double` and `Complex` are **weak** (the C# literal is
the Python literal); `char`, `Half` and `decimal` — which have no Python literal — plus every `NDArray`
(0-d included) are **strong**. `NDArray.Scalar(1L)` is therefore the escape hatch: `np.r_[i8, 1L]` is
int8, `np.r_[i8, NDArray.Scalar(1L)]` is int64. A weak integer that does not fit the adopted dtype
raises `OverflowException("Python integer 1000 out of bounds for int8")` rather than wrapping, reusing
`NDExprTypeRules.CheckIntLiteralFits`; a weak float saturates to ±inf instead (`np.r_[f4, 1e300]`).
An all-literal key whose integer does not fit int64 lifts the default to **uint64** (`np.r_[2**63]`
and `np.r_[2**64-1]` are both uint64 upstream; in C# only a `ulong` can carry the value).
NumSharp has a single complex width, so NumPy's `result_type(float32, 1j) → complex64` collapses onto
`Complex`.

**Edge sweep (post-ship).** Probed outside the corpus envelope along the magnitude / structural /
resource / null / acceptance / adversarial / call-form dimensions. Three fixes came out of it — the
O(1) `ndmin` expansion, the uint64 weak default and `ix_(null)`, all described above. Matched and worth not
re-proving: `-0.0` keeps its sign bit, NaN/±inf/subnormals round-trip, the weak-integer boundaries are
exact to ±1 (`i8+127` fine, `+128` raises), `9e18:9e18+2` collapses to empty as it does upstream,
huge slice expressions raise an honest `OutOfMemoryException` rather than wrapping a byte count, and
NumPy's error ORDER holds (a bad directive is reported before the shapes behind it). `np.r_`/`np.c_`
are **stateless singletons** — a directive is copied to locals per call, so it neither persists to the
next call nor leaks across threads (checked under `Parallel.For`). Deliberate divergences: a `null`
ENTRY throws (NumPy builds an object array, a dtype NumSharp lacks — same call as `fill_diagonal`),
`np.ix_("abc")` is accepted as a char sequence (NumSharp's house string→char-array conversion; NumPy
rejects it as 0-d), and a sub-1e-15 slice step reports `np.arange`'s inherited "step can't be 0".

**Deliberate divergence — no `bmat` branch.** NumPy routes a single bare string key into
`matrixlib.bmat`, which resolves the words in it against the CALLER'S Python frame
(`sys._getframe().f_back`) as variable names. C# cannot do that, and the branch raises `NameError` for
every string literal anyway (`np.r_['1 2; 3 4']` → "name '1' is not defined"), so a lone string gets
the same reading as any other entry — which is what makes `np.r_["0:5"]` work at all.

**Perf (NPY/NS, best-of-21, Release, 100K / 10M).** Three cost classes, reported apart because a
geomean across them would describe nothing: **O(N) concatenation** (`r_[a,b]`, `r_[a,0,0,b]`, `c_[a,b]`)
**1.61× / 1.19×**, **O(N) generation** (the arange and imaginary-step linspace branches) **2.63× / 1.16×**,
and the **O(1) view** `ix_` **1.29× / 3.80×**. No cell below 1.0. The 10M ratios compress toward parity
because both sides are memory-bandwidth bound there — `r_`'s own overhead above `np.concatenate` measures
0.94 ms on a 27 ms call (3.6%), and two extra weak-scalar entries add ~0.06 ms.

See `Creation/np.{r_,c_}.cs`, `Indexing/np.{ix_,s_}.cs`.

**`np.ogrid` — the open-mesh grid.** Port of NumPy 2.x `numpy.ogrid` (the `sparse=True` instance of
`nd_grid`). It is NOT an `AxisConcatenator` — it is its own class — but it borrows the SAME slice
grammar via `AxisConcatenator.ParseSliceToken`/`SliceSpec`, so slices are spelled as strings
(`np.ogrid["0:3", "0:5"]` ≙ `np.ogrid[0:3, 0:5]`; one string may carry comma-separated slices;
`Slice` objects work too) with NO directives. A **single** slice returns a bare 1-D array (arange, or
linspace for an imaginary `Nj` step — stop inclusive); **multiple** slices return an open mesh — N
arrays each shape 1 in every axis but its own, built by `arange`→(astype)→`reshape`-to-`(1,…,size,…,1)`.
The whole mesh shares ONE dtype (NumPy's single `result_type` over every slice bound): int64 iff every
field of every slice is an integer literal, else float64 — so `np.ogrid["0:2", "0.0:2"]` yields TWO
float64 arrays, and an imaginary step floats the whole mesh. A single slice may omit its stop
(`np.ogrid["5:"]` = `arange(0, 5)`); a **multi** slice may NOT (NumPy sizes each axis from `stop-start`
and leaks an `AttributeError` — NumSharp raises a clear `ValueError`), and an imaginary step always
needs a stop. **Return type** `OGridResult` stands in for NumPy's array-or-tuple polymorphism (C#
cannot return both from one indexer): it converts implicitly to a bare `NDArray` for a single slice,
to `NDArray[]` for a mesh, and `Deconstruct`s (`var (y, x) = np.ogrid["0:3", "0:5"];`) — collapsing
NumPy's bare-array-vs-1-tuple trailing-comma distinction. Imaginary/float grids are **bit-exact** with
NumPy (same `linspace`/`arange` code path as `r_`). Values/shapes/dtypes verified against NumPy 2.4.2;
**≥1.5× faster on every measured variation** (geomean ~4.5×; NPY/NS). Gate:
`Creation/np.ogrid.Test.cs` (18). See `Creation/np.ogrid.cs`.

**`np.mgrid` — the dense-mesh twin of `ogrid`.** Port of NumPy 2.x `numpy.mgrid` (the `sparse=False`
`nd_grid`). Shares `ogrid`'s parsing and grammar via the extracted `np.ParseGridSpecs` / `GridMeshDtype`
helpers (same string-slice spelling, same shared-dtype rule, same missing-stop/imaginary `ValueError`s
with `mgrid` wording). A **single** slice returns a bare 1-D array (identical to `ogrid` single);
**multiple** slices return ONE dense stacked array of shape `(N, size_0, …, size_{N-1})` — `result[k]`
is the k-th coordinate broadcast across the grid. Built NumPy's way — `np.indices(sizes, typ)` then
each **non-trivial** axis (start≠0 / step≠1 / imaginary) is rescaled by overwriting its layer with the
1-D line broadcast across the grid (`np.copyto`); a pure `0:n` axis needs no post-pass, so the common
`np.mgrid[0:a, 0:b]` is a single fused `indices` fill. This is why `mgrid` gets its OWN build rather than
reusing `ogrid`'s lines: **NumPy sizes each axis by `ceil((stop-start)/step)`, which may be negative** —
`mgrid[5:0, 0:3]` raises `"negative dimensions are not allowed"` (via `indices`) where `ogrid[5:0, 0:3]`
clamps to an empty axis, and a zero step is a divide-by-zero — a genuine NumPy divergence between the two
that an arange-line-based `mgrid` would have silently papered over. **Return type** `MGridResult` is
NumPy's single array: implicit to `NDArray` (the stack), plus `Deconstruct`
(`var (x, y) = np.mgrid["0:5", "0:3"];`) and indexing into the per-axis grids. **This replaced the legacy
non-NumPy `np.mgrid(NDArray, NDArray)` method** (and its `NDArray.mgrid(rhs)` twin) — a deliberate
breaking change, since a property named `mgrid` cannot coexist with a method of the same name. 13-case
differential matrix bit-exact vs NumPy 2.4.2; **≥1.5× faster on every measured variation** (geomean
~6×; NPY/NS). Gate: `Creation/np.mgrid.Test.cs` (15). See `Creation/np.mgrid.cs`, `Creation/np.ogrid.cs`
(shared helpers).

**`np.meshgrid` — coordinate matrices from coordinate vectors (completed to full NumPy).** Port of NumPy
2.x `numpy.meshgrid` (`numpy/lib/_function_base_impl.py`). Replaced the old **2-input `Kwargs` stub** (which
was 2-D only and whose `copy` flag was a no-op) with the full variadic API: `meshgrid(x1, x2[, x3], string
indexing="xy", bool sparse=false, bool copy=true)` plus a `meshgrid(NDArray[] xi, …)` core for 4+ inputs.
Each input is flattened to its own open-mesh axis; **`"xy"`** (Cartesian, default) swaps the first two axes
vs **`"ij"`** (matrix) — 2-D outputs are `(N, M)` under xy and `(M, N)` under ij; **`sparse`** keeps the
open-mesh `(1,…,Ni,…,1)` shapes; **`copy`** materializes contiguous grids (false → non-contiguous broadcast
views, NumPy-verbatim). Each grid **PRESERVES its input's dtype** (not promoted). Bad `indexing` →
`ValueError "Valid values for `indexing` are 'xy' and 'ij'."`. **Return type** `MeshgridResult` — NumPy's
tuple: implicit to `NDArray[]`, `Deconstruct` (`var (xx, yy) = np.meshgrid(x, y);`), indexer. The old
`Kwargs` class and the `(NDArray, NDArray)` return are gone; in-repo callers migrated (the `NewDtypes`
sweep, the fuzz `OpRegistry` meshgrid case). Differential-verified bit-exact vs NumPy 2.4.2 (shape + dtype
+ values) across xy/ij × sparse × 1/2/3/4-D × dtype-preservation × copy-view. **Perf (NPY/NS):** dense grids
at scale are **4.7×–5.2×** (1000², 3-D 50³) and `copy=false` views **2.3×** — NumSharp's bulk broadcast-copy
beats NumPy's there; SMALL dense/sparse grids are **0.5×–1.2×**, bound by NumSharp's per-`NDArray`
construction/`.copy()` floor (~1.6 µs), the same ceiling documented for `fill_diagonal` — not the meshgrid
algorithm (it cannot use `mgrid`'s fused `indices` kernel because its values are arbitrary input vectors, not
coordinates). Gate: `Creation/np.meshgrid.Test.cs` (12). See `Creation/np.meshgrid.cs`.

**`np.ix_` — audited, no change.** The open-mesh index builder (`Indexing/np.ix_.cs`, documented above) was
re-verified against NumPy 2.4.2 across 1–3 operands incl. bool masks, empty (→intp), float-preserve, single,
and the 2-D `ValueError` — all matching, and it is already the differential-fuzz `ix_` tier (green). The only
"difference" is C# `int` literals being int32 where Python's are int64, which is language-wide, not an `ix_`
behaviour.

### Iteration
`nditer`, `ndenumerate`, `ndindex`, `nested_iters`, `flatiter` (plus the pre-existing `broadcast`), and
the NumSharp-extension typed forms `nditer<T>` / `nditer_chunks<T>`

**`np.nested_iters(op, axes, flags=null, op_flags=null, op_dtypes=null, order='K', casting="safe",
buffersize=0)` → `NDIterator[]`** — the port of NumPy's `NpyIter_NestedIters` (`nditer_pywrap.c`).
Returns one `np.NDIterator` per entry in `axes` (outermost first), all iterating the SAME operand buffer
over disjoint axis subsets; advancing an outer iterator re-bases every inner one to its new position, so
`foreach (var _ in i) foreach (var _ in j) …` walks nested loops. Built on the EXISTING primitives —
each level is an ordinary `NDIterator` with `op_axes = [axes[level]]` sharing iter0's operands, linked
parent→child, and re-based via `NDIterRef.ResetBasePointers` (which already existed). The nested levels
switch from the default publish-then-advance `MoveNext` to **NumPy's advance-on-entry protocol**
(`npyiter_next`'s `started` flag) so the child re-base and the in-body `multi_index`/`value` stay aligned
(guarded by `_nestedMode`, so plain `nditer` is untouched — 662 nditer tests green). Validation is
verbatim NumPy (`ValueError`): `axes` ≥ 2 entries ("axes must have at least 2 entries for nested
iteration"), no axis reused ("An axis is used more than once"), negative axis ("An axis is out of
bounds"); an axis past the operand's ndim is rejected at construction as `IncorrectShapeException` (house
convention) where NumPy raises `ValueError`. **The `multi_index` path — the documented primary use — is
bit-exact** with NumPy 2.4.2 across single/multiple operands, any axis order, any depth (coords AND values
pair identically). **The bare value stream (no `multi_index`) now matches NumPy's memory order too:** the
op_axes contiguity check (`NDIter.CheckAllOperandsContiguous`) judges the ITERATOR's remapped layout, not
the full operand, so a non-contiguous axis SUBSET — e.g. inner axes `[0,2]` of a C-contiguous `(2,3,2)`,
strides `(6,1)` — no longer coalesces into F-order (`0,6,1,7,…` → NumPy's `0,1,6,7,…`); verified bit-exact
over all 2- and 3-level axes partitions × value-stream/`multi_index` (a plain non-contiguous nditer was
always correct — the bug was specific to op_axes subsets). Value-read perf is bound by the same per-element
`it[0]` NDArray-view cost documented below (~4.6× on a 1M walk); the iteration ENGINE is fast. See
`APIs/np.nested_iters.cs` + the nested support in `APIs/np.nditer.cs` (the contiguity fix is in
`Backends/Iterators/NDIter.cs`); gate: `APIs/np.nested_iters.Test.cs` (7).

**`NDArray.flatiter` / `np.flat(a)` → `np.FlatIterator`** — a WRITE-THROUGH, C-order flat iterator, the
analog of NumPy's `flatiter` (the type of `a.flat`). NumSharp's `NDArray.flat` returns a raveled `NDArray`
and is deeply embedded (~15 core call sites), so its return type is IMMOVABLE and NumPy's `a.flat → flatiter`
surface cannot be reclaimed; the iterator lives on the NEW `flatiter` accessor instead. It fixes a genuine
`flat` defect: `flat` COPIES for a non-contiguous layout (via `reshape`), so `a.T.flat[i] = v` is silently
lost — `FlatIterator` reads AND writes THROUGH to the base in logical C-order for every layout (transposed,
sliced, strided, negative-stride, broadcast), because every element maps a flat index through the base's
strides (`Shape.TransformOffset`, via `GetAtIndex`/`SetAtIndex`) with **no per-element allocation**. Surface
(probed against 2.4.2): `this[long]` scalar get/set (negative wraps; OOB → `IndexError` "index N is out of
bounds for size M"); fancy (`int[]`/`long[]`/`NDArray`) and slice-string (`"1:4"`, `"::2"`) get/set; the
`index`/`coords` cursor (past-end `coords` leaves the slowest axis UNwrapped, matching NumPy: index==size on
`(2,3)` → `(2,0)`); `Base`, `size`, `copy()` (a fresh 1-D C-order array); and C-order iteration that shares
the cursor (NumPy's `iter(f) is f`, so a second pass RESUMES). Cross-dtype scalar assignment casts with NumPy
semantics: to an integer dtype an in-range value truncates toward zero, but an OUT-OF-RANGE **weak** scalar (a
C# primitive — NumSharp's NEP50 analog of a Python int/float) RAISES `OverflowException` ("Python integer 300
out of bounds for int8") rather than wrapping, matching NumPy's `a.flat[i]=v` bounds check (floats truncate
first, then the truncated value is range-checked; NaN/inf raise; a bool target never does). A **strong** scalar
— an `NDArray`, i.e. the fancy/slice setters or `a.astype` — still wraps, as NumPy wraps an `np.int64` scalar.
To a float/complex/decimal dtype it uses `Convert` (exact). **Perf (NPY/NS, Release, 100K, non-contiguous):**
same-dtype set **1.8×**, cross-dtype→float set **2.4×**, get **6.3×** — all FASTER than NumPy (the
allocation-free `GetAtIndex`/`SetAtIndex` hot path); cross-dtype→integer set is correctness-first (`astype`,
slower, rarer). `x.flat` is UNCHANGED (still the raveled `NDArray`). See `APIs/np.flatiter.cs`; gate:
`APIs/np.flatiter.Test.cs` (13).

The three NumPy iteration objects, all following the `np.broadcast` house shape — a lowercase
factory returning a PascalCase nested class — and all **their own iterator** (NumPy's `iter(x) is x`),
so a second enumeration RESUMES rather than restarting. Indices are `long` (NumPy's `intp`) and every
step yields a FRESH index array, never a recycled buffer.

- **`np.ndindex(...)` → `np.NDIndex`** — C-order odometer over a shape's index space, last axis
  fastest. `params long[]` takes both spellings NumPy allows (`np.ndindex(3, 2)` and
  `np.ndindex(arr.shape)`); a plain `int[]` overload exists because array covariance won't widen it
  (it is deliberately NOT `params`, or `np.ndindex()` would be ambiguous). `ndindex()` and an empty
  shape both yield ONE empty index (Python's `product()` with no iterables); any zero-length
  dimension yields nothing; a negative dimension raises `ArgumentException "negative dimensions are
  not allowed"` at CONSTRUCTION, before a single index is produced. Deliberately **not** NDIter-backed
  — NumPy itself moved `ndindex` off `nditer` onto `itertools.product`, and there are no operands to
  walk, only a counter. `Indexing/np.ndindex.cs`.
- **`np.ndenumerate(arr)` → `np.NDEnumerate`** — yields `(index, value)` for every element. Port of
  NumPy's `asarray(arr).flat` + `flatiter.coords`, so the order is always LOGICAL C-order whatever the
  layout: F-contiguous, transposed, reversed, sliced and broadcast views all read through their own
  strides (`Shape.TransformOffset`), a 0-d array yields exactly one pair with an EMPTY index, and an
  empty array yields nothing. NumSharp's `NDArray.flat` is a raveled `NDArray` rather than a `flatiter`
  object (no `coords` cursor), so the coordinates come from an odometer advanced in lockstep with the
  flat position — which is what `flatiter.coords` is. `np.ndenumerate<T>(arr)` is a NumSharp extension
  yielding unboxed `T`. `Indexing/np.ndenumerate.cs`.
- **`np.nditer(...)` → `np.NDIterator`** — the public managed face of `NDIterRef`, i.e. the port of
  NumPy's `nditer_pywrap.c` (argument conversion, flag-string parsing, property surface, iteration
  protocol) over the C iterator NumSharp already had. Full signature parity
  (`op, flags, op_flags, op_dtypes, order='K', casting="safe", op_axes, itershape, buffersize`) with
  all 13 global and 15 per-op flag strings (incl. NumPy's `grow_inner`/`growinner` alias) and verbatim
  error texts — `Unexpected iterator global flag "…"`, `Unexpected per-op iterator flag "…"`,
  `order must be one of 'C', 'F', 'A', or 'K' (got 'Q')`, `casting must be one of …`,
  `Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked`,
  `Iteration of zero-sized operands is not enabled`, `Must provide at least one operand`. Property
  surface: `nop`, `ndim`, `shape`, `itersize`, `operands`, `dtypes`, `value`, `it[i]`, `index`,
  `multi_index`, `iterindex`, `iterrange`, `itviews`, `finished`, `has_index`, `has_multi_index`,
  `has_delayed_bufalloc`, `iterationneedsapi`; methods `iternext`, `reset`, `close`, `copy`,
  `remove_axis`, `remove_multi_index`, `enable_external_loop`, `debug_print`. `APIs/np.nditer.cs`.

**Three traps `np.nditer` had to solve — do not re-break:**
- **`NDIterRef` is a `ref struct` and cannot be a class field.** The bridge is
  `NDIterRef.Detach`/`Borrow` (`Backends/Iterators/NDIter.Detach.cs`), which hands the heap
  `NDIterState*` — plus operands AND pending COPY_IF_OVERLAP write-backs — to the managed owner and
  re-borrows a non-owning `NDIterRef` per call. The pre-existing `ReleaseState`/`FreeState` pair is
  NOT enough on its own: it nulls the state but strands `_writebackOriginals` on the dying ref struct,
  so a copied-for-overlap temp would never be written back. `np.nditer` therefore MUST be disposed
  (`close()`/`using`), exactly like NumPy's `with np.nditer(...) as it:`.
- **`GetEnumerator()` must NOT return `this`.** `foreach` disposes the enumerator it obtains, and this
  class's `Dispose` frees unmanaged state — so returning `this` (the `np.Broadcast` pattern, safe only
  because Broadcast owns no unmanaged memory) CLOSES the iterator at the end of any `foreach`/LINQ
  pass and every later property read throws. A thin wrapper with a no-op `Dispose` shares the same
  live cursor, so the observable semantics are unchanged.
- **`MoveNext` publishes THEN advances.** NumPy's `__next__` returns `self.value` and *then* calls
  `iternext()`, which is why after consuming one element the cursor already reads 1 and a `copy()`
  taken there continues from the SECOND element. Safe because `value` captures the operand's ABSOLUTE
  data pointer — except under buffered `external_loop`, where the view aliases a buffer the next step
  refills (NumPy has the identical hazard, hence its `[x.copy() for x in it]` idiom).

`np.nditer` also fills a gap `NDIterRef` leaves: an ALLOCATE operand with no `op_dtypes` entry throws
in the engine, while NumPy INFERS the dtype by promoting the operands that have one — so
`np.nditer([a, null])` allocates `int64` for an `int64` input and `np.nditer([int_a, float_b, null])`
allocates `float64`. That inference lives in the wrapper, where NumPy puts it.

**Perf (NPY/NS, higher = NumSharp faster; best-of-9, Release, 100K float64):** `ndindex` **1.5×**,
`ndenumerate` **3.3×**, `nditer` with `multi_index` **8.0×**, `external_loop` ~parity — but the
per-element `it[0]` loop is **0.16–0.18×** (56–76 ms across runs; the variance is GC, it allocates
100K `NDArray`s). The iteration ENGINE is not the problem: a bare `iternext()` walk of 100K elements
takes 0.61 ms against NumPy's 2.60 ms for the same bare walk (**4.3×**). The whole deficit is the
per-element `NDArray` view (~0.6 µs each) — `new NDArray(storage, shape)` re-aliases the storage, so
the hot 0-d path takes the direct slice ctor instead (measured 2× cheaper) and the strided
`external_loop` path (once per CHUNK, not per element) keeps the storage route.

**Do NOT try to fix `it[0]` by re-seating a cached view** — measured, and it is a silent-wrong-answer
trap. `UnmanagedStorage` keeps **three** synchronized address caches: the public `byte* Address`, the
`IArraySlice InternalArray`, and a per-dtype `ArraySlice<T> _arrayXxx` field. Re-seating `Address`
alone makes `GetAtIndex`/`ToString`/casts correct while `np.sum(it[0])` and `it[0] * 2` silently
return **0**; re-seating `Address` + `InternalArray` fixes the reductions and still leaves the
arithmetic operators and comparisons stale. The re-seat is fast (0.68 ms boxed, ~16× NumPy, vs 59 ms
today) but is only safe if every cache is updated behind one `UnmanagedStorage` method — and any
future cache added there re-breaks it silently. That is the core-wide change, and this is why.
The NumSharp-idiomatic fast paths — the typed iterators below, `np.evaluate`, vectorized ops — avoid
the per-element view entirely instead.

**Gate:** unit tests only (`Indexing/np.ndindex.Test.cs`, `Indexing/np.ndenumerate.Test.cs`,
`APIs/np.nditer.Test.cs` — 83 tests from probed NumPy 2.4.2 output). These are iteration PROTOCOLS,
not value-producing ops, so they have no `(dtype, shape, bytes)` result for the differential-fuzz
corpus to bit-compare and are deliberately absent from `gen_oracle.py`/`OpRegistry.cs`.

### Typed iteration — `np.nditer<T>` / `np.nditer_chunks<T>` (NumSharp extension)

The unboxed counterparts of `np.nditer`, over the same `NDIterRef` engine. NumPy has no equivalent
(Python has no unboxed generics), so there is no oracle tier — the gate is
`APIs/np.nditer.Typed.Test.cs` (23 tests), which pins them against the boxed `np.nditer` and
`ndenumerate` across every layout and all 15 dtypes.

```csharp
foreach (ref double x in np.nditer<double>(a, writeable: true)) x *= 2;              // element-wise
foreach (Span<double> c in np.nditer_chunks<double>(a, writeable: true))             // chunk-wise
    TensorPrimitives.Multiply(c, 2.0, c);
```

**Perf (NPY/NS, 100K float64, vs NumPy's `for x in np.nditer(a)` = 6.681 ms):** `np.nditer<T>`
**40×** (0.167 ms), `nditer_chunks<T>` **41×** (0.162 ms), chunks + `Vector<T>` **249×** (0.027 ms).
The raw-pointer floor is 0.047 ms, so the element walk costs ~1.6 ns/element over it — that residue is
`MoveNext`, and a leaner one-cursor variant measured no better. Layout cost is flat (0.16–0.19 ms for
contiguous / F / reversed / offset / broadcast).

**Five things this design had to get right — do not re-break:**
- **`GetEnumerator()` must NOT return `this`.** Unlike `np.Broadcast`, these own unmanaged state, so
  the first `foreach` frees what the second would walk — a use-after-free, caught by a test. The
  factory returns a **`readonly struct` enumerable holding no unmanaged state**, whose
  `GetEnumerator()` builds a FRESH `NDIterRef` (0.4 µs). So re-enumeration restarts — deliberately
  unlike the class-based `np.nditer`, which is its own iterator (NumPy's `iter(x) is x`) and resumes.
- **`foreach` disposes a `ref struct` enumerator through the same pattern**, no `IDisposable` needed;
  that is what frees the state. Hand-driven `MoveNext` loops must dispose themselves.
- **The walk is chunk-driven, not element-driven.** `EXTERNAL_LOOP` is always on and `MoveNext` walks
  the inner loop with pointer arithmetic, touching the iterator only at a chunk boundary — a
  contiguous array is ONE chunk, so per-element iterator cost is ~zero. The naive
  `Iternext()`-per-element form measured 0.47 ms, 3× worse.
- **Never buffered.** A `ref`/`Span` into a buffer dangles the moment the next fill lands; unbuffered
  pointers are absolute into the operand, which is also why the enumerator can advance the iterator
  *before* handing out the captured chunk.
- **Order is `'K'` (memory order), matching `np.nditer` exactly** — probed: `a[:, ::-1]` of
  `arange(6).reshape(2,3)` yields `0 1 2 3 4 5` in BOTH libraries, and `2 1 0 5 4 3` under `order='C'`.
  It is also what makes reversed/F/transposed views coalesce to one chunk. Pass `order: 'C'` for the
  logical order `ndenumerate` uses.

Two deliberate divergences, both pinned by tests: a **dtype mismatch throws** (a `ref` cannot convert,
so reinterpreting the bytes would hand out garbage — `np.nditer<double>` on an int32 array is an
error, not a cast), and an **empty array iterates zero times** where the boxed form demands NumPy's
`zerosize_ok` flag (forcing `if (a.size > 0)` around every `foreach` is not how C# collections
behave). `nditer_chunks<T>` additionally rejects a **non-unit-stride** inner loop up front —
`a[":, ::2"]` — because a `Span<T>` is contiguous by definition; the single-operand inner stride is
fixed for the whole iteration, so this is detected at `GetEnumerator` and never mid-loop.

**Four `NDIterRef` defects found while building this (all routed around, none fixed):**
`GetInnerLoopSizePtr()` **access-violates on a 0-d operand** (reads `Shape[-1]`; the internal
`ResolveInnerLoopCount` already guards it, the public accessor does not); `GetInnerStrideArray()`
returns **element** strides while the kernel contract (`ForEach`/`ExecuteGeneric`) takes **byte**
strides from the private `GetInnerLoopByteStrides()`, with no public byte-stride accessor;
`ResetToIterIndexRange` + `EXTERNAL_LOOP` **does not clamp the inner count to the range**, so it reads
past the array end — which blocks parallel range-splitting (NumPy requires `NPY_ITER_RANGED` at
construction for exactly this reason; splitting by array SLICE works today and measured **3.2×** on 8
workers); and NumPy's unbuffered `external_loop` coalesces `[:, ::2]` into **1** chunk where NumSharp
emits **500**. The first two are absorbed by `TypedIterHelpers.ReadInnerLoop`.

### Fused Expressions (NumSharp extension)
`evaluate` — `np.evaluate(expr[, operands][, out])` compiles an `NDExpr` tree to ONE NDIter pass: every elementwise node runs inside a single inner-loop kernel, so chained expressions allocate no intermediates and read each operand once (NumPy-ecosystem equivalent: `numexpr.evaluate`; measured 3.2–6.1× faster than NumPy 2.4.2 on 4M chains, 1.2–4× over NumSharp's own unfused chains — gate: the `benchmark/fusion` subsystem of `benchmark/run_benchmark.py`).

```csharp
NDArray r = np.evaluate((NDExpr)a * b + 2);                          // fused a*b+2
NDArray d = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b)); // a,b dedup → 3 operand streams
NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b), @out: x);        // one-pass sum(a*b), no temp
```

Semantics (all probed against NumPy 2.4.2, pinned in `NDEvaluateTests.cs`):
- **Dtypes follow NumPy result_type PER NODE** (`NDExpr.Typing.cs`): NEP50 strong-strong incl. the int/float tier crossing (`i4+f4→f8`); weak python-scalar literals (`i4+2→i4`, `f2+2.5→f2`, `bool+2→i64`, out-of-range → OverflowError); `true_divide` ints→f64; `arctan2` tier floats; `power`/`remainder`/`floor_divide` bool→i8; unary math tiers (`bool/i8→f16`, `i16→f32`, `i32+→f64`); comparisons→bool. `(i4*i4)+f8` wraps the multiply in int32 before promoting — bit-compatible with the unfused NumPy sequence.
- **Reductions are root-only**: `NDExpr.Sum/Prod/Min/Max/Mean(expr)` run a one-pass accumulating kernel over the inputs (NumPy reduce dtypes; f16/f32 sums accumulate in f64 and cast back (more precise than NumPy's pairwise); min/max NaN-propagate; empty: sum=0, prod=1, mean=NaN, min/max raise).
- Repeated NDArray references deduplicate to one iterator operand; `out=` follows the ufunc rules above; `ExecuteExpression` (Tier 3C) throws without `EXTERNAL_LOOP` (the ~40× per-element foot-gun) — `np.evaluate` configures the iterator itself.

### Sorting & Searching
`argmax`, `argmin`, `argpartition`, `argsort`, `argwhere`, `flatnonzero`, `lexsort`, `nanargmax`, `nanargmin`, `nonzero`, `partition`, `searchsorted`, `sort`, `sort_complex`

**The issue-#623 six** — `partition`/`argpartition`/`lexsort`/`nanargmax`/`nanargmin`/`sort_complex` (all probed
against 2.4.2; gates `Sorting/np.{partition,lexsort,nanargmax,sort_complex}.Test.cs` (108) + ~2,170 differential
cases in the `sort`/`nanreduce` fuzz tiers) — are compositions/wrappers over the kernels that already shipped:
introselect (`Utilities/QuickSelect.cs`, the median/percentile primitive), the stable radix argsort
(`AxisSort`), and the `ArgMax`/`ArgMin` engine reductions. No third sort/select core was added.
- `partition(a, kth, axis=-1, kind='introselect', order=null)` + in-place `ndarray.partition` — the kth element(s)
  land in final sorted position, sides unordered. `Backends/Default/Sorting/AxisPartition.cs` drives one introselect
  line kernel per 1-D lane through **`AxisSort.DriveAllButAxis`** (the same NDIter IterAllButAxis loop the sorts
  use). Validation is NumPy's probed ORDER with verbatim texts: kind (`select kind must be 'introselect' (got '…')`)
  → order (`Cannot specify order when the array has no fields.`) → axis → writeable (`partition array is read-only`)
  → kth (`kth(=N) out of bounds (M)` reporting the POST-wrap value; bounds SKIPPED for size-0 arrays; the kth list
  is sorted so partitions compose, and multi-kth segments confine each introselect). Floats NaN-compact (NaN sorts
  last, ORIGINAL bit patterns preserved in the tail — unlike np.sort's canonicalizing radix); Complex uses NumPy's
  CDOUBLE_LT comparator, whose NaN order is positional ((1,NaN) BEFORE (NaN,1)) and deliberately NOT compacted.
  C# reads an empty `int[]` kth as NumPy's `np.array([], intp)` — a valid no-op copy (Python's bare `[]` is float64
  → TypeError; a typed array carries no such ambiguity). Result is a fresh C-contiguous copy (house np.sort
  convention; NumPy's `copy(order='K')` keeps F-order for F-inputs — values identical). **The NDArray-kth overload**
  (NumPy's array form, on np.partition/np.argpartition AND the ndarray methods) is what makes NumPy's kth-dtype
  rejections REACHABLE — bool kth `Booleans unacceptable as partition index`, non-integer (incl. Decimal)
  `Partition index must be integer` (TypeError), >1-D `object too deep for desired array` — with NumPy's probed
  STAGING: kind → order → too-deep (BEFORE axis: `partition(a, [[1]], axis=9)` reports too-deep) → axis →
  bool/integer dtype → wrap/bounds. Integer kth values cast to intp MODULARLY like NumPy (uint64 2^63 wraps
  negative and its bounds error reports the wrapped value; 2^64−1 wraps to a LEGAL −1); Char rides the integer
  route (house call); 0-d and any layout accepted.
- `argpartition` — same driver returning int64 indices; the source is only read (broadcast views legal). Built on
  the **index-tracking QuickSelect overloads** (`PartitionAtMany(T*, long*, …)` — values and their original indices
  swap in lockstep, the `RadixSort.ArgSortU32/U64` pattern). A 0-d input ravels to `(1,)` FIRST — NumPy's arg-side
  `PyArray_CheckAxis` quirk (axis/kth errors then report dimension/size 1, result `[0]`) while `np.partition(0-d)`
  raises AxisError; float NaN indices tail in encounter order (argsort's exact policy). **The fuzz tier pins the
  DERIVED kth-values** (`take(partition(a,ks),ks)` / the take_along_axis gather at ks): the arrangement BETWEEN kth
  anchors is introselect-implementation-specific on BOTH sides, so whole-output bytes are not contractual — the
  two-sided ≤/≥ invariant is unit-test-pinned instead.
- `lexsort(keys, axis=-1)` — indirect stable sort, LAST key primary. Pure composition reproducing `PyArray_LexSort`
  pass for pass: successive **stable** argsorts first→last key, each re-sorting the running permutation via
  `take_along_axis` — bit-identical indices to NumPy's mergesort passes because NumSharp's radix argsort is stable,
  ties included. Verbatim errors: `need sequence of keys with len > 0 in lexsort` (TypeError), `all keys need to be
  the same shape` (ValueError); 0-d keys let axis 0/-1 slip (NumPy's back-compat quirk) and size ≤ 1 early-returns
  a 0-filled int64 array. The single-NDArray overload reads the first axis as the key list (a 1-D input degenerates
  to N scalar keys → the 0-d `0`, NumPy's probed quirk).
- `nanargmax`/`nanargmin(a, axis=null, out=null, keepdims=false)` — NumPy's `_replace_nan(∓inf)` + all-NaN-slice guard
  (`ValueError("All-NaN slice encountered")`) then `argmax`/`argmin`; int/bool/char/decimal pass through untouched
  (NumPy's `mask=None` branch). The flat form returns `long`; the `int?`-axis overload returns the int64 NDArray and
  handles `keepdims` with `axis=null` (shape `(1,)*ndim`). **`out=` follows NumPy's probed argmax contract:** any
  bool/int dtype safe-castable to intp is accepted (bool through uint32 — uint64/floats raise the verbatim
  `Cannot cast array data from dtype('…') to dtype('int64') according to the rule 'safe'` TypeError), the shape must
  EQUAL the keepdims-adjusted result (`output array does not match result of np.argmax.` — nanargmin leaks
  "argmin"), the int64 indices cast UNSAFELY into it (an int8 out wraps index 200 → −56, a bool out reads
  index ≠ 0 — probed), the SAME instance returns, a strided out writes through its base, and the all-NaN guard
  fires BEFORE out is touched. Positional slot 3 is `out` (NumPy's order) — pass `keepdims:` by name; the implicit
  bool→NDArray conversion would otherwise bind a positional bool to `out`. Axis validation runs UP FRONT for every
  dtype path — the engine's argmax silently wraps out-of-range negative axes (`np.argmax(a2d, -3)` → `[1 1]`, a
  pre-existing argmax bug this family refuses to inherit) — with NumPy's rule that a 0-d array accepts axis 0/-1.
  The documented NaN+Inf untrusted-tie caveats fall out of the composition identically (probed cases pinned).
  **Auditing this family's dtype cells exposed and fixed three ENGINE argmax/argmin bugs** (all pre-existing, all
  now oracle-gated): flat `argmax(Decimal)` rode the IL arg kernel whose Bgt/Blt cannot compare a 16-byte struct
  and SILENTLY mis-answered (`argmax([3,9,1,5])` → 0) — Decimal now takes the same `ExecuteReducing` fallback
  family as Half/Complex (`DecimalArg{Max,Min}Kernel`); flat `argmax(Char)` threw NotSupportedException while the
  axis path supported char all along — it now rides the zero-copy uint16 reinterpret its amax already uses; and
  the strided/scalar IL step compared floats with `Bgt_Un`/`Blt_Un`, whose UNORDERED branch also fires when the
  ACCUM holds NaN, so every later finite value stole the index (`argmax` of a transposed `[nan,2,3,nan]` walked to
  3 instead of NumPy's first-NaN-wins 0; contiguous inputs were immune via the C# NaN helpers) — fixed to ordered
  `Bgt`/`Blt`. The reduce fuzz tier now generates the flat `axis=None` argmax/argmin cells whose absence hid all
  three (the old skip excluded flat argmax entirely; keepdims=False flat cells replay via
  `NDArray.Scalar(np.argmax(a))`, and the decimal oracle gained flat argmax cases).
- `sort_complex(a)` — copy + sort along the LAST axis **in the input's own dtype** (an int64 past 2^53 orders by
  its exact value), then up-cast. NumSharp's single `Complex` absorbs NumPy's complex64 cells (int8/16, uint8/16 —
  values identical, width documented; those four dtypes are excluded from the fuzz tier and unit-test-pinned).
  0-d leaks `ndarray.sort()`'s own `axis -1 is out of bounds for array of dimension 0`, exactly as NumPy leaks it.
  See `Sorting_Searching_Counting/np.{partition,argpartition,lexsort,nanargmax,sort_complex}.cs`.

### Linear Algebra
`diag`, `diagflat`, `diag_indices`, `diag_indices_from`, `diagonal`, `dot`, `einsum` (subscripts parsed; contraction pending), `fill_diagonal`, `inner`, `mask_indices`, `matmul`, `matvec`, `outer`, `tensordot`, `trace`, `tril`, `tril_indices`, `tril_indices_from`, `triu`, `triu_indices`, `triu_indices_from`, `vdot`, `vecdot`, `vecmat`

`np.linalg.*`: `cholesky`, `cond`, `cross`, `det`, `diagonal`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `inv`, `lstsq`, `matmul`, `matrix_norm`, `matrix_power`, `matrix_rank`, `matrix_transpose`, `multi_dot`, `norm`, `outer`, `pinv`, `qr`, `slogdet`, `solve`, `svd`, `svdvals`, `tensordot`, `tensorinv`, `tensorsolve`, `trace`, `vecdot`, `vector_norm` (+ `LinAlgError`)

### The BLAS/LAPACK API surface — implemented vs. NSE-throwing shell

Everything NumPy computes through the OpenBLAS binary the interop package bundles now has its public
API, its NumPy-exact validation and its engine seam. Whether the NUMERICS exist splits cleanly in
two, and the split is a property of the problem rather than a decision:

| | Members | With no backend installed |
|---|---|---|
| **Products** (CBLAS) | `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `linalg.multi_dot`, `matrix_power` (n ≥ 0), the `linalg` Array-API forms, `linalg.norm` except matrix ord ∈ {2, -2, 'nuc'} | **They compute.** Managed kernels; the `IBlasBackend` invariant holds — a backend changes WHICH implementation runs, never WHETHER an answer exists. |
| **Factorisations** (LAPACK) | `cholesky`, `det`, `slogdet`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `inv`, `pinv`, `lstsq`, `qr`, `solve`, `svd`, `svdvals`, `matrix_rank`, `cond`, `tensorinv`, `tensorsolve`, `matrix_power` (n < 0), `norm` matrix ord ∈ {2, -2, 'nuc'} | **`NotSupportedException` from `TensorEngine`.** Core ships no managed LU/QR/SVD/eigensolver, so there is nothing to fall back to. The message names the NumPy API, the LAPACK routine and the `IBlasBackend.Try*` member to implement. |

**The seam is `TensorEngine.Blas` — one property, not two.** `IBlasBackend.LinearAlgebra.cs` adds 15
`Try*` members, ALL of them **default interface implementations returning false**, which is what let
the file exist without touching `OpenBlasBackend`. (Default interface members are
interface-dispatched and invisible on the implementing class — pinned by a test that types the
variable as `IBlasBackend` precisely because it must.) LAPACK belongs on this property because it
lives inside the very binary already bundled: scipy-openblas ships `gesv`/`getrf`/`potrf`/`geev`/
`syevd`/`heevd`/`gesdd`/`geqrf`/`orgqr`/`gelsd` next to the BLAS symbols. `TensorEngine`'s members
are `virtual`, and each reads `Blas` into a **local** before calling it (GEMM_PARITY §9 — a
test-then-call on a settable property was ~2 % NREs under a concurrent `Disable()`).

**Validation runs before the throw, so the error contract is gated NOW** and does not move when the
numerics land — `_assert_stacked_2d`/`_assert_stacked_square`/`_commonType` are ported verbatim
(`LinAlgError("{n}-dimensional array given. Array must be at least two-dimensional")`,
`"Last 2 dimensions of the array must be square"`, `TypeError("array type float16 is unsupported in
linalg")` — and `Decimal`/`Char`, which have no NumPy dtype, take float16's exit rather than
inventing a new error).

**Six NumPy behaviours worth not re-deriving** (all probed against 2.4.2; the whole 132-case matrix
was replayed through both libraries — 121 bit-exact, 11 expected NSE, 0 unexplained differences):

- **`np.inner` reports `b`'s shape ALREADY TRANSPOSED.** It is `matrixproduct(a, swapaxes(b,-1,-2))`,
  so `np.inner(ones((2,3)), ones((3,2)))` raises "shapes (2,3) and **(2,3)** not aligned".
- **`np.vdot`'s length mismatch is a RESHAPE error, and its own length check is dead code.**
  `array_vdot` flattens both operands through ONE reused `PyArray_Dims` buffer holding `-1`;
  `_fix_unknown_dimension` writes the resolved length back into it, so the second flatten asks for
  `(a.size,)`. Hence `vdot(ones(3), ones(5))` → "cannot reshape array of size 5 into shape (3,)", and
  `"vectors have different lengths"` in the C source is unreachable.
- **`vecdot` reduces in the LOOP dtype, not NEP50's accumulator** — its loops are `'ii->i'`, so an
  int32 pair stays int32 where `np.sum` would give int64. `vecdot`/`vecmat`/`vdot` conjugate their
  FIRST operand; `inner`/`matvec` do not.
- **`axes=` names the core axes per operand, and is the general form of `axis=`.** One entry per
  input AND output — `np.vecdot(m, v, axes: [[-1],[-1],[]])`. The output entry may be omitted ONLY
  where the output has no core axes, which in this family is `vecdot` alone; supplying both `axes`
  and `axis` raises `TypeError("cannot specify both 'axis' and 'axes'")`. For `matvec`/`vecmat` the
  output entry says where the result's core axis LANDS, so
  `np.matvec(a, b, axes: [[-2,-1],[-1],[0]])` transposes the answer.
- **`axis=`/`keepdims=` compute only on `vecdot` — but exist on all three.** `matvec`/`vecmat` have
  two DISTINCT core dims and NumPy raises `TypeError` for both, so they are offered rather than
  omitted: a ported call then fails the same way with the same message instead of failing to
  compile. The keepdims text names input 1's core rank, which differs per signature (1 for
  `matvec`, 2 for `vecmat`).
- **`keepdims=True` COMPOSES with `axes=` on `vecdot`.** Under keepdims the output gains the reduced
  core dimension (kept as size 1), so a PROVIDED output entry must be length 1 (`[[1],[1],[1]]`) and
  says WHERE the kept axis lands in the output — but it may still be OMITTED (`[[1],[1]]`), in which
  case the kept axis goes to the END. An explicit empty output entry (`[[1],[1],[]]`) then
  CONTRADICTS the kept dimension and is the AxisError "operand 2 has 1 core dimensions, but 0
  dimensions are specified"; the mirror-image `[[1],[1],[1]]` WITHOUT keepdims is "has 0 … but 1
  specified". The output axis is validated against the output rank (`axis N is out of bounds for
  array of dimension M`). The whole interaction routes through one `NormalizeAxes` override for the
  effective output core rank — the earlier code rejected every `axes`+`keepdims` call outright.
- **`np.linalg.matrix_rank` below rank 2 is a PREDICATE, not a count.** NumPy short-circuits with
  `int(not all(A == 0))`, so `matrix_rank([1,2,3])` is **1**, not 3, and an empty operand is 0
  (`all([])` is true). Easy to get wrong, because a count returns a plausible number for every
  non-degenerate input.
- **`tensordot` has THREE distinct rejections, reproduced in NumPy's order.** A repeated RAW axis
  value is `ValueError("duplicate axes are not allowed in tensordot")`, checked first, ahead of any
  shape or range test (so `([5,5],[0,1])` is a duplicate, not an index error). An out-of-range
  contraction axis is `IndexError("tuple index out of range")` — NumPy indexes the shape TUPLE with
  the raw axis (`as_[axes_a[k]]`), so `tensordot(a2d, b2d, 3)` (the int count exceeds ndim) and
  `([5,0],[0,1])` are index errors, NOT shape mismatches. Mismatched extents and unequal list lengths
  are `ValueError("shape-mismatch for sum")`; because the extent comparison short-circuits the loop,
  a mismatch found earlier HIDES a later out-of-range axis (`([0,1],[0,5])` settles as the shape
  mismatch). A negative count contracts NOTHING (`range(-axes,0)` is empty for `axes ≤ 0`), so
  `axes=-1` ≡ `axes=0` ≡ the outer product.
- **`np.linalg.multi_dot` requires every non-endpoint operand to be 2-D.** NumPy promotes a 1-D
  first/last operand (row/column) and THEN runs `_assert_2d` over all of them, so a 0-D, a 3-D+, or
  a 1-D operand in a MIDDLE position raises `LinAlgError("{ndim}-dimensional array given. Array must
  be two-dimensional")`. The guard is load-bearing, not cosmetic: without it the Cormen cost model
  reads a phantom `dimensions[1]` off a non-matrix and the chain either leaks a raw bounds error or —
  worse — returns a silently wrong-shaped product (a 3-D middle used to give `(2,4,2)` where NumPy
  refuses). `out=` is threaded into the FINAL dot, so it takes the TWO-dimensional product shape even
  for a vector endpoint (`multi_dot([v, B, C], out)` wants `(1,k)` and returns a raveled `(k,)` view,
  not `(k,)` in and out). Exactly two operands is a plain `np.dot` and skips the 2-D check entirely.
- **The `np.linalg` Array-API forms are NOT aliases.** `linalg.trace`/`linalg.diagonal` reduce the
  **last** two axes where `np.trace`/`np.diagonal` take the **first** (a `(2,3,3)` stack gives `(2,)`
  vs `(3,)`); `linalg.outer` demands 1-D where `np.outer` flattens anything; `linalg.cross` demands
  3-vectors. Each is implemented against its own contract.

**Two fixes rode along.** `NDArray.matrix_power(-1)` used to `throw new Exception("matrix_power just
work with int >= 0")` — never NumPy's rule (`a**-n` is `inv(a)**n`); it now takes that route, plus
binary exponentiation, a `LinAlgError` for non-square input, and an identity in the operand's own
dtype rather than always float64. And `np.dot`/`np.matmul`'s "not aligned" messages rendered shapes
as `(2, 3)` where NumPy writes `(2,3)` — now `Shape.ToPythonTuple()` (`ToString()` is left alone; it
renders for humans everywhere else).

**Known adjacent bug, NOT fixed here:** `np.stack` runs its operands through `atleast_1d` first, so
stacking three 0-d arrays gives `(3,1)` where NumPy gives `(3,)`. `linalg.cross` routes around it
with `expand_dims`+`concatenate`.

### `np.einsum` — the subscript language is complete, the contraction is not

The parser is a port of `PyArray_EinsteinSum`'s parsing block plus `parse_operand_subscripts` and
`parse_output_subscripts` (`numpy/_core/src/multiarray/einsum.cpp`) — `EinsumSubscripts.cs`. **Every
expression NumPy accepts parses, every expression NumPy rejects is rejected with NumPy's own text,
and the output SHAPE is resolved** — it is carried in the `NotSupportedException` the engine then
raises, which is what let the whole surface be differential-tested against NumPy without a kernel
(75 cases: 73 exact, 2 the documented divergence below). Both spellings work: the subscripts string
and NumPy's sublist form `np.einsum(a, [0,1], b, [1,2], [0,2])`.

**NumPy has TWO einsum parsers** — the C one behind the default `optimize=False` and a Python one
(`einsumfunc.py`) behind the `optimize` path — and they word the same rejection differently.
NumSharp reproduces the C one whatever `optimize` says, since that is what a default call hits.

**The label encoding is NumPy's**, one `sbyte` per operand dimension: positive = the label's ASCII
code at first occurrence, **negative = the offset back to that first occurrence** (a repeated label,
i.e. a diagonal), `0` = a broadcast dimension from an ellipsis. So `"abbcbc"` over 6 dims is
`[97,98,-1,99,-3,-2]`.

**Six things that are easy to get wrong, all probed:**
- **The two operand-count messages read BACKWARDS.** NumPy walks operand by operand checking the
  delimiter after each chunk, so ONE operand against TWO terms raises *"more operands provided to
  einstein sum function than specified in the subscripts string"*. Reproduced verbatim anyway.
- **An impossible diagonal has TWO messages.** `nop == 1 && out == null &&` nothing summed takes
  NumPy's return-a-view attempt, which says *"dimensions in **single operand** for collapsing index
  …"*; anything else says *"dimensions in **operand {i}** for …"*. So `einsum("ii->i", a)` and
  `einsum("ii", a)` fail differently on the same array.
- **Every diagonal is validated BEFORE any cross-operand extent.** NumPy collapses all operands'
  diagonals first, so `einsum("ij,jj->ij", …)` reports operand 1's diagonal even though operand 0
  already fixed a conflicting `j`. Doing it in one pass reverses the two errors.
- **The implicit output is ASCII-ordered**, so an upper-case label sorts BEFORE a lower-case one:
  `einsum("zA", a)` transposes a `(2,3)` and `einsum("Az", a)` does not.
- **The sublist alphabet is `'ABC…Zabc…z'` — UPPER case first** (NumPy 2.4.2's `einsum_symbols`).
  Index order therefore equals ASCII order, which is what makes an inferred output come back in the
  caller's numbering. The intuitive guess (`a-z` then `A-Z`) is SILENTLY wrong, not an error: it only
  shows on an expression mixing the halves, where `einsum(a, [0,26])` on `(2,3)` returns `(3,2)`.
- **Spaces are legal anywhere** and skipped; a `>` without its `-` is scanned as a label and rejected
  as an invalid subscript, while a lone `-` gets the missing-arrow message.

**One deliberate divergence** (`[Misaligned]`): when a label's extents disagree between operands,
NumPy's C path leaks `NpyIter`'s *"operands could not be broadcast together with remapped shapes
[original->remapped]: (2,3)->(2,newaxis,3) (4,2)->(2,4)"* — iterator axis bookkeeping rather than
anything about the contraction. NumSharp raises the wording NumPy's OWN `einsumfunc.py` parser uses
for the identical condition: `Size of label 'j' for operand 1 (3) does not match previous terms (4).`
(whose two sizes also read swapped against the sentence — the first is the size already recorded).

**Pass the keywords BY NAME** — `np.einsum("ij->i", ops, @out: dst)`. They are keyword-only in NumPy,
and naming them is also what keeps them unambiguous here: NumSharp converts scalars to `NDArray`
implicitly, so a fully positional 7-argument call matches both the `params` overload and the full one
and the compiler rejects it.

**Not implemented and deliberately out of scope:** `np.einsum_path` (NumPy's contraction planner —
its own function, ~400 lines of optimal/greedy search), and the contraction kernel itself. Unlike
`np.linalg`, einsum is NOT waiting on a backend — a summation kernel over an arbitrary label set and
a path planner are NumSharp's own work — so the seam has no `TryEinsum` and the message points at
`np.tensordot` rather than at a package to install. Gate:
`LinearAlgebra/EinsumSubscriptParityTests.cs` (21).

**Signature parity is its own gate.** `LinAlgSignatureParityTests` reflects over the whole surface
and asserts each function has an overload whose parameter NAMES appear in NumPy's ORDER, plus the
18 defaults NumPy states concretely — because renaming `UPLO` to `uplo` or swapping `keepdims` and
`ord` compiles, passes every value test, and silently breaks every named-argument call ported from
Python. It also asserts no `np.linalg` member is missing. Two defaults C# cannot spell (`object`
parameters may only default to null) are carried by a **null sentinel** and pinned behaviourally:
`matrix_norm`'s `ord='fro'` and `vector_norm`'s `ord=2`. Four NumPy ufunc keywords are deliberately
ABSENT from the gufuncs — `casting`, `order`, `subok`, `signature` — because NumSharp models none of
them anywhere in its ufunc surface (`signature` is what `dtype` already does; `subok` concerns
ndarray subclasses the library does not have), and accepting-then-ignoring would be worse.

Gate: `LinearAlgebra/{BlasProductApiTests, LinAlgErrorParityTests, LinAlgEngineSeamTests,
LinAlgSignatureParityTests, EinsumSubscriptParityTests}.cs` (92 tests), backed by a 451-case NumPy
differential matrix replayed through both libraries (dtype sweep × 10, memory-layout sweep × 8, the
`axes=` paths, einsum's whole subscript grammar, every rejection): 435 exact, 5 expected NSE, 11
documented divergences with verbatim messages, **0 unexplained**. `LinAlgEngineSeamTests`' first region is a checklist — **delete a case as each
implementation lands**. These are API/error contracts rather than value-producing kernels, so they
are unit tests and deliberately absent from the differential-fuzz corpus.

**Two known type-only divergences, messages verbatim on both sides:** the reshape/alignment errors
raise `IncorrectShapeException` where NumPy raises `ValueError` (long-standing house convention,
see the reshape section), and `AxisError` derives from `ArgumentOutOfRangeException` with
`paramName: "axis"`, so .NET appends `(Parameter 'axis')` to `.Message` — which is why the whole
library asserts axis texts with wildcards.

**The diagonal / triangular family** (NumPy's `_twodim_base_impl.py` + `_index_tricks_impl.py`; all probed against 2.4.2, tests in `Indexing/np.{diag,tri,diag_indices}.Test.cs`, oracle tiers in `manip`/`matmul`):

- `diag(v, k=0)` — **the two branches differ in kind**: a 1-D input *constructs* a fresh writeable C-contiguous `(n, n)` matrix (`n = v.size + |k|`) with `v` on the k-th diagonal; a 2-D input *extracts* via `diagonal(v, k)` and so returns a **read-only view sharing storage** (probed: `shares_memory` True, `writeable` False — the docstring's "a copy" is wrong). Anything else raises `ValueError("Input must be 1- or 2-d.")`. The construction path allocates the zero matrix and aliases a **writeable** length-`v.size`, stride-`(n+1)` view onto it, letting the ordinary copy machinery fill the diagonal — NumPy's `res.flat[i::n+1] = v` with no element loop.
- `diagflat(v, k=0)` — as `diag`'s 1-D branch but accepts **any** rank: the input is raveled in **logical C order** first (so an F-contiguous or strided view is read row-major, not memory-order), 0-d → `(1,1)`.
- `tri(N, M=None, k=0, dtype=float)` — ones at and below the k-th diagonal. NumPy computes `greater_equal.outer(arange(N), arange(-k, M-k)).astype(dtype)` (two full N×M passes + a bool temp); NumSharp blits a prefix of one pre-built ones row into each row, dtype-agnostically. Negative `N`/`M` clamp to an empty axis (NumPy reaches them through `arange`).
- `tril(m, k=0)` / `triu(m, k=0)` — always a **fresh writeable C-contiguous copy**, dtype preserved, applied to the last two axes. Two NumPy spellings leak through and are reproduced: a **1-D input yields a 2-D result** (`m.shape[-2:]` of `(n,)` is `(n,)`, so the mask is `tri(n)` → `(n,n)` and the row broadcasts down it — `tril([1,2,3])` → `[[1,0,0],[1,2,0],[1,2,3]]`), and a **0-d input raises** NumPy's leaked `tri()` missing-argument error, verbatim: `tri() missing 1 required positional argument: 'N'`. Because the retained columns of each row form one contiguous run, the mask is skipped entirely: the result is filled row-by-row (blit the kept run, zero the rest). Sources whose last axis is not unit-stride fall back to the literal `where(tri(...), m, 0)` composition.
- `fill_diagonal(a, val, wrap=False)` — **in place, returns void**. Values **tile cyclically and truncate**, they do not broadcast (`a.flat[:end:step] = val` semantics — probed: `[1,2,3,4]` into a 6×6 gives `[1,2,3,4,1,2]`, an over-long list is cut, and an **empty** one is a silent no-op). `wrap` only affects tall matrices. Errors verbatim: `array must be at least 2-d`, `All dimensions of input must be of equal length` (ndim > 2, non-hyper-cubic), `underlying array is read-only`. Implemented with **no element loop** — the targets resolve to real strides, so it is one aliased stride-`Σstrides` view (or, when wrapping, `ceil(count/cols)` equally-strided blocks), which is why it writes correctly through transposed, sliced, F-order and negative-stride views (verified for `wrap` × non-contiguous too, the one combination the oracle tier does not reach). `val` is `object` and accepts everything NumPy's array_like does — scalars, `NDArray`, C# arrays (incl. rectangular 2-D, which ravel), and `IEnumerable` collections. Only genuine scalars take the `NDArray.Scalar` fast path (it converts via `IConvertible`); arrays, collections, `Half` and `Complex` route through `asanyarray`. A **null** `val` throws `ArgumentNullException` — a deliberate divergence from NumPy, where `None` reaches the dtype converter and yields nan for a float array (TypeError for an integral one); in C# a null is a caller bug and silently writing nan would hide it.
- `diag_indices(n, ndim=2)` / `diag_indices_from(arr)` / `tril_indices(n, k=0, m=None)` / `triu_indices` / `tril_indices_from(arr, k=0)` / `triu_indices_from` / `mask_indices(n, mask_func, k=0)` — return `NDArray<long>[]` (the house tuple convention; `intp` is int64). `diag_indices` returns **the same array instance** in every slot, matching NumPy's `(idx,) * ndim` aliasing (a write through one is visible through all); `ndim <= 0` gives an empty array. `mask_indices`' arity follows the mask function's rank, so passing `diag` yields **one** index array. The `tril/triu_indices` generators skip NumPy's N×M bool mask + two broadcast coordinate grids + boolean gathers entirely: the triangle is separable, so the row/column runs are emitted directly in `O(output)` — **4.4× NumPy at 10M**.

**Perf (NPY/NS, best-of-rounds, Release):** geomean **1.83× at N=10M** (no cell below 1.0; min 1.13×) and **1.40× at N=100K**. `tril_indices`/`triu_indices` lead at 4.40×/4.55× (10M). The single sub-1.0 cell is `fill_diagonal` at 100K (0.62×, 3.7 µs vs 2.3 µs) — it touches only `min(rows, cols)` elements, so the call is dominated by NDArray/Shape construction rather than work; at 10M it is 1.23×. The triangular fills pick their allocation strategy by size: below **64 MiB** the buffer comes back dirty so they allocate uninitialised and write every byte exactly once, above it `np.zeros` rides free OS zero-pages and only the retained run is touched (measured cliff — 64 MB: 6.0 ms vs 11.0 ms write-once; 72 MB: 18.9 ms vs 13.5 ms the other way; see `np.tri.cs` → `WriteOnceMaxBytes`).

### Random (`np.random.*`)
`bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `dirichlet`, `exponential`, `f`, `gamma`, `geometric`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `logseries`, `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `permutation`, `poisson`, `power`, `rand`, `randint`, `randn`, `random_sample`, `rayleigh`, `seed`, `shuffle`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `uniform`, `vonmises`, `wald`, `weibull`, `zipf`

### File I/O
`fromfile`, `load`, `load_npy`, `load_npz`, `save`, `savez`, `savez_compressed`, `tofile`

The `.npy`/`.npz` stack is a **port of NumPy 2.4.2's `numpy/lib/_format_impl.py`** (NEP-01) in
`IO/{NpyFormat,NpzFile,PyLiteral}.cs`, and the writer is **byte-for-byte identical to NumPy's own
`np.save`** — not merely readable by it. Format versions **1.0 / 2.0 / 3.0** (2- vs 4-byte header
length; latin-1 vs UTF-8), **64-byte `ARRAY_ALIGN`** data alignment (so the data is mmap-ready),
**`fortran_order`** both ways (write via the transpose; read via reshape-reversed + transpose), and
**big-endian** files byte-swapped to native on read (per-component: `>c16` swaps in 8s, `>U1` in 4s).
Headers parse through a real Python-literal parser (`PyLiteral`, standing in for `ast.literal_eval`)
rather than `IndexOf`/regex, so key order, whitespace, quoting, trailing commas and the Python 2 `3L`
suffix all work; `max_header_size` (default 10000) bounds hostile input and `allow_pickle` lifts it.

`np.load` returns `object` — `NDArray` for `.npy`, `NpzFile` for `.npz` — mirroring NumPy's
content-dependent return; **`np.load_npy` / `np.load_npz` are the typed forms** and need no cast.
`NpzFile` is lazy + cached, takes `"w"` or `"w.npy"` as keys, exposes `.Files` stripped, supports
`npz.f.weights` dot access (NumPy's `BagObj`) and **must be disposed**.

| Dtype | Maps to | Notes |
|-------|---------|-------|
| Boolean/SByte/Byte | `\|b1` / `\|i1` / `\|u1` | single-byte types take the `\|` prefix, never `<` |
| Int16…UInt64, Half, Single, Double | `<i2`…`<u8`, `<f2`, `<f4`, `<f8` | direct |
| Complex | `<c16` | `<c8` widens to `Complex` on read |
| Char | `<U1` | 2-byte UTF-16 ↔ 4-byte UCS-4; non-BMP rejected (needs a surrogate pair) |
| Decimal | — | `NotSupportedException`: no NumPy dtype (cast to Double) |

Object arrays, structured/subarray dtypes, `datetime64`/`timedelta64`, `\|S`/`<U`n>1/`\|V`, `<f16`
and `<c32` are parsed then rejected with a precise message — including the zero-width `<U0`/`\|S0`/`\|V0`,
which are *valid* NumPy dtypes and so report "unsupported", never "invalid descriptor".

**`mmap_mode` is implemented** (`NpyFormat.OpenMemmap`, port of NumPy's `open_memmap` + `numpy.memmap`):
`np.load(path, mmap_mode=…)` returns an `NDArray` backed by a `MemoryMappedFile` view — zero-copy,
the mapping released with the array (and its views) on dispose/GC. Modes match what NumPy actually
does *through `np.load`*: **`r`/`readonly`** → read-only (non-writeable) view, **`r+`** → read-write
(flushes to disk), **`c`** → copy-on-write (not flushed). The other four documented spellings validate
but fail downstream in NumPy too, reproduced verbatim: **`w+`** → `TypeError "object of type 'NoneType'
has no len()"`, **`write`/`copyonwrite`/`readwrite`** → `ValueError "invalid mode: '<mode>b'"` (they
contain the letter *w*, so `open_memmap` misroutes them into its create branch — NumSharp does **not**
truncate the file as NumPy's `w+` does). Validation happens **only on the `.npy` branch**: an `.npz`
ignores `mmap_mode` entirely (even an invalid value → `NpzFile`), matching NumPy. mmap needs a real
**path** — a stream/`byte[]` npy raises NumPy's `ValueError "…Memmap cannot use existing file handles"`.
NumSharp-specific rejections (`NotSupportedException`): big-endian and `<U1`/`<c8` files, which NumSharp
byte-swaps / widens on read and so cannot present as a zero-copy view. Design: `docs/MMAP_MODE_HANDOVER.md`;
tests: `IO/NpyMemmapTests.cs` (16).

**Known intentional divergences** (differential-verified, everything else agrees): the 5 unsupported
dtypes above are the *only* files NumPy loads that NumSharp refuses. NumPy reports `shape: (True,)`
and overflowing dims via a later TypeError/OverflowError while NumSharp rejects them up front — both
refuse, the text differs. Big-endian files byte-swap to native (NumPy keeps a byte-swapped dtype),
and `Char` round-trips U+0000 as a NUL where NumPy's `<U` reports `''` (the bytes are identical —
`<U` treats NUL as string padding).

**Three traps this port has already fallen into — do not re-break:**
- **`(1)` is not a tuple.** Only a comma makes one in Python, so `'shape': (1)` is the *int* 1 and
  NumPy rejects it. `PyLiteral.ParseTuple` tracks `sawComma` for exactly this.
- **A header-length field is attacker-controlled.** `ReadBytes` sizes its buffer from what the stream
  actually holds (seekable ⇒ exact) and grows as bytes arrive — never from `count`. A 28-byte file
  can claim a 4 GB header, and `new byte[claim]` turns that into a gigabyte spike; NumPy shrugs it
  off in ~2 KB because `fp.read(n)` allocates only what it returns. Pinned by
  `HostileHeaderLength_DoesNotAllocateTheClaim` (measures ~1.2 KB).
- **`NpzFile`'s name map needs NumPy's TWO passes**, not one interleaved loop: `dict(zip(stripped,
  full))` then `.update(zip(full, full))`. For an archive holding both `a` and `a.npy` — which both
  strip to key `a` — pass 2 re-points `a` at the entry literally *named* `a`; one loop resolves it to
  whichever came last. Within a pass, later entries win (a zip may repeat a name, and Python's dict
  keeps the last; .NET's `GetEntry` would return the first). Pinned by the `npz_*_names` cases.
- **Never buffer an npz member to sniff its magic.** `MemoryStream`/`byte[]` cap at 2 GB, so buffering
  made NumSharp *write* 5 GiB archives it then failed to *read* ("Stream was too long") while NumPy
  read them fine — and it doubled the cost of every ordinary load. `NpyFormat.ReadArray` is strictly
  forward-only, so `StartsWithNpyMagic` sniffs on one `entry.Open()` and the reader streams from a
  fresh one. Pinned by `Npz_StreamsMembers_RatherThanBufferingThem`.

**Size limits (measured, not assumed):** `.npy` and `.npz` both round-trip **> 4 GiB** — verified at
5 GiB against every 32-bit boundary (`int.MaxValue`, `uint.MaxValue`) with NumPy reading the result,
and the 5 GiB header is byte-identical to NumPy's. Array data lives in unmanaged memory and both IO
paths stream in 256 KB chunks, so no managed buffer ever scales with the array. Zip64 comes from
.NET's `ZipArchive` on demand (NumPy forces `force_zip64=True` for the same reason, gh-10776). The
only 2 GB ceilings left are the inherently `byte[]`-returning conveniences — `np.save(arr)`,
`np.savez(...)` → `byte[]`, and `NpzFile.GetRawBytes` — which is a `byte[]` limit, not a format one;
use the path/stream overloads above 2 GB. Tests: `NpyLargeFileTests` (`[HighMemory][LongIndexing]`,
excluded from CI; needs ~8 GB RAM + ~6 GB disk).

**Gates:** `test/oracle/gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip` (286 committed cases of real
NumPy output) replayed by `NpyOracleTests` under the **`NpyOracle`** category — read / byte-exact
write / header-only write / verbatim error / round-trip / npz / live-view write / hostile-allocation.
Reverse interop (NumPy reading NumSharp, the direction byte-equality can't cover for `.npz`) is the
manual gate `python test/oracle/verify_npy_interop.py`.

### Printing / Formatting
`array2string`, `array_repr`, `array_str`, `format_float_positional`, `format_float_scientific`, `get_printoptions`, `printoptions` (IDisposable context), `set_printoptions`

`NDArray.ToString()` is a **byte-exact port of NumPy 2.4.2's array printing** (`numpy/_core/arrayprint.py` + `dragon4.c`): `ToString()` / `ToString(false)` → `np.array_str` (`[0 1 2]`, the `str()` form), `ToString(true)` → `np.array_repr` (`array([0, 1, 2], dtype=…)`, the `repr()` form). Covers decimal-point float alignment, the maxprec/unique/fixed floatmodes, exp-format cutoffs (per-dtype, native-precision ratio), nan/inf fields, complex, summarization at `threshold` (with `…` and edgeitems), line wrapping at `linewidth`, the 0-d `str`-vs-`repr` asymmetry (`5.0` vs `5.`), and repr dtype/shape suffixes. Float digit generation leans on .NET's shortest-round-trip `ToString("R")` (== Dragon4 unique) but routes **all rounding** through `ToString("F"|"E"+precision)` (rounds the true binary value, IEEE half-to-even) — never the shortest string (the latter diverges ~50 % on adversarial ties). NumSharp's `Char` dtype uses string rendering (no NumPy equivalent). Validated against NumPy 2.4.2 across ~18 000 fuzz cases.

### Other
`around`, `asscalar`, `copyto`, `round_`, `size`, `multithreading` (NumSharp extension — `np.multithreading(enabled, max_threads)` opt-in threaded kernels)

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`, unary `-`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&`, `|`, `!`

### Indexing
- Integer and slice indexing (`nd[0]`, `nd[1:3]`, `nd[::-1]`)
- Boolean masking (`nd[mask]`) — read-only
- Fancy indexing (`nd[indices]`)
- **Raw `int[]`/`long[]` as the SOLE index is FANCY** (NumPy parity): `nd[new int[]{0,2}]`
  selects rows 0 and 2 (shape `(2,…)`), NOT the element at coordinate `(0,2)`. `nd[0,2]` (separate
  scalar ints) stays coordinate (`HAS_INTEGER`). **Coordinate (element/sub-array) access moved to the
  shim `nd.GetData(int[]/long[])`** (`Backends/NDArray.cs`) — e.g. `nd.GetData(0, 2)`. Internal
  coordinate callers must use `GetData`, not `nd[coordArray]`.
  The typed `NDArray<T>.this[int[]]` is unaffected — it returns a scalar `T` (inherently coordinate).

---

## Core Source Files

| Component | Location |
|-----------|----------|
| NDArray | `Backends/NDArray.cs` |
| UnmanagedStorage | `Backends/Unmanaged/UnmanagedStorage.cs` |
| Shape | `View/Shape.cs` |
| Slice | `View/Slice.cs` |
| TensorEngine | `Backends/TensorEngine.cs` |
| DefaultEngine | `Backends/Default/DefaultEngine.*.cs` |
| np API | `APIs/np.cs` |
| Diagonal / triangular family | `Creation/np.tri.cs`, `Indexing/np.{diag,tril,diag_indices,tril_indices,fill_diagonal}.cs` |
| unique family | `Manipulation/NDArray.unique.cs` + `NDArray.unique.Kwargs.cs` (sort+mask core + axis path), `Manipulation/np.unique.cs` (`np.unique`), `Manipulation/np.unique_values.cs` (Array-API `unique_values`/`unique_counts`/`unique_inverse`/`unique_all` + result structs), `Manipulation/NDArray.unique.Hash.cs` (int + complex hash fast path, splitmix64). Design + measured perf decisions: `docs/UNIQUE_DESIGN.md` |
| Selection family | `Indexing/np.{take,take_along_axis,put,place,select}.cs`; IL kernels `Backends/Kernels/Direct/DirectILKernelGenerator.{Take,TakeAlongAxis,Put,Place,Select}.cs` (`Select` = fused single-pass reverse-`ConditionalSelect` chain; `TakeAlongAxis` = whole-array strided-odometer gather, byte-width-keyed) |
| BLAS/LAPACK seam | `Backends/IBlasBackend.cs` + `IBlasBackend.LinearAlgebra.cs` (15 default `Try*`), `Backends/TensorEngine.LinearAlgebra.cs` (virtuals + `LinAlgHelper`) |
| CBLAS product family | `LinearAlgebra/np.{inner,vdot,vecdot,matvec,vecmat,tensordot}.cs`, `LinearAlgebra/GufuncGuard.cs` |
| einsum | `LinearAlgebra/np.einsum.cs`, `LinearAlgebra/EinsumSubscripts.cs` (port of `einsum.cpp`'s parser) |
| `np.linalg` module | `LinearAlgebra/linalg/np.linalg.cs` (class + `_assert_*`/`_commonType` ports) and `np.linalg.{solve,inv,det,eig,svd,qr,cholesky,lstsq,norm,multi_dot,matrix_power,arrayapi}.cs`; `Exceptions/LinAlgError.cs` |
| Grid / slice-expression DSL | `Creation/np.r_.cs` (`AxisConcatenator` + `RClass`), `Creation/np.c_.cs`, `Creation/np.ogrid.cs` (`OGridClass` + `OGridResult` + shared `nd_grid` helpers), `Creation/np.mgrid.cs` (`MGridClass` + `MGridResult`), `Creation/np.meshgrid.cs` (`MeshgridResult`), `Indexing/np.{ix_,s_}.cs` |
| Array printing (NumPy parity) | `Backends/Printing/{PrintOptions,Dragon4,ElementFormatters,ArrayFormatter}.cs`, `APIs/np.array2string.cs`, `Casting/NdArray.ToString.cs` |
| Iterators | `Backends/Iterators/NDIter.cs`, `NDIter.Detach.cs` (ref-struct → managed-owner bridge) |
| Iteration APIs (nditer/ndindex/ndenumerate) | `APIs/np.nditer.cs`, `Indexing/np.{ndindex,ndenumerate}.cs` |
| Typed iteration (nditer&lt;T&gt;/nditer_chunks&lt;T&gt;) | `APIs/np.nditer.Typed.cs` (ref-struct enumerators + `TypedIterHelpers`) |
| Expression DSL (np.evaluate) | `Backends/Iterators/NDExpr.cs` (nodes + emission), `NDExpr.Typing.cs` (per-node NumPy result_type pass), `NDExpr.Evaluate.cs` (array leaves, binding, reductions, operators), `Backends/Default/Math/DefaultEngine.Evaluate.cs` (host) |
| ILKernelGenerator | `Backends/Kernels/ILKernelGenerator*.cs` (per-chunk, NDIter-driven) |
| DirectILKernelGenerator | `Backends/Kernels/Direct/DirectILKernelGenerator.*.cs` (whole-array, 63 partials) |
| Kernel shared infra | `Backends/Kernels/{VectorMethodCache,ScalarMethodCache,KernelOp,StrideDetector,SimdMatMul.*}.cs` |
| .npy/.npz format (NEP-01) | `IO/NpyFormat.cs` (port of NumPy's `_format_impl.py`), `IO/NpzFile.cs` (lazy archive), `IO/PyLiteral.cs` (header parser ≙ `ast.literal_eval`), `APIs/np.{save,load}.cs` |
| float32 NumPy-exact math (exp/log/sin/cos) | `Utilities/NDFloatMath.cs`, `Utilities/NDFloatMath.Simd.cs` |
| Type info | `Utilities/InfoOf.cs` |
| Generic NDArray | `Generics/NDArray\`1.cs` |

---

## Implementation Patterns

### Pattern 1: Compose existing np functions
```csharp
public static NDArray std(NDArray a, int? axis = null, ...)
{
    var mean_val = np.mean(a, axis, keepdims: true);
    return np.sqrt(np.mean(np.power(a - mean_val, 2), axis));
}
```

### Pattern 2: Delegate to TensorEngine
```csharp
public static NDArray sum(NDArray a, int? axis = null, ...)
{
    return a.TensorEngine.Sum(a, axis, typeCode, keepdims);
}
```
Use Pattern 2 when low-level optimization is needed.

## Type Switch Pattern

```csharp
switch (nd.typecode)
{
    case NPTypeCode.Boolean: return Process<bool>(nd);
    case NPTypeCode.Byte: return Process<byte>(nd);
    case NPTypeCode.SByte: return Process<sbyte>(nd);
    case NPTypeCode.Int16: return Process<short>(nd);
    case NPTypeCode.UInt16: return Process<ushort>(nd);
    case NPTypeCode.Int32: return Process<int>(nd);
    case NPTypeCode.UInt32: return Process<uint>(nd);
    case NPTypeCode.Int64: return Process<long>(nd);
    case NPTypeCode.UInt64: return Process<ulong>(nd);
    case NPTypeCode.Char: return Process<char>(nd);
    case NPTypeCode.Half: return Process<Half>(nd);
    case NPTypeCode.Single: return Process<float>(nd);
    case NPTypeCode.Double: return Process<double>(nd);
    case NPTypeCode.Decimal: return Process<decimal>(nd);
    case NPTypeCode.Complex: return Process<Complex>(nd);
    default: throw new NotSupportedException();
}
```

## GitHub Issues

Create issues on `SciSharp/NumSharp` via `gh issue create`. `GH_TOKEN` is available via the `env-tokens` skill.

### Feature / Enhancement

- **Overview**: 1-2 sentence summary of what and why
- **Problem**: What's broken or missing, why it matters
- **Proposal**: What to change, with a task checklist (`- [ ]`)
- **Evidence**: Data, benchmarks, or references supporting the proposal
- **Scope / Non-goals**: What this issue does NOT cover (prevent scope creep)
- **Benchmark / Performance** (if applicable): Before/after numbers, methodology, what to measure
- **Breaking changes** table (if any): Change | Impact | Migration
- **Related issues**: Link dependencies

### Bug Report

- **Overview**: 1-2 sentence summary of the bug and its impact
- **Reproduction**: Minimal code to trigger the bug
- **Expected**: Correct behavior (include NumPy output as source of truth)
- **Actual**: What NumSharp does instead (error message, wrong output, crash)
- **Workaround** (if any): How users can avoid the bug today
- **Root cause** (if known): File, line, why it happens
- **Related issues**: Link duplicates or upstream causes

## Performance Convention

All NumSharp-vs-NumPy benchmark ratios are reported as **NPY/NS**:

> **ratio = NumPy_ms / NumSharp_ms** — **`>1` = NumSharp FASTER, `<1` = NumSharp slower, `=1` = parity.**

**Higher is better.** Equivalently `speedup = NumPy_time / NumSharp_time`. A cell of
`0.5` means NumSharp takes 2× NumPy's time; `2.0` means NumSharp is 2× faster. Use this
direction **everywhere** — matrices, geomeans, commit messages, and the per-subsystem
`*_sheet.py` outputs (`nditer`/`layout`/`operand`/`cast`/`fusion`) — so one glance answers "are we faster?".

- Icons used in the matrices: ✅ `≥1.0` · 🟡 `≥0.5` · 🟠 `≥0.2` · 🔴 `<0.2`.
- Timing scripts MUST run `dotnet run -c Release - < script.cs` (Debug taints hand-written kernels ~2×; see `benchmark/CLAUDE.md`).
- best-of-rounds (take the min), warmup excluded, correctness checked before every timed row.
- The canonical harness is `benchmark/run_benchmark.py` — the op/dtype/N matrix plus five appended subsystems: `benchmark/nditer/` (iterator aspects), `benchmark/layout/` (reduction/copy/elementwise × memory layout × dtype), `benchmark/operand/` (1-D/scalar/mixed-operand/broadcast layouts), `benchmark/cast/` (astype src→dst matrix), `benchmark/fusion/` (np.evaluate). The `run-benchmarks.ps1` "Status Icons" table reports the *inverse* (NS/NPY, lower = better) — prefer this NPY/NS convention.
- The op/dtype/N matrix is **BenchmarkDotNet** (`benchmark/NumSharp.Benchmark.CSharp`) timed against a **warm NumPy** process (`benchmark/NumSharp.Benchmark.Python`) across **1K / 100K / 10M × all 15 dtypes** (~615 ops/size), joined on `(op, dtype, N)`. Two methodology guards beyond `-c Release`: the **InProcessEmit** toolchain (sibling `.claude/worktrees/` checkouts hold same-named projects the out-of-process toolchain refuses to build) and a **25 ms-capped 50-iteration** job (so µs–ms array ops skip BenchmarkDotNet's nanosecond invocation ramp — without it the full matrix takes days). The NDIter subsystem reports a section that crashes all retries (the known intermittent `AccessViolation`) as **NA/IGNORED**, never a failure.
- **What we commit & reference is `benchmark/history/`, not the gitignored `benchmark/results/<ts>/` raw scratch.** Every run writes a committable snapshot `benchmark/history/<date>_<sha>/` (MANIFEST + report.{md,json,csv} + numpy-results.json + all subsystem `*_results.{md,tsv}` + cards — the json/csv are gitignored at the benchmark root, so this is their only tracked home) and repoints `benchmark/history/latest` (a git symlink). Built by `benchmark/scripts/snapshot_history.py` (called by `run_benchmark.py`; `--no-history` opts out). Reference the stable `benchmark/history/latest/benchmark-report.md`. The publish ritual (run → review → commit `benchmark/history/`) is what `.github/workflows/benchmark.yml` performs post-release. See `benchmark/CLAUDE.md` → "History snapshots & the publish ritual".

## Build & Test

```bash
# Build (silent, errors only)
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
```

### Strong naming (identity, NOT authenticity)

Every assembly in the repo is strong-named with **`Open.snk` at the repo root**, configured once in
**`Directory.Build.props`** — `SignAssembly` defaults to true and any project opts out with
`<SignAssembly>false</SignAssembly>`. Public key token: **`cc7b13ffcd2ddd51`**.

**The key is Microsoft's PUBLISHED open-source key** — byte-identical to
`dotnet/arcade`'s `src/Microsoft.DotNet.Arcade.Sdk/tools/snk/Open.snk`, the same identity carried by
`netstandard`, `System.Memory` and `System.Buffers`, and the same key **TensorFlow.NET** signs with
(which is why we keep it: a NumSharp-owned key would break the cross-repo
`InternalsVisibleTo("TensorFlowNET.UnitTest")`). Its private half is public by design, so this is
assembly **identity** — version binding, telling this `NumSharp` from another — and **not** a
security property. `InternalsVisibleTo` is therefore not an access control. Real authenticity is
Authenticode + NuGet author signing, needs a code-signing certificate, and does not exist here yet.

**Consequences to know:**
- Every `InternalsVisibleTo` MUST carry `PublicKey=` — keyless is `CS1726`, a hard error. There are
  7: five in `NumSharp.Core/Assembly/Properties.cs`, two in `NumSharp.Interop.OpenBLAS/AssemblyInfo.cs`.
  The literal is held in `NumSharpFriendKey` / `BlasFriendKey` consts so it is written once per assembly.
- `dotnet run` scripts outside the repo root need signing properties to see internals — see
  "Scripting with `dotnet run`".
- Gate: `test/NumSharp.UnitTest/Assembly/StrongNameTests.cs` (6 tests) asserts the token, the full
  public key, that friend access actually resolves, and that no friend declaration is keyless.

**How it was broken before (2019 → 2026-07):** `SignAssembly` sat in a `Publish|AnyCPU`-only
PropertyGroup while CI builds and packs `Release`, so **every published NumSharp shipped unsigned**
(`PublicKeyToken=null`) — and the `Publish` configuration did not even compile, because all seven
friend declarations were keyless. A `#if !SIGNING` guard was meant to cover that, but `SIGNING` was
defined only in the *test* csproj, a different assembly, where it did nothing. Nothing asserted the
identity, so none of it surfaced. Hence the runtime gate rather than trusting the build.

### Running Tests

Tests use **MSTest v3** framework with source-generated test discovery.

```bash
# Run from test directory
cd test/NumSharp.UnitTest

# All tests (includes OpenBugs - expected failures)
dotnet test --no-build

# Exclude OpenBugs (CI-style - only real failures)
dotnet test --no-build --filter "TestCategory!=OpenBugs"

# Run ONLY OpenBugs tests
dotnet test --no-build --filter "TestCategory=OpenBugs"

# Exclude multiple categories
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"

# Run specific test class
dotnet test --no-build --filter "ClassName~BinaryOpTests"

# Run specific test method
dotnet test --no-build --filter "Name~Add_Int32_SameType"
```

### Output Formatting

```bash
# Results only (no messages, no stack traces)
dotnet test --no-build 2>&1 | grep -E "^(failed|skipped|Test run|  total:|  failed:|  succeeded:|  skipped:|  duration:)"

# Results with messages (no stack traces)
dotnet test --no-build 2>&1 | grep -v "^    at " | grep -v "^     at " | grep -v "^    ---" | grep -v "^  from K:" | sed 's/AssertFailedException: //'

# Verbose output (shows passed tests too)
dotnet test --no-build -v normal
```

## Test Categories

Tests use typed category attributes defined in `TestCategory.cs`. Adding new bug reproductions or platform-specific tests only requires the right attribute — no CI workflow changes.

| Category | Attribute | Purpose | CI Behavior |
|----------|-----------|---------|-------------|
| `OpenBugs` | `[OpenBugs]` | Known-failing bug reproductions. Remove when fixed. | **EXCLUDED** via filter |
| `Misaligned` | `[Misaligned]` | Documents NumSharp vs NumPy behavioral differences. | Runs (tests pass) |
| `WindowsOnly` | `[WindowsOnly]` | Requires GDI+/System.Drawing.Common | Excluded on non-Windows |
| `LongIndexing` | `[LongIndexing]` | Arrays with size > int.MaxValue (>2B elements) | Runs (excluded only if also HighMemory) |
| `HighMemory` | `[HighMemory]` | Requires 8GB+ RAM | **EXCLUDED** via filter |
| `LargeMemoryTest` | `[LargeMemoryTest]` | Memory-heavy non-bugs (combines OpenBugs+HighMemory) | **EXCLUDED** via filter |
| `FuzzMatrix` | `[TestCategory("FuzzMatrix")]` | NumPy differential gate — replays the committed oracle corpus bit-exact (see Differential-Fuzz Pipeline). | Runs (the gate) |

### How CI Excludes Categories

The CI pipeline (`.github/workflows/build-and-release.yml`) uses MSTest's `--filter` to exclude categories:

```yaml
- name: Test (net10.0)
  run: |
    dotnet test test/NumSharp.UnitTest/NumSharp.UnitTest.csproj \
      --configuration Release --no-build --framework net10.0 \
      --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
```

This filter excludes all tests with `[OpenBugs]` or `[HighMemory]` attributes from CI runs. Tests pass locally when the bug is fixed — then remove the `[OpenBugs]` attribute.

### Usage

```csharp
// Class-level (all tests in class)
[TestClass]
[OpenBugs]
public class BroadcastBugTests { ... }

// Method-level
[TestMethod]
[OpenBugs]
public void BroadcastWriteCorruptsData() { ... }

// Documenting behavioral differences (NOT excluded from CI)
[TestMethod]
[Misaligned]
public void BroadcastSlice_MaterializesInNumSharp() { ... }
```

### Local Filtering

```bash
# Exclude OpenBugs (same as CI)
dotnet test --filter "TestCategory!=OpenBugs"

# Run ONLY OpenBugs tests (to verify fixes)
dotnet test --filter "TestCategory=OpenBugs"

# Run ONLY Misaligned tests
dotnet test --filter "TestCategory=Misaligned"

# Combine multiple exclusions
dotnet test --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=WindowsOnly"
```

**OpenBugs files**: `OpenBugs.cs` (general), `OpenBugs.Bitmap.cs` (bitmap), `OpenBugs.ApiAudit.cs` (API audit), `OpenBugs.ILKernelBattle.cs` (IL kernel), `OpenBugs.Random.cs` (random), `OpenBugs.BroadcastReduce.cs` (broadcast reduce).

## Differential-Fuzz Pipeline (NumPy oracle)

Proves every NDIter-backed op is **bit-identical** to NumPy 2.4.2 across the input space. NumPy is the oracle: Python generates a **committed, bytes-exact corpus**; the C# harness replays the operand bytes and bit-compares — **no Python at test time or in CI**.

```
test/oracle/                          corpus generators (NumPy 2.4.2 — run by hand / nightly soak)
  layout_catalog.py                   memory-layout builders (the 40 variations: 26 single + 9 pair + 5 where)
  gen_oracle.py                       deterministic op matrices (astype/binary/unary/reduce/where/… — ~159 ops);
                                      per-mode dtype axes widened to ALL_DTYPES; Char woven into 18 tier files
                                      via the uint16 proxy (char_tier, relabelled uint16->char)
  gen_decimal_oracle.cs               INDEPENDENT C# oracle for Decimal (no NumPy analog): naive scalar
                                      System.Decimal math -> decimal_{unary,binary,reduce,scan,power,
                                      varstd,matmul,astype,stat,where,sort,manip}.jsonl
  gen_index_oracle.py                 getter/setter index oracle (token index over 15 base recipes)
  fuzz_random.py                      seeded random fuzzer (imports the other two)
  gen_npy_oracle.py                   .npy/.npz format oracle: REAL np.save/savez output + a manifest ->
                                      NumSharp.UnitTest/IO/corpus/npy_oracle.zip (286 cases)
  interop_write.cs                    the reverse direction: NumSharp writes, verify_npy_interop.py reads
  verify_npy_interop.py               manual gate — real NumPy reads NumSharp's output (needs Python + SDK)
test/NumSharp.UnitTest/Fuzz/          C# replay harness (no Python)
  FuzzCorpus.cs                       rebuilds EXACT NDArray views from (dtype,shape,strides,offset,bytes)
  OpRegistry.cs                       op-name → NumSharp call (pairs 1:1 with gen_oracle.py)
  BitDiff.cs / Shrinker.cs            bit-exact compare (NaN tokenized; Decimal by canonical VALUE so 1.0m≡1.00m) / shrink a failure to 1 element
  MisalignedRegistry.cs               the documented, excused divergences (intended differences and excused cases)
  FuzzCorpusTests.cs                  one [FuzzMatrix] test per op-corpus file (checks dtype + shape + bytes + error parity)
  IndexOracleTests.cs                 index get/set differential gate (curated + dtype + seeded-random tiers)
  MetamorphicTests.cs                 oracle-free invariants (round-trips / involutions / identities — no NumPy)
  HarnessSelfTests.cs                 proves the harness has teeth (BitDiff detects value/NaN/-0 diffs; non-vacuous)
  corpus/*.jsonl                      committed corpus (~79K cases / 48 tiers; op corpus ~65K incl. 4.4K Char woven + 695 Decimal + 2.3K specials + 100 precision/truth-bearing + 287 products + 146 random-stream), copied to test output via the csproj glob
test/NumSharp.UnitTest/IO/            .npy/.npz format gate (no Python)
  NpyOracleCorpus.cs                  opens npy_oracle.zip, rebuilds arrays from the manifest
  NpyOracleTests.cs                   one [NpyOracle] test per claim: read / byte-exact write / header-only
                                      write / verbatim error / round-trip / npz / live-view write /
                                      hostile-allocation / corpus non-vacuity
  NpyLargeFileTests.cs                the 2 GB and 4 GB walls; the 5 GiB pair is [HighMemory] (not in CI)
  AllocationTests.cs                  MinAllocated: best-of-N, since GetTotalAllocatedBytes is process-wide
  corpus/npy_oracle.zip               committed real-NumPy output (286 cases), copied via the csproj glob
  NumpyCompatibilityTests.cs          hand-made NumPy fixtures (test_compat/) — v2.0/v3.0/fortran/scalar/
  NpyFormatVersionTests.cs            empty/bool/multi/compressed. They resolve test_compat RELATIVE to the
                                      working dir and must fail loudly when it is absent: they used to
                                      `return` silently, and nothing copied the fixtures, so on a clean
                                      checkout all 13 passed while asserting NOTHING. The csproj now copies
                                      test_compat\* — do not drop that rule.
```

- **Generators live in `test/oracle/`** and write the corpus into `test/NumSharp.UnitTest/Fuzz/corpus/` (path resolved relative to `test/oracle/`). CI replays the committed corpus, never the generators.
- **Result kinds & error messages.** `expected.kind` widens the corpus past "one array": `array`
  (default — dtype+shape+bytes), `scalar` (a C# scalar wrapped 0-d), `dtype` (compared by NumPy
  dtype NAME — gates the promotion table directly), `text` (printing, verbatim) and `tuple` (N
  slots, **arity asserted**). `error: {type, text}` records NumPy's exception verbatim, so error
  cases assert the actual MESSAGE, not merely that something was thrown (the legacy
  `expects_throw`-only cases keep the weaker assertion). Three tiers use them: **`iter.jsonl`**
  (4,605 — materialized `ndindex`/`ndenumerate`/`nditer`/`broadcast` TRACES: the value stream,
  multi_index, c_index/f_index and external_loop chunk lengths across order C/F/A/K × 22 layouts;
  traversal ORDER has no other gate, and every other tier depends on it), **`dtype_text.jsonl`**
  (2,111 — `promote_types`/`result_type`/`min_scalar_type`/`can_cast`/`isscalar`/`iscomplexobj`/
  `isrealobj`/`size`/`array_str`/`array_repr`/`nonzero` at any rank) and **`errors_full.jsonl`**
  (688 raising cells over 22 distinct NumPy messages — the cells every value generator SKIPS) and
  **`out_where.jsonl`** (3,727 — ufunc `out=`/`where=` over 7 out layouts × 9 mask layouts × 28
  ufuncs; each case records BOTH the returned view AND the entire base buffer behind `out`, so
  masked-off slots are proven to keep their prior contents and a kernel writing outside a
  strided/offset/negstride window is caught — previously reached only through `maximum_out`/
  `minimum_out`/`clip_out`, 11 contiguous unmasked cases each).
  A fourth, **** (3,727), gates ufunc / across 7 out layouts x 9 mask
  layouts x 28 ufuncs: each case records BOTH the returned view and the ENTIRE base buffer behind
  , so masked-off slots are proven to retain their prior contents and a kernel writing outside
  a strided/offset/negstride view window is caught (previously reached only via maximum_out/
  minimum_out/clip_out — 11 contiguous, unmasked cases each).
  Comparators: `FuzzCorpusTests.Kinds.cs`; callables: `OpRegistry.Kinds.cs`; ledger: `Fuzz/README.md`
  → Table 0.
- **Three `FuzzMatrix` gates**: `FuzzCorpusTests` (the op corpus — ~65K cases across the tiers, checking dtype + shape + bytes + error parity + per-file minimum-count floors; Char woven into 18 tier files, 12 `Decimal*` tiers: unary/binary/reduce/scan/power/varstd/matmul/astype/stat/where/sort/manip), `IndexOracleTests` (the index oracle — `index_curated` 2,273 + `index_dtype` 104 + `index_setter_dtype` 10 (cross-dtype cast-on-set) + `index_random` 10,000; the advanced-indexing parity gate), and `MetamorphicTests` (12 NumPy-free invariants incl. Half/Complex/Decimal/bool/char + strided views). A failing op case auto-shrinks to a 1-element repro.
- **Dtype coverage**: per-mode dtype axes widened toward `ALL_DTYPES`. **Char** (no NumPy dtype) is woven into 18 tier files via the uint16 proxy (`gen_oracle.char_tier`, relabelled uint16→char). **Decimal** (no NumPy analog) rides an independent C# oracle (`gen_decimal_oracle.cs`, naive scalar `System.Decimal`; incl. axis reductions, empty, negative int powers). Verified Char/clip-bool/round/dot bugs are carved from the green corpus and reproduced under `[OpenBugs]` (`OpenBugs.Char.cs`, `OpenBugs.DtypeCoverage.cs`, `OpenBugs.FuzzGaps.cs`) — NOT excused in `MisalignedRegistry`.
- **Regenerate** (deterministic; needs `numpy==2.4.2`): `python test/oracle/gen_oracle.py <mode>` (modes: `smoke astype_full binary divmod_power comparison unary reduce where place matmul rounding bitwise unary_extra nanreduce scan stat logic modf manip sort tail params aliasing copyto errors groupa specials precision products random_parity`; `precision` additionally needs `mpmath`) + `python test/oracle/gen_index_oracle.py` (the `index_*` tiers) + `python test/oracle/fuzz_random.py 1234 2000 random_smoke.jsonl` + `dotnet run test/oracle/gen_decimal_oracle.cs` (the `decimal_*` tiers), then `dotnet build` (copies the corpus to test output).
- **Truthful vs precise** (`precision.jsonl`, 72 truth-bearing cases): each case carries `expected.truth` — the correctly-rounded mathematical reference (exact `Fraction` / 200-bit mpmath, generator-side only) — over precision-ADVERSARIAL inputs (wide-magnitude/cancellation sums at N≤2049, large-mean variance, near-1 products, expm1/log1p small-|x|) the ordinary 8–36-element pools cannot express. Policy (the vision is byte-identical NumPy parity): **bit-exact to NumPy passes without truth ever being read — precise never fails**; truth only adjudicates divergences. Not-less-truthful than NumPy → excused "prefer-precise" parity debt (being MORE accurate than NumPy is still a divergence to close by porting NumPy's algorithm, never a win); less truthful beyond 4×/+8 ULP slack → precision LOSS, red unless a bounded known-bug branch covers it (`MisalignedRegistry` P1–P3; the unbounded summation blanket is gated on truth-absence so losses can't hide in it). Findings on arrival, excused bounded ≤256 ULP (P3): f32 var/std accumulation 55/26 ULP vs truth (NumPy 3/2), negative-stride reduce path 11–32 ULP (NumPy exact).
- **Products & random streams**: `products.jsonl` (287) is the FIRST value gate for `inner`/`vdot`/`vecdot`/`matvec`/`vecmat`/`tensordot`/`linalg.multi_dot`/`linalg.matrix_power` — small-exact operands across all 13 dtypes (bit-parity holds even vs NumPy's BLAS since depth-4 float sums are exact; conjugation + loop-dtype rules gated) plus deep-K truth-bearing f32/f64 cases adjudicated prefer-precise. `random_parity.jsonl` (38, portable MT19937/uniform-arithmetic dists, hard everywhere) + `random_parity_host.jsonl` (108, libm-consuming transform/rejection samplers — win-amd64-authored, Inconclusive off-Windows like matmul_parity) pin the documented byte-identical seed claim the statistical tests couldn't. Findings: 8 samplers carved + pinned under `OpenBugsRandom.RandomParity_*` (gamma shape<1 via 2-arg, f, pareto, standard_cauchy, binomial, negative_binomial, multinomial, multivariate_normal); chisquare/wald/noncentral_f/dirichlet ride a scoped ≤8-ULP (32 wald) R1 envelope; C-long int outputs recorded widened to NumSharp's fixed int64.
- **Run the gate**: `dotnet test --filter "TestCategory=FuzzMatrix"`. Each case is bit-exact (pass), a documented difference in `MisalignedRegistry` (excused, never silent), or a failure (red). Full divergence ledger: `test/NumSharp.UnitTest/Fuzz/README.md`.

### The `.npy`/`.npz` format oracle (same philosophy, separate corpus)

Same shape as above — NumPy is the oracle, the corpus is committed, no Python at test time — but the
claim is stronger: NumSharp's writer must be **byte-identical** to `np.save`, not merely readable.

- **Regenerate**: `python test/oracle/gen_npy_oracle.py` (deterministic; needs `numpy==2.4.2`), then `dotnet build`.
- **Run the gate**: `dotnet test --filter "TestCategory=NpyOracle"` — 286 cases across every dtype ×
  {0-d, empty, 1-D, 2-D, 3-D, unit, empty-2d/3d} × {C, F, strided, reversed, offset, broadcast,
  transposed} × versions {1.0, 2.0, 3.0} × {little, big, native} endian, plus value fidelity (NaN
  payloads incl. sNaN, subnormals, signed zero, integer extremes, BMP seams) and 42
  malformed / unsupported / hostile files whose messages must match NumPy verbatim.
- **`kind: "header"` cases exist because two writer branches are unreachable from a real array.** The
  growth padding is normally INVISIBLE — shrink the body by 5 chars and the alignment padding grows by
  5, leaving the file identical — so a wrong growth axis (`shape[0]` vs `shape[-1]`) passes every
  ordinary test; it only shows on shapes that tip the header across a 64-byte bucket, and those need
  10¹⁷ elements to allocate. The v1.0→v2.0 auto-selection boundary likewise sits at 21,817 dimensions.
  Both are driven through the header dict, exactly as NumPy's `_write_array_header` was.
- **Reverse interop** (manual; needs Python + the SDK): `python test/oracle/verify_npy_interop.py` has
  NumSharp write 28 files and real NumPy read them. This is the only way to prove `.npz` output,
  whose ZIP framing legitimately differs from Python's `zipfile` and so cannot be byte-compared.

## CI Pipeline

`.github/workflows/build-and-release.yml` — test on 3 OSes (Windows/Ubuntu/macOS), build NuGet on tag push, create GitHub Release, publish to nuget.org. The `FuzzMatrix` gate runs here (replays the committed corpora; no Python).

`.github/workflows/fuzz-soak.yml` — nightly soak: sweeps seeds through `test/oracle/fuzz_random.py` (~1M fresh cases/night), replays them, and uploads any failing corpus; copy a shrunk repro into `Fuzz/corpus/regressions/` to pin it on every CI thereafter.

`.github/workflows/benchmark.yml` — decoupled post-release perf run (triggers on a published Release or manual dispatch — a slow/failed benchmark must never gate a release). Runs the whole `benchmark/run_benchmark.py` harness (op/dtype/N matrix + the five subsystems), renders the DocFX benchmark pages, and commits the refreshed report + cards + a `benchmark/history/<date>_<sha>/` snapshot (+ the `latest` symlink) to master with `[skip ci]`.

## Scripting with `dotnet run` (.NET 10 file-based apps)

### Accessing Internal Members

NumSharp has many key types/fields/methods marked `internal` (Shape.dimensions, Shape.strides, NDArray.Storage, np._FindCommonType, etc.). To access them from a `dotnet run` script, override the assembly name to match an existing `InternalsVisibleTo` entry — **and sign the script**:

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
```

**How it works:** NumSharp declares `[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript, PublicKey=…")]` in `src/NumSharp.Core/Assembly/Properties.cs`. The `#:property AssemblyName=NumSharp.DotNetRunScript` directive overrides the script's assembly name (which normally derives from the filename) to match, granting full access to all `internal` and `protected internal` members.

**Why the two signing properties.** NumSharp is strong-named (repo-root `Directory.Build.props`), so its friend declarations name a `PublicKey` — and a keyed `InternalsVisibleTo` matches ONLY an assembly signed with that key. An unsigned script with the right *name* is not a match, and the failure is a compile error on the first internal member, not a warning.

A script living **under the repo root** needs neither property: `Directory.Build.props` is an implicit file for file-based apps, so it inherits signing automatically (this is why `test/oracle/*.cs` just works). A script run from **anywhere else** — the scratchpad, a temp dir — inherits nothing and must pass both properties itself. Use an absolute path for the key.

### Accessing Unsafe code
NumSharp uses unsafe in many places, hence include `#:property AllowUnsafeBlocks=true` in scripts.

### Script Template (copy-paste ready)

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
```

(The last two are required only for scripts OUTSIDE the repo root — see above.)

### Key Internal Members Available

| Member | What it exposes |
|--------|----------------|
| `shape.dimensions` | Raw long[] of dimension sizes |
| `shape.strides` | Raw long[] of stride values |
| `shape.size` | Internal field: total element count |
| `shape.offset` | Base offset into storage (NumPy-aligned) |
| `shape.bufferSize` | Size of underlying buffer |
| `shape._flags` | Cached ArrayFlags bitmask |
| `shape.IsWriteable` | False for broadcast views (NumPy behavior) |
| `shape.IsBroadcasted` | Has any stride=0 with dimension > 1 |
| `shape.IsSimpleSlice` | IsSliced && !IsBroadcasted |
| `shape.OriginalSize` | Product of non-broadcast dimensions |
| `arr.Storage` | Underlying `UnmanagedStorage` |
| `arr.GetTypeCode` | `NPTypeCode` of the array |
| `arr.Array` | `IArraySlice` — raw data access |
| `np._FindCommonType(...)` | Type promotion logic |
| `np.powerOrder` | Type promotion ordering |
| `NPTypeCode.GetGroup()` | Type category (int/uint/float/etc.) |
| `NPTypeCode.GetPriority()` | Type priority for promotion |
| `NPTypeCode.AsNumpyDtypeName()` | NumPy dtype name (e.g. "int32") |
| `Shape.NewScalar()` | Create scalar shapes |

### Common Public NDArray Properties

| Property | Description |
|----------|-------------|
| `nd.shape` | Dimensions as `int[]` |
| `nd.ndim` | Number of dimensions |
| `nd.size` | Total element count |
| `nd.dtype` | Element type as `Type` |
| `nd.typecode` | Element type as `NPTypeCode` |
| `nd.T` | Transpose (swaps axes) |
| `nd.flat` | 1D iterator over elements |

## Adding New Features

1. Read NumPy docs for the function
2. **Run actual Python code** to observe exact behavior and fuzzy all possible inputs to define a behavior matrix.
3. Check existing similar implementations
4. Implement behavior matching exactly that of numpy.
5. Write tests based on observed NumPy output
6. Handle all 15 dtypes

---

## Q&A - Design & Architecture

**Q: Why unmanaged memory instead of Span<T>/Memory<T>?**
A: Benchmarking showed unmanaged memory was fastest. NDArray is self-managed memory allocation optimized for performance.

**Q: Why are there TWO classes — `ILKernelGenerator` AND `DirectILKernelGenerator`?**
A: They encode two different kernel-driving contracts. `DirectILKernelGenerator` (63 partials in `Direct/`) emits whole-array kernels: one call processes the entire array; the kernel walks dimensions/strides itself. `ILKernelGenerator` (root) emits per-chunk kernels matching NumPy's `PyUFuncGenericFunction` contract: the iterator (`NDIterRef`) drives the loop and the kernel only processes one chunk. The split makes the contract explicit in the type name. A kernel lives in whichever class matches how it is driven — `ILKernelGenerator` for kernels run as an NDIter inner loop (`np.where`, flat/pairwise reductions), `DirectILKernelGenerator` for kernels that own their full-array traversal.

**Q: When should I write kernels in `ILKernelGenerator` vs `DirectILKernelGenerator`?**
A: Match the driving contract. A kernel that runs as the inner loop of an NDIter pass (the iterator advances the operand pointers, the kernel handles one chunk) belongs in `ILKernelGenerator`. A kernel that takes the whole array plus its shape/strides and walks it itself belongs in `DirectILKernelGenerator`.

**Q: Why is TensorEngine abstracted?**
A: So alternative backends (GPU/CUDA, SIMD intrinsics, MKL/BLAS) can plug in behind one contract; `DefaultEngine` (pure C#) is the engine.

**Q: How closely does the API match NumPy?**
A: Goal is as close as possible - all edge cases included (NaN handling, multi-type operations, broadcasting). Target is NumPy 2.x.

**Q: Does np.random match NumPy's random state/seed behavior?**
A: Yes, 1-to-1 matching. The bit generator is **MT19937** (NumPy's Mersenne Twister, `RandomSampling/MT19937.cs`) with Marsaglia-polar Gaussian + cached-Gaussian carry and NumPy's state-tuple `get_state`/`set_state` format, so a given seed produces byte-identical sequences to NumPy 2.4.2.

**Q: What are the primary use cases?**
A: Anything that can use the capabilities - porting Python ML code, standalone .NET scientific computing, integration with TensorFlow.NET/ML.NET.

**Q: Which subsystem is the most intricate?**
A: Slicing/broadcasting — offset/stride calculations with contiguity detection. The `readonly struct Shape` with cached `ArrayFlags` centralizes it behind O(1) flag checks.

**Q: How is NumPy compatibility validated?**
A: Written by hand based on NumPy docs and original tests. Testing philosophy: run actual NumPy code, observe output, replicate 1-to-1 in C#.

**Q: What's the pattern for adding new np.* functions?**
A: Sometimes uses other np functions (no DefaultEngine needed). Sometimes requires DefaultEngine for optimization. Tests should be based on actually running NumPy code and imitating the outcome.

**Q: Are breaking changes acceptable?**
A: Yes - breaking changes are accepted to align with NumPy 2.x behavior.

---

## Q&A - Core Components

**Q: What are the three pillars of NumSharp?**
A: `NDArray` (user-facing API), `UnmanagedStorage` (raw memory management), and `Shape` (dimensions, strides, coordinate translation). They work together: NDArray wraps Storage which uses Shape for offset calculations.

**Q: What is Shape responsible for?**
A: Shape is a `readonly struct` containing dimensions, strides, offset, bufferSize, and cached `ArrayFlags`. Key properties: `IsScalar`, `IsContiguous`, `IsSliced`, `IsBroadcasted`, `IsWriteable`, `IsSimpleSlice`. Methods: `GetOffset(coords)`, `GetCoordinates(offset)`. NumPy-aligned: broadcast views are read-only (`IsWriteable = false`).

**Q: How does slicing work internally?**
A: The `Slice` class parses Python notation (e.g., "1:5:2") into `Start`, `Stop`, `Step`. It converts to `SliceDef` (absolute indices) for computation. `SliceDef.Merge()` handles recursive slicing (slice of a slice).

**Q: What are the special Slice instances?**
A: `Slice.All` (`:` - all elements), `Slice.Ellipsis` (`...` - fill dimensions), `Slice.NewAxis` (insert dimension), `Slice.Index(n)` (single element, reduces dimensionality).

**Q: What is NDIter?**
A: The NumPy-aligned multi-operand iterator. It handles C/F/A/K order, broadcasting, external loops, buffering, casting, masks, reductions, and synchronized traversal for copy and elementwise kernels. Copy and multi-operand execution go through `NDIter.Copy` and the multi-operand iterator.

**Q: How does broadcasting work?**
A: Shapes align from the right. Dimensions must be equal OR one must be 1. Dimension of 1 "stretches" to match. Implemented via `DefaultEngine.Broadcast()` which resolves compatible shapes.

**Q: What is InfoOf<T>?**
A: Static type information cache to avoid runtime reflection. Provides `InfoOf<T>.Size` (bytes), `InfoOf<T>.NPTypeCode`, `InfoOf<T>.Zero`, `InfoOf<T>.MaxValue/MinValue`.

**Q: What is NDArray<T>?**
A: Generic typed wrapper providing type-safe access. Returns `T` from indexer instead of NDArray. Has typed `Address` pointer (`T*`) and `Array` property (`ArraySlice<T>`).

**Q: When does DefaultEngine use parallelization?**
A: Parallelization is minimal. Most operations use SIMD vectorization instead for performance.

---

## Q&A - Operations & Operators

**Q: How do arithmetic operators work?**
A: All operators (`+`, `-`, `*`, `/`, `%`, unary `-`) are defined in `NDArray.Primitive.cs`. They delegate to `TensorEngine.Add()`, `Subtract()`, etc. Scalar operands are wrapped via `NDArray.Scalar()`.

**Q: How do comparison operators work?**
A: Element-wise comparisons (`==`, `!=`, `>`, `<`, etc.) return `NDArray<bool>`. Defined in `NDArray.Equals.cs`, `NDArray.Greater.cs`, etc. Support broadcasting.

**Q: What indexing modes are supported?**
A: Integer indices, string slices (`"1:3, :"`), Slice objects, boolean masks, fancy indexing (NDArray<int> indices), and mixed combinations. All in `Selection/NDArray.Indexing*.cs`.

**Q: How is linear algebra implemented?**
A: Core ops (`dot`, `matmul`, `outer`) in `LinearAlgebra/`; `trace`/`diagonal` in `Indexing/`. The rest of what NumPy sends to OpenBLAS — the product family (`inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`) and the whole `np.linalg` module — sits in `LinearAlgebra/` and `LinearAlgebra/linalg/`, dispatching through `TensorEngine.Blas`. The products compute with managed kernels; the LAPACK factorisations raise `NotSupportedException` until a backend is installed. See "The BLAS/LAPACK API surface" above.

**Q: Why does `np.linalg.inv` throw `NotSupportedException` when `np.matmul` works fine without a backend?**
A: Because there is a managed matrix product and there is no managed SVD. `IBlasBackend`'s invariant — installing a backend changes WHICH implementation runs, never WHETHER an answer exists — holds for every product NumSharp offers, since a portable GEMM is a hundred lines. It cannot hold for a factorisation: a portable `gesdd`, `geev` or `gelsd` is a library in its own right, and NumSharp.Core carries none. The API, its NumPy-exact validation and its verbatim error messages all exist and are tested; only the numerics wait. The exception names the NumPy API, the LAPACK routine and the `IBlasBackend.Try*` member to implement, so it reads as an install instruction.

---

## Q&A - Development

**Q: What's in the test suite?**
A: MSTest v3 framework in `test/NumSharp.UnitTest/`. Many tests adapted from NumPy's own test suite, plus the differential-fuzz corpora. Broad coverage across operations, dtypes, and edge cases. Uses source-generated test discovery (no special flags needed).

**Q: What .NET version is targeted?**
A: Library multi-targets `net8.0` and `net10.0`. Tests also multi-target both frameworks.

**Q: What are the main dependencies?**
A: No external runtime dependencies. `System.Memory` and `System.Runtime.CompilerServices.Unsafe` are built into the .NET 8+ runtime.

**Q: What projects use NumSharp?**
A: TensorFlow.NET, ML.NET integrations, Gym.NET, Pandas.NET, and various scientific computing projects.

**Q: Can I save/load NumPy files?**
A: Yes, and byte-exactly. `np.save` writes `.npy`, `np.savez`/`np.savez_compressed` write `.npz`, and `np.load` reads both (returning `object`; `np.load_npy`/`np.load_npz` are the typed forms). `IO/NpyFormat.cs` is a port of NumPy 2.4.2's `_format_impl.py`, so a saved file is byte-for-byte what NumPy's own `np.save` would produce — versions 1.0/2.0/3.0, 64-byte alignment, `fortran_order`, and big-endian reads included. See the File I/O section above for the dtype map and the gates.

**Q: Why does `np.load` return `object` instead of `NDArray`?**
A: Because NumPy's does: `np.load` yields an `ndarray` for a `.npy` and an `NpzFile` for a `.npz`, dispatching on the file's magic bytes rather than its extension, and C# has no union type to express that. `np.load_npy` and `np.load_npz` are the typed escapes — prefer them when the kind is known; each also reports a directed error if handed the other kind.

**Q: What random distributions are supported?**
A: An extensive NumPy-matching set (40+ distributions and samplers — see the Supported np.* APIs → Random list), all on the `NumPyRandom` class in `RandomSampling/`, with 1-to-1 seed/state parity to NumPy.

---

## Detailed Documentation

@ARCHITECTURE.md for comprehensive technical details and @CONTRIBUTING.md for development workflow.
