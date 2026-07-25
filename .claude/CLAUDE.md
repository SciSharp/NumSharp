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
    internal readonly int _flags;        // Cached ArrayFlags bitmask
    internal readonly int _hashCode;     // Precomputed hash code
    internal readonly int size;          // Total element count
    internal readonly int[] dimensions;  // Dimension sizes
    internal readonly int[] strides;     // Stride values (0 = broadcast dimension)
    internal readonly int bufferSize;    // Size of underlying buffer
    internal readonly int offset;        // Base offset into storage
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
`arange`, `array`, `asanyarray`, `asarray`, `asarray_chkfinite`, `ascontiguousarray`, `asfortranarray`, `asmatrix`, `copy`, `empty`, `empty_like`, `eye`, `frombuffer`, `full`, `full_like`, `identity`, `linspace`, `meshgrid`, `mgrid`, `ones`, `ones_like`, `require`, `tri`, `zeros`, `zeros_like`

The `as*` conversion family mirrors NumPy: `asarray_chkfinite(a, dtype=None, order='K')` = `asarray` then raise `ValueError("array must not contain infs or NaNs")` if a **float-family** dtype (Half/Single/Double/Complex — NumPy's `typecodes['AllFloat']`; Decimal/int/bool skip the check) holds any inf/NaN, via a **fused single-pass NaN-poison SIMD reduction** (`Backends/Kernels/FiniteScan.cs`: `acc += v - v` — +0 for finite, absorbing-NaN for non-finite; AVX2 gather + reversed-contiguous fast path for strided/negative-stride views; ~2–27× NumPy contiguous, ≥1× strided). `require(a, dtype=None, requirements=None)` parses C/F/A/W/O/E flags (+aliases; single-string requirements iterate by char like NumPy, so `"F_CONTIGUOUS"` as one string raises), resolves an order and copies only if a remaining ALIGNED/WRITEABLE/OWNDATA flag is unsatisfied (ALIGNED is always true in NumSharp, so only broadcast-non-writeable and views force a copy). `asmatrix(data, dtype=None)` returns a **2-D view** (NumSharp has no `matrix` subclass — the deprecated NumPy one; no `*`-as-matmul/`.H`/`.I`): 0-D→(1,1), 1-D→(1,N), 2-D unchanged, >2-D drops length-1 axes and must land on 2-D else `ValueError("shape too large to be a matrix.")`; also parses matrix strings (`"1 2; 3 4"`). See `Creation/np.{asarray_chkfinite,require,asmatrix}.cs`.

### Shape Manipulation
`append`, `array_split`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `block`, `c_`, `column_stack`, `concat`, `concatenate`, `delete`, `dsplit`, `dstack`, `expand_dims`, `flatten`, `flip`, `fliplr`, `flipud`, `hsplit`, `hstack`, `insert`, `matrix_transpose`, `moveaxis`, `pad`, `permute_dims`, `r_`, `ravel`, `repeat`, `reshape`, `resize`, `roll`, `rollaxis`, `rot90`, `split`, `squeeze`, `stack`, `swapaxes`, `tile`, `transpose`, `trim_zeros`, `unique`, `unstack`, `vsplit`, `vstack`

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

**`np.exp` at float32 is BIT-EXACT with NumPy**, not merely close: `Utilities/NDFloatMath.cs` is a port of
`simd_exp_FLOAT` — the kernel NumPy 2.4.2 actually runs (`numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`,
the `SIMD_AVX2_FMA3` instantiation; on Windows the AVX-512 variant is `!defined(_MSC_VER)`-gated and never
compiled). Cody-Waite range reduction `y = x - k·ln2` with `k = rint(x·log2 e)`, a 5th-over-2nd-order Remez
minimax ratio, then scale by `2^k`. It is deliberately NOT correctly rounded — NumPy documents 2.52 ULP of its
own error at `x = 0xc2781e37` — so `MathF.Exp` (the previous kernel, ~correctly rounded) differed from NumPy on
**~35% of all float32 inputs** by 1-2 ULP, which the blanket `unary ~ULP` fuzz excuse was silently absorbing.
Verified **exhaustively: all 2³² float32 bit patterns** agree with NumPy 2.4.2, through both the SIMD kernel and
the scalar entry point (chunked-checksum sweep; the corpus tier `exp_f32.jsonl` pins the discriminating inputs in
CI, and `MisalignedRegistry`'s unary-ULP branch now EXCLUDES exp/Single so a 1-ULP regression turns the gate red).
Specials follow from NumPy blanking the NaN lanes before the range compares: any NaN — quiet, signalling, either
sign, any payload — returns the canonical `0x7fc00000` (note .NET's own `float.NaN` is `0xffc00000`);
`x ≥ 88.72283935546875` (incl. `+inf`) → `+inf`; `x ≤ -103.97208404541015625` (incl. `-inf`) → `+0`; the
`[-104, -87.3]` band produces subnormals through NumPy's `fma_scalef_ps` split. NumPy also raises the FP
overflow/underflow status flags (a `RuntimeWarning`); NumSharp models no FP status word — pre-existing, not
introduced here. **Host pin:** the quadrant's `mul`+`add` is FUSED because MSVC 19.44 (the numpy==2.4.2 win-amd64
wheel's compiler) contracted it — observable at `x = 0xc26d0e6c`, where `x·log2 e` lands on the exact tie `-85.5`
and the two spellings round `k` to -85 vs -86, a 1-ULP difference in the result. Same pin class as the
MSVC-pinned cast kernels (`Fuzz/README.md` → "Host-dependent values"). **Perf (NPY/NS, Release, best-of-25 warm,
`out=`):** 10M **1.14×**, 100K **1.22×**, 1K 0.69× — i.e. NumSharp is now FASTER than NumPy on the sizes where
throughput dominates, having been 0.33× before. The port is 2.4-3.4× faster than the `MathF.Exp` kernel it
replaces because it vectorizes: `Utilities/NDFloatMath.Simd.cs` carries `Vector{128,256,512}<float>` overloads
(the algorithm is purely elementwise, so lane `i` is bit-identical to the scalar call), gated by
`DirectILKernelGenerator.ExpVectorSimdAvailable` on hardware FMA — exactly as NumPy gates its own. The 1K cell is
fixed per-call dispatch overhead (~0.8 µs), not throughput: per element it is 1.30 ns at N=1K vs 0.50 ns at
N=100K. NumPy's whole-vector denormal branch is KEPT rather than folded into a branchless select — measured, the
extra `vdivps` on every vector cost ~0.17 ns/element, a fifth of the kernel.

### Math — Reductions
`all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `average`, `average_returned`, `count_nonzero`, `cumprod`, `cumsum`, `diff`, `ediff1d`, `max`, `mean`, `median`, `min`, `percentile`, `prod`, `ptp`, `quantile`, `std`, `sum`, `var`

### Math — NaN-Aware
`nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanpercentile`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar`

### Bitwise
`bitwise_and`, `bitwise_or`, `bitwise_xor` (ufunc `out=`/`where=`/`dtype=` supported; float/complex/decimal INPUTS raise NumPy's coercion TypeError while a float/complex/decimal `dtype=` raises the no-loop text — distinct messages, both probed; probed order: bad `where` → no-loop → out-cast → shape), `invert`, `left_shift`, `right_shift`

### Comparison & Logic
`all`, `allclose`, `any`, `array_equal`, `equal`, `fmax`, `fmin`, `greater`, `greater_equal`, `isclose`, `iscomplex`, `iscomplexobj`, `isfinite`, `isinf`, `isnan`, `isreal`, `isrealobj`, `isscalar`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `maximum`, `minimum`, `not_equal`

The six comparisons and `isnan`/`isfinite`/`isinf` expose **ONE NumPy-shaped overload each** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` (no bare/out split). It returns plain `NDArray` — NumPy's `np.less(a, b, out=f64)` returns the f64 out itself; `True→1` at any numeric out dtype since bool casts same_kind to all of them. A plain call still returns an `NDArray<bool>` *instance* (TensorEngine contract), so the typed wrapper is one zero-alloc cast away and the C# comparison operators (`==`, `<`, …) keep the `NDArray<bool>` static type via `AsGeneric<bool>()`. `dtype=` is validate-only (probed 2.4.2): bool loops only — `dtype: Boolean` is a no-op, anything else raises `No loop matching the specified signature and casting was found for ufunc <name>`. Comparisons compare at `result_type(lhs, rhs)` inside the kernel (probed: `greater(i8 2^53+1, f8 2^53)` → False, `equal` → True). Engine members follow the house order `(inputs, typeCode, out, where)`.

### Type Promotion & Dtype
`can_cast`, `common_type`, `find_common_type`, `finfo`, `iinfo`, `issubdtype`, `min_scalar_type`, `mintypecode`, `promote_types`, `result_type`

### Selection
`compress`, `extract`, `index_exp`, `indices`, `ix_`, `place`, `put`, `ravel_multi_index`, `s_`, `take`, `unravel_index`, `where`

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

### Iteration
`nditer`, `ndenumerate`, `ndindex` (plus the pre-existing `broadcast`), and the NumSharp-extension
typed forms `nditer<T>` / `nditer_chunks<T>`

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
`argmax`, `argmin`, `argsort`, `argwhere`, `flatnonzero`, `nonzero`, `searchsorted`, `sort`

### Linear Algebra
`diag`, `diagflat`, `diag_indices`, `diag_indices_from`, `diagonal`, `dot`, `fill_diagonal`, `mask_indices`, `matmul`, `outer`, `trace`, `tril`, `tril_indices`, `tril_indices_from`, `triu`, `triu_indices`, `triu_indices_from`

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
`around`, `asscalar`, `copyto`, `round_`, `size`, `multithreading` (NumSharp extension — `np.multithreading(enabled, max_threads)` opt-in threaded kernels), `parity_matmul` (NumSharp extension — opt-in **byte-identical** `np.dot`/`np.matmul`)

`np.parity_matmul(enabled, library = null, threads = 0)` makes `np.dot` / `np.matmul` reproduce NumPy's float32/float64 results **BIT-FOR-BIT**, by P/Invoking the *same CBLAS binary NumPy calls* through a route-for-route port of NumPy's two matrix-product dispatchers. **Off by default; NumSharp's SIMD GEMM is untouched and stays the fast path.** It exists because no portable algorithm can match NumPy here: for f32/f64 mat@mat NumPy always calls cblas (since gh-23588 it copies non-blasable operands into a temp rather than take the portable loop), and scipy-openblas' `sgemm` uses an arch-specific **multi-accumulator** scheme matching neither a sequential mul+add chain nor a sequential FMA chain — NumSharp's own GEMM differed on **94.5 %** of a `(128,784)@(784,128)` f32 product (max ~976 K ULP) and **45 %** at K=10. Both dispatchers are mirrored because they are different C code and **disagree** when an operand is not blasable (a stride-2 matrix @ vector: `np.dot` takes gemv-on-a-copy, `np.matmul` the portable loop — 278/300 elements differ): `Dot` → `cblas_matrixproduct` + the N-D `dotfunc` tail (which is NOT gemm), `MultiplyMatrix` → `@TYPE@_matmul`'s five routes incl. the `a @ a.T` **syrk** shortcut and NumPy's copy/transpose rules. Strides are ported in ELEMENTS (NumPy's are bytes — same logic with `itemsize == 1`). Scope is Single/Double; every other dtype keeps the native path (integer products are bit-exact by construction — modular addition is associative). **Three load-bearing details:** the result bits depend on the BLAS **thread count** (1/2/4/24 threads give four different answers), a **named library is binding** (an explicit path or `NUMSHARP_PARITY_BLAS` is never silently substituted — parity is a claim about one binary), and the parity path is *faster* than the native GEMM anyway (1.7–13×; 2048³ f64 293 ms vs 3860 ms). Gate: the **host-pinned** `matmul_parity` corpus tier (342 cases — 342 bit-exact on, 294 divergent off) which goes `Inconclusive`, never red, on a host that cannot load the pinned library. See `docs/GEMM_PARITY.md`, `Backends/Kernels/Blas/`, issue #626.

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
| Grid / slice-expression DSL | `Creation/np.r_.cs` (`AxisConcatenator` + `RClass`), `Creation/np.c_.cs`, `Indexing/np.{ix_,s_}.cs` |
| Array printing (NumPy parity) | `Backends/Printing/{PrintOptions,Dragon4,ElementFormatters,ArrayFormatter}.cs`, `APIs/np.array2string.cs`, `Casting/NdArray.ToString.cs` |
| Iterators | `Backends/Iterators/NDIter.cs`, `NDIter.Detach.cs` (ref-struct → managed-owner bridge) |
| Iteration APIs (nditer/ndindex/ndenumerate) | `APIs/np.nditer.cs`, `Indexing/np.{ndindex,ndenumerate}.cs` |
| Typed iteration (nditer&lt;T&gt;/nditer_chunks&lt;T&gt;) | `APIs/np.nditer.Typed.cs` (ref-struct enumerators + `TypedIterHelpers`) |
| Expression DSL (np.evaluate) | `Backends/Iterators/NDExpr.cs` (nodes + emission), `NDExpr.Typing.cs` (per-node NumPy result_type pass), `NDExpr.Evaluate.cs` (array leaves, binding, reductions, operators), `Backends/Default/Math/DefaultEngine.Evaluate.cs` (host) |
| ILKernelGenerator | `Backends/Kernels/ILKernelGenerator*.cs` (per-chunk, NDIter-driven) |
| DirectILKernelGenerator | `Backends/Kernels/Direct/DirectILKernelGenerator.*.cs` (whole-array, 63 partials) |
| Kernel shared infra | `Backends/Kernels/{VectorMethodCache,ScalarMethodCache,KernelOp,StrideDetector,SimdMatMul.*}.cs` |
| .npy/.npz format (NEP-01) | `IO/NpyFormat.cs` (port of NumPy's `_format_impl.py`), `IO/NpzFile.cs` (lazy archive), `IO/PyLiteral.cs` (header parser ≙ `ast.literal_eval`), `APIs/np.{save,load}.cs` |
| float32 NumPy-exact math (exp) | `Utilities/NDFloatMath.cs`, `Utilities/NDFloatMath.Simd.cs` |
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
  corpus/*.jsonl                      committed corpus (~76K cases / 43 tiers; op corpus ~63K incl. 4.4K Char woven + 695 Decimal), copied to test output via the csproj glob
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
- **Three `FuzzMatrix` gates**: `FuzzCorpusTests` (the op corpus — ~63K cases across the tiers, checking dtype + shape + bytes + error parity + per-file minimum-count floors; Char woven into 18 tier files, 12 `Decimal*` tiers: unary/binary/reduce/scan/power/varstd/matmul/astype/stat/where/sort/manip), `IndexOracleTests` (the index oracle — `index_curated` 2,273 + `index_dtype` 104 + `index_setter_dtype` 10 (cross-dtype cast-on-set) + `index_random` 10,000; the advanced-indexing parity gate), and `MetamorphicTests` (12 NumPy-free invariants incl. Half/Complex/Decimal/bool/char + strided views). A failing op case auto-shrinks to a 1-element repro.
- **Dtype coverage**: per-mode dtype axes widened toward `ALL_DTYPES`. **Char** (no NumPy dtype) is woven into 18 tier files via the uint16 proxy (`gen_oracle.char_tier`, relabelled uint16→char). **Decimal** (no NumPy analog) rides an independent C# oracle (`gen_decimal_oracle.cs`, naive scalar `System.Decimal`; incl. axis reductions, empty, negative int powers). Verified Char/clip-bool/round/dot bugs are carved from the green corpus and reproduced under `[OpenBugs]` (`OpenBugs.Char.cs`, `OpenBugs.DtypeCoverage.cs`, `OpenBugs.FuzzGaps.cs`) — NOT excused in `MisalignedRegistry`.
- **Regenerate** (deterministic; needs `numpy==2.4.2`): `python test/oracle/gen_oracle.py <mode>` (modes: `smoke astype_full binary divmod_power comparison unary reduce where place matmul rounding bitwise unary_extra nanreduce scan stat logic modf manip sort tail params aliasing copyto errors groupa`) + `python test/oracle/gen_index_oracle.py` (the `index_*` tiers) + `python test/oracle/fuzz_random.py 1234 2000 random_smoke.jsonl` + `dotnet run test/oracle/gen_decimal_oracle.cs` (the `decimal_*` tiers), then `dotnet build` (copies the corpus to test output).
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

NumSharp has many key types/fields/methods marked `internal` (Shape.dimensions, Shape.strides, NDArray.Storage, np._FindCommonType, etc.). To access them from a `dotnet run` script, override the assembly name to match an existing `InternalsVisibleTo` entry:

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
```

**How it works:** NumSharp declares `[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript")]` in `src/NumSharp.Core/Assembly/Properties.cs`. The `#:property AssemblyName=NumSharp.DotNetRunScript` directive overrides the script's assembly name (which normally derives from the filename) to match, granting full access to all `internal` and `protected internal` members.

### Accessing Unsafe code
NumSharp uses unsafe in many places, hence include `#:property AllowUnsafeBlocks=true` in scripts.

### Script Template (copy-paste ready)

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
```

### Key Internal Members Available

| Member | What it exposes |
|--------|----------------|
| `shape.dimensions` | Raw int[] of dimension sizes |
| `shape.strides` | Raw int[] of stride values |
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
A: Core ops (`dot`, `matmul`, `outer`) in `LinearAlgebra/`; `trace`/`diagonal` in `Indexing/`.

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
