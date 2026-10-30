---
paths:
  - "src/NumSharp.Core/Backends/Kernels/**"
---

# IL kernel generators

Kernels are emitted at run time through `System.Reflection.Emit.DynamicMethod`, by two classes split by who drives the
loop.

## `DirectILKernelGenerator`: whole-array kernels

`src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.*.cs`, one partial per kernel family (binary,
mixed-type, shift, unary, comparison, flat and axis reductions, scans, masking, casts and copies, selection, matmul,
clip, the polynomial families, …). One call processes the whole array: the kernel walks dimensions and strides itself,
and an iterator, if any, only sets it up. A typical signature:

```csharp
delegate void DirectKernel(void* input, void* output, long* strides, long* shape, int ndim, long iterSize);
```

Execution paths:
1. SimdFull: both operands contiguous, a SIMD-capable dtype → a vector loop plus a scalar tail.
2. ScalarFull: both contiguous, a non-SIMD dtype (Decimal) → a scalar loop.
3. General: strided or broadcast → coordinate-based iteration.

## `ILKernelGenerator`: NDIter-driven per-chunk kernels

`src/NumSharp.Core/Backends/Kernels/ILKernelGenerator*.cs` (the folder root, not `Direct/`): `np.where`, the flat and
pairwise reductions (float16 runs included), scans and the polynomial evaluation kernels. A kernel processes one
inner-loop chunk per call while `NDIterRef` drives the loop and advances the pointers between calls (NumPy's
`PyUFuncGenericFunction` model), through `NDIterRef.Execute(kernelKey)` / `ForEach`:

```csharp
unsafe delegate void NDInnerLoopFunc(
    void** dataptrs,   // [nop] current operand pointers
    long*  strides,    // [nop] per-operand byte stride of the inner loop
    long   count,      // elements in this chunk
    void*  auxdata);   // op-specific extras (e.g. an axis index)
```

## Shared infrastructure (`Backends/Kernels/`)

- `VectorMethodCache.cs`, `ScalarMethodCache.cs`: reflection caches (Vector128/256/512, Math/MathF).
- `KernelOp.cs`: the `BinaryOp`, `UnaryOp`, `ReductionOp`, `ComparisonOp` and `ExecutionPath` enums.
- `BinaryKernel.cs`, `CopyKernel.cs`, `ReductionKernel.cs`, `ScalarKernel.cs`: kernel key structs and delegate types.
- `StrideDetector.cs`: layout classification (Contiguous / Strided / Broadcast).
- `SimdMatMul.*.cs`: matmul SIMD primitives.

## Coverage

| Category | Operations |
|----------|------------|
| Binary | Add, Sub, Mul, Div, Power, FloorDivide, BitwiseAnd/Or/Xor |
| Shift | LeftShift, RightShift (SIMD against a scalar, a scalar loop against an array) |
| Unary | Negate, Abs, Sign, Sqrt, Cbrt, Square, Reciprocal, Floor, Ceil, Truncate, trig, Exp, Log, BitwiseNot |
| Reduction | Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any (flat and per axis) |
| Scan | CumSum, CumProd (elementwise SIMD, axis support) |
| Comparison | Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual |
| Clip / Modf | Clip, Modf (SIMD helpers) |
| Cast (SIMD, contiguous and strided) | see below |

Casts:
- float→int32 (cvtt); float/complex128→uint32 and →uint64 (an AVX2 mod-2³² reduce and hi/lo split, NumPy's modular
  wrap); float→{int8, uint8, int16, uint16, char} (cvtt + narrow); int→a narrower int (narrow); int→float
  (`.Cast.IntToFloat`: {int8, uint8, int16, uint16, char, int32, uint32} → {float32, float64}; int64/uint64 → float64 by
  the exponent-bias splice, one round-to-nearest-even rounding).
- float16 through Giesen's bit tricks: f16→f32 keeps NaN payloads, f16→f64 keeps a signalling NaN signalling, X→f16
  keeps sNaN; int64/uint64→f16 clamps |v| ≥ 65520 to a ±70000 sentinel, packs the low 32 bits, converts, then narrows.
- complex128→{int, bool} through a real-part deinterleave; *→bool is `!= 0`, float16 truthiness is
  `(bits & 0x7fff) != 0` (±0 → False, NaN/inf → True).
- Strided rows gather (4- and 8-byte elements) or stage gather → buffer → convert. 1- and 2-byte strided and reversed
  views use SIMD lane shuffles (`.Cast.SubwordCopy`, `.Cast.SubwordNarrow`, `.Cast.SubwordWiden`, and float16→bool in
  `.Cast.ToBool`); 4- and 8-byte same-type strided copies are `.Cast.WordCopy` (reverse, deinterleave, gather).
- NaN, ±inf and out-of-range float→unsigned inputs are a C-undefined conversion: NumSharp reproduces the MSVC NumPy
  answer (glibc/gcc's differs, and so do one build's vectorized and scalar loops), so those cells are regression pins,
  not portable parity (`test/NumSharp.Tests.Oracle/Fuzz/README.md` → "Host-dependent values").

## Rules

- The direct binary route writes its result in C-coordinate order over the coalesced dims (`ExecuteKernelCoalesced`
  passes no result strides), so an F-allocated result is correct there only when every dim coalesces into one run.
  `HasNonScalarBroadcastOperand` keeps that route from allocating F for a broadcast set, and the small-broadcast
  shortcut sends all-F sets to `NDIter`, which allocates F. A broadcast all-F set on that route (float16 add, subtract,
  multiply, divide) is walked with its axes reversed: C order over the reversed dims is the F memory order. Gates: the
  binary tiers' `pp_fbcast_*` layouts, `Backends/Kernels/FOrderBroadcastBinaryTests.cs`.
- A strided copy or cast never reads outside its view. The two-load deinterleave of the 1- and 2-byte strided kernels
  (`a[::2]`) starts its second load one element early and shifts the wanted half down instead of masking, so the load
  ends on the group's last element; the 4- and 8-byte kernels bound their loop. Gate:
  `Backends/Kernels/StridedCastFenceTests.cs` (every source a view over `FencedMemory`, all 15 × 15 dtype pairs, 1-D and
  2-D layouts, lengths around each kernel's vector group, and `np.zeros(new Shape(40 * 4096), NPTypeCode.Byte)["1::2"].copy()`).
- Emit a per-element copy as a typed `Ldind`/`Stind` move for the dtype width, not `cpblk` of a runtime-tiny size
  (a memcpy call where a `mov` belongs is about 2× slower).

## float16

The BCL has no vector float16 arithmetic and no F16C (`Vector<Half>` throws). Three techniques, chosen by whether the
operation needs real float math, all bit-exact with NumPy:
- Bit-level AVX2 on raw ushort lanes: sign / negative / absolute, min / max / ptp / nanmin / nanmax (flat and per axis,
  every layout: NumPy's sequential HALF loops, the first NaN and the first ±0 in NumPy's visiting order),
  maximum / minimum / fmax / fmin, the six comparisons, clip, and floor / ceil / trunc / rint. They compare through the
  sign-magnitude order map `key = bits ^ (0x8000 | asr(bits, 15))`, whose NaN keys lie outside the finite band.
- SIMD widen-compute-narrow for add / subtract / multiply / divide: Giesen f16→f32, one float32 vector operation, a
  round-to-nearest-even narrow, on every layout (row and column broadcasts, strided and reversed views gathered in
  256-element blocks, broadcast F sets walked in F order, `out=`).
- Scalar widen-unary for 16 transcendentals (`EmitUnaryHalfViaFloat`: `(Half)MathF.X((float)h)`, NumPy's npy_half
  model): conversion-bound, at parity with NumPy, which is the ceiling.

`np.power` on float16 is NumPy's HALF_power: an exact widen (a signalling NaN stays signalling), the CRT `powf`,
`npy_float_to_half`, and no scalar-exponent fast path (`HalfPower` runs a vector widen and narrow around one `MathF.Pow`
loop per 256-element block).

Traps: portable `Vector256<ushort>` operations do not lower to AVX2 (30× slower); RyuJIT swaps the operands of
commutative intrinsics, so NaN priority needs an explicit blend; AVX2 has no 16-bit variable shifts.

## NaN operand order of float add and multiply (a win-amd64 pin)

When both operands of a float32/float64 add or multiply are NaN, the result is one operand's NaN, and the compiled loop
decides which (x86 `addpd`/`mulpd` return their first source operand quieted; the operations are commutative, so the
compiler picks the order). numpy 2.4.2 win-amd64 (MSVC 19.44):
- add and multiply keep the SECOND operand's NaN on contiguous, strided, reversed, `out=`, `where=` and transposed
  operands, buffered 2-D broadcasts, and 0-d × 0-d;
- except the AVX2 scalar-broadcast SIMD loop (one input of stride 0, the other input and the out contiguous, longer
  than 2 vectors): its blocks of 2 vectors (8 float64 / 16 float32 lanes) keep the BROADCAST operand's NaN and its
  scalar tail the ARRAY operand's. The split restarts per inner-loop call: per `where=` run, and per `np.getbufsize()`
  chunk when an operand is cast (a cast 0-d or ≤ bufsize 1-D input is copied whole: one call);
- subtract and divide keep the FIRST operand's NaN; float16 the second everywhere; scalar math (`np.float64`) the first;
- reductions and scans run their own compiled orders, which NumSharp does not follow. `sum`/`prod` are pairwise (MSVC's
  leaf loop keeps the accumulator's NaN, its tree combine the second operand's); `cumsum`/`cumprod` keep the NEW
  element's NaN at each step, where NumSharp's scan kernels keep the accumulator's (`DirectILKernelGenerator.Scan.cs`).

NumSharp encodes the order explicitly (`DirectILKernelGenerator.NaNOperandOrder.cs`), because RyuJIT may swap a
commutative operation's operands, so the IL operand order proves nothing:
- vector: `(va & (vb == vb)) op vb` (a NaN in `vb` zeroes `va`, so the hardware can only return `vb`);
- scalar: `r = a op b; if (r != r && b != b) r = b + b`;
- broadcast: a cold region path (`EmitBroadcastNaNRegion`), taken only when the broadcast value is NaN; the hot loops
  keep the plain instruction. The NDIter, `out=` and `where=` routes pass the rule (`BroadcastNaNRule`: the split per
  call, the buffer chunk) as kernel auxdata, computed by `DefaultEngine.NumPyBroadcastNaNRule`.

It is wired into the MixedType kernels (SimdFull / ScalarFull / SimdScalarLeft / SimdScalarRight / SimdChunk /
General), the Tier-3B inner-loop shell and its strided broadcast dispatch, the 0-d × 0-d scalar delegate and NDExpr's
`BinaryNode`; a single `np.evaluate` node with a 0-d operand delegates to the direct kernel. It costs two µops per add or
multiply vector.

Linux NumPy (the GCC build) orders the operands differently on the same CPU: its contiguous SIMD body keeps the first
operand's NaN and its tail the second's, its broadcast loop the broadcast operand's everywhere, its strided / `out=` /
`where=` paths the first. The order is a compiler artifact, pinned to win-amd64 like the float16 and complex NaN-sign
contracts.

Gates: the oracle tier `nan_order.jsonl` (NaN payloads compared raw through `params.nan_bits` on x86 hosts with
X86_V3) and `Math/NaNOperandOrderTests.cs`. Open, pinned with NumPy's raw bits in
`test/NumSharp.Tests/OpenBugs.NaNOperandOrder.cs` (one test per class):
- `cumsum`/`cumprod` of float32, float64 and complex128 (complex128 per part);
- complex128 add (array layouts), multiply and divide: MSVC's complex loops choose the NaN per part and, for add, per
  unrolled vector pair; NumSharp's complex `cumprod` also multiplies element 0 by the identity instead of copying it;
- N-D broadcasts NumPy runs one row per call: an outer product with two rows such as `(2,1)+(1,64)`, and `(m,n)+(m,1)`
  with n > 4096;
- a 0-d operand against an N-D view NumPy buffers (a row slice `x[:, :k]`, reversed rows or columns, a 3-D slice);
- a reversed cast operand of 9 or more elements;
- `where=` together with a cast;
- `np.evaluate` trees deeper than one node with a NaN 0-d or size-1 broadcast input.

## Narrow rows: the 2-D block kernel

`ND2DElementwiseKernel` (`DirectILKernelGenerator.InnerLoop2D.cs`) loops the outer axis itself with per-operand outer
byte strides and is called once per 2-D block, instead of once per row under EXTERNAL_LOOP (the per-row odometer
advance and the per-chunk kernel's SIMD-viability prologue dominate narrow rows). It reuses the per-chunk kernel's
scalar and vector emit bodies, so results are byte-identical.
- Dispatch: `NDIterRef.{Is2DElementwiseShape,TryExecute2DElementwise}`, wired into the three packed-key
  `ExecuteElementWise*` entry points (`out=` and allocating unary/binary routes). It requires: unbuffered, no `where=`
  mask, EXTERNAL_LOOP, NDim ≥ 2, the output's inner axis element-contiguous, every input's inner axis contiguous or
  broadcast (stride 1 or 0).
- The contract carries `innerByteStrides` and dispatches once per call on the inner-mode pattern (unary {C, S}, binary
  {CC, SC, CS}); each pattern is a full 2-D loop with the S operand loaded and broadcast once per row.
- 32- and 64-bit lanes on a 256-bit AVX2 host finish each row's `innerCount % lanes` remainder, or the whole row when
  `innerCount < lanes`, with one `MaskLoad`→body→`MaskStore` (the mask hoisted per call; integer masked-off lanes filled
  with 1 so a software vector divide cannot throw). Other hosts and 1- and 2-byte lanes keep the scalar tail.
- A once-per-call row-shape dispatch picks generic rows, one-vector rows (`innerCount == lanes`: one full vector per row,
  no inner loop) or masked rows. A null vector body or a non-SIMD dtype set gets a scalar-only 2-D block (constant-stride
  addressing when every inner stride is its own element size, runtime strides otherwise).
- Leading axes that do not fold into the block (`x[::2, :, :w]`) are walked by a per-block odometer in the dispatcher.
- `positive` has an identity vector body, so a contiguous `positive` is a SIMD copy.
- NDIter merges adjacent axes every operand walks contiguously in iteration order
  (`NDIterCoalescing.CoalesceAxesIterationOrder`: `stride[d] == stride[d+1]*shape[d+1]`, or a size-1 stride-0 axis) on
  the non-all-contiguous branch, as NumPy's `npyiter_coalesce_axes` does unconditionally. So `a[:, ::2]` is one strided
  chunk, as NumPy hands it out, and a 3-D `x[:, :, :w]` reaches the kernel as `(rows, w)`.
- `where=`: `DirectILKernelGenerator.InnerLoop2D.Masked.cs` (`NDIterRef.Is2DMaskedElementwiseShape` /
  `TryExecute2DMasked`, wired into both `ExecuteElementWise` overloads, the packed-key one included). The mask rides as
  the trailing operand: a byte-per-element mask becomes each vector group's lane mask (pmovzx + compare) feeding the
  tails' masked load/body/store, a `(rows, 1)` mask gates whole rows, and ops without a vector body get a masked scalar
  block. The tail word is assembled from exactly `tail` bytes, so the mask array is never over-read.
- Not covered: copy-class ops at w = 2/3 have no half-vector store; buffered cast routes run per chunk; a strided inner
  row that does not coalesce stays on the per-chunk gather path.

## Three or more inputs with a broadcast input: pattern stages

The 1-D shell, the 2-D block kernel and the `where=` 2-D block kernel dispatch on the inputs' C/S pattern once per call
(`DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs`; up to 4 inputs, 15 patterns). The all-C loop stays inline;
every other pattern is a STAGE: a DynamicMethod with the kernel's own signature, reached by one direct call and compiled
by the JIT on its first call (a DynamicMethod that another one calls compiles when the call first executes). 2^N copies
of a 4×-unrolled SIMD loop in one method would exhaust the JIT's inline budget.
- The gather route does the same for two or more inputs: a stride-0 input next to strided ones is read once and
  broadcast instead of gathered (`np.clip(x[::2], lo, hi)`, `np.add(x[::2], 5.0, out=o)`); the caller's vector body is
  unchanged, so the add/multiply NaN operand order holds.
- Users: np.clip's iterator route, the binary ufuncs' gather route (two inputs), and a caller's own 3- or 4-input kernel
  through `NDIterRef.ExecuteElementWise` / `ExecuteElementWiseTernary` / `ExecuteExpression`. Not `np.evaluate`, whose
  fused kernels come from their own generator (`CompileFusedInnerLoop` / `EmitFusedSimdLoop`).
- Gates: `Backends/Kernels/BroadcastPatternStageTests.cs` (every pattern × 1-D / 2-D / `where=` / gather against the
  materialized all-C route) and the `clip_ufunc` tier.
- Diagnosing a slow stage: `DOTNET_JitDisasm="<method name>*"` + `DOTNET_JitStdOutFile`, then count `call`
  instructions. A method whose inline budget is spent inlines nothing more, `AggressiveInlining` included, so split it.
