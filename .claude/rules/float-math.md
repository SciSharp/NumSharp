---
paths:
  - "src/NumSharp.Core/Utilities/{NDFloatMath,NDComplexMath}*.cs"
  - "src/NumSharp.Core/Math/NDArray.{Correlate,Convolve}*.cs"
  - "src/NumSharp.Core/Backends/Default/Math/{Sliding/SlidingDot,Default.Correlate,Default.Convolve,Default.Sinc}*.cs"
  - "src/NumSharp.Core/Math/np.{correlate,convolve,sinc,arcsinh,arccosh,arctanh,exp,exp2,expm1,log,log1p,sin,cos,tanh}*.cs"
  - "src/NumSharp.Core/Backends/ISlidingDotBackend.cs"
  - "test/NumSharp.Tests/Math/**"
---

# Float math: sliding dot products, NumPy's float kernels, complex NaN signs

## `np.correlate` / `np.convolve`

NumPy's `_pyarray_correlate` + `small_correlate` (`Backends/Default/Math/Sliding/SlidingDot.cs`).
`np.correlate(a, v, mode='valid')` is `c_k = Σ_n a_{n+k}·conj(v_n)`; `np.convolve(a, v, mode='full')` is
`correlate(a, v[::-1])` (no conjugation).
- Layers: NumPy's C functions `multiarray.correlate` / `correlate2` (`PyArray_Correlate` / `PyArray_Correlate2` with
  `_pyarray_correlate`'s checks) are the engine members `TensorEngine.Correlate` / `Correlate2`
  (`Backends/Default/Math/Default.Correlate.cs`); np.convolve's Python statements (`ndmin=1`, its own `a cannot be empty`
  / `v cannot be empty`, the swap by length, `v[::-1]`) are `TensorEngine.Convolve` (`Default.Convolve.cs`); the four
  facades are one call each, `NDArray.convolve` / `correlate` keeping `[NDScoped]` so the weaver roots their operands at
  the public entry (`OperandRootsTests`).
- One 3-region engine for both: the left ramp (scalar), the middle (vectorized over OUTPUT positions, `Vector<T>`
  separate multiply-add for the 10 SIMD dtypes; char through ushort), the right ramp, plus a length-gated `DotSimd` inner
  reduction (length ≥ 32) for the ramps and few-output tails of large kernels. correlate is non-commutative: when
  `len(a) < len(v)` the operands swap and the output is time-reversed (`_pyarray_revert`, in the result's own buffer as
  NumPy does: `SlidingDot.RevertInPlace`), and a complex `v` is conjugated.
- The mode is NumPy's `PyArray_CorrelatemodeConverter` (`SlidingDot.ParseSlidingMode`): case-SENSITIVE; a text whose
  first letter names a mode but is not its exact name is `ValueError("Use one of 'valid', 'same', or 'full' for
  convolve/correlate mode")`, any other text `ValueError("mode must be one of 'valid', 'same', or 'full' (got 'x')")`
  with the argument in Python's repr; a null mode is NumPy's `None`, `TypeError("convolve/correlate mode not
  understood")`; a lone surrogate is CPython's `UnicodeEncodeError`. np.convolve checks emptiness before the mode;
  np.correlate parses the mode before its operands. NumPy's integer modes (0 / 1 / 2) have no counterpart.
- `MaterializeForSliding` copies a one-dimensional view of the compute dtype (np.convolve's reversed kernel, a stepped,
  offset or broadcast operand) with ONE call of the strided move kernel (`CopyOneDimensional`).
- Bit parity without a backend: vectorizing over outputs keeps each output's `Σ_t a[i+t]·k[t]` in t order, so it is
  byte-identical to the sequential sum, hence to NumPy, for the small-kernel regime. Integer, bool, decimal and Half are
  exact at every size (integer SIMD wrap and the reordered reduction are modular-exact). Long float32/float64 kernels,
  and complex128 at any length, are bounded-ULP, because NumPy reduces those positions through cblas `?dot` (`?dotu`).
- NumPy's dotfunc per dtype in its scalar regime (scipy-openblas' `?dot` is a plain scalar sum below its vector block),
  reproduced exactly, so every position shorter than the block is byte-exact without the backend:
  - ddot: a sequential double sum below 16 terms (`DotSimd`);
  - sdot: float32 products rounded, then summed in a DOUBLE below 32 terms (`SdotManaged`);
  - zdotu: four separate double sums `Σar·br − Σai·bi`, `Σar·bi + Σai·br` below 8 terms (`ZdotuManaged`), then its
    RESULT built with C99 complex arithmetic: scipy-openblas' `openblas_make_complex_double(re, im)` is
    `re + im*_Complex_I`, whose real part is `re + im*0`, so an infinite or NaN imaginary part makes the real part NaN
    (`ZdotuResult`, also on the blocked kernel's stores: `np.convolve([1+0j], [inf+0j])` is `nan+nanj`). Only when both
    operands reach CDOUBLE_dot with a positive stride: `PyArray_Correlate` passes a one-element operand through with its
    own stride, and np.convolve's kernel is `v[::-1]`, so a fresh one-element kernel arrives with stride -16, which
    `blas_stride` refuses; CDOUBLE_dot's plain loop runs (`CdoubleDotPlain`). The caller decides with
    `DotOperandBlasable` (np.convolve reverses its kernel's stride, np.correlate's `conj(v)` is always fresh, a cast is
    always fresh). The backend is skipped there too, as NumPy skips cblas.
- Long products without the backend (`Backends/Default/Math/Sliding/SlidingDot.Long.cs`): blocked kernels over the positions at or past
  the vector block. B consecutive outputs share one loop over the kernel (a broadcast, one load per vector, FMAs into
  eight chains); the head/tail steps read tiny zero-padded edge windows, which requires a FINITE kernel (inf/NaN declines
  to the per-position kernels). float16 (HALF_dot: a sequential float32 sum at every length) has an exact blocked kernel,
  byte-exact at every position. complex128 is one 256-bit FMA per complex MAC on both sides, so only NumPy's per-position
  overhead is left to win.
- `SlidingCorrelateInto` is the pointer-level core, which the numpy.polynomial series arena calls for its `np.convolve`
  statements.
- Byte parity with `NumSharp.Interop.OpenBLAS`: NumPy reduces every ramp position, and the middle whenever
  `small_correlate` declines (a real kernel longer than 11, or any complex kernel), with its per-dtype `dotfunc` (chunked
  cblas `?dot`/`?dotu` accumulated in double). When `TensorEngine.Blas` also implements `ISlidingDotBackend`
  (`Backends/ISlidingDotBackend.cs`), `SlidingCorrelate` routes exactly those positions through the scipy-openblas
  `sdot`/`ddot`/`zdotu` NumPy calls (`OpenBlasEngine.SlidingDot`, reusing the product family's `@name@_dot`), so
  `np.correlate`/`np.convolve` are byte-identical to NumPy for float32/float64/complex128 under the products' three
  parity conditions (build, thread count, DYNAMIC_ARCH kernel). The `small_correlate` regime (real float, n2 ≤ 11) stays
  managed (already byte-exact), so no call mixes managed and native positions; a backend that declines the dtype (a
  real-only CBLAS handed complex) leaves the managed kernel in place. It is per position (one backend `?dot` per output,
  as NumPy calls `dotfunc`). Gate: `SlidingDotLiveParityTests` (live NumPy); the managed path is the `groupa` tier's
  default.

## Inverse hyperbolic: `arcsinh` / `arccosh` / `arctanh`

(+ the Array-API aliases `asinh`/`acosh`/`atanh`, the same ufuncs.) They follow the `arcsin`/`sinh` seam (`ASinh`/
`ACosh`/`ATanh` → `ExecuteUnaryOp` → `UnaryOp.{Asinh,Acosh,Atanh}`) with the `f(x, out=, where=, dtype=)` surface plus
positional-dtype convenience overloads.
- Real float32/float64 are byte-identical to NumPy 2.4.2: `Math.Asinh/Acosh/Atanh` and the `MathF` twins call the same
  MSVC `ucrtbase` CRT as `npy_asinh/acosh/atanh` (specials, ±inf, NaN, subnormals and ±0 included).
- These are the one arc family NumPy has no active SIMD kernel for on win-amd64 (its SVML `asinh`/`acosh`/`atanh` are
  AVX-512/Linux-gated), so byte parity requires the scalar CRT on both sides, and parity is the speed ceiling. The kernel
  is a non-unrolled scalar loop (`EmitUnaryScalarLoop`; there is no `Vector<double>.Asinh`) with raw-pointer direct
  calls. Do not unroll it (the CRT call latency dominates) or vectorize it (`TensorPrimitives` is itself scalar and
  0-ULP for these three precisely because no correctly rounded vector form exists; a SIMD polynomial would diverge).
- Integer/bool tier to float per `ResolveUnaryFloatReturnType` (bool/i8/u8 → f16, i16/u16/char → f32, i32+ → f64).
  float16 is byte-exact through `Half.Asinh/Acosh/Atanh` (they compute in float32, NumPy's `astype 'e'->'f'`). Decimal
  goes through the decimal→double bridge (15 significant digits in the valid domain; out-of-domain NaN/inf throws
  `OverflowException` on `(decimal)NaN`, like every decimal transcendental: `arcsin(2m)`, `sqrt(-1m)`, `log(0m)`).
- complex128 derives from the byte-exact `Asin`/`Acos`/`Atan` (= `Catanh`) ports through NumPy's msun involution
  `I·conj(·)`, a pure component swap: `asinh(z) = swap(asin(swap z))`, `atanh(z) = catanh(z)`, `acosh(z) =
  cacosh_formula(acos(z))`, within the complex-unary family's ≤ 3 ULP envelope. `arccosh` inherits `arccos`'s
  sub-DBL_MIN-imaginary edge (`MisalignedRegistry` branch (7), the complex-unary cases beyond 3 ULP).
- Gates: `Math/InverseHyperbolicTests.cs`, `NpApiOverloadTests_UnaryMath`, the `unary_extra`/`specials` tiers.

## `np.sinc`

The normalized sinc `sin(pi*x)/(pi*x)`, a port of NumPy 2.4.2's `numpy/lib/_function_base_impl.py::sinc`; a plain
function with no `out=`/`where=`/`dtype=`: the facade `Math/np.sinc.cs` calls `TensorEngine.Sinc` (DefaultEngine's fused
pass in `Backends/Default/Math/Default.Sinc.cs`). NumPy's composition is `y = pi*x; eps = finfo(y.dtype).eps if y.dtype.kind
== 'f' else 1e-20; y = where(y != 0, y, eps); sin(y)/y`.
- The dtype follows `pi*x` (a NEP 50 weak float): bool, every integer width and Char → float64 (not np.sin's i8 → f16
  tiers); float16/float32/float64 kept; complex → complex128. The zero test is on `pi*x`, not `x`, so sub-underflow
  denormals behave as in NumPy.
- One fused `np.evaluate` pass: the pi multiply, the zero → eps select, `sin` and the divide fold into one inner-loop
  kernel over the shared `t = pi*x` subtree.
- Bit-exact for every real dtype (float64 arithmetic is IEEE-exact; float32 `sin` is NumPy's kernel, float64 `sin` the
  shared `ucrtbase` `Math.Sin`). complex128's per-component bits are not reproducible (NumSharp's complex sin differs
  from NumPy's UCRT `csin` within the ≤ 3-ULP envelope), so complex is excluded from the byte corpus and pinned by an
  allclose unit test; the value is accurate to about 2.5 ULP relative. Decimal rides the same expression through the
  decimal→double bridge.
- The float64 cells are host-libm-sensitive near the zero crossings (`sin(pi*k) ≈ 0`), so the `sinc` tier is host-pinned
  to win-amd64 (`RunHostLibmCorpus`, Inconclusive elsewhere). Gate: `Math/np.sinc.Test.cs`.

## float32 kernels ported from NumPy

`exp`, `log`, `sin`, `cos` and `tanh` at float32 are bit-exact with NumPy, because `Utilities/NDFloatMath.cs` ports the
kernels NumPy 2.4.2 runs instead of calling the platform libm: `simd_exp_FLOAT` and `simd_log_FLOAT`
(`loops_exponent_log.dispatch.c.src`, the `SIMD_AVX2_FMA3` instantiation; on Windows the AVX-512 variant is
`!defined(_MSC_VER)`-gated and never compiled) and `simd_sincos_f32` (`loops_trigonometric.dispatch.cpp`,
`NPY_SIMD_FMA3`). `rad2deg`/`deg2rad` form their constant at the operand's precision, as NumPy's `RAD2DEG`/`DEG2RAD`
macros do. None of these kernels is correctly rounded (NumPy documents 2.52 ULP for exp, 3.83 for log, 0.647 for the
sine polynomial), so `MathF.Exp`/`Log`/`Sin`/`Cos` differ from NumPy on many inputs.
- Each is verified over all 2³² float32 bit patterns through both the SIMD kernel and the scalar entry point. The tier
  `numpy_f32_kernels.jsonl` pins the discriminating inputs, and `MisalignedRegistry`'s unary-ULP branch excludes these
  ops at a float32 result (`NumPyPortedFloat32Kernels`), so a 1-ULP regression fails.
- `tanh` is also a float64 port: NumPy ships its own table-driven kernel at both widths (`simd_tanh_f32`/`simd_tanh_f64`
  in `loops_hyperbolic.dispatch.cpp.src`, from Intel SVML), where exp/log/sin/cos fall through to the platform's scalar
  `npy_*` at float64 and already agree. tanh is odd, so the kernel works on `|x|` and ORs the sign back:
  `[0, SATURATION_THRESHOLD)` splits into 32 subintervals at float32 / 16 at float64; the exponent field of `|x|` indexes
  a table of per-subinterval minimax coefficients plus a recentring offset `b`; the answer is a degree-6 / 16 Horner
  evaluation of `P(|x| - b)`. The top subintervals hold `1 + 0·y + …`, so the saturated range returns exactly ±1 through
  the same arithmetic. The FMA is not a host pin here: NumPy spells its Horner steps as `hn::MulAdd`, an explicit fused
  multiply-add. Verified over all 2³² float32 inputs and, at float64, every sign × exponent × 2²⁰ mantissa prefix plus
  random full-width patterns. Gates: `numpy_f32_kernels.jsonl`, `numpy_f64_kernels.jsonl`, with tanh carved out of the
  unary-ULP excuse at both widths (`NumPyPortedFloat64Kernels`).
- The tanh coefficient fetch is a transpose, not a gather: a subinterval's 8 float32 coefficients are 8 consecutive
  floats (one `Vector256`), so the whole table read is 8 contiguous row loads and an 8×8 transpose, where a gather per
  coefficient issues 8 `vgatherdps`. float64 stays scalar (its lookup would cost 18 gathers per 4 lanes). `Vector512` is
  composed from two verified `Vector256` halves.
- Specials (mostly consequences of NumPy blanking NaN lanes before its range compares): any NaN (quiet, signalling,
  either sign, any payload) returns `0x7fc00000` (.NET's `float.NaN` is `0xffc00000`); `exp` saturates at
  `x ≥ 88.72283935546875` → `+inf` and `x ≤ -103.97208404541015625` → `+0`, the band between giving subnormals through
  `fma_scalef_ps`'s exponent split; `log` returns `-inf` at ±0 and a NEGATIVE NaN for a negative argument (a positive one
  for a NaN argument); `sin`/`cos` hand arguments past their Cody-Waite limits (117435.992 and 71476.0625) to the
  platform libm, as NumPy does, so those inputs are libm-dependent by construction. NumPy also raises FP status flags
  (`RuntimeWarning`), which NumSharp does not model.
- Host pin: each kernel's quadrant uses a FUSED multiply-add, because MSVC 19.44 (the numpy==2.4.2 win-amd64 wheel's
  compiler) contracts NumPy's separate multiply and add. At `x = 0xc26d0e6c`, `x·log2 e` lands on the exact tie `-85.5`,
  and the two spellings round `k` to -85 and -86.
- `Utilities/NDFloatMath.Simd.cs` carries `Vector{128,256,512}<float>` overloads (each algorithm is elementwise, so lane
  `i` is bit-identical to the scalar call), gated by `DirectILKernelGenerator.NumPyFloatKernelSimdAvailable` on hardware
  FMA as NumPy gates its own; without FMA the scalar entry point runs, same bits. The rarely taken subnormal branches stay
  branches (folding them into branchless selects costs a fifth of each kernel), and the vector overloads carry
  `AggressiveOptimization` (without it tier-0 JIT runs the first calls far slower than steady state).

## float32 `exp2`

The numpy 2.4.2 win-amd64 wheel has no vector exp2 (its SVML `simd_exp2` is gated on `NPY_HAVE_AVX512_SKX &&
NPY_CAN_LINK_SVML`), so Windows runs the scalar MSVC CRT `exp2f`, and no reproducible NumPy kernel exists to match; a
pure-float polynomial cannot reach the CRT's near-correctly-rounded result. `NDFloatMath.Exp2` computes
`2^x = 2^n · 2^r` (`n = rint(x)`, `r ∈ [-0.5, 0.5]`) with `2^r` evaluated in DOUBLE (a degree-8 minimax, 2^-37.7
relative error), narrowed once to float, then scaled by `2^n` in the exponent field (exact for every normal result).
Result: within 1 ULP of NumPy on every input; the differing inputs all give subnormal outputs (the narrow plus the
denormal split round twice). The 2^n scale stays in float integer operations because `Vector256.ConvertToInt64` is
AVX2-emulated and slower than NumPy. Vectorized in `NDFloatMath.Simd.cs` (floats widen to two `Vector<double>` halves
with `WidenLower`/`WidenUpper`, the polynomial runs per half, `Narrow` packs `2^r` back), gated by
`NumPyFloatKernelSimdAvailable`, lane-identical to the scalar entry point. It stays under the ≤ 2-ULP unary excuse, not in
the bit-exact `NumPyPortedFloat32Kernels` list. Specials: NaN → `0x7fc00000`, `x ≥ 128` → `+inf`, `x ≤ -150` → `+0`,
`±0` → `1`, integer `x` exact (`exp2(10) = 1024`). Other dtypes use `Math.Pow` (double/int/Half) or
`NDComplexMath.Exp2`. Gates: the `unary_extra`/`specials` tiers, `Math/Exp2KernelTests.cs`.

## The platform libm: `expm1` / `log1p`

`expm1`/`log1p` at float32/float64 call the platform libm and are computed as `Exp(x)-1` / `Log(1+x)`: catastrophic for
small |x| (`expm1(1e-8)` returns 0 where NumPy returns 1e-8) and wrong on the sign of zero (`expm1(-0.0)` gives `+0.0`).
Neither is portable (NumPy calls the CRT for both), so bit parity is out of reach; the accuracy defect is a separate,
open bug (.NET's `float.ExpM1`/`double.ExpM1`/`LogP1` are themselves `Exp(x)-1`/`Log(1+x)`). float16 loops and float64
`exp`/`log`/`sin`/`cos` agree with NumPy bit for bit on win-amd64.

## complex128 unary ufuncs: NaN signs

The complex128 unary ufuncs in `Utilities/NDComplexMath.cs` (sqrt, log, log2, log10, log1p, exp, exp2, expm1, square,
reciprocal, sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, conjugate, negative,
positive) match NumPy 2.4.2 bit for bit including the NaN SIGN. win-amd64 NumPy runs MSVC UCRT's complex functions, whose
"produce a NaN" slots emit the positive `NPY_NAN` (`0x7ff8…`), while .NET's `double.NaN` is `0xfff8…`. Per path:
- a positive-NaN constant where NumPy canonicalises;
- `hypot`/`cexp`/`ctanh` propagate the input NaN's sign;
- `csinh`/`ccosh` canonicalise, so `csin`/`ccos` follow the transform negate;
- `square` is a true `vfmaddsub` with `re*im` in the addend slot;
- `reciprocal` is `CDOUBLE_reciprocal`'s `-1/d` division, not a `-scl` negate.

The oracle compares these ops' complex NaNs raw-byte (`ComplexNanContractOps` + `BitDiff.Compare(nanBitExact)` + a
`DiffHasSignFlip` hard failure) over the ±0/±inf/±NaN grid with both NaN signs. Complex and float16 `sign`/`maximum` keep
the input NaN bits, as NumPy does. complex128 runs scalar through `System.Numerics.Complex` operators / `Complex.Pow`,
about 2× slower than NumPy.
