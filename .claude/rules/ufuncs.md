---
paths:
  - "src/NumSharp.Core/Math/**"
  - "src/NumSharp.Core/Operations/**"
  - "src/NumSharp.Core/Backends/Default/Math/*.cs"
  - "src/NumSharp.Core/Utilities/{NDDivision,NDGcdLcm}*.cs"
  - "test/NumSharp.Tests/Math/**"
---

# Elementwise ufuncs: arithmetic, `out=` / `where=` / `dtype=`, bitwise

## The ufunc surface

- Each elementwise ufunc exposes ONE NumPy-shaped overload, `f(x[, x2], NDArray out = null, NDArray where = null,
  NPTypeCode? dtype = null)`, mirroring `f(x1[, x2], /, out=None, *, where=True, dtype=None)`: `out` is the
  second/third positional slot as in NumPy, and `where`/`dtype` are reachable by name without `out`.
- `dtype=` selects the LOOP: the computation runs at that precision even with `out=` (`sqrt([2.], out=f64, dtype=f32)`
  stores the float32-rounded value; `add(0.1, 0.2, out=f64, dtype=f32)` stores `0.30000001…`); `power(2, -1, dtype: f64)`
  is 0.5 (the negative-integer-exponent ValueError applies only to integer loops); `power(10, 11, dtype: f64)` is 1e11
  exactly; `add(True, True, dtype: i32)` is 2 (the bool→logical-OR remap keys off the final loop dtype);
  `negative(bool, dtype: f64)` is legal. Inputs must reach the loop through same_kind casts (`Cannot cast ufunc
  '<name>' input [N] from ...`; binary errors are indexed, unary errors are not).
- Loop-existence gates raise `No loop matching ... ufunc <name>`: the float-only ufuncs (sqrt, exp, log, trig, `divide`,
  `true_divide`) reject an int or bool dtype; the bitwise ufuncs reject float, complex and decimal dtypes.
- `positive` is a full ufunc with identity loops at every dtype except bool: `positive(bool)` and `dtype: bool` raise the
  `did not contain a loop with signature matching types <class 'numpy.dtypes.XDType'> -> ...` texts;
  `positive(bool, dtype: f64)` works.
- `round_`/`around` follow NumPy's non-ufunc shape `round(a, int decimals = 0, NDArray out = null)`: `out=` only, the
  second positional is `decimals`, and a cast error at `decimals != 0` names ufunc 'multiply' (NumPy's composition).
- `rint` is the true ufunc form of round-half-to-even: float-tier (bool/i8/u8 → f16, i16/u16 → f32, i32+ → f64; floats
  and complex keep their dtype), reusing `UnaryOp.Round`'s kernel (complex rounds both parts; `dtype=<int>` has no loop).
  `round_`/`around` keep integer dtypes (np.round's integer path is an identity copy).
- floor/ceil/trunc have identity loops on every bool and integer dtype. The loop dtype comes from the input tier
  (`sinh(i1, out=f8)` stores float16-precision values). `reciprocal` of an integer 0 is the signed MinValue.
  `sign`/`positive` reject bool with the no-loop text. Bitwise and invert raise the no-loop TypeError for float inputs.
- Positional-dtype overloads (`np.sqrt(x, NPTypeCode.Single)`, `(x, Type)`) exist for source compatibility; they are not
  NumPy call forms (NumPy's second positional is `out`).
- Tests: `Math/UfuncDtypeOverloadTests.cs`.

## `out=` and `where=`

Supported on the binary ufuncs `add`/`subtract`/`multiply`/`divide`/`true_divide`/`mod`/`power`/`floor_divide`/
`arctan2`/`bitwise_and`/`bitwise_or`/`bitwise_xor` and the unary ufuncs `sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/`abs`/
`absolute`/`negative`/`square`/`log2`/`log10`/`log1p`/`exp2`/`expm1`/`cbrt`/`sign`/`floor`/`ceil`/`trunc`/`reciprocal`/
`sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/`arctan`/`deg2rad` (`radians`)/`rad2deg` (`degrees`)/`invert` (`bitwise_not`)/
`rint`. Engine plumbing: `Backends/Default/Math/DefaultEngine.UfuncOut.cs`.
- `out` joins the broadcast but is never stretched, needs a same_kind cast from the loop dtype (resolved from the
  inputs), is returned itself, and may alias an input (overlap-safe through COPY_IF_OVERLAP). `where` must be bool,
  broadcasts and joins the output shape; masked-off `out` slots keep their contents.
- NumPy's order of the refusals, every ufunc: the `dtype=` class check, a read-only `out`, a non-bool mask (both parsed
  by `UfuncLoopCode` for every operand), the loop (no-loop / input cast), the out cast, then the shapes in NumPy's two
  phases: every operand broadcast together first, the failure listing the inputs, then `out`, then `where` (a trailing
  space), and only then `out`'s no-stretch check against the joint shape (`ResolveUfuncIterationShape`,
  `BroadcastUfuncInputs`). A fresh result under `where=` takes NumPy's iterator layout, the mask voting
  (`WhereResultShape`: F operands under an F mask give an F result). The ufuncs that handle `out=` themselves (`ldexp`,
  `divmod`, `modf`, `frexp`) keep their own texts and order.
- A weak Python int (a 0-d C# integer literal) is range-checked against the FINAL loop dtype before any kernel runs
  (`ExecuteBinaryOp` → `CheckWeakPyIntOperand`: `np.maximum(int8, 1000)` is NumPy's `OverflowError: Python integer 1000
  out of bounds for int8`; a bool array meets the int in the int64 loop; np.power checks it before its negative-power
  error), and an int past 2^53 is re-packed into a float16 / float32 loop (`RepackWeakIntForFloatLoop`). The comparisons
  pack every weak scalar the same way (`comparison.md`).
- A passed-in masked out is flagged READWRITE (NumPy's masked ufunc rule), so an out that overlaps an input keeps its
  masked-off positions instead of receiving buffer contents.
- Trivial loop: a binary call whose operands are all trivially iterable (one dtype, out C- or F-contiguous, array inputs
  of out's dims in its order or one size-1 input, out disjoint from each input or the input itself) skips the iterator
  and calls the DirectIL kernel a fresh result uses (`TryTrivialBinaryOpInto`, `DefaultEngine.BinaryOp.cs`), as NumPy's
  `try_trivial_single_output_loop` does for a provided out. float16 add/subtract/multiply/divide into an out take the
  SIMD widen-compute-narrow kernels; for those four ops the route also takes broadcast and strided inputs into a C or F
  out (`TryHalfArithBroadcastInto`: the strided widen-compute-narrow kernel walks the out's memory order; an input that
  is the out itself runs in place; any other overlap stays with the iterator's copy). Gates:
  `Backends/Kernels/TrivialBinaryOutTests.cs` (every dtype through both routes, byte-compared, with the declines and the
  overlap rule) and `Math/HalfArithKernelTests.cs`.
- A one-element operand with more dimensions than the other declines the binary, comparison and shift scalar-broadcast
  shortcuts, so the result keeps NumPy's shape (`(7,) op (1,1)` is `(1,7)`). Gate:
  `Backends/Kernels/OneElementOperandRankTests.cs`.

## `np.fix`, `np.fabs`

- `fix(x, out=None)` is a verbatim delegation to `trunc(x, out=out)` (NumPy's `_ufunclike_impl.py`): dtype preservation
  (int/bool identity, floats truncated), NaN/±inf pass-through, the sign of `-0.0`, and trunc's error surface (complex
  and text raise trunc's `TypeError` texts, which name `trunc`, as NumPy's fix leaks them). No `where`/`dtype` parameter.
  The statement is `TensorEngine.Fix` (`Backends/TensorEngine.UfuncLike.cs`; `Backends/Default/Math/Default.Fix.cs` calls
  the operand engine's `Truncate`); the facade `Math/np.fix.cs` keeps the matrix rule. Gated by the `unary` tier's `trunc`
  corpus through the `fix → trunc` alias and `Math/np.fix.Test.cs`.
- `fabs` is the float-only absolute value (NumPy's `TD(flts, f='fabs', astype={'e':'f'})`): it promotes bool/int to
  float (NEP 50: bool/int8/uint8 → float16, int16/uint16/char → float32, int32/uint32/int64/uint64 → float64;
  float16/float32/float64/decimal kept) and has no complex loop. The operation is `abs` on the float loops (clear the
  sign bit), so `UnaryOp.Fabs` aliases `UnaryOp.Abs` at every kernel emit site; it is a separate enum value only so that
  `UfuncName(UnaryOp.Fabs) == "fabs"` names the ufunc in errors. The promoting path casts int→float before the abs
  (`fabs(int.MinValue)` is the exact float magnitude). Bit-exact with NumPy, including `-0.0 → +0.0`, `±inf → +inf` and a
  negative NaN whose sign bit is cleared with its payload kept. Complex is refused with NumPy's three error paths: no
  `dtype=` → `TypeError` "ufunc 'fabs' not supported for the input types…"; `dtype=` float → "Cannot cast ufunc 'fabs'
  input from complex128 to <float> with casting rule 'same_kind'"; `dtype=` complex/int → "No loop matching the
  specified signature and casting was found for ufunc fabs". Gates: `Math/np.fabs.Test.cs`, the `unary`/`errors_full`
  tiers.

## `np.clip`

A port of `numpy/_core/_methods.py::_clip` (behind `np.clip` and `ndarray.clip`), of `fromnumeric.clip`'s keyword rule
and of the ufunc call it routes to. Signature `clip(a, a_min=null, a_max=null, out=null, dtype=null, min=null, max=null,
where=null)`; `ndarray.clip(min, max, out, dtype, where)`. A 0-d integer bound (any C# integer literal, or a char) is
NumPy's Python int.
- Keyword rule (raised before any ufunc error, a read-only out included): `min=`/`max=` alone are the bounds; both
  positionals plus a keyword → ValueError "Passing `min` or `max` keyword argument when `a_min` and `a_max` are provided
  is forbidden."; one positional plus a keyword → TypeError "clip() missing 1 required positional argument: 'a_min'" (or
  'a_max'). C# has no "omitted" distinct from null, so a null positional counts as omitted only when a keyword is given
  (`[Misaligned]`: `np.clip(a, 1)` is legal here).
- Drop rule: on an integer array, a bound at or past `iinfo(a.dtype)` (`min <= iinfo.min`, `max >= iinfo.max`) cannot
  clip anything and is dropped: `np.clip(uint8, 0, 256)` returns the array unchanged, `np.clip(uint8, -1, 100)` only
  clips from above.
- Routing: no bound left → `positive` (a typed copy with positive's rules: `np.clip(bool_array)` raises its no-loop
  TypeError); one bound → `minimum`/`maximum`; two → the clip kernel. The routed name is what every cast error quotes.
  `where=` and `dtype=` ride along.
- Packing: a kept bound that does not fit the LOOP dtype raises OverflowError "Python integer 1000 out of bounds for int8"
  before any element is read, so an empty array raises too (`NDExprTypeRules.CheckPyIntPacksIntoLoop`, NumPy's
  `@TYPE@_safe_pyint_setitem`).
- Range vs loop: the drop rule uses the INPUT's range, while `dtype=` moves the packing: `np.clip(int8, 1000, 2000,
  dtype: np.int16)` is legal (all 1000).
- Platform texts: past ±2³¹ NumPy's text follows the platform C `long`; NumSharp pins the LP64 (Linux/macOS) texts. A
  value past long.MaxValue reports "Python int too large to convert to C long", except into uint32/uint64.
- Casts: `dtype=` validates each input same_kind, in operand order (a Python int reaches any integer or inexact loop but
  not bool, named int64), so `np.clip(f64, min: 3.0, dtype: int32)` raises. `out=` accepts any same_kind-reachable dtype
  (computed at the loop dtype, then cast in through the iterator's buffers). A given `out` is returned itself.
- Broadcasting (NpyIter's joint broadcast): the array, both bounds, `out` and `where` broadcast together: a bound may be
  larger than the array (`np.clip(arange(3), zeros((2,3)), 1)` is (2,3)), and an out larger than every input receives
  the broadcast; out itself never stretches. NumPy's texts: "operands could not be broadcast together with shapes (3,)
  (4,) () (3,) " (every operand, 0-d as `()`, trailing space) and "non-broadcastable output operand with shape (3,)
  doesn't match the broadcast shape (2,3)".
- `where=` must be bool (`ArgumentException` "Cannot cast array data from dtype('int64') to dtype('bool') according to
  the rule 'safe'", NumPy's TypeError text). Masked-off positions of a given out keep their contents; of a fresh result
  they are uninitialized, as in NumPy.
- Overlap: an out that overlaps an input other than as its exact elementwise alias (`np.clip(x[:-1], 1, 4, out=x[1:])`)
  is computed as if from a copy (NDIter COPY_IF_OVERLAP); the in-place `np.clip(x, lo, hi, out=x)` is not copied. With
  `where=` the out is READWRITE, so masked-off positions come back intact.
- Error order: keyword rule → read-only out → where= dtype → packing → dtype= input casts → out cast → broadcast / out
  shape.
- Promotion: 0-d bounds use the ufunc array-scalar table (`np._FindCommonArrayScalarType`), so a Char array with a float
  bound promotes to float64 instead of truncating the bound into char.
- The float loop: NumPy's clip loop runs its CONST fast path (`_npy_clip_const_minmax_`: `x < lo ? lo : x; x > hi ? hi :
  x`; a tie keeps the element, so `np.clip(-0.0, 0, 1)` is -0.0; a NaN bound fills the call with that NaN's bits)
  whenever both bound strides of an inner-loop call are 0, else the general `MIN(MAX())` (a tie takes the bound). Which
  one runs depends on NumPy's trivial-loop test and NpyIter's axis order, coalescing and buffering; a (2,1) bound against
  a (2,3) array runs const when NpyIter does not buffer it, and a one-element call off the trivial loop runs const.
  `Default.ClipNDArray.ConstLoop.cs` asks the shared port of that decision (`DefaultEngine.UfuncInnerSteps.cs`). On x86
  the const compare-selects are `MAXPD(lo, x)` / `MINPD(hi, x)` (the second operand on a tie or a NaN).
- Result layout: a fresh result takes NumPy's ufunc output layout (`Shape.NpyIterOutputShape`: the operands' memory
  order, the mask votes, C wins ties); a dense F or permuted input is clipped as a C run and relabelled
  (`NDArray.AdoptLayout`).
- Two execution routes (`Backends/Default/Math/Default.ClipNDArray{,.Iter,.ConstLoop}.cs`): the FLAT route (no `where=`,
  every operand one dense run after the layout permutation, no harmful overlap) runs the whole-array kernel
  (`DirectILKernelGenerator.Clip.cs`: an output-alignment peel, then aligned stores; past 16 MiB out of place,
  non-temporal stores + sfence); everything else is NumPy's structure, ONE NDIter over (a, lo, hi, out[, where]) driving
  a Tier-3B clip kernel (`DirectILKernelGenerator.Clip.InnerLoop.cs`) through the broadcast-pattern stages
  (`kernels.md`). The flat route reads scalar bounds in place or converts them into stack slots (no 0-d astype per call);
  kernel bodies and op-flag arrays are cached per key.
- Divergences: a strong 0-d NDArray bound (NumPy's `np.int64(5)` / `np.array(5)`) cannot be told from a
  literal, so it is treated as a Python int; the keyword rule's null-as-omitted reading; `where=` with an overlapping out
  of ANOTHER dtype (NumPy round-trips the masked-off values through the loop dtype, NumSharp keeps them exactly); the
  `where=` mask's error TYPE; `np.broadcast_to` / `np.broadcast_arrays` keep an extent-1 axis's original stride where
  NumPy writes 0 (only a one-element bound built by broadcast_to can change clip's const-loop decision through it);
  `NDExpr.Clip` (np.evaluate) has no drop rule, throws on two null bounds and on an out-of-range literal, and runs the
  general loop on ties.
- Gates: the `clip_pyint` and `clip_ufunc` tiers, `Math/np.clip.{PythonIntBounds,Ufunc}.Test.cs`,
  `Backends/Kernels/BroadcastPatternStageTests.cs`.

## The divmod family

- `np.remainder` is an exact alias of `np.mod` (floored remainder; the result takes the divisor's sign).
- `np.fmod` is `BinaryOp.Fmod`, the C remainder (truncated division; the result takes the dividend's sign): `fmod(-7,3)
  == -1` where `mod(-7,3) == 2`. It rides the `Mod`/`FloorDivide` seam (`ExecuteBinaryOp` → the scalar per-element kernel
  `EmitFmodOperation` → `NDDivision.Fmod*`), so it inherits every layout, NEP 50 promotion (an integer stays integer,
  bool → int8), out=/where=/dtype= and every dtype. Integer `fmod` is C# `%` with the ÷0 → 0 and int/long MIN % -1
  guards; float `fmod` is C# `a % b` (bit-identical to C `fmod`); complex is refused (NumPy's TypeError).
- `np.divmod` returns `(floor_divide(x1, x2), remainder(x1, x2))`, byte for byte across ÷0 → (±inf, nan) / (0, 0) for
  integers, signed MIN / -1 → (MIN, 0), ±inf, nan and -0.0. Its `out=(q, r)` / `where=` / `dtype=` path composes
  `FloorDivide` and `Mod`; the common path is one fused IL kernel (`DirectILKernelGenerator.DivMod.cs`: a 4×-unrolled
  scalar two-in/two-out loop over `NDDivision.Divmod*(a, b, out mod)` for contiguous, promoted operands). The integer
  helpers use one division (`q = n / d; r = n - q * d`).
- float16 `mod`/`floor_divide`/`fmod` compute in float32 (NumPy's HALF loop `astype 'e'->'f'`) through the
  `NDDivision.*Single` helpers, so float16 ÷0 gives ±inf, bit-exact.
- Gates: the `divmod_power`/`multioutput` tiers and the Math unit tests.

## `np.float_power`

Power at a minimum precision of float64 (NumPy's `dd->d`/`DD->D` loops only). `Default.FloatPower` resolves the float
loop (complex128 if either operand is complex, else float64) and runs the power kernel on that loop
(`ExecuteBinaryOp(BinaryOp.Power, …)`, `Math.Pow` / `ComplexPowNumPy`), never np.power's scalar-exponent fast path:
NumPy's float_power loop has none, so `float_power(757.7617475757002, 2.0)` is pow's `...03c2`, not `x*x`'s `...03c1`.
- Every real input (bool, integers, float16, float32, decimal, char) promotes to float64 and a complex operand to
  complex128; with no integer loop, a negative integer exponent is legal (`float_power(2, -1) = 0.5`).
- `dtype=` may select only float64 or complex128 (else "No loop matching…for ufunc float_power"); out=/where= and every
  layout ride Power's machinery.
- Error order (NumPy's): read-only out → non-bool where → dtype no loop → complex input cannot cast to float64 → out cast
  → shape, raised with the `float_power` name.
- Complex inherits power's complex divergence (F5: `npy_cpow` against the host `cpow`, ~ULP for non-integer and large
  integer exponents, plus inf/NaN edges): bit-exact on the integer-exponent branch, excused otherwise.
- pow on every element on both sides: parity is the speed ceiling (no SIMD `pow` exists).
- Gates: the `divmod_power`/`out_where`/`specials`/`power_fast` tiers, `Math/np.float_power.Test.cs`.

## `np.power`'s float loops

NumPy 2.4.2's DOUBLE_power / FLOAT_power test the exponent's STEP in each inner-loop call: with `steps[1] == 0` they read
the exponent once and write `1.0/x`, `1.0`, `sqrt(x)`, `x` (raw bits; a signalling NaN kept) or `x*x` for -1, 0, 0.5,
1, 2 (compared in the loop dtype: -0.0 takes the 0 branch, NaN none); any other exponent or a nonzero step is pow. The
answers differ (`757.7617475757002 ** 2`: `x*x` `...03c1`, pow `...03c2`; `-0.0 ** 0.5`: sqrt -0.0, pow +0.0;
`-inf ** 0.5`: sqrt NaN, pow +inf), so which calls the ufunc makes is part of the answer.
- The decision is ported once (`Backends/Default/Math/DefaultEngine.UfuncInnerSteps.cs`, `NumPyUfuncInputSteps`, shared
  with np.clip's const loop): `check_for_trivial_loop` (a cast or unaligned 0-d / ≤ bufsize 1-D input is copied; any
  other, or an out of another dtype or unaligned, refuses the trivial loop), `try_trivial_single_output_loop` (a 0-d input
  has step 0, a 1-D operand its own stride, an N-D operand the item size), then NpyIter (fill_axisdata, the axis sort,
  flips, COPY_IF_OVERLAP, coalescing, `npyiter_find_buffering_setup`'s cost model with `np.getbufsize()`; a read-only 0-d
  cast input is copied eagerly; a buffered single-stride broadcast keeps step 0). The decision is uniform over a call.
  Consequences: a 0-d exponent is the fast path on every route (plain, out=, where=); `(1,) ** (1,)` is pow (the trivial
  loop's 1-D step) but `(5,) ** (1,)` and a one-element call with a cast out are fast (NpyIter, step 0); a column
  exponent `(R,1)` against `(R,n)` is fast only when NpyIter leaves it unbuffered (n > bufsize / 2 at the default; with
  every input cast, rows of exactly bufsize stop the core and stay fast).
- Execution (`Default.Power.StrideZero.cs`, `PowerFloatLoop`): a single-valued exponent with a fast value runs the SIMD
  unary kernel (square / sqrt / reciprocal / positive) or a fill with 1, out= included, where= as compute-then-masked
  np.copyto; a varying exponent that NumPy's calls see with step 0 runs one NDIter pass with NumPy's per-call branch per
  element and, when both inputs are the loop dtype, a vector body (`PowerStrideZeroVector128/256/512`, generic over
  float/double: the broadcast loops hand it a uniform exponent vector, one compare selects one vector op); everything
  else is pow. float16 (HALF_power) and np.float_power have no fast path.
- Open divergences: np.evaluate's Power node has no fast path, so `np.kaiser` differs from NumPy by 1–2 ULP in some
  windows; `np.power(unsigned, -1)` raises ValueError where NumPy raises OverflowError.
- Gates: the `power_fast` tier (zero excuses), `Math/np.power.StrideZero.Test.cs`.

## `np.gcd` / `np.lcm`

NumPy's `npy_gcd@c@`/`npy_lcm@c@` and the `@TYPE@_gcd`/`@TYPE@_lcm` integer loops. Integer-only: bool, half, single,
double, decimal, complex and the uint64+signed → float64 NEP 50 pair name no loop and raise `ufunc '{name}' did not
contain a loop with signature matching types (…, …) -> None` (TypeError; the `dtype=` variant closes with
`-> {dtype}DType`). Both operands and the output share one integer dtype; a bool with an integer promotes to that
integer's loop. `BinaryOp.Gcd`/`BinaryOp.Lcm` ride the `Fmod` seam (`EmitGcdLcmOperation` → `Utilities/NDGcdLcm.{Gcd,Lcm}*`,
one helper per integer dtype including Char), so they inherit every layout, out=/where=/dtype= and weak scalars.
- The helpers port NumPy: an unsigned Euclidean core (`while (a != 0) { c = a; a = b % a; b = c; }`), signed = the core
  on the magnitudes reinterpreted back, `lcm = gcd == 0 ? 0 : |a| / gcd * |b|`. A signed result can be negative when the
  magnitude wraps (`gcd(int8 -128, -128) == -128`, `lcm(int32.Min, 1) == int32.Min`), and the lcm product wraps the dtype
  (`lcm(int16 21000, 14000) == -23536`).
- No SIMD (a data-dependent loop), like NumPy: there is no vector integer division or vector TZCNT, and binary/Stein's
  GCD is slower than division-based Euclid on x86. Parity is the speed ceiling.
- Gates: `Math/np.gcd.Test.cs`, the `gcd` tier.

## Bitwise

`bitwise_and`, `bitwise_or`, `bitwise_xor` (out=/where=/dtype= supported), `invert`, `left_shift`, `right_shift`.
Float/complex/decimal INPUTS raise NumPy's coercion TypeError, while a float/complex/decimal `dtype=` raises the no-loop
text (two different messages). Error order: a bad `where` → no loop → the out cast → the shape.
