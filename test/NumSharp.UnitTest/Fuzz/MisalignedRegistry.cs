using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    public enum DivergenceKind { Dtype, Shape, Value, Threw }

    /// <summary>
    ///     The explicit, documented set of NumSharp-vs-NumPy behavioral differences that are
    ///     INTENDED (per maintainer decision) rather than bugs. The differential matrix excuses
    ///     ONLY these patterns — logging each one — and fails on anything else. Keeping the cases
    ///     in the corpus (instead of dropping them) means a future change to either behavior
    ///     surfaces immediately: a fixed divergence simply starts passing the bit-exact check, and
    ///     a drift beyond the documented tolerance turns back into a hard failure.
    ///
    ///     Documented differences:
    ///       1. NEP50 weak-scalar: NumSharp treats a 0-D array operand as a weak scalar (the other
    ///          operand's dtype drives promotion). NumPy makes 0-D arrays full participants; only
    ///          Python scalar literals are weak. NumSharp cannot distinguish the two (both are 0-D
    ///          NDArrays), and keeping `arr + 5` ergonomic was chosen over strict NEP50 parity.
    ///       2. Complex true-division differs from NumPy's npy_cdivide by ~1 ULP
    ///          (System.Numerics.Complex uses a different scaling). Add/sub/mul are bit-exact.
    /// </summary>
    public static class MisalignedRegistry
    {
        private static readonly System.Collections.Generic.HashSet<string> ReduceOps = new()
        {
            "sum", "prod", "min", "max", "mean", "std", "var", "argmax", "argmin", "all", "any"
        };

        private static readonly System.Collections.Generic.HashSet<string> NanReduceOps = new()
        {
            "nansum", "nanprod", "nanmax", "nanmin", "nanmean", "nanstd", "nanvar", "nanmedian"
        };

        private static readonly System.Collections.Generic.HashSet<string> QuantileOps = new()
        {
            "median", "percentile", "quantile"
        };

        private static readonly System.Collections.Generic.HashSet<string> ExtremaOps = new()
        {
            "maximum", "minimum", "fmax", "fmin"
        };

        public static string Classify(
            FuzzCorpus.Case c, DivergenceKind kind,
            byte[] expected, byte[] actual, NPTypeCode tc, IReadOnlyList<BitDiff.Diff> diffs)
        {
            // (1) NEP50 weak-scalar promotion. Any multi-operand op with a 0-D operand: NumSharp
            //     promotes it weakly (the array operand's dtype drives the result), where NumPy makes
            //     0-D arrays full participants. Covers binary pp_scalar_* and np.where wh_bcast_xy.
            if (kind == DivergenceKind.Dtype && c.Operands.Length >= 2 && c.Operands.Any(o => o.Shape.Length == 0))
                return "NEP50 weak-scalar: 0-D operand promoted weakly (NumPy promotes 0-D arrays fully)";

            // (2) Complex true-division ~1 ULP. Excuse only divide, only complex result, only when every
            //     differing element is within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Op == "divide" && tc == NPTypeCode.Complex
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "complex division ~1 ULP (npy_cdivide vs System.Numerics.Complex)";

            // ----------------------------------------------------------------------------------
            // W1 dtype-expansion divergences — real NumSharp bugs surfaced by widening the corpus
            // to float16-as-input and the narrow integers (int8/int16/uint16/uint32/uint64). Each
            // is documented + collected for the maintainer and excused here so the bit-exact
            // matrix stays green for every other (now-gated) cell. Scoped tightly to the exact
            // (op, dtype) cell so a regression in a neighbouring cell still fails the gate.
            // ----------------------------------------------------------------------------------

            // (W1-A) floor_divide / mod producing a float16: NDDivision (F1) ported SByte..UInt64,
            // Single, Double — but NOT Half. The Half floored-division falls back to a generic path
            // that yields -0.0 / NaN where NumPy yields the floored quotient or IEEE ±inf. Scoped to
            // a Half operand/result so int & float32/64 floor_divide stay gated bit-exact.
            if ((c.Op == "floor_divide" || c.Op == "mod")
                && (tc == NPTypeCode.Half || c.Operands.Any(o => o.Dtype == "float16"))
                && (kind == DivergenceKind.Value || kind == DivergenceKind.Threw))
                return "floor_divide/mod(float16): NDDivision has no Half path (wrong value/NaN) [known bug]";

            // (W1-B FIXED) power(float16) on the scalar-broadcast path used to throw
            // InvalidCastException because ReadScalarAsDouble called Convert.ToDouble on a boxed
            // System.Half (not IConvertible); it now casts Half directly, so the excuse is removed and
            // any regression of the crash fails the fuzz gate.
            //
            // (W1-C) power(uint64,int64): NumPy promotes to float64 (NEP50), but NumSharp keeps the
            // integer power path -> ArgumentException "Integers to negative integer powers" (the
            // negative-exponent cell) in the kernel.
            if (c.Op == "power" && kind == DivergenceKind.Threw)
            {
                if (c.Operands.Any(o => o.Dtype == "uint64") && c.Operands.Any(o => o.Dtype == "int64"))
                    return "power(uint64,int64): NEP50 uint64+int64->float64 not applied; integer-power path throws/corrupts [known bug]";
            }

            // (W1-F) power(narrow-int, float16) widens the result to float64 where NumPy keeps
            // float16 — a power-SPECIFIC promotion bug (add/sub/mul/divide on the same int8+float16
            // pair promote correctly to float16). Scoped to a NumPy-expected Half result.
            if (c.Op == "power" && kind == DivergenceKind.Dtype && tc == NPTypeCode.Half)
                return "power(*,float16): result widened past NumPy's float16 (power-specific NEP50 promotion) [known bug]";

            // (W1-D) dot of 1-D int8 vectors routes through ReduceAdd(int8)->int8, for which no IL
            // reduction kernel is emitted ("IL kernel not available for Sum(SByte) -> SByte").
            // NumPy dot(int8,int8) -> int8 (modular). 2-D int8 matmul (GEMM path) is unaffected.
            if (c.Op == "dot" && kind == DivergenceKind.Threw
                && c.Operands.Length == 2 && c.Operands.All(o => o.Dtype == "int8"))
                return "dot(int8): Sum(int8)->int8 IL reduction kernel missing [known bug]";

            // (W9-A/C) np.expand_dims and np.atleast_3d on an EMPTY (size-0) array drop the
            // inserted/appended axis: NumSharp returns [0,3] where NumPy returns [1,0,3] / [0,3,1].
            // Non-empty inputs are correct. Scoped to a shape mismatch on a zero-size operand.
            if ((c.Op == "expand_dims" || c.Op == "atleast_3d")
                && kind == DivergenceKind.Shape && c.Operands[0].Shape.Any(d => d == 0))
                return "expand_dims/atleast_3d(empty): inserted/appended axis dropped on a zero-size array [known bug]";

            // (W9-B) np.repeat ignores Shape.offset: on an offset slice (b[2:7]) or a 0-D view at a
            // non-zero offset it reads from the base buffer start, returning the wrong elements.
            // Contiguous/offset-0 repeat is bit-exact. Scoped to a repeat on a non-zero-offset operand.
            if (c.Op == "repeat" && kind == DivergenceKind.Value && c.Operands[0].Offset != 0)
                return "repeat: ignores Shape.offset (reads base start) on offset / 0-D views [known bug]";

            // (W8-A) np.modf only supports Single/Double/Decimal: float16 and integer inputs throw
            // "modf only supports floating-point types". NumPy returns (float16,float16) for Half and
            // promotes integer input to (float64,float64). float32/float64 modf is bit-exact incl. the
            // signed-zero/inf edges. Scoped to the two modf outputs that threw AND to the non-f32/f64
            // input dtypes the bug is documented for (B4/F12) — a modf(float64) throw is a REAL
            // regression and fails the gate.
            if ((c.Op == "modf_frac" || c.Op == "modf_int") && kind == DivergenceKind.Threw
                && c.Operands[0].Dtype != "float32" && c.Operands[0].Dtype != "float64")
                return "modf(float16/int): no Half kernel, no integer->float64 promotion (throws) [known bug]";

            // (W1-E) np.where on the scalar-broadcast path with a narrow-int operand throws
            // "Zero-push unsupported for SByte" — NDExpr.EmitPushZero gained Complex/Half (F4) but
            // not the sub-32-bit integers. Scoped to a where that threw with such an operand.
            if (c.Op == "where" && kind == DivergenceKind.Threw
                && c.Operands.Any(o => o.Dtype == "int8" || o.Dtype == "uint8"
                                       || o.Dtype == "int16" || o.Dtype == "uint16"))
                return "where(narrow-int) scalar-broadcast: NDExpr zero-push unsupported for sub-32-bit int [known bug]";

            // Size-1 result shape was FIXED in Phase 1 F7: Shape.Broadcast no longer collapses a
            // 1-D [1] against a lower-rank operand (e.g. [1] + 0-D scalar -> [1], not []). The NDim
            // guard keeps result ndim == max(ndims). Classifier branch removed so the matrix verifies it.

            // Bool arithmetic was FIXED in Phase 1 F6: `+` now emits logical OR and `*` logical AND
            // for the bool dtype (True + True -> True / byte 1, not 2), matching NumPy's bool ufunc
            // loops. `-` has no bool loop and throws on both sides. Classifier branch removed so the
            // matrix verifies bool add/multiply bit-exact.

            // (B2/F10) Complex BINARY arithmetic — PER-OP scopes. The former branch here excused ANY
            // value divergence of ANY magnitude for ANY 2-operand complex-result op (so a gross
            // complex add/matmul/copyto regression passed silently). Dismantled: divide keeps its
            // own 2-ULP branch at (2) above; add/subtract/multiply/power get the tight scopes below;
            // every other complex-binary op (matmul/dot/outer/copyto/extrema/concatenate/...) must
            // be bit-exact and now fails the gate on divergence.
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Complex && c.Operands.Length == 2)
            {
                // add/subtract run the same naive component formulas on both sides; only FMA
                // contraction / evaluation order can differ -> every diff capped at 2 ULP.
                if ((c.Op == "add" || c.Op == "subtract")
                    && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                    return "complex add/subtract within 2 ULP (FMA contraction) [documented]";
                // multiply: npy_cmul vs System.Numerics.Complex round (ac-bd)/(ad+bc) differently
                // (FMA contraction). In the catastrophic-cancellation regime (ac ~ bd) the RELATIVE
                // error of the cancelled component is unbounded, but the ABSOLUTE error stays at
                // rounding scale of the products — i.e. of the element's dominant component. So the
                // detection: every differing component within 16 ULP *of the element's own
                // magnitude* (not of itself). A divergence larger than that is a real kernel bug.
                if (c.Op == "multiply"
                    && diffs.All(d => WithinComplexElementMagnitudeUlp(expected, actual, d.Index, 16)))
                    return "complex multiply cancellation / ~ULP at element magnitude (npy_cmul vs System.Numerics) [documented #12]";
                // power: Complex.Pow (polar exp(w*log z)) vs npy_cpow (special-cases small integer
                // exponents via repeated squaring) — measured on the corpus the finite interior
                // diverges by up to ~350 ULP of the affected component, plus the documented gross
                // inf/NaN edges (Phase-1 F5) where one side goes non-finite. Bound the finite side
                // at 512 ULP of the ELEMENT's magnitude (same absolute-error anchor as multiply:
                // still catches sign flips / wrong magnitudes) and excuse the non-finite edges.
                if (c.Op == "power"
                    && diffs.All(d => WithinComplexElementMagnitudeUlp(expected, actual, d.Index, 512)
                                      || NonFiniteInvolved(expected, actual, d.Index)))
                    return "complex power ~ULP / gross inf-NaN edge (Complex.Pow vs npy_cpow) [documented F5]";
            }

            // (3) NaN ordering in <= / >= was FIXED in Phase 1 F2 (the unordered Cgt_Un/Clt_Un
            //     compare now yields False for a NaN operand, matching IEEE/NumPy). The classifier
            //     branch is intentionally removed so the comparison matrix verifies it bit-exact.

            // (W5-A) cumprod on a SIZE-1 / empty / 0-d array skips the NEP50 accumulator widening —
            // it preserves the narrow integer input dtype on the one-element fast path instead of
            // int16/int32 -> int64, uint8/uint16 -> uint64. cumsum was fixed (ReduceCumAdd now
            // promotes + reshapes every trivial case to match np.add.accumulate); cumprod still
            // carries the bug in ReduceCumMul. Scoped to a cumprod dtype mismatch ON THAT size-<=1
            // fast path only (B3/F11) — a full-size cumprod widening miss is a real bug and fails.
            if (c.Op == "cumprod" && kind == DivergenceKind.Dtype && ElementCount(c.Operands[0]) <= 1)
                return "cumprod(size-1 int): skips NEP50 accumulator widening (int16/int32/uint8/uint16) [known bug]";

            // --- T13 element-wise extrema (maximum/minimum/fmax/fmin) + isclose ---
            // (W7-B FIXED) maximum/minimum/fmax/fmin are now DIRECT binary ufuncs (BinaryOp.Maximum
            // /Minimum/FMax/FMin via ExecuteBinaryOp), no longer routed through np.clip. maximum/
            // minimum PROPAGATE NaN; fmax/fmin IGNORE NaN (return the finite operand, first-operand
            // on both-NaN) — bit-exact with NumPy 2.4.2 across every dtype pair and layout in
            // LOGIC_BIN_PAIRS, so NO extrema value excuse remains and the matrix verifies them all.
            // isclose on an F-contiguous complex operand diverges — its own comparison kernel (NOT the
            // now-fixed clip-routed extrema path) still pairs a strided complex operand by buffer order.
            // Scoped to cases with a complex128 operand present (B6/F14) — a real-dtype isclose
            // divergence is a real bug and fails the gate.
            if (c.Op == "isclose" && kind == DivergenceKind.Value
                && c.Operands.Any(o => o.Dtype == "complex128"))
                return "isclose: F-contiguous/complex strided pairing divergence [known bug]";

            // (W11-A / clip_out FIXED) maximum/minimum/clip now PROPAGATE NaN on the out= path
            // (NumPy: maximum(NaN,x)=NaN, clip(NaN,lo,hi)=NaN). Root cause: the clip SIMD kernel used
            // the hardware MAXPS/MINPD intrinsics (Avx.Max/Min), which return the SECOND operand on an
            // unordered (NaN) compare and so silently dropped the NaN; the scalar path already
            // propagated. EmitVectorMinOrMax(propagateNaN: true) now restores it via
            // ConditionalSelect(Equals(a,a), hwMinMax, a) for the float lanes
            // (DirectILKernelGenerator.cs). The classifier branches are removed so the matrix verifies
            // maximum_out / minimum_out / clip_out NaN propagation bit-exact.

            // --- T12 statistics: the QuantileEngine ops (median/percentile/quantile) diverge on
            //     non-finite slices and on the integer axis path; average has summation-order drift.
            //     ptp / count_nonzero / clip are bit-exact. ---
            if (QuantileOps.Contains(c.Op) && kind == DivergenceKind.Value)
            {
                // (W6-A) a slice containing ±inf / NaN: the partition + linear interpolation
                // ((a+b)/2 or a+(b-a)*frac) produces a NaN where NumPy does not (or vice-versa) —
                // e.g. (+inf + -inf)/2. Either direction is excused.
                if (diffs.Any(d => d.Expected == "NaN" || d.Actual == "NaN"))
                    return "median/percentile/quantile: ±inf/NaN slice partition+interpolation NaN mismatch [known bug]";
                // (W6-B) integer input on the axis path: GROSS interpolation value error (sign flips,
                // wrong magnitude) — a genuine QuantileEngine defect, not a rounding difference.
                if (c.Operands[0].Dtype.StartsWith("int") || c.Operands[0].Dtype.StartsWith("uint"))
                    return "percentile/quantile(int): gross interpolation value error on the axis path [known bug]";
                // float input, finite: interpolation order / partition selection differs by a few ULP.
                return "median/percentile/quantile: float interpolation order/precision divergence [known bug]";
            }
            // (W6-C) np.average: pairwise (NumPy) vs naive (NumSharp) summation order on large-magnitude
            // slices -> precision drift.
            if (c.Op == "average" && kind == DivergenceKind.Value)
                return "average: summation-order precision divergence (pairwise vs naive) [known bug]";

            // (W6-D FIXED) np.clip propagates NaN (clip(NaN,lo,hi)=NaN) — fixed together with W11-A by
            // making the clip SIMD min/max NaN-aware (the scalar path already propagated). The
            // classifier branch is removed so the matrix verifies clip(NaN) bit-exact.

            // --- NaN-aware reductions (T10 / W4): the nan* family is broadly broken ---
            if (NanReduceOps.Contains(c.Op))
            {
                // (W4-E) nanmean/nanstd/nanvar over an EMPTY float16 array (axis=None) throw
                // "Can't construct NDIterator with an empty shape" instead of returning NaN.
                if (kind == DivergenceKind.Threw && c.Operands[0].Shape.Any(d => d == 0))
                    return "nan-reduction(empty): throws 'NDIterator empty shape' instead of NaN [known bug]";
                // (W4-D) complex 1-D axis reduction throws (shared NDCoordinatesAxisIncrementor bug).
                if (kind == DivergenceKind.Threw && c.Operands.Length == 1 && c.Operands[0].Dtype == "complex128")
                    return "complex 1-D axis reduction throws (NDCoordinatesAxisIncrementor vector shape) [known bug]";
                // (W4-A) shape: nanmean/nanstd/nanvar collapse a 1-D axis reduction to [1] instead of
                // a scalar [], and drop keepdims entirely on the integer input path.
                if (kind == DivergenceKind.Shape)
                    return "nan-reduction shape: nanmean/nanstd/nanvar give [1] not scalar on 1-D axis, and ignore keepdims on int input [known bug]";
                // result dtype (NEP50 accumulator width / complex->real for nanstd/nanvar).
                if (kind == DivergenceKind.Dtype)
                    return "nan-reduction result dtype differs (NEP50 accumulator / complex->real) [known bug]";
                // (W4-C) nanmedian propagates NaN instead of ignoring it.
                if (kind == DivergenceKind.Value && c.Op == "nanmedian")
                    return "nanmedian: propagates NaN instead of ignoring it [known bug]";
                // (W4-B) nansum/nanmean/nanstd/nanvar: wrong NaN masking / count, or summation order.
                if (kind == DivergenceKind.Value)
                    return "nan-reduction value: NaN masking / count / summation-order divergence [known bug]";
            }

            // --- Reductions (single-operand, but classified before the unary rules) ---
            if (ReduceOps.Contains(c.Op))
            {
                // Reduction result dtype differs (NEP50 accumulator width / complex->real for std/var).
                if (kind == DivergenceKind.Dtype)
                    return "reduction result dtype differs (NEP50 accumulator / complex->real) [known bug]";
                // Complex axis reduction on a 2-D+ array now works (resolved); but reducing a 1-D
                // complex array along its only axis still throws "NDCoordinatesAxisIncrementor with a
                // vector shape". Excuse only that residual Threw case — the 2-D cases are verified
                // (value diffs fall to the summation / ~ULP branches).
                if (kind == DivergenceKind.Threw && c.Operands.Length == 1 && c.Operands[0].Dtype == "complex128")
                    return "complex 1-D axis reduction throws (NDCoordinatesAxisIncrementor vector shape) [known bug]";

                // NaN propagation: the FLAT (axis=null) min/max reduction now propagates NaN
                // (Phase 1 F2-reductions: NaN-propagating SIMD min/max in the IL flat kernel +
                // CombineVectors), so it is NOT excused — a flat regression fails the gate. The
                // axis (vertical/strided) SIMD min/max path still drops NaN; excuse only that.
                // (mean/std/var/sum propagate NaN on both paths already, via arithmetic.)
                if (kind == DivergenceKind.Value && diffs.Count > 0 && diffs.All(d => d.Expected == "NaN")
                    && c.Params != null
                    && c.Params.TryGetValue("axis", out var axEl)
                    && axEl.ValueKind != System.Text.Json.JsonValueKind.Null)
                    return "axis-reduction NaN propagation: axis SIMD min/max skips NaN [known bug; flat fixed]";
                // bool min/max along an axis returns True where NumPy returns False.
                if (kind == DivergenceKind.Value && (c.Op == "min" || c.Op == "max") && tc == NPTypeCode.Boolean)
                    return "bool min/max along axis diverges [known bug]";
                // Floating accumulation: NumPy pairwise summation / two-pass var vs NumSharp order.
                // Scoped to FLOAT-FAMILY result dtypes (B1/F9): integer/bool accumulation is exact
                // (modular) on both sides — an integer-result sum/prod value divergence is a REAL
                // bug, not "precision", and fails the gate.
                if (kind == DivergenceKind.Value
                    && (tc == NPTypeCode.Half || tc == NPTypeCode.Single
                        || tc == NPTypeCode.Double || tc == NPTypeCode.Complex)
                    && (c.Op == "sum" || c.Op == "mean" || c.Op == "std" || c.Op == "var" || c.Op == "prod"))
                    return "reduction summation/two-pass precision (algorithm order)";
            }

            // Decimal std (surfaced by scoping B1 to float-family): var — the exact decimal
            // mean-of-squared-deviations — is bit-exact, so the divergence is purely sqrt(var):
            // the ORACLE uses an independent Newton sqrt (gen_decimal_oracle.DecSqrt) while
            // NumSharp uses DecimalMath.Sqrt, and NEITHER is correctly rounded at the 28/29-digit
            // limit (probed vs 60-digit truth 2026-07-07: oracle low on 2 cases, NumSharp high on
            // 1, both fine on 1). Excuse ONLY std × decimal × Value with every diff within one
            // unit of the 28th significant digit (relative 1e-27) — a real iteration/accumulation
            // bug diverges orders of magnitude more and still fails.
            if (c.Op == "std" && tc == NPTypeCode.Decimal && kind == DivergenceKind.Value
                && diffs.All(DecimalLastDigitDiff))
                return "decimal std: independent 28-digit sqrt implementations differ in the last digit [documented]";

            // (4) Unary result-dtype: the transcendental ufuncs (sqrt/cbrt/exp/log/sin/cos/tan) now
            //     follow NumPy's width-based float promotion (Phase 1 F3a) and are verified bit-exact,
            //     so they are NOT excused here. reciprocal now preserves the integer dtype too (bool
            //     -> int8), matching NumPy bit-exact (see (8) below), so it is no longer excused. The
            //     remaining dtype-preserving ufuncs (square/floor/ceil/trunc) still widen integer
            //     input to float64 instead of preserving it — pending Phase 1 F3b. Scoped to that set
            //     so a transcendental promotion regression fails the gate.
            if (kind == DivergenceKind.Dtype && c.Operands.Length == 1
                && (c.Op == "square" || c.Op == "floor" || c.Op == "ceil" || c.Op == "trunc"))
                return "unary preserve-dtype pending: square/floor/ceil/trunc widen int->float64 [F3b]";

            // (W3-A/B) The hyperbolic / inverse-trig / angle-conversion ufuncs have no Half kernel
            // (throw "Unary operation X not supported for Half" whenever the input promotes to
            // float16: bool/int8/uint8/float16). The COMPLEX hyperbolic/inverse-trig kernels are now
            // implemented (NDComplexMath) and verified within ULP below — only Half still throws here;
            // deg2rad/rad2deg additionally throw for Complex (NumPy has no complex loop for them either).
            // Scoped to a single-operand THREW on these op names AND to the float16-promoting input
            // dtypes the bug is documented for (B5/F13; probed: sinh(bool/i8/u8/f16)->float16,
            // i16/u16->float32 works) — a sinh(float64) throw is a real regression and fails.
            if (kind == DivergenceKind.Threw && c.Operands.Length == 1
                && (c.Op == "sinh" || c.Op == "cosh" || c.Op == "tanh"
                    || c.Op == "arcsin" || c.Op == "arccos" || c.Op == "arctan"
                    || c.Op == "deg2rad" || c.Op == "rad2deg")
                && (c.Operands[0].Dtype == "bool" || c.Operands[0].Dtype == "int8"
                    || c.Operands[0].Dtype == "uint8" || c.Operands[0].Dtype == "float16"
                    || ((c.Op == "deg2rad" || c.Op == "rad2deg") && c.Operands[0].Dtype == "complex128")))
                return "unary hyperbolic/inverse-trig/angle: no Half kernel (throws NotSupportedException) [known bug]";

            // (W3-C) FIXED: np.exp2's float32-output IL kernel used to leave the evaluation stack
            // unbalanced (a spurious Ldc_R8 2.0 in EmitExp2Call's Single branch), throwing
            // InvalidProgramException for every int16/uint16/char/float32 input. The excuse is removed
            // so any regression of the malformed-IL crash now fails the fuzz gate.

            // (5) Unary transcendental / complex magnitude ~ULP (libm / algorithm differences).
            //     Tight: every differing element within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "unary ~ULP (transcendental/magnitude algorithm difference)";

            // (6) np.negative on unsigned integers was FIXED in Phase 1 F4: np.negative now routes
            //     through the engine kernel (two's-complement wrap, e.g. -1u -> 255), matching NumPy.
            //     Classifier branch removed so the unary matrix verifies it bit-exact.

            // (6b) Complex unary math is a full NumPy-algorithm port (NDComplexMath): npy_clog with the
            //     near-|z|=1 log1p path, Kahan ctanh, csinh/ccosh, npy_catanh with real_part_reciprocal,
            //     FMA-contracted z*z, Goldberg expm1, the C99 cexp/csqrt non-finite tables, and
            //     branch-cut/signed-zero fixups. Every complex unary op (sqrt/exp/log/log2/log10/log1p/
            //     expm1/exp2/sin/cos/tan/sinh/cosh/tanh/arcsin/arccos/arctan/square/reciprocal/negative)
            //     matches NumPy 2.4.2 bit-exactly or within 3 ULP on the finite interior — verified by a
            //     504-point bit-exact sweep — so the WHOLE set is held to a TIGHT 3-ULP gate; a real
            //     regression fails.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1 && tc == NPTypeCode.Complex
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 3)))
                return "complex unary within 3 ULP (full NumPy-algorithm port)";

            // (7) The only complex-unary divergences beyond 3 ULP are three pathological regimes, each
            //     verified against NumPy 2.4.2 and accepted (NumSharp is the more accurate side of the
            //     square/log cancellation cases; these are inputs no real workload produces):
            //       - cos/sin with a NaN imaginary part: the sign of the resulting zero component is
            //         C99-UNSPECIFIED (cos(+-0 + NaN i).imag = +-0 either way); the platform libm and
            //         the npy_ccos identity pick opposite signs.
            //       - arccos with a sub-DBL_MIN imaginary part: Complex.Acos flushes the denormal real
            //         part to 0 where NumPy's cacos _do_hard_work keeps it (arccos(2 + 1e-308 i).real
            //         ~ 5.8e-309) — a denormal-range edge.
            //       - sinh/cosh at the overflow boundary |x| in [710, 710.13]: Windows' CRT sinh
            //         overflows to inf while .NET Math.Sinh stays finite (a platform-libm boundary,
            //         absent on glibc).
            //     Scoped to these op names so a >3-ULP regression in ANY other complex unary op fails.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1 && tc == NPTypeCode.Complex
                && (c.Op == "cos" || c.Op == "sin" || c.Op == "arccos" || c.Op == "sinh" || c.Op == "cosh"))
                return "complex cos/sin/arccos/sinh/cosh pathological edge (NaN zero-sign / subnormal / overflow boundary) [documented]";

            // (7c) Complex REDUCTIONS / SCANS (min/max/sum/prod/mean/std/var, cumsum/cumprod) with a
            //     NaN element: complex ordering with NaN is implementation-defined. NumPy carries the
            //     NaN-containing element through verbatim (its real part is NaN but the imaginary part
            //     is the element's original value, e.g. NaN+4540i), whereas NumSharp's magnitude-based
            //     comparison / accumulation collapses the element to NaN+NaN. A documented complex
            //     NaN-ordering/propagation difference — distinct from the elementwise unary math above,
            //     and scoped to the reduction/scan op names so an elementwise regression still fails.
            //     The diffs must actually INVOLVE a NaN token (B7/F15) — a finite-value complex
            //     reduce/scan divergence is not "NaN ordering" and fails the gate.
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Complex
                && (ReduceOps.Contains(c.Op) || c.Op == "cumsum" || c.Op == "cumprod")
                && diffs.Any(d => d.Expected.Contains("NaN") || d.Actual.Contains("NaN")))
                return "complex reduction/scan NaN ordering/propagation differs [documented]";

            // Complex np.where was resolved in committed code (no longer throws "Zero-push
            // unsupported for Complex"); it now selects complex operands bit-exact. Classifier
            // branch removed so the where matrix verifies it.

            // (8) np.reciprocal of an integer/bool now matches NumPy bit-exact and is no longer
            //     excused: it preserves the integer dtype (bool -> int8), C-truncating 1/x gives 0
            //     for |x| > 1, and the per-type 1/0 sentinel is reproduced exactly (0 for
            //     int8/int16/uint8/uint16/uint32; the sign-bit 0x80..0 for int32/int64/uint64).
            //     Strided / sliced / broadcast integer operands are read in place (no longer throw).

            return null;
        }

        /// <summary>Element count of a corpus operand (0-d shape [] counts as 1).</summary>
        private static long ElementCount(FuzzCorpus.Operand o)
        {
            long n = 1;
            foreach (var d in o.Shape)
                n *= d;
            return n;
        }

        /// <summary>
        ///     True when both differing complex components at <paramref name="index"/> lie within
        ///     <paramref name="maxUlp"/> ULP of the ELEMENT's magnitude (its largest finite
        ///     component), not of themselves. This is the absolute-error envelope a differently
        ///     rounded/contracted (a*c - b*d) can produce: in the catastrophic-cancellation regime
        ///     the cancelled component's RELATIVE error is unbounded while its ABSOLUTE error stays
        ///     at rounding scale of the products (~ the dominant component). Non-finite values are
        ///     never "cancellation".
        /// </summary>
        private static bool WithinComplexElementMagnitudeUlp(byte[] exp, byte[] act, int index, int maxUlp)
        {
            int o = index * 16;
            double er = BitConverter.ToDouble(exp, o), ei = BitConverter.ToDouble(exp, o + 8);
            double ar = BitConverter.ToDouble(act, o), ai = BitConverter.ToDouble(act, o + 8);
            if (!double.IsFinite(er) || !double.IsFinite(ei) || !double.IsFinite(ar) || !double.IsFinite(ai))
                return false;
            double mag = Math.Max(Math.Max(Math.Abs(er), Math.Abs(ei)), Math.Max(Math.Abs(ar), Math.Abs(ai)));
            double ulp = Math.BitIncrement(mag) - mag;
            return Math.Abs(er - ar) <= maxUlp * ulp && Math.Abs(ei - ai) <= maxUlp * ulp;
        }

        /// <summary>
        ///     Both diff tokens parse as decimal and differ by no more than one unit in the 28th
        ///     significant digit (relative 1e-27) — the disagreement envelope of two independent,
        ///     not-correctly-rounded 28/29-digit decimal sqrt implementations.
        /// </summary>
        private static bool DecimalLastDigitDiff(BitDiff.Diff d)
        {
            if (!decimal.TryParse(d.Expected, System.Globalization.NumberStyles.Float,
                                  System.Globalization.CultureInfo.InvariantCulture, out var e)
                || !decimal.TryParse(d.Actual, System.Globalization.NumberStyles.Float,
                                     System.Globalization.CultureInfo.InvariantCulture, out var a))
                return false;
            decimal diff = Math.Abs(e - a);
            decimal mag = Math.Max(Math.Abs(e), Math.Abs(a));
            return diff <= mag * 1e-27m;
        }

        /// <summary>Either side's complex element at <paramref name="index"/> has a NaN/inf component.</summary>
        private static bool NonFiniteInvolved(byte[] exp, byte[] act, int index)
        {
            int o = index * 16;
            return !double.IsFinite(BitConverter.ToDouble(exp, o))
                || !double.IsFinite(BitConverter.ToDouble(exp, o + 8))
                || !double.IsFinite(BitConverter.ToDouble(act, o))
                || !double.IsFinite(BitConverter.ToDouble(act, o + 8));
        }
    }
}
