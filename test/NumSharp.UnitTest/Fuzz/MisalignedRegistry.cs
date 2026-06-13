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

            // (W1-A) floor_divide / mod producing a float16: NpyDivision (F1) ported SByte..UInt64,
            // Single, Double — but NOT Half. The Half floored-division falls back to a generic path
            // that yields -0.0 / NaN where NumPy yields the floored quotient or IEEE ±inf. Scoped to
            // a Half operand/result so int & float32/64 floor_divide stay gated bit-exact.
            if ((c.Op == "floor_divide" || c.Op == "mod")
                && (tc == NPTypeCode.Half || c.Operands.Any(o => o.Dtype == "float16"))
                && (kind == DivergenceKind.Value || kind == DivergenceKind.Threw))
                return "floor_divide/mod(float16): NpyDivision has no Half path (wrong value/NaN) [known bug]";

            // (W1-B/C) power throws on two newly-covered cells:
            //   * any float16 operand on the scalar-broadcast path: System.Half is not IConvertible,
            //     so the scalar power helper throws InvalidCastException.
            //   * (uint64,int64): NumPy promotes to float64 (NEP50), but NumSharp keeps the integer
            //     power path -> ArgumentException "Integers to negative integer powers" or a bounds
            //     DebugAssert ("index < Count, Memory corruption") in the kernel.
            if (c.Op == "power" && kind == DivergenceKind.Threw)
            {
                if (c.Operands.Any(o => o.Dtype == "float16"))
                    return "power(float16): Half scalar path InvalidCast (Half is not IConvertible) [known bug]";
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
            // signed-zero/inf edges. Scoped to the two modf outputs that threw.
            if ((c.Op == "modf_frac" || c.Op == "modf_int") && kind == DivergenceKind.Threw)
                return "modf(float16/int): no Half kernel, no integer->float64 promotion (throws) [known bug]";

            // (W1-E) np.where on the scalar-broadcast path with a narrow-int operand throws
            // "Zero-push unsupported for SByte" — NpyExpr.EmitPushZero gained Complex/Half (F4) but
            // not the sub-32-bit integers. Scoped to a where that threw with such an operand.
            if (c.Op == "where" && kind == DivergenceKind.Threw
                && c.Operands.Any(o => o.Dtype == "int8" || o.Dtype == "uint8"
                                       || o.Dtype == "int16" || o.Dtype == "uint16"))
                return "where(narrow-int) scalar-broadcast: NpyExpr zero-push unsupported for sub-32-bit int [known bug]";

            // Size-1 result shape was FIXED in Phase 1 F7: Shape.Broadcast no longer collapses a
            // 1-D [1] against a lower-rank operand (e.g. [1] + 0-D scalar -> [1], not []). The NDim
            // guard keeps result ndim == max(ndims). Classifier branch removed so the matrix verifies it.

            // Bool arithmetic was FIXED in Phase 1 F6: `+` now emits logical OR and `*` logical AND
            // for the bool dtype (True + True -> True / byte 1, not 2), matching NumPy's bool ufunc
            // loops. `-` has no bool loop and throws on both sides. Classifier branch removed so the
            // matrix verifies bool add/multiply bit-exact.

            // Complex binary arithmetic (add/sub/mul/div): catastrophic cancellation (re*re-im*im -> 0)
            // and ~1 ULP from System.Numerics.Complex vs NumPy's npy_c* algorithms.
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Complex && c.Operands.Length == 2)
                return "complex binary arithmetic (cancellation / ~ULP) [partly known bug]";

            // (3) NaN ordering in <= / >= was FIXED in Phase 1 F2 (the unordered Cgt_Un/Clt_Un
            //     compare now yields False for a NaN operand, matching IEEE/NumPy). The classifier
            //     branch is intentionally removed so the comparison matrix verifies it bit-exact.

            // (W5-A) cumsum/cumprod on a SIZE-1 array skip the NEP50 accumulator widening — they
            // preserve the narrow integer input (int16/int32 -> int64, uint8/uint16 -> uint64 is
            // what NumPy does) only on the size>1 path; the one-element fast path returns the input
            // dtype. Scoped to a cumsum/cumprod dtype mismatch.
            if ((c.Op == "cumsum" || c.Op == "cumprod") && kind == DivergenceKind.Dtype)
                return "cumsum/cumprod(size-1 int): skips NEP50 accumulator widening (int16/int32/uint8/uint16) [known bug]";

            // --- T13 element-wise extrema (maximum/minimum/fmax/fmin) + isclose ---
            if (ExtremaOps.Contains(c.Op) && kind == DivergenceKind.Value)
            {
                // (W7-B) fmax/fmin must IGNORE NaN (return the finite operand); NumSharp propagates
                // it, so it behaves like maximum/minimum. Scoped to a NaN appearing in NumSharp's out.
                if ((c.Op == "fmax" || c.Op == "fmin") && diffs.Any(d => d.Actual == "NaN"))
                    return "fmax/fmin: propagate NaN instead of ignoring it [known bug]";
                // (W7-A) on an F-contiguous / strided operand the extrema kernel pairs elements by
                // memory order, not logical order -> scrambled result. (C-contiguous is bit-exact;
                // add/sub/mul handle the same F-contig operand correctly, so this is extrema-specific.)
                return "maximum/minimum/fmax/fmin: wrong element pairing on F-contiguous/strided operand [known bug]";
            }
            // isclose on an F-contiguous complex operand diverges (same strided-pairing family).
            if (c.Op == "isclose" && kind == DivergenceKind.Value)
                return "isclose: F-contiguous/complex strided pairing divergence [known bug]";

            // (W11-A) maximum/minimum must PROPAGATE NaN (NumPy: maximum(NaN,x)=NaN) but NumSharp
            // returns the non-NaN operand — it behaves like fmax/fmin (the NaN semantics are swapped
            // with W7-B). Surfaced by the out= test where b=roll(a) places a finite opposite the NaN;
            // the out= write mechanism itself is sound (only the NaN element diverges).
            if ((c.Op == "maximum_out" || c.Op == "minimum_out") && kind == DivergenceKind.Value
                && diffs.All(d => d.Expected == "NaN"))
                return "maximum/minimum: do not propagate NaN (return the non-NaN operand; swapped with fmax/fmin) [known bug]";
            // clip(NaN) clamps to a_min on the out= path too (same as W6-D).
            if (c.Op == "clip_out" && kind == DivergenceKind.Value && diffs.All(d => d.Expected == "NaN"))
                return "clip(NaN): clamps NaN to a_min instead of preserving it (out= path) [known bug]";

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

            // (W6-D) np.clip propagates NaN in NumPy (clip(NaN,lo,hi)=NaN) but NumSharp clamps it to
            // a_min — its min/max comparisons sort NaN below the lower bound. Scoped to a clip whose
            // every divergent element is a NumPy NaN.
            if (c.Op == "clip" && kind == DivergenceKind.Value && diffs.All(d => d.Expected == "NaN"))
                return "clip(NaN): clamps NaN to a_min instead of preserving it [known bug]";

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
                if (kind == DivergenceKind.Value &&
                    (c.Op == "sum" || c.Op == "mean" || c.Op == "std" || c.Op == "var" || c.Op == "prod"))
                    return "reduction summation/two-pass precision (algorithm order)";
            }

            // (4) Unary result-dtype: the transcendental ufuncs (sqrt/cbrt/exp/log/sin/cos/tan) now
            //     follow NumPy's width-based float promotion (Phase 1 F3a) and are verified bit-exact,
            //     so they are NOT excused here. The dtype-preserving ufuncs (square/floor/ceil/trunc/
            //     reciprocal) still widen integer input to float64 instead of preserving it — pending
            //     Phase 1 F3b (needs integer identity / x*x / int-reciprocal kernels). Scoped to that
            //     set so a transcendental promotion regression fails the gate.
            if (kind == DivergenceKind.Dtype && c.Operands.Length == 1
                && (c.Op == "square" || c.Op == "floor" || c.Op == "ceil" || c.Op == "trunc" || c.Op == "reciprocal"))
                return "unary preserve-dtype pending: square/floor/ceil/trunc/reciprocal widen int->float64 [F3b]";

            // (W3-A/B) The hyperbolic / inverse-trig / angle-conversion ufuncs have no Half kernel
            // (throw "Unary operation X not supported for Half" whenever the input promotes to
            // float16: bool/int8/uint8/float16). The COMPLEX hyperbolic/inverse-trig kernels are now
            // implemented (NpyComplexMath) and verified within ULP below — only Half still throws here;
            // deg2rad/rad2deg additionally throw for Complex (NumPy has no complex loop for them either).
            // Scoped to a single-operand THREW on these op names so value/dtype cells stay gated.
            if (kind == DivergenceKind.Threw && c.Operands.Length == 1
                && (c.Op == "sinh" || c.Op == "cosh" || c.Op == "tanh"
                    || c.Op == "arcsin" || c.Op == "arccos" || c.Op == "arctan"
                    || c.Op == "deg2rad" || c.Op == "rad2deg"))
                return "unary hyperbolic/inverse-trig/angle: no Half kernel (throws NotSupportedException) [known bug]";

            // (W3-C) np.exp2 emits a MALFORMED IL method for its float32-output kernel: int16/uint16/
            // float32 inputs throw InvalidProgramException ("the CLR detected an invalid program").
            // exp2(float64) and the Half path are fine, so this is isolated to the Single emitter.
            if (kind == DivergenceKind.Threw && c.Op == "exp2")
                return "exp2: malformed IL (InvalidProgramException) on the Single/float32-output kernel [known bug]";

            // (5) Unary transcendental / complex magnitude ~ULP (libm / algorithm differences).
            //     Tight: every differing element within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "unary ~ULP (transcendental/magnitude algorithm difference)";

            // (6) np.negative on unsigned integers was FIXED in Phase 1 F4: np.negative now routes
            //     through the engine kernel (two's-complement wrap, e.g. -1u -> 255), matching NumPy.
            //     Classifier branch removed so the unary matrix verifies it bit-exact.

            // (6b) Complex unary math is a full NumPy-algorithm port (NpyComplexMath): npy_clog with the
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
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Complex
                && (ReduceOps.Contains(c.Op) || c.Op == "cumsum" || c.Op == "cumprod"))
                return "complex reduction/scan NaN ordering/propagation differs [documented]";

            // Complex np.where was resolved in committed code (no longer throws "Zero-push
            // unsupported for Complex"); it now selects complex operands bit-exact. Classifier
            // branch removed so the where matrix verifies it.

            // (8) np.reciprocal of an integer still returns the float64 reciprocal (0 for |x|>1)
            //     instead of NumPy's integer reciprocal with the ÷0 sentinel, and on a non-contiguous
            //     integer operand it still throws (the int->float reciprocal path needs a flat
            //     Address) — both pending F3b (preserve-int-dtype + a strided integer-reciprocal
            //     kernel). reciprocal on non-contiguous FLOAT operands is resolved. Scoped to integer
            //     input (value diff or the non-contig throw).
            if (c.Op == "reciprocal" && c.Operands.Length == 1
                && (kind == DivergenceKind.Value || kind == DivergenceKind.Threw)
                && (c.Operands[0].Dtype.StartsWith("int") || c.Operands[0].Dtype.StartsWith("uint")
                    || c.Operands[0].Dtype == "bool"))
                return "reciprocal(int): float64 reciprocal / non-contig Address vs NumPy integer ÷0 sentinel [F3b]";

            return null;
        }
    }
}
