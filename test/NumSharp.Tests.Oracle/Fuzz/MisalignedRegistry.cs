using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     How a case diverged. <see cref="ErrorText"/> is the error-parity upgrade: the op DID
    ///     throw (so it is not <see cref="Threw"/>, which means "threw where NumPy returned a
    ///     value"), but its message is not the one NumPy produced. <see cref="Arity"/> is a
    ///     tuple-kind result with the wrong number of slots, and <see cref="Text"/> a
    ///     text-kind result whose string differs.
    ///
    ///     <para>
    ///     NOTE the <c>diffs.Count > 0</c> guard on every ULP branch below. <c>diffs.All(...)</c>
    ///     is VACUOUSLY TRUE for an empty diff list, so a divergence carrying no per-element diffs
    ///     — an error-parity gap, a text mismatch, a wrong arity — would otherwise be silently
    ///     excused as "within 2 ULP" by a branch that never examined anything.
    ///     </para>
    /// </summary>
    public enum DivergenceKind { Dtype, Shape, Value, Threw, ErrorText, Arity, Text }

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
    ///       2. Complex arithmetic ULP envelopes vs NumPy's npy_c* algorithms (each per-op,
    ///          measured, and bounded — see the B2 branch): divide within 2 ULP (npy_cdivide
    ///          scaling); add/subtract within 2 ULP (FMA contraction); multiply within 16 ULP of
    ///          the ELEMENT magnitude (catastrophic-cancellation regime); power within 512
    ///          element-magnitude ULP or at a documented inf/NaN edge (Complex.Pow vs npy_cpow,
    ///          Bug Ledger L6). Every other complex-binary op is gated bit-exact.
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

        /// <summary>The CBLAS product family gated by the products tier (P3's f32 deep scope).</summary>
        private static readonly System.Collections.Generic.HashSet<string> ProductOps = new()
        {
            "inner", "vdot", "vecdot", "matvec", "vecmat", "tensordot"
        };

        /// <summary>
        ///     The np.fft transform ops (NOT the helpers: fftfreq/rfftfreq are always float64 and
        ///     fftshift/ifftshift preserve the input dtype, so neither ever diverges). Only these
        ///     carry the float32/float16 -> complex128/float64 promotion divergence (F1 below).
        /// </summary>
        private static readonly System.Collections.Generic.HashSet<string> FftTransformOps = new()
        {
            "fft", "ifft", "rfft", "irfft", "hfft", "ihfft",
            "fft2", "ifft2", "fftn", "ifftn", "rfft2", "irfft2", "rfftn", "irfftn"
        };

        /// <summary>
        ///     The unary ops whose float32 loop NumSharp ports from NumPy's own kernel rather than
        ///     delegating to the platform libm, and which are therefore held BIT-EXACT (no ULP
        ///     envelope). Adding an op here without a matching port would silently turn a real
        ///     divergence into a hard failure — which is the intent, but only once the port exists.
        /// </summary>
        private static readonly System.Collections.Generic.HashSet<string> NumPyPortedFloat32Kernels = new()
        {
            "exp", "log", "sin", "cos", "rad2deg", "deg2rad", "tanh"
        };

        /// <summary>
        ///     The same claim at float64. Almost empty on purpose: at f8 the platform libm already
        ///     agrees with NumPy bit-for-bit for exp/log/sin/cos, so there was nothing to port and
        ///     nothing to carve out. tanh is the exception — NumPy ships its OWN table-driven kernel
        ///     at f8 too (loops_hyperbolic), which is why the BCL's Math.Tanh diverged on 8.1% of
        ///     inputs and why NDFloatMath now owns both widths.
        /// </summary>
        private static readonly System.Collections.Generic.HashSet<string> NumPyPortedFloat64Kernels = new()
        {
            "tanh"
        };

        /// <summary>
        ///     Single-operand ops that are PURE ARITHMETIC (convolution / Horner / powers / leading-
        ///     zero normalisation / normalized dot) rather than transcendental-libm — so they have NO
        ///     legitimate reason to differ from NumPy by a ULP, and are held BIT-EXACT. Carved out of
        ///     the blanket "unary ~ULP" excuse below for the same reason the ported float32 kernels
        ///     are: the excuse's rationale ("transcendental/magnitude algorithm difference") does not
        ///     apply, so a ≤2-ULP drift here is a regression, not algorithm noise, and must fail the
        ///     gate. Probed byte-exact vs NumPy 2.4.2 (poly.jsonl / the cov/corrcoef cases in
        ///     products.jsonl). The 2-operand siblings (polyval/polyadd/polymul/cross/einsum/…) never
        ///     reach the unary branch and are already held strict.
        /// </summary>
        private static readonly System.Collections.Generic.HashSet<string> ByteExactArithmeticUnaryOps = new()
        {
            "poly", "polyder", "polyint", "vander", "poly1d_coeffs", "poly1d_fromroots",
            "cov", "conjugate", "real", "imag", "vector_norm", "matrix_norm"
        };

        public static string Classify(
            FuzzCorpus.Case c, DivergenceKind kind,
            byte[] expected, byte[] actual, NPTypeCode tc, IReadOnlyList<BitDiff.Diff> diffs,
            byte[] truth = null)
        {
            // (1) NEP50 weak-scalar promotion. Any multi-operand op with a 0-D operand: NumSharp
            //     promotes it weakly (the array operand's dtype drives the result), where NumPy makes
            //     0-D arrays full participants. Covers binary pp_scalar_* and np.where wh_bcast_xy.
            if (kind == DivergenceKind.Dtype && c.Operands.Length >= 2 && c.Operands.Any(o => o.Shape.Length == 0))
                return "NEP50 weak-scalar: 0-D operand promoted weakly (NumPy promotes 0-D arrays fully)";

            // (2) Complex true-division ~1 ULP. Excuse only divide, only complex result, only when every
            //     differing element is within 2 ULP — a gross error still fails.
            if (kind == DivergenceKind.Value && c.Op == "divide" && tc == NPTypeCode.Complex
                && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "complex division ~1 ULP (npy_cdivide vs System.Numerics.Complex)";

            // corrcoef normalizes a complex covariance matrix with two in-place complex/real
            // divisions, so it inherits the same npy_cdivide-vs-System.Numerics rounding as the
            // direct divide ufunc. Keep this composition explicit — the former blanket complex-
            // unary excuse hid the one-ULP diagonal cell despite corrcoef being documented exact.
            if (kind == DivergenceKind.Value && c.Op == "corrcoef" && tc == NPTypeCode.Complex
                && c.Operands.Any(o => o.Dtype == "complex128")
                && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "corrcoef(complex): normalization inherits npy_cdivide vs System.Numerics "
                     + "rounding (bounded <=2 ULP) [documented]";

            // logaddexp / logaddexp2: log(exp(x1)+exp(x2)) / log2(2**x1+2**x2). Math.Exp / double.Exp2
            // / MathF.Exp are bit-identical to NumPy's ucrtbase exp/exp2/expf, but the compound uses
            // log1p, and NumPy's ucrtbase log1p is CLOSED — the managed fdlibm port agrees to <=1 ULP
            // (5% of log1p evaluations differ by 1). So logaddexp is <=1 ULP, logaddexp2 <=2 ULP (the
            // LOG2E * log1p product), float16 via float32 <=2. nextafter is NOT here (bit-exact via
            // BitIncrement/BitDecrement). Bounded per-element; a gross error still fails.
            if (kind == DivergenceKind.Value && (c.Op == "logaddexp" || c.Op == "logaddexp2")
                && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "logaddexp/logaddexp2: managed fdlibm log1p <=2 ULP vs NumPy's closed ucrtbase log1p [documented]";

            // (F1) np.fft over a float32/float16 input. NumSharp has ONE complex type (complex128) and
            //      no complex64, so it returns complex128 (fft/rfft/ihfft/the N-D forms) or float64
            //      (irfft/hfft) where NumPy 2.x returns complex64 / float32 / float16. A dtype-ONLY
            //      divergence: the VALUES are BIT-IDENTICAL to NumPy 2.4.2 (NumSharp reproduces NumPy's
            //      exact per-loop precision — double+round OR the single-precision engine, see
            //      FFT_PARITY.md §7). CompareArray no longer merely excuses these: it UP-CASTS NumPy's
            //      complex64/float32/float16 bytes and bit-compares (IsFftFloatCell / FftFloatValuesMatch),
            //      so a compute divergence would fail even though the dtype mismatch is excused here. Two
            //      shapes reach this branch: the mappable one (float32/float16 expected from irfft/hfft)
            //      and the UNNAMEABLE complex64 (fft/rfft/ihfft/...), both routed as a Dtype divergence.
            //      float64/complex128/int/bool inputs are CONTRACTUAL (bit-exact) and are NOT touched
            //      here; the helpers never diverge (see FftTransformOps). A real complex64 dtype (#569)
            //      would flip the dtype cell automatically, values unchanged.
            if (kind == DivergenceKind.Dtype && FftTransformOps.Contains(c.Op)
                && c.Operands.Length >= 1
                && (c.Operands[0].Dtype == "float32" || c.Operands[0].Dtype == "float16"))
                return "np.fft(float32/float16): NumPy returns complex64/float32/float16; NumSharp has no "
                     + "complex64 and returns complex128/float64 — VALUES bit-exact (harness-verified), "
                     + "dtype-only divergence pending #569 [documented]";

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
                    && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                    return "complex add/subtract within 2 ULP (FMA contraction) [documented]";
                // multiply: npy_cmul vs System.Numerics.Complex round (ac-bd)/(ad+bc) differently
                // (FMA contraction). In the catastrophic-cancellation regime (ac ~ bd) the RELATIVE
                // error of the cancelled component is unbounded, but the ABSOLUTE error stays at
                // rounding scale of the products — i.e. of the element's dominant component. So the
                // detection: every differing component within 16 ULP *of the element's own
                // magnitude* (not of itself). A divergence larger than that is a real kernel bug.
                if (c.Op == "multiply"
                    && diffs.Count > 0 && diffs.All(d => WithinComplexElementMagnitudeUlp(expected, actual, d.Index, 16)))
                    return "complex multiply cancellation / ~ULP at element magnitude (npy_cmul vs System.Numerics) [documented #12]";
                // power: Complex.Pow (polar exp(w*log z)) vs npy_cpow (special-cases small integer
                // exponents via repeated squaring) — measured on the corpus the finite interior
                // diverges by up to ~350 ULP of the affected component, plus the documented gross
                // inf/NaN edges (Phase-1 F5) where one side goes non-finite. Bound the finite side
                // at 512 ULP of the ELEMENT's magnitude (same absolute-error anchor as multiply:
                // still catches sign flips / wrong magnitudes) and excuse the non-finite edges.
                if (c.Op == "power"
                    && diffs.Count > 0 && diffs.All(d => WithinComplexElementMagnitudeUlp(expected, actual, d.Index, 512)
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
            // (FIXED) isclose complex128 no longer diverges — the excuse here is removed. isclose used
            // to cast BOTH operands to float64 up front, which for a complex operand dropped the
            // imaginary part (and paired an F-contiguous/strided complex operand by buffer order), so
            // every discriminating complex case failed. IsClose now computes in NumPy's exact
            // result_type(x, result_type(y, 1.0)) — complex stays complex, float32 stays float32 — so
            // the whole complex128 tier (contig/F/strided/broadcast/negstride) is bit-exact with NumPy
            // 2.4.2 and needs no excuse (verified with this branch disabled).

            // (S2) fmax/fmin ±0-tie sign on a negative-stride float32 view. NumPy's OWN fmax/fmin
            //      pick which zero to return on a (+0,-0) tie by SIMD path: its array loop returns
            //      the second operand (so array fmax(+0,-0) = -0, unlike the +0 its scalar path
            //      gives). NumSharp matches that array behaviour on contiguous/strided operands but
            //      diverges on the REVERSED f32 lane order, so the sign of the tied zero differs.
            //      Non-contractual — NumPy is itself path-inconsistent here. Scoped razor-tight:
            //      float32, and BOTH the expected and actual token are a signed zero — a wrong
            //      NON-zero fmax/fmin result still fails.
            if ((c.Op == "fmax" || c.Op == "fmin") && kind == DivergenceKind.Value
                && tc == NPTypeCode.Single && diffs.Count > 0 && diffs.All(IsSignedZeroPairSingle))
                return "fmax/fmin ±0-tie sign on a reversed float32 view (NumPy's own fmax ±0 sign is "
                     + "SIMD-path-dependent) [documented non-contractual]";

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

            // =================================================================================
            // (P1/P2) Truthful-vs-precise adjudication — cases carrying expected.truth (the
            // precision tier's correctly-rounded mathematical reference). POLICY, per the
            // project vision (byte-identical parity to NumPy): "precise" (bit-exact to NumPy)
            // passes WITHOUT ever reaching this code — truth is consulted only on an existing
            // divergence, so it can never turn a NumPy-matching result red. On a divergence it
            // answers the one question the parity bytes cannot: WHICH side lost precision.
            //   * NOT-LESS-truthful than NumPy (every diff within TruthSlack of NumPy's own
            //     distance to truth) -> excused as prefer-precise PARITY DEBT. Being MORE
            //     accurate than NumPy is still a divergence to close by porting NumPy's
            //     algorithm (the exp/log/sin/cos/tanh route), never a win — hence the label.
            //     Two labels so the triage differentiates toward-truth from within-noise.
            //   * LESS truthful (beyond slack) -> genuine precision LOSS: fall through, where
            //     only a tightly-scoped known-bug branch (e.g. S1 expm1/log1p) may still excuse
            //     it; otherwise the gate is red. The unbounded reduction blanket below is gated
            //     on truth==null precisely so a truth-bearing loss cannot hide in it.
            // The slack (4x relative, +8 absolute ULP) absorbs SIMD lane-count variation across
            // hosts (V128/V256/V512 pick different accumulation orders) while still failing the
            // losses that matter — measured 512 ULP for a naive f32 wide-sum, ~6.7e7 ULP for
            // Exp(x)-1 expm1 — by orders of magnitude.
            if (kind == DivergenceKind.Value && truth != null && expected != null
                && truth.Length == expected.Length && diffs.Count > 0
                && (tc == NPTypeCode.Half || tc == NPTypeCode.Single
                    || tc == NPTypeCode.Double || tc == NPTypeCode.Complex))
            {
                bool notLessTruthful = true, anyTowardTruth = false;
                foreach (var d in diffs)
                {
                    long dNS = BitDiff.UlpDistance(actual, truth, d.Index, tc);
                    long dNPY = BitDiff.UlpDistance(expected, truth, d.Index, tc);
                    if (dNS > TruthSlack(dNPY)) { notLessTruthful = false; break; }
                    if (dNS < dNPY) anyTowardTruth = true;
                }
                if (notLessTruthful && anyTowardTruth)
                    return "prefer-precise: diverges from NumPy TOWARD the correctly-rounded truth "
                         + "— parity debt (port NumPy's algorithm), not precision loss [documented]";
                if (notLessTruthful)
                    return "prefer-precise: diverges from NumPy within truth-equivalence slack "
                         + "(neither side less accurate) [documented]";
                // less truthful than NumPy: fall through — known-bug branches or a red gate.
            }

            // (P3) The precision losses the truth-bearing tiers DISCOVERED on arrival
            //      (2026-08-14), excused as tracked known bugs with a measured bound — kept in
            //      the corpus so a kernel fix flips them bit-exact automatically and a WORSE
            //      regression still fails:
            //        * float32 var/std accumulation: 55/26 ULP from truth where NumPy's two-pass
            //          pairwise sits at 3/2 (contiguous wide-magnitude input);
            //        * the NEGATIVE-STRIDE reduce path (sum/mean/var/std): 11-32 ULP from truth
            //          where NumPy is EXACT on the same reversed view — the backward traversal
            //          accumulates in a worse order than the contiguous kernel;
            //        * float32 DEEP product contractions (inner/tensordot at K=2049, products
            //          tier): a few elements land 1-2 ULP past the prefer-precise slack while
            //          NumPy's BLAS sgemm multi-accumulator stays ≤1 ULP from exact (f64 deep
            //          products are prefer-precise-excused or bit-exact; vdot/vecdot/matvec/
            //          vecmat f32 sit WITHIN the slack).
            //      Bounded at 256 ULP-vs-truth (≈5x the measured worst, room for cross-host SIMD
            //      lane variation); a loss beyond that — or in any other cell — is red.
            if (kind == DivergenceKind.Value && truth != null && expected != null
                && truth.Length == expected.Length && diffs.Count > 0
                && ((c.Layout == "negstride_1d"
                     && (c.Op == "sum" || c.Op == "mean" || c.Op == "var" || c.Op == "std"))
                    || (tc == NPTypeCode.Single && (c.Op == "var" || c.Op == "std"))
                    || (tc == NPTypeCode.Single && ProductOps.Contains(c.Op)))
                && diffs.All(d => BitDiff.UlpDistance(actual, truth, d.Index, tc) <= 256))
                return "precision-loss (known): f32 var/std accumulation + negative-stride reduction "
                     + "accumulation + f32 deep product contraction lose ULP vs truth where NumPy "
                     + "stays near-exact (bounded ≤256) [known bug]";

            // (R1) np.random transform samplers within a few ULP of NumPy on the SAME CRT:
            //      chisquare / wald / noncentral_f / dirichlet compose their draws with a
            //      slightly different arithmetic ordering than NumPy's C (measured ≤5/≤24/≤3/≤3
            //      ULP on the corpus; the underlying uniform/gauss STREAM is bit-identical — a
            //      stream slip produces gross divergence and still fails, as the eight carved
            //      samplers in gen_random_parity did). Per-dist caps: 32 for wald (its
            //      inverse-Gaussian composition drifts the most), 8 for the rest.
            if (c.Op == "rnd" && kind == DivergenceKind.Value && diffs.Count > 0
                && c.Params != null && c.Params.TryGetValue("dist", out var rndDist)
                && rndDist.GetString() is "chisquare" or "wald" or "noncentral_f" or "dirichlet"
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc,
                                                    rndDist.GetString() == "wald" ? 32 : 8)))
                return "rnd transform ~ULP: chisquare/wald/noncentral_f/dirichlet arithmetic "
                     + "ordering differs from NumPy's C composition (stream identical) [documented]";

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
                if (kind == DivergenceKind.Value && diffs.Count > 0 && diffs.Count > 0 && diffs.All(d => d.Expected == "NaN")
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
                // bug, not "precision", and fails the gate. Gated on truth==null: this blanket is
                // UNBOUNDED, so a truth-bearing case (precision tier) must be adjudicated by the
                // P1/P2 truthful branch above instead — a divergence that is LESS truthful than
                // NumPy would otherwise hide in here.
                if (kind == DivergenceKind.Value && truth == null
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
                && diffs.Count > 0 && diffs.All(DecimalLastDigitDiff))
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
            //
            //     CARVED OUT: the float32 loops NumSharp ports from NumPy itself. Those are no
            //     longer "whatever the platform libm does" — NDFloatMath holds ports of
            //     simd_exp_FLOAT, simd_log_FLOAT and simd_sincos_f32, each bit-exact over ALL 2^32
            //     float32 inputs (exhaustively verified, not sampled), and rad2deg/deg2rad now form
            //     their constant at float precision exactly as NumPy's RAD2DEG/DEG2RAD macros do.
            //     The carve-out covers float32 input AND the narrow-integer inputs (int16/uint16/
            //     char) whose NumPy loop is that same 'f->f' kernel. So ANY float32 divergence in
            //     these ops is now a regression, not an algorithm difference, and must fail the gate
            //     — which is the whole point of narrowing an excuse rather than leaving a green
            //     blanket over a fixed op. Other result dtypes stay excused on purpose: float16 runs
            //     NumPy's separate loops_half kernels, and float64 exp/log/sin/cos are the
            //     platform's scalar npy_* calls, none of which NumSharp reproduces bit-for-bit.
            //
            //     tanh is carved out at BOTH widths (see NumPyPortedFloat64Kernels): it is the one
            //     op here for which NumPy ships its own kernel at float64 as well, so f8 tanh is a
            //     port too and a 1-ULP drift there is likewise a regression, not libm noise.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1
                && !(tc == NPTypeCode.Single && NumPyPortedFloat32Kernels.Contains(c.Op))
                && !(tc == NPTypeCode.Double && NumPyPortedFloat64Kernels.Contains(c.Op))
                && !ByteExactArithmeticUnaryOps.Contains(c.Op)
                && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "unary ~ULP (transcendental/magnitude algorithm difference)";

            // (S1) expm1 / log1p are computed by NumSharp as Exp(x)-1 / Log(1+x) rather than a
            //      dedicated small-|x| kernel, so near zero they lose precision: a subnormal result
            //      flushes to 0, expm1(-0.0) gives +0.0, and a moderate result can drift a ULP.
            //      NumPy calls the CRT (npy_expm1 / npy_log1p), which is not reproducible here, so
            //      this is a DOCUMENTED non-portable accuracy defect (see the CLAUDE.md math
            //      section: "expm1(1e-8) returns 0 where NumPy returns 1e-8"). Bounded: every diff
            //      is within 2 ULP OR within an absolute ~ulp(1) envelope of the dtype, so a gross
            //      error (wrong magnitude/sign at a non-tiny result) exceeds it and still fails.
            //      Surfaced by the specials tier's subnormal / tiny / -0 inputs — the ordinary pools
            //      never reach the catastrophic small-|x| band. Placed AFTER the ~ULP branch so a
            //      large-|x| expm1 (all diffs within 2 ULP) keeps the generic label.
            if (kind == DivergenceKind.Value && (c.Op == "expm1" || c.Op == "log1p")
                && c.Operands.Length == 1 && diffs.Count > 0
                && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)
                                  || BitDiff.WithinAbs(expected, actual, d.Index, tc, Expm1Log1pAbsTol(tc))))
                return "expm1/log1p computed as Exp(x)-1 / Log(1+x): small-|x| precision loss / -0 / "
                     + "subnormal flush, bounded abs error [documented non-portable defect]";

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
                && !ByteExactArithmeticUnaryOps.Contains(c.Op)
                && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 3)))
                return "complex unary within 3 ULP (full NumPy-algorithm port)";

            // (7) The only complex-unary divergences beyond 3 ULP are three pathological regimes, each
            //     verified against NumPy 2.4.2 and accepted (NumSharp is the more accurate side of the
            //     square/log cancellation cases; these are inputs no real workload produces):
            //       - cos/sin with a NaN imaginary part: the sign of the resulting zero component is
            //         C99-UNSPECIFIED (cos(+-0 + NaN i).imag = +-0 either way); the platform libm and
            //         the npy_ccos identity pick opposite signs.
            //       - arccos with a sub-DBL_MIN imaginary part: Complex.Acos flushes the denormal real
            //         part to 0 where NumPy's cacos _do_hard_work keeps it (arccos(2 + 1e-308 i).real
            //         ~ 5.8e-309) — a denormal-range edge. arccosh INHERITS this exact case: NumSharp
            //         computes cacosh via the msun formula on Acos (cacosh(z) = ±I·cacos(z)), so the
            //         flushed denormal becomes arccosh(2 + 1e-308 i).imag = 0 vs NumPy's 5.77e-309.
            //         (arcsinh = swap(asin(swap)) and arctanh = catanh are fully ≤3 ULP — asin/atan
            //         have no such edge — so they stay under the generic 3-ULP branch above.)
            //       - sinh/cosh at the overflow boundary |x| in [710, 710.13]: Windows' CRT sinh
            //         overflows to inf while .NET Math.Sinh stays finite (a platform-libm boundary,
            //         absent on glibc).
            //     Scoped to these op names so a >3-ULP regression in ANY other complex unary op fails.
            if (kind == DivergenceKind.Value && c.Operands.Length == 1 && tc == NPTypeCode.Complex
                && (c.Op == "cos" || c.Op == "sin" || c.Op == "arccos" || c.Op == "arccosh"
                    || c.Op == "sinh" || c.Op == "cosh"))
                return "complex cos/sin/arccos/arccosh/sinh/cosh pathological edge (NaN zero-sign / subnormal / overflow boundary) [documented]";

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

            // (S3) complex matmul / dot / outer with an INFINITE operand. A complex product
            //      (inf+0j)*(x+0j) is inf + (inf*0)j = inf + nan·j, and accumulating it, NumPy's
            //      zgemm / npy_cmul carry the C99 Annex-G "recover-infinities" fixup and collapse the
            //      cell to (nan,nan), while NumSharp's managed complex product leaves (inf,nan).
            //      Complex arithmetic with infinities is C99-UNSPECIFIED, so this is non-contractual
            //      — and it is confined to the infinite cells: the NaN-propagation cells (a NaN
            //      anywhere -> (nan,nan)) match bit-for-bit, and EVERY real-dtype matmul/dot/outer
            //      specials case is bit-exact (the managed float GEMM propagates NaN/inf like NumPy's
            //      BLAS on these order-independent operands). Scoped to a non-finite complex diff so a
            //      finite-value complex product divergence still fails.
            if (kind == DivergenceKind.Value && tc == NPTypeCode.Complex
                && (c.Op == "matmul" || c.Op == "dot" || c.Op == "outer")
                && diffs.Count > 0 && diffs.All(d => NonFiniteInvolved(expected, actual, d.Index)))
                return "complex matmul/dot/outer infinite-operand product: C99-unspecified complex-"
                     + "infinity arithmetic (zgemm inf-recovery vs managed product) [documented]";

            // Complex np.where was resolved in committed code (no longer throws "Zero-push
            // unsupported for Complex"); it now selects complex operands bit-exact. Classifier
            // branch removed so the where matrix verifies it.

            // (8) np.reciprocal of an integer/bool now matches NumPy bit-exact and is no longer
            //     excused: it preserves the integer dtype (bool -> int8), C-truncating 1/x gives 0
            //     for |x| > 1, and the per-type 1/0 sentinel is reproduced exactly (0 for
            //     int8/int16/uint8/uint16/uint32; the sign-bit 0x80..0 for int32/int64/uint64).
            //     Strided / sliced / broadcast integer operands are read in place (no longer throw).

            // =================================================================================
            // Result-kind and error-parity tiers (iter / dtype_text / errors_full). These gate
            // claims the corpus could not express before: traversal ORDER, tuple ARITY, dtype and
            // text results, and NumPy's actual error MESSAGE. Everything below was measured when
            // those tiers were first run, and each branch is scoped to the exact cell so a
            // neighbouring regression still fails.
            // =================================================================================

            // (K1) np.nditer traversal with order='A' over a TRANSPOSED 3-D operand — NumSharp
            // picks a different axis ordering than NumPy's stride-sorted 'A'/'K' heuristic, so the
            // value stream, the multi_index stream and BOTH tracked-index streams (c_index/f_index)
            // all differ. Scoped to nditer_* on that one layout; every other layout/order is gated
            // exactly.
            //
            // This excuse used to also cover `strided_2d_cols` and `negstride_2d_offset` for a
            // second cause — external_loop CHUNKING: NumSharp's constructor only coalesced axes when
            // every operand was contiguous and nothing was broadcast, so a uniformly strided
            // a[:, ::2] ran one chunk per row where NumPy's unconditional npyiter_coalesce_axes hands
            // out ONE strided chunk ([8] vs [4], [6] vs [1]). NDIterCoalescing.CoalesceAxesIterationOrder
            // (2026-09-03) merges in iteration order exactly as NumPy does; probed against 2.4.2 on
            // all three layouts × C/F/A/K, the chunk lengths and value streams are identical except
            // the 'A'-order transposed_3d case above, so those two layouts are gated bit-exactly now.
            if (c.Op != null && (c.Op == "nditer" || c.Op.StartsWith("nditer_"))
                && c.Layout == "transposed_3d")
                return "nditer traversal: order='A' axis ordering over a transposed 3-D operand [known bug]";

            // (K2) np.isscalar on a 0-D ARRAY. NumPy answers False — a 0-d ndarray is an array, not
            // a scalar, which is one of its best-known gotchas — while NumSharp answers True.
            if (c.Op == "isscalar" && c.Operands.Length == 1 && c.Operands[0].Shape.Length == 0)
                return "isscalar(0-d array): NumSharp True, NumPy False (a 0-d ndarray is not a scalar) [known bug]";

            // (K3) np.nonzero on a 0-D array. NumPy 2.x REFUSES it ("Calling nonzero on 0d arrays is
            // not allowed. Use np.atleast_1d(scalar).nonzero() instead."); NumSharp returns a tuple.
            if (c.Op == "nonzero_all" && c.Operands.Length == 1 && c.Operands[0].Shape.Length == 0)
                return "nonzero(0-d): NumPy raises ValueError, NumSharp returns a tuple [known bug]";

            // (K4) Complex-input ufunc rejection — the WORDING, not the decision. Both sides refuse
            // cbrt/floor/ceil/trunc/deg2rad/rad2deg/floor_divide/mod on complex input; NumPy raises
            // its ufunc TypeError, NumSharp a NotSupportedException with its own text. (The bitwise
            // and invert loops DO produce NumPy's message verbatim, so the machinery exists — these
            // kernels simply do not use it.)
            if (kind == DivergenceKind.ErrorText && ComplexRejectOps.Contains(c.Op)
                && c.Operands.Any(o => o.Dtype == "complex128"))
                return "complex ufunc rejection wording: NotSupportedException('operation X not supported "
                     + "for Complex') vs NumPy's ufunc TypeError [known gap]";

            // (K5) …and on a ZERO-SIZE complex operand the rejection is skipped ENTIRELY: NumSharp
            // returns an empty result because the kernel never runs, where NumPy still raises —
            // NumPy validates the LOOP (can this dtype be handled at all?), not the data.
            if (kind == DivergenceKind.Value && ComplexRejectOps.Contains(c.Op)
                && c.Operands.Any(o => o.Dtype == "complex128")
                && c.Operands.Any(o => o.Shape.Any(d => d == 0)))
                return "complex ufunc rejection SKIPPED on a zero-size operand (NumPy validates the "
                     + "loop, not the data) [known bug]";

            // (K6/K8 RETIRED) power with a negative integer exponent — for BOTH an integer base and
            // a bool base (which promotes to an integer loop) — now raises NumPy's clean
            // ValueError("Integers to negative integer powers are not allowed.") exactly. The former
            // bugs (bool loop missing the guard; the int path tripping Debug.Fail("index < Count,
            // Memory corruption expected") and reading out of bounds in RELEASE) were fixed in
            // Default.Power.cs: the guard keys off the PROMOTED loop type (ResolvePowerResultType, so
            // a bool base counts) and the negative-exponent pre-scan reads by flat index through
            // Storage.GetAtIndex<T> (layout-correct for strided/broadcast/(N,M) exponents) instead of
            // the coordinate GetXxx(long) overload that walked off axis 0. No excuse needed.

            // (K9) np.result_type over a mixed signed/unsigned pair where one operand is 0-D THROWS
            // (ArgumentException 'Destination array was not long enough' / OverflowException)
            // instead of resolving a dtype.
            if (kind == DivergenceKind.Threw && c.Op == "result_type_arrays")
                return "result_type(mixed signed/unsigned, 0-D operand): throws instead of resolving "
                     + "a promotion [known bug]";

            // (K10) ufunc out=/where= with a read-only BROADCAST out. NumPy refuses it
            // ("non-broadcastable output operand …" / read-only output). NumSharp either raises
            // with different wording or — worse — WRITES THROUGH IT, which contradicts its own
            // design rule that a broadcast view is non-writeable (Shape.IsWriteable == false).
            if ((kind == DivergenceKind.ErrorText || kind == DivergenceKind.Value
                 || kind == DivergenceKind.Arity || kind == DivergenceKind.Shape)
                && c.Layout == "out_broadcast")
                return "ufunc out= on a read-only broadcast view: NumSharp writes through it (or "
                     + "refuses with different wording) where NumPy raises [known bug]";

            // (K12) isnan into a STRIDED bool out writes False where NumPy writes True — the
            // 1-byte store walks the buffer rather than the view, so the True results land on the
            // wrong elements. A CONTIGUOUS bool out is correct, which is exactly why this needed a
            // strided-out axis to surface at all.
            if (kind == DivergenceKind.Value && c.Op == "out_unary"
                && c.Layout == "out_strided"
                && c.Params != null && c.Params.TryGetValue("ufunc", out var ufEl)
                && ufEl.GetString() == "isnan")
                return "isnan into a strided bool out=: results land on the wrong elements "
                     + "(buffer walked instead of the view) [known bug]";

            // (K11) out=/where= float32 transcendentals (exp/sin) and isnan: every differing
            // element is within 2 ULP — the same envelope branch (5) documents for the plain unary
            // path, which cannot be reached here because `out` and `where` are operands too (so
            // Operands.Length > 1).
            //
            // OPEN QUESTION, deliberately recorded rather than smoothed over: exp(1.0f) inside a
            // (4,5) float32 array comes back 0x402df854 from np.exp(x), np.exp(x, out) AND
            // np.exp(x, out, where) alike, while NumPy — and the committed unary.jsonl expectation
            // for the very same values, shape and dtype — say 0x402df855. The unary tier is green,
            // so the same op on the same data disagrees depending on how the array was built.
            // That points at kernel/path selection, not at out=; it is scoped here only so this
            // tier can gate everything else it covers.
            if (kind == DivergenceKind.Value && c.Op != null && c.Op.StartsWith("out_")
                && diffs.Count > 0 && diffs.All(d => BitDiff.WithinUlp(expected, actual, d.Index, tc, 2)))
                return "out=/where= float32 transcendental within 2 ULP — see the exp(1.0f) "
                     + "path-dependence note at branch K11 [open question]";

            // (K7) NEP50 weak-scalar, reached through the ERROR path rather than the dtype one.
            // Documented difference (1) at the top of this file: NumSharp treats a 0-D operand as a
            // WEAK scalar, so pairings NumPy refuses outright (int64 with uint64, which have no
            // common integer type) instead promote and succeed.
            //
            // EXCLUDES the out=/where= tier: there a 0-D operand is usually the `where` MASK
            // (scalar_true / scalar_false), which has nothing to do with promotion — without this
            // guard the branch silently swallowed 60 mask cases it had no business classifying.
            if ((kind == DivergenceKind.Value || kind == DivergenceKind.ErrorText)
                && (c.Op == null || !c.Op.StartsWith("out_"))
                && c.Operands.Length >= 2 && c.Operands.Any(o => o.Shape.Length == 0))
                return "NEP50 weak-scalar (error path): a 0-D operand promotes weakly, so a pairing "
                     + "NumPy refuses succeeds instead";

            return null;
        }

        /// <summary>
        ///     The ufuncs NumPy has no complex loop for. NumSharp refuses them too, but with its own
        ///     exception type and wording — see branches (K4)/(K5).
        /// </summary>
        private static readonly System.Collections.Generic.HashSet<string> ComplexRejectOps = new()
        {
            "cbrt", "floor", "ceil", "trunc", "deg2rad", "rad2deg", "floor_divide", "mod"
        };

        /// <summary>Element count of a corpus operand (0-d shape [] counts as 1).</summary>
        private static long ElementCount(FuzzCorpus.Operand o)
        {
            long n = 1;
            foreach (var d in o.Shape)
                n *= d;
            return n;
        }

        /// <summary>
        ///     The prefer-precise threshold: NumSharp's ULP distance to truth may exceed NumPy's by
        ///     at most 4x relative or +8 absolute (whichever is larger) to count as
        ///     not-less-truthful. Saturating — a NumPy distance of long.MaxValue (opposite-sign vs
        ///     truth) admits anything, since NumPy itself is maximally far.
        /// </summary>
        private static long TruthSlack(long dNPY)
            => dNPY >= (long.MaxValue - 8) / 4 ? long.MaxValue : Math.Max(4 * dNPY, dNPY + 8);

        /// <summary>
        ///     The absolute-error envelope for the expm1/log1p Exp(x)-1 / Log(1+x) signature — a few
        ///     ULP of 1.0 for the dtype (its worst absolute error near zero), well below any gross
        ///     wrong-magnitude bug at a non-tiny result.
        /// </summary>
        private static double Expm1Log1pAbsTol(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Double => 1e-13,
            NPTypeCode.Single => 1e-6,
            NPTypeCode.Half => 1e-2,
            _ => 0.0,
        };

        /// <summary>
        ///     Both the expected and actual Single tokens are a signed zero. BitDiff prints Single
        ///     bytes low-to-high, so +0.0 is "00000000" and -0.0 (0x80000000) is "00000080".
        /// </summary>
        private static bool IsSignedZeroPairSingle(BitDiff.Diff d) =>
            (d.Expected == "00000000" || d.Expected == "00000080")
            && (d.Actual == "00000000" || d.Actual == "00000080");

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
