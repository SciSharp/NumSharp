using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The result-kind and error-parity half of the differential matrix.
    ///
    ///     The original corpus could only express ONE comparable thing: a single NDArray, checked
    ///     as (dtype, shape, C-contiguous bytes). That shape excluded three whole classes of op
    ///     from the gate — ones returning a tuple of arrays, ones returning a dtype or a scalar,
    ///     and ones returning text — and it reduced every raising case to "something was thrown".
    ///     <see cref="FuzzCorpus.Expected.Kind"/> and <see cref="FuzzCorpus.Case.Error"/> close
    ///     that gap; the comparators live here so the shared harness file stays thin.
    /// </summary>
    public partial class FuzzCorpusTests
    {
        /// <summary>
        ///     The single-operand complex ufuncs backed by <c>NDComplexMath</c>, whose NaN SIGN (and
        ///     signed-zero) NumSharp reproduces bit-for-bit against NumPy 2.4.2 win-amd64 (MSVC UCRT
        ///     complex functions). For these — and ONLY these — a complex128 result's NaN is compared
        ///     by raw bytes rather than tokenized, so the sign fix is gated and a regression fails.
        ///     Verified byte-exact over the full ±0/±inf/NaN × ±0/±inf/NaN special-value grid (every
        ///     "produce a NaN" slot is the positive NPY_NAN; the negatives are genuine 0/0·inf-inf or a
        ///     transform negate, which reproduce on x86 by construction). abs/absolute return float64
        ///     and so never reach the Complex branch; the complex BINARY ops (add/mul/power/…) keep the
        ///     tokenizing comparison (their NaN is non-contractual — see MisalignedRegistry F5/#12).
        /// </summary>
        private static readonly HashSet<string> ComplexNanContractOps = new()
        {
            "sqrt", "log", "log2", "log10", "log1p", "exp", "exp2", "expm1", "square", "reciprocal",
            "sin", "cos", "tan", "sinh", "cosh", "tanh",
            "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh",
            "conjugate", "conj", "negative", "positive",
            // sign(z)=(+NaN,+NaN) for any NaN/both-inf (canonical); abs/absolute(z)=+NaN (npy_cabs) — the
            // latter returns float64, so the gate keys on a complex OPERAND (below), not the result dtype.
            "sign", "abs", "absolute"
        };

        /// <summary>
        ///     The float16 reductions whose result is a SELECTED input element — NumPy's HALF_maximum / HALF_minimum /
        ///     HALF_fmax / HALF_fmin (numpy/_core/src/umath/loops.c.src) are sequential loops that copy one operand, never
        ///     compute a new value. So the result's bits — which NaN (payload, sign, a signalling NaN unquieted), which ±0 —
        ///     follow from the reduction's visiting order alone, with no libm, compiler or CPU in the path: portable, and
        ///     contractual. For a float16 operand these ops are compared by RAW BYTES and any difference is a HARD failure
        ///     (no registry excuse: a selection has no rounding to excuse, and the generic ULP excuses read NaN-vs-NaN as
        ///     0 ULP). Gated across every tier that holds them (reduce, nanreduce, specials, instance, half_minmax).
        /// </summary>
        private static readonly HashSet<string> HalfSelectionContractOps = new() { "max", "min", "nanmax", "nanmin" };

        /// <summary>
        ///     True when the case reduces exactly one float16 operand to a float16 result — the shape of every case the
        ///     float16 min/max contracts (<see cref="HalfSelectionContractOps"/>, <see cref="IsHalfPtpContractCase"/>) cover.
        /// </summary>
        /// <param name="c">The corpus case.</param>
        /// <param name="tc">The type code of the result being compared.</param>
        /// <returns>
        ///     <see langword="true"/> for a float16 result of a single float16 operand; <see langword="false"/> for a
        ///     <c>dtype=</c> cast or a multi-operand case, whose bits the contracts do not describe.
        /// </returns>
        private static bool IsSingleHalfOperand(FuzzCorpus.Case c, NPTypeCode tc)
            => tc == NPTypeCode.Half && c.Operands.Length == 1 && c.Operands[0].Dtype == "float16";

        /// <summary>
        ///     True when the case is a float16 <c>ptp</c> whose bits are a contract on this host.
        /// </summary>
        /// <param name="c">The corpus case.</param>
        /// <param name="tc">The type code of the result being compared.</param>
        /// <returns>
        ///     <see langword="true"/> on an x86 host (<see cref="NanSignIsX86Reference"/>) for a <c>ptp</c> of one float16
        ///     operand; its bytes then compare raw and any difference is a hard failure.
        /// </returns>
        /// <remarks>
        ///     NumPy's <c>_ptp</c> (numpy/_core/_methods.py) is <c>subtract(maximum.reduce(a), minimum.reduce(a))</c>: two of
        ///     the selections above, which return the SAME first NaN, and one HALF_subtract, which computes in float32 and
        ///     narrows once — so a signalling NaN comes back quieted, identically on every host. One step is hardware
        ///     dependent: <c>inf - inf</c> makes the CPU's default NaN, negative on x86 (the recorded win-amd64 0xfe00) and
        ///     positive on arm64. So the raw-byte comparison holds on x86 hosts only; elsewhere NaN stays tokenized, as for
        ///     every other op.
        /// </remarks>
        private static bool IsHalfPtpContractCase(FuzzCorpus.Case c, NPTypeCode tc)
            => NanSignIsX86Reference && c.Op == "ptp" && IsSingleHalfOperand(c, tc);

        /// <summary>
        ///     The complex NaN-SIGN contract above is authored against numpy 2.4.2 win-amd64 (MSVC UCRT),
        ///     and NumSharp reproduces it in managed code + x64 SSE — so it holds bit-for-bit on the whole
        ///     x86 family (Windows AND Linux, shared deterministic SSE), but NOT on AArch64: a fresh NaN
        ///     from a genuine 0/0·inf-inf inside the complex arithmetic carries the CPU's default-NaN
        ///     sign, which is POSITIVE on ARM where x86 emits NEGATIVE. Off the x86 reference the raw-byte
        ///     NaN-sign compare would hard-fail via <see cref="BitDiff.DiffHasSignFlip"/> on a platform
        ///     artifact, so the sign contract is gated to x86 and NaN tokenizes elsewhere (the finite
        ///     complex VALUES in these tiers are proven cross-platform — they were macOS-green before this
        ///     comparison existed). Same host-pin RULE as the RunHostLibmCorpus tiers, applied per-op.
        /// </summary>
        private static readonly bool NanSignIsX86Reference =
            System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture
                is System.Runtime.InteropServices.Architecture.X86
                or System.Runtime.InteropServices.Architecture.X64;

        /// <summary>
        ///     True on a host where the NaN-operand-order tier (<c>nan_order.jsonl</c>) is a byte contract:
        ///     an x86 process with NumPy's X86_V3 feature set (AVX2, FMA3, BMI1, BMI2, LZCNT).
        /// </summary>
        /// <remarks>
        ///     The tier is authored against numpy 2.4.2 win-amd64, whose AVX2 scalar-broadcast loop
        ///     splits its NaN choice at blocks of two 32-byte vectors. NumSharp derives that block from
        ///     the host's features exactly as NumPy's dispatcher would (16-byte vectors without X86_V3),
        ///     so on a host without X86_V3 BOTH libraries would split at a different point than the
        ///     recorded bytes and the comparison would fail on a platform artifact. Such hosts, and
        ///     arm64 (whose NumPy is compiled by another compiler with another operand order), compare
        ///     NaN as one token instead — the same host rule as <see cref="NanSignIsX86Reference"/>.
        ///     NumSharp's own NaN choice does not depend on the operating system, so Linux x64 is gated
        ///     like Windows.
        /// </remarks>
        private static readonly bool NaNOperandOrderHost =
            NanSignIsX86Reference
            && System.Runtime.Intrinsics.X86.Avx2.IsSupported
            && System.Runtime.Intrinsics.X86.Fma.IsSupported
            && System.Runtime.Intrinsics.X86.Bmi1.IsSupported
            && System.Runtime.Intrinsics.X86.Bmi2.IsSupported
            && System.Runtime.Intrinsics.X86.Lzcnt.IsSupported;

        /// <summary>
        ///     True for a case of the NaN-operand-order tier on a host where that tier is a byte contract.
        /// </summary>
        /// <param name="c">The corpus case to test.</param>
        /// <returns>
        ///     <see langword="true"/> when the case carries <c>params.nan_bits = true</c> (written only by
        ///     <c>gen_oracle.gen_nan_order</c>) and <see cref="NaNOperandOrderHost"/> holds; then its NaN
        ///     payloads compare raw and any payload or sign difference fails outright.
        /// </returns>
        /// <remarks>
        ///     When both operands of a float add or multiply are NaN, the result is the NaN of the
        ///     operand the compiled loop put first. Every other tier tokenizes NaN, so this marker is the
        ///     one place that choice is gated; it rides on the case rather than the op name because the
        ///     same ops (<c>add</c>, <c>multiply</c>, <c>out_binary</c>, <c>evaluate</c>) appear in
        ///     tiers whose layouts include NumSharp's documented residuals.
        /// </remarks>
        private static bool IsNaNOperandOrderCase(FuzzCorpus.Case c)
            => NaNOperandOrderHost
               && c.Params != null
               && c.Params.TryGetValue("nan_bits", out var marker)
               && marker.ValueKind == System.Text.Json.JsonValueKind.True;

        // ---- new tiers ---------------------------------------------------------------------

        /// <summary>
        ///     Iterator TRACES. np.ndindex / np.ndenumerate / np.nditer / np.broadcast produce no
        ///     array, so they were deliberately absent from this corpus — but their MATERIALIZED
        ///     iteration is an ordinary array artifact, and it is the one that matters: traversal
        ///     ORDER. NumPy is the only oracle for how C/F/A/K resolve, how external_loop chunks,
        ///     and what a strided/broadcast/F-contiguous operand yields. Every other tier depends
        ///     on NDIter agreeing with NumPy here, so nothing else in the corpus can catch a drift
        ///     in it.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Iter() => RunCorpus("iter.jsonl");

        /// <summary>
        ///     The non-array result kinds: dtype-returning promotion helpers (result_type /
        ///     promote_types / min_scalar_type), scalar-returning predicates (can_cast / isscalar /
        ///     iscomplexobj / isrealobj / size), text-returning printing (array_str / array_repr),
        ///     and tuple-returning selection (nonzero / meshgrid) where ARITY is asserted too.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void DtypeText() => RunCorpus("dtype_text.jsonl");

        /// <summary>
        ///     Public multi-output APIs. Every slot is recorded so the gate asserts tuple arity
        ///     as well as each output's dtype, shape and bytes; this supersedes the older
        ///     one-piece-at-a-time coverage for split-like functions.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void MultiOutput() => RunCorpus("multioutput.jsonl");

        /// <summary>
        ///     Error parity with NumPy's ACTUAL message. The deterministic op matrices are re-run
        ///     by the generator keeping only the cells where NumPy raises — cells the value tiers
        ///     silently skip — and each records the exception type and text verbatim. This is a
        ///     separate corpus on purpose: it leaves the (large, shared) value tiers byte-identical
        ///     rather than rewriting 87K committed lines to interleave error rows.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void ErrorsFull() => RunCorpus("errors_full.jsonl");

        /// <summary>
        ///     ufunc <c>out=</c> / <c>where=</c>. Previously reached only through maximum_out /
        ///     minimum_out / clip_out (11 cases each, contiguous out, no mask), so nothing these
        ///     parameters actually promise was gated. Every case records TWO slots — what the call
        ///     returned, and the ENTIRE base buffer behind <c>out</c> — because:
        ///
        ///     <list type="bullet">
        ///       <item><c>where</c> masking is defined by what does NOT change, so the out array's
        ///             prior contents are an operand and are re-checked afterwards.</item>
        ///       <item>a strided / offset / negstride / F-order / transposed <c>out</c> is where a
        ///             kernel that walks the buffer instead of the view corrupts elements OUTSIDE
        ///             the window — which a view-shaped comparison cannot see.</item>
        ///     </list>
        ///
        ///     A read-only (broadcast) <c>out</c> must be refused; those ride the error machinery.
        /// </summary>
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void OutWhere() => RunCorpus("out_where.jsonl");

        // ---- comparators -------------------------------------------------------------------

        /// <summary>
        ///     The original (dtype, shape, bytes) comparison, factored out so tuple slots reuse it
        ///     verbatim. <paramref name="slot"/> labels which tuple slot is being compared (null
        ///     for a plain result) so a failure names it. When the expectation carries a memory facet
        ///     (<see cref="FuzzCorpus.Expected.Mem"/>, the view-flags tiers) it is compared too, against
        ///     <paramref name="operands"/>, once dtype and shape agree.
        /// </summary>
        /// <param name="c">The corpus case.</param>
        /// <param name="result">NumSharp's result (one array, or one tuple slot).</param>
        /// <param name="exp">NumPy's expectation for it.</param>
        /// <param name="slot">The tuple-slot label, or null for a plain result.</param>
        /// <param name="failures">Receives one line per unexcused divergence.</param>
        /// <param name="documented">Counts the registry-excused divergences by reason.</param>
        /// <param name="operands">The case's reconstructed operands, for the facet's shares / same relations; null
        ///     when the caller has none (the facet is then skipped — only the view-flags tiers record one, and their
        ///     replay always passes the operands).</param>
        private static void CompareArray(
            FuzzCorpus.Case c, NDArray result, FuzzCorpus.Expected exp, string slot,
            List<string> failures, Dictionary<string, int> documented, NDArray[] operands = null)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            string at = slot == null ? "" : $" {slot}";

            // NumPy can produce a dtype NumSharp has no NPTypeCode for — complex64, from an fft over
            // a float32/float16 input (NumSharp promotes to double and returns complex128). That
            // unnameable dtype IS the divergence: route it as a Dtype difference for the registry to
            // excuse, exactly as a mappable dtype mismatch would be. No other tier records an
            // unmappable dtype (verified), so this only fires for the fft float32/float16 cells.
            NPTypeCode tc;
            try { tc = FuzzCorpus.DtypeToTC(exp.Dtype); }
            catch (NotSupportedException) when (OpRegistry.IsPolynomialOp(c.Op) && exp.Dtype == "complex64")
            {
                // numpy.polynomial makes complex64 values under NEP 50 (a float16/float32 value meeting a Python
                // complex, or a complex64 NumPy scalar meeting a narrow value): complex64 SCALARS in its scalar code and
                // complex64 ARRAY loops in mapdomain's `off + scl*x`. NumSharp emulates both in float32 exactly (the
                // scalar engine; the mapdomain complex64 kernels) and carries the value in its one complex dtype,
                // complex128 (#569). The generator records exactly those results, so the VALUES must match NumPy's:
                // up-cast NumPy's complex64 (exact) and compare, NaN tokenized as for every binary op. The polynomial
                // tiers take no registry excuse, so this is the one documented divergence they report — dtype-only,
                // never a value.
                if (result.typecode == NPTypeCode.Complex && Complex64ValuesMatch(exp, result))
                    Bump(documented, "complex64 result carried as complex128 (#569: no complex64 dtype; values exact)");
                else
                    failures.Add($"{c.Id} [{c.Layout}]{at}: NumPy complex64 result, NumSharp {result.typecode}{result.Shape}: " +
                                 "VALUES diverge after up-cast (not dtype-only)");
                return;
            }
            catch (NotSupportedException)
            {
                var dr = MisalignedRegistry.Classify(c, DivergenceKind.Dtype, null, null, default, empty);
                if (dr == null)
                    failures.Add($"{c.Id} [{c.Layout}]{at}: NumPy dtype '{exp.Dtype}' has no NumSharp analog");
                // F1 excuses the complex64 dtype — but the VALUES must still be bit-exact (the divergence
                // is dtype-ONLY, not a compute gap). Up-cast NumPy's complex64 and bit-compare.
                else if (IsFftFloatCell(c, exp, result) && !FftFloatValuesMatch(exp, result))
                    failures.Add($"{c.Id} [{c.Layout}]{at}: fft {exp.Dtype} VALUES diverge from NumPy after up-cast (not dtype-only)");
                else Bump(documented, dr);
                return;
            }

            // NEP50: result dtype must match NumPy exactly (the headline promotion failure).
            if (result.typecode != tc)
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Dtype, null, null, tc, empty);
                if (reason == null)
                    failures.Add($"{c.Id} [{c.Layout}]{at}: result dtype {result.typecode} != NumPy {exp.Dtype}");
                // As above (irfft/hfft returning float64 where NumPy returns float32/float16): the excused
                // dtype divergence must still carry bit-exact values.
                else if (IsFftFloatCell(c, exp, result) && !FftFloatValuesMatch(exp, result))
                    failures.Add($"{c.Id} [{c.Layout}]{at}: fft {exp.Dtype} VALUES diverge from NumPy after up-cast (not dtype-only)");
                else Bump(documented, reason);
                return;
            }
            // Broadcasting: result shape must match NumPy.
            if (!ShapeEquals(result.Shape.dimensions, exp.Shape))
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Shape, null, null, tc, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]{at}: result shape [{string.Join(",", result.Shape.dimensions)}] " +
                                  $"!= NumPy [{string.Join(",", exp.Shape)}]");
                return;
            }

            // The memory facet (view-flags tiers): checked once dtype and shape agree, reported on its own line so a
            // value divergence of the same case stays visible beside it. No registry branch excuses a facet: the
            // tiers that record one are held to zero excuses.
            if (exp.Mem != null && operands != null)
            {
                string mem = CompareMem(result, exp.Mem, operands);
                if (mem != null)
                    failures.Add($"{c.Id} [{c.Layout}]{at}: memory facet: {mem}");
            }

            var actual = FuzzCorpus.ResultBytes(result);
            var expected = FuzzCorpus.FromHex(exp.Buffer);

            // The complex-unary ufuncs (NDComplexMath) reproduce NumPy 2.4.2's NaN SIGN bit-for-bit
            // (MSVC UCRT convention: sqrt/log/exp/... +NaN, csin/ctan via the transform's -Re negate,
            // ctanh sign-propagating, sign/abs canonicalise to +NaN), so for those ops the NaN sign is
            // CONTRACTUAL and is compared by raw bytes. The trigger keys on a complex OPERAND, not the
            // result dtype, so `abs`/`absolute` (complex128 -> float64) is gated too. Everywhere else NaN
            // stays tokenized (a differing payload/sign is non-contractual — order/algorithm-dependent).
            // The nan_order tier's cases (params.nan_bits) gate the float add / multiply NaN-operand
            // order the same way: their NaN payloads are contractual, so they compare raw as well.
            // float16 max / min / nanmax / nanmin copy an input element (HalfSelectionContractOps): raw bytes, portable.
            // float16 ptp subtracts two of those selections: raw bytes on x86 hosts (IsHalfPtpContractCase).
            bool halfSelection = HalfSelectionContractOps.Contains(c.Op) && IsSingleHalfOperand(c, tc);
            bool halfPtp = IsHalfPtpContractCase(c, tc);
            bool nanExact = halfSelection
                            || halfPtp
                            || (ComplexNanContractOps.Contains(c.Op)
                                && (tc == NPTypeCode.Complex || c.Operands.Any(o => o.Dtype == "complex128"))
                                && NanSignIsX86Reference)   // win-amd64 NaN-sign contract; tokenize off x86 (arm64)
                            || IsNaNOperandOrderCase(c);

            // Bit-exact to NumPy ("precise") passes HERE, before truth is ever read — matching
            // NumPy's bytes is the contract, and truth can never turn a precise result red.
            var diffs = BitDiff.Compare(expected, actual, tc, nanExact);
            if (diffs.Count == 0)
                return;

            var truth = exp.Truth == null ? null : FuzzCorpus.FromHex(exp.Truth);

            // A pure NaN-sign / signed-zero flip in a NaN-contract complex-unary op is a HARD failure:
            // it must bypass the ULP / pathological-edge excuses, which report NaN-vs-NaN as 0 ULP and
            // would otherwise swallow the regression this gate exists to catch. A finite rounding
            // residual (nanExact but no sign flip) still flows through the registry (the ≤3-ULP excuse).
            // A float16 selection has no rounding residual at all, and float16 ptp rounds exactly as NumPy's HALF_subtract
            // does (float32 arithmetic, one narrowing): every difference is hard.
            bool signFlip = nanExact && diffs.Any(d => BitDiff.DiffHasSignFlip(expected, actual, d.Index, tc));
            var vreason = signFlip || halfSelection || halfPtp ? null
                : MisalignedRegistry.Classify(c, DivergenceKind.Value, expected, actual, tc, diffs, truth);
            if (vreason != null)
            {
                Bump(documented, vreason);
                return;
            }

            // Shrinking rebuilds the case as a 1-element repro, which only makes sense for the
            // elementwise single-array shape — skip it for tuple slots.
            var shrunk = slot == null ? Shrinker.ShrinkElementwise(c, diffs[0].Index) : null;
            failures.Add($"{c.Id} [{c.Layout}]{at}: " +
                (halfSelection ? "float16 selection contract violation "
                    : halfPtp ? "float16 ptp contract violation "
                    : signFlip ? "NaN-sign/signed-zero contract violation " : "") +
                string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}" +
                    TruthNote(expected, actual, truth, d.Index, tc))) +
                (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : "") +
                (shrunk != null ? $"\n      minimal repro: {shrunk}" : ""));
        }

        /// <summary>
        ///     An fft transform cell where NumSharp promoted the result (complex128/float64) because it
        ///     has no complex64 dtype, while NumPy returned complex64/float32/float16 — the F1 divergence.
        ///     Helpers (fftfreq/rfftfreq/fftshift/ifftshift) never promote, so they are excluded.
        /// </summary>
        private static bool IsFftFloatCell(FuzzCorpus.Case c, FuzzCorpus.Expected exp, NDArray result)
            => c.Op.Contains("fft") && !c.Op.Contains("freq") && !c.Op.Contains("shift")
               && (exp.Dtype == "complex64" || exp.Dtype == "float32" || exp.Dtype == "float16")
               && (result.typecode == NPTypeCode.Complex || result.typecode == NPTypeCode.Double);

        /// <summary>
        ///     Verify the fft float32/float16 VALUES are bit-identical to NumPy despite the dtype
        ///     divergence: up-cast NumPy's complex64/float32/float16 result bytes to NumSharp's
        ///     complex128/float64 precision (an exact widening) and bit-compare. True == dtype-only.
        /// </summary>
        private static bool FftFloatValuesMatch(FuzzCorpus.Expected exp, NDArray result)
        {
            byte[] want = FuzzCorpus.FromHex(exp.Buffer);   // NumPy complex64 / float32 / float16
            byte[] got = FuzzCorpus.ResultBytes(result);    // NumSharp complex128 / float64
            byte[] up;
            switch (exp.Dtype)
            {
                case "complex64":                            // 2 float32 -> 2 float64 per element
                case "float32":                              // float32 -> float64
                    up = new byte[want.Length * 2];
                    for (int i = 0; i + 4 <= want.Length; i += 4)
                        BitConverter.GetBytes((double)BitConverter.ToSingle(want, i)).CopyTo(up, i * 2);
                    break;
                case "float16":                              // float16 -> float64
                    up = new byte[want.Length * 4];
                    for (int i = 0; i + 2 <= want.Length; i += 2)
                        BitConverter.GetBytes((double)BitConverter.Int16BitsToHalf(BitConverter.ToInt16(want, i))).CopyTo(up, i * 4);
                    break;
                default:
                    return false;
            }
            if (up.Length != got.Length) return false;
            for (int i = 0; i < up.Length; i++)
                if (up[i] != got[i]) return false;
            return true;
        }

        /// <summary>
        ///     Whether NumSharp's complex128 <paramref name="result"/> holds exactly NumPy's complex64 values: the
        ///     expected bytes (two float32 per element) are widened exactly to two float64 and compared with the
        ///     result through <see cref="BitDiff.Compare(byte[],byte[],NPTypeCode)"/> — same shape, every finite
        ///     component and signed zero bit-exact, NaN tokenized (which NaN payload survives a float32 operation is
        ///     non-contractual, as for every binary op).
        /// </summary>
        /// <param name="exp">NumPy's expected result (dtype complex64).</param>
        /// <param name="result">NumSharp's result (complex128).</param>
        /// <returns>True when the values match: the divergence is the dtype alone.</returns>
        private static bool Complex64ValuesMatch(FuzzCorpus.Expected exp, NDArray result)
        {
            if (!ShapeEquals(result.Shape.dimensions, exp.Shape))
                return false;
            byte[] want = FuzzCorpus.FromHex(exp.Buffer);
            byte[] up = new byte[want.Length * 2];
            for (int i = 0; i + 4 <= want.Length; i += 4)
                BitConverter.GetBytes((double)BitConverter.ToSingle(want, i)).CopyTo(up, i * 2);
            byte[] got = FuzzCorpus.ResultBytes(result);
            return up.Length == got.Length && BitDiff.Compare(up, got, NPTypeCode.Complex).Count == 0;
        }

        /// <summary>
        ///     For a truth-bearing failure, say WHO lost precision right in the failure line:
        ///     NumSharp's and NumPy's ULP distances to the correctly-rounded reference. An
        ///     untruthful divergence then reads e.g. "(truth-ulp NS=512 NPY=2)" — the precision
        ///     loss quantified — instead of two opaque hex tokens.
        /// </summary>
        private static string TruthNote(byte[] expected, byte[] actual, byte[] truth, int index, NPTypeCode tc)
        {
            if (truth == null || truth.Length != expected.Length)
                return "";
            long dNS = BitDiff.UlpDistance(actual, truth, index, tc);
            long dNPY = BitDiff.UlpDistance(expected, truth, index, tc);
            static string F(long d) => d == long.MaxValue ? "max" : d.ToString();
            return $" (truth-ulp NS={F(dNS)} NPY={F(dNPY)})";
        }

        /// <summary>
        ///     Multi-output ops: compare the ARITY first, then every slot. The older which/piece
        ///     params record one slot per case and so can never catch a tuple that came back with
        ///     the wrong number of entries (np.nonzero on a 2-D array being the standing example).
        /// </summary>
        private static void CompareTuple(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var got = OpRegistry.ApplyTuple(c.Op, c.Params, operands);
            var want = c.Expected.Slots ?? Array.Empty<FuzzCorpus.Expected>();

            if (got.Length != want.Length)
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Arity, null, null, default,
                                                         Array.Empty<BitDiff.Diff>());
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]: tuple arity {got.Length} != NumPy {want.Length}");
                return;
            }

            for (int i = 0; i < want.Length; i++)
                CompareArray(c, got[i], want[i], $"slot[{i}]", failures, documented, operands);

            // The view-flags tiers also record which slots share memory and which are one object (np.nonzero's columns
            // of one buffer, np.diag_indices' repeated slot, np.meshgrid's views): NumSharp's slots must pair the same way.
            if (c.Expected.Xshare != null || c.Expected.Xsame != null)
            {
                string cross = CompareCrossSlots(got, c.Expected.Xshare, c.Expected.Xsame);
                if (cross != null)
                    failures.Add($"{c.Id} [{c.Layout}]: tuple slots: {cross}");
            }
        }

        // ---- the memory facet (view-flags tiers) -------------------------------------------------

        /// <summary>The flag names in <see cref="FuzzCorpus.MemFacet.Flags"/> order, for failure lines.</summary>
        private static readonly string[] MemFlagNames = { "C_CONTIGUOUS", "F_CONTIGUOUS", "OWNDATA", "WRITEABLE", "ALIGNED" };

        /// <summary>
        ///     Compares NumSharp's result against NumPy's memory facet: the five flags, the normalized byte strides, and
        ///     per operand whether the result shares its memory and whether it IS the operand.
        /// </summary>
        /// <param name="result">NumSharp's result array.</param>
        /// <param name="want">NumPy's facet.</param>
        /// <param name="operands">The case's operands, in descriptor order (the facet's per-operand lists follow it).</param>
        /// <returns>Null when every part agrees; otherwise one line naming each part that differs.</returns>
        /// <remarks>
        ///     Strides are normalized on NumSharp's side exactly as the generator normalized NumPy's: an axis of extent
        ///     0 or 1, and every axis of an empty array, reads 0. <c>np.shares_memory</c> is NumSharp's own exact solver
        ///     (<c>max_work = -1</c>), NumPy's answer being exact too; an empty array shares memory with nothing on either
        ///     side. Identity is <see cref="object.ReferenceEquals"/>, NumPy's <c>is</c>.
        /// </remarks>
        private static string CompareMem(NDArray result, FuzzCorpus.MemFacet want, NDArray[] operands)
        {
            var parts = new List<string>();
            var f = result.flags;
            bool[] got = { f.c_contiguous, f.f_contiguous, f.owndata, f.writeable, f.aligned };
            if (want.Flags != null)
                for (int i = 0; i < Math.Min(got.Length, want.Flags.Length); i++)
                    if (got[i] != want.Flags[i])
                        parts.Add($"{MemFlagNames[i]} {got[i]} (NumPy {want.Flags[i]})");

            if (want.Strides != null)
            {
                long[] dims = result.Shape.dimensions;
                long[] bytes = result.strides;
                bool empty = result.size == 0;
                var norm = new long[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    norm[i] = empty || dims[i] <= 1 ? 0 : bytes[i];
                if (!norm.SequenceEqual(want.Strides))
                    parts.Add($"strides ({string.Join(",", norm)}) (NumPy ({string.Join(",", want.Strides)}))");
            }

            int n = Math.Min(operands.Length, Math.Max(want.Shares?.Length ?? 0, want.Same?.Length ?? 0));
            for (int i = 0; i < n; i++)
            {
                var op = operands[i];
                if (want.Same != null && i < want.Same.Length)
                {
                    bool same = ReferenceEquals(result, op);
                    if (same != want.Same[i])
                        parts.Add(want.Same[i]
                            ? $"returned a new object, NumPy returns operand {i} itself"
                            : $"returned operand {i} itself, NumPy returns a new object");
                }
                if (want.Shares != null && i < want.Shares.Length && op is not null)
                {
                    bool shares = np.shares_memory(result, op);
                    if (shares != want.Shares[i])
                        parts.Add(want.Shares[i]
                            ? $"a COPY of operand {i}, NumPy returns a VIEW of it"
                            : $"a VIEW of operand {i}, NumPy returns memory of its own");
                }
            }
            return parts.Count == 0 ? null : string.Join("; ", parts);
        }

        /// <summary>
        ///     Compares which slots of a tuple result share memory and which are one object, against NumPy's pairs.
        /// </summary>
        /// <param name="slots">NumSharp's tuple slots.</param>
        /// <param name="wantShare">NumPy's sharing pairs (null when not recorded).</param>
        /// <param name="wantSame">NumPy's identity pairs (null when not recorded).</param>
        /// <returns>Null when both pair sets agree; otherwise a line naming the differing pairs.</returns>
        private static string CompareCrossSlots(NDArray[] slots, int[][] wantShare, int[][] wantSame)
        {
            var gotShare = new HashSet<(int, int)>();
            var gotSame = new HashSet<(int, int)>();
            for (int i = 0; i < slots.Length; i++)
                for (int j = i + 1; j < slots.Length; j++)
                {
                    if (ReferenceEquals(slots[i], slots[j]))
                        gotSame.Add((i, j));
                    if (slots[i] is not null && slots[j] is not null && np.shares_memory(slots[i], slots[j]))
                        gotShare.Add((i, j));
                }

            var parts = new List<string>();
            void Diff(string what, HashSet<(int, int)> got, int[][] want)
            {
                if (want == null)
                    return;
                var w = new HashSet<(int, int)>(want.Select(p => (p[0], p[1])));
                if (!got.SetEquals(w))
                    parts.Add($"{what} pairs {{{string.Join(" ", got.OrderBy(x => x).Select(x => $"{x.Item1}-{x.Item2}"))}}} " +
                              $"(NumPy {{{string.Join(" ", w.OrderBy(x => x).Select(x => $"{x.Item1}-{x.Item2}"))}}})");
            }
            Diff("shared-memory", gotShare, wantShare);
            Diff("same-object", gotSame, wantSame);
            return parts.Count == 0 ? null : string.Join("; ", parts);
        }

        /// <summary>
        ///     Promotion helpers return a dtype, not an array — compare NumPy's dtype NAME. Routed
        ///     through the registry as a Dtype divergence so the documented NEP50 weak-scalar
        ///     difference covers result_type over a 0-D operand, exactly as it covers the result
        ///     dtype of a binary op over the same operands.
        /// </summary>
        private static void CompareDtypeResult(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var got = OpRegistry.ApplyDtype(c.Op, c.Params, operands).AsNumpyDtypeName();
            if (string.Equals(got, c.Expected.Value, StringComparison.Ordinal))
                return;

            var reason = MisalignedRegistry.Classify(c, DivergenceKind.Dtype, null, null, default,
                                                     Array.Empty<BitDiff.Diff>());
            if (reason != null) Bump(documented, reason);
            else failures.Add($"{c.Id} [{c.Layout}]: dtype result '{got}' != NumPy '{c.Expected.Value}'");
        }

        /// <summary>Printing returns text — compare verbatim (this is a byte-exact port claim).</summary>
        private static void CompareTextResult(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var got = OpRegistry.ApplyText(c.Op, c.Params, operands);
            if (string.Equals(got, c.Expected.Value, StringComparison.Ordinal))
                return;

            var reason = MisalignedRegistry.Classify(c, DivergenceKind.Text, null, null, default,
                                                     Array.Empty<BitDiff.Diff>());
            if (reason != null) Bump(documented, reason);
            else failures.Add($"{c.Id} [{c.Layout}]: text differs\n" +
                              $"      numpy:    {Show(c.Expected.Value)}\n" +
                              $"      numsharp: {Show(got)}");
        }

        // ---- error parity ------------------------------------------------------------------

        /// <summary>
        ///     NumPy's exception classes and the .NET types NumSharp is allowed to answer them
        ///     with. The taxonomies genuinely do not align 1:1 (NumPy's ValueError covers what .NET
        ///     splits across ArgumentException / InvalidOperationException / FormatException, and
        ///     NumSharp deliberately raises its own IncorrectShapeException / AxisOutOfRangeException),
        ///     so the TYPE check is a class-level sanity net and the MESSAGE carries the contract.
        /// </summary>
        private static readonly Dictionary<string, string[]> ErrorTypeMap = new()
        {
            ["ValueError"] = new[]
            {
                "ArgumentException", "ArgumentNullException", "ArgumentOutOfRangeException",
                "IncorrectShapeException", "AxisOutOfRangeException", "InvalidOperationException",
                "FormatException", "OverflowException", "NotSupportedException",
                // ndarray.item's size guard (NumPy: ValueError "can only convert an array of
                // size 1 to a Python scalar") is the house IncorrectSizeException.
                "IncorrectSizeException"
            },
            ["TypeError"] = new[]
            {
                "NotSupportedException", "InvalidCastException", "ArgumentException",
                "ArgumentNullException", "InvalidOperationException", "UFuncTypeException",
                // IncorrectTypeException now derives from TypeError (the ufunc "No loop matching…" /
                // "not supported for the input types" / digitize / i0 messages are all NumPy TypeErrors),
                // but its .GetType().Name stays "IncorrectTypeException", so name it here explicitly.
                "IncorrectTypeException"
            },
            ["IndexError"] = new[]
            {
                "IndexOutOfRangeException", "ArgumentOutOfRangeException", "ArgumentException",
                "AxisOutOfRangeException"
            },
            ["AxisError"] = new[]
            {
                "AxisOutOfRangeException", "ArgumentOutOfRangeException", "ArgumentException",
                "IndexOutOfRangeException"
            },
            ["OverflowError"] = new[] { "OverflowException", "ArgumentException" },
            ["ZeroDivisionError"] = new[] { "DivideByZeroException", "ArithmeticException" },
            ["MemoryError"] = new[] { "OutOfMemoryException" },
            ["NotImplementedError"] = new[] { "NotImplementedException", "NotSupportedException" },
        };

        /// <summary>
        ///     Assert error parity. Three tiers, weakest first:
        ///     (1) it must throw at all — a silent result where NumPy raises is always a failure;
        ///     (2) if NumPy's exception was recorded, the .NET type must be a plausible counterpart;
        ///     (3) and the message must match NumPy's verbatim.
        ///     Tiers 2-3 route through <see cref="MisalignedRegistry"/> so a documented wording
        ///     difference is excused-but-printed rather than silently accepted.
        /// </summary>
        private static void CheckError(
            FuzzCorpus.Case c, NDArray[] operands, List<string> failures, Dictionary<string, int> documented)
        {
            var empty = Array.Empty<BitDiff.Diff>();
            Exception caught = null;
            try
            {
                _ = OpRegistry.Invoke(c.Expected?.KindOrArray ?? "array", c.Op, c.Params, operands);
            }
            catch (Exception e)
            {
                caught = e;
            }

            if (caught == null)
            {
                var reason = MisalignedRegistry.Classify(c, DivergenceKind.Value, null, null, default, empty);
                if (reason != null) Bump(documented, reason);
                else failures.Add($"{c.Id} [{c.Layout}]: NumPy raises " +
                                  $"{c.Error?.Type ?? "an error"} but NumSharp produced a result (error-parity gap)");
                return;
            }

            // Legacy cases carry no recorded exception — "it threw" is all they ever asserted.
            if (c.Error == null)
                return;

            string gotType = caught.GetType().Name;
            string gotText = NormalizeMessage(caught.Message);
            // Trimmed like NumSharp's side (NormalizeMessage): NumPy's iterator texts END in a space
            // ("operands could not be broadcast together with shapes (3,) (4,) "), which a one-sided trim
            // could never match.
            string wantText = (c.Error.Text ?? "").Trim();

            // NumSharp names several exceptions exactly after NumPy's (Exceptions/{ValueError,
            // TypeError,IndexError,AxisError}.cs), so an identical name is the strongest possible
            // match and is accepted before the taxonomy map is consulted at all.
            bool typeOk = gotType == c.Error.Type
                          || !ErrorTypeMap.TryGetValue(c.Error.Type, out var allowed)
                          || allowed.Contains(gotType);
            bool textOk = string.Equals(gotText, wantText, StringComparison.Ordinal);

            if (typeOk && textOk)
                return;

            var er = MisalignedRegistry.Classify(c, DivergenceKind.ErrorText, null, null, default, empty);
            if (er != null)
            {
                Bump(documented, er);
                return;
            }

            failures.Add($"{c.Id} [{c.Layout}]: error mismatch\n" +
                         $"      numpy:    {c.Error.Type}: {Show(wantText)}\n" +
                         $"      numsharp: {gotType}: {Show(gotText)}");
        }

        /// <summary>
        ///     Strip .NET's framing so the comparison is about the MESSAGE, not the platform:
        ///     ArgumentException appends " (Parameter 'name')" and multi-line messages carry the
        ///     parameter on its own line. NumPy's text is otherwise compared byte-for-byte.
        /// </summary>
        private static string NormalizeMessage(string m)
        {
            if (string.IsNullOrEmpty(m))
                return "";

            int p = m.IndexOf(" (Parameter '", StringComparison.Ordinal);
            if (p >= 0)
                m = m.Substring(0, p);

            int nl = m.IndexOf("\nParameter name:", StringComparison.Ordinal);
            if (nl >= 0)
                m = m.Substring(0, nl);

            return m.Trim();
        }

        /// <summary>Render a string for a one-line failure report: escaped and length-capped.</summary>
        private static string Show(string s)
        {
            if (s == null)
                return "<null>";
            s = s.Replace("\r", "\\r").Replace("\n", "\\n");
            return s.Length <= 200 ? $"\"{s}\"" : $"\"{s.Substring(0, 200)}…\" ({s.Length} chars)";
        }
    }
}
