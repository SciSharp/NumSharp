using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Tests.Fuzz;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Regression pins for the differential-fuzz GATE-TIGHTENING pass (COMPLETENESS_PLAN
    ///     WS-BUGS): every excuse branch in <see cref="NumSharp.Tests.Fuzz.MisalignedRegistry"/>
    ///     was narrowed from a blanket to its documented (op, dtype, kind) cell, and each REAL bug
    ///     the tightening exposed is either fixed in src (verified by a normal test here) or pinned
    ///     under [OpenBugs] below. NumPy 2.4.2 is the source of truth for every assertion.
    /// </summary>
    [TestClass]
    public class FuzzGateRegressionTests
    {
        // ============================================================================
        //  F16 / B8: np.invert on non-integer dtypes must throw CLEANLY, never crash.
        //
        //  Historically np.invert(float) reached the IL BitwiseNot emitter with a dtype
        //  it has no opcode for and executed an ILLEGAL CPU INSTRUCTION
        //  (ExecutionEngineException — killed the whole test host; documented in
        //  test/oracle/gen_oracle.py's errors tier as the reason the invert spec was
        //  omitted). The loop-resolution guard in Default.Invert.cs now rejects every
        //  non-(bool|integer) loop dtype up front with NumPy's verbatim TypeError:
        //      "ufunc 'invert' not supported for the input types, and the inputs could
        //       not be safely coerced to any supported types according to the casting
        //       rule ''safe''"           (probed identical on NumPy 2.4.2)
        //  UfuncUnaryBatchOutWhereTests.Invert_FloatInputs_NotSupportedText_AndOrder
        //  pins the float64 message/order; these pin the remaining non-integer dtypes
        //  (float64 kept per plan-spec, plus Half / complex128 / decimal — decimal has
        //  no NumPy analog but must take the same clean-throw path, not the crash).
        // ============================================================================

        [TestMethod]
        public void Invert_Double_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new double[] { 1.5 }));
            act.Should().Throw<TypeError>()
               .WithMessage("ufunc 'invert' not supported for the input types, and the inputs " +
                            "could not be safely coerced to any supported types according to the casting rule ''safe''");
        }

        [TestMethod]
        public void Invert_Half_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new Half[] { (Half)1.5f }));
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }

        [TestMethod]
        public void Invert_Complex_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new System.Numerics.Complex[] { new(1, 2) }));
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }

        [TestMethod]
        public void Invert_Decimal_ThrowsCleanly_NoHostCrash()
        {
            Action act = () => np.invert(np.array(new decimal[] { 1.5m }));
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }

        [TestMethod]
        public void InvertOperator_Float_ThrowsCleanly_NoHostCrash()
        {
            var a = np.array(new float[] { 1.5f, 2.5f });
            Action act = () => { var _ = ~a; };
            act.Should().Throw<TypeError>().WithMessage("*ufunc 'invert' not supported*");
        }
    }

    /// <summary>
    ///     Tightness self-tests for <see cref="MisalignedRegistry.Classify"/> (B1-B7): each
    ///     excuse branch was narrowed from a blanket to its documented (op, dtype, kind) cell,
    ///     and these pin the NEGATIVE space — a synthetic GROSS regression in a neighbouring
    ///     cell must classify as NOT excused (null), while the documented divergence stays
    ///     excused. Without these, a future "helpful" re-broadening of a branch would be
    ///     invisible until a real regression sailed through the gate.
    /// </summary>
    [TestClass]
    public class MisalignedRegistryTightnessTests
    {
        private static FuzzCorpus.Case Case(string op, params (string dtype, long[] shape)[] operands)
            => new FuzzCorpus.Case
            {
                Id = "selftest/" + op,
                Op = op,
                Layout = "selftest",
                Operands = operands.Select(o => new FuzzCorpus.Operand
                {
                    Dtype = o.dtype,
                    Shape = o.shape,
                    Strides = o.shape.Select(_ => 1L).ToArray(),
                    Offset = 0,
                    BufferSize = o.shape.Aggregate(1L, (a, b) => a * b),
                }).ToArray(),
            };

        private static byte[] C128(double re, double im)
            => BitConverter.GetBytes(re).Concat(BitConverter.GetBytes(im)).ToArray();

        private static readonly BitDiff.Diff[] OneDiff = { new BitDiff.Diff(0, "aa", "bb") };

        private static double UlpUp(double v, int n = 1)
            => BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(v) + n);

        // ---- B1: integer-result reduction value divergence is NOT "summation precision" ----

        [TestMethod]
        public void B1_IntegerSumValueDiff_NotExcused()
        {
            var c = Case("sum", ("int32", new long[] { 8 }));
            var exp = BitConverter.GetBytes(100L);
            var act = BitConverter.GetBytes(101L);
            MisalignedRegistry.Classify(c, DivergenceKind.Value, exp, act, NPTypeCode.Int64,
                new[] { new BitDiff.Diff(0, "64", "65") }).Should().BeNull(
                "integer accumulation is exact/modular — a wrong int64 sum is a real bug");
        }

        [TestMethod]
        public void B1_FloatSumValueDiff_StillExcused()
        {
            var c = Case("sum", ("float64", new long[] { 8 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.3), BitConverter.GetBytes(UlpUp(0.3)), NPTypeCode.Double, OneDiff)
                .Should().NotBeNull("float summation-order drift is the documented divergence");
        }

        // ---- B2: complex binary per-op scopes ----

        [TestMethod]
        public void B2_ComplexAdd_GrossDiff_NotExcused()
        {
            var c = Case("add", ("complex128", new long[] { 1 }), ("complex128", new long[] { 1 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 2.0), C128(1.5, 2.0), NPTypeCode.Complex, OneDiff).Should().BeNull(
                "complex add is only excused within 2 ULP; 0.5 absolute is a real kernel bug");
        }

        [TestMethod]
        public void B2_ComplexAdd_OneUlp_StillExcused()
        {
            var c = Case("add", ("complex128", new long[] { 1 }), ("complex128", new long[] { 1 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 2.0), C128(UlpUp(1.0), 2.0), NPTypeCode.Complex, OneDiff).Should().NotBeNull();
        }

        [TestMethod]
        public void B2_ComplexMultiply_WrongMagnitude_NotExcused()
        {
            var c = Case("multiply", ("complex128", new long[] { 1 }), ("complex128", new long[] { 1 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1e10, 1.0), C128(2e10, 1.0), NPTypeCode.Complex, OneDiff).Should().BeNull(
                "a 2x-magnitude error is not the cancellation regime");
        }

        [TestMethod]
        public void B2_ComplexMultiply_CancellationResidue_StillExcused()
        {
            // expected exactly 0 (NumPy's rounding), actual a tiny residue at rounding scale of
            // the dominant component 1e10 (ulp(1e10) ~ 1.9e-6) — the documented regime.
            var c = Case("multiply", ("complex128", new long[] { 1 }), ("complex128", new long[] { 1 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(0.0, 1e10), C128(1e-6, 1e10), NPTypeCode.Complex, OneDiff).Should().NotBeNull();
        }

        [TestMethod]
        public void B2_ComplexPower_FiniteSignFlip_NotExcused()
        {
            var c = Case("power", ("complex128", new long[] { 1 }), ("complex128", new long[] { 1 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 0.5), C128(-1.0, 0.5), NPTypeCode.Complex, OneDiff).Should().BeNull(
                "a finite sign flip is far outside the 512 element-magnitude-ULP envelope");
        }

        [TestMethod]
        public void B2_ComplexMatmul_AnyDiff_NotExcused()
        {
            // The old blanket excused ANY 2-operand complex value diff — matmul included.
            var c = Case("matmul", ("complex128", new long[] { 2, 2 }), ("complex128", new long[] { 2, 2 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 2.0), C128(UlpUp(1.0), 2.0), NPTypeCode.Complex, OneDiff).Should().BeNull(
                "complex matmul has no excuse branch — it must be bit-exact");
        }

        // ---- B3: cumprod dtype excuse only on the size<=1 fast path ----

        [TestMethod]
        public void B3_CumprodDtype_FullSize_NotExcused()
        {
            MisalignedRegistry.Classify(Case("cumprod", ("int16", new long[] { 5 })),
                DivergenceKind.Dtype, null, null, NPTypeCode.Int64, System.Array.Empty<BitDiff.Diff>())
                .Should().BeNull("a full-size cumprod NEP50 widening miss is a real bug");
        }

        [TestMethod]
        public void B3_CumprodDtype_SizeOne_StillExcused()
        {
            MisalignedRegistry.Classify(Case("cumprod", ("int16", new long[] { 1 })),
                DivergenceKind.Dtype, null, null, NPTypeCode.Int64, System.Array.Empty<BitDiff.Diff>())
                .Should().NotBeNull("the documented ReduceCumMul bug is the size-1 fast path");
        }

        // ---- B4: modf threw excuse excludes f32/f64 ----

        [TestMethod]
        public void B4_ModfFloat64Threw_NotExcused()
        {
            MisalignedRegistry.Classify(Case("modf_frac", ("float64", new long[] { 4 })),
                DivergenceKind.Threw, null, null, default, System.Array.Empty<BitDiff.Diff>())
                .Should().BeNull("modf(float64) throwing would be a real regression");
        }

        [TestMethod]
        public void B4_ModfFloat16Threw_StillExcused()
        {
            MisalignedRegistry.Classify(Case("modf_frac", ("float16", new long[] { 4 })),
                DivergenceKind.Threw, null, null, default, System.Array.Empty<BitDiff.Diff>())
                .Should().NotBeNull();
        }

        // ---- B5: hyperbolic threw excuse scoped to the float16-promoting inputs ----

        [TestMethod]
        public void B5_SinhFloat64Threw_NotExcused()
        {
            MisalignedRegistry.Classify(Case("sinh", ("float64", new long[] { 4 })),
                DivergenceKind.Threw, null, null, default, System.Array.Empty<BitDiff.Diff>())
                .Should().BeNull("sinh(float64) throwing would be a real regression");
        }

        [TestMethod]
        public void B5_SinhInt8Threw_StillExcused()
        {
            MisalignedRegistry.Classify(Case("sinh", ("int8", new long[] { 4 })),
                DivergenceKind.Threw, null, null, default, System.Array.Empty<BitDiff.Diff>())
                .Should().NotBeNull("int8 promotes to float16 — the documented no-Half-kernel bug");
        }

        // ---- B6: isclose value excuse requires a complex128 operand ----

        [TestMethod]
        public void B6_IscloseRealValueDiff_NotExcused()
        {
            MisalignedRegistry.Classify(
                Case("isclose", ("float64", new long[] { 4 }), ("float64", new long[] { 4 })),
                DivergenceKind.Value, new byte[] { 0 }, new byte[] { 1 }, NPTypeCode.Boolean,
                new[] { new BitDiff.Diff(0, "00", "01") }).Should().BeNull(
                "the documented isclose bug involves a complex operand; real-dtype divergence is new");
        }

        [TestMethod]
        public void B6_IscloseComplexValueDiff_StillExcused()
        {
            MisalignedRegistry.Classify(
                Case("isclose", ("complex128", new long[] { 4 }), ("float64", new long[] { 4 })),
                DivergenceKind.Value, new byte[] { 0 }, new byte[] { 1 }, NPTypeCode.Boolean,
                new[] { new BitDiff.Diff(0, "00", "01") }).Should().NotBeNull();
        }

        // ---- B7: complex reduce/scan excuse requires a NaN token in the diffs ----

        [TestMethod]
        public void B7_ComplexMinFiniteDiff_NotExcused()
        {
            var c = Case("min", ("complex128", new long[] { 4 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 2.0), C128(3.0, 4.0), NPTypeCode.Complex,
                new[] { new BitDiff.Diff(0, "aa:bb", "cc:dd") }).Should().BeNull(
                "a finite complex min divergence is not 'NaN ordering'");
        }

        [TestMethod]
        public void B7_ComplexMinNaNDiff_StillExcused()
        {
            var c = Case("min", ("complex128", new long[] { 4 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(double.NaN, 4540.0), C128(double.NaN, double.NaN), NPTypeCode.Complex,
                new[] { new BitDiff.Diff(0, "NaN:aa", "NaN:NaN") }).Should().NotBeNull();
        }

        // ---- decimal std last-digit scope (surfaced by B1) ----

        [TestMethod]
        public void DecimalStd_GrossDiff_NotExcused()
        {
            var c = Case("std", ("decimal", new long[] { 8 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value, null, null, NPTypeCode.Decimal,
                new[] { new BitDiff.Diff(0, "100", "101") }).Should().BeNull(
                "a 1% decimal std error is an iteration/accumulation bug, not sqrt rounding");
        }

        [TestMethod]
        public void DecimalStd_LastDigit_StillExcused()
        {
            var c = Case("std", ("decimal", new long[] { 8 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value, null, null, NPTypeCode.Decimal,
                new[] { new BitDiff.Diff(0, "3278.9071286096267380354468786", "3278.9071286096267380354468787") })
                .Should().NotBeNull("one unit in the 28th significant digit is the documented sqrt envelope");
        }

        // ---- B8: the unary ~ULP excuse no longer covers float32 exp ----
        // NDFloatMath.Exp ports NumPy's simd_exp_FLOAT and is bit-exact over all 2^32 float32
        // inputs, so a 1-ULP exp/f32 drift is a regression rather than "a libm difference". These
        // two tests are the ratchet: the first fails if someone re-widens the excuse, the second
        // fails if the carve-out is widened past exp's float32 loop.

        private static byte[] F32(float v) => BitConverter.GetBytes(v);

        private static float UlpUpF(float v, int n = 1)
            => BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(v) + n);

        [TestMethod]
        public void B8_ExpFloat32_FromInt16Input_NotExcused()
        {
            // int16 -> float32 exp runs NumPy's SAME 'f->f' kernel, so it is held bit-exact too.
            var c = Case("exp", ("int16", new long[] { 4 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                F32(2.7182817f), F32(UlpUpF(2.7182817f)), NPTypeCode.Single, OneDiff).Should().BeNull(
                "the narrow-integer exp loop is the float32 kernel — same bit-exact claim");
        }

        [TestMethod]
        public void B8_ExpFloat64_OneUlp_StillExcused()
        {
            var c = Case("exp", ("float64", new long[] { 4 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(Math.E), BitConverter.GetBytes(UlpUp(Math.E)), NPTypeCode.Double, OneDiff)
                .Should().NotBeNull("float64 exp is still the platform libm vs NumPy's npy_exp");
        }

        [TestMethod]
        [DataRow("exp")]
        [DataRow("log")]
        [DataRow("sin")]
        [DataRow("cos")]
        [DataRow("rad2deg")]
        [DataRow("deg2rad")]
        [DataRow("tanh")]
        public void B8_PortedFloat32Kernels_OneUlp_NotExcused(string op)
        {
            var c = Case(op, ("float32", new long[] { 4 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                F32(0.84147096f), F32(UlpUpF(0.84147096f)), NPTypeCode.Single, OneDiff).Should().BeNull(
                $"float32 {op} is a bit-exact port of NumPy's own kernel — 1 ULP is a regression");
        }

        /// <summary>
        /// tanh is the only one of these ports that also replaces NumPy's FLOAT64 loop, so it is the
        /// only op whose f8 divergences must also stop being excused. The companion test below pins
        /// the other direction — that f8 exp/log/sin/cos, which are still the platform libm, keep it.
        /// </summary>
        [TestMethod]
        public void B8_PortedFloat64Tanh_OneUlp_NotExcused()
        {
            var c = Case("tanh", ("float64", new long[] { 4 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.7615941559557649), BitConverter.GetBytes(UlpUp(0.7615941559557649)), NPTypeCode.Double, OneDiff)
                .Should().BeNull("float64 tanh is a bit-exact port of NumPy's own kernel — 1 ULP is a regression");
        }

        [TestMethod]
        public void B8_UnportedUnaryFloat64_OneUlp_StillExcused()
        {
            // At float64 the platform libm already agrees with NumPy for these, so there is no port
            // and the ~ULP envelope still applies. tanh is deliberately absent — see above.
            foreach (var op in new[] { "exp", "log", "sin", "cos", "expm1", "log1p", "exp2" })
            {
                var c = Case(op, ("float64", new long[] { 4 }));
                MisalignedRegistry.Classify(c, DivergenceKind.Value,
                    BitConverter.GetBytes(0.7615941559557649), BitConverter.GetBytes(UlpUp(0.7615941559557649)), NPTypeCode.Double, OneDiff)
                    .Should().NotBeNull($"float64 {op} is not ported — it keeps the ~ULP envelope");
            }
        }

        [TestMethod]
        public void B8_UnportedUnaryFloat32_OneUlp_StillExcused()
        {
            // expm1/log1p/exp2 are still the platform libm at float32, so they keep the documented
            // envelope. If one of them is ported later, it moves into the carve-out set and this
            // test must move with it — which is exactly what tanh just did.
            foreach (var op in new[] { "expm1", "log1p", "exp2", "arctan" })
            {
                var c = Case(op, ("float32", new long[] { 4 }));
                MisalignedRegistry.Classify(c, DivergenceKind.Value,
                    F32(0.84147096f), F32(UlpUpF(0.84147096f)), NPTypeCode.Single, OneDiff)
                    .Should().NotBeNull($"{op} is not ported — it keeps the ~ULP envelope");
            }
        }

        // ---- P: truthful-vs-precise (the precision tier's truth-bearing adjudication) ----
        // Precise-doesn't-fail is STRUCTURAL (Classify is only reached after a NumPy byte
        // mismatch), so these pin the divergent side: not-less-truthful is excused, a gross
        // loss is not, and the P3 known-loss scope is bounded and layout/op-scoped.

        [TestMethod]
        public void P_TowardTruth_Excused()
        {
            // NumSharp lands ON truth, NumPy 2 ULP off -> excused as parity debt (toward truth).
            var c = Case("sum", ("float64", new long[] { 8 }));
            var truth = BitConverter.GetBytes(0.3);
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(UlpUp(0.3, 2)), BitConverter.GetBytes(0.3), NPTypeCode.Double,
                OneDiff, truth).Should().NotBeNull(
                "diverging from NumPy toward the correctly-rounded truth is prefer-precise parity debt");
        }

        [TestMethod]
        public void P_EquallyTruthful_Excused()
        {
            // Both sides 2 ULP from truth, opposite directions -> within truth-equivalence slack.
            var c = Case("sum", ("float64", new long[] { 8 }));
            var truth = BitConverter.GetBytes(0.3);
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(UlpUp(0.3, 2)), BitConverter.GetBytes(UlpUp(0.3, -2)), NPTypeCode.Double,
                OneDiff, truth).Should().NotBeNull("neither side is less accurate — excused, logged");
        }

        [TestMethod]
        public void P_GrossLoss_NotExcused()
        {
            // NumPy exact, NumSharp 100000 ULP off truth, outside every known-loss scope ->
            // a genuine precision loss must be RED (the truth==null gate on the unbounded
            // reduction blanket is what keeps it from hiding there).
            var c = Case("sum", ("float64", new long[] { 8 }));
            var truth = BitConverter.GetBytes(0.3);
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.3), BitConverter.GetBytes(UlpUp(0.3, 100000)), NPTypeCode.Double,
                OneDiff, truth).Should().BeNull(
                "less truthful than NumPy beyond slack is precision loss, not summation-order noise");
        }

        [TestMethod]
        public void P_KnownLoss_NegstrideSum_Excused_ButBounded()
        {
            // The documented negstride accumulation loss (measured 32 ULP) is excused ≤256...
            var c = Case("sum", ("float64", new long[] { 8 }));
            c.Layout = "negstride_1d";
            var truth = BitConverter.GetBytes(0.3);
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.3), BitConverter.GetBytes(UlpUp(0.3, 32)), NPTypeCode.Double,
                OneDiff, truth).Should().NotBeNull("the measured negstride loss is a tracked known bug");

            // ...and STAYS bounded: 1000 ULP on the same cell is a worse regression -> red.
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.3), BitConverter.GetBytes(UlpUp(0.3, 1000)), NPTypeCode.Double,
                OneDiff, truth).Should().BeNull("the known-loss excuse is bounded at 256 ULP-vs-truth");
        }

        [TestMethod]
        public void P_TruthlessCase_KeepsBlanketExcuse()
        {
            // Without truth (every pre-existing tier) the float summation blanket still applies —
            // the truth==null gate narrows it only where adjudication is possible.
            var c = Case("sum", ("float64", new long[] { 8 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.3), BitConverter.GetBytes(UlpUp(0.3)), NPTypeCode.Double,
                OneDiff).Should().NotBeNull("truthless tiers keep the documented summation excuse");
        }
    }
}
