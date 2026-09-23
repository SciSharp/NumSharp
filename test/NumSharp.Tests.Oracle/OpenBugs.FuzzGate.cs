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

        /// <summary>
        /// The cancellation residue that complex multiply USED to leave (expected exactly 0, actual a tiny
        /// residue at the rounding scale of the dominant component 1e10) is no longer excused.
        /// </summary>
        /// <remarks>
        /// Complex multiply became bit-exact (NDComplexMath ports NumPy's fused <c>simd_cmul</c>,
        /// dbc0b3b3), and 074d4d08 deliberately removed its excuse so a residue fails the FuzzMatrix
        /// gate. This test used to assert the excuse still existed; it went red when the excuse was
        /// removed and was only noticed once CI reached the Oracle step. It now pins the tight contract.
        /// </remarks>
        [TestMethod]
        public void B2_ComplexMultiply_CancellationResidue_NoLongerExcused()
        {
            var c = Case("multiply", ("complex128", new long[] { 1 }), ("complex128", new long[] { 1 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(0.0, 1e10), C128(1e-6, 1e10), NPTypeCode.Complex, OneDiff).Should().BeNull(
                "complex multiply is bit-exact (dbc0b3b3); its excuse was retired in 074d4d08, so even a rounding-scale residue is a real regression");
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

        [TestMethod]
        public void Corrcoef_ComplexGrossDiff_NotExcused()
        {
            var c = Case("corrcoef", ("complex128", new long[] { 2, 3 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 0.0), C128(2.0, 0.0), NPTypeCode.Complex, OneDiff)
                .Should().BeNull("the corrcoef composition is bounded to inherited divide ULP noise");
        }

        [TestMethod]
        public void Corrcoef_ComplexOneUlp_StillExcused()
        {
            var c = Case("corrcoef", ("complex128", new long[] { 2, 3 }));
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                C128(1.0, 0.0), C128(UlpUp(1.0), 0.0), NPTypeCode.Complex, OneDiff)
                .Should().NotBeNull("complex corrcoef inherits the documented npy_cdivide rounding");
        }

        // ---- B3: cumprod dtype excuse only on the size<=1 fast path ----

        [TestMethod]
        public void B3_CumprodDtype_FullSize_NotExcused()
        {
            MisalignedRegistry.Classify(Case("cumprod", ("int16", new long[] { 5 })),
                DivergenceKind.Dtype, null, null, NPTypeCode.Int64, System.Array.Empty<BitDiff.Diff>())
                .Should().BeNull("a full-size cumprod NEP50 widening miss is a real bug");
        }

        /// <summary>
        /// The size-1 cumprod NEP50 widening miss is no longer excused; like the full-size case, a dtype
        /// divergence there is a real bug.
        /// </summary>
        /// <remarks>
        /// ReduceCumMul's size-≤1 and empty branches cast to the accumulating type (int16/int32 → int64,
        /// uint8/uint16 → uint64), so cumprod is bit-exact on dtype at n ≤ 1. 7fd70a3b removed the excuse so
        /// a regression turns the gate red. This test used to assert the excuse still existed; it now
        /// pins the removal.
        /// </remarks>
        [TestMethod]
        public void B3_CumprodDtype_SizeOne_NoLongerExcused()
        {
            MisalignedRegistry.Classify(Case("cumprod", ("int16", new long[] { 1 })),
                DivergenceKind.Dtype, null, null, NPTypeCode.Int64, System.Array.Empty<BitDiff.Diff>())
                .Should().BeNull("the size-<=1 fast path now widens (7fd70a3b), so a dtype miss there is a real regression");
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

        // ---- B6: isclose value diffs are NOT excused (real never was; complex128 FIXED in fbbda5f0) ----

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
        public void B6_IscloseComplexValueDiff_NotExcused()
        {
            MisalignedRegistry.Classify(
                Case("isclose", ("complex128", new long[] { 4 }), ("float64", new long[] { 4 })),
                DivergenceKind.Value, new byte[] { 0 }, new byte[] { 1 }, NPTypeCode.Boolean,
                new[] { new BitDiff.Diff(0, "00", "01") }).Should().BeNull(
                "isclose complex128 is now bit-exact (fixed in fbbda5f0); the excuse was removed");
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

        // ---- K1: the public nditer trace uses the same narrow layout carve as nditer_* ----

        [TestMethod]
        public void K1_NditerDirect_Transposed3d_StillExcused()
        {
            var c = Case("nditer", ("int32", new long[] { 2, 3, 4 }));
            c.Layout = "transposed_3d";
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(1), BitConverter.GetBytes(2), NPTypeCode.Int32, OneDiff)
                .Should().NotBeNull("the renamed public trace must retain the measured K1 carve");
        }

        [TestMethod]
        public void K1_NditerDirect_ContiguousNeighbour_NotExcused()
        {
            var c = Case("nditer", ("int32", new long[] { 2, 3, 4 }));
            c.Layout = "c_contiguous_3d";
            MisalignedRegistry.Classify(c, DivergenceKind.Value,
                BitConverter.GetBytes(1), BitConverter.GetBytes(2), NPTypeCode.Int32, OneDiff)
                .Should().BeNull("K1 is limited to the three measured layouts");
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

        // ---- H1: float32 sin/cos past NumPy's Cody-Waite limit is the HOST libm ----

        /// <summary>2^32: past both limits; macos-latest's Apple libm rounds its float32 sine 1 ULP from ucrtbase.</summary>
        private const float PastLimit = 4294967296f;

        /// <summary>
        /// A float32 sin/cos case in the given tier shape, carrying <paramref name="inputs"/> as its input
        /// buffer. With <paramref name="outTier"/> it is shaped like an out_where cell: op <c>out_unary</c>,
        /// the ufunc in <c>params.ufunc</c>, input + out operands.
        /// </summary>
        /// <param name="ufunc">The unary ufunc name the case exercises.</param>
        /// <param name="outTier">True for the out=/where= tier's <c>out_unary</c> shape, false for a plain unary op.</param>
        /// <param name="inputs">The float32 input values, written as the operand's little-endian hex buffer.</param>
        /// <returns>The synthetic case.</returns>
        private static FuzzCorpus.Case SinCosCase(string ufunc, bool outTier, params float[] inputs)
        {
            var bytes = inputs.SelectMany(BitConverter.GetBytes).ToArray();
            var input = new FuzzCorpus.Operand
            {
                Dtype = "float32", Shape = new long[] { inputs.Length }, Strides = new long[] { 1 },
                Offset = 0, BufferSize = inputs.Length, Buffer = Convert.ToHexString(bytes),
            };
            if (!outTier)
                return new FuzzCorpus.Case { Id = "selftest/" + ufunc, Op = ufunc, Layout = "selftest", Operands = new[] { input } };
            using var doc = System.Text.Json.JsonDocument.Parse($"{{\"ufunc\":\"{ufunc}\"}}");
            return new FuzzCorpus.Case
            {
                Id = "selftest/out_unary/" + ufunc, Op = "out_unary", Layout = "out_c",
                Params = new System.Collections.Generic.Dictionary<string, System.Text.Json.JsonElement>
                    { ["ufunc"] = doc.RootElement.GetProperty("ufunc").Clone() },
                Operands = new[] { input, input },
            };
        }

        /// <summary>The float32 bits <paramref name="ulps"/> representable steps away from <paramref name="v"/>.</summary>
        /// <param name="v">The starting value.</param>
        /// <param name="ulps">How many ULPs to move (may be negative).</param>
        /// <returns>The little-endian bytes of the moved value.</returns>
        private static byte[] F32(float v, int ulps = 0)
            => BitConverter.GetBytes(BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(v) + ulps));

        /// <summary>
        /// The macOS shape of the divergence: NumSharp's element is exactly this host's libm answer for a
        /// past-limit input, and the reference is 1 ULP away. It is excused off the reference host and
        /// NEVER on it, for both the plain op and the out_unary tier and for sin and cos alike.
        /// <see cref="MisalignedRegistry.Classify"/> is pinned to follow whichever host this is.
        /// </summary>
        [TestMethod]
        public void H1_PastLimitHostLibmAnswer_OneUlp_ExcusedOnlyOffTheReferenceHost()
        {
            foreach (var ufunc in new[] { "sin", "cos" })
            foreach (var outTier in new[] { false, true })
            {
                var c = SinCosCase(ufunc, outTier, PastLimit, .5f);
                float host = ufunc == "sin" ? MathF.Sin(PastLimit) : MathF.Cos(PastLimit);
                var actual = F32(host).Concat(F32(.5f)).ToArray();
                var expected = F32(host, -1).Concat(F32(.5f)).ToArray();
                var diffs = new[] { new BitDiff.Diff(0, "ref", "host") };

                MisalignedRegistry.IsHostLibmSinCosHandoff(c, DivergenceKind.Value, expected, actual,
                    NPTypeCode.Single, diffs, libmReferenceHost: false).Should().BeTrue(
                    $"{ufunc} (out tier {outTier}): a past-limit lane is the host libm on both sides");
                MisalignedRegistry.IsHostLibmSinCosHandoff(c, DivergenceKind.Value, expected, actual,
                    NPTypeCode.Single, diffs, libmReferenceHost: true).Should().BeFalse(
                    "the reference host reproduces ucrtbase exactly, so it must stay strict");
                (MisalignedRegistry.Classify(c, DivergenceKind.Value, expected, actual, NPTypeCode.Single, diffs) != null)
                    .Should().Be(!MisalignedRegistry.IsLibmReferenceHost,
                    "Classify applies H1 exactly when this is not the libm reference host");
            }
        }

        /// <summary>
        /// The negative space. An in-range lane (the bit-exact port) is not excused off the reference host
        /// either, even when a past-limit input sits in the same case. Neither is a gross miss on a
        /// past-limit lane, a value that is not this host's libm answer, another op, or a non-float32 slot.
        /// </summary>
        [TestMethod]
        public void H1_InRangeLane_GrossMiss_OtherOpOrDtype_NotExcused()
        {
            var c = SinCosCase("sin", true, PastLimit, .5f);
            float host = MathF.Sin(PastLimit);
            float port = NumSharp.Utilities.NDFloatMath.Sin(.5f);

            // An in-range lane 1 ULP off: the port regressed. Its value is no libm answer for 2^32.
            MisalignedRegistry.IsHostLibmSinCosHandoff(c, DivergenceKind.Value,
                F32(host).Concat(F32(port)).ToArray(), F32(host).Concat(F32(port, 1)).ToArray(), NPTypeCode.Single,
                new[] { new BitDiff.Diff(1, "port", "port+1") }, libmReferenceHost: false)
                .Should().BeFalse("the in-range sin port is bit-exact on every host");

            // The same 1-ULP in-range miss with NO past-limit input in the case at all.
            MisalignedRegistry.IsHostLibmSinCosHandoff(SinCosCase("sin", true, .5f), DivergenceKind.Value,
                F32(port), F32(port, 1), NPTypeCode.Single, new[] { new BitDiff.Diff(0, "port", "port+1") },
                libmReferenceHost: false).Should().BeFalse("no lane of this case reaches the libm handoff");

            // A past-limit lane whose reference is 64 ULP away: beyond any libm rounding difference.
            MisalignedRegistry.IsHostLibmSinCosHandoff(c, DivergenceKind.Value,
                F32(host, -64).Concat(F32(port)).ToArray(), F32(host).Concat(F32(port)).ToArray(), NPTypeCode.Single,
                new[] { new BitDiff.Diff(0, "ref", "host") }, libmReferenceHost: false)
                .Should().BeFalse("a gross miss is not a libm rounding difference");

            // NumSharp's element is NOT this host's libm answer, even though it is within 1 ULP of the reference.
            MisalignedRegistry.IsHostLibmSinCosHandoff(c, DivergenceKind.Value,
                F32(host, 2).Concat(F32(port)).ToArray(), F32(host, 1).Concat(F32(port)).ToArray(), NPTypeCode.Single,
                new[] { new BitDiff.Diff(0, "ref", "other") }, libmReferenceHost: false)
                .Should().BeFalse("the kernel returns exactly MathF.Sin past the limit; anything else is a bug");

            // Other ops and dtypes: tan is libm everywhere and has its own excuse, float64 sin is not the port.
            MisalignedRegistry.IsHostLibmSinCosHandoff(SinCosCase("tan", true, PastLimit), DivergenceKind.Value,
                F32(MathF.Tan(PastLimit), -1), F32(MathF.Tan(PastLimit)), NPTypeCode.Single,
                new[] { new BitDiff.Diff(0, "ref", "host") }, libmReferenceHost: false).Should().BeFalse();
            MisalignedRegistry.IsHostLibmSinCosHandoff(c, DivergenceKind.Value,
                BitConverter.GetBytes(0.25), BitConverter.GetBytes(UlpUp(0.25)), NPTypeCode.Double,
                OneDiff, libmReferenceHost: false).Should().BeFalse();
        }
    }
}
