using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Tests for the <c>np.emath</c> (<c>numpy.lib.scimath</c>) automatic-domain math module. Every
    ///     expected value is probed against NumPy 2.4.2. These are the branch-cut-aware siblings of the
    ///     ordinary ufuncs: where <see cref="np.sqrt(NDArray)"/> returns <c>nan</c> for a negative real,
    ///     <see cref="np.emath"/>.<c>sqrt</c> returns the complex value. The module is a pure composition
    ///     over the existing (already-fuzzed) <c>np.*</c> ufuncs plus a domain scan, so this suite is its
    ///     gate — the emath-specific behavior being the whole-array real→complex promotion decision.
    ///     <para>
    ///     Two documented divergences are pinned as <see cref="MisalignedAttribute"/>: F1 (NumSharp has no
    ///     complex64, so a triggering float32/small-int input yields complex128 double precision rather than
    ///     NumPy's complex64 single precision) and F5 (complex <c>power</c> with a non-integer exponent is
    ///     <c>allclose</c>, not bit-exact — the <c>Complex.Pow</c> vs <c>npy_cpow</c> ULP divergence).
    ///     </para>
    /// </summary>
    [TestClass]
    public class np_emath_Test
    {
        private const double Pi = System.Math.PI;

        /// <summary>
        /// Assert a real 1-D (or 0-d) result reproduces the NumPy-probed doubles: bit-for-bit on the pinned
        /// win-amd64 CRT host, within <see cref="HostLibm.OffHostUlpBudget"/> ULP elsewhere. The emath values
        /// run through libm (acos, atanh, …), and glibc / Apple libm round some inputs one ULP differently
        /// from <c>ucrtbase</c>, which is why the matching <c>emath</c> corpus tier is host-pinned too.
        /// </summary>
        /// <param name="r">The result under test (must be float64).</param>
        /// <param name="expected">The NumPy 2.4.2 values, in C order.</param>
        private static void AssertReal(NDArray r, params double[] expected)
        {
            Assert.AreEqual(NPTypeCode.Double, r.typecode, "expected a float64 result");
            var got = r.Data<double>();
            Assert.AreEqual(expected.Length, got.Count, "length");
            for (int i = 0; i < expected.Length; i++)
                Assert.IsTrue(HostLibm.Matches(expected[i], got[i]), HostLibm.Describe($"element {i}", expected[i], got[i]));
        }

        /// <summary>
        /// Assert a complex128 1-D result reproduces the NumPy-probed values per component: bit-for-bit on
        /// the pinned win-amd64 CRT host, within the <see cref="HostLibm"/> budget elsewhere (the sign bit —
        /// including a signed zero — stays strict on every host).
        /// </summary>
        /// <param name="r">The result under test (must be complex128).</param>
        /// <param name="expected">The NumPy 2.4.2 values, in C order.</param>
        private static void AssertComplexExact(NDArray r, params Complex[] expected)
        {
            Assert.AreEqual(NPTypeCode.Complex, r.typecode, "expected a complex128 result");
            var got = r.Data<Complex>();
            Assert.AreEqual(expected.Length, got.Count, "length");
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.IsTrue(HostLibm.Matches(expected[i], got[i]),
                    HostLibm.Describe($"element {i} real", expected[i].Real, got[i].Real) + "; " +
                    HostLibm.Describe($"element {i} imag", expected[i].Imaginary, got[i].Imaginary));
            }
        }

        /// <summary>Assert a complex128 result is close to expected (for the F5 complex-power divergence).</summary>
        private static void AssertComplexClose(NDArray r, double rtol, params Complex[] expected)
        {
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            var got = r.Data<Complex>();
            Assert.AreEqual(expected.Length, got.Count);
            for (int i = 0; i < expected.Length; i++)
                Assert.IsTrue(Complex.Abs(got[i] - expected[i]) <= rtol * (1 + Complex.Abs(expected[i])),
                    $"element {i}: got {got[i]} expected {expected[i]}");
        }

        // ─────────────────────────────── sqrt ───────────────────────────────

        [TestMethod]
        public void sqrt_RealNonNegative_StaysReal()
        {
            // Non-triggering: identical to np.sqrt; int promotes to float64.
            AssertReal(np.emath.sqrt(np.array(new[] { 1, 4, 9, 16 })), 1.0, 2.0, 3.0, 4.0);
        }

        [TestMethod]
        public void sqrt_RealNegative_PromotesToComplex()
        {
            // Any negative real element promotes the WHOLE array to complex128.
            AssertComplexExact(np.emath.sqrt(np.array(new[] { -1.0, 4.0 })), new Complex(0, 1), new Complex(2, 0));
        }

        [TestMethod]
        public void sqrt_Scalar_NegativeIsComplex_PositiveIsReal()
        {
            // A scalar input yields a 0-D array. sqrt(-1) == 1j; sqrt(1) == 1.0.
            var neg = np.emath.sqrt((NDArray)(-1.0));
            Assert.AreEqual(NPTypeCode.Complex, neg.typecode);
            Assert.AreEqual(new Complex(0, 1), neg.Data<Complex>()[0]);
            var pos = np.emath.sqrt((NDArray)1.0);
            Assert.AreEqual(NPTypeCode.Double, pos.typecode);
            Assert.AreEqual(1.0, pos.Data<double>()[0]);
        }

        [TestMethod]
        public void sqrt_NegativeZero_DoesNotTrigger()
        {
            // -0.0 < 0 is false, so -0.0 stays real; np.sqrt(-0.0) == -0.0 (IEEE, sign preserved).
            var r = np.emath.sqrt(np.array(new[] { -0.0, 0.5 }));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(unchecked((long)0x8000000000000000UL), BitConverter.DoubleToInt64Bits(r.Data<double>()[0]));
        }

        [TestMethod]
        public void sqrt_NaN_DoesNotTrigger_ButNaNNegElsewhereDoes()
        {
            // nan < 0 is false: [nan, 4] stays real. But [nan, -1] has a negative → complex.
            AssertReal(np.emath.sqrt(np.array(new[] { double.NaN, 4.0 })), double.NaN, 2.0);
            var r = np.emath.sqrt(np.array(new[] { double.NaN, -1.0 }));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            Assert.AreEqual(new Complex(0, 1), r.Data<Complex>()[1]);
        }

        [TestMethod]
        public void sqrt_Bool_StaysReal_Float16()
        {
            // bool never negative → real np.sqrt(bool) → float16 [1, 0].
            var r = np.emath.sqrt(np.array(new[] { true, false, true }));
            Assert.AreEqual(NPTypeCode.Half, r.typecode);
            CollectionAssert.AreEqual(new[] { (Half)1, (Half)0, (Half)1 }, r.Data<Half>().ToArray());
        }

        [TestMethod]
        public void sqrt_Empty_StaysReal()
        {
            var r = np.emath.sqrt(np.array(new double[0]));
            Assert.AreEqual(NPTypeCode.Double, r.typecode);
            Assert.AreEqual(0, r.size);
        }

        [TestMethod]
        public void sqrt_2D_PromotesWholeArray()
        {
            var r = np.emath.sqrt(np.array(new[] { -1.0, 4.0, 9.0, 16.0 }).reshape(2, 2));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            CollectionAssert.AreEqual(new[] { 2L, 2L }, r.shape.Select(x => (long)x).ToArray());
            AssertComplexExact(r.reshape(4), new Complex(0, 1), new Complex(2, 0), new Complex(3, 0), new Complex(4, 0));
        }

        [TestMethod]
        public void sqrt_ComplexInput_AppliesComplexUfunc()
        {
            var r = np.emath.sqrt(np.array(new[] { new Complex(1, 2), new Complex(-3, 0) }));
            AssertComplexExact(r, Complex.Sqrt(new Complex(1, 2)), Complex.Sqrt(new Complex(-3, 0)));
        }

        // ─────────────────────────────── log family ───────────────────────────────

        [TestMethod]
        public void log_NegativeReal_ComplexPrincipalValue()
        {
            // log(-e) == 1 + pi·i.
            var r = np.emath.log((NDArray)(-System.Math.E));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            var z = r.Data<Complex>()[0];
            Assert.AreEqual(1.0, z.Real, 1e-15);
            Assert.AreEqual(Pi, z.Imaginary, 1e-15);
        }

        [TestMethod]
        public void log_Zero_IsNegInf_StaysReal()
        {
            AssertReal(np.emath.log(np.array(new[] { 0.0, System.Math.E })), double.NegativeInfinity, 1.0);
        }

        [TestMethod]
        public void log_NegativeZero_DoesNotTrigger()
        {
            // -0.0 does not trigger; log(-0.0) == -inf (real), log(4) real.
            AssertReal(np.emath.log(np.array(new[] { -0.0, 4.0 })), double.NegativeInfinity, System.Math.Log(4.0));
        }

        [TestMethod]
        public void log10_NegativeReals_Complex()
        {
            var r = np.emath.log10(np.array(new[] { -10.0, -100.0, 100.0 }));
            AssertComplexExact(r,
                new Complex(1.0, 1.3643763538418412),
                new Complex(2.0, 1.3643763538418412),
                new Complex(2.0, 0.0));
        }

        [TestMethod]
        public void log2_Positive_StaysReal()
        {
            AssertReal(np.emath.log2(np.array(new[] { 4.0, 8.0 })), 2.0, 3.0);
        }

        [TestMethod]
        public void log2_NegativeReals_Complex_BitExact()
        {
            // Probed 2.4.2: log2(-8) real part is 2.9999999999999996 (not 3.0), imag 4.532360141827193.
            var r = np.emath.log2(np.array(new[] { -4.0, -8.0, 8.0 }));
            AssertComplexExact(r,
                new Complex(2.0, 4.532360141827193),
                new Complex(2.9999999999999996, 4.532360141827193),
                new Complex(2.9999999999999996, 0.0));
        }

        // ─────────────────────────────── logn ───────────────────────────────

        [TestMethod]
        public void logn_PositiveArgs_Real()
        {
            AssertReal(np.emath.logn((NDArray)2.0, np.array(new[] { 4.0, 8.0 })), 2.0, 3.0);
        }

        [TestMethod]
        public void logn_NegativeX_Complex()
        {
            var r = np.emath.logn((NDArray)2.0, np.array(new[] { -4.0, -8.0, 8.0 }));
            AssertComplexExact(r,
                new Complex(2.0, 4.532360141827193),
                new Complex(2.9999999999999996, 4.532360141827193),
                new Complex(2.9999999999999996, 0.0));
        }

        [TestMethod]
        public void logn_NegativeBase_Complex()
        {
            // A negative BASE (n) also promotes: logn(-2, 8) = log(8) / log(-2+0j) is complex.
            var r = np.emath.logn((NDArray)(-2.0), np.array(new[] { 8.0 }));
            AssertComplexExact(r, new Complex(0.1392609706362244, -0.6311808726237906));
        }

        [TestMethod]
        public void logn_ArrayBase_Real()
        {
            // logn([2,10],[8,100]) = [log2(8), log10(100)] = [3, 2].
            AssertReal(np.emath.logn(np.array(new[] { 2.0, 10.0 }), np.array(new[] { 8.0, 100.0 })), 3.0, 2.0);
        }

        // ─────────────────────────────── power ───────────────────────────────

        [TestMethod]
        public void power_PositiveBase_Integer()
        {
            // [2,4]**2 stays integer (no negative base, no negative exponent).
            var r = np.emath.power(np.array(new[] { 2, 4 }), (NDArray)2);
            Assert.IsTrue(r.typecode == NPTypeCode.Int32 || r.typecode == NPTypeCode.Int64);
            CollectionAssert.AreEqual(new[] { 4L, 16L }, r.Data<int>().Select(x => (long)x).ToArray());
        }

        [TestMethod]
        public void power_NegativeExponent_PromotesExponentToFloat()
        {
            // A negative exponent promotes p to float (NumPy's _fix_int_lt_zero = p*1.0), so the integer
            // base does NOT hit np.power's "Integers to negative integer powers" error.
            AssertReal(np.emath.power(np.array(new[] { 2, 4 }), (NDArray)(-2)), 0.25, 0.0625);
        }

        [TestMethod]
        public void power_NegativeBase_IntegerExponent_ComplexBitExact()
        {
            // Negative base → complex; integer exponent → bit-exact complex power (4-0j, 16+0j).
            var r = np.emath.power(np.array(new[] { -2, 4 }), (NDArray)2);
            AssertComplexExact(r, new Complex(4, -0.0), new Complex(16, 0));
        }

        [TestMethod]
        public void power_ArrayExponent_Integer()
        {
            var r = np.emath.power(np.array(new[] { 2, 4 }), np.array(new[] { 2, 4 }));
            CollectionAssert.AreEqual(new[] { 4L, 256L }, r.Data<int>().Select(x => (long)x).ToArray());
        }

        [TestMethod]
        [Misaligned]
        public void power_NegativeBase_NonIntegerExponent_Allclose_F5()
        {
            // F5: complex power with a NON-integer exponent takes Complex.Pow, which is allclose to NumPy's
            // npy_cpow but not bit-exact. NumPy: power(-8, 1/3) ≈ (1 + 1.7320508075688772j).
            var r = np.emath.power((NDArray)(-8.0), (NDArray)(1.0 / 3.0));
            AssertComplexClose(r, 1e-12, new Complex(1.0, 1.7320508075688772));
        }

        // ─────────────────────────────── arccos / arcsin ───────────────────────────────

        /// <summary>
        /// arccos(0.5) = pi/3 stays a real scalar. Bit-exact on the pinned CRT host; Apple's libm returns
        /// the neighbouring double (…976 vs …979), so elsewhere it is held to the <see cref="HostLibm"/> budget.
        /// </summary>
        [TestMethod]
        public void arccos_InDomain_StaysReal()
        {
            var r = np.emath.arccos((NDArray)0.5);
            AssertReal(r, 1.0471975511965979);
        }

        [TestMethod]
        public void arccos_AbsGreaterThanOne_Complex_WithNegZeroImag()
        {
            // |2| > 1 promotes the whole array. arccos(1+0j) == -0j (real 0, imag -0.0) — sign matters.
            var r = np.emath.arccos(np.array(new[] { 1.0, 2.0 }));
            AssertComplexExact(r, new Complex(0.0, -0.0), new Complex(0.0, -1.3169578969248166));
            // Pin the -0.0 imaginary sign bit explicitly.
            Assert.AreEqual(unchecked((long)0x8000000000000000UL),
                BitConverter.DoubleToInt64Bits(r.Data<Complex>()[0].Imaginary));
        }

        [TestMethod]
        public void arccos_Boundary_AbsEqualsOne_StaysReal()
        {
            // The trigger is strictly |x| > 1, so |x| == 1 stays real: arccos([1,-1]) = [0, pi].
            AssertReal(np.emath.arccos(np.array(new[] { 1.0, -1.0 })), 0.0, Pi);
        }

        [TestMethod]
        public void arcsin_InDomain_StaysReal()
        {
            AssertReal(np.emath.arcsin(np.array(new[] { 0.0, 1.0 })), 0.0, 1.5707963267948966);
        }

        [TestMethod]
        public void arcsin_AbsGreaterThanOne_Complex()
        {
            var r = np.emath.arcsin(np.array(new[] { 0.0, 2.0 }));
            AssertComplexExact(r, new Complex(0.0, 0.0), new Complex(1.5707963267948966, 1.3169578969248166));
        }

        // ─────────────────────────────── arctanh ───────────────────────────────

        /// <summary>
        /// arctanh(0.5) stays a real scalar. Bit-exact on the pinned CRT host; glibc returns the neighbouring
        /// double (…548 vs …549), so elsewhere it is held to the <see cref="HostLibm"/> budget.
        /// </summary>
        [TestMethod]
        public void arctanh_InDomain_StaysReal()
        {
            var r = np.emath.arctanh((NDArray)0.5);
            AssertReal(r, 0.5493061443340549);
        }

        [TestMethod]
        public void arctanh_Boundary_One_StaysReal_IsInf()
        {
            // |1| > 1 is false, so x == 1 stays real and yields +inf (matching NumPy, RuntimeWarning aside).
            AssertReal(np.emath.arctanh((NDArray)1.0), double.PositiveInfinity);
        }

        [TestMethod]
        public void arctanh_AbsGreaterThanOne_Complex()
        {
            // arctanh([0.5, 2.0]): 2.0 triggers; 0.5→0.549+0j, 2.0→0.549+1.5708j.
            var r = np.emath.arctanh(np.array(new[] { 0.5, 2.0 }));
            AssertComplexExact(r,
                new Complex(0.5493061443340549, 0.0),
                new Complex(0.5493061443340549, 1.5707963267948966));
        }

        [TestMethod]
        public void arctanh_ComplexInput_AppliesComplexUfunc()
        {
            // arctanh(1j) == 0.7853981633974483j.
            var r = np.emath.arctanh(np.array(new[] { new Complex(0, 1) }));
            AssertComplexExact(r, new Complex(0.0, 0.7853981633974483));
        }

        // ─────────────────────────────── layout coverage ───────────────────────────────

        [TestMethod]
        public void NonContiguous_Layouts_MatchContiguous()
        {
            // The fused domain scan handles contiguous float; non-contiguous falls back to the composition,
            // which reads through strides. All must agree with NumPy — reversed, transposed and strided.
            var m = np.array(new[] { -1.0, 4.0, 9.0, -16.0, 25.0, -36.0 }).reshape(2, 3);

            // reversed rows: [[-16,25,-36],[-1,4,9]] → complex.
            var rev = np.emath.sqrt(m["::-1"]).reshape(6);
            AssertComplexExact(rev, new Complex(0, 4), new Complex(5, 0), new Complex(0, 6),
                                    new Complex(0, 1), new Complex(2, 0), new Complex(3, 0));

            // transposed (F-order view) → complex, same set of values in column order.
            Assert.AreEqual(NPTypeCode.Complex, np.emath.sqrt(m.T).typecode);

            // strided arccos, non-triggering (all |x|<=1) stays real and matches NumPy.
            var w = np.array(new[] { 0.1, 0.5, 0.9, -0.2, 0.3, -0.4 });
            AssertReal(np.emath.arccos(w["::-1"]),
                1.9823131728623846, 1.2661036727794992, 1.7721542475852274,
                0.45102681179626236, 1.0471975511965979, 1.4706289056333368);
        }

        // ─────────────────────────────── F1 (no complex64) ───────────────────────────────

        [TestMethod]
        [Misaligned]
        public void F1_Float32Trigger_YieldsComplex128_NotComplex64()
        {
            // NumSharp has no complex64: a triggering float32 input yields complex128 (double precision)
            // where NumPy yields complex64 (single precision). The VALUE is NumSharp's correct
            // double-precision answer (== NumPy computing the same input in float64 domain).
            var r = np.emath.sqrt(np.array(new[] { -1.0f, 4.0f }));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode); // NumPy would be complex64
            AssertComplexExact(r, new Complex(0, 1), new Complex(2, 0));
        }

        [TestMethod]
        [Misaligned]
        public void F1_UInt8Trigger_YieldsComplex128()
        {
            // uint8 is in NumPy's _tocomplex "→ csingle" set, so NumPy gives complex64; NumSharp complex128.
            // Values match NumPy's float64-domain arccos.
            var r = np.emath.arccos(np.array(new byte[] { 0, 1, 2, 3 }));
            Assert.AreEqual(NPTypeCode.Complex, r.typecode);
            AssertComplexExact(r,
                new Complex(1.5707963267948966, -0.0),
                new Complex(0.0, -0.0),
                new Complex(0.0, -1.3169578969248166),
                new Complex(0.0, -1.762747174039086));
        }
    }
}
