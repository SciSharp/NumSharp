using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.sinc — the normalized sinc function <c>sin(pi*x)/(pi*x)</c>, with the removable singularity
    ///     at every zero of <c>pi*x</c> filled by its limit value 1. All expected values are probed from
    ///     NumPy 2.4.2 (win-amd64).
    ///     <para>
    ///     Result dtype follows <c>pi*x</c> (NEP 50 weak-float): bool / every integer width / Char all
    ///     promote to float64; float16/float32/float64 are preserved; complex → complex128. Every REAL
    ///     dtype is BIT-EXACT with NumPy (gated bit-for-bit by the differential-fuzz "sinc" tier on the
    ///     win-amd64 CRT). COMPLEX128 is the one dtype that is NOT byte-reproducible — sinc composes
    ///     sin ∘ divide and the complex-sin ULP envelope amplifies through the division to tens of ULP —
    ///     so it is excluded from the byte corpus and pinned here by an <c>allclose</c> assertion instead.
    ///     </para>
    /// </summary>
    [TestClass]
    public class NpSincTest
    {
        // A tight tolerance for the host-libm real values (the fuzz tier pins them bit-exact on win-amd64;
        // here we assert value-correctness portably). Complex uses a looser bound for its ULP envelope.
        private const double Tol = 1e-13;

        /// <summary>sinc(0) is the limit value 1 — EXACTLY 1.0 for every float width (sin(eps)/eps rounds to 1).</summary>
        [TestMethod]
        public void Sinc_ZeroIsExactlyOne()
        {
            np.sinc(np.array(new[] { 0.0 })).GetAtIndex<double>(0).Should().Be(1.0);
            np.sinc(np.array(new[] { 0.0f })).GetAtIndex<float>(0).Should().Be(1.0f);
            ((double)(Half)np.sinc(np.array(new[] { (Half)0 })).GetValue(0)).Should().Be(1.0);
            // -0.0 also takes the limit (pi*-0.0 == 0 is falsy -> replaced by eps).
            np.sinc(np.array(new[] { -0.0 })).GetAtIndex<double>(0).Should().Be(1.0);
        }

        /// <summary>
        /// Basic float64 values vs NumPy 2.4.2 (probed on win-amd64): bit-exact on the pinned CRT host,
        /// within the <see cref="HostLibm"/> ULP budget elsewhere. sinc runs through libm <c>sin</c>, and glibc
        /// / Apple libm round sinc(0.25)'s <c>sin</c> one ULP differently (…061 vs …062), which is also why the
        /// <c>sinc</c> corpus tier is host-pinned.
        /// </summary>
        [TestMethod]
        public void Sinc_BasicValues_Float64()
        {
            var r = np.sinc(np.array(new[] { 0.5, 0.25, 0.1, 1.0, 2.0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            var expected = new[]
            {
                0.6366197723675814,        // sinc(1/2) = 2/pi
                0.9003163161571062,
                0.983631643083466,
                // sinc(integer != 0) is a tiny residual (sin(pi*k) is not exactly 0 in float), NOT literally 0.
                3.8981718325193755e-17,
                -3.8981718325193755e-17,
            };
            for (int i = 0; i < expected.Length; i++)
            {
                double got = r.GetAtIndex<double>(i);
                HostLibm.Matches(expected[i], got).Should().BeTrue(HostLibm.Describe($"sinc element {i}", expected[i], got));
            }
        }

        /// <summary>The docstring example — first values of sinc(linspace(-4, 4, 41)) match NumPy.</summary>
        [TestMethod]
        public void Sinc_Docstring_Linspace()
        {
            var r = np.sinc(np.linspace(-4.0, 4.0, 41));
            r.size.Should().Be(41);
            r.GetAtIndex<double>(1).Should().BeApproximately(-4.92362781e-02, 1e-9);
            r.GetAtIndex<double>(2).Should().BeApproximately(-8.40918587e-02, 1e-9);
            r.GetAtIndex<double>(3).Should().BeApproximately(-8.90384387e-02, 1e-9);
            // The centre (x == 0) is exactly the limit 1.
            r.GetAtIndex<double>(20).Should().Be(1.0);
        }

        /// <summary>sinc is an EVEN function: sinc(-x) == sinc(x) bit-for-bit (both go through |.|-symmetric sin/pi*x).</summary>
        [TestMethod]
        public void Sinc_IsEvenFunction()
        {
            var xs = new[] { 0.3, 1.7, 2.9, 0.05, 3.33 };
            var pos = np.sinc(np.array(xs));
            var neg = np.sinc(np.array(new[] { -0.3, -1.7, -2.9, -0.05, -3.33 }));
            for (int i = 0; i < xs.Length; i++)
                neg.GetAtIndex<double>(i).Should().Be(pos.GetAtIndex<double>(i), $"even at index {i}");
        }

        /// <summary>Integer input promotes to float64 (pi*x weak-float rule), NOT np.sin's i8->f16 tier.</summary>
        [TestMethod]
        public void Sinc_IntPromotesToFloat64()
        {
            var r = np.sinc(np.array(new[] { 0, 1, 2, 3, 4 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.dtype.Should().Be(np.float64);
            r.GetAtIndex<double>(0).Should().Be(1.0);
            r.GetAtIndex<double>(1).Should().Be(3.8981718325193755e-17);
            r.GetAtIndex<double>(2).Should().Be(-3.8981718325193755e-17);
        }

        /// <summary>int8/uint64 (narrow + widest unsigned) also go to float64 — the promotion is width-independent.</summary>
        [TestMethod]
        public void Sinc_AllIntWidths_ToFloat64()
        {
            np.sinc(np.array(new sbyte[] { 0, 1, 2 })).typecode.Should().Be(NPTypeCode.Double);
            np.sinc(np.array(new ulong[] { 0, 1, 2 })).typecode.Should().Be(NPTypeCode.Double);
        }

        /// <summary>Boolean input promotes to float64 (True -> the sinc(1) residual, False -> 1.0).</summary>
        [TestMethod]
        public void Sinc_BoolPromotesToFloat64()
        {
            var r = np.sinc(np.array(new[] { false, true }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(1.0);
            r.GetAtIndex<double>(1).Should().Be(3.8981718325193755e-17);
        }

        /// <summary>Char (NumSharp's uint16-family dtype) promotes to float64 like the other integers.</summary>
        [TestMethod]
        public void Sinc_CharPromotesToFloat64()
        {
            var r = np.sinc(np.array(new char[] { (char)0, (char)1, (char)2 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(1.0);
        }

        /// <summary>float16 and float32 inputs are PRESERVED (not widened) — pi is a weak scalar.</summary>
        [TestMethod]
        public void Sinc_FloatWidthsPreserved()
        {
            np.sinc(np.array(new[] { (Half)0.5, (Half)1 })).typecode.Should().Be(NPTypeCode.Half);
            np.sinc(np.array(new[] { 0.5f, 1f })).typecode.Should().Be(NPTypeCode.Single);
            np.sinc(np.array(new[] { 0.5, 1.0 })).typecode.Should().Be(NPTypeCode.Double);
        }

        /// <summary>inf/-inf/NaN all yield NaN (sin(inf)=NaN, NaN/anything=NaN); the value is NaN regardless of sign.</summary>
        [TestMethod]
        public void Sinc_InfAndNan_GiveNan()
        {
            var r = np.sinc(np.array(new[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN }));
            double.IsNaN(r.GetAtIndex<double>(0)).Should().BeTrue();
            double.IsNaN(r.GetAtIndex<double>(1)).Should().BeTrue();
            double.IsNaN(r.GetAtIndex<double>(2)).Should().BeTrue();
        }

        /// <summary>A 0-d (scalar) input yields a 0-d result (NumPy returns a scalar), NOT a 1-D array.</summary>
        [TestMethod]
        public void Sinc_ScalarInput_ZeroDim()
        {
            var r = np.sinc(NDArray.Scalar(0.5));
            r.ndim.Should().Be(0);
            r.GetAtIndex<double>(0).Should().Be(0.6366197723675814);
        }

        /// <summary>An empty input yields an empty float64 array (shape preserved, no crash).</summary>
        [TestMethod]
        public void Sinc_Empty_Float64Empty()
        {
            var r = np.sinc(np.array(new double[] { }));
            r.size.Should().Be(0);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        /// <summary>
        ///     Non-contiguous inputs (negative-stride reversed view, transposed 2-D view) are read through
        ///     their own strides and produce the same values as the contiguous form — sinc handles every layout.
        /// </summary>
        [TestMethod]
        public void Sinc_NonContiguousLayouts()
        {
            // reversed (negative-stride) view
            var contig = np.sinc(np.array(new[] { 0.0, 0.25, 0.5, 1.0 }));
            var rev = np.sinc(np.array(new[] { 0.0, 0.25, 0.5, 1.0 })["::-1"]);
            for (int i = 0; i < 4; i++)
                rev.GetAtIndex<double>(i).Should().Be(contig.GetAtIndex<double>(3 - i), $"reversed elem {i}");

            // transposed 2-D view
            var m = np.arange(6.0).reshape(2, 3).T;   // (3,2) transposed
            var rm = np.sinc(m);
            rm.shape.Should().Equal(3, 2);
            rm.typecode.Should().Be(NPTypeCode.Double);
        }

        /// <summary>
        ///     COMPLEX128 pin (excluded from the byte-exact fuzz corpus because the per-component BITS are
        ///     not reproducible — sinc composes sin∘divide and NumSharp's complex sin differs from NumPy's
        ///     UCRT csin within its ≤3-ULP-relative envelope). The complex VALUE is nonetheless accurate:
        ///     validated ≤~2.5 ULP RELATIVE (max 5.65e-16 relative error) and np.allclose(rtol=1e-5,
        ///     atol=1e-8) passing on ALL 200,000 random samples vs NumPy 2.4.2 — so it is pinned here with
        ///     the same magnitude-based closeness NumPy's own allclose uses, not a per-component bit compare.
        /// </summary>
        [TestMethod]
        public void Sinc_Complex_AllcloseToNumpy()
        {
            var r = np.sinc(np.array(new[] { new Complex(0, 0), new Complex(0.5, 0.5),
                                             new Complex(1, 2), new Complex(0, 1) }));
            r.typecode.Should().Be(NPTypeCode.Complex);

            // sinc(0) is exactly the limit 1 + 0j.
            var z0 = (Complex)r.GetValue(0);
            z0.Real.Should().Be(1.0);
            z0.Imaginary.Should().Be(0.0);

            // Interior points: numpy-allclose (magnitude of the complex error vs atol + rtol*|expected|).
            AssertClose((Complex)r.GetValue(1), new Complex(0.7986963159564631, -0.7986963159564631));
            AssertClose((Complex)r.GetValue(2), new Complex(-34.09033869939481, -17.04516934969741));
            AssertClose((Complex)r.GetValue(3), new Complex(3.676077910374978, 0.0));
        }

        /// <summary>A null input is rejected up front (NumPy would fail inside asanyarray(None)).</summary>
        [TestMethod]
        public void Sinc_Null_Throws()
        {
            Action act = () => np.sinc((NDArray)null);
            act.Should().Throw<ArgumentNullException>();
        }

        // Complex closeness helper — NumPy's allclose semantics on the COMPLEX magnitude of the error
        // (|got - expected| <= atol + rtol*|expected|), NOT a per-component compare: when one component is
        // tiny relative to |z| its raw-bit ULP can be large while the complex value stays ~2.5 ULP accurate,
        // so the magnitude form is the correct closeness test (matches np.allclose, which passed on all 200K).
        private static void AssertClose(Complex got, Complex expected)
        {
            double atol = 1e-8, rtol = 1e-5;   // NumPy allclose defaults
            (got - expected).Magnitude
                .Should().BeLessThanOrEqualTo(atol + rtol * expected.Magnitude, "|complex error|");
        }
    }
}
