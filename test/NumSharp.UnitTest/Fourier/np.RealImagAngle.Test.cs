using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Fourier
{
    /// <summary>
    ///     np.real / np.imag / np.angle — the complex-number accessor set completed alongside the FFT
    ///     work (with np.conjugate they are the four basic complex accessors, and the standard post-FFT
    ///     spectrum extractors: for A = np.fft.fft(a), A.real / A.imag are the components and np.angle(A)
    ///     is the phase spectrum). All values/dtypes/semantics probed against NumPy 2.4.2 and verified
    ///     bit-exact.
    ///
    ///     Semantics (NumPy 2.4.2):
    ///     - real(complex): float64 VIEW onto the real lane (shares memory, writeable). complex128 -> float64.
    ///     - imag(complex): float64 VIEW onto the imaginary lane (shares memory, writeable).
    ///     - real(real): the input itself (dtype preserved). imag(real): a READ-ONLY zeros of same shape/dtype.
    ///     - angle(z) = arctan2(imag, real); deg=True multiplies by 180/pi at the result's own tier.
    ///       Real input tier: bool/int32+/float64 -> float64, int8/uint8/float16 -> float16,
    ///       int16/uint16/float32 -> float32.
    /// </summary>
    [TestClass]
    public class NpRealImagAngleTest : TestClass
    {
        private const double Pi = Math.PI;

        // =====================================================================================
        // np.real / np.imag — COMPLEX input: float64 lane VIEW (shared memory, writeable)
        // =====================================================================================

        [TestMethod]
        public void Real_Complex_ExtractsRealLane_Float64()
        {
            var z = new NDArray(new[] { new Complex(1, 2), new Complex(3, -4), new Complex(5, 6) });
            var r = np.real(z);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.shape.Should().BeEquivalentTo(new[] { 3 });
            r.Data<double>().Should().BeEquivalentTo(new[] { 1.0, 3.0, 5.0 });
        }

        [TestMethod]
        public void Imag_Complex_ExtractsImaginaryLane_Float64()
        {
            var z = new NDArray(new[] { new Complex(1, 2), new Complex(3, -4), new Complex(5, 6) });
            var im = np.imag(z);
            im.typecode.Should().Be(NPTypeCode.Double);
            im.Data<double>().Should().BeEquivalentTo(new[] { 2.0, -4.0, 6.0 });
        }

        [TestMethod]
        public void Real_Complex_IsAWriteableViewThatWritesThrough()
        {
            // np.real(z)[0] = 100 must modify z[0]'s real part (NumPy shares memory and is writeable).
            var z = new NDArray(new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6) });
            var r = np.real(z);
            r.Shape.IsWriteable.Should().BeTrue();
            r.SetData(100.0, 0);
            z.Data<Complex>()[0].Should().Be(new Complex(100, 2));
        }

        [TestMethod]
        public void Imag_Complex_IsAWriteableViewThatWritesThrough()
        {
            var z = new NDArray(new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6) });
            var im = np.imag(z);
            im.Shape.IsWriteable.Should().BeTrue();
            im.SetData(-9.0, 2);
            z.Data<Complex>()[2].Should().Be(new Complex(5, -9));
        }

        [TestMethod]
        public void RealImag_Complex_SignedZeroPreservedThroughView()
        {
            // real(-0-0j) is -0.0, imag(0-0j) is -0.0 — the lane view preserves the sign bit.
            var z = new NDArray(new[] { new Complex(-0.0, -0.0) });
            double.IsNegative(np.real(z).Data<double>()[0]).Should().BeTrue();
            double.IsNegative(np.imag(z).Data<double>()[0]).Should().BeTrue();
        }

        [TestMethod]
        public void RealImag_Complex_Transposed_ReadsThroughStrides()
        {
            var z = new NDArray(new[]
            {
                new Complex(1, 10), new Complex(2, 20), new Complex(3, 30),
                new Complex(4, 40), new Complex(5, 50), new Complex(6, 60)
            }).reshape(2, 3).transpose(); // (3,2), non-contiguous
            np.real(z).Data<double>().Should().BeEquivalentTo(new[] { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 });
            np.imag(z).Data<double>().Should().BeEquivalentTo(new[] { 10.0, 40.0, 20.0, 50.0, 30.0, 60.0 });
        }

        [TestMethod]
        public void RealImag_Complex_NegativeStride_ReadsReversed()
        {
            var z = new NDArray(new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8) })["::-1"];
            np.real(z).Data<double>().Should().BeEquivalentTo(new[] { 7.0, 5.0, 3.0, 1.0 });
            np.imag(z).Data<double>().Should().BeEquivalentTo(new[] { 8.0, 6.0, 4.0, 2.0 });
        }

        [TestMethod]
        public void RealImag_Complex_SlicedOffset_ReadsThroughOffset()
        {
            var z = new NDArray(new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8) })["1:3"];
            np.real(z).Data<double>().Should().BeEquivalentTo(new[] { 3.0, 5.0 });
            np.imag(z).Data<double>().Should().BeEquivalentTo(new[] { 4.0, 6.0 });
        }

        [TestMethod]
        public void RealImag_Complex_BroadcastView_IsReadOnly()
        {
            // real/imag of a read-only broadcast complex stays read-only (NumPy parity).
            var z = np.broadcast_to(new NDArray(new[] { new Complex(7, 8) }), new Shape(3));
            var r = np.real(z);
            r.Shape.IsWriteable.Should().BeFalse();
            r.Data<double>().Should().BeEquivalentTo(new[] { 7.0, 7.0, 7.0 });
            np.imag(z).Data<double>().Should().BeEquivalentTo(new[] { 8.0, 8.0, 8.0 });
        }

        [TestMethod]
        public void RealImag_Complex_ZeroD()
        {
            var z = NDArray.Scalar(new Complex(2, 3));
            np.real(z).GetDouble(0).Should().Be(2.0);
            np.imag(z).GetDouble(0).Should().Be(3.0);
        }

        [TestMethod]
        public void RealImag_Complex_Empty()
        {
            var z = np.array(new Complex[] { }).reshape(0);
            np.real(z).size.Should().Be(0);
            np.imag(z).size.Should().Be(0);
            np.real(z).typecode.Should().Be(NPTypeCode.Double);
        }

        // =====================================================================================
        // np.real / np.imag — REAL input: real is the input itself; imag is READ-ONLY zeros
        // =====================================================================================

        [TestMethod]
        public void Real_RealInput_ReturnsInputUnchanged_SameInstance()
        {
            var a = new NDArray(new[] { 1.0, 2.0, 3.0 });
            var r = np.real(a);
            ReferenceEquals(r, a).Should().BeTrue("np.real of a real array IS the array (matches NumPy `a.real is a`)");
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void Imag_RealInput_IsReadOnlyZeros_DtypePreserved()
        {
            var a = new NDArray(new[] { 1.0f, 2.0f, 3.0f });
            var im = np.imag(a);
            im.typecode.Should().Be(NPTypeCode.Single);          // dtype preserved
            im.Shape.IsWriteable.Should().BeFalse();              // NumPy: imag of real is read-only
            im.Data<float>().Should().BeEquivalentTo(new[] { 0.0f, 0.0f, 0.0f });
        }

        [DataTestMethod]
        // Non-bool numeric dtypes where [1,2,3] round-trips (bool collapses to [1,1,1], tested separately).
        [DataRow(NPTypeCode.Byte)]
        [DataRow(NPTypeCode.SByte)]
        [DataRow(NPTypeCode.Int16)]
        [DataRow(NPTypeCode.UInt16)]
        [DataRow(NPTypeCode.Int32)]
        [DataRow(NPTypeCode.UInt32)]
        [DataRow(NPTypeCode.Int64)]
        [DataRow(NPTypeCode.UInt64)]
        [DataRow(NPTypeCode.Char)]
        [DataRow(NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double)]
        [DataRow(NPTypeCode.Decimal)]
        public void RealImag_RealDtypes_RealPreservesDtype_ImagIsZerosOfSameDtype(NPTypeCode dt)
        {
            var a = new NDArray(new[] { 1, 2, 3 }).astype(dt);
            np.real(a).typecode.Should().Be(dt);
            np.real(a).astype(np.float64).Data<double>().Should().BeEquivalentTo(new[] { 1.0, 2.0, 3.0 });

            var im = np.imag(a);
            im.typecode.Should().Be(dt);
            im.Shape.IsWriteable.Should().BeFalse();
            im.astype(np.float64).Data<double>().Should().BeEquivalentTo(new[] { 0.0, 0.0, 0.0 });
        }

        [TestMethod]
        public void RealImag_Bool_RealIsInput_ImagIsFalse()
        {
            var a = new NDArray(new[] { true, false, true });
            np.real(a).typecode.Should().Be(NPTypeCode.Boolean);
            np.real(a).Data<bool>().Should().BeEquivalentTo(new[] { true, false, true });

            var im = np.imag(a);
            im.typecode.Should().Be(NPTypeCode.Boolean);
            im.Data<bool>().Should().BeEquivalentTo(new[] { false, false, false });
        }

        [TestMethod]
        public void Imag_RealInput_Empty()
        {
            var a = np.array(new double[] { });
            np.imag(a).size.Should().Be(0);
            np.imag(a).typecode.Should().Be(NPTypeCode.Double);
        }

        // =====================================================================================
        // np.angle — COMPLEX input (the phase spectrum). complex128 -> float64.
        // =====================================================================================

        [TestMethod]
        public void Angle_Complex_FourQuadrants_Radians()
        {
            var z = new NDArray(new[] { new Complex(1, 1), new Complex(1, -1), new Complex(-1, 1), new Complex(-1, -1) });
            var a = np.angle(z);
            a.typecode.Should().Be(NPTypeCode.Double);
            var d = a.Data<double>();
            d[0].Should().BeApproximately(Pi / 4, 1e-15);
            d[1].Should().BeApproximately(-Pi / 4, 1e-15);
            d[2].Should().BeApproximately(3 * Pi / 4, 1e-15);
            d[3].Should().BeApproximately(-3 * Pi / 4, 1e-15);
        }

        [TestMethod]
        public void Angle_Complex_Axes_Zero_And_SignedZeroConvention()
        {
            // NumPy: np.angle([0., -0j... ]) convention — atan2 sign rules.
            var z = new NDArray(new[]
            {
                new Complex(1, 0), new Complex(0, 1), new Complex(-1, 0), new Complex(0, -1),
                new Complex(0, 0), new Complex(-0.0, 0.0)
            });
            var d = np.angle(z).Data<double>();
            d[0].Should().Be(0.0);                          // +real axis
            d[1].Should().BeApproximately(Pi / 2, 1e-15);   // +imag axis
            d[2].Should().BeApproximately(Pi, 1e-15);       // -real axis
            d[3].Should().BeApproximately(-Pi / 2, 1e-15);  // -imag axis
            d[4].Should().Be(0.0);                          // 0+0j
            d[5].Should().BeApproximately(Pi, 1e-15);       // -0+0j -> atan2(+0,-0) = pi
        }

        [TestMethod]
        public void Angle_Complex_Degrees()
        {
            var z = new NDArray(new[] { new Complex(1, 1), new Complex(1, -1), new Complex(-1, 1), new Complex(0, 1) });
            var d = np.angle(z, deg: true);
            d.typecode.Should().Be(NPTypeCode.Double);
            d.Data<double>()[0].Should().BeApproximately(45.0, 1e-12);
            d.Data<double>()[1].Should().BeApproximately(-45.0, 1e-12);
            d.Data<double>()[2].Should().BeApproximately(135.0, 1e-12);
            d.Data<double>()[3].Should().BeApproximately(90.0, 1e-12);
        }

        // =====================================================================================
        // np.angle — REAL input: 0 for positive, pi for negative; per-dtype float tier.
        // =====================================================================================

        [TestMethod]
        public void Angle_RealInput_ZeroForPositive_PiForNegative()
        {
            var a = new NDArray(new[] { 1.0, -1.0, 0.0, 2.0, -3.0 });
            var d = np.angle(a);
            d.typecode.Should().Be(NPTypeCode.Double);
            d.Data<double>()[0].Should().Be(0.0);
            d.Data<double>()[1].Should().BeApproximately(Pi, 1e-15);
            d.Data<double>()[2].Should().Be(0.0);
            d.Data<double>()[3].Should().Be(0.0);
            d.Data<double>()[4].Should().BeApproximately(Pi, 1e-15);
        }

        [DataTestMethod]
        // NumPy 2.4.2 real-input angle tier (probed): bool/int32+/float64 -> float64;
        // int8/uint8/float16 -> float16; int16/uint16/float32 -> float32.
        [DataRow(NPTypeCode.Boolean, NPTypeCode.Double)]
        [DataRow(NPTypeCode.SByte, NPTypeCode.Half)]
        [DataRow(NPTypeCode.Byte, NPTypeCode.Half)]
        [DataRow(NPTypeCode.Int16, NPTypeCode.Single)]
        [DataRow(NPTypeCode.UInt16, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Int32, NPTypeCode.Double)]
        [DataRow(NPTypeCode.UInt32, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Int64, NPTypeCode.Double)]
        [DataRow(NPTypeCode.UInt64, NPTypeCode.Double)]
        [DataRow(NPTypeCode.Half, NPTypeCode.Half)]
        [DataRow(NPTypeCode.Single, NPTypeCode.Single)]
        [DataRow(NPTypeCode.Double, NPTypeCode.Double)]
        public void Angle_RealInput_DtypeTierMatchesNumpy(NPTypeCode input, NPTypeCode expected)
        {
            var a = new NDArray(new[] { 1, 2 }).astype(input);
            np.angle(a).typecode.Should().Be(expected);
            np.angle(a, deg: true).typecode.Should().Be(expected);
        }

        [TestMethod]
        public void Angle_RealInput_Float16_Degrees_BitExactWithNumpy()
        {
            // NumPy computes `a *= 180/pi` in the tier's precision: angle(int8 negative, deg=True) at
            // float16 is 179.9 (bits 0x599f), NOT 180 — the multiply rounds in float16. Bit-exact pin.
            var a = new NDArray(new[] { 1, -1 }).astype(NPTypeCode.SByte); // -> float16
            var d = np.angle(a, deg: true);
            d.typecode.Should().Be(NPTypeCode.Half);
            BitConverter.HalfToUInt16Bits((Half)d.GetValue(0)).Should().Be((ushort)0x0000);   // 0
            BitConverter.HalfToUInt16Bits((Half)d.GetValue(1)).Should().Be((ushort)0x599f);   // 179.9 (NumPy-exact)
        }

        // =====================================================================================
        // FFT companion — np.angle(np.fft.fft(x)) and np.fft.fft(x).real / .imag
        // =====================================================================================

        [TestMethod]
        public void Angle_And_Real_Of_FFT_Output_ComputeTheSpectrum()
        {
            var x = new NDArray(new[] { 1.0, 2.0, 1.0, -1.0, 1.5, 0.0, 0.5, 2.0 });
            var A = np.fft.fft(x);

            // A.real / A.imag are float64 lane views onto the complex spectrum.
            var re = np.real(A);
            var im = np.imag(A);
            re.typecode.Should().Be(NPTypeCode.Double);
            re.GetDouble(0).Should().BeApproximately(7.0, 1e-12); // DC bin = sum(x)

            // np.angle(A) is the phase spectrum; combined with |A| it reconstructs A.
            var ang = np.angle(A);
            ang.typecode.Should().Be(NPTypeCode.Double);
            for (int i = 0; i < A.size; i++)
            {
                double reci = re.GetDouble(i), imci = im.GetDouble(i);
                ang.GetDouble(i).Should().BeApproximately(Math.Atan2(imci, reci), 1e-12);
            }

            // amplitude spectrum np.abs(A) and phase np.angle(A) are the documented FFT accessors.
            ang.GetDouble(0).Should().Be(0.0); // real positive DC -> phase 0
        }

        // =====================================================================================
        // [Misaligned] — the single documented divergence: NaN sign bit.
        // =====================================================================================

        [TestMethod]
        [Misaligned]
        public void Angle_Complex_NaN_YieldsDotNetNegativeQuietNaN()
        {
            // NumPy canonicalizes a NaN angle to the positive quiet NaN 0x7ff8000000000000; NumSharp's
            // arctan2 returns .NET's NEGATIVE quiet NaN 0xfff8000000000000 (sign-bit only, same magnitude
            // and quiet-ness) — the same .NET-NaN class documented for the float32 kernels. FFT output is
            // finite, so the FFT-companion phase spectrum is unaffected.
            var z = new NDArray(new[] { new Complex(double.NaN, 1), new Complex(1, double.NaN) });
            var d = np.angle(z).Data<double>();
            double.IsNaN(d[0]).Should().BeTrue();
            double.IsNaN(d[1]).Should().BeTrue();
            BitConverter.DoubleToUInt64Bits(d[0]).Should().Be(0xfff8000000000000);
        }
    }
}
