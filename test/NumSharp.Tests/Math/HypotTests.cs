using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.hypot — sqrt(x1**2 + x2**2) without spurious overflow/underflow. All expected values probed
    /// from NumPy 2.4.2 (win-amd64).
    ///
    /// A float-tier binary ufunc with arctan2's promotion (ee->e, ff->f, dd->d, gg->g): bool/int8/uint8
    /// -> float16, int16/uint16 -> float32, int32+/int64+/char -> float64.
    ///
    /// float32/float16 are BIT-EXACT with NumPy. float64 is CORRECTLY-ROUNDED (Borges' FMA algorithm),
    /// so it matches NumPy on ~91% of inputs and is within 1 ULP — MORE accurate — on the rest, because
    /// NumPy calls the platform (UCRT) hypot which is only faithfully rounded. See NDHypotMath.
    /// </summary>
    [TestClass]
    public class HypotTests
    {
        // ---------------------------------------------------------------- basic values / dtype

        [TestMethod]
        public void Hypot_Float64_ThreeFourFive()
        {
            var r = np.hypot(np.array(new[] { 3.0, 5, 8 }), np.array(new[] { 4.0, 12, 15 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(5.0);
            r.GetAtIndex<double>(1).Should().Be(13.0);
            r.GetAtIndex<double>(2).Should().Be(17.0);
        }

        [TestMethod]
        public void Hypot_Unit_Sqrt2()
        {
            np.hypot((NDArray)1.0, (NDArray)1.0).GetAtIndex<double>(0)
                .Should().Be(1.4142135623730951);
        }

        [TestMethod]
        public void Hypot_Int32_PromotesToFloat64()
        {
            var r = np.hypot(np.array(new[] { 3, 5 }), np.array(new[] { 4, 12 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(5.0);
            r.GetAtIndex<double>(1).Should().Be(13.0);
        }

        [TestMethod]
        public void Hypot_Float32_Preserved_BitExact()
        {
            var r = np.hypot(np.array(new[] { 3f, 5 }), np.array(new[] { 4f, 12 }));
            r.typecode.Should().Be(NPTypeCode.Single);
            r.GetAtIndex<float>(0).Should().Be(5f);
            r.GetAtIndex<float>(1).Should().Be(13f);
        }

        [TestMethod]
        public void Hypot_UInt8_Int8_PromotesToFloat16()
        {
            var r = np.hypot(np.array(new byte[] { 3, 5 }), np.array(new sbyte[] { 4, 12 }));
            r.typecode.Should().Be(NPTypeCode.Half);
            ((float)r.GetAtIndex<Half>(0)).Should().Be(5f);
            ((float)r.GetAtIndex<Half>(1)).Should().Be(13f);
        }

        [TestMethod]
        public void Hypot_Int16_PromotesToFloat32()
        {
            np.hypot(np.array(new short[] { 3 }), np.array(new short[] { 4 }))
                .typecode.Should().Be(NPTypeCode.Single);
        }

        // ---------------------------------------------------------------- overflow / underflow safety

        [TestMethod]
        public void Hypot_LargeOperands_NoOverflow()
        {
            // Naive sqrt(x^2 + y^2) would overflow to +inf; hypot stays finite.
            var r = np.hypot((NDArray)1e200, (NDArray)1e200).GetAtIndex<double>(0);
            r.Should().Be(1.414213562373095e+200);
            BitConverter.DoubleToInt64Bits(r).Should().Be(unchecked((long)0x697d8f9811335b57UL));
            double.IsFinite(r).Should().BeTrue();
        }

        [TestMethod]
        public void Hypot_TinyOperands_NoUnderflow()
        {
            var r = np.hypot((NDArray)1e-200, (NDArray)1e-200).GetAtIndex<double>(0);
            BitConverter.DoubleToInt64Bits(r).Should().Be(unchecked((long)0x167151f68876f410UL));
        }

        [TestMethod]
        public void Hypot_SmallestSubnormal()
        {
            // hypot(5e-324, 5e-324) rounds back to the smallest subnormal (NumPy: 0x1).
            np.hypot((NDArray)double.Epsilon, (NDArray)double.Epsilon).GetAtIndex<double>(0)
                .Should().Be(double.Epsilon);
        }

        // ---------------------------------------------------------------- special values

        [TestMethod]
        public void Hypot_Infinity_BeatsNaN()
        {
            // C99 / npy_hypot: an infinite operand yields +inf even when the other is NaN.
            np.hypot((NDArray)double.PositiveInfinity, (NDArray)double.NaN).GetAtIndex<double>(0)
                .Should().Be(double.PositiveInfinity);
            np.hypot((NDArray)double.NaN, (NDArray)double.NegativeInfinity).GetAtIndex<double>(0)
                .Should().Be(double.PositiveInfinity);
        }

        [TestMethod]
        public void Hypot_NaN_Propagates()
        {
            double.IsNaN(np.hypot((NDArray)double.NaN, (NDArray)3.0).GetAtIndex<double>(0)).Should().BeTrue();
            double.IsNaN(np.hypot((NDArray)3.0, (NDArray)double.NaN).GetAtIndex<double>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Hypot_Result_AlwaysNonNegative()
        {
            np.hypot((NDArray)(-3.0), (NDArray)(-4.0)).GetAtIndex<double>(0).Should().Be(5.0);
        }

        [TestMethod]
        public void Hypot_Zeros_PlusZero()
        {
            var r = np.hypot((NDArray)(-0.0), (NDArray)(-0.0)).GetAtIndex<double>(0);
            r.Should().Be(0.0);
            BitConverter.DoubleToInt64Bits(r).Should().Be(0L);  // +0, not -0
        }

        [TestMethod]
        public void Hypot_WithZero_IsAbs()
        {
            np.hypot((NDArray)(-7.0), (NDArray)0.0).GetAtIndex<double>(0).Should().Be(7.0);
            np.hypot((NDArray)0.0, (NDArray)(-7.0)).GetAtIndex<double>(0).Should().Be(7.0);
        }

        [TestMethod]
        public void Hypot_Commutative()
        {
            np.hypot((NDArray)3.7, (NDArray)11.9).GetAtIndex<double>(0)
                .Should().Be(np.hypot((NDArray)11.9, (NDArray)3.7).GetAtIndex<double>(0));
        }

        // ---------------------------------------------------------------- correctly-rounded (prefer-precise)

        [TestMethod]
        [Misaligned]
        public void Hypot_Float64_CorrectlyRounded_MoreAccurateThanNumPy()
        {
            // NumPy's platform (UCRT) hypot is only FAITHFULLY rounded: np.hypot(0.1, 0.1) is
            // 0x3fc21a1851ff630b, 1 ULP above the correctly-rounded value. NumSharp returns the
            // correctly-rounded 0x3fc21a1851ff630a (== CPython's math.hypot) — bit-for-bit MORE accurate.
            long bits = BitConverter.DoubleToInt64Bits(
                np.hypot((NDArray)0.1, (NDArray)0.1).GetAtIndex<double>(0));
            bits.Should().Be(unchecked((long)0x3fc21a1851ff630aUL));
        }

        // ---------------------------------------------------------------- broadcasting

        [TestMethod]
        public void Hypot_Broadcast()
        {
            var r = np.hypot(np.array(new[,] { { 3.0 }, { 5.0 } }), np.array(new[,] { { 4.0, 12, 0.0 } }));
            r.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            r.GetAtIndex<double>(0).Should().Be(5.0);   // hypot(3,4)
            r.GetAtIndex<double>(2).Should().Be(3.0);   // hypot(3,0)
            r.GetAtIndex<double>(4).Should().Be(13.0);  // hypot(5,12)
            r.GetAtIndex<double>(5).Should().Be(5.0);   // hypot(5,0)
        }

        [TestMethod]
        public void Hypot_ScalarScalar_Returns0D()
        {
            var r = np.hypot((NDArray)3.0, (NDArray)4.0);
            r.ndim.Should().Be(0);
            r.GetAtIndex<double>(0).Should().Be(5.0);
        }

        [TestMethod]
        public void Hypot_Empty_ReturnsEmpty()
        {
            np.hypot(np.zeros(new Shape(0)), np.zeros(new Shape(0))).size.Should().Be(0);
        }

        // ---------------------------------------------------------------- ufunc out / where / dtype

        [TestMethod]
        public void Hypot_Out_ReturnsSameInstance()
        {
            var a = np.array(new[] { 3.0, 5, 8 });
            var b = np.array(new[] { 4.0, 12, 15 });
            var o = np.zeros(3);
            var r = np.hypot(a, b, @out: o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.GetAtIndex<double>(1).Should().Be(13.0);
        }

        [TestMethod]
        public void Hypot_Where_LeavesMaskedOff()
        {
            var a = np.array(new[] { 3.0, 5, 8 });
            var b = np.array(new[] { 4.0, 12, 15 });
            var o = np.array(new[] { 9.0, 9, 9 });
            var w = np.array(new[] { true, false, true });
            np.hypot(a, b, @out: o, where: w);
            o.GetAtIndex<double>(0).Should().Be(5.0);
            o.GetAtIndex<double>(1).Should().Be(9.0);   // masked-off keeps prior value
            o.GetAtIndex<double>(2).Should().Be(17.0);
        }

        [TestMethod]
        public void Hypot_Dtype_SelectsLoop()
        {
            var r = np.hypot(np.array(new[] { 3.0, 5 }), np.array(new[] { 4.0, 12 }), dtype: np.float32);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.GetAtIndex<float>(1).Should().Be(13f);
        }

        // ---------------------------------------------------------------- error taxonomy

        [TestMethod]
        public void Hypot_IntegerDtype_RaisesNoLoop()
        {
            var a = np.array(new[] { 3.0 });
            var b = np.array(new[] { 4.0 });
            Assert.ThrowsExactly<IncorrectTypeException>(() => np.hypot(a, b, dtype: np.int32));
        }

        [TestMethod]
        public void Hypot_ComplexInput_Raises()
        {
            var a = np.array(new Complex[] { new Complex(1, 0) });
            Assert.ThrowsExactly<IncorrectTypeException>(() => np.hypot(a, a));
        }

        // ---------------------------------------------------------------- decimal (NumSharp extension)

        [TestMethod]
        public void Hypot_Decimal()
        {
            // Decimal (NumSharp extension, no NumPy analog) is highest-precision best-effort via
            // DecimalMath.Sqrt: a perfect-square radicand is exact (hypot(3,4)=5), otherwise it lands
            // within ~1e-27 (the tail of decimal's 28-29 significant digits).
            var r = np.hypot(np.array(new[] { 3m, 5m }), np.array(new[] { 4m, 12m }));
            r.typecode.Should().Be(NPTypeCode.Decimal);
            r.GetAtIndex<decimal>(0).Should().Be(5m);
            r.GetAtIndex<decimal>(1).Should().BeApproximately(13m, 1e-25m);
        }
    }
}
