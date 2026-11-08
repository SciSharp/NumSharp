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
    /// Every float dtype is BIT-EXACT with NumPy: NumPy calls the platform (UCRT) hypot, which is the plain
    /// sqrt(x*x + y*y) with a power-of-two rescale outside the safe range — only faithfully rounded — and
    /// NDHypotMath is that model (parity QA-25; it was the correctly rounded Borges kernel before). A NaN
    /// operand comes back as UCRT returns it, the first one with its payload and sign. See NDHypotMath.
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

        // ---------------------------------------------------------------- UCRT's rounding (parity QA-25)

        /// <summary>
        /// NumPy's platform (UCRT) hypot is only FAITHFULLY rounded: <c>np.hypot(0.1, 0.1)</c> is
        /// <c>0x3fc21a1851ff630b</c>, 1 ULP above the correctly rounded value (CPython's <c>math.hypot</c> gives
        /// <c>...630a</c>). NumSharp computes UCRT's <c>sqrt(x*x + y*y)</c> and returns NumPy's bits, on the scalar
        /// path, the contiguous SIMD path (length 8) and the scalar-broadcast path. Until parity QA-25 this test was
        /// [Misaligned] and pinned the correctly rounded <c>...630a</c>.
        /// </summary>
        [TestMethod]
        public void Hypot_Float64_IsUcrtsFaithfulRounding_LikeNumPy()
        {
            long bits = BitConverter.DoubleToInt64Bits(
                np.hypot((NDArray)0.1, (NDArray)0.1).GetAtIndex<double>(0));
            bits.Should().Be(unchecked((long)0x3fc21a1851ff630bUL));

            using var eight = np.full(new Shape(8), 0.1);
            using var vector = np.hypot(eight, eight);
            using var broadcast = np.hypot(eight, NDArray.Scalar(0.1));
            for (int i = 0; i < 8; i++)
            {
                BitConverter.DoubleToInt64Bits(vector.GetAtIndex<double>(i)).Should().Be(unchecked((long)0x3fc21a1851ff630bUL), "the SIMD lane");
                BitConverter.DoubleToInt64Bits(broadcast.GetAtIndex<double>(i)).Should().Be(unchecked((long)0x3fc21a1851ff630bUL), "the broadcast lane");
            }
        }

        /// <summary>
        /// UCRT returns its FIRST NaN operand untouched — payload and sign — and an infinite operand beats a NaN
        /// (probed through ucrtbase and np.hypot: <c>hypot(1, -nan(3))</c> is that <c>-nan(3)</c>; <c>hypot(nan(4),
        /// -nan(5))</c> is <c>nan(4)</c>). NumSharp returned the canonical .NET NaN before parity QA-25 (the oracle
        /// tokenizes NaNs, so only a raw-bit test sees it).
        /// </summary>
        [TestMethod]
        public void Hypot_Float64_NaN_IsTheFirstNaNOperandVerbatim()
        {
            static double D(ulong bits) => BitConverter.Int64BitsToDouble(unchecked((long)bits));
            static ulong H(double x, double y) => (ulong)BitConverter.DoubleToInt64Bits(np.hypot((NDArray)x, (NDArray)y).GetAtIndex<double>(0));

            H(D(0x7ff8000000000001), 1.0).Should().Be(0x7ff8000000000001UL);
            H(D(0xfff8000000000002), 1.0).Should().Be(0xfff8000000000002UL);
            H(1.0, D(0xfff8000000000003)).Should().Be(0xfff8000000000003UL);
            H(D(0x7ff8000000000004), D(0xfff8000000000005)).Should().Be(0x7ff8000000000004UL);
            H(double.PositiveInfinity, D(0xfff8000000000006)).Should().Be(0x7ff0000000000000UL);
        }

        /// <summary>
        /// The complex functions built on <c>npy_hypot</c> share the UCRT model (parity QA-25): NumPy's <c>nc_log1p</c>
        /// is <c>log(hypot(x + 1, y))</c>, and near <c>|z + 1| = 1</c> a 1-ULP hypot difference is a 5e-4 relative error
        /// after the log. <c>np.log1p(complex(-1.3339345080004047e-16, 9.344550712680563e-07)).real</c> is
        /// <c>4.365396932825163e-13</c> in NumPy; the block-form hypot NumSharp used gave <c>4.3631764867759135e-13</c>
        /// (4.4e12 ULP away). <c>np.sqrt</c> (csqrt's <c>hypot(a, b)</c>) takes the same kernel; a 1.2M-part dense
        /// sample of log1p and sqrt is bit-identical to NumPy.
        /// </summary>
        [TestMethod]
        public void ComplexLog1pAndSqrt_UseUcrtsHypot_LikeNumPy()
        {
            using var z = np.array(new[] { new Complex(-1.3339345080004047e-16, 9.344550712680563e-07) });

            using var log1p = np.log1p(z);
            log1p.GetAtIndex<Complex>(0).Real.Should().Be(4.365396932825163e-13);

            // csqrt computes t = sqrt((a + hypot(a, b)) * 0.5): for this z the block form's hypot is 1 ULP off UCRT's and
            // both parts came out 1 ULP from NumPy (0x...588 / 0x...36f).
            using var w = np.array(new[] { new Complex(0.6441349919701778, -0.7410804937363649) });
            using var sqrt = np.sqrt(w);
            BitConverter.DoubleToInt64Bits(sqrt.GetAtIndex<Complex>(0).Real).Should().Be(unchecked((long)0x3fecda8002ac9589UL));
            BitConverter.DoubleToInt64Bits(sqrt.GetAtIndex<Complex>(0).Imaginary).Should().Be(unchecked((long)0xbfda4cf7dd94936eUL));
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
