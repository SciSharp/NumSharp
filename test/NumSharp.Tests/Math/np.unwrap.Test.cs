using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.unwrap — unwrap a signal by taking the complement of large deltas w.r.t. a period.
    ///     All expected values are probed from NumPy 2.4.2 (win-amd64). Bit-exactness across the
    ///     dtype × layout × parameter space is gated by the differential-fuzz "unwrap" tier (702
    ///     cases); this suite pins the documented behaviours, the dtype/shape contracts and the
    ///     error taxonomy.
    ///     <para>
    ///     Two overloads mirror NumPy's keyword-only <c>period</c>: the <c>double</c> overload is
    ///     the default (float period, used for radian phase) and the <c>long</c> overload is the
    ///     integer-preserving path (NumPy's <c>period=&lt;int&gt;</c>). For an INTEGER input the two
    ///     paths compute the SAME values and differ only in output dtype.
    ///     </para>
    /// </summary>
    [TestClass]
    public class NpUnwrapTest
    {
        private const double Tol = 1e-13;

        // ---- the NumPy documentation examples ------------------------------------------------

        /// <summary>The canonical radian-phase example (float64, default period 2*pi).</summary>
        [TestMethod]
        public void Unwrap_Phase_Float64_Default()
        {
            // phase = linspace(0, pi, 5); phase[3:] += pi
            var phase = np.linspace(0, System.Math.PI, 5);
            phase["3:"] += System.Math.PI;
            var r = np.unwrap(phase);
            r.dtype.Should().Be(np.float64);
            r.shape.Should().Equal(5);
            // NumPy 2.4.2: [0, pi/4, pi/2, -pi/4, 0]
            r.GetDouble(0).Should().BeApproximately(0.0, Tol);
            r.GetDouble(1).Should().BeApproximately(System.Math.PI / 4, Tol);
            r.GetDouble(2).Should().BeApproximately(System.Math.PI / 2, Tol);
            r.GetDouble(3).Should().BeApproximately(-System.Math.PI / 4, Tol);
            r.GetDouble(4).Should().BeApproximately(0.0, Tol);
        }

        /// <summary>Integer period keeps the integer dtype and value (period=4 -> [0,1,2,3,4]).</summary>
        [TestMethod]
        public void Unwrap_IntegerPeriod_KeepsIntegerDtype()
        {
            var r = np.unwrap(np.array(new[] { 0, 1, 2, -1, 0 }), period: 4);
            r.dtype.Should().Be(np.int32);          // input int32 preserved (NumSharp int[] -> int32)
            r.ToArray<int>().Should().Equal(0, 1, 2, 3, 4);

            np.unwrap(np.array(new[] { 1, 2, 3, 4, 5, 6, 1, 2, 3 }), period: 6)
                .ToArray<int>().Should().Equal(1, 2, 3, 4, 5, 6, 7, 8, 9);

            np.unwrap(np.array(new[] { 2, 3, 4, 5, 2, 3, 4, 5 }), period: 4)
                .ToArray<int>().Should().Equal(2, 3, 4, 5, 6, 7, 8, 9);
        }

        /// <summary>The degrees example (period=360, float).</summary>
        [TestMethod]
        public void Unwrap_Degrees_Period360()
        {
            var deg = np.mod(np.linspace(0, 720, 19), 360.0) - 180.0;
            var r = np.unwrap(deg, period: 360.0);
            r.dtype.Should().Be(np.float64);
            // NumPy: monotone ramp -180, -140, ..., 540
            double[] expected = { -180, -140, -100, -60, -20, 20, 60, 100, 140,
                                  180, 220, 260, 300, 340, 380, 420, 460, 500, 540 };
            for (int i = 0; i < expected.Length; i++)
                r.GetDouble(i).Should().BeApproximately(expected[i], 1e-10);
        }

        // ---- the float vs integer period distinction -----------------------------------------

        /// <summary>An integer input with a FLOAT period yields float64 (weak-float promotion).</summary>
        [TestMethod]
        public void Unwrap_IntInput_FloatPeriod_IsFloat64()
        {
            var r = np.unwrap(np.array(new[] { 0, 1, 2, -1, 0 }), period: 4.0);
            r.dtype.Should().Be(np.float64);
            r.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0);
        }

        /// <summary>
        ///     Integer and float periods compute the SAME values for an integer input (an integer
        ///     difference never lands in the fractional interval gap) — only the dtype differs.
        /// </summary>
        [TestMethod]
        public void Unwrap_OddIntegerPeriod_NotAmbiguous()
        {
            // period=5 is odd => boundary NOT ambiguous on the integer path.
            np.unwrap(np.array(new[] { 0, 1, 2, 3, 4, 0, 1, 2 }), period: 5)
                .ToArray<int>().Should().Equal(0, 1, 2, 3, 4, 5, 6, 7);
            np.unwrap(np.array(new[] { 0, 2, 5 }), period: 5)
                .ToArray<int>().Should().Equal(0, 2, 0);
        }

        // ---- discont ---------------------------------------------------------------------------

        /// <summary>
        ///     A discont larger than period/2 suppresses unwrapping of jumps below it; a discont
        ///     smaller than period/2 has no effect (clamped up to period/2).
        /// </summary>
        [TestMethod]
        public void Unwrap_Discont_Behaviour()
        {
            var p = np.array(new[] { 0.0, 3.0, 6.0, 9.0 });    // jumps of 3, period 2*pi ~ 6.283
            // default discont (pi ~ 3.14): every jump (3) is below it -> no unwrap.
            np.unwrap(p).ToArray<double>().Should().Equal(0.0, 3.0, 6.0, 9.0);
            // discont below period/2 is clamped -> still no unwrap.
            np.unwrap(p, discont: 1.0).ToArray<double>().Should().Equal(0.0, 3.0, 6.0, 9.0);
            // discont above period/2 (5) -> still no unwrap here (jumps are 3 < 5), unchanged.
            np.unwrap(p, discont: 5.0).ToArray<double>().Should().Equal(0.0, 3.0, 6.0, 9.0);
        }

        // ---- axis ------------------------------------------------------------------------------

        /// <summary>Unwrap along a chosen axis of a 2-D array (integer period, axis 0).</summary>
        [TestMethod]
        public void Unwrap_Axis0_2D()
        {
            var b = np.array(new int[,] { { 0, 0 }, { 1, 2 }, { 2, 4 }, { -1, 1 }, { 0, 3 } });
            var r = np.unwrap(b, period: 4, discont: null, axis: 0);
            r.dtype.Should().Be(np.int32);
            r.shape.Should().Equal(5, 2);
            // NumPy 2.4.2 column-wise: [[0,0],[1,2],[2,4],[3,5],[4,7]]
            r.ToArray<int>().Should().Equal(0, 0, 1, 2, 2, 4, 3, 5, 4, 7);
        }

        /// <summary>Negative axis normalizes; axis=-2 == axis=0 for a 2-D array.</summary>
        [TestMethod]
        public void Unwrap_NegativeAxis_Normalizes()
        {
            var b = np.array(new int[,] { { 0, 0 }, { 1, 2 }, { 2, 4 }, { -1, 1 }, { 0, 3 } });
            var r0 = np.unwrap(b, period: 4, axis: 0);
            var rm2 = np.unwrap(b, period: 4, axis: -2);
            rm2.ToArray<int>().Should().Equal(r0.ToArray<int>());
        }

        // ---- dtype coverage --------------------------------------------------------------------

        /// <summary>float32/float16 preserve their width (weak scalars never widen the float).</summary>
        [TestMethod]
        public void Unwrap_Float32_Float16_PreserveWidth()
        {
            np.unwrap(np.array(new[] { 0f, 1f, 2f, 3f, 0f }), period: 4.0f).dtype.Should().Be(np.float32);
            np.unwrap(np.array(new[] { 0f, 1f, 2f, 3f, 0f }).astype(np.float16)).dtype.Should().Be(np.float16);
        }

        /// <summary>A bool input on the integer path promotes to int64 (weak int is higher-kind).</summary>
        [TestMethod]
        public void Unwrap_Bool_IntegerPeriod_IsInt64()
        {
            var r = np.unwrap(np.array(new[] { true, false, true, true }), period: 4);
            r.dtype.Should().Be(np.int64);
            r.ToArray<long>().Should().Equal(1, 0, 1, 1);   // all diffs 0/1 < discont(2) -> no correction
        }

        /// <summary>An unsigned input with a FLOAT period is valid (-> float64), unlike the integer path.</summary>
        [TestMethod]
        public void Unwrap_Unsigned_FloatPeriod_IsFloat64()
        {
            var r = np.unwrap(np.array(new byte[] { 0, 1, 2, 3, 0 }), period: 4.0);
            r.dtype.Should().Be(np.float64);
        }

        /// <summary>Decimal (a NumSharp-only dtype) is treated as a float family and preserved.</summary>
        [TestMethod]
        public void Unwrap_Decimal_Preserved()
        {
            var r = np.unwrap(np.array(new decimal[] { 0m, 3m, 6m, 9m }), period: 4.0);
            r.dtype.Should().Be(np.@decimal);
            r.ToArray<decimal>().Should().Equal(0m, -1m, -2m, -3m);
        }

        // ---- edge cases ------------------------------------------------------------------------

        /// <summary>A single element (or empty) has no adjacent difference — the input is returned (as the out dtype).</summary>
        [TestMethod]
        public void Unwrap_SingleAndEmpty()
        {
            np.unwrap(np.array(new[] { 5.0 })).ToArray<double>().Should().Equal(5.0);
            np.unwrap(np.array(new[] { 5 }), period: 4).ToArray<int>().Should().Equal(5);
            np.unwrap(np.array(new double[] { })).size.Should().Be(0);
            np.unwrap(np.array(new int[] { }), period: 4).dtype.Should().Be(np.int32);
        }

        /// <summary>The result is an independent copy — mutating it does not touch the input.</summary>
        [TestMethod]
        public void Unwrap_ReturnsIndependentCopy()
        {
            var p = np.array(new[] { 0.0, 1.0, 2.0, 3.0 });
            var r = np.unwrap(p);
            r[0] = 99.0;
            p.GetDouble(0).Should().Be(0.0, "unwrap must not alias its input");
        }

        /// <summary>A non-contiguous (reversed) input is read through its strides, result is C-contiguous.</summary>
        [TestMethod]
        public void Unwrap_ReversedInput()
        {
            var full = np.array(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var rev = full["::-1"];                      // [4,3,2,1,0], negative stride
            var r = np.unwrap(rev, period: 4.0);
            // diff is -1 everywhere (< discont 2) so no correction -> equals the reversed values.
            r.ToArray<double>().Should().Equal(4.0, 3.0, 2.0, 1.0, 0.0);
        }

        // ---- error taxonomy --------------------------------------------------------------------

        /// <summary>A 0-D input is rejected (NumPy's inner diff requires at least 1-D).</summary>
        [TestMethod]
        public void Unwrap_ZeroDim_Throws()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() => np.unwrap(np.array(5.0)));
            ex.Message.Should().Contain("at least one dimensional");
        }

        /// <summary>An out-of-range axis raises AxisError reporting the original axis.</summary>
        [TestMethod]
        public void Unwrap_AxisOutOfRange_Throws()
        {
            var b = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            var ex = Assert.ThrowsException<AxisError>(() => np.unwrap(b, period: 4, axis: 5));
            ex.Message.Should().Contain("axis 5 is out of bounds for array of dimension 2");
        }

        /// <summary>A complex input reproduces NumPy's TypeError (mod has no complex loop).</summary>
        [TestMethod]
        public void Unwrap_Complex_Throws_TypeError()
        {
            var z = np.array(new[] { new Complex(0, 0), new Complex(1, 1), new Complex(2, 2) });
            var ex = Assert.ThrowsException<TypeError>(() => np.unwrap(z, period: 4));
            ex.Message.Should().Contain("ufunc 'remainder' not supported");
        }

        /// <summary>
        ///     An unsigned integer with an INTEGER period reproduces NumPy's OverflowError: the
        ///     negative interval_low cannot be cast to the unsigned dtype. Fires for empty inputs too.
        /// </summary>
        [TestMethod]
        public void Unwrap_UnsignedIntegerPeriod_Throws_Overflow()
        {
            var ex = Assert.ThrowsException<OverflowException>(
                () => np.unwrap(np.array(new byte[] { 0, 1, 2, 3, 0 }), period: 4));
            ex.Message.Should().Contain("out of bounds for uint8");

            // Empty unsigned still raises (the scalar-cast check is size-independent, matching NumPy).
            Assert.ThrowsException<OverflowException>(
                () => np.unwrap(np.array(new byte[] { }), period: 4));
        }

        /// <summary>A signed integer too narrow for the derived interval also overflows (int8, period 300).</summary>
        [TestMethod]
        public void Unwrap_NarrowSignedInteger_Throws_Overflow()
        {
            var ex = Assert.ThrowsException<OverflowException>(
                () => np.unwrap(np.array(new sbyte[] { 0, 1, 2, 3, 0 }), period: 300));
            ex.Message.Should().Contain("-150 out of bounds for int8");
        }
    }
}
