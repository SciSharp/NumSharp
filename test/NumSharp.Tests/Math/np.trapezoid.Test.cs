using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.trapezoid — composite trapezoidal integration along an axis. All expected values are
    ///     from NumPy 2.4.2 (probed directly). The result is a pure composition
    ///     <c>sum(d*(y[1:]+y[:-1])/2, axis)</c>, so it is BIT-exact with NumPy for finite data; these
    ///     assertions cover the docstring examples, the dtype tier (NEP50), scalar/coordinate spacing,
    ///     the axis forms, integer overflow-wrap, and the degenerate/error corners. (`np.trapz` was
    ///     REMOVED in NumPy 2.x — only `trapezoid` exists.)
    /// </summary>
    [TestClass]
    public class NpTrapezoidTest
    {
        // ---------------------------- docstring examples ----------------------------

        [TestMethod]
        public void Trapezoid_Basic()
        {
            // np.trapezoid([1,2,3]) -> 4.0 (a 0-d scalar)
            var r = np.trapezoid(np.array(new long[] { 1, 2, 3 }));
            r.ndim.Should().Be(0);
            r.dtype.Should().Be(np.float64);
            r.GetDouble(0).Should().Be(4.0);
        }

        [TestMethod]
        public void Trapezoid_Dx()
        {
            // np.trapezoid([1,2,3], dx=2) -> 8.0
            np.trapezoid(np.array(new long[] { 1, 2, 3 }), dx: 2).GetDouble(0).Should().Be(8.0);
        }

        [TestMethod]
        public void Trapezoid_XCoords()
        {
            // np.trapezoid([1,2,3], x=[4,6,8]) -> 8.0
            np.trapezoid(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 4, 6, 8 }))
                .GetDouble(0).Should().Be(8.0);
        }

        [TestMethod]
        public void Trapezoid_ReversedX_IntegratesInReverse()
        {
            // np.trapezoid([1,2,3], x=[8,6,4]) -> -8.0
            np.trapezoid(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 8, 6, 4 }))
                .GetDouble(0).Should().Be(-8.0);
        }

        // ---------------------------- dtype tier (NEP50) ----------------------------

        [TestMethod]
        public void Trapezoid_IntegerInputs_ToFloat64()
        {
            foreach (var t in new[] { np.int8, np.uint8, np.int16, np.uint16, np.int32,
                                      np.uint32, np.int64, np.uint64 })
            {
                var r = np.trapezoid(np.array(new double[] { 1, 2, 3, 4 }).astype(t));
                r.dtype.Should().Be(np.float64, $"integer {t} integrates to float64");
                r.GetDouble(0).Should().Be(7.5);
            }
        }

        [TestMethod]
        public void Trapezoid_Float32_StaysFloat32()
        {
            var r = np.trapezoid(np.array(new float[] { 1, 2, 3, 4 }));
            r.dtype.Should().Be(np.float32);
            ((double)r.GetAtIndex<float>(0)).Should().Be(7.5);
        }

        [TestMethod]
        public void Trapezoid_Float16_StaysFloat16()
        {
            var r = np.trapezoid(np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 }));
            r.dtype.Should().Be(np.float16);
            ((double)r.GetAtIndex<Half>(0)).Should().Be(7.5);
        }

        [TestMethod]
        public void Trapezoid_Bool_ToFloat64()
        {
            // bool -> float64; the bool+bool sum is logical, then *1.0/2.0 promotes.
            var r = np.trapezoid(np.array(new bool[] { true, true, true, true, true }));
            r.dtype.Should().Be(np.float64);
            r.GetDouble(0).Should().Be(2.0);
        }

        [TestMethod]
        public void Trapezoid_Complex_StaysComplex128()
        {
            var r = np.trapezoid(np.array(new System.Numerics.Complex[]
                { new(1, 1), new(2, 2), new(3, 3) }));
            r.dtype.Should().Be(np.complex128);
            r.GetAtIndex<System.Numerics.Complex>(0).Should().Be(new System.Numerics.Complex(4, 4));
        }

        [TestMethod]
        public void Trapezoid_Uint8_Overflow_WrapsBeforePromotion()
        {
            // NumPy computes y[1:]+y[:-1] in uint8 (wraps at 256) BEFORE the float promotion:
            // (200+180)%256=124, (180+150)%256=74, (150+240)%256=134; sum/2 = (124+74+134)/2 = 166.
            var r = np.trapezoid(np.array(new byte[] { 200, 180, 150, 240 }));
            r.dtype.Should().Be(np.float64);
            r.GetDouble(0).Should().Be(166.0);
        }

        // ---------------------------- axis handling ----------------------------

        [TestMethod]
        public void Trapezoid_2D_Axis0()
        {
            // np.trapezoid(arange(6).reshape(2,3), axis=0) -> [1.5, 2.5, 3.5]
            var r = np.trapezoid(np.arange(6).reshape(2, 3), axis: 0);
            r.shape.Should().Equal(3);
            r.Data<double>().Should().Equal(new[] { 1.5, 2.5, 3.5 });
        }

        [TestMethod]
        public void Trapezoid_2D_Axis1_IsDefault()
        {
            // axis=1 == default axis=-1 -> [2., 8.]
            np.trapezoid(np.arange(6).reshape(2, 3), axis: 1).Data<double>().Should().Equal(new[] { 2.0, 8.0 });
            np.trapezoid(np.arange(6).reshape(2, 3)).Data<double>().Should().Equal(new[] { 2.0, 8.0 });
        }

        [TestMethod]
        public void Trapezoid_2D_X1D_AlongAxis()
        {
            // np.trapezoid(arange(6).reshape(2,3), x=[0,2,4], axis=1) -> [4., 16.]
            var r = np.trapezoid(np.arange(6).reshape(2, 3), np.array(new long[] { 0, 2, 4 }), axis: 1);
            r.Data<double>().Should().Equal(new[] { 4.0, 16.0 });
        }

        // ---------------------------- degenerate corners ----------------------------

        [TestMethod]
        public void Trapezoid_SingleElement_IsZero()
        {
            var r = np.trapezoid(np.array(new long[] { 5 }));
            r.ndim.Should().Be(0);
            r.GetDouble(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Trapezoid_Empty_IsZero()
        {
            np.trapezoid(np.array(new double[] { })).GetDouble(0).Should().Be(0.0);
        }

        [TestMethod]
        public void Trapezoid_2D_UnitAxis_IsZeros()
        {
            // np.trapezoid(zeros((2,1))+3, axis=1) -> [0., 0.]
            var a = np.full(new Shape(2, 1), 3.0);
            np.trapezoid(a, axis: 1).Data<double>().Should().Equal(new[] { 0.0, 0.0 });
        }

        // ---------------------------- errors ----------------------------

        [TestMethod]
        public void Trapezoid_0D_Throws()
        {
            // NumPy leaks an IndexError; NumSharp raises the clearer axis-out-of-bounds error.
            Action act = () => np.trapezoid(np.array(5.0));
            act.Should().Throw<ArgumentOutOfRangeException>();
        }
    }
}
