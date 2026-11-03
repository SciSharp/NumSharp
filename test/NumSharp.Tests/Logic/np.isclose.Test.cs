using System.Linq;

namespace NumSharp.Tests.Logic
{
    [TestClass]
    public class np_isclose_Test
    {
        [TestMethod]
        public void np_isclose_1D()
        {
            //>>> np.isclose([1e10, 1e-7], [1.00001e10,1e-8])
            //array([True, False])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-7}, new[] {1.00001e10, 1e-8}).Data<bool>(), new[] {true, false}));

            //>>> np.isclose([1e10, 1e-8], [1.00001e10,1e-9])
            //array([True, True])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-8}, new[] {1.00001e10, 1e-9}).Data<bool>(), new[] {true, true}));

            //>>> np.isclose([1e10, 1e-8], [1.0001e10, 1e-9])
            //array([False, True])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-8}, new[] {1.0001e10, 1e-9}).Data<bool>(), new[] {false, true}));

            //>>> np.isclose([1.0, np.nan], [1.0, np.nan])
            //array([True, False])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e10, 1e-8}, new[] {1.0001e10, 1e-9}).Data<bool>(), new[] {false, true}));

            //>>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
            //array([True, True])
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1.0, double.NaN}, new[] {1.0, double.NaN}, equal_nan: true).Data<bool>(), new[] {true, true}));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {double.NaN, 1.0}, new[] {1.0, double.NaN}, equal_nan: true).Data<bool>(), new[] {false, false}));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {double.NaN, 1.0}, new[] {1.0, double.NaN}).Data<bool>(), new[] {false, false}));

            //>>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
            //array([True, False], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-8, 1e-7}, new[] {0.0, 0.0}).Data<bool>(), new[] {true, false}));

            //>>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
            //array([False, False], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-100, 1e-7}, new[] {0.0, 0.0}, atol: 0.0).Data<bool>(), new[] {false, false}));

            //>>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
            //array([True, True], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-10, 1e-10}, new[] {1e-20, 0.0}).Data<bool>(), new[] {true, true}));

            //>>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
            //array([False, True], dtype= bool)
            Assert.IsTrue(Enumerable.SequenceEqual(np.isclose(new[] {1e-10, 1e-10}, new[] {1e-20, 0.999999e-10}, atol: 0.0).Data<bool>(), new[] {false, true}));
        }

        /// <summary>
        ///     NumPy's formula is <c>less_equal(abs(x - y), atol + rtol * abs(y)) &amp; isfinite(y) | (x == y)</c>, with no
        ///     test of the difference itself: an infinite tolerance accepts an infinite difference (<c>inf &lt;= inf</c>).
        ///     NumPy 2.4.2: <c>np.isclose([1e308, inf], [-1e308, 1.0], rtol=0, atol=inf)</c> is <c>[True, True]</c> (the
        ///     first difference overflows, the second pair has a finite reference); so is <c>rtol=inf</c>. The parity QA pin
        ///     QA-37: an extra <c>isfinite(diff)</c> term made both False.
        /// </summary>
        [TestMethod]
        public void InfiniteDifference_InfiniteTolerance_IsClose()
        {
            using var x = np.array(new[] { 1e308, double.PositiveInfinity });
            using var y = np.array(new[] { -1e308, 1.0 });

            using var byAtol = np.isclose(x, y, rtol: 0.0, atol: double.PositiveInfinity);
            using var byRtol = np.isclose(x, y, rtol: double.PositiveInfinity, atol: 0.0);

            byAtol.ToArray<bool>().Should().Equal(true, true);
            byRtol.ToArray<bool>().Should().Equal(true, true);
            np.allclose(np.array(new[] { 1e308 }), np.array(new[] { -1e308 }), rtol: 0.0, atol: double.PositiveInfinity)
                .Should().BeTrue("allclose is all(isclose(...))");
        }

        /// <summary>
        ///     A finite tolerance that overflows the computation dtype is infinite there too: float32 operands compute in
        ///     float32, so <c>atol=1e300</c> packs to inf (a weak Python float) and the float32 difference 3e38 - -3e38 is inf:
        ///     <c>np.isclose(float32 [3e38, -3e38], [-3e38, 3e38], rtol=0, atol=1e300)</c> is <c>[True, True]</c> in NumPy
        ///     2.4.2.
        /// </summary>
        [TestMethod]
        public void Float32_ToleranceOverflowingTheDtype_AcceptsAnInfiniteDifference()
        {
            using var x = np.array(new[] { 3e38f, -3e38f });
            using var y = np.array(new[] { -3e38f, 3e38f });

            using var close = np.isclose(x, y, rtol: 0.0, atol: 1e300);

            close.ToArray<bool>().Should().Equal(true, true);
        }

        /// <summary>
        ///     What the formula still refuses (NumPy 2.4.2): a NaN difference fails <c>&lt;=</c> whatever the tolerance
        ///     (<c>np.isclose([nan], [nan], atol=inf)</c> is <c>[False]</c>), an infinite reference is close only to itself
        ///     (<c>isfinite(y)</c> gates the tolerance, <c>x == y</c> answers <c>inf == inf</c>), and equal_nan joins two NaNs.
        /// </summary>
        [TestMethod]
        public void NaNAndInfiniteReference_KeepNumPysAnswers()
        {
            var nan = System.BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            using var nans = np.array(new[] { nan });
            using var infs = np.array(new[] { double.PositiveInfinity, double.PositiveInfinity });
            using var nearInf = np.array(new[] { 1e308, double.PositiveInfinity });

            np.isclose(nans, nans, atol: double.PositiveInfinity).ToArray<bool>().Should().Equal(false);
            np.isclose(nans, nans, atol: double.PositiveInfinity, equal_nan: true).ToArray<bool>().Should().Equal(true);
            np.isclose(nearInf, infs, rtol: 0.0, atol: double.PositiveInfinity).ToArray<bool>().Should().Equal(false, true);
        }
    }
}
