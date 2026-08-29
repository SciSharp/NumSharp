using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.interp — 1-D linear interpolation. All expected values probed from NumPy 2.4.2.
    /// Port of compiled_base.c::arr_interp (binary_search_with_guess + slope table + NaN fixup) and
    /// the _function_base_impl.py wrapper (period normalization, complex fp). Bit-exact with NumPy.
    /// </summary>
    [TestClass]
    public class InterpTests
    {
        private static readonly NDArray Xp = np.array(new[] { 1.0, 2, 3 });
        private static readonly NDArray Fp = np.array(new[] { 3.0, 2, 0 });

        [TestMethod]
        public void Basic()
        {
            var r = np.interp(np.array(new[] { 0.0, 1, 1.5, 2.72, 3.14 }), Xp, Fp);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(3.0);
            r.GetAtIndex<double>(1).Should().Be(3.0);
            r.GetAtIndex<double>(2).Should().Be(2.5);
            r.GetAtIndex<double>(3).Should().BeApproximately(0.55999999999999961, 1e-15);
            r.GetAtIndex<double>(4).Should().Be(0.0);
        }

        [TestMethod]
        public void LeftRight_Defaults_AreEndpoints()
        {
            var r = np.interp(np.array(new[] { -1.0, 5 }), Xp, Fp);
            r.GetAtIndex<double>(0).Should().Be(3.0);   // fp[0]
            r.GetAtIndex<double>(1).Should().Be(0.0);   // fp[-1]
        }

        [TestMethod]
        public void LeftRight_Custom()
        {
            var r = np.interp(np.array(new[] { -1.0, 5 }), Xp, Fp, left: -100, right: 99);
            r.GetAtIndex<double>(0).Should().Be(-100.0);
            r.GetAtIndex<double>(1).Should().Be(99.0);
        }

        [TestMethod]
        public void NaN_And_Inf_X()
        {
            var r = np.interp(np.array(new[] { double.NaN, 2.5, double.PositiveInfinity, double.NegativeInfinity }), Xp, Fp);
            double.IsNaN(r.GetAtIndex<double>(0)).Should().BeTrue();     // NaN x -> NaN
            r.GetAtIndex<double>(1).Should().Be(1.0);
            r.GetAtIndex<double>(2).Should().Be(0.0);                    // +inf > xp[-1] -> right (fp[-1])
            r.GetAtIndex<double>(3).Should().Be(3.0);                    // -inf < xp[0]  -> left  (fp[0])
        }

        [TestMethod]
        public void SinglePoint_Xp()
        {
            var r = np.interp(np.array(new[] { 0.0, 2, 5 }), np.array(new[] { 2.0 }), np.array(new[] { 9.0 }));
            r.GetAtIndex<double>(0).Should().Be(9.0);
            r.GetAtIndex<double>(1).Should().Be(9.0);
            r.GetAtIndex<double>(2).Should().Be(9.0);
        }

        [TestMethod]
        public void Period_WrapsCoordinates()
        {
            var x = np.array(new[] { -180.0, -170, -185, 185, -10, -5, 0, 365 });
            var xp = np.array(new[] { 190.0, -190, 350, -350 });
            var fp = np.array(new[] { 5.0, 10, 3, 4 });
            var r = np.interp(x, xp, fp, period: 360);
            var expected = new[] { 7.5, 5.0, 8.75, 6.25, 3.0, 3.25, 3.5, 3.75 };
            for (int i = 0; i < expected.Length; i++)
                r.GetAtIndex<double>(i).Should().BeApproximately(expected[i], 1e-12);
        }

        [TestMethod]
        public void EqualDx_UsesLeftEndpointNoDivByZero()
        {
            // dx[j] == x avoids the non-finite interpolation branch.
            var r = np.interp(np.array(new[] { 1.5 }), np.array(new[] { 1.0, 1, 2 }), np.array(new[] { 10.0, 20, 30 }));
            r.GetAtIndex<double>(0).Should().Be(25.0);
        }

        [TestMethod]
        public void ScalarX_Returns0d()
        {
            var r = np.interp((NDArray)2.5, Xp, Fp);
            r.ndim.Should().Be(0);
            r.GetAtIndex<double>(0).Should().Be(1.0);
        }

        [TestMethod]
        public void HighDimX_PreservesShape()
        {
            var x = np.array(new[,] { { 0.5, 2.5 }, { 1.5, 3.5 } });
            var r = np.interp(x, Xp, Fp);
            r.shape.Should().Equal(2, 2);
            r.GetAtIndex<double>(0).Should().Be(3.0);
            r.GetAtIndex<double>(1).Should().Be(1.0);
            r.GetAtIndex<double>(2).Should().Be(2.5);
            r.GetAtIndex<double>(3).Should().Be(0.0);
        }

        [TestMethod]
        public void IntegerInputs_CastToFloat64()
        {
            var r = np.interp(np.array(new[] { 2 }), np.array(new[] { 1, 2, 3 }), np.array(new[] { 30, 20, 0 }));
            r.typecode.Should().Be(NPTypeCode.Double);
            r.GetAtIndex<double>(0).Should().Be(20.0);
        }

        // ---------------------------------------------------------------- complex fp

        [TestMethod]
        public void ComplexFp()
        {
            var r = np.interp(np.array(new[] { 1.5, 4.0 }), np.array(new[] { 2.0, 3, 5 }),
                np.array(new[] { new Complex(0, 1), Complex.Zero, new Complex(2, 3) }));
            r.typecode.Should().Be(NPTypeCode.Complex);
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(0, 1));    // x=1.5 < xp[0] -> left = fp[0] = 1j
            r.GetAtIndex<Complex>(1).Should().Be(new Complex(1, 1.5));
        }

        [TestMethod]
        public void ComplexFp_ComplexLeft()
        {
            var r = np.interp(np.array(new[] { -1.0 }), np.array(new[] { 2.0, 3, 5 }),
                np.array(new[] { new Complex(0, 1), Complex.Zero, new Complex(2, 3) }), left: new Complex(7, 8));
            r.GetAtIndex<Complex>(0).Should().Be(new Complex(7, 8));
        }

        // ---------------------------------------------------------------- errors (verbatim NumPy text)

        [TestMethod]
        public void EmptyXp_Raises()
        {
            Action act = () => np.interp(np.array(new[] { 1.0 }), np.zeros(0, typeof(double)), np.zeros(0, typeof(double)));
            act.Should().Throw<ArgumentException>().WithMessage("array of sample points is empty");
        }

        [TestMethod]
        public void LengthMismatch_Raises()
        {
            Action act = () => np.interp(np.array(new[] { 1.0 }), np.array(new[] { 1.0, 2 }), np.array(new[] { 1.0, 2, 3 }));
            act.Should().Throw<ArgumentException>().WithMessage("fp and xp are not of the same length.");
        }

        [TestMethod]
        public void PeriodZero_Raises()
        {
            Action act = () => np.interp(np.array(new[] { 1.0 }), np.array(new[] { 1.0, 2 }), np.array(new[] { 1.0, 2 }), period: 0);
            act.Should().Throw<ArgumentException>().WithMessage("period must be a non-zero value");
        }

        [TestMethod]
        public void Xp2D_Raises()
        {
            Action act = () => np.interp(np.array(new[] { 1.0 }), np.array(new[,] { { 1.0, 2 } }), np.array(new[,] { { 1.0, 2 } }));
            act.Should().Throw<ArgumentException>().WithMessage("object too deep for desired array");
        }
    }
}
