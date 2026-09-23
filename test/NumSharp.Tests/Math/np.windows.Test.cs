using System;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     Window functions np.bartlett / np.blackman / np.hamming / np.hanning / np.kaiser and the
    ///     internal Bessel i0 that kaiser is built on. All expected values are from NumPy 2.4.2
    ///     (probed directly). bartlett is pure IEEE arithmetic and is asserted BIT-exact; the trig /
    ///     Bessel windows call the host libm (Math.Cos / Math.Exp) whose last bit is platform-specific,
    ///     so they are asserted within a tight tolerance here (the differential-fuzz "windows" tier
    ///     pins them BIT-exact on the win-amd64 CRT). Covers the docstring examples, the empty / single
    ///     / even / odd corners, dtype, symmetry, and kaiser's beta sweep across i0's Chebyshev split.
    /// </summary>
    [TestClass]
    public class NpWindowsTest
    {
        private const double Tol = 1e-12;

        private static void AssertClose(NDArray got, double[] expected)
        {
            got.dtype.Should().Be(np.float64);
            got.ndim.Should().Be(1);
            got.size.Should().Be(expected.Length);
            var g = got.Data<double>().ToArray();
            for (int i = 0; i < expected.Length; i++)
                g[i].Should().BeApproximately(expected[i], Tol * (1.0 + System.Math.Abs(expected[i])),
                    $"element {i}");
        }

        // ---------------------------- bartlett (BIT-exact, portable) ----------------------------

        [TestMethod]
        public void Bartlett_Values5()
        {
            // np.bartlett(5) -> [0. , 0.5, 1. , 0.5, 0. ]
            np.bartlett(5).Data<double>().Should().Equal(new[] { 0.0, 0.5, 1.0, 0.5, 0.0 });
        }

        [TestMethod]
        public void Bartlett_Docstring12()
        {
            // np.bartlett(12) — the docstring example, BIT-exact (pure division/add).
            np.bartlett(12).Data<double>().Should().Equal(new[]
            {
                0.0, 0.18181818181818177, 0.36363636363636365, 0.5454545454545454, 0.7272727272727273,
                0.9090909090909091, 0.9090909090909091, 0.7272727272727273, 0.5454545454545454,
                0.36363636363636365, 0.18181818181818177, 0.0
            });
        }

        [TestMethod]
        public void Bartlett_Even2_IsBothZero()
        {
            // np.bartlett(2) -> [0., 0.]
            np.bartlett(2).Data<double>().Should().Equal(new[] { 0.0, 0.0 });
        }

        // ---------------------------- hanning / hamming / blackman ----------------------------

        [TestMethod]
        public void Hanning_Values5()
        {
            AssertClose(np.hanning(5), new[] { 0.0, 0.5, 1.0, 0.5, 0.0 });
        }

        [TestMethod]
        public void Hanning_Docstring12()
        {
            AssertClose(np.hanning(12), new[]
            {
                0.0, 0.07937323358440945, 0.29229249349905684, 0.5711574191366425, 0.8274303669726426,
                0.9797464868072487, 0.9797464868072487, 0.8274303669726426, 0.5711574191366425,
                0.29229249349905684, 0.07937323358440945, 0.0
            });
        }

        [TestMethod]
        public void Hamming_Values5()
        {
            AssertClose(np.hamming(5), new[] { 0.08, 0.54, 1.0, 0.54, 0.08 });
        }

        [TestMethod]
        public void Hamming_Even2_IsEndpointFloor()
        {
            // np.hamming(2) -> [0.08, 0.08] (the raised-cosine floor a0 - a1 = 0.08)
            AssertClose(np.hamming(2), new[] { 0.08, 0.08 });
        }

        [TestMethod]
        public void Blackman_Values5()
        {
            // NumPy's tiny negative endpoint (-1.39e-17) is preserved.
            AssertClose(np.blackman(5), new[] { -1.3877787807814457e-17, 0.34, 0.9999999999999999, 0.34, -1.3877787807814457e-17 });
        }

        [TestMethod]
        public void Blackman_Docstring12()
        {
            AssertClose(np.blackman(12), new[]
            {
                -1.3877787807814457e-17, 0.032606434624560324, 0.159903634783434, 0.4143979812474828,
                0.7360451799107798, 0.9670467694337431, 0.9670467694337431, 0.7360451799107798,
                0.4143979812474828, 0.159903634783434, 0.032606434624560324, -1.3877787807814457e-17
            });
        }

        // ---------------------------- kaiser ----------------------------

        [TestMethod]
        public void Kaiser_Values5_Beta5()
        {
            AssertClose(np.kaiser(5, 5.0), new[]
            {
                0.036710892271286676, 0.5528517696991324, 1.0, 0.5528517696991324, 0.036710892271286676
            });
        }

        [TestMethod]
        public void Kaiser_Docstring12_Beta14()
        {
            // beta=14 pushes the middle argument past i0's x==8 Chebyshev split (_i0_2 branch).
            AssertClose(np.kaiser(12, 14.0), new[]
            {
                7.726866835270368e-06, 0.003460091937889429, 0.04652001885143685, 0.22973712022098938,
                0.5998853159552917, 0.9456748983666177, 0.9456748983666177, 0.5998853159552917,
                0.22973712022098938, 0.04652001885143685, 0.003460091937889429, 7.726866835270368e-06
            });
        }

        [TestMethod]
        public void Kaiser_Beta0_IsRectangular()
        {
            // beta=0 -> i0(0)/i0(0) = 1 everywhere (rectangular window).
            AssertClose(np.kaiser(3, 0.0), new[] { 1.0, 1.0, 1.0 });
            AssertClose(np.kaiser(8, 0.0), new[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });
        }

        [TestMethod]
        public void Kaiser_Even2_Beta5()
        {
            AssertClose(np.kaiser(2, 5.0), new[] { 0.036710892271286676, 0.036710892271286676 });
        }

        // ---------------------------- shared edge cases ----------------------------

        [TestMethod]
        public void AllWindows_EmptyForMLessThan1()
        {
            foreach (int M in new[] { 0, -1, -2 })
            {
                foreach (var w in new[] { np.bartlett(M), np.blackman(M), np.hamming(M), np.hanning(M), np.kaiser(M, 5.0) })
                {
                    w.size.Should().Be(0);
                    w.ndim.Should().Be(1);
                    w.dtype.Should().Be(np.float64);
                }
            }
        }

        [TestMethod]
        public void AllWindows_OnesForM1()
        {
            foreach (var w in new[] { np.bartlett(1), np.blackman(1), np.hamming(1), np.hanning(1), np.kaiser(1, 14.0) })
            {
                w.Data<double>().Should().Equal(new[] { 1.0 });
                w.size.Should().Be(1);
                w.dtype.Should().Be(np.float64);
            }
        }

        [TestMethod]
        public void AllWindows_ReturnFloat64_AndCorrectShape()
        {
            np.bartlett(7).dtype.Should().Be(np.float64);
            np.blackman(7).dtype.Should().Be(np.float64);
            np.hamming(7).dtype.Should().Be(np.float64);
            np.hanning(7).dtype.Should().Be(np.float64);
            np.kaiser(7, 8.6).dtype.Should().Be(np.float64);
            np.hanning(33).shape.Should().Equal(33);
            np.kaiser(64, 5.0).shape.Should().Equal(64);
        }

        [TestMethod]
        public void AllWindows_AreSymmetric()
        {
            foreach (var w in new[] { np.bartlett(13), np.blackman(13), np.hamming(13), np.hanning(13), np.kaiser(13, 8.6) })
            {
                var d = w.Data<double>().ToArray();
                int n = d.Length;
                for (int i = 0; i < n; i++)
                    d[i].Should().Be(d[n - 1 - i], $"window is symmetric at {i}");
            }
        }

        // ---------------------------- internal Bessel i0 (kaiser's engine) ----------------------------

        [TestMethod]
        public void BesselI0_MatchesNumpy()
        {
            // np.i0([0, 1, 2, 3.75, 5, 8, 10, 20]) — both Chebyshev branches (x<=8 and x>8).
            double[] xs = { 0.0, 1.0, 2.0, 3.75, 5.0, 8.0, 10.0, 20.0 };
            double[] exp = { 1.0, 1.2660658777520082, 2.279585302336067, 9.118945860844567,
                             27.239871823604442, 427.56411572180474, 2815.716628466254, 43558282.559553534 };
            for (int i = 0; i < xs.Length; i++)
                np.BesselI0(xs[i]).Should().BeApproximately(exp[i], 1e-9 * (1.0 + System.Math.Abs(exp[i])), $"i0({xs[i]})");

            // i0 is even.
            np.BesselI0(-3.0).Should().Be(np.BesselI0(3.0));
            np.BesselI0(0.0).Should().Be(1.0);
        }

        // ---------------------------- float-M parity (NumPy's `M: _FloatLike_co`) ----------------------------

        [TestMethod]
        public void Hanning_FloatM_YieldsFractionalLength()
        {
            // NumPy accepts a non-integer M: len == len(arange(1-M, M, 2)). hanning(5.7) -> 6 samples.
            AssertClose(np.hanning(5.7), new[]
            {
                0.0, 0.38408992486623594, 0.9462594179299406, 0.8228139257794013,
                0.2034101276353224, 0.039675056661785446
            });
        }

        [TestMethod]
        public void Bartlett_FloatM_Exact()
        {
            // np.bartlett(2.5) -> [0., 0.6666...] (pure arithmetic, BIT-exact).
            np.bartlett(2.5).Data<double>().Should().Equal(new[] { 0.0, 0.6666666666666667 });
        }

        [TestMethod]
        public void Kaiser_FractionalMBetween0And1_IsOneElement_NotEmpty()
        {
            // kaiser has NO M<1 guard: arange(0, 0.5) = [0], so kaiser(0.5, β) is ONE element,
            // not empty (the one structural difference from the cosine windows). M<=0 stays empty.
            AssertClose(np.kaiser(0.5, 5.0), new[] { 0.036710892271286676 });
            np.kaiser(0.0, 5.0).size.Should().Be(0);
            np.hanning(0.5).size.Should().Be(0);   // cosine window DOES guard M<1 -> empty
        }

        [TestMethod]
        public void IntM_And_DoubleM_AgreeForIntegerValues()
        {
            // An int caller binds via implicit int->double and gets the identical result.
            np.hanning(5).Data<double>().Should().Equal(np.hanning(5.0).Data<double>().ToArray());
            np.kaiser(12, 14.0).Data<double>().Should().Equal(np.kaiser(12.0, 14.0).Data<double>().ToArray());
        }
    }
}
