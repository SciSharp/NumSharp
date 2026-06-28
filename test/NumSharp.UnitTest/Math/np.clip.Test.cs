using System;
using System.Collections.Generic;
using System.Text;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class np_clip_test
    {
        [TestMethod]
        public void Case1()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, 3, max).Should().BeOfValues(3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Case2()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, max, null).Should().BeOfValues(8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 11).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Case3()
        {
            var a = np.arange(12).reshape(3, 4);
            var max = np.repeat(8, 12).reshape(3, 4);
            np.clip(a, null, max).Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8).And.BeShaped(3, 4);
        }

        // Regression for W6-D: np.clip PROPAGATES NaN in NumPy (clip(NaN, lo, hi) = NaN), it does NOT
        // clamp NaN to a_min. NumSharp clips via the SIMD min/max kernel whose hardware MAXPS/MINPD
        // dropped the NaN; the NaN-aware float path now restores propagation. Verified vs NumPy 2.4.2.

        private static void AssertElems(NDArray actual, double[] expected, string because)
        {
            actual.size.Should().Be(expected.Length, because);
            var f = actual.astype(NPTypeCode.Double);
            for (int i = 0; i < expected.Length; i++)
            {
                double v = f.GetDouble(i);
                if (double.IsNaN(expected[i]))
                    double.IsNaN(v).Should().BeTrue($"element {i} should be NaN ({because})");
                else
                    v.Should().Be(expected[i], $"element {i} ({because})");
            }
        }

        [TestMethod]
        public void Clip_ScalarBounds_PropagatesNaN()
        {
            double nan = double.NaN;
            var d = np.array(new double[] { nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, nan, 11 });
            // NumPy: np.clip(d, 2, 5) = [nan, 2, 2, 3, 4, 5, 5, 5, 5, 5, nan, 5]
            AssertElems(np.clip(d, (NDArray)2.0, (NDArray)5.0),
                new double[] { nan, 2, 2, 3, 4, 5, 5, 5, 5, 5, nan, 5 },
                "clip(NaN, lo, hi) must preserve NaN, not clamp to a_min");
        }

        [TestMethod]
        public void Clip_ArrayBounds_PropagatesNaN()
        {
            double nan = double.NaN;
            // NaN in the value AND in both bounds (the clip fuzz-op shape, array a_min/a_max).
            var a = np.array(new double[] { nan, 1, 5, nan, 9, 2, nan, 8, 3, 10, 0, nan });
            var lo = np.array(new double[] { 0, 2, nan, 1, 3, nan, 4, 0, nan, 5, 1, 2 });
            var hi = np.array(new double[] { 3, nan, 7, 8, nan, 6, 9, nan, 5, 8, nan, 9 });
            // NumPy: every element with a NaN in value/lo/hi -> NaN; only index 9 (10 clipped to [5,8]) -> 8.
            AssertElems(np.clip(a, lo, hi),
                new double[] { nan, nan, nan, nan, nan, nan, nan, nan, nan, 8, nan, nan },
                "clip with array bounds must propagate NaN from value or either bound");
        }

        [TestMethod]
        public void Clip_OutAlias_PropagatesNaN()
        {
            double nan = double.NaN;
            var a = np.array(new double[] { nan, 1, 5, nan, 9, 2, nan, 8, 3, 10, 0, nan });
            var lo = np.array(new double[] { 0, 2, nan, 1, 3, nan, 4, 0, nan, 5, 1, 2 });
            var hi = np.array(new double[] { 3, nan, 7, 8, nan, 6, 9, nan, 5, 8, nan, 9 });
            np.clip(a, lo, hi, a); // out = a (aliases the input)
            AssertElems(a,
                new double[] { nan, nan, nan, nan, nan, nan, nan, nan, nan, 8, nan, nan },
                "clip(a, lo, hi, out=a) must propagate NaN through the aliased out= write");
        }
    }
}
