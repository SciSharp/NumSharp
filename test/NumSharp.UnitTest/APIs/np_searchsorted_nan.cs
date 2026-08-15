using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.APIs
{
    /// <summary>
    /// searchsorted NaN handling — validated against NumPy 2.4.2. NumPy's float comparators
    /// (numpy_tag.h: less / less_equal = !less(b,a)) implement a NaN-as-largest total order:
    /// a NaN key sorts to the END, and a NaN key never corrupts a following key via the
    /// monotonic-bound carry. Regression pins for the DirectILKernelGenerator.Search.cs fix.
    /// </summary>
    [TestClass]
    public class NpSearchsortedNaNTests
    {
        private static readonly double NaN = double.NaN;

        private static NDArray Bins() => np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 });

        private static void AssertLongs(NDArray result, params long[] expected)
        {
            Assert.AreEqual(expected.Length, (int)result.size, "size");
            var flat = result.flatten();
            for (int i = 0; i < expected.Length; i++)
                Assert.AreEqual(expected[i], flat.GetInt64((long)i), $"element {i}");
        }

        // ---- side='right' (less_equal) ----

        [TestMethod]
        public void Right_LoneNaN_GoesToEnd()
        {
            // np.searchsorted(bins, [nan], 'right') = [5]
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { NaN }), "right"), 5);
        }

        [TestMethod]
        public void Right_NaN_ThenFiniteKey_DoesNotCorrupt()
        {
            // np.searchsorted(bins, [nan,1.0], 'right') = [5,2]  (was [5,5] before the carry fix)
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { NaN, 1.0 }), "right"), 5, 2);
        }

        [TestMethod]
        public void Right_Finite_ThenNaN()
        {
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { 1.0, NaN }), "right"), 2, 5);
        }

        [TestMethod]
        public void Right_ConsecutiveNaNs_ThenFinite()
        {
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { NaN, NaN, 1.0 }), "right"), 5, 5, 2);
        }

        [TestMethod]
        public void Right_DescendingWithNaN()
        {
            // np.searchsorted(bins, [3.0,nan,1.0], 'right') = [3,5,2]
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { 3.0, NaN, 1.0 }), "right"), 3, 5, 2);
        }

        // ---- side='left' (less) ----

        [TestMethod]
        public void Left_LoneNaN_GoesToEnd()
        {
            // np.searchsorted(bins, [nan], 'left') = [5]  (was 0 before the bisect fix)
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { NaN }), "left"), 5);
        }

        [TestMethod]
        public void Left_NaN_ThenFiniteKey()
        {
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { NaN, 1.0 }), "left"), 5, 1);
        }

        [TestMethod]
        public void Left_Finite_ThenNaN()
        {
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { 1.0, NaN }), "left"), 1, 5);
        }

        [TestMethod]
        public void Left_DescendingWithNaN()
        {
            AssertLongs(np.searchsorted(Bins(), np.array(new[] { 3.0, NaN, 1.0 }), "left"), 3, 5, 1);
        }

        // ---- scalar overload path + float32 ----

        [TestMethod]
        public void ScalarNaNKey_GoesToEnd()
        {
            Assert.AreEqual(5L, np.searchsorted(Bins(), double.NaN, "right"));
            Assert.AreEqual(5L, np.searchsorted(Bins(), double.NaN, "left"));
        }

        [TestMethod]
        public void Float32_NaN_GoesToEnd()
        {
            var bins = np.array(new float[] { 0f, 1f, 2.5f, 4f, 10f });
            AssertLongs(np.searchsorted(bins, np.array(new float[] { float.NaN, 1f }), "right"), 5, 2);
        }
    }
}
