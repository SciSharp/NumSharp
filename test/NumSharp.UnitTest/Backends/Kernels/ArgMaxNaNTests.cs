using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels
{
    /// <summary>
    /// Comprehensive tests for ArgMax/ArgMin based on NumPy 2.4.2 behavior.
    /// </summary>
    [TestClass]
    public class ArgMaxMinTests
    {
        #region NaN Handling

        [TestMethod]
        public void ArgMax_NaN_InMiddle_ReturnsNaNIndex()
        {
            // NumPy: argmax([1.0, nan, 3.0]) = 1 (NaN wins)
            var a = np.array(new[] { 1.0, double.NaN, 3.0 });
            long result = np.argmax(a);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void ArgMax_NaN_AtStart_ReturnsNaNIndex()
        {
            // NumPy: argmax([nan, 1.0, 3.0]) = 0 (first NaN wins)
            var a = np.array(new[] { double.NaN, 1.0, 3.0 });
            long result = np.argmax(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMax_NaN_AtEnd_ReturnsNaNIndex()
        {
            // NumPy: argmax([1.0, 3.0, nan]) = 2 (NaN wins)
            var a = np.array(new[] { 1.0, 3.0, double.NaN });
            long result = np.argmax(a);

            Assert.AreEqual(2, result);
        }

        [TestMethod]
        public void ArgMax_MultipleNaN_ReturnsFirstNaNIndex()
        {
            // NumPy: argmax([nan, nan, 1.0]) = 0 (first NaN wins)
            var a = np.array(new[] { double.NaN, double.NaN, 1.0 });
            long result = np.argmax(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMin_NaN_InMiddle_ReturnsNaNIndex()
        {
            // NumPy: argmin([1.0, nan, 3.0]) = 1 (NaN wins - same as argmax!)
            var a = np.array(new[] { 1.0, double.NaN, 3.0 });
            long result = np.argmin(a);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void ArgMin_NaN_AtStart_ReturnsNaNIndex()
        {
            // NumPy: argmin([nan, 1.0, 3.0]) = 0
            var a = np.array(new[] { double.NaN, 1.0, 3.0 });
            long result = np.argmin(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMax_Inf_Vs_NaN()
        {
            // NumPy: argmax([inf, nan, 2.0]) = 1 (NaN wins over Inf)
            var a = np.array(new[] { double.PositiveInfinity, double.NaN, 2.0 });
            long result = np.argmax(a);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void ArgMin_NegInf_Vs_NaN()
        {
            // NumPy: argmin([-inf, nan, 2.0]) = 1 (NaN wins over -Inf)
            var a = np.array(new[] { double.NegativeInfinity, double.NaN, 2.0 });
            long result = np.argmin(a);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void ArgMax_Float32_NaN()
        {
            // NumPy: argmax(float32 with nan) = 1
            var a = np.array(new[] { 1.0f, float.NaN, 3.0f });
            long result = np.argmax(a);

            Assert.AreEqual(1, result);
        }

        #endregion

        #region Empty Array

        [TestMethod]
        public void ArgMax_EmptyArray_ThrowsArgumentException()
        {
            // NumPy: argmax([]) raises ValueError
            var a = np.array(Array.Empty<double>());

            Assert.ThrowsException<ArgumentException>(() => np.argmax(a));
        }

        [TestMethod]
        public void ArgMin_EmptyArray_ThrowsArgumentException()
        {
            // NumPy: argmin([]) raises ValueError
            var a = np.array(Array.Empty<double>());

            Assert.ThrowsException<ArgumentException>(() => np.argmin(a));
        }

        #endregion

        #region Boolean Support

        [TestMethod]
        public void ArgMax_Boolean_ReturnsFirstTrueIndex()
        {
            // NumPy: argmax([False, True, False, True]) = 1 (first True)
            var a = np.array(new[] { false, true, false, true });
            long result = np.argmax(a);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void ArgMin_Boolean_ReturnsFirstFalseIndex()
        {
            // NumPy: argmin([False, True, False, True]) = 0 (first False)
            var a = np.array(new[] { false, true, false, true });
            long result = np.argmin(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMax_Boolean_AllFalse_ReturnsZero()
        {
            // NumPy: argmax([False, False, False]) = 0 (first occurrence)
            var a = np.array(new[] { false, false, false });
            long result = np.argmax(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMin_Boolean_AllTrue_ReturnsZero()
        {
            // NumPy: argmin([True, True, True]) = 0 (first occurrence)
            var a = np.array(new[] { true, true, true });
            long result = np.argmin(a);

            Assert.AreEqual(0, result);
        }

        #endregion

        #region Scalar and Single Element

        [TestMethod]
        public void ArgMax_SingleElement_ReturnsZero()
        {
            // NumPy: argmax([42]) = 0
            var a = np.array(new[] { 42 });
            long result = np.argmax(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMax_Scalar_ReturnsZero()
        {
            // NumPy: argmax(scalar) = 0
            var a = NDArray.Scalar(42);
            long result = np.argmax(a);

            Assert.AreEqual(0, result);
        }

        #endregion

        #region Ties (First Occurrence)

        [TestMethod]
        public void ArgMax_Ties_ReturnsFirstOccurrence()
        {
            // NumPy: argmax([5, 1, 5, 3, 5]) = 0 (first max)
            var a = np.array(new[] { 5, 1, 5, 3, 5 });
            long result = np.argmax(a);

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void ArgMin_Ties_ReturnsFirstOccurrence()
        {
            // NumPy: argmin([5, 1, 5, 1, 5]) = 1 (first min)
            var a = np.array(new[] { 5, 1, 5, 1, 5 });
            long result = np.argmin(a);

            Assert.AreEqual(1, result);
        }

        #endregion

        #region Inf Handling

        [TestMethod]
        public void ArgMax_PositiveInf()
        {
            // NumPy: argmax([1, inf, 3]) = 1
            var a = np.array(new[] { 1.0, double.PositiveInfinity, 3.0 });
            long result = np.argmax(a);

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void ArgMin_NegativeInf()
        {
            // NumPy: argmin([1, -inf, 3]) = 1
            var a = np.array(new[] { 1.0, double.NegativeInfinity, 3.0 });
            long result = np.argmin(a);

            Assert.AreEqual(1, result);
        }

        #endregion

        #region 2D With Axis

        [TestMethod]
        public void ArgMax_2D_Axis0()
        {
            // NumPy: argmax([[1, 5, 3], [4, 2, 6]], axis=0) = [1, 0, 1]
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, 0);

            CollectionAssert.AreEqual(new long[] { 1, 0, 1 }, result.ToArray<long>());
        }

        [TestMethod]
        public void ArgMax_2D_Axis1()
        {
            // NumPy: argmax([[1, 5, 3], [4, 2, 6]], axis=1) = [1, 2]
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, 1);

            CollectionAssert.AreEqual(new long[] { 1, 2 }, result.ToArray<long>());
        }

        [TestMethod]
        public void ArgMax_2D_AxisNegative()
        {
            // NumPy: argmax([[1, 5, 3], [4, 2, 6]], axis=-1) = [1, 2] (same as axis=1)
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmax(a, -1);

            CollectionAssert.AreEqual(new long[] { 1, 2 }, result.ToArray<long>());
        }

        [TestMethod]
        public void ArgMin_2D_Axis0()
        {
            // NumPy: argmin([[1, 5, 3], [4, 2, 6]], axis=0) = [0, 1, 0]
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmin(a, 0);

            CollectionAssert.AreEqual(new long[] { 0, 1, 0 }, result.ToArray<long>());
        }

        [TestMethod]
        public void ArgMin_2D_Axis1()
        {
            // NumPy: argmin([[1, 5, 3], [4, 2, 6]], axis=1) = [0, 1]
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            var result = np.argmin(a, 1);

            CollectionAssert.AreEqual(new long[] { 0, 1 }, result.ToArray<long>());
        }

        #endregion

        #region NDArray Instance Methods

        [TestMethod]
        public void NDArray_ArgMax_Instance()
        {
            var a = np.array(new[] { 1, 5, 3 });
            long result = a.argmax();

            Assert.AreEqual(1, result);
        }

        [TestMethod]
        public void NDArray_ArgMin_Instance()
        {
            // NumPy: argmin([1, 5, 3]) = 0 (index of minimum value 1)
            var a = np.array(new[] { 1, 5, 3 });
            long result = a.argmin();

            Assert.AreEqual(0, result);
        }

        [TestMethod]
        public void NDArray_ArgMax_WithAxis_ReturnsNDArray()
        {
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            NDArray result = a.argmax(1);

            CollectionAssert.AreEqual(new long[] { 1, 2 }, result.ToArray<long>());
        }

        [TestMethod]
        public void NDArray_ArgMin_WithAxis_ReturnsNDArray()
        {
            var a = np.array(new[,] { { 1, 5, 3 }, { 4, 2, 6 } });
            NDArray result = a.argmin(1);

            CollectionAssert.AreEqual(new long[] { 0, 1 }, result.ToArray<long>());
        }

        #endregion
    }
}
