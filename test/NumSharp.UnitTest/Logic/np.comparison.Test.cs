using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Tests for np.equal, np.not_equal, np.less, np.greater, np.less_equal, np.greater_equal
    /// These are thin wrappers around operators that exist for NumPy API compatibility.
    /// </summary>
    public class np_comparison_Test
    {
        #region np.equal

        [Test]
        public void equal_ArrayArray()
        {
            // NumPy: np.equal([1, 2, 3], [1, 0, 3]) -> [True, False, True]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 0, 3 });
            var result = np.equal(a, b);

            Assert.AreEqual(new[] { 3 }, result.shape);
            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        [Test]
        public void equal_ArrayScalar()
        {
            // NumPy: np.equal([1, 2, 3], 2) -> [False, True, False]
            var a = np.array(new[] { 1, 2, 3 });
            var result = np.equal(a, 2);

            Assert.AreEqual(new[] { 3 }, result.shape);
            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        [Test]
        public void equal_ScalarArray()
        {
            // NumPy: np.equal(2, [1, 2, 3]) -> [False, True, False]
            var b = np.array(new[] { 1, 2, 3 });
            var result = np.equal(2, b);

            Assert.AreEqual(new[] { 3 }, result.shape);
            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        [Test]
        public void equal_Float64()
        {
            // NumPy: np.equal([1.0, 2.0, 3.0], [1.0, 2.1, 3.0]) -> [True, False, True]
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var b = np.array(new[] { 1.0, 2.1, 3.0 });
            var result = np.equal(a, b);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        #endregion

        #region np.not_equal

        [Test]
        public void not_equal_ArrayArray()
        {
            // NumPy: np.not_equal([1, 2, 3], [1, 0, 3]) -> [False, True, False]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 0, 3 });
            var result = np.not_equal(a, b);

            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        [Test]
        public void not_equal_ArrayScalar()
        {
            // NumPy: np.not_equal([1, 2, 3], 2) -> [True, False, True]
            var a = np.array(new[] { 1, 2, 3 });
            var result = np.not_equal(a, 2);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        #endregion

        #region np.less

        [Test]
        public void less_ArrayArray()
        {
            // NumPy: np.less([1, 2, 3], [2, 2, 2]) -> [True, False, False]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 2, 2, 2 });
            var result = np.less(a, b);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        [Test]
        public void less_ArrayScalar()
        {
            // NumPy: np.less([1, 2, 3], 2) -> [True, False, False]
            var a = np.array(new[] { 1, 2, 3 });
            var result = np.less(a, 2);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        [Test]
        public void less_ScalarArray()
        {
            // NumPy: np.less(2, [1, 2, 3]) -> [False, False, True]
            var b = np.array(new[] { 1, 2, 3 });
            var result = np.less(2, b);

            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        #endregion

        #region np.greater

        [Test]
        public void greater_ArrayArray()
        {
            // NumPy: np.greater([1, 2, 3], [2, 2, 2]) -> [False, False, True]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 2, 2, 2 });
            var result = np.greater(a, b);

            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        [Test]
        public void greater_ArrayScalar()
        {
            // NumPy: np.greater([1, 2, 3], 2) -> [False, False, True]
            var a = np.array(new[] { 1, 2, 3 });
            var result = np.greater(a, 2);

            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        #endregion

        #region np.less_equal

        [Test]
        public void less_equal_ArrayArray()
        {
            // NumPy: np.less_equal([1, 2, 3], [2, 2, 2]) -> [True, True, False]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 2, 2, 2 });
            var result = np.less_equal(a, b);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        [Test]
        public void less_equal_ArrayScalar()
        {
            // NumPy: np.less_equal([1, 2, 3], 2) -> [True, True, False]
            var a = np.array(new[] { 1, 2, 3 });
            var result = np.less_equal(a, 2);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsFalse(result.GetBoolean(2));
        }

        #endregion

        #region np.greater_equal

        [Test]
        public void greater_equal_ArrayArray()
        {
            // NumPy: np.greater_equal([1, 2, 3], [2, 2, 2]) -> [False, True, True]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 2, 2, 2 });
            var result = np.greater_equal(a, b);

            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        [Test]
        public void greater_equal_ArrayScalar()
        {
            // NumPy: np.greater_equal([1, 2, 3], 2) -> [False, True, True]
            var a = np.array(new[] { 1, 2, 3 });
            var result = np.greater_equal(a, 2);

            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        #endregion

        #region Broadcasting

        [Test]
        public void comparison_Broadcasting2D()
        {
            // NumPy: np.equal([[1, 2], [3, 4]], [1, 4]) -> [[True, False], [False, True]]
            var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new[] { 1, 4 });
            var result = np.equal(a, b);

            Assert.AreEqual(new[] { 2, 2 }, result.shape);
            Assert.IsTrue(result.GetBoolean(0, 0));
            Assert.IsFalse(result.GetBoolean(0, 1));
            Assert.IsFalse(result.GetBoolean(1, 0));
            Assert.IsTrue(result.GetBoolean(1, 1));
        }

        [Test]
        public void less_Broadcasting()
        {
            // NumPy: np.less([[1], [2], [3]], [1, 2, 3]) broadcasts to (3, 3)
            var a = np.array(new[,] { { 1 }, { 2 }, { 3 } });  // Shape (3, 1)
            var b = np.array(new[] { 1, 2, 3 });                // Shape (3,)
            var result = np.less(a, b);

            // Result shape should be (3, 3)
            Assert.AreEqual(new[] { 3, 3 }, result.shape);

            // Row 0: [1 < 1, 1 < 2, 1 < 3] = [False, True, True]
            Assert.IsFalse(result.GetBoolean(0, 0));
            Assert.IsTrue(result.GetBoolean(0, 1));
            Assert.IsTrue(result.GetBoolean(0, 2));

            // Row 1: [2 < 1, 2 < 2, 2 < 3] = [False, False, True]
            Assert.IsFalse(result.GetBoolean(1, 0));
            Assert.IsFalse(result.GetBoolean(1, 1));
            Assert.IsTrue(result.GetBoolean(1, 2));

            // Row 2: [3 < 1, 3 < 2, 3 < 3] = [False, False, False]
            Assert.IsFalse(result.GetBoolean(2, 0));
            Assert.IsFalse(result.GetBoolean(2, 1));
            Assert.IsFalse(result.GetBoolean(2, 2));
        }

        #endregion

        #region Different dtypes

        [Test]
        public void equal_MixedTypes()
        {
            // NumPy: np.equal(int32_array, float64_array) works with type promotion
            var a = np.array(new[] { 1, 2, 3 });         // int32
            var b = np.array(new[] { 1.0, 2.0, 3.0 });   // float64
            var result = np.equal(a, b);

            Assert.IsTrue(result.GetBoolean(0));
            Assert.IsTrue(result.GetBoolean(1));
            Assert.IsTrue(result.GetBoolean(2));
        }

        [Test]
        public void less_MixedTypes()
        {
            // NumPy: np.less(int32_array, float64_array)
            var a = np.array(new[] { 1, 2, 3 });         // int32
            var b = np.array(new[] { 1.5, 2.5, 2.5 });   // float64
            var result = np.less(a, b);

            Assert.IsTrue(result.GetBoolean(0));   // 1 < 1.5
            Assert.IsTrue(result.GetBoolean(1));   // 2 < 2.5
            Assert.IsFalse(result.GetBoolean(2));  // 3 < 2.5
        }

        #endregion
    }
}
