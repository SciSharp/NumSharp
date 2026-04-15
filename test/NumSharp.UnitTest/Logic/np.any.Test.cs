using System;
using System.Linq;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Logic
{
    [TestClass]
    public class NpAnyTest
    {
        [TestMethod]
        public void Any1DArrayTest()
        {
            // 测试1维数组
            var arr = np.array(new int[] { 0, 1, 2 });
            var result = np.any(arr, axis: 0, keepdims: false);
            Assert.AreEqual(true, result.GetBoolean(0));
        }

        [TestMethod]
        public void Any2DArrayTest()
        {
            // 测试2维数组
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result = np.any(arr, axis: 0, keepdims: false);
            var expected = np.array(new bool[] { true, false });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [TestMethod]
        public void Any2DArrayWithAxis1Test()
        {
            // 测试2维数组，axis=1
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result = np.any(arr, axis: 1, keepdims: false);
            var expected = np.array(new bool[] { false, true });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [TestMethod]
        public void AnyWithKeepdimsTest()
        {
            // 测试keepdims参数
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result = np.any(arr, axis: 0, keepdims: true);
            // 结果形状应该是 (1, 2) 而不是 (2,)
            Assert.AreEqual(2, result.ndim);
            Assert.AreEqual(1, result.shape[0]);
            Assert.AreEqual(2, result.shape[1]);
        }

        [TestMethod]
        public void AnyWithNegativeAxisTest()
        {
            // 测试负轴
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result1 = np.any(arr, axis: 1, keepdims: false);  // axis=1
            var result2 = np.any(arr, axis: -1, keepdims: false); // axis=-1 (应该是等价的)
            Assert.IsTrue(Enumerable.SequenceEqual(result1.Data<bool>(), result2.Data<bool>()));
        }

        [TestMethod]
        public void AnyAllZerosTest()
        {
            // 测试全零数组
            var arr = np.array(new int[,] { { 0, 0 }, { 0, 0 } });
            var result = np.any(arr, axis: 0, keepdims: false);
            var expected = np.array(new bool[] { false, false });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [TestMethod]
        public void AnyAllNonZerosTest()
        {
            // 测试全非零数组
            var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var result = np.any(arr, axis: 0, keepdims: false);
            var expected = np.array(new bool[] { true, true });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [TestMethod]
        public void AnyInvalidAxisTest()
        {
            // Test invalid axis - should throw ArgumentOutOfRangeException
            var arr = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.any(arr, axis: 5, keepdims: false));
        }

        [TestMethod]
        public void AnyScalarArrayTest()
        {
            // NumPy: np.array(5) creates 0D array, np.any(a) returns True
            // NumSharp now correctly creates 0D arrays for scalars
            var arr = np.array(5);
            Assert.AreEqual(0, arr.ndim, "np.array(5) should create 0D array");

            // np.any without axis works on 0D arrays and returns bool
            var result = np.any(arr);
            Assert.AreEqual(true, result);
        }

        [TestMethod]
        public void Any0DArray_WithAxis0_ReturnsScalar()
        {
            // NumPy 2.x: np.any(0D_array, axis=0) returns 0D boolean scalar
            var arr = np.array(5);
            Assert.AreEqual(0, arr.ndim);

            var result = np.any(arr, axis: 0);
            Assert.AreEqual(0, result.ndim, "Result should be 0D");
            Assert.AreEqual(true, (bool)result);
        }

        [TestMethod]
        public void Any0DArray_WithAxisNeg1_ReturnsScalar()
        {
            // NumPy 2.x: np.any(0D_array, axis=-1) is equivalent to axis=0
            var arr = np.array(0);  // falsy value
            var result = np.any(arr, axis: -1);
            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(false, (bool)result);
        }

        [TestMethod]
        public void Any0DArray_WithInvalidAxis_Throws()
        {
            // NumPy 2.x: np.any(0D_array, axis=1) raises AxisError
            var arr = np.array(5);
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.any(arr, axis: 1));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.any(arr, axis: 2));
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.any(arr, axis: -2));
        }

        [TestMethod]
        public void Any0DArray_WithKeepdims_Returns0D()
        {
            // NumPy 2.x: keepdims=True on 0D still returns 0D (no axes to keep)
            var arr = np.array(5);
            var result = np.any(arr, axis: 0, keepdims: true);
            Assert.AreEqual(0, result.ndim, "keepdims on 0D should still be 0D");
            Assert.AreEqual(true, (bool)result);
        }

        [TestMethod]
        public void AnyNullArrayTest()
        {
            // Test null array - should throw ArgumentNullException
            NDArray arr = null;
            Assert.ThrowsException<ArgumentNullException>(() => np.any(arr, axis: 0, keepdims: false));
        }
    }
}