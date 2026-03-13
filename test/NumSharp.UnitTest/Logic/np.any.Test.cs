using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Logic
{
    public class NpAnyTest
    {
        [Test]
        public void Any1DArrayTest()
        {
            // 测试1维数组
            var arr = np.array(new int[] { 0, 1, 2 });
            var result = np.any(arr, axis: 0, keepdims: false);
            Assert.AreEqual(true, result.GetBoolean(0));
        }

        [Test]
        public void Any2DArrayTest()
        {
            // 测试2维数组
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result = np.any(arr, axis: 0, keepdims: false);
            var expected = np.array(new bool[] { true, false });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [Test]
        public void Any2DArrayWithAxis1Test()
        {
            // 测试2维数组，axis=1
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result = np.any(arr, axis: 1, keepdims: false);
            var expected = np.array(new bool[] { false, true });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [Test]
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

        [Test]
        public void AnyWithNegativeAxisTest()
        {
            // 测试负轴
            var arr = np.array(new int[,] { { 0, 0 }, { 1, 0 } });
            var result1 = np.any(arr, axis: 1, keepdims: false);  // axis=1
            var result2 = np.any(arr, axis: -1, keepdims: false); // axis=-1 (应该是等价的)
            Assert.IsTrue(Enumerable.SequenceEqual(result1.Data<bool>(), result2.Data<bool>()));
        }

        [Test]
        public void AnyAllZerosTest()
        {
            // 测试全零数组
            var arr = np.array(new int[,] { { 0, 0 }, { 0, 0 } });
            var result = np.any(arr, axis: 0, keepdims: false);
            var expected = np.array(new bool[] { false, false });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [Test]
        public void AnyAllNonZerosTest()
        {
            // 测试全非零数组
            var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var result = np.any(arr, axis: 0, keepdims: false);
            var expected = np.array(new bool[] { true, true });
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), expected.Data<bool>()));
        }

        [Test]
        public void AnyInvalidAxisTest()
        {
            // Test invalid axis - should throw ArgumentOutOfRangeException
            var arr = np.array(new int[,] { { 0, 1 }, { 2, 3 } });
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.any(arr, axis: 5, keepdims: false));
        }

        [Test]
        [Misaligned]  // NumPy: np.array(5) creates 0D, NumSharp: creates 1D with shape (1,)
        public void AnyScalarArrayTest()
        {
            // NumPy: np.array(5) creates 0D array, np.any(a, axis=0) returns True
            // NumSharp: np.array(5) creates 1D array with shape (1,), np.any(a, axis=0) also returns True
            var arr = np.array(5);
            var result = np.any(arr, axis: 0, keepdims: false);
            Assert.AreEqual(true, result.GetBoolean(0));
        }

        [Test]
        public void AnyNullArrayTest()
        {
            // Test null array - should throw ArgumentNullException
            NDArray arr = null;
            Assert.ThrowsException<ArgumentNullException>(() => np.any(arr, axis: 0, keepdims: false));
        }
    }
}