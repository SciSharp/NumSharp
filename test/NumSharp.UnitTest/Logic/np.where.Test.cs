using System;
using System.Linq;
using TUnit.Core;
using NumSharp.UnitTest.Utilities;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Comprehensive tests for np.where matching NumPy 2.x behavior.
    ///
    /// NumPy signature: where(condition, x=None, y=None, /)
    /// - Single arg: returns np.nonzero(condition)
    /// - Three args: element-wise selection with broadcasting
    /// </summary>
    public class np_where_Test
    {
        #region Single Argument (nonzero equivalent)

        [Test]
        public void Where_SingleArg_1D_ReturnsIndices()
        {
            // np.where([0, 1, 0, 2, 0, 3]) -> (array([1, 3, 5]),)
            var arr = np.array(new[] { 0, 1, 0, 2, 0, 3 });
            var result = np.where(arr);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(1L, 3L, 5L);
        }

        [Test]
        public void Where_SingleArg_2D_ReturnsTupleOfIndices()
        {
            // np.where([[0, 1, 0], [2, 0, 3]]) -> (array([0, 1, 1]), array([1, 0, 2]))
            var arr = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } });
            var result = np.where(arr);

            Assert.AreEqual(2, result.Length);
            result[0].Should().BeOfValues(0L, 1L, 1L);  // row indices
            result[1].Should().BeOfValues(1L, 0L, 2L);  // col indices
        }

        [Test]
        public void Where_SingleArg_Boolean_ReturnsNonzero()
        {
            var arr = np.array(new[] { true, false, true, false, true });
            var result = np.where(arr);

            Assert.AreEqual(1, result.Length);
            result[0].Should().BeOfValues(0L, 2L, 4L);
        }

        [Test]
        public void Where_SingleArg_Empty_ReturnsEmptyIndices()
        {
            var arr = np.array(new int[0]);
            var result = np.where(arr);

            Assert.AreEqual(1, result.Length);
            Assert.AreEqual(0, result[0].size);
        }

        [Test]
        public void Where_SingleArg_AllFalse_ReturnsEmptyIndices()
        {
            var arr = np.array(new[] { false, false, false });
            var result = np.where(arr);

            Assert.AreEqual(1, result.Length);
            Assert.AreEqual(0, result[0].size);
        }

        [Test]
        public void Where_SingleArg_AllTrue_ReturnsAllIndices()
        {
            var arr = np.array(new[] { true, true, true });
            var result = np.where(arr);

            result[0].Should().BeOfValues(0L, 1L, 2L);
        }

        [Test]
        public void Where_SingleArg_3D_ReturnsTupleOfThreeArrays()
        {
            // 2x2x2 array with some non-zero elements
            var arr = np.zeros(new[] { 2, 2, 2 }, NPTypeCode.Int32);
            arr[0, 0, 1] = 1;
            arr[1, 1, 0] = 1;
            var result = np.where(arr);

            Assert.AreEqual(3, result.Length);
            result[0].Should().BeOfValues(0L, 1L);  // dim 0
            result[1].Should().BeOfValues(0L, 1L);  // dim 1
            result[2].Should().BeOfValues(1L, 0L);  // dim 2
        }

        #endregion

        #region Three Arguments (element-wise selection)

        [Test]
        public void Where_ThreeArgs_Basic_SelectsCorrectly()
        {
            // np.where(a < 5, a, 10*a) for a = arange(10)
            var a = np.arange(10);
            var result = np.where(a < 5, a, 10 * a);

            result.Should().BeOfValues(0L, 1L, 2L, 3L, 4L, 50L, 60L, 70L, 80L, 90L);
        }

        [Test]
        public void Where_ThreeArgs_BooleanCondition()
        {
            var cond = np.array(new[] { true, false, true, false });
            var x = np.array(new[] { 1, 2, 3, 4 });
            var y = np.array(new[] { 10, 20, 30, 40 });
            var result = np.where(cond, x, y);

            result.Should().BeOfValues(1, 20, 3, 40);
        }

        [Test]
        public void Where_ThreeArgs_2D()
        {
            // np.where([[True, False], [True, True]], [[1, 2], [3, 4]], [[9, 8], [7, 6]])
            var cond = np.array(new bool[,] { { true, false }, { true, true } });
            var x = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var y = np.array(new int[,] { { 9, 8 }, { 7, 6 } });
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 2);
            Assert.AreEqual(1, (int)result[0, 0]);
            Assert.AreEqual(8, (int)result[0, 1]);
            Assert.AreEqual(3, (int)result[1, 0]);
            Assert.AreEqual(4, (int)result[1, 1]);
        }

        [Test]
        public void Where_ThreeArgs_NonBoolCondition_TreatsAsTruthy()
        {
            // np.where([0, 1, 2, 0], 100, -100) -> [-100, 100, 100, -100]
            var cond = np.array(new[] { 0, 1, 2, 0 });
            var result = np.where(cond, 100, -100);

            result.Should().BeOfValues(-100, 100, 100, -100);
        }

        #endregion

        #region Scalar Arguments

        [Test]
        public void Where_ScalarX()
        {
            var cond = np.array(new[] { true, false, true, false });
            var y = np.array(new[] { 10, 20, 30, 40 });
            var result = np.where(cond, 99, y);

            result.Should().BeOfValues(99, 20, 99, 40);
        }

        [Test]
        public void Where_ScalarY()
        {
            var cond = np.array(new[] { true, false, true, false });
            var x = np.array(new[] { 1, 2, 3, 4 });
            var result = np.where(cond, x, -1);

            result.Should().BeOfValues(1, -1, 3, -1);
        }

        [Test]
        public void Where_BothScalars()
        {
            var cond = np.array(new[] { true, false, true, false });
            var result = np.where(cond, 1, 0);

            result.Should().BeOfValues(1, 0, 1, 0);
        }

        [Test]
        public void Where_ScalarFloat()
        {
            var cond = np.array(new[] { true, false });
            var result = np.where(cond, 1.5, 2.5);

            Assert.AreEqual(typeof(double), result.dtype);
            Assert.AreEqual(1.5, (double)result[0], 1e-10);
            Assert.AreEqual(2.5, (double)result[1], 1e-10);
        }

        #endregion

        #region Broadcasting

        [Test]
        public void Where_Broadcasting_ScalarY()
        {
            // np.where(a < 4, a, -1) for 3x3 array
            var arr = np.array(new int[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } });
            var result = np.where(arr < 4, arr, -1);

            result.Should().BeShaped(3, 3);
            Assert.AreEqual(0, (int)result[0, 0]);
            Assert.AreEqual(1, (int)result[0, 1]);
            Assert.AreEqual(2, (int)result[0, 2]);
            Assert.AreEqual(-1, (int)result[1, 2]);
            Assert.AreEqual(-1, (int)result[2, 2]);
        }

        [Test]
        public void Where_Broadcasting_DifferentShapes()
        {
            // cond: (2,1), x: (3,), y: (1,3) -> result: (2,3)
            var cond = np.array(new bool[,] { { true }, { false } });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new int[,] { { 10, 20, 30 } });
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 3);
            // Row 0: cond=True, so x values
            Assert.AreEqual(1, (int)result[0, 0]);
            Assert.AreEqual(2, (int)result[0, 1]);
            Assert.AreEqual(3, (int)result[0, 2]);
            // Row 1: cond=False, so y values
            Assert.AreEqual(10, (int)result[1, 0]);
            Assert.AreEqual(20, (int)result[1, 1]);
            Assert.AreEqual(30, (int)result[1, 2]);
        }

        [Test]
        public void Where_Broadcasting_ColumnVector()
        {
            // cond: (3,1), x: scalar, y: (1,4) -> result: (3,4)
            var cond = np.array(new bool[,] { { true }, { false }, { true } });
            var x = 1;
            var y = np.array(new int[,] { { 10, 20, 30, 40 } });
            var result = np.where(cond, x, y);

            result.Should().BeShaped(3, 4);
            // Row 0: all 1s
            for (int j = 0; j < 4; j++)
                Assert.AreEqual(1, (int)result[0, j]);
            // Row 1: y values
            Assert.AreEqual(10, (int)result[1, 0]);
            Assert.AreEqual(40, (int)result[1, 3]);
            // Row 2: all 1s
            for (int j = 0; j < 4; j++)
                Assert.AreEqual(1, (int)result[2, j]);
        }

        #endregion

        #region Type Promotion

        [Test]
        public void Where_TypePromotion_IntFloat_ReturnsFloat64()
        {
            var cond = np.array(new[] { true, false });
            var result = np.where(cond, 1, 2.5);

            Assert.AreEqual(typeof(double), result.dtype);
            Assert.AreEqual(1.0, (double)result[0], 1e-10);
            Assert.AreEqual(2.5, (double)result[1], 1e-10);
        }

        [Test]
        public void Where_TypePromotion_Int32Int64_ReturnsInt64()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new int[] { 1 });
            var y = np.array(new long[] { 2 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(long), result.dtype);
        }

        [Test]
        public void Where_TypePromotion_FloatDouble_ReturnsDouble()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new float[] { 1.5f });
            var y = np.array(new double[] { 2.5 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(double), result.dtype);
        }

        #endregion

        #region Edge Cases

        [Test]
        public void Where_EmptyArrays_ThreeArgs()
        {
            var cond = np.array(new bool[0]);
            var x = np.array(new int[0]);
            var y = np.array(new int[0]);
            var result = np.where(cond, x, y);

            Assert.AreEqual(0, result.size);
        }

        [Test]
        public void Where_SingleElement()
        {
            var cond = np.array(new[] { true });
            var result = np.where(cond, 42, 0);

            Assert.AreEqual(1, result.size);
            Assert.AreEqual(typeof(int), result.dtype);  // same-type scalars preserve type
            Assert.AreEqual(42, (int)result[0]);
        }

        [Test]
        public void Where_AllTrue_ReturnsAllX()
        {
            var cond = np.array(new[] { true, true, true });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });
            var result = np.where(cond, x, y);

            result.Should().BeOfValues(1, 2, 3);
        }

        [Test]
        public void Where_AllFalse_ReturnsAllY()
        {
            var cond = np.array(new[] { false, false, false });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });
            var result = np.where(cond, x, y);

            result.Should().BeOfValues(10, 20, 30);
        }

        [Test]
        public void Where_LargeArray()
        {
            var size = 100000;
            var cond = np.arange(size) % 2 == 0;  // alternating True/False
            var x = np.ones(size, NPTypeCode.Int32);
            var y = np.zeros(size, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            Assert.AreEqual(size, result.size);
            // Even indices should be 1, odd should be 0
            Assert.AreEqual(1, (int)result[0]);
            Assert.AreEqual(0, (int)result[1]);
            Assert.AreEqual(1, (int)result[2]);
        }

        #endregion

        #region NumPy Output Verification

        [Test]
        public void Where_NumPyExample1()
        {
            // From NumPy docs: np.where([[True, False], [True, True]],
            //                          [[1, 2], [3, 4]], [[9, 8], [7, 6]])
            // Expected: array([[1, 8], [3, 4]])
            var cond = np.array(new bool[,] { { true, false }, { true, true } });
            var x = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var y = np.array(new int[,] { { 9, 8 }, { 7, 6 } });
            var result = np.where(cond, x, y);

            Assert.AreEqual(1, (int)result[0, 0]);
            Assert.AreEqual(8, (int)result[0, 1]);
            Assert.AreEqual(3, (int)result[1, 0]);
            Assert.AreEqual(4, (int)result[1, 1]);
        }

        [Test]
        public void Where_NumPyExample2()
        {
            // From NumPy docs: np.where(a < 5, a, 10*a) for a = arange(10)
            // Expected: array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
            var a = np.arange(10);
            var result = np.where(a < 5, a, 10 * a);

            result.Should().BeOfValues(0L, 1L, 2L, 3L, 4L, 50L, 60L, 70L, 80L, 90L);
        }

        [Test]
        public void Where_NumPyExample3()
        {
            // From NumPy docs: np.where(a < 4, a, -1) for specific array
            // Expected: array([[ 0,  1,  2], [ 0,  2, -1], [ 0,  3, -1]])
            var a = np.array(new int[,] { { 0, 1, 2 }, { 0, 2, 4 }, { 0, 3, 6 } });
            var result = np.where(a < 4, a, -1);

            Assert.AreEqual(0, (int)result[0, 0]);
            Assert.AreEqual(1, (int)result[0, 1]);
            Assert.AreEqual(2, (int)result[0, 2]);
            Assert.AreEqual(0, (int)result[1, 0]);
            Assert.AreEqual(2, (int)result[1, 1]);
            Assert.AreEqual(-1, (int)result[1, 2]);
            Assert.AreEqual(0, (int)result[2, 0]);
            Assert.AreEqual(3, (int)result[2, 1]);
            Assert.AreEqual(-1, (int)result[2, 2]);
        }

        #endregion

        #region Dtype Coverage

        [Test]
        public void Where_Dtype_Byte()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new byte[] { 1, 2 });
            var y = np.array(new byte[] { 10, 20 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(byte), result.dtype);
            result.Should().BeOfValues((byte)1, (byte)20);
        }

        [Test]
        public void Where_Dtype_Int16()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new short[] { 1, 2 });
            var y = np.array(new short[] { 10, 20 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(short), result.dtype);
            result.Should().BeOfValues((short)1, (short)20);
        }

        [Test]
        public void Where_Dtype_Int32()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new int[] { 1, 2 });
            var y = np.array(new int[] { 10, 20 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(int), result.dtype);
            result.Should().BeOfValues(1, 20);
        }

        [Test]
        public void Where_Dtype_Int64()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new long[] { 1, 2 });
            var y = np.array(new long[] { 10, 20 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(long), result.dtype);
            result.Should().BeOfValues(1L, 20L);
        }

        [Test]
        public void Where_Dtype_Single()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new float[] { 1.5f, 2.5f });
            var y = np.array(new float[] { 10.5f, 20.5f });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(float), result.dtype);
            Assert.AreEqual(1.5f, (float)result[0], 1e-6f);
            Assert.AreEqual(20.5f, (float)result[1], 1e-6f);
        }

        [Test]
        public void Where_Dtype_Double()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new double[] { 1.5, 2.5 });
            var y = np.array(new double[] { 10.5, 20.5 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(double), result.dtype);
            Assert.AreEqual(1.5, (double)result[0], 1e-10);
            Assert.AreEqual(20.5, (double)result[1], 1e-10);
        }

        [Test]
        public void Where_Dtype_Boolean()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new bool[] { true, true });
            var y = np.array(new bool[] { false, false });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(bool), result.dtype);
            Assert.IsTrue((bool)result[0]);
            Assert.IsFalse((bool)result[1]);
        }

        #endregion
    }
}
