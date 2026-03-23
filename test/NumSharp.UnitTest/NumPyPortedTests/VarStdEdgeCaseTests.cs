using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.NumPyPortedTests
{
    /// <summary>
    /// Tests ported from NumPy test_numeric.py for np.std and np.var.
    /// Covers edge cases including empty arrays, ddof, keepdims, and axis combinations.
    /// </summary>
    public class VarStdEdgeCaseTests
    {
        #region Basic std Tests (from test_std)

        [Test]
        public void Std_2DArray_Global()
        {
            // NumPy: std([[1,2,3],[4,5,6]]) = 1.707825127659933
            var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.std(A);
            var value = (double)result;
            Assert.IsTrue(Math.Abs(value - 1.707825127659933) <1e-10);
        }

        [Test]
        public void Std_2DArray_Axis0()
        {
            // NumPy: std([[1,2,3],[4,5,6]], axis=0) = [1.5, 1.5, 1.5]
            var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.std(A, axis: 0);
            result.Should().BeShaped(3);
            var data = result.GetData<double>();
            Assert.IsTrue(Math.Abs(data[0] - 1.5) <1e-10);
            Assert.IsTrue(Math.Abs(data[1] - 1.5) <1e-10);
            Assert.IsTrue(Math.Abs(data[2] - 1.5) <1e-10);
        }

        [Test]
        public void Std_2DArray_Axis1()
        {
            // NumPy: std([[1,2,3],[4,5,6]], axis=1) = [0.81649658, 0.81649658]
            var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.std(A, axis: 1);
            result.Should().BeShaped(2);
            var data = result.GetData<double>();
            Assert.IsTrue(Math.Abs(data[0] - 0.81649658) <1e-6);
            Assert.IsTrue(Math.Abs(data[1] - 0.81649658) <1e-6);
        }

        #endregion

        #region Basic var Tests (from test_var)

        [Test]
        public void Var_2DArray_Global()
        {
            // NumPy: var([[1,2,3],[4,5,6]]) = 2.9166666666666665
            var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.var(A);
            var value = (double)result;
            Assert.IsTrue(Math.Abs(value - 2.9166666666666665) <1e-10);
        }

        [Test]
        public void Var_2DArray_Axis0()
        {
            // NumPy: var([[1,2,3],[4,5,6]], axis=0) = [2.25, 2.25, 2.25]
            var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.var(A, axis: 0);
            result.Should().BeShaped(3);
            var data = result.GetData<double>();
            Assert.IsTrue(Math.Abs(data[0] - 2.25) <1e-10);
            Assert.IsTrue(Math.Abs(data[1] - 2.25) <1e-10);
            Assert.IsTrue(Math.Abs(data[2] - 2.25) <1e-10);
        }

        [Test]
        public void Var_2DArray_Axis1()
        {
            // NumPy: var([[1,2,3],[4,5,6]], axis=1) = [0.66666667, 0.66666667]
            var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result = np.var(A, axis: 1);
            result.Should().BeShaped(2);
            var data = result.GetData<double>();
            Assert.IsTrue(Math.Abs(data[0] - 0.66666667) <1e-6);
            Assert.IsTrue(Math.Abs(data[1] - 0.66666667) <1e-6);
        }

        #endregion

        #region ddof Tests (Sample vs Population)

        [Test]
        public void Std_Ddof0_PopulationStd()
        {
            // NumPy: std([1,2,3,4,5], ddof=0) = 1.4142135623730951
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.std(a, ddof: 0);
            var value = (double)result;
            Assert.IsTrue(Math.Abs(value - 1.4142135623730951) <1e-10);
        }

        [Test]
        public void Std_Ddof1_SampleStd()
        {
            // NumPy: std([1,2,3,4,5], ddof=1) = 1.5811388300841898
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.std(a, ddof: 1);
            var value = (double)result;
            Assert.IsTrue(Math.Abs(value - 1.5811388300841898) <1e-10);
        }

        [Test]
        public void Var_Ddof0_PopulationVar()
        {
            // NumPy: var([1,2,3,4,5], ddof=0) = 2.0
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.var(a, ddof: 0);
            var value = (double)result;
            Assert.AreEqual(2.0, value);
        }

        [Test]
        public void Var_Ddof1_SampleVar()
        {
            // NumPy: var([1,2,3,4,5], ddof=1) = 2.5
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.var(a, ddof: 1);
            var value = (double)result;
            Assert.AreEqual(2.5, value);
        }

        #endregion

        #region keepdims Tests

        [Test]
        public void Std_Keepdims_True()
        {
            // NumPy: std(arange(12).reshape(3,4), axis=1, keepdims=True).shape = (3, 1)
            var a = np.arange(12).reshape(3, 4);
            var result = np.std(a, axis: 1, keepdims: true);
            result.Should().BeShaped(3, 1);
        }

        [Test]
        public void Std_Keepdims_False()
        {
            // NumPy: std(arange(12).reshape(3,4), axis=1, keepdims=False).shape = (3,)
            var a = np.arange(12).reshape(3, 4);
            var result = np.std(a, axis: 1, keepdims: false);
            result.Should().BeShaped(3);
        }

        [Test]
        public void Var_Keepdims_True()
        {
            var a = np.arange(12).reshape(3, 4);
            var result = np.var(a, axis: 1, keepdims: true);
            result.Should().BeShaped(3, 1);
        }

        [Test]
        public void Var_Keepdims_False()
        {
            var a = np.arange(12).reshape(3, 4);
            var result = np.var(a, axis: 1, keepdims: false);
            result.Should().BeShaped(3);
        }

        #endregion

        #region Empty Array Tests

        [Test]
        public void Std_EmptyArray_ReturnsNaN()
        {
            // NumPy: std([]) = nan (with RuntimeWarning)
            var a = np.array(new double[0]);
            var result = np.std(a);
            Assert.IsTrue(double.IsNaN((double)result));
        }

        [Test]
        public void Var_EmptyArray_ReturnsNaN()
        {
            // NumPy: var([]) = nan (with RuntimeWarning)
            var a = np.array(new double[0]);
            var result = np.var(a);
            Assert.IsTrue(double.IsNaN((double)result));
        }

        #endregion

        #region Single Element Tests

        [Test]
        public void Std_SingleElement_ReturnsZero()
        {
            // NumPy: std([5]) = 0.0
            var a = np.array(new double[] { 5.0 });
            var result = np.std(a);
            Assert.AreEqual(0.0, (double)result);
        }

        [Test]
        public void Var_SingleElement_ReturnsZero()
        {
            // NumPy: var([5]) = 0.0
            var a = np.array(new double[] { 5.0 });
            var result = np.var(a);
            Assert.AreEqual(0.0, (double)result);
        }

        [Test]
        public void Std_SingleElement_Ddof1_ReturnsNaN()
        {
            // NumPy: std([5], ddof=1) = nan (division by zero)
            var a = np.array(new double[] { 5.0 });
            var result = np.std(a, ddof: 1);
            Assert.IsTrue(double.IsNaN((double)result));
        }

        [Test]
        public void Var_SingleElement_Ddof1_ReturnsNaN()
        {
            // NumPy: var([5], ddof=1) = nan (division by zero)
            var a = np.array(new double[] { 5.0 });
            var result = np.var(a, ddof: 1);
            Assert.IsTrue(double.IsNaN((double)result));
        }

        #endregion

        #region Dtype Handling

        [Test]
        public void Std_Int32_ReturnsFloat64()
        {
            // NumPy: std(int32 array) returns float64
            var arr = np.array(new int[] { 1, 2, 3, 4, 5 });
            var result = np.std(arr);
            Assert.AreEqual(np.float64, result.dtype);
        }

        [Test]
        public void Std_Int64_ReturnsFloat64()
        {
            var arr = np.array(new long[] { 1, 2, 3, 4, 5 });
            var result = np.std(arr);
            Assert.AreEqual(np.float64, result.dtype);
        }

        [Test]
        public void Std_Float32_ReturnsFloat32()
        {
            // NumPy: std(float32 array) returns float32
            var arr = np.array(new float[] { 1f, 2f, 3f, 4f, 5f });
            var result = np.std(arr);
            Assert.AreEqual(np.float32, result.dtype);
        }

        [Test]
        public void Std_Float64_ReturnsFloat64()
        {
            var arr = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var result = np.std(arr);
            Assert.AreEqual(np.float64, result.dtype);
        }

        #endregion

        #region 3D Array Axis Combinations

        [Test]
        public void Std_3DArray_Axis0()
        {
            // NumPy: std(arange(24).reshape(2,3,4), axis=0).shape = (3, 4)
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.std(a, axis: 0);
            result.Should().BeShaped(3, 4);
        }

        [Test]
        public void Std_3DArray_Axis1()
        {
            // NumPy: std(arange(24).reshape(2,3,4), axis=1).shape = (2, 4)
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.std(a, axis: 1);
            result.Should().BeShaped(2, 4);
        }

        [Test]
        public void Std_3DArray_Axis2()
        {
            // NumPy: std(arange(24).reshape(2,3,4), axis=2).shape = (2, 3)
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.std(a, axis: 2);
            result.Should().BeShaped(2, 3);
        }

        [Test]
        public void Std_3DArray_AxisNone()
        {
            // NumPy: std(arange(24).reshape(2,3,4), axis=None).shape = ()
            var a = np.arange(24).reshape(2, 3, 4);
            var result = np.std(a);  // axis defaults to None (flatten)
            Assert.IsTrue(result.Shape.IsScalar);
        }

        #endregion

        #region NaN in Data

        [Test]
        public void Std_WithNaN_ReturnsNaN()
        {
            // NumPy: std([1,2,nan,4,5]) = nan
            var a = np.array(new double[] { 1.0, 2.0, double.NaN, 4.0, 5.0 });
            var result = np.std(a);
            Assert.IsTrue(double.IsNaN((double)result));
        }

        [Test]
        public void Var_WithNaN_ReturnsNaN()
        {
            // NumPy: var([1,2,nan,4,5]) = nan
            var a = np.array(new double[] { 1.0, 2.0, double.NaN, 4.0, 5.0 });
            var result = np.var(a);
            Assert.IsTrue(double.IsNaN((double)result));
        }

        #endregion

        #region Negative Axis

        [Test]
        public void Std_NegativeAxis()
        {
            // NumPy: std(a, axis=-1) is same as axis=last dimension
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var result1 = np.std(a, axis: -1);
            var result2 = np.std(a, axis: 1);

            result1.Should().BeShaped(2);
            result2.Should().BeShaped(2);

            var data1 = result1.GetData<double>();
            var data2 = result2.GetData<double>();

            Assert.IsTrue(Math.Abs(data1[0] - data2[0]) <1e-10);
            Assert.IsTrue(Math.Abs(data1[1] - data2[1]) <1e-10);
        }

        #endregion

        #region All Same Values

        [Test]
        public void Std_AllSameValues_ReturnsZero()
        {
            var a = np.full(5.0, 10);  // full(fill_value, shape...)
            var result = np.std(a);
            Assert.AreEqual(0.0, (double)result);
        }

        [Test]
        public void Var_AllSameValues_ReturnsZero()
        {
            var a = np.full(5.0, 10);  // full(fill_value, shape...)
            var result = np.var(a);
            Assert.AreEqual(0.0, (double)result);
        }

        #endregion

        #region Large Array

        [Test]
        public void Std_LargeArray()
        {
            // Test numerical stability with large arrays
            var a = np.arange(10000.0);  // float64 from double overload
            var result = np.std(a);
            // std of [0, 1, ..., 9999] = sqrt((n^2-1)/12) where n=10000
            // = sqrt(99999999/12) ~ 2886.75
            var expected = Math.Sqrt((10000.0 * 10000.0 - 1.0) / 12.0);
            Assert.IsTrue(Math.Abs((double)result - expected) <1.0);
        }

        #endregion

        #region Strided Array

        [Test]
        public void Std_StridedArray()
        {
            var a = np.arange(20);
            var strided = a["::" + 2];  // [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
            var result = np.std(strided);

            // std of [0,2,4,6,8,10,12,14,16,18]
            // mean = 9, variance = ((0-9)^2 + ... + (18-9)^2)/10 = 33
            // std = sqrt(33) ~ 5.745
            var value = (double)result;
            Assert.IsTrue(value >5.0);
            Assert.IsTrue(value <6.0);
        }

        #endregion
    }
}
