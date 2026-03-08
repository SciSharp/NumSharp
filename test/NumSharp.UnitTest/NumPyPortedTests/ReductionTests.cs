using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.NumPyPortedTests;

/// <summary>
/// Attribute to track which NumPy test file and test function a test was ported from.
/// </summary>
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Class)]
public class NumPyPortedAttribute : Attribute
{
    /// <summary>
    /// The source test file and function, e.g. "test_numeric.py::test_sum"
    /// </summary>
    public string Source { get; }

    public NumPyPortedAttribute(string source) => Source = source;
}

/// <summary>
/// Tests ported from NumPy's test suite for reduction operations.
/// These tests verify NumSharp's behavior matches NumPy 2.4.2.
///
/// Source files:
/// - numpy/_core/tests/test_numeric.py
/// - numpy/_core/tests/test_multiarray.py
/// - numpy/_core/tests/test_umath.py
/// </summary>
public class ReductionTests : TestClass
{
    #region np.sum tests

    [Test]
    [NumPyPorted("test_numeric.py::test_sum")]
    public void Sum_Axis1_Keepdims()
    {
        // Python: np.sum([[1,2,3],[4,5,6],[7,8,9]], axis=1, keepdims=True)
        // Returns: [[6], [15], [24]]
        var m = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.sum(m, axis: 1, keepdims: true);

        result.shape.Should().ContainInOrder(3, 1);
        // NumPy 2.x: sum of int32 returns int64
        result.GetInt64(0, 0).Should().Be(6);
        result.GetInt64(1, 0).Should().Be(15);
        result.GetInt64(2, 0).Should().Be(24);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_sum")]
    public void Sum_Axis0()
    {
        // Python: np.sum([[1,2,3],[4,5,6],[7,8,9]], axis=0)
        // Returns: [12, 15, 18]
        var m = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.sum(m, axis: 0);

        result.shape.Should().ContainInOrder(3);
        result.GetInt64(0).Should().Be(12);
        result.GetInt64(1).Should().Be(15);
        result.GetInt64(2).Should().Be(18);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_sum")]
    public void Sum_NoAxis()
    {
        // Python: np.sum([[1,2,3],[4,5,6],[7,8,9]])
        // Returns: 45
        var m = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        var result = np.sum(m);

        result.Shape.IsScalar.Should().BeTrue();
        result.GetInt64(0).Should().Be(45);
    }

    [Test]
    [NumPyPorted("test_numeric.py")]
    public void Sum_EmptyArray()
    {
        // Python: np.sum([])
        // Returns: 0.0
        var result = np.sum(np.array(new double[0]));

        result.Shape.IsScalar.Should().BeTrue();
        result.GetDouble(0).Should().Be(0.0);
    }

    [Test]
    [NumPyPorted("test_numeric.py")]
    public void Sum_Int32_Returns_Int64()
    {
        // NumPy 2.x (NEP50): sum of int32 returns int64
        var arr = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(arr);

        result.dtype.Should().Be<long>();
    }

    #endregion

    #region np.prod tests

    [Test]
    [NumPyPorted("test_numeric.py::test_prod")]
    public void Prod_AxisNeg1()
    {
        // Python: np.prod([[1,2,3,4],[5,6,7,9],[10,3,4,5]], axis=-1)
        // Returns: [24, 1890, 600]
        var arr = np.array(new int[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 9 }, { 10, 3, 4, 5 } });
        var result = np.prod(arr, axis: -1);

        result.shape.Should().ContainInOrder(3);
        result.GetInt64(0).Should().Be(24);
        result.GetInt64(1).Should().Be(1890);
        result.GetInt64(2).Should().Be(600);
    }

    [Test]
    [NumPyPorted("test_numeric.py")]
    public void Prod_EmptyArray()
    {
        // Python: np.prod([])
        // Returns: 1.0 (multiplicative identity)
        var result = np.prod(np.array(new double[0]));

        result.Shape.IsScalar.Should().BeTrue();
        result.GetDouble(0).Should().Be(1.0);
    }

    #endregion

    #region np.mean tests

    [Test]
    [NumPyPorted("test_numeric.py::test_mean")]
    public void Mean_2DArray()
    {
        // Python: np.mean([[1,2,3],[4,5,6]])
        // Returns: 3.5
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.mean(A);

        result.GetDouble(0).Should().BeApproximately(3.5, 1e-10);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_mean")]
    public void Mean_Axis0()
    {
        // Python: np.mean([[1,2,3],[4,5,6]], axis=0)
        // Returns: [2.5, 3.5, 4.5]
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.mean(A, axis: 0);

        result.shape.Should().ContainInOrder(3);
        result.GetDouble(0).Should().BeApproximately(2.5, 1e-10);
        result.GetDouble(1).Should().BeApproximately(3.5, 1e-10);
        result.GetDouble(2).Should().BeApproximately(4.5, 1e-10);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_mean")]
    public void Mean_Axis1()
    {
        // Python: np.mean([[1,2,3],[4,5,6]], axis=1)
        // Returns: [2.0, 5.0]
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.mean(A, axis: 1);

        result.shape.Should().ContainInOrder(2);
        result.GetDouble(0).Should().BeApproximately(2.0, 1e-10);
        result.GetDouble(1).Should().BeApproximately(5.0, 1e-10);
    }

    #endregion

    #region np.std tests

    [Test]
    [NumPyPorted("test_numeric.py::test_std")]
    public void Std_2DArray()
    {
        // Python: np.std([[1,2,3],[4,5,6]])
        // Returns: 1.707825127659933
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.std(A);

        result.GetDouble(0).Should().BeApproximately(1.707825127659933, 1e-10);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_std")]
    public void Std_Axis0()
    {
        // Python: np.std([[1,2,3],[4,5,6]], axis=0)
        // Returns: [1.5, 1.5, 1.5]
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.std(A, axis: 0);

        result.shape.Should().ContainInOrder(3);
        result.GetDouble(0).Should().BeApproximately(1.5, 1e-10);
        result.GetDouble(1).Should().BeApproximately(1.5, 1e-10);
        result.GetDouble(2).Should().BeApproximately(1.5, 1e-10);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_std")]
    public void Std_Axis1()
    {
        // Python: np.std([[1,2,3],[4,5,6]], axis=1)
        // Returns: [0.816496580927726, 0.816496580927726]
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.std(A, axis: 1);

        result.shape.Should().ContainInOrder(2);
        result.GetDouble(0).Should().BeApproximately(0.816496580927726, 1e-10);
        result.GetDouble(1).Should().BeApproximately(0.816496580927726, 1e-10);
    }

    #endregion

    #region np.var tests

    [Test]
    [NumPyPorted("test_numeric.py::test_var")]
    public void Var_2DArray()
    {
        // Python: np.var([[1,2,3],[4,5,6]])
        // Returns: 2.9166666666666665
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.var(A);

        result.GetDouble(0).Should().BeApproximately(2.9166666666666665, 1e-10);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_var")]
    public void Var_Axis0()
    {
        // Python: np.var([[1,2,3],[4,5,6]], axis=0)
        // Returns: [2.25, 2.25, 2.25]
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.var(A, axis: 0);

        result.shape.Should().ContainInOrder(3);
        result.GetDouble(0).Should().BeApproximately(2.25, 1e-10);
        result.GetDouble(1).Should().BeApproximately(2.25, 1e-10);
        result.GetDouble(2).Should().BeApproximately(2.25, 1e-10);
    }

    [Test]
    [NumPyPorted("test_numeric.py::test_var")]
    public void Var_Axis1()
    {
        // Python: np.var([[1,2,3],[4,5,6]], axis=1)
        // Returns: [0.6666666666666666, 0.6666666666666666]
        var A = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var result = np.var(A, axis: 1);

        result.shape.Should().ContainInOrder(2);
        result.GetDouble(0).Should().BeApproximately(0.6666666666666666, 1e-10);
        result.GetDouble(1).Should().BeApproximately(0.6666666666666666, 1e-10);
    }

    #endregion

    #region np.argmax tests (no axis - returns long)

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_BasicSigned()
    {
        // Python: np.argmax([1, 2, 3, 4, -4, -3, -2, -1])
        // Returns: 3
        var a = np.array(new int[] { 1, 2, 3, 4, -4, -3, -2, -1 });
        var result = np.argmax(a);

        result.Should().Be(3L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_MaxAtStart()
    {
        // Python: np.argmax([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        // Returns: 0 (first occurrence)
        var a = np.array(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 });
        var result = np.argmax(a);

        result.Should().Be(0L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_MaxAtEnd()
    {
        // Python: np.argmax([0,1,2,3,4,5,6,7])
        // Returns: 7
        var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });
        var result = np.argmax(a);

        result.Should().Be(7L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax::test_maximum_signed_integers")]
    public void Argmax_Int32_MaxValue()
    {
        // Python: np.argmax(np.array([1, 2147483647, -2147483648], dtype=np.int32))
        // Returns: 1
        var a = np.array(new int[] { 1, int.MaxValue, int.MinValue });
        var result = np.argmax(a);

        result.Should().Be(1L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax::test_maximum_signed_integers")]
    public void Argmax_Int64_MaxValue()
    {
        // Python: np.argmax(np.array([1, 9223372036854775807, -9223372036854775808], dtype=np.int64))
        // Returns: 1
        var a = np.array(new long[] { 1, long.MaxValue, long.MinValue });
        var result = np.argmax(a);

        result.Should().Be(1L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_Boolean_TrueAtEnd()
    {
        // Python: np.argmax([False, False, False, False, True])
        // Returns: 4
        var a = np.array(new bool[] { false, false, false, false, true });
        var result = np.argmax(a);

        result.Should().Be(4L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_Boolean_TrueAtStart()
    {
        // Python: np.argmax([True, False, False, False, False])
        // Returns: 0
        var a = np.array(new bool[] { true, false, false, false, false });
        var result = np.argmax(a);

        result.Should().Be(0L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_Float_NaN_AtEnd()
    {
        // Python: np.argmax([0., 1., 2., 3., np.nan])
        // Returns: 4 (NaN returns its index)
        var a = np.array(new double[] { 0, 1, 2, 3, double.NaN });
        var result = np.argmax(a);

        result.Should().Be(4L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_Float_NaN_InMiddle()
    {
        // Python: np.argmax([0., 1., 2., np.nan, 3.])
        // Returns: 3
        var a = np.array(new double[] { 0, 1, 2, double.NaN, 3 });
        var result = np.argmax(a);

        result.Should().Be(3L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmax")]
    public void Argmax_Float_NaN_AtStart()
    {
        // Python: np.argmax([np.nan, 0., 1., 2., 3.])
        // Returns: 0
        var a = np.array(new double[] { double.NaN, 0, 1, 2, 3 });
        var result = np.argmax(a);

        result.Should().Be(0L);
    }

    #endregion

    #region np.argmin tests (no axis - returns long)

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_BasicSigned()
    {
        // Python: np.argmin([1, 2, 3, 4, -4, -3, -2, -1])
        // Returns: 4
        var a = np.array(new int[] { 1, 2, 3, 4, -4, -3, -2, -1 });
        var result = np.argmin(a);

        result.Should().Be(4L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_MinNotAtStart()
    {
        // Python: np.argmin([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0])
        // Returns: 8 (first occurrence of min)
        var a = np.array(new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 });
        var result = np.argmin(a);

        result.Should().Be(8L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_MinAtStart()
    {
        // Python: np.argmin([0,1,2,3,4,5,6,7])
        // Returns: 0
        var a = np.array(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });
        var result = np.argmin(a);

        result.Should().Be(0L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin::test_minimum_signed_integers")]
    public void Argmin_Int64_MinValue()
    {
        // Python: np.argmin(np.array([1, -9223372036854775808, -9223372036854775807], dtype=np.int64))
        // Returns: 1
        var a = np.array(new long[] { 1, long.MinValue, long.MinValue + 1 });
        var result = np.argmin(a);

        result.Should().Be(1L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_Boolean_FalseAtEnd()
    {
        // Python: np.argmin([True, True, True, True, False])
        // Returns: 4
        var a = np.array(new bool[] { true, true, true, true, false });
        var result = np.argmin(a);

        result.Should().Be(4L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_Boolean_FalseAtStart()
    {
        // Python: np.argmin([False, True, True, True, True])
        // Returns: 0
        var a = np.array(new bool[] { false, true, true, true, true });
        var result = np.argmin(a);

        result.Should().Be(0L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_Float_NaN_AtEnd()
    {
        // Python: np.argmin([0., 1., 2., 3., np.nan])
        // Returns: 4 (NaN returns its index)
        var a = np.array(new double[] { 0, 1, 2, 3, double.NaN });
        var result = np.argmin(a);

        result.Should().Be(4L);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmin")]
    public void Argmin_Float_NaN_AtStart()
    {
        // Python: np.argmin([np.nan, 0., 1., 2., 3.])
        // Returns: 0
        var a = np.array(new double[] { double.NaN, 0, 1, 2, 3 });
        var result = np.argmin(a);

        result.Should().Be(0L);
    }

    #endregion

    #region np.argmax/argmin with axis (returns NDArray)

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmaxArgminCommon::test_np_argmin_argmax_keepdims")]
    public void Argmax_WithAxis_Keepdims_Shape()
    {
        // Python: np.argmax([[1,2],[3,4]], axis=0, keepdims=True).shape
        // Returns: (1, 2)
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.argmax(a, axis: 0, keepdims: true);

        result.shape.Should().ContainInOrder(1, 2);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmaxArgminCommon")]
    public void Argmax_WithAxis_2D()
    {
        // Python: np.argmax([[1,2],[3,4]], axis=0)
        // Returns: [1, 1] (row 1 has max in both columns)
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.argmax(a, axis: 0);

        result.shape.Should().ContainInOrder(2);
        result.GetInt64(0).Should().Be(1);
        result.GetInt64(1).Should().Be(1);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestArgmaxArgminCommon")]
    public void Argmin_WithAxis_2D()
    {
        // Python: np.argmin([[1,2],[3,4]], axis=0)
        // Returns: [0, 0] (row 0 has min in both columns)
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.argmin(a, axis: 0);

        result.shape.Should().ContainInOrder(2);
        result.GetInt64(0).Should().Be(0);
        result.GetInt64(1).Should().Be(0);
    }

    #endregion

    #region np.max / np.min tests

    [Test]
    [NumPyPorted("test_multiarray.py::TestMinMax::test_scalar")]
    public void Max_Scalar_Axis0()
    {
        // Python: np.amax(1, axis=0)
        // Returns: 1
        var result = np.amax(np.array(1), axis: 0);

        result.GetInt32(0).Should().Be(1);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestMinMax::test_scalar")]
    public void Min_Scalar_AxisNone()
    {
        // Python: np.amin(1, axis=None)
        // Returns: 1
        var result = np.amin(np.array(1), axis: null);

        result.GetInt32(0).Should().Be(1);
    }

    [Test]
    [NumPyPorted("test_multiarray.py::TestMinMax::test_axis")]
    public void Max_2D_Axis1()
    {
        // Python: np.amax([[1, 2, 3]], axis=1)
        // Returns: [3]
        var a = np.array(new int[,] { { 1, 2, 3 } });
        var result = np.amax(a, axis: 1);

        result.shape.Should().ContainInOrder(1);
        result.GetInt32(0).Should().Be(3);
    }

    [Test]
    [NumPyPorted("test_umath.py::TestMaximum::test_reduce")]
    public void Max_Float_WithNaN()
    {
        // Python: np.max([0., np.nan, 1.])
        // Returns: nan (NaN propagates)
        var a = np.array(new double[] { 0, double.NaN, 1 });
        var result = np.amax(a);

        double.IsNaN(result.GetDouble(0)).Should().BeTrue();
    }

    [Test]
    [NumPyPorted("test_umath.py::TestMinimum::test_reduce")]
    public void Min_Float_WithNaN()
    {
        // Python: np.min([0., np.nan, 1.])
        // Returns: nan (NaN propagates)
        var a = np.array(new double[] { 0, double.NaN, 1 });
        var result = np.amin(a);

        double.IsNaN(result.GetDouble(0)).Should().BeTrue();
    }

    #endregion

    #region np.all tests (no axis - returns bool)

    [Test]
    [NumPyPorted("test_logical")]
    public void All_AllTrue()
    {
        // Python: np.all([True, True, True])
        // Returns: True
        var result = np.all(np.array(new bool[] { true, true, true }));

        result.Should().BeTrue();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void All_OneFalse()
    {
        // Python: np.all([True, False, True])
        // Returns: False
        var result = np.all(np.array(new bool[] { true, false, true }));

        result.Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void All_EmptyArray()
    {
        // Python: np.all([])
        // Returns: True (vacuous truth)
        var result = np.all(np.array(new bool[0]));

        result.Should().BeTrue();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void All_IntegersAllNonZero()
    {
        // Python: np.all([1, 2, 3])
        // Returns: True (non-zero = True)
        var result = np.all(np.array(new int[] { 1, 2, 3 }));

        result.Should().BeTrue();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void All_IntegersWithZero()
    {
        // Python: np.all([1, 0, 3])
        // Returns: False (zero = False)
        var result = np.all(np.array(new int[] { 1, 0, 3 }));

        result.Should().BeFalse();
    }

    #endregion

    #region np.all tests (with axis - returns NDArray<bool>)

    [Test]
    [NumPyPorted("test_logical")]
    public void All_Axis0()
    {
        // Python: np.all([[True, False], [True, True]], axis=0)
        // Returns: [True, False]
        var a = np.array(new bool[,] { { true, false }, { true, true } });
        var result = np.all(a, axis: 0);

        result.shape.Should().ContainInOrder(2);
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void All_Axis0_Keepdims()
    {
        // Python: np.all([[True, False], [True, True]], axis=0, keepdims=True)
        // Returns: [[True, False]]
        var a = np.array(new bool[,] { { true, false }, { true, true } });
        var result = np.all(a, axis: 0, keepdims: true);

        result.shape.Should().ContainInOrder(1, 2);
        result.GetBoolean(0, 0).Should().BeTrue();
        result.GetBoolean(0, 1).Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void All_Axis1_Keepdims()
    {
        // Python: np.all([[True, False], [True, True]], axis=1, keepdims=True)
        // Returns: [[False], [True]]
        var a = np.array(new bool[,] { { true, false }, { true, true } });
        var result = np.all(a, axis: 1, keepdims: true);

        result.shape.Should().ContainInOrder(2, 1);
        result.GetBoolean(0, 0).Should().BeFalse();
        result.GetBoolean(1, 0).Should().BeTrue();
    }

    #endregion

    #region np.any tests (no axis - returns bool)

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_AllFalse()
    {
        // Python: np.any([False, False, False])
        // Returns: False
        var result = np.any(np.array(new bool[] { false, false, false }));

        result.Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_OneTrue()
    {
        // Python: np.any([False, True, False])
        // Returns: True
        var result = np.any(np.array(new bool[] { false, true, false }));

        result.Should().BeTrue();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_EmptyArray()
    {
        // Python: np.any([])
        // Returns: False
        var result = np.any(np.array(new bool[0]));

        result.Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_IntegersAllZero()
    {
        // Python: np.any([0, 0, 0])
        // Returns: False
        var result = np.any(np.array(new int[] { 0, 0, 0 }));

        result.Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_IntegersOneNonZero()
    {
        // Python: np.any([0, 1, 0])
        // Returns: True
        var result = np.any(np.array(new int[] { 0, 1, 0 }));

        result.Should().BeTrue();
    }

    #endregion

    #region np.any tests (with axis - returns NDArray<bool>)

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_Axis0()
    {
        // Python: np.any([[True, False], [False, False]], axis=0)
        // Returns: [True, False]
        var a = np.array(new bool[,] { { true, false }, { false, false } });
        var result = np.any(a, axis: 0);

        result.shape.Should().ContainInOrder(2);
        result.GetBoolean(0).Should().BeTrue();
        result.GetBoolean(1).Should().BeFalse();
    }

    [Test]
    [NumPyPorted("test_logical")]
    public void Any_Axis0_Keepdims()
    {
        // Python: np.any([[True, False], [True, True]], axis=0, keepdims=True)
        // Returns: [[True, True]]
        var a = np.array(new bool[,] { { true, false }, { true, true } });
        var result = np.any(a, axis: 0, keepdims: true);

        result.shape.Should().ContainInOrder(1, 2);
        result.GetBoolean(0, 0).Should().BeTrue();
        result.GetBoolean(0, 1).Should().BeTrue();
    }

    #endregion

    #region keepdims tests

    [Test]
    [NumPyPorted("test_ufunc.py::test_keepdims_argument")]
    public void Sum_Keepdims_Shape()
    {
        // Python: np.sum([[1,2],[3,4]], axis=0, keepdims=True).shape
        // Returns: (1, 2)
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(a, axis: 0, keepdims: true);

        result.shape.Should().ContainInOrder(1, 2);
    }

    [Test]
    [NumPyPorted("test_ufunc.py::test_keepdims_argument")]
    public void Sum_NoKeepdims_Shape()
    {
        // Python: np.sum([[1,2],[3,4]], axis=0, keepdims=False).shape
        // Returns: (2,)
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(a, axis: 0, keepdims: false);

        result.shape.Should().ContainInOrder(2);
    }

    [Test]
    [NumPyPorted("test_ufunc.py::test_keepdims_argument")]
    public void Max_Keepdims()
    {
        // Python: np.max([[1,2],[3,4]], axis=0, keepdims=True)
        // Returns: [[3, 4]]
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.amax(a, axis: 0, keepdims: true);

        result.shape.Should().ContainInOrder(1, 2);
        result.GetInt32(0, 0).Should().Be(3);
        result.GetInt32(0, 1).Should().Be(4);
    }

    #endregion

    #region 3D array tests

    [Test]
    [NumPyPorted("test_numeric.py")]
    public void Sum_3D_Axis0()
    {
        // Python: np.sum(np.arange(24).reshape(2,3,4), axis=0)
        // Shape: (3, 4)
        // Values: [[12,14,16,18],[20,22,24,26],[28,30,32,34]]
        var a = np.arange(24).reshape(2, 3, 4);
        var result = np.sum(a, axis: 0);

        result.shape.Should().ContainInOrder(3, 4);
        result.GetInt64(0, 0).Should().Be(12);
        result.GetInt64(0, 1).Should().Be(14);
        result.GetInt64(1, 0).Should().Be(20);
        result.GetInt64(2, 3).Should().Be(34);
    }

    [Test]
    [NumPyPorted("test_numeric.py")]
    public void Sum_3D_Axis1()
    {
        // Python: np.sum(np.arange(24).reshape(2,3,4), axis=1).shape
        // Returns: (2, 4)
        var a = np.arange(24).reshape(2, 3, 4);
        var result = np.sum(a, axis: 1);

        result.shape.Should().ContainInOrder(2, 4);
    }

    [Test]
    [NumPyPorted("test_numeric.py")]
    public void Sum_3D_AxisNeg1()
    {
        // Python: np.sum(np.arange(24).reshape(2,3,4), axis=-1).shape
        // Returns: (2, 3)
        var a = np.arange(24).reshape(2, 3, 4);
        var result = np.sum(a, axis: -1);

        result.shape.Should().ContainInOrder(2, 3);
    }

    #endregion
}
