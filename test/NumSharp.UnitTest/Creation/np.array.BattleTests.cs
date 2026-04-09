using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest;

/// <summary>
/// Battle tests for np.array() comparing against NumPy behavior.
/// These tests validate that np.array correctly creates NDArrays from:
/// - Scalars (0-dimensional arrays)
/// - 1D through 4D arrays
/// - Jagged arrays
/// - Existing NDArrays
/// - With ndmin parameter
/// - With copy parameter
/// </summary>
public class np_array_BattleTests
{
    #region Scalars (0-dimensional)

    [Test]
    public void Array_ScalarInt32_Creates0DimArray()
    {
        // NumPy: np.array(42) creates shape=(), ndim=0
        var arr = np.array(42);
        arr.ndim.Should().Be(0);
        arr.shape.Should().BeEmpty();
        arr.size.Should().Be(1);
        ((int)arr).Should().Be(42);
    }

    [Test]
    public void Array_ScalarInt64_Creates0DimArray()
    {
        var arr = np.array(42L);
        arr.ndim.Should().Be(0);
        arr.shape.Should().BeEmpty();
        arr.dtype.Should().Be(typeof(long));
    }

    [Test]
    public void Array_ScalarDouble_Creates0DimArray()
    {
        var arr = np.array(3.14);
        arr.ndim.Should().Be(0);
        arr.shape.Should().BeEmpty();
        arr.dtype.Should().Be(typeof(double));
        ((double)arr).Should().BeApproximately(3.14, 0.001);
    }

    [Test]
    public void Array_ScalarBoolTrue_Creates0DimArray()
    {
        var arr = np.array(true);
        arr.ndim.Should().Be(0);
        arr.dtype.Should().Be(typeof(bool));
        ((bool)arr).Should().BeTrue();
    }

    [Test]
    public void Array_ScalarBoolFalse_Creates0DimArray()
    {
        var arr = np.array(false);
        arr.ndim.Should().Be(0);
        ((bool)arr).Should().BeFalse();
    }

    [Test]
    public void Array_ScalarNegative_Creates0DimArray()
    {
        var arr = np.array(-99);
        arr.ndim.Should().Be(0);
        ((int)arr).Should().Be(-99);
    }

    [Test]
    public void Array_ScalarZero_Creates0DimArray()
    {
        var arr = np.array(0);
        arr.ndim.Should().Be(0);
        ((int)arr).Should().Be(0);
    }

    [Test]
    public void Array_ScalarLargeValue_Creates0DimArray()
    {
        var arr = np.array(long.MaxValue);
        arr.ndim.Should().Be(0);
        ((long)arr).Should().Be(long.MaxValue);
    }

    #endregion

    #region 1D Arrays

    [Test]
    public void Array_1DInt32_CreatesVector()
    {
        var arr = np.array(new int[] { 1, 2, 3 });
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 3 });
        arr.dtype.Should().Be(typeof(int));
    }

    [Test]
    public void Array_1DInt64_CreatesVector()
    {
        var arr = np.array(new long[] { 1, 2, 3 });
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 3 });
        arr.dtype.Should().Be(typeof(long));
    }

    [Test]
    public void Array_1DFloat32_CreatesVector()
    {
        var arr = np.array(new float[] { 1f, 2f, 3f });
        arr.ndim.Should().Be(1);
        arr.dtype.Should().Be(typeof(float));
    }

    [Test]
    public void Array_1DFloat64_CreatesVector()
    {
        var arr = np.array(new double[] { 1.0, 2.0, 3.0 });
        arr.ndim.Should().Be(1);
        arr.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Array_1DBool_CreatesVector()
    {
        var arr = np.array(new bool[] { true, false, true });
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 3 });
        arr.dtype.Should().Be(typeof(bool));
    }

    [Test]
    public void Array_1DSingleElement_CreatesVector()
    {
        // NumPy: np.array([42]) creates shape=(1,), ndim=1
        var arr = np.array(new int[] { 42 });
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 1 });
    }

    [Test]
    public void Array_1DEmpty_CreatesEmptyVector()
    {
        var arr = np.array(new int[0]);
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 0 });
        arr.size.Should().Be(0);
    }

    [Test]
    public void Array_1DNegative_CreatesVector()
    {
        var arr = np.array(new int[] { -1, -2, -3 });
        arr.GetInt32(0).Should().Be(-1);
        arr.GetInt32(2).Should().Be(-3);
    }

    #endregion

    #region 2D Arrays

    [Test]
    public void Array_2DInt32_CreatesMatrix()
    {
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 2 });
        arr.dtype.Should().Be(typeof(int));
    }

    [Test]
    public void Array_2DDouble_CreatesMatrix()
    {
        var arr = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 2 });
        arr.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Array_2DSingleRow_CreatesMatrix()
    {
        var arr = np.array(new int[,] { { 1, 2, 3 } });
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 1, 3 });
    }

    [Test]
    public void Array_2DSingleCol_CreatesMatrix()
    {
        var arr = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 3, 1 });
    }

    [Test]
    public void Array_2D1x1_CreatesMatrix()
    {
        var arr = np.array(new int[,] { { 42 } });
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 1, 1 });
    }

    #endregion

    #region 3D+ Arrays

    [Test]
    public void Array_3D_CreatesTensor()
    {
        var arr = np.array(new int[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
        arr.ndim.Should().Be(3);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 2, 2 });
    }

    [Test]
    public void Array_4D_CreatesTensor()
    {
        var arr = np.array(new int[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } } });
        arr.ndim.Should().Be(4);
        arr.shape.Should().BeEquivalentTo(new[] { 1, 2, 2, 2 });
    }

    #endregion

    #region Jagged Arrays

    [Test]
    public void Array_2DJagged_CreatesMatrix()
    {
        var arr = np.array(new int[][] { new[] { 1, 2 }, new[] { 3, 4 } });
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 2 });
    }

    [Test]
    public void Array_3DJagged_CreatesTensor()
    {
        var arr = np.array(new int[][][] {
            new[] { new[] { 1, 2 }, new[] { 3, 4 } },
            new[] { new[] { 5, 6 }, new[] { 7, 8 } }
        });
        arr.ndim.Should().Be(3);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 2, 2 });
    }

    #endregion

    #region ndmin Parameter

    [Test]
    public void Array_Ndmin1_On1DArray_NoChange()
    {
        var arr = np.array(new int[] { 1, 2, 3 }, ndmin: 1);
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 3 });
    }

    [Test]
    public void Array_Ndmin2_On1DArray_AddsLeadingDim()
    {
        // NumPy: np.array([1,2,3], ndmin=2) -> shape (1, 3)
        var arr = np.array(new int[] { 1, 2, 3 }, ndmin: 2);
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 1, 3 });
    }

    [Test]
    public void Array_Ndmin3_On1DArray_AddsTwoLeadingDims()
    {
        // NumPy: np.array([1,2,3], ndmin=3) -> shape (1, 1, 3)
        var arr = np.array(new int[] { 1, 2, 3 }, ndmin: 3);
        arr.ndim.Should().Be(3);
        arr.shape.Should().BeEquivalentTo(new[] { 1, 1, 3 });
    }

    [Test]
    public void Array_Ndmin3_On2DArray_AddsOneLeadingDim()
    {
        // NumPy: np.array([[1,2],[3,4]], ndmin=3) -> shape (1, 2, 2)
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } }, ndmin: 3);
        arr.ndim.Should().Be(3);
        arr.shape.Should().BeEquivalentTo(new[] { 1, 2, 2 });
    }

    [Test]
    public void Array_Ndmin1_On2DArray_NoChange()
    {
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } }, ndmin: 1);
        arr.ndim.Should().Be(2);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 2 });
    }

    #endregion

    #region From Existing NDArray

    [Test]
    public void Array_FromNDArray_DefaultCopyFalse_SharesMemory()
    {
        var original = np.array(new int[] { 1, 2, 3 });
        var alias = np.array(original, copy: false);

        alias[0] = np.array(999);
        original.GetInt32(0).Should().Be(999, "copy=false should share memory");
    }

    [Test]
    public void Array_FromNDArray_CopyTrue_Independent()
    {
        var original = np.array(new int[] { 1, 2, 3 });
        var copied = np.array(original, copy: true);

        copied[0] = np.array(999);
        original.GetInt32(0).Should().Be(1, "copy=true should not affect original");
    }

    [Test]
    public void Array_FromNDArray_PreservesShape()
    {
        var original = np.arange(6).reshape(2, 3);
        var arr = np.array(original);
        arr.shape.Should().BeEquivalentTo(new[] { 2, 3 });
    }

    #endregion

    #region Params Syntax

    [Test]
    public void Array_ParamsMultipleValues_Creates1DArray()
    {
        var arr = np.array(1, 2, 3);
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 3 });
    }

    #endregion

    #region Special Values

    [Test]
    public void Array_WithInfinity_Preserved()
    {
        var arr = np.array(new double[] { 1, double.PositiveInfinity, double.NegativeInfinity });
        arr.GetDouble(1).Should().Be(double.PositiveInfinity);
        arr.GetDouble(2).Should().Be(double.NegativeInfinity);
    }

    [Test]
    public void Array_WithNaN_Preserved()
    {
        var arr = np.array(new double[] { 1, double.NaN, 2 });
        double.IsNaN(arr.GetDouble(1)).Should().BeTrue();
    }

    #endregion

    #region Edge Cases

    [Test]
    public void Array_EmptyInt_CreatesEmptyWithCorrectDtype()
    {
        var arr = np.array(new int[0]);
        arr.shape.Should().BeEquivalentTo(new[] { 0 });
        arr.dtype.Should().Be(typeof(int));
    }

    [Test]
    public void Array_EmptyDouble_CreatesEmptyWithCorrectDtype()
    {
        var arr = np.array(new double[0]);
        arr.shape.Should().BeEquivalentTo(new[] { 0 });
        arr.dtype.Should().Be(typeof(double));
    }

    [Test]
    public void Array_EmptyBool_CreatesEmptyWithCorrectDtype()
    {
        var arr = np.array(new bool[0]);
        arr.shape.Should().BeEquivalentTo(new[] { 0 });
        arr.dtype.Should().Be(typeof(bool));
    }

    [Test]
    public void Array_String_CreatesCharArray()
    {
        // NumSharp-specific: string creates char array
        var arr = np.array("hello");
        arr.ndim.Should().Be(1);
        arr.shape.Should().BeEquivalentTo(new[] { 5 });
        arr.dtype.Should().Be(typeof(char));
    }

    #endregion

    #region All Supported Dtypes

    [Test]
    public void Array_Byte_Supported()
    {
        var arr = np.array(new byte[] { 1, 2, 3 });
        arr.dtype.Should().Be(typeof(byte));
    }

    [Test]
    public void Array_Int16_Supported()
    {
        var arr = np.array(new short[] { 1, 2, 3 });
        arr.dtype.Should().Be(typeof(short));
    }

    [Test]
    public void Array_UInt16_Supported()
    {
        var arr = np.array(new ushort[] { 1, 2, 3 });
        arr.dtype.Should().Be(typeof(ushort));
    }

    [Test]
    public void Array_UInt32_Supported()
    {
        var arr = np.array(new uint[] { 1, 2, 3 });
        arr.dtype.Should().Be(typeof(uint));
    }

    [Test]
    public void Array_UInt64_Supported()
    {
        var arr = np.array(new ulong[] { 1, 2, 3 });
        arr.dtype.Should().Be(typeof(ulong));
    }

    #endregion
}
