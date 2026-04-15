using System;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for NumPy 2.x (NEP50) dtype promotion rules.
/// These tests verify that reduction operations return the correct output dtype.
///
/// NumPy 2.x promotion rules:
/// - sum/prod of integer types: promote to int64/uint64
/// - sum/prod of float32: PRESERVE float32 (NEP50 change)
/// - sum/prod of float64: preserve float64
/// - mean always returns float64
/// - min/max preserve input dtype
/// - var/std always return float64
/// </summary>
public class DtypePromotionTests
{
    #region Sum Dtype Promotion

    [TestMethod]
    public async Task Sum_Int32_ReturnsInt64()
    {
        // NumPy: np.sum(np.array([1, 2, 3], dtype=np.int32)).dtype == int64
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result.GetInt64(0)).IsEqualTo(6L);
    }

    [TestMethod]
    public async Task Sum_UInt32_ReturnsUInt64()
    {
        // NumPy: np.sum(np.array([1, 2, 3], dtype=np.uint32)).dtype == uint64
        var a = np.array(new uint[] { 1, 2, 3 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.UInt64);
        await Assert.That(result.GetUInt64(0)).IsEqualTo(6UL);
    }

    [TestMethod]
    public async Task Sum_Int16_ReturnsInt64()
    {
        // NumPy: np.sum(np.array([1, 2, 3], dtype=np.int16)).dtype == int64
        var a = np.array(new short[] { 1, 2, 3 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result.GetInt64(0)).IsEqualTo(6L);
    }

    [TestMethod]
    public async Task Sum_UInt16_ReturnsUInt64()
    {
        // NumPy: np.sum(np.array([1, 2, 3], dtype=np.uint16)).dtype == uint64
        var a = np.array(new ushort[] { 1, 2, 3 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.UInt64);
        await Assert.That(result.GetUInt64(0)).IsEqualTo(6UL);
    }

    [TestMethod]
    public async Task Sum_Byte_ReturnsUInt64()
    {
        // NumPy: np.sum(np.array([1, 2, 3], dtype=np.uint8)).dtype == uint64
        var a = np.array(new byte[] { 1, 2, 3 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.UInt64);
        await Assert.That(result.GetUInt64(0)).IsEqualTo(6UL);
    }

    [TestMethod]
    public async Task Sum_Float32_ReturnsFloat32()
    {
        // NumPy 2.x (NEP50): np.sum(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype == float32
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Single);
        await Assert.That(result.GetSingle(0)).IsEqualTo(6.0f);
    }

    [TestMethod]
    public async Task Sum_Float64_ReturnsFloat64()
    {
        // NumPy: np.sum(np.array([1.0, 2.0, 3.0], dtype=np.float64)).dtype == float64
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
        await Assert.That(result.GetDouble(0)).IsEqualTo(6.0);
    }

    [TestMethod]
    public async Task Sum_Int64_ReturnsInt64()
    {
        // NumPy: np.sum(np.array([1, 2, 3], dtype=np.int64)).dtype == int64
        var a = np.array(new long[] { 1, 2, 3 });
        var result = np.sum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result.GetInt64(0)).IsEqualTo(6L);
    }

    #endregion

    #region Prod Dtype Promotion

    [TestMethod]
    public async Task Prod_Int32_ReturnsInt64()
    {
        // NumPy: np.prod(np.array([1, 2, 3], dtype=np.int32)).dtype == int64
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.prod(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        await Assert.That(result.GetInt64(0)).IsEqualTo(6L);
    }

    [TestMethod]
    public async Task Prod_Float32_ReturnsFloat32()
    {
        // NumPy 2.x (NEP50): np.prod(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype == float32
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.prod(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Single);
        await Assert.That(result.GetSingle(0)).IsEqualTo(6.0f);
    }

    #endregion

    #region Mean Dtype Promotion

    [TestMethod]
    public async Task Mean_Int32_ReturnsFloat64()
    {
        // NumPy: np.mean(np.array([1, 2, 3], dtype=np.int32)).dtype == float64
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.mean(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
        await Assert.That(result.GetDouble(0)).IsEqualTo(2.0);
    }

    [TestMethod]
    public async Task Mean_Float32_ReturnsFloat64()
    {
        // NumSharp: np.mean(float32_array) returns float64 by default
        // This differs from NumPy 2.x which returns float32 per NEP50
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.mean(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
        await Assert.That(result.GetDouble(0)).IsEqualTo(2.0);
    }

    [TestMethod]
    public async Task Mean_Float64_ReturnsFloat64()
    {
        // NumPy: np.mean(np.array([1.0, 2.0, 3.0], dtype=np.float64)).dtype == float64
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.mean(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
        await Assert.That(result.GetDouble(0)).IsEqualTo(2.0);
    }

    #endregion

    #region Min/Max Dtype Promotion (Preserve Input)

    [TestMethod]
    public async Task Min_Int32_ReturnsInt32()
    {
        // NumPy: np.min(np.array([1, 2, 3], dtype=np.int32)).dtype == int32
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.amin(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int32);
        await Assert.That(result.GetInt32(0)).IsEqualTo(1);
    }

    [TestMethod]
    public async Task Max_Int32_ReturnsInt32()
    {
        // NumPy: np.max(np.array([1, 2, 3], dtype=np.int32)).dtype == int32
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.amax(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int32);
        await Assert.That(result.GetInt32(0)).IsEqualTo(3);
    }

    [TestMethod]
    public async Task Min_Float32_ReturnsFloat32()
    {
        // NumPy: np.min(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype == float32
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.amin(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Single);
        await Assert.That(result.GetSingle(0)).IsEqualTo(1.0f);
    }

    [TestMethod]
    public async Task Max_Float64_ReturnsFloat64()
    {
        // NumPy: np.max(np.array([1.0, 2.0, 3.0], dtype=np.float64)).dtype == float64
        var a = np.array(new double[] { 1.0, 2.0, 3.0 });
        var result = np.amax(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
        await Assert.That(result.GetDouble(0)).IsEqualTo(3.0);
    }

    #endregion

    #region Var/Std Dtype Promotion

    [TestMethod]
    public async Task Var_Int32_ReturnsFloat64()
    {
        // NumPy: np.var(np.array([1, 2, 3], dtype=np.int32)).dtype == float64
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.var(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task Std_Int32_ReturnsFloat64()
    {
        // NumPy: np.std(np.array([1, 2, 3], dtype=np.int32)).dtype == float64
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.std(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
    }

    [TestMethod]
    [OpenBugs] // NumPy 2.x returns float64 for var(float32), NumSharp returns float32
    public async Task Var_Float32_NumPyReturnsFloat64()
    {
        // NumPy: np.var(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype == float64
        // NumPy always promotes to float64 for var/std for numerical stability
        // NumSharp follows GetComputingType() which preserves float32
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.var(a);

        // This test documents NumPy expected behavior (fails currently)
        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
    }

    [TestMethod]
    public async Task Var_Float32_ReturnsFloat32()
    {
        // NumSharp's current behavior - preserves float32 via GetComputingType()
        // This differs from NumPy which always returns float64 for numerical stability
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.var(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Single);
    }

    #endregion

    #region Axis Reduction Dtype Promotion

    [TestMethod]
    public async Task Sum_Int32_Axis_ReturnsInt64()
    {
        // NumPy: np.sum(np.array([[1, 2], [3, 4]], dtype=np.int32), axis=0).dtype == int64
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.sum(a, axis: 0);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        result.Should().BeOfValues(4L, 6L);
    }

    [TestMethod]
    public async Task Sum_Float32_Axis_ReturnsFloat32()
    {
        // NumPy 2.x (NEP50): np.sum(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), axis=0).dtype == float32
        var a = np.array(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } });
        var result = np.sum(a, axis: 0);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Single);
        result.Should().BeOfValues(4.0f, 6.0f);
    }

    [TestMethod]
    public async Task Min_Int32_Axis_ReturnsInt32()
    {
        // NumPy: np.min(np.array([[1, 2], [3, 4]], dtype=np.int32), axis=0).dtype == int32
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.amin(a, axis: 0);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int32);
        result.Should().BeOfValues(1, 2);
    }

    [TestMethod]
    public async Task Mean_Int32_Axis_ReturnsFloat64()
    {
        // NumPy: np.mean(np.array([[1, 2], [3, 4]], dtype=np.int32), axis=0).dtype == float64
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var result = np.mean(a, axis: 0);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Double);
        result.Should().BeOfValues(2.0, 3.0);
    }

    #endregion

    #region CumSum Dtype Promotion

    [TestMethod]
    public async Task CumSum_Int32_ReturnsInt64()
    {
        // NumPy: np.cumsum(np.array([1, 2, 3], dtype=np.int32)).dtype == int64
        var a = np.array(new int[] { 1, 2, 3 });
        var result = np.cumsum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Int64);
        result.Should().BeOfValues(1L, 3L, 6L);
    }

    [TestMethod]
    public async Task CumSum_Float32_ReturnsFloat32()
    {
        // NumPy 2.x (NEP50): np.cumsum(np.array([1.0, 2.0, 3.0], dtype=np.float32)).dtype == float32
        var a = np.array(new float[] { 1.0f, 2.0f, 3.0f });
        var result = np.cumsum(a);

        await Assert.That(result.typecode).IsEqualTo(NPTypeCode.Single);
        result.Should().BeOfValues(1.0f, 3.0f, 6.0f);
    }

    #endregion
}
