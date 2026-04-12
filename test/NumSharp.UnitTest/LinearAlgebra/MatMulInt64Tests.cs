using System;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.LinearAlgebra;

/// <summary>
/// Tests for np.matmul and np.dot with int64 indexing support.
/// Verifies SIMD path works correctly for contiguous arrays.
/// </summary>
public class MatMulInt64Tests
{
    #region Basic Correctness Tests

    [Test]
    public async Task MatMul_Float64_2x2_Correct()
    {
        // NumPy:
        // a = np.array([[1., 2.], [3., 4.]])
        // b = np.array([[5., 6.], [7., 8.]])
        // np.matmul(a, b) = [[19., 22.], [43., 50.]]
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(19.0);
        await Assert.That(result.GetDouble(0, 1)).IsEqualTo(22.0);
        await Assert.That(result.GetDouble(1, 0)).IsEqualTo(43.0);
        await Assert.That(result.GetDouble(1, 1)).IsEqualTo(50.0);
    }

    [Test]
    public async Task MatMul_Float32_2x2_Correct()
    {
        var a = np.array(new float[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new float[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        await Assert.That(result.GetSingle(0, 0)).IsEqualTo(19.0f);
        await Assert.That(result.GetSingle(0, 1)).IsEqualTo(22.0f);
        await Assert.That(result.GetSingle(1, 0)).IsEqualTo(43.0f);
        await Assert.That(result.GetSingle(1, 1)).IsEqualTo(50.0f);
    }

    [Test]
    public async Task MatMul_Int32_2x2_Correct()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new int[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        await Assert.That(result.GetInt32(0, 0)).IsEqualTo(19);
        await Assert.That(result.GetInt32(0, 1)).IsEqualTo(22);
        await Assert.That(result.GetInt32(1, 0)).IsEqualTo(43);
        await Assert.That(result.GetInt32(1, 1)).IsEqualTo(50);
    }

    #endregion

    #region SIMD Path Tests (Contiguous Arrays)

    [Test]
    public async Task MatMul_ContiguousArrays_UsesSimdPath()
    {
        // Create contiguous arrays that should trigger SIMD path
        var a = np.arange(100).reshape(10, 10).astype(NPTypeCode.Double);
        var b = np.arange(100).reshape(10, 10).astype(NPTypeCode.Double);

        // Verify arrays are contiguous
        await Assert.That(a.Shape.IsContiguous).IsTrue();
        await Assert.That(b.Shape.IsContiguous).IsTrue();

        var result = np.matmul(a, b);

        // Verify shape
        await Assert.That(result.shape[0]).IsEqualTo(10);
        await Assert.That(result.shape[1]).IsEqualTo(10);

        // Verify a specific element: result[0,0] = sum(a[0,:] * b[:,0])
        // a[0,:] = [0,1,2,...,9], b[:,0] = [0,10,20,...,90]
        // sum = 0*0 + 1*10 + 2*20 + ... + 9*90 = 10+40+90+160+250+360+490+640+810 = 2850
        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(2850.0);
    }

    [Test]
    public async Task MatMul_LargerMatrices_Correct()
    {
        // Test with larger matrices to exercise blocking in SIMD path
        var a = np.ones(new Shape(64, 64), NPTypeCode.Double);
        var b = np.ones(new Shape(64, 64), NPTypeCode.Double);

        var result = np.matmul(a, b);

        // All elements should be 64 (sum of 64 ones)
        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(64.0);
        await Assert.That(result.GetDouble(32, 32)).IsEqualTo(64.0);
        await Assert.That(result.GetDouble(63, 63)).IsEqualTo(64.0);
    }

    [Test]
    public async Task MatMul_NonSquare_MxN_NxP()
    {
        // Test non-square matrix multiplication
        // (3x4) @ (4x5) = (3x5)
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Double);
        var b = np.arange(20).reshape(4, 5).astype(NPTypeCode.Double);

        await Assert.That(a.Shape.IsContiguous).IsTrue();
        await Assert.That(b.Shape.IsContiguous).IsTrue();

        var result = np.matmul(a, b);

        await Assert.That(result.shape[0]).IsEqualTo(3);
        await Assert.That(result.shape[1]).IsEqualTo(5);

        // NumPy: result[0,0] = 0*0 + 1*5 + 2*10 + 3*15 = 70
        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(70.0);
    }

    #endregion

    #region np.dot Tests

    [Test]
    public async Task Dot_Float64_VectorVector()
    {
        // NumPy: np.dot([1,2,3], [4,5,6]) = 32
        var a = np.array(new double[] { 1, 2, 3 });
        var b = np.array(new double[] { 4, 5, 6 });

        var result = np.dot(a, b);

        await Assert.That((double)result).IsEqualTo(32.0);
    }

    [Test]
    public async Task Dot_Float32_VectorVector()
    {
        var a = np.array(new float[] { 1, 2, 3 });
        var b = np.array(new float[] { 4, 5, 6 });

        var result = np.dot(a, b);

        await Assert.That((float)result).IsEqualTo(32.0f);
    }

    [Test]
    public async Task Dot_MatrixVector()
    {
        // NumPy: np.dot([[1,2],[3,4]], [5,6]) = [17, 39]
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new double[] { 5, 6 });

        var result = np.dot(a, b);

        await Assert.That(result.shape[0]).IsEqualTo(2);
        await Assert.That(result.GetDouble(0)).IsEqualTo(17.0);
        await Assert.That(result.GetDouble(1)).IsEqualTo(39.0);
    }

    [Test]
    public async Task Dot_MatrixMatrix_SameAsMatMul()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });

        var dotResult = np.dot(a, b);
        var matmulResult = np.matmul(a, b);

        // For 2D arrays, dot and matmul should give same result
        await Assert.That(dotResult.GetDouble(0, 0)).IsEqualTo(matmulResult.GetDouble(0, 0));
        await Assert.That(dotResult.GetDouble(0, 1)).IsEqualTo(matmulResult.GetDouble(0, 1));
        await Assert.That(dotResult.GetDouble(1, 0)).IsEqualTo(matmulResult.GetDouble(1, 0));
        await Assert.That(dotResult.GetDouble(1, 1)).IsEqualTo(matmulResult.GetDouble(1, 1));
    }

    #endregion

    #region Edge Cases

    [Test]
    public async Task MatMul_1x1_Matrices()
    {
        var a = np.array(new double[,] { { 3 } });
        var b = np.array(new double[,] { { 4 } });

        var result = np.matmul(a, b);

        await Assert.That(result.shape[0]).IsEqualTo(1);
        await Assert.That(result.shape[1]).IsEqualTo(1);
        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(12.0);
    }

    [Test]
    public async Task MatMul_Identity_PreservesMatrix()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var identity = np.eye(2);

        var result = np.matmul(a, identity);

        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(1.0);
        await Assert.That(result.GetDouble(0, 1)).IsEqualTo(2.0);
        await Assert.That(result.GetDouble(1, 0)).IsEqualTo(3.0);
        await Assert.That(result.GetDouble(1, 1)).IsEqualTo(4.0);
    }

    [Test]
    public async Task MatMul_ZeroMatrix_ReturnsZeros()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var zeros = np.zeros(new Shape(2, 2), NPTypeCode.Double);

        var result = np.matmul(a, zeros);

        await Assert.That(result.GetDouble(0, 0)).IsEqualTo(0.0);
        await Assert.That(result.GetDouble(0, 1)).IsEqualTo(0.0);
        await Assert.That(result.GetDouble(1, 0)).IsEqualTo(0.0);
        await Assert.That(result.GetDouble(1, 1)).IsEqualTo(0.0);
    }

    #endregion

    #region Various DTypes

    [Test]
    public async Task MatMul_Int64_Correct()
    {
        var a = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new long[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        await Assert.That(result.GetInt64(0, 0)).IsEqualTo(19L);
        await Assert.That(result.GetInt64(1, 1)).IsEqualTo(50L);
    }

    [Test]
    public async Task Dot_LargeContiguousArrays()
    {
        // Test with larger arrays to ensure SIMD path handles size correctly
        var rng = np.random.RandomState(42);
        var a = rng.randn(100, 100);
        var b = rng.randn(100, 100);

        // Should not throw
        var result = np.dot(a, b);

        await Assert.That(result.shape[0]).IsEqualTo(100);
        await Assert.That(result.shape[1]).IsEqualTo(100);
    }

    #endregion
}
