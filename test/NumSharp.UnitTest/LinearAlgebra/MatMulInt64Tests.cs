using System;
using System.Linq;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra;

/// <summary>
/// Tests for np.matmul and np.dot with int64 indexing support.
/// Verifies SIMD path works correctly for contiguous arrays.
/// </summary>
[TestClass]
public class MatMulInt64Tests
{
    #region Basic Correctness Tests

    [TestMethod]
    public async Task MatMul_Float64_2x2_Correct()
    {
        // NumPy:
        // a = np.array([[1., 2.], [3., 4.]])
        // b = np.array([[5., 6.], [7., 8.]])
        // np.matmul(a, b) = [[19., 22.], [43., 50.]]
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        result.GetDouble(0, 0).Should().Be(19.0);
        result.GetDouble(0, 1).Should().Be(22.0);
        result.GetDouble(1, 0).Should().Be(43.0);
        result.GetDouble(1, 1).Should().Be(50.0);
    }

    [TestMethod]
    public async Task MatMul_Float32_2x2_Correct()
    {
        var a = np.array(new float[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new float[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        result.GetSingle(0, 0).Should().Be(19.0f);
        result.GetSingle(0, 1).Should().Be(22.0f);
        result.GetSingle(1, 0).Should().Be(43.0f);
        result.GetSingle(1, 1).Should().Be(50.0f);
    }

    [TestMethod]
    public async Task MatMul_Int32_2x2_Correct()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new int[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        result.GetInt32(0, 0).Should().Be(19);
        result.GetInt32(0, 1).Should().Be(22);
        result.GetInt32(1, 0).Should().Be(43);
        result.GetInt32(1, 1).Should().Be(50);
    }

    #endregion

    #region SIMD Path Tests (Contiguous Arrays)

    [TestMethod]
    public async Task MatMul_ContiguousArrays_UsesSimdPath()
    {
        // Create contiguous arrays that should trigger SIMD path
        var a = np.arange(100).reshape(10, 10).astype(NPTypeCode.Double);
        var b = np.arange(100).reshape(10, 10).astype(NPTypeCode.Double);

        // Verify arrays are contiguous
        a.Shape.IsContiguous.Should().BeTrue();
        b.Shape.IsContiguous.Should().BeTrue();

        var result = np.matmul(a, b);

        // Verify shape
        result.shape[0].Should().Be(10);
        result.shape[1].Should().Be(10);

        // Verify a specific element: result[0,0] = sum(a[0,:] * b[:,0])
        // a[0,:] = [0,1,2,...,9], b[:,0] = [0,10,20,...,90]
        // sum = 0*0 + 1*10 + 2*20 + ... + 9*90 = 10+40+90+160+250+360+490+640+810 = 2850
        result.GetDouble(0, 0).Should().Be(2850.0);
    }

    [TestMethod]
    public async Task MatMul_LargerMatrices_Correct()
    {
        // Test with larger matrices to exercise blocking in SIMD path
        var a = np.ones(new Shape(64, 64), NPTypeCode.Double);
        var b = np.ones(new Shape(64, 64), NPTypeCode.Double);

        var result = np.matmul(a, b);

        // All elements should be 64 (sum of 64 ones)
        result.GetDouble(0, 0).Should().Be(64.0);
        result.GetDouble(32, 32).Should().Be(64.0);
        result.GetDouble(63, 63).Should().Be(64.0);
    }

    [TestMethod]
    public async Task MatMul_NonSquare_MxN_NxP()
    {
        // Test non-square matrix multiplication
        // (3x4) @ (4x5) = (3x5)
        var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Double);
        var b = np.arange(20).reshape(4, 5).astype(NPTypeCode.Double);

        a.Shape.IsContiguous.Should().BeTrue();
        b.Shape.IsContiguous.Should().BeTrue();

        var result = np.matmul(a, b);

        result.shape[0].Should().Be(3);
        result.shape[1].Should().Be(5);

        // NumPy: result[0,0] = 0*0 + 1*5 + 2*10 + 3*15 = 70
        result.GetDouble(0, 0).Should().Be(70.0);
    }

    #endregion

    #region np.dot Tests

    [TestMethod]
    public async Task Dot_Float64_VectorVector()
    {
        // NumPy: np.dot([1,2,3], [4,5,6]) = 32
        var a = np.array(new double[] { 1, 2, 3 });
        var b = np.array(new double[] { 4, 5, 6 });

        var result = np.dot(a, b);

        ((double)result).Should().Be(32.0);
    }

    [TestMethod]
    public async Task Dot_Float32_VectorVector()
    {
        var a = np.array(new float[] { 1, 2, 3 });
        var b = np.array(new float[] { 4, 5, 6 });

        var result = np.dot(a, b);

        ((float)result).Should().Be(32.0f);
    }

    [TestMethod]
    public async Task Dot_MatrixVector()
    {
        // NumPy: np.dot([[1,2],[3,4]], [5,6]) = [17, 39]
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new double[] { 5, 6 });

        var result = np.dot(a, b);

        result.shape[0].Should().Be(2);
        result.GetDouble(0).Should().Be(17.0);
        result.GetDouble(1).Should().Be(39.0);
    }

    [TestMethod]
    public async Task Dot_MatrixMatrix_SameAsMatMul()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });

        var dotResult = np.dot(a, b);
        var matmulResult = np.matmul(a, b);

        // For 2D arrays, dot and matmul should give same result
        dotResult.GetDouble(0, 0).Should().Be(matmulResult.GetDouble(0, 0));
        dotResult.GetDouble(0, 1).Should().Be(matmulResult.GetDouble(0, 1));
        dotResult.GetDouble(1, 0).Should().Be(matmulResult.GetDouble(1, 0));
        dotResult.GetDouble(1, 1).Should().Be(matmulResult.GetDouble(1, 1));
    }

    #endregion

    #region Edge Cases

    [TestMethod]
    public async Task MatMul_1x1_Matrices()
    {
        var a = np.array(new double[,] { { 3 } });
        var b = np.array(new double[,] { { 4 } });

        var result = np.matmul(a, b);

        result.shape[0].Should().Be(1);
        result.shape[1].Should().Be(1);
        result.GetDouble(0, 0).Should().Be(12.0);
    }

    [TestMethod]
    public async Task MatMul_Identity_PreservesMatrix()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var identity = np.eye(2);

        var result = np.matmul(a, identity);

        result.GetDouble(0, 0).Should().Be(1.0);
        result.GetDouble(0, 1).Should().Be(2.0);
        result.GetDouble(1, 0).Should().Be(3.0);
        result.GetDouble(1, 1).Should().Be(4.0);
    }

    [TestMethod]
    public async Task MatMul_ZeroMatrix_ReturnsZeros()
    {
        var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
        var zeros = np.zeros(new Shape(2, 2), NPTypeCode.Double);

        var result = np.matmul(a, zeros);

        result.GetDouble(0, 0).Should().Be(0.0);
        result.GetDouble(0, 1).Should().Be(0.0);
        result.GetDouble(1, 0).Should().Be(0.0);
        result.GetDouble(1, 1).Should().Be(0.0);
    }

    #endregion

    #region Various DTypes

    [TestMethod]
    public async Task MatMul_Int64_Correct()
    {
        var a = np.array(new long[,] { { 1, 2 }, { 3, 4 } });
        var b = np.array(new long[,] { { 5, 6 }, { 7, 8 } });

        var result = np.matmul(a, b);

        result.GetInt64(0, 0).Should().Be(19L);
        result.GetInt64(1, 1).Should().Be(50L);
    }

    [TestMethod]
    public async Task Dot_LargeContiguousArrays()
    {
        // Test with larger arrays to ensure SIMD path handles size correctly
        var rng = np.random.RandomState(42);
        var a = rng.randn(100, 100);
        var b = rng.randn(100, 100);

        // Should not throw
        var result = np.dot(a, b);

        result.shape[0].Should().Be(100);
        result.shape[1].Should().Be(100);
    }

    #endregion
}
