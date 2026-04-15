using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Kernels;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for SIMD-optimized axis reduction kernels.
/// </summary>
[TestClass]
public class AxisReductionSimdTests
{
    [TestMethod]
    public void Sum_Axis0_2D_Contiguous()
    {
        // Create array [[1, 2, 3], [4, 5, 6]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        // Sum along axis 0: [5, 7, 9]
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(5, (int)result[0]);
        Assert.AreEqual(7, (int)result[1]);
        Assert.AreEqual(9, (int)result[2]);
    }

    [TestMethod]
    public void Sum_Axis1_2D_Contiguous()
    {
        // Create array [[1, 2, 3], [4, 5, 6]]
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        // Sum along axis 1: [6, 15]
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6, (int)result[0]);
        Assert.AreEqual(15, (int)result[1]);
    }

    [TestMethod]
    public void Sum_Axis1_LargeArray_UsesSimd()
    {
        // Create a large 2D array to ensure SIMD path is used
        int rows = 100;
        int cols = 256;  // Multiple of Vector256 count
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i * cols + j + 1;

        var arr = np.array(data);

        // Sum along axis 1 (inner axis, contiguous)
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(rows);

        // Verify results
        for (int i = 0; i < rows; i++)
        {
            double expected = 0;
            for (int j = 0; j < cols; j++)
                expected += i * cols + j + 1;
            Assert.AreEqual(expected, (double)result[i], 1e-10);
        }
    }

    [TestMethod]
    public void Sum_Axis0_LargeArray_UsesSimd()
    {
        // Create a large 2D array to ensure SIMD path is used
        int rows = 256;  // Multiple of Vector256 count
        int cols = 100;
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i * cols + j + 1;

        var arr = np.array(data);

        // Sum along axis 0 (outer axis, strided access for inner elements)
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Verify results
        for (int j = 0; j < cols; j++)
        {
            float expected = 0;
            for (int i = 0; i < rows; i++)
                expected += i * cols + j + 1;
            Assert.AreEqual(expected, (float)result[j], 1e-3f);
        }
    }

    [TestMethod]
    public void Max_Axis1_2D()
    {
        var arr = np.array(new int[,] { { 3, 1, 4 }, { 1, 5, 9 } });

        var result = np.max(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(4, (int)result[0]);
        Assert.AreEqual(9, (int)result[1]);
    }

    [TestMethod]
    public void Min_Axis0_2D()
    {
        var arr = np.array(new double[,] { { 3.0, 1.0, 4.0 }, { 1.0, 5.0, 9.0 } });

        var result = np.min(arr, axis: 0);

        result.Should().BeShaped(3);
        Assert.AreEqual(1.0, (double)result[0], 1e-10);
        Assert.AreEqual(1.0, (double)result[1], 1e-10);
        Assert.AreEqual(4.0, (double)result[2], 1e-10);
    }

    [TestMethod]
    public void Prod_Axis1_2D()
    {
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var result = np.prod(arr, axis: 1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6L, (long)result[0]);
        Assert.AreEqual(120L, (long)result[1]);
    }

    [TestMethod]
    public void Sum_Axis2_3D_ContiguousInnerAxis()
    {
        // 2x3x4 array
        var data = new int[2, 3, 4];
        int val = 1;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 4; k++)
                    data[i, j, k] = val++;

        var arr = np.array(data);

        // Sum along axis 2 (innermost, contiguous)
        var result = np.sum(arr, axis: 2);

        result.Should().BeShaped(2, 3);

        // Verify: each slice along axis 2 is summed
        // [0,0,:] = 1+2+3+4 = 10
        // [0,1,:] = 5+6+7+8 = 26
        // etc.
        Assert.AreEqual(10L, (long)result[0, 0]);
        Assert.AreEqual(26L, (long)result[0, 1]);
        Assert.AreEqual(42L, (long)result[0, 2]);
        Assert.AreEqual(58L, (long)result[1, 0]);
    }

    [TestMethod]
    public void Sum_NegativeAxis()
    {
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        // axis=-1 is same as axis=1 for 2D array
        var result = np.sum(arr, axis: -1);

        result.Should().BeShaped(2);
        Assert.AreEqual(6L, (long)result[0]);
        Assert.AreEqual(15L, (long)result[1]);
    }

    [TestMethod]
    public void Sum_Keepdims()
    {
        var arr = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var result = np.sum(arr, axis: 1, keepdims: true);

        result.Should().BeShaped(2, 1);
        Assert.AreEqual(6L, (long)result[0, 0]);
        Assert.AreEqual(15L, (long)result[1, 0]);
    }

    [TestMethod]
    public void AxisReductionKernel_IsAvailable()
    {
        // Test that the kernel is available for supported types
        var key = new AxisReductionKernelKey(
            NPTypeCode.Double,
            NPTypeCode.Double,
            ReductionOp.Sum,
            InnerAxisContiguous: true
        );

        var kernel = ILKernelGenerator.TryGetAxisReductionKernel(key);

        // Kernel may be null if IL generation is disabled, but should not throw
        // If SIMD is available, kernel should be non-null
        if (ILKernelGenerator.VectorBits > 0 && ILKernelGenerator.Enabled)
        {
            Assert.IsNotNull(kernel);
        }
    }

    [TestMethod]
    public void Sum_AllDtypes_Axis1()
    {
        // Test that axis reduction works for various dtypes
        var intArr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var floatArr = np.array(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var doubleArr = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        var intResult = np.sum(intArr, axis: 1);
        var floatResult = np.sum(floatArr, axis: 1);
        var doubleResult = np.sum(doubleArr, axis: 1);

        Assert.AreEqual(3L, (long)intResult[0]);
        Assert.AreEqual(7L, (long)intResult[1]);

        Assert.AreEqual(3f, (float)floatResult[0], 1e-6f);
        Assert.AreEqual(7f, (float)floatResult[1], 1e-6f);

        Assert.AreEqual(3.0, (double)doubleResult[0], 1e-10);
        Assert.AreEqual(7.0, (double)doubleResult[1], 1e-10);
    }
}
