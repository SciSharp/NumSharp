using System;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Benchmark tests for axis reduction SIMD optimizations.
/// Tests correctness of strided SIMD (AVX2 gather) and parallel outer loop.
/// Issue #576: Complete SIMD axis reductions
/// </summary>
public class AxisReductionBenchmarkTests
{
    #region Correctness Tests - Strided Access with SIMD Gather

    [Test]
    public void Sum_Axis0_LargeStrided_CorrectResults()
    {
        // Sum along axis 0 requires strided access (rows are not contiguous)
        // This tests AVX2 gather for float
        int rows = 1000;
        int cols = 100;
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i + 1; // Each row has value (i+1)

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Each column sum = sum(1 to 1000) = 1000 * 1001 / 2 = 500500
        float expected = (float)(rows * (rows + 1)) / 2;
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(expected, result.GetSingle(j), 1.0f, $"Column {j} mismatch");
        }
    }

    [Test]
    public void Sum_Axis0_LargeStrided_Double_CorrectResults()
    {
        // Test AVX2 gather for double
        int rows = 1000;
        int cols = 100;
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i + 1;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        double expected = (double)(rows * (rows + 1)) / 2;
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(expected, result.GetDouble(j), 1e-6, $"Column {j} mismatch");
        }
    }

    [Test]
    public void Max_Axis0_LargeStrided_CorrectResults()
    {
        // Test strided Max with AVX2 gather
        int rows = 500;
        int cols = 200;
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i * cols + j; // Increasing values

        var arr = np.array(data);
        var result = np.amax(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Max along axis 0 should be from the last row
        for (int j = 0; j < cols; j++)
        {
            float expected = (rows - 1) * cols + j;
            Assert.AreEqual(expected, result.GetSingle(j), 1e-3f, $"Column {j} mismatch");
        }
    }

    [Test]
    public void Min_Axis0_LargeStrided_CorrectResults()
    {
        // Test strided Min with AVX2 gather
        int rows = 500;
        int cols = 200;
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i * cols + j;

        var arr = np.array(data);
        var result = np.amin(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Min along axis 0 should be from the first row
        for (int j = 0; j < cols; j++)
        {
            double expected = j;
            Assert.AreEqual(expected, result.GetDouble(j), 1e-10, $"Column {j} mismatch");
        }
    }

    [Test]
    public void Prod_Axis0_SmallStrided_CorrectResults()
    {
        // Test strided Prod (small to avoid overflow)
        int rows = 4;
        int cols = 10;
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i + 1; // Values 1, 2, 3, 4

        var arr = np.array(data);
        var result = np.prod(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Prod = 1 * 2 * 3 * 4 = 24
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(24.0f, result.GetSingle(j), 1e-3f, $"Column {j} mismatch");
        }
    }

    #endregion

    #region Correctness Tests - Parallel Outer Loop

    [Test]
    public void Sum_Axis1_LargeOutput_ParallelPath()
    {
        // Large number of output elements triggers parallel outer loop
        // With threshold = 1000, this should use parallel path
        int rows = 2000; // > 1000 output elements
        int cols = 100;
        var data = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = j + 1;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(rows);

        // Each row sum = 1 + 2 + ... + 100 = 5050
        double expected = (double)(cols * (cols + 1)) / 2;
        for (int i = 0; i < rows; i++)
        {
            Assert.AreEqual(expected, result.GetDouble(i), 1e-10, $"Row {i} mismatch");
        }
    }

    [Test]
    public void Mean_Axis0_LargeOutput_ParallelPath()
    {
        // Test Mean with parallel outer loop
        int rows = 100;
        int cols = 2000; // > 1000 output elements
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i;

        var arr = np.array(data);
        var result = np.mean(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Mean of 0, 1, 2, ..., 99 = 49.5
        float expected = (float)(rows - 1) / 2;
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(expected, result.GetDouble(j), 1e-3, $"Column {j} mismatch");
        }
    }

    [Test]
    public void Max_Axis1_LargeOutput_ParallelPath()
    {
        // Test Max with parallel outer loop
        int rows = 1500; // > 1000 output elements
        int cols = 50;
        var data = new int[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = j;

        var arr = np.array(data);
        var result = np.amax(arr, axis: 1);

        result.Should().BeShaped(rows);

        // Max in each row is cols - 1
        for (int i = 0; i < rows; i++)
        {
            Assert.AreEqual(cols - 1, result.GetInt32(i), $"Row {i} mismatch");
        }
    }

    #endregion

    #region Correctness Tests - Integer Types (non-gather path)

    [Test]
    public void Sum_Axis0_LargeStrided_Int32_CorrectResults()
    {
        // Int32 uses scalar strided path (no gather)
        int rows = 1000;
        int cols = 100;
        var data = new int[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = 1;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Each column sum = 1000 (sum of 1000 ones)
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual((long)rows, result.GetInt64(j), $"Column {j} mismatch");
        }
    }

    [Test]
    public void Sum_Axis0_LargeStrided_Int64_CorrectResults()
    {
        // Int64 uses scalar strided path (no gather)
        int rows = 1000;
        int cols = 100;
        var data = new long[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i + 1;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        long expected = (long)(rows * (rows + 1)) / 2;
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(expected, result.GetInt64(j), $"Column {j} mismatch");
        }
    }

    #endregion

    #region Edge Cases

    [Test]
    public void Sum_Axis0_SmallOutput_SequentialPath()
    {
        // Small output (< 1000) should use sequential path
        int rows = 100;
        int cols = 5; // Only 5 output elements
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = 1.0f;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual((float)rows, result.GetSingle(j), 1e-3f);
        }
    }

    [Test]
    public void Sum_Axis0_SmallAxisSize_CorrectResults()
    {
        // Very small axis size (< 8, below gather vector count)
        int rows = 3; // Very small axis
        int cols = 100;
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = i + 1;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        // Sum = 1 + 2 + 3 = 6
        for (int j = 0; j < cols; j++)
        {
            Assert.AreEqual(6.0f, result.GetSingle(j), 1e-3f);
        }
    }

    [Test]
    public void Sum_3D_MiddleAxis_Strided()
    {
        // 3D array, reduce along middle axis
        int d0 = 10, d1 = 100, d2 = 20;
        var data = new double[d0, d1, d2];
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    data[i, j, k] = 1.0;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(d0, d2);

        // Each element should be d1 (sum of d1 ones)
        for (int i = 0; i < d0; i++)
            for (int k = 0; k < d2; k++)
                Assert.AreEqual((double)d1, result.GetDouble(i, k), 1e-10);
    }

    [Test]
    public void Sum_WithNaN_StridedPath()
    {
        // NaN propagation in strided path
        int rows = 100;
        int cols = 10;
        var data = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i, j] = 1.0f;

        // Set one NaN in each column's middle
        for (int j = 0; j < cols; j++)
            data[50, j] = float.NaN;

        var arr = np.array(data);
        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);

        // All results should be NaN due to NaN propagation
        for (int j = 0; j < cols; j++)
        {
            Assert.IsTrue(float.IsNaN(result.GetSingle(j)), $"Column {j} should be NaN");
        }
    }

    #endregion

    #region Performance Verification (not timing, just that it runs)

    [Test]
    public void Sum_VeryLarge_Axis0_CompletesWithoutError()
    {
        // Very large array to stress test parallel + gather paths
        int rows = 5000;
        int cols = 500; // 500 output elements, some parallelism
        var arr = np.ones(new int[] { rows, cols }, dtype: np.float64);

        var result = np.sum(arr, axis: 0);

        result.Should().BeShaped(cols);
        Assert.AreEqual((double)rows, result.GetDouble(0), 1e-10);
    }

    [Test]
    public void Sum_VeryLarge_Axis1_CompletesWithoutError()
    {
        // Very large output to stress test parallel path
        int rows = 5000; // 5000 output elements -> parallel
        int cols = 500;
        var arr = np.ones(new int[] { rows, cols }, dtype: np.float32);

        var result = np.sum(arr, axis: 1);

        result.Should().BeShaped(rows);
        Assert.AreEqual((float)cols, result.GetSingle(0), 1e-3f);
    }

    #endregion

    #region AVX2 Feature Detection

    [Test]
    public void Avx2IsSupported_ReportStatus()
    {
        // Informational test to report AVX2 status
        Console.WriteLine($"AVX2 Supported: {Avx2.IsSupported}");
        Console.WriteLine($"AVX Supported: {Avx.IsSupported}");
        Console.WriteLine($"SSE42 Supported: {Sse42.IsSupported}");

        // This test always passes - it just reports status
        Assert.IsTrue(true);
    }

    #endregion
}
