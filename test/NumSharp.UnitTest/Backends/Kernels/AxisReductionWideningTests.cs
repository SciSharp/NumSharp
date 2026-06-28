using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for the widening-SIMD axis-reduction kernels
/// (DirectILKernelGenerator.Reduction.Axis.Widening.cs): NEP50 promotion
/// pairs where the accumulator is wider than the input — narrow ints to
/// int64/uint64 and any int to double (mean).
///
/// Includes the regression test for the uint32 leading-axis column-order
/// bug: the previous kernel built accumulators via Avx2.UnpackLow/UnpackHigh,
/// which interleave PER 128-BIT LANE, silently swapping output columns
/// (2,3) with (4,5) in every group of eight.
/// </summary>
[TestClass]
public class AxisReductionWideningTests
{
    #region uint32 column-order regression (was silently wrong)

    [TestMethod]
    public void Sum_UInt32_Axis0_ColumnOrder_Regression()
    {
        // 64 rows x 40 cols, column j filled with j.
        // NumPy: np.sum(a, axis=0)[j] == 64 * j for every j.
        // The old UnpackLow/High kernel returned columns (0,1,4,5,2,3,6,7)
        // within each group of 8 — e.g. col 2 got 64*4 instead of 64*2.
        var a = np.ones(new Shape(64, 40), np.uint32);
        for (int j = 0; j < 40; j++)
            a[":", j.ToString()] = (NDArray)(uint)j;

        var s = np.sum(a, axis: 0);

        s.Should().BeShaped(40);
        for (int j = 0; j < 40; j++)
            Assert.AreEqual((ulong)(64 * j), s.GetUInt64(j), $"column {j}");
    }

    #endregion

    #region narrow-int sums — NEP50 dtype + values (all six pairs)

    [TestMethod]
    public void Sum_Int16_Axis0_NegativeValues_MatchesNumPy()
    {
        // NumPy: a = (np.arange(12, dtype=np.int16) - 5).reshape(3, 4)
        //        np.sum(a, axis=0) -> array([-3, 0, 3, 6]); dtype int64
        var a = (np.arange(12) - 5).astype(np.int16).reshape(3, 4);
        var s = np.sum(a, axis: 0);

        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        s.Should().BeShaped(4);
        Assert.AreEqual(-3L, s.GetInt64(0));
        Assert.AreEqual(0L, s.GetInt64(1));
        Assert.AreEqual(3L, s.GetInt64(2));
        Assert.AreEqual(6L, s.GetInt64(3));
    }

    [TestMethod]
    public void Sum_Int16_Axis1_NegativeValues_MatchesNumPy()
    {
        // NumPy: np.sum((np.arange(12, dtype=np.int16) - 5).reshape(3, 4), axis=1)
        //        -> array([-14, 2, 18]); dtype int64
        var a = (np.arange(12) - 5).astype(np.int16).reshape(3, 4);
        var s = np.sum(a, axis: 1);

        Assert.AreEqual(NPTypeCode.Int64, s.GetTypeCode);
        s.Should().BeShaped(3);
        Assert.AreEqual(-14L, s.GetInt64(0));
        Assert.AreEqual(2L, s.GetInt64(1));
        Assert.AreEqual(18L, s.GetInt64(2));
    }

    [TestMethod]
    public void Sum_AllNarrowPairs_BothAxes_MatchScalarReference()
    {
        // 67x37 with awkward tails (not multiples of any vector width).
        // Signed dtypes include negatives to exercise sign extension through
        // the int32 scratch accumulator.
        VerifyAgainstScalar((np.arange(67 * 37) % 11 - 5).astype(np.int16).reshape(67, 37), NPTypeCode.Int64);
        VerifyAgainstScalar((np.arange(67 * 37) % 11).astype(np.uint16).reshape(67, 37), NPTypeCode.UInt64);
        VerifyAgainstScalar((np.arange(67 * 37) % 11 - 5).astype(np.int8).reshape(67, 37), NPTypeCode.Int64);
        VerifyAgainstScalar((np.arange(67 * 37) % 11).astype(np.uint8).reshape(67, 37), NPTypeCode.UInt64);
        VerifyAgainstScalar((np.arange(67 * 37) % 11 - 5).astype(np.int32).reshape(67, 37), NPTypeCode.Int64);
        VerifyAgainstScalar((np.arange(67 * 37) % 11).astype(np.uint32).reshape(67, 37), NPTypeCode.UInt64);
    }

    private static void VerifyAgainstScalar(NDArray a, NPTypeCode expectedDtype)
    {
        int rows = (int)a.shape[0], cols = (int)a.shape[1];
        string name = a.GetTypeCode.ToString();

        var s0 = np.sum(a, axis: 0);
        Assert.AreEqual(expectedDtype, s0.GetTypeCode, $"{name} axis0 dtype");
        for (int j = 0; j < cols; j++)
        {
            long want = 0;
            for (int i = 0; i < rows; i++) want += Convert.ToInt64(a[i, j].GetValue(0));
            Assert.AreEqual(want, Convert.ToInt64(s0[j].GetValue(0)), $"{name} sum axis0 col {j}");
        }

        var s1 = np.sum(a, axis: 1);
        Assert.AreEqual(expectedDtype, s1.GetTypeCode, $"{name} axis1 dtype");
        for (int i = 0; i < rows; i++)
        {
            long want = 0;
            for (int j = 0; j < cols; j++) want += Convert.ToInt64(a[i, j].GetValue(0));
            Assert.AreEqual(want, Convert.ToInt64(s1[i].GetValue(0)), $"{name} sum axis1 row {i}");
        }
    }

    #endregion

    #region chunk-drain boundaries (int32 scratch overflow safety)

    [TestMethod]
    public void Sum_Int16_Axis0_AxisLongerThanChunk_DrainsCorrectly()
    {
        // axisSize 20000 > ChunkI16 (16384) — exercises the scratch drain
        // between row chunks. Values span negatives.
        var a = ((np.arange(20000 * 5) % 9) - 4).astype(np.int16).reshape(20000, 5);
        var s = np.sum(a, axis: 0);

        for (int j = 0; j < 5; j++)
        {
            long want = 0;
            for (int i = 0; i < 20000; i++) want += Convert.ToInt64(a[i, j].GetValue(0));
            Assert.AreEqual(want, s.GetInt64(j), $"col {j}");
        }
    }

    [TestMethod]
    public void Sum_Int16_Axis1_RowLongerThanChunk_DrainsCorrectly()
    {
        // Row length 40000 > ChunkI16 — exercises the innermost per-chunk
        // int32 -> int64 drain.
        var a = ((np.arange(3 * 40000) % 9) - 4).astype(np.int16).reshape(3, 40000);
        var s = np.sum(a, axis: 1);

        for (int i = 0; i < 3; i++)
        {
            long want = 0;
            for (int j = 0; j < 40000; j++) want += Convert.ToInt64(a[i, j].GetValue(0));
            Assert.AreEqual(want, s.GetInt64(i), $"row {i}");
        }
    }

    #endregion

    #region layouts — 3-D slab path and sliced views

    [TestMethod]
    public void Sum_Int16_3D_MiddleAxis_SlabPath()
    {
        // axis=1 of a 3-D array drives the leading-axis kernel with
        // outerSize > 1 (per-slab origin arithmetic).
        var a = ((np.arange(5 * 7 * 9) % 13) - 6).astype(np.int16).reshape(5, 7, 9);
        var s = np.sum(a, axis: 1);

        s.Should().BeShaped(5, 9);
        for (int i = 0; i < 5; i++)
            for (int k = 0; k < 9; k++)
            {
                long want = 0;
                for (int j = 0; j < 7; j++) want += Convert.ToInt64(a[i, j, k].GetValue(0));
                Assert.AreEqual(want, s.GetInt64(i, k), $"[{i},{k}]");
            }
    }

    [TestMethod]
    public void Sum_Int16_SlicedRows_InnerSlabPath()
    {
        // a[::2, :] keeps the inner dim contiguous (axisStride != innerSize)
        // — the inner-slab variant of the leading-axis kernel.
        var basea = ((np.arange(80 * 37) % 11) - 5).astype(np.int16).reshape(80, 37);
        var a = basea["::2, :"];
        var s = np.sum(a, axis: 0);

        for (int j = 0; j < 37; j++)
        {
            long want = 0;
            for (int i = 0; i < 40; i++) want += Convert.ToInt64(a[i, j].GetValue(0));
            Assert.AreEqual(want, s.GetInt64(j), $"col {j}");
        }
    }

    [TestMethod]
    public void Sum_Int16_SlicedColumns_ScalarFallback()
    {
        // a[:, ::2] breaks inner contiguity — falls back to the typed scalar
        // path, which must agree with the SIMD kernels.
        var basea = ((np.arange(40 * 30) % 11) - 5).astype(np.int16).reshape(40, 30);
        var a = basea[":, ::2"];
        var s0 = np.sum(a, axis: 0);
        var dense = a.copy();
        var expected = np.sum(dense, axis: 0);

        for (int j = 0; j < 15; j++)
            Assert.AreEqual(expected.GetInt64(j), s0.GetInt64(j), $"col {j}");
    }

    #endregion

    #region mean (double accumulator) and prod (generic tier)

    [TestMethod]
    public void Mean_Int16_Axis0_MatchesNumPy()
    {
        // NumPy: np.mean((np.arange(12, dtype=np.int16) - 5).reshape(3, 4), axis=0)
        //        -> array([-1., 0., 1., 2.]); dtype float64
        var a = (np.arange(12) - 5).astype(np.int16).reshape(3, 4);
        var m = np.mean(a, axis: 0);

        Assert.AreEqual(NPTypeCode.Double, m.GetTypeCode);
        Assert.AreEqual(-1.0, m.GetDouble(0), 1e-12);
        Assert.AreEqual(0.0, m.GetDouble(1), 1e-12);
        Assert.AreEqual(1.0, m.GetDouble(2), 1e-12);
        Assert.AreEqual(2.0, m.GetDouble(3), 1e-12);
    }

    [TestMethod]
    public void Mean_UInt8_Axis1_MatchesScalarReference()
    {
        var a = (np.arange(50 * 23) % 11).astype(np.uint8).reshape(50, 23);
        var m = np.mean(a, axis: 1);

        Assert.AreEqual(NPTypeCode.Double, m.GetTypeCode);
        for (int i = 0; i < 50; i++)
        {
            double want = 0;
            for (int j = 0; j < 23; j++) want += Convert.ToDouble(a[i, j].GetValue(0));
            want /= 23;
            Assert.AreEqual(want, m.GetDouble(i), 1e-9, $"row {i}");
        }
    }

    [TestMethod]
    public void Prod_Int16_Axis0_MatchesNumPy()
    {
        // NumPy: np.prod([[1,2,3],[2,3,1],[3,1,2]] int16, axis=0) = [6,6,6]; int64
        var a = np.array(new short[,] { { 1, 2, 3 }, { 2, 3, 1 }, { 3, 1, 2 } });
        var p = np.prod(a, axis: 0);

        Assert.AreEqual(NPTypeCode.Int64, p.GetTypeCode);
        Assert.AreEqual(6L, p.GetInt64(0));
        Assert.AreEqual(6L, p.GetInt64(1));
        Assert.AreEqual(6L, p.GetInt64(2));
    }

    [TestMethod]
    public void Prod_Int16_Axis0_Int64Wraparound_MatchesNumPy()
    {
        // 70 twos: 2^70 wraps modulo 2^64 to 0 in NumPy's int64 accumulator.
        // The old scalar path computed through double and produced garbage
        // for this case; exact int64 multiplication matches NumPy.
        var a = np.full(new Shape(70, 3), 2, typeof(short));
        var p = np.prod(a, axis: 0);

        Assert.AreEqual(NPTypeCode.Int64, p.GetTypeCode);
        for (int j = 0; j < 3; j++)
            Assert.AreEqual(0L, p.GetInt64(j), $"col {j}");
    }

    #endregion
}
