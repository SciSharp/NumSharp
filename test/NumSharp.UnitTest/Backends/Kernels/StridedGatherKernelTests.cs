using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests for the AVX2 hardware-gather strided SIMD paths (Phase 2a / roadmap
/// Wave 3): the Tier-3B inner-loop shell's gather-load (+ contiguous-store and
/// per-lane scatter-store) branches in DirectILKernelGenerator.InnerLoop.cs,
/// and the gather section of the 1-D strided flat reduction in
/// DirectILKernelGenerator.Reduction.cs.
///
/// On machines without AVX2/V256 the gather branches are not emitted and the
/// same shapes run the scalar-strided fallback — these tests then validate
/// that fallback, so they are hardware-independent correctness gates.
///
/// Also pins the NpyIterFlags collision regression: the NumSharp extension
/// flags used to alias the shifted NumPy flags (GATHER_ELIGIBLE==ONEITERATION,
/// CONTIGUOUS==GROWINNER, EARLY_EXIT==DELAYBUF, PARALLEL_SAFE==REDUCE), so
/// setting GATHER_ELIGIBLE made ForEach run a single inner loop and silently
/// skip every row after the first.
/// </summary>
[TestClass]
public class StridedGatherKernelTests
{
    private static readonly (Type dtype, string name)[] GatherDtypes =
    {
        (np.int32, "i32"),
        (np.uint32, "u32"),
        (np.float32, "f32"),
        (np.int64, "i64"),
        (np.uint64, "u64"),
        (np.float64, "f64"),
    };

    private static double Scalar(NDArray nd) => Convert.ToDouble(nd.GetValue(0));

    #region strided binary (gather-load + contiguous store)

    [TestMethod]
    public void StridedBinary_AllGatherDtypes_FullScan()
    {
        // n=1004 is not a multiple of the 4x-unrolled vector step, so the
        // main gather loop, single-vector remainder AND scalar tail all run.
        // Stride 3 exercises a non-power-of-two index vector.
        foreach (var (dt, name) in GatherDtypes)
        {
            const int n = 1004;
            var w1 = np.arange(3 * n + 1).astype(dt);
            var w2 = (np.arange(3 * n + 1) % 13 + 1).astype(dt);
            var sum = w1["::3"] + w2["::3"];

            for (int i = 0; i < n; i++)
            {
                double expect = Convert.ToDouble(w1[3 * i].GetValue(0)) + Convert.ToDouble(w2[3 * i].GetValue(0));
                Assert.AreEqual(expect, Scalar(sum[i]), $"{name} element {i}");
            }
        }
    }

    [TestMethod]
    public void StridedBinary_MixedStrides()
    {
        // lhs stride 2, rhs stride 3 — distinct per-operand index vectors.
        const int n = 500;
        var w1 = np.arange(2 * n).astype(np.float64) + 1.0;
        var w2 = np.arange(3 * n).astype(np.float64) + 2.0;
        var r = w1["::2"] + w2["::3"];
        for (int i = 0; i < n; i += 17)
            Assert.AreEqual(w1.GetDouble(2 * i) + w2.GetDouble(3 * i), r.GetDouble(i), $"element {i}");
        Assert.AreEqual(w1.GetDouble(2 * (n - 1)) + w2.GetDouble(3 * (n - 1)), r.GetDouble(n - 1));
    }

    [TestMethod]
    public void StridedBinary_TailOnly_SmallCount()
    {
        // count < one vector: the gather loops must skip straight to the tail.
        var w = np.arange(20).astype(np.float32) + 1f;
        var r = w["::3"] + w["::3"]; // 7 elements
        for (int i = 0; i < 7; i++)
            Assert.AreEqual(2 * w.GetSingle(3 * i), r.GetSingle(i), $"element {i}");
    }

    #endregion

    #region strided unary (incl. the multi-row 2-D shape that caught the flag collision)

    [TestMethod]
    public void StridedUnary_2D_AllRowsComputed_FlagCollisionRegression()
    {
        // sqrt(a[::2, ::2]) iterates row-by-row via EXTERNAL_LOOP. With the
        // old flag collision (GATHER_ELIGIBLE==ONEITERATION) only row 0 was
        // computed and rows 1+ silently stayed garbage.
        var big = (np.arange(40_000).astype(np.float64) + 1.0).reshape(200, 200);
        var v = big["::2, ::2"];
        var r = np.sqrt(v);

        for (int i = 0; i < 100; i += 7)
            for (int j = 0; j < 100; j += 11)
                Assert.AreEqual(Math.Sqrt(v.GetDouble(i, j)), r.GetDouble(i, j), 1e-12, $"({i},{j})");
        Assert.AreEqual(Math.Sqrt(v.GetDouble(99, 99)), r.GetDouble(99, 99), 1e-12, "last element");
    }

    [TestMethod]
    public void StridedUnary_NegativeStride()
    {
        // Negative byte strides produce negative (sign-extended) gather
        // indices — must read backwards correctly.
        var w = np.arange(2000).astype(np.float32) + 1f;
        var r = np.sqrt(w["::-2"]);
        for (int i = 0; i < 1000; i += 73)
            Assert.AreEqual(MathF.Sqrt(w.GetSingle(2000 - 1 - 2 * i)), r.GetSingle(i), 1e-5f, $"element {i}");
    }

    #endregion

    #region strided output (per-lane scatter store)

    [TestMethod]
    public void StridedOutput_ScatterStore_ThroughIterator()
    {
        // out is itself strided -> the gather+scatter branch. Buffers are
        // disjoint so no overlap machinery participates.
        foreach (var (dt, name) in GatherDtypes)
        {
            const int n = 333;
            var w1 = np.arange(2 * n).astype(dt);
            var w2 = (np.arange(2 * n) % 7 + 1).astype(dt);
            var outBuf = np.zeros(new Shape(2 * n), dt);
            var v1 = w1["::2"];
            var v2 = w2["::2"];
            var vOut = outBuf["::2"];

            using (var iter = NpyIterRef.MultiNew(3, new[] { v1, v2, vOut },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY }))
            {
                var tc = v1.GetTypeCode;
                iter.ExecuteElementWiseBinary(tc, tc, tc,
                    il => DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, tc),
                    il => DirectILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, tc),
                    $"gather_tests_scatter_add_{tc}");
            }

            for (int i = 0; i < n; i += 13)
            {
                double expect = Convert.ToDouble(w1[2 * i].GetValue(0)) + Convert.ToDouble(w2[2 * i].GetValue(0));
                Assert.AreEqual(expect, Scalar(outBuf[2 * i]), $"{name} out element {i}");
                Assert.AreEqual(0.0, Scalar(outBuf[2 * i + 1]), $"{name} odd slot {i} must stay untouched");
            }
        }
    }

    #endregion

    #region strided flat reductions (gather + 4 accumulators)

    [TestMethod]
    public void StridedReduction_SumMinMax_MatchContiguousReference()
    {
        // Odd count so the gather bulk + scalar tail both contribute.
        var wf = (np.arange(200_001).astype(np.float64) % 97.0) + 1.0;
        var sv = wf["::2"];
        var copy = sv.copy();

        Assert.AreEqual((double)np.sum(copy), (double)np.sum(sv), 1e-9, "sum f64");
        Assert.AreEqual((double)np.min(copy), (double)np.min(sv), "min f64");
        Assert.AreEqual((double)np.max(copy), (double)np.max(sv), "max f64");

        var wi = (np.arange(100_001) % 7 + 1).astype(np.int64);
        Assert.AreEqual((long)np.sum(wi["::2"].copy()), (long)np.sum(wi["::2"]), "sum i64");

        var wu = (np.arange(50_001) % 11 + 1).astype(np.uint32);
        Assert.AreEqual(
            Convert.ToDouble(np.sum(wu["::3"].copy()).GetValue(0)),
            Convert.ToDouble(np.sum(wu["::3"]).GetValue(0)), "sum u32 stride 3");

        var w32 = (np.arange(99_999).astype(np.float32) % 31f) + 1f;
        // f32 strided sum vs contiguous reference: same dtype, same vector
        // accumulation order class — allow tiny fp reassociation slack.
        Assert.AreEqual((float)np.sum(w32["::3"].copy()), (float)np.sum(w32["::3"]), 1.0f, "sum f32 stride 3");
    }

    [TestMethod]
    public void StridedReduction_NaNMinMax_ParityWithContiguous()
    {
        // Float min/max NaN behavior must be identical between the strided
        // gather path and the contiguous SIMD path (shared combine helpers).
        var w = np.arange(64).astype(np.float64) + 1.0;
        w[9] = np.array(double.NaN);
        var sv = w["1::2"]; // includes index 9
        double strided = (double)np.min(sv);
        double contig = (double)np.min(sv.copy());
        Assert.AreEqual(contig.Equals(strided), true, $"strided={strided} contig={contig}");
        double stridedMax = (double)np.max(sv);
        double contigMax = (double)np.max(sv.copy());
        Assert.AreEqual(contigMax.Equals(stridedMax), true, $"max strided={stridedMax} contig={contigMax}");
    }

    [TestMethod]
    public void StridedReduction_NegativeStride()
    {
        var w = (np.arange(10_001).astype(np.float64) % 13.0) + 1.0;
        Assert.AreEqual((double)np.sum(w["::-2"].copy()), (double)np.sum(w["::-2"]), 1e-9, "sum reversed-strided");
    }

    [TestMethod]
    public void StridedReduction_SmallCounts()
    {
        // Below the gather threshold the scalar loop must produce identical
        // results (gather section exits immediately, offset stays positioned).
        for (int n = 1; n <= 40; n++)
        {
            var w = np.arange(3 * n).astype(np.float64) + 1.0;
            var sv = w["::3"];
            Assert.AreEqual((double)np.sum(sv.copy()), (double)np.sum(sv), 1e-12, $"n={n}");
        }
    }

    #endregion

    #region iterator flag integrity (collision regression)

    [TestMethod]
    public void NpyIterFlags_ExtensionFlags_DoNotAliasNumPyFlags()
    {
        // The four NumSharp extension flags must not collide with any other
        // NpyIterFlags member (they used to alias GROWINNER/ONEITERATION/
        // DELAYBUF/REDUCE, silently corrupting multi-row iteration).
        var extension = new[]
        {
            NpyIterFlags.CONTIGUOUS, NpyIterFlags.GATHER_ELIGIBLE,
            NpyIterFlags.EARLY_EXIT, NpyIterFlags.PARALLEL_SAFE,
        };
        foreach (NpyIterFlags value in Enum.GetValues<NpyIterFlags>())
        {
            if (value == NpyIterFlags.None || Array.IndexOf(extension, value) >= 0)
                continue;
            foreach (var ext in extension)
                Assert.AreEqual(0u, (uint)(value & ext),
                    $"{ext} collides with {value} (0x{(uint)value:X8})");
        }
    }

    #endregion
}
