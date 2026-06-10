using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Iterators;

/// <summary>
/// Wave 4 (roadmap): windowed buffered iteration + DELAY_BUFALLOC + the
/// buffered-cast Advance fix (handover bug (b)).
///
/// Before this work, a buffered non-reduce iterator had NO buffered iternext:
/// construction did one eager fill and the plain advancers neither refilled
/// nor wrote back, so any iteration larger than one buffer (default 8192
/// elements) silently processed only the first fill — and the
/// Strides×ElementSizes pointer math used the BUFFER dtype size, corrupting
/// every reposition under an active cast (int32 source buffered as float64
/// advanced 8 bytes per element instead of 4).
///
/// These tests are the multi-fill gates the handover demanded: every one of
/// them iterates N &gt; 8192 (or forces a small buffersize) so the
/// flush → jump → refill machinery and the SrcElementSizes pointer math are
/// both load-bearing.
/// </summary>
[TestClass]
public class NpyIterBufferedWindowTests
{
    private const int MultiFillN = 20_005; // 3 windows: 8192 + 8192 + 3621

    private static readonly NpyIterPerOpFlags Elw = NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;

    // =====================================================================
    // Per-element protocol (NumPy nditer without external_loop)
    // =====================================================================

    [TestMethod]
    public void PerElement_BufferedCast_MultiWindow_Int32ToFloat64()
    {
        // NumPy 2.4.2:
        //   it = np.nditer([arr], flags=['buffered'], op_flags=[['readonly']],
        //                  op_dtypes=['float64'], casting='safe')
        //   -> iterates per element, values == arr exactly, across buffer refills.
        var arr = np.arange(MultiFillN).astype(np.int32);

        using var iter = NpyIterRef.New(
            arr, NpyIterGlobalFlags.BUFFERED,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            NPTypeCode.Double);

        long i = 0;
        do
        {
            double v = iter.GetValue<double>(0);
            if (v != i)
                Assert.Fail($"element {i}: {v} (window boundary at 8192/16384 — bug (b) regression)");
            i++;
        } while (iter.Iternext());

        Assert.AreEqual(MultiFillN, i, "must traverse ALL windows, not just the first fill");
    }

    [TestMethod]
    public void PerElement_BufferedCast_StridedSource_MultiWindow()
    {
        // Strided int32 source viewed as float64 — the refill repositions via
        // GotoIterIndex; with the old ElementSizes multiplier the second
        // window started 2x too far into the array.
        var wide = np.arange(2 * MultiFillN).astype(np.int32);
        var view = wide["::2"];

        using var iter = NpyIterRef.New(
            view, NpyIterGlobalFlags.BUFFERED,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            NPTypeCode.Double);

        long i = 0;
        do
        {
            double v = iter.GetValue<double>(0);
            if (v != 2 * i)
                Assert.Fail($"element {i}: {v}, expected {2 * i}");
            i++;
        } while (iter.Iternext());

        Assert.AreEqual(MultiFillN, i);
    }

    // =====================================================================
    // Window drivers (kernel consumes one fill per call)
    // =====================================================================

    [TestMethod]
    public void WindowDriver_BufferedCastBinary_MultiFill_Values()
    {
        // i32 + f64 with the iterator casting i32 -> f64 in windows; the
        // same-dtype f64 SIMD body runs over the buffers. Checks every
        // element so a wrong second-window position cannot hide.
        var a = np.arange(MultiFillN).astype(np.int32);
        var b = np.arange(MultiFillN).astype(np.float64) * 0.5;
        var outNd = np.empty(new Shape(MultiFillN), np.float64);

        using (var iter = NpyIterRef.MultiNew(3, new[] { a, b, outNd },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED |
            NpyIterGlobalFlags.GROWINNER | NpyIterGlobalFlags.DELAY_BUFALLOC,
            NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | Elw, NpyIterPerOpFlags.READONLY | Elw, NpyIterPerOpFlags.WRITEONLY | Elw },
            new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double }))
        {
            iter.ExecuteElementWiseBinary(
                NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double,
                il => DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Double),
                il => DirectILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Double),
                "wave4_tests_add_f64");
        }

        for (int i = 0; i < MultiFillN; i++)
        {
            double expect = i + i * 0.5;
            if (outNd.GetDouble(i) != expect)
                Assert.Fail($"element {i}: {outNd.GetDouble(i)} vs {expect}");
        }
    }

    [TestMethod]
    public void WindowDriver_BufferedCastOutput_WriteFlushAcrossWindows()
    {
        // The OUTPUT needs the cast (compute f64 -> store f32): windowed
        // write-back must flush every window, incl. the final one (which is
        // flushed by the iternext that returns false / the single-fill path).
        var src = np.arange(MultiFillN).astype(np.float64) + 0.25;
        var outNd = np.empty(new Shape(MultiFillN), np.float32);

        using (var iter = NpyIterRef.MultiNew(2, new[] { src, outNd },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED |
            NpyIterGlobalFlags.GROWINNER | NpyIterGlobalFlags.DELAY_BUFALLOC,
            NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | Elw, NpyIterPerOpFlags.WRITEONLY | Elw },
            new[] { NPTypeCode.Double, NPTypeCode.Double }))
        {
            // identity copy in f64; the iterator casts the buffered output to f32
            iter.ExecuteElementWiseUnary(
                NPTypeCode.Double, NPTypeCode.Double,
                il => { },
                null,
                "wave4_tests_identity_f64");
        }

        for (int i = 0; i < MultiFillN; i += 977)
            Assert.AreEqual((float)(i + 0.25), outNd.GetSingle(i), $"element {i}");
        Assert.AreEqual((float)(MultiFillN - 1 + 0.25), outNd.GetSingle(MultiFillN - 1), "last window must flush");
    }

    [TestMethod]
    public void WindowDriver_2D_StridedCastSource_RowAlignedWindows()
    {
        // (100,100) strided int32 view cast to f64, buffersize 4096 -> NumPy
        // fills row-aligned 4000/4000/2000 (verified against numpy 2.4.2).
        // Whatever the exact split, every element must be correct.
        var big = np.arange(40_000).astype(np.int32).reshape(200, 200);
        var v = big["::2, ::2"]; // (100,100), strides (400,2) elements
        var outNd = np.empty(new Shape(100, 100), np.float64);

        using (var iter = NpyIterRef.AdvancedNew(2, new[] { v, outNd },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.GROWINNER,
            NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | Elw, NpyIterPerOpFlags.WRITEONLY | Elw },
            new[] { NPTypeCode.Double, NPTypeCode.Double },
            bufferSize: 4096))
        {
            iter.ExecuteElementWiseUnary(
                NPTypeCode.Double, NPTypeCode.Double,
                il => { }, null, "wave4_tests_identity_f64");
        }

        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 100; j += 13)
                Assert.AreEqual((double)v.GetInt32(i, j), outNd.GetDouble(i, j), $"({i},{j})");
        Assert.AreEqual((double)v.GetInt32(99, 99), outNd.GetDouble(99, 99));
    }

    // =====================================================================
    // DELAY_BUFALLOC
    // =====================================================================

    [TestMethod]
    public unsafe void DelayBufalloc_DefersAllocation_AutoEnsuresOnExecute()
    {
        var arr = np.arange(MultiFillN).astype(np.int32);
        var outNd = np.empty(new Shape(MultiFillN), np.float64);

        using var iter = NpyIterRef.MultiNew(2, new[] { arr, outNd },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.DELAY_BUFALLOC,
            NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_UNSAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | Elw, NpyIterPerOpFlags.WRITEONLY | Elw },
            new[] { NPTypeCode.Double, NPTypeCode.Double });

        // Construction must NOT have allocated buffers (the whole point:
        // NumPy nditer raises if iterated without reset; we defer instead).
        Assert.IsTrue((iter.RawState->ItFlags & (uint)NpyIterFlags.DELAYBUF) != 0, "DELAYBUF flag set");
        Assert.IsTrue(iter.RawState->GetBuffer(0) == null, "no buffer before first use");
        Assert.AreEqual(0L, iter.RawState->BufTransferSize, "no window before first use");

        iter.ExecuteElementWiseUnary(
            NPTypeCode.Double, NPTypeCode.Double,
            il => { }, null, "wave4_tests_identity_f64");

        Assert.IsTrue((iter.RawState->ItFlags & (uint)NpyIterFlags.DELAYBUF) == 0, "DELAYBUF cleared after ensure");
        Assert.IsTrue(iter.RawState->GetBuffer(0) != null, "buffer materialized");

        for (int i = 0; i < MultiFillN; i += 991)
            Assert.AreEqual((double)i, outNd.GetDouble(i), $"element {i}");
        Assert.AreEqual((double)(MultiFillN - 1), outNd.GetDouble(MultiFillN - 1));
    }

    [TestMethod]
    public unsafe void DelayBufalloc_Reset_MaterializesBuffers()
    {
        var arr = np.arange(100).astype(np.int32);

        using var iter = NpyIterRef.New(
            arr, NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.DELAY_BUFALLOC,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            NPTypeCode.Double);

        Assert.IsTrue(iter.RawState->GetBuffer(0) == null, "deferred at construction");
        Assert.IsTrue(iter.Reset(), "NumPy contract: Reset materializes delayed buffers");
        Assert.IsTrue(iter.RawState->GetBuffer(0) != null, "allocated by Reset");
        Assert.AreEqual(0.0, iter.GetValue<double>(0));
    }

    // =====================================================================
    // Production np.* routes that use the machinery
    // =====================================================================

    [TestMethod]
    public void Production_PromotingUnary_Sqrt_Int32_MultiFill()
    {
        // sqrt(i32) -> f64 routes through the buffered-cast unary path
        // (A/B-measured 1.36x faster than the fused per-element convert).
        var a = np.arange(MultiFillN).astype(np.int32);
        var r = np.sqrt(a);

        Assert.AreEqual(NPTypeCode.Double, r.typecode);
        for (int i = 0; i < MultiFillN; i += 983)
            Assert.AreEqual(Math.Sqrt(i), r.GetDouble(i), 1e-12, $"element {i}");
        Assert.AreEqual(Math.Sqrt(MultiFillN - 1), r.GetDouble(MultiFillN - 1), 1e-12, "last element (multi-fill)");
    }

    [TestMethod]
    public void Production_PromotingUnary_Sqrt_Strided2D()
    {
        var big = np.arange(40_000).astype(np.int32).reshape(200, 200);
        var v = big["::2, ::2"];
        var r = np.sqrt(v);

        Assert.AreEqual(NPTypeCode.Double, r.typecode);
        for (int i = 0; i < 100; i += 7)
            for (int j = 0; j < 100; j += 11)
                Assert.AreEqual(Math.Sqrt(v.GetInt32(i, j)), r.GetDouble(i, j), 1e-12, $"({i},{j})");
        Assert.AreEqual(Math.Sqrt(v.GetInt32(99, 99)), r.GetDouble(99, 99), 1e-12);
    }

    [TestMethod]
    public void Production_MixedBinary_ValueParity_AllPromotionClasses()
    {
        // The binary route keeps the FUSED per-element-convert body (A/B:
        // beats the buffered-cast architecture for every cheap-op class) —
        // these pin the NumPy-identical values + dtypes at multi-fill sizes.
        int n = MultiFillN;

        var i32 = np.arange(n).astype(np.int32);
        var f64 = np.arange(n).astype(np.float64) + 0.5;
        var add = i32 + f64;
        Assert.AreEqual(NPTypeCode.Double, add.typecode);
        Assert.AreEqual((n - 1) + (n - 1) + 0.5, add.GetDouble(n - 1));

        var u8 = (np.arange(n) % 200).astype(np.uint8);
        var addU = u8 + i32;
        Assert.AreEqual(NPTypeCode.Int32, addU.typecode);
        Assert.AreEqual((n - 1) % 200 + (n - 1), addU.GetInt32(n - 1));

        var i8 = (np.arange(n) % 200).astype(np.@byte);  // np.byte == sbyte (NumPy parity)
        var addS = i8 + i32;
        Assert.AreEqual(NPTypeCode.Int32, addS.typecode);
        // numpy: np.int8(19804 % 200 = 4) + 19804 ... wraps via sbyte
        int expectSByte = unchecked((sbyte)((n - 1) % 200)) + (n - 1);
        Assert.AreEqual(expectSByte, addS.GetInt32(n - 1));

        var f32 = np.arange(n).astype(np.float32);
        var mix = f32 + f64;
        Assert.AreEqual(NPTypeCode.Double, mix.typecode);
        Assert.AreEqual((n - 1) + (n - 1) + 0.5, mix.GetDouble(n - 1));

        var cmp = i32 < f64;
        Assert.AreEqual(NPTypeCode.Boolean, cmp.typecode);
        Assert.IsTrue(cmp.GetBoolean(0));
        Assert.IsTrue(cmp.GetBoolean(n - 1));
    }

    // =====================================================================
    // ExecuteBinary (Layer-2 bridge) over the windowed machinery
    // =====================================================================

    [TestMethod]
    public void ExecuteBinary_BufferedCast_MultiFill()
    {
        // The casting-bridge path: ExecuteBinary on a BUFFERED iterator with
        // op_dtypes — previously only the first 8192 elements were computed
        // (no buffered iternext existed) and casts corrupted the advance.
        var a = np.arange(MultiFillN).astype(np.int32);
        var b = np.arange(MultiFillN).astype(np.int32);
        var outNd = np.empty(new Shape(MultiFillN), np.float64);

        using (var iter = NpyIterRef.MultiNew(3, new[] { a, b, outNd },
            NpyIterGlobalFlags.BUFFERED,
            NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY },
            new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double }))
        {
            iter.ExecuteBinary(BinaryOp.Add);
        }

        for (int i = 0; i < MultiFillN; i += 997)
            Assert.AreEqual(2.0 * i, outNd.GetDouble(i), $"element {i}");
        Assert.AreEqual(2.0 * (MultiFillN - 1), outNd.GetDouble(MultiFillN - 1), "last window");
    }

    // =====================================================================
    // Linearity criterion (NumPy: 'buffered' buffers only when REQUIRED)
    // =====================================================================

    [TestMethod]
    public unsafe void Buffered_LinearSameDtypeOperand_StaysUnbuffered()
    {
        // A same-dtype 1-D strided view is a single linear walk — NumPy does
        // not buffer it ('buffered' enables buffering when required); the
        // kernel reads the array directly through its true stride.
        var wide = np.arange(64).astype(np.float64);
        var view = wide["::2"];

        using var iter = NpyIterRef.New(view, NpyIterGlobalFlags.BUFFERED);

        Assert.IsTrue(iter.RawState->GetBuffer(0) == null, "linear operand must not be buffered");
        Assert.AreEqual(16L, iter.RawState->GetBufStride(0), "BufStrides carries the TRUE inner byte stride");

        // And a cast operand IS buffered:
        var view2 = wide["::2"].astype(np.int32);
        using var iter2 = NpyIterRef.New(view2, NpyIterGlobalFlags.BUFFERED,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING, NPTypeCode.Double);
        Assert.IsTrue(iter2.RawState->GetBuffer(0) != null, "cast operand must be buffered");
        Assert.AreEqual(8L, iter2.RawState->GetBufStride(0), "buffered operand stride = buffer element size");
    }
}
