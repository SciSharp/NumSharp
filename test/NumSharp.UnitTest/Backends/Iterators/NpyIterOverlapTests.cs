using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.UnitTest.Backends.Iterators;

/// <summary>
/// COPY_IF_OVERLAP + memory-overlap solver tests (NpyMemOverlap — a port of
/// NumPy's mem_overlap.c, and the FORCECOPY/write-back machinery in
/// NpyIterRef construction/Dispose, NumPy nditer_constr.c:3083-3311).
///
/// All expected values were produced by running NumPy 2.4.2:
///   - B-cases: ufunc calls with overlapping out= (np.add/np.multiply)
///   - S-cases: np.shares_memory (exact) and np.may_share_memory (bounds)
///
/// Without COPY_IF_OVERLAP, write-ahead overlap silently corrupts:
/// add(a[:-1], a[:-1], out=a[1:]) used to cascade to [1,2,4,6,8,16,32,64].
/// </summary>
[TestClass]
public class NpyIterOverlapTests
{
    private const NpyIterPerOpFlags ELW = NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;

    private static NDArray Arange8() => np.arange(8).astype(np.float64) + 1.0;

    /// <summary>
    /// Production-equivalent binary op through the Tier-3B per-chunk route
    /// with NumPy's ufunc iterator flags (EXTERNAL_LOOP | COPY_IF_OVERLAP,
    /// all operands OVERLAP_ASSUME_ELEMENTWISE).
    /// </summary>
    private static void RunBinary(NDArray in0, NDArray in1, NDArray @out, BinaryOp op)
    {
        using var iter = NpyIterRef.MultiNew(3, new[] { in0, in1, @out },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[]
            {
                NpyIterPerOpFlags.READONLY | ELW,
                NpyIterPerOpFlags.READONLY | ELW,
                NpyIterPerOpFlags.WRITEONLY | ELW,
            });
        iter.ExecuteElementWiseBinary(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double,
            il => DirectILKernelGenerator.EmitScalarOperation(il, op, NPTypeCode.Double),
            il => DirectILKernelGenerator.EmitVectorOperation(il, op, NPTypeCode.Double),
            $"overlap_tests_{op}_f64");
    }

    private static void AssertValues(NDArray got, params double[] expect)
    {
        Assert.AreEqual(expect.Length, (int)got.size, "size mismatch");
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetDouble(i), $"element {i}");
    }

    #region COPY_IF_OVERLAP ufunc behaviors (expected values from NumPy)

    [TestMethod]
    public void Overlap_WriteAhead_ForcesCopy()
    {
        // numpy: np.add(a[:-1], a[:-1], out=a[1:]) -> [1,2,4,6,8,10,12,14]
        // without protection the kernel reads its own freshly written values
        // and cascades to [1,2,4,6,8,16,32,64].
        var a = Arange8();
        RunBinary(a[":-1"], a[":-1"], a["1:"], BinaryOp.Add);
        AssertValues(a, 1, 2, 4, 6, 8, 10, 12, 14);
    }

    [TestMethod]
    public void Overlap_ForwardDirection()
    {
        // numpy: np.add(a[1:], a[1:], out=a[:-1]) -> [4,6,8,10,12,14,16,8]
        var a = Arange8();
        RunBinary(a["1:"], a["1:"], a[":-1"], BinaryOp.Add);
        AssertValues(a, 4, 6, 8, 10, 12, 14, 16, 8);
    }

    [TestMethod]
    public void Overlap_ExactAlias_NoCopy()
    {
        // numpy: np.add(a, a, out=a) -> 2a. With OVERLAP_ASSUME_ELEMENTWISE on
        // all operands, exact aliasing must NOT force a temporary (NumPy
        // nditer_constr.c:3130-3152) — verified via the operand array.
        var a = Arange8();
        using (var iter = NpyIterRef.MultiNew(3, new[] { a, a, a },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | ELW, NpyIterPerOpFlags.READONLY | ELW, NpyIterPerOpFlags.WRITEONLY | ELW }))
        {
            var ops = iter.GetOperandArray();
            Assert.IsNotNull(ops);
            Assert.IsTrue(ReferenceEquals(ops[2], a), "exact alias must not be force-copied");
            iter.ExecuteElementWiseBinary(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double,
                il => DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Double),
                il => DirectILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Double),
                "overlap_tests_Add_f64");
        }
        AssertValues(a, 2, 4, 6, 8, 10, 12, 14, 16);
    }

    [TestMethod]
    public void Overlap_InterleavedDisjoint_NoCopy_SolverProvesNo()
    {
        // numpy: np.add(a[::2], a[1::2], out=a[::2]) -> [3,2,7,4,11,6,15,8].
        // Even/odd interleaved views never share a byte; the Diophantine
        // solver proves it at max_work=1, so no temporary is made.
        var a = Arange8();
        var even = a["::2"];
        var odd = a["1::2"];
        using (var iter = NpyIterRef.MultiNew(3, new[] { even, odd, even },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | ELW, NpyIterPerOpFlags.READONLY | ELW, NpyIterPerOpFlags.WRITEONLY | ELW }))
        {
            var ops = iter.GetOperandArray();
            Assert.IsNotNull(ops);
            Assert.IsTrue(ReferenceEquals(ops[2], even), "disjoint interleaved views must not be copied");
            iter.ExecuteElementWiseBinary(NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double,
                il => DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, NPTypeCode.Double),
                il => DirectILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Double),
                "overlap_tests_Add_f64");
        }
        AssertValues(a, 3, 2, 7, 4, 11, 6, 15, 8);
    }

    [TestMethod]
    public void Overlap_2D_ColumnShift()
    {
        // numpy: x = arange(25).reshape(5,5)+1; np.add(x[:,:-1], x[:,:-1], out=x[:,1:])
        var x = (np.arange(25).astype(np.float64) + 1.0).reshape(5, 5);
        RunBinary(x[":, :-1"], x[":, :-1"], x[":, 1:"], BinaryOp.Add);
        AssertValues(x["0"], 1, 2, 4, 6, 8);
        AssertValues(x["1"], 6, 12, 14, 16, 18);
        AssertValues(x["4"], 21, 42, 44, 46, 48);
    }

    [TestMethod]
    public void Overlap_NegativeStrideOut()
    {
        // numpy: np.add(a, 1, out=a[::-1]) -> [9,8,7,6,5,4,3,2]
        var a = Arange8();
        RunBinary(a, NDArray.Scalar(1.0), a["::-1"], BinaryOp.Add);
        AssertValues(a, 9, 8, 7, 6, 5, 4, 3, 2);
    }

    [TestMethod]
    public void Overlap_Multiply_BothDirections()
    {
        // numpy: np.multiply(a[:-2], a[2:], out=a[1:-1]) -> [1,3,8,15,24,35,48,8]
        var a = Arange8();
        RunBinary(a[":-2"], a["2:"], a["1:-1"], BinaryOp.Multiply);
        AssertValues(a, 1, 3, 8, 15, 24, 35, 48, 8);
    }

    #endregion

    #region solver matrix — np.shares_memory (exact) / np.may_share_memory (bounds)

    private static void AssertShare(NDArray u, NDArray v, bool exact, bool bounds, string name)
    {
        var rExact = NpyMemOverlap.SolveMayShareMemory(u, v, -1);
        var rBounds = NpyMemOverlap.SolveMayShareMemory(u, v, 0);
        Assert.AreEqual(exact, rExact == MemOverlap.Yes, $"{name}: exact (np.shares_memory)");
        Assert.AreEqual(bounds, rBounds != MemOverlap.No, $"{name}: bounds (np.may_share_memory)");
    }

    [TestMethod]
    public void Solver_MatchesNumPySharesMemory()
    {
        var a = np.arange(64).astype(np.float64);
        var b2d = np.arange(80).astype(np.uint8).reshape(4, 20);

        AssertShare(a[":3"], a["5:"], exact: false, bounds: false, "a[:3] vs a[5:]");
        AssertShare(a["::2"], a["1::2"], exact: false, bounds: true, "a[::2] vs a[1::2]");
        AssertShare(a[":10"], a["5:15"], exact: true, bounds: true, "a[:10] vs a[5:15]");
        AssertShare(a["::4"], a["2::4"], exact: false, bounds: true, "a[::4] vs a[2::4]");
        AssertShare(a["::4"], a["2::2"], exact: true, bounds: true, "a[::4] vs a[2::2]");
        AssertShare(a, a, exact: true, bounds: true, "a vs a");
        AssertShare(a["::-1"], a["::2"], exact: true, bounds: true, "a[::-1] vs a[::2]");
        AssertShare(b2d[":, ::7"], b2d[":, 3::3"], exact: false, bounds: true, "2d[:,::7] vs 2d[:,3::3]");
        AssertShare(b2d["0"], b2d["1"], exact: false, bounds: false, "2d row0 vs row1");
        AssertShare(b2d[":, 0"], b2d[":, 1"], exact: false, bounds: true, "2d col0 vs col1");
        AssertShare(b2d[":2, :"], b2d["2:, :"], exact: false, bounds: false, "2d top vs bottom");
        AssertShare(np.broadcast_to(a[":1"], new Shape(5)), a[":1"], exact: true, bounds: true, "broadcast vs self");
        AssertShare(a["1:7"], a["6:8"], exact: true, bounds: true, "a[1:7] vs a[6:8]");
        AssertShare(a["1:7"], a["7:8"], exact: false, bounds: false, "a[1:7] vs a[7:8]");
    }

    [TestMethod]
    public void Solver_EmptyArrays_NeverShare()
    {
        var a = np.arange(64).astype(np.float64);
        var empty = a["3:3"];
        Assert.AreEqual(MemOverlap.No, NpyMemOverlap.SolveMayShareMemory(empty, a, -1));
        Assert.AreEqual(MemOverlap.No, NpyMemOverlap.SolveMayShareMemory(a, empty, 0));
    }

    [TestMethod]
    public void Solver_InternalOverlap()
    {
        // Broadcast (stride-0 on a non-unit dim) maps many indices to one
        // address -> internal overlap. Contiguous and plain strided views
        // do not.
        var a = np.arange(64).astype(np.float64);
        var bc = np.broadcast_to(np.arange(3).astype(np.float64), new Shape(4, 3));
        Assert.AreEqual(MemOverlap.Yes, NpyMemOverlap.SolveMayHaveInternalOverlap(bc, -1), "broadcast view");
        Assert.AreEqual(MemOverlap.No, NpyMemOverlap.SolveMayHaveInternalOverlap(a, -1), "contiguous");
        Assert.AreEqual(MemOverlap.No, NpyMemOverlap.SolveMayHaveInternalOverlap(a["::2"], -1), "strided");
        Assert.AreEqual(MemOverlap.No, NpyMemOverlap.SolveMayHaveInternalOverlap(a["::-1"], -1), "reversed");
    }

    #endregion

    #region Layer-2 contiguous-output contract guard

    [TestMethod]
    public void ExecuteBinary_StridedOutput_Throws()
    {
        // The legacy whole-array kernels behind ExecuteBinary ignore output
        // strides (they assume a fresh contiguous result). This used to write
        // a strided output CONTIGUOUSLY — silent corruption. Now it throws.
        var a = Arange8();
        var c = np.zeros(new Shape(8), np.float64);
        Assert.ThrowsException<InvalidOperationException>(() =>
        {
            using var iter = NpyIterRef.MultiNew(3, new[] { a["::2"], a["1::2"], c["::2"] },
                NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ExecuteBinary(BinaryOp.Add);
        });
    }

    [TestMethod]
    public void ExecuteBinary_OffsetContiguousOutput_Works()
    {
        // Offset-contiguous output (a[1:]) satisfies the kernel contract; with
        // COPY_IF_OVERLAP the overlapping case is also numerically correct.
        var a = Arange8();
        using (var iter = NpyIterRef.MultiNew(3, new[] { a[":-1"], a[":-1"], a["1:"] },
            NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
            new[] { NpyIterPerOpFlags.READONLY | ELW, NpyIterPerOpFlags.READONLY | ELW, NpyIterPerOpFlags.WRITEONLY | ELW }))
        {
            iter.ExecuteBinary(BinaryOp.Add);
        }
        AssertValues(a, 1, 2, 4, 6, 8, 10, 12, 14);
    }

    #endregion

    #region production np.* routes still correct (flags are now passed by default)

    [TestMethod]
    public void ProductionBinary_NonOverlapping_Unaffected()
    {
        // The production binary route now passes COPY_IF_OVERLAP; outputs are
        // freshly allocated so behavior must be identical.
        var a = np.arange(16).astype(np.float64);
        var b = np.arange(16).astype(np.float64) * 2.0;
        var r = a["::2"] + b["::2"];
        for (int i = 0; i < 8; i++)
            Assert.AreEqual(2 * i + 4.0 * i, r.GetDouble(i), $"element {i}");
    }

    #endregion
}
