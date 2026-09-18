using System;
using System.Numerics;

namespace NumSharp.Tests.Indexing;

/// <summary>
/// Tests for <see cref="np.put_along_axis(NDArray,NDArray,NDArray,int?)"/> — the per-slice scatter,
/// setter twin of <see cref="np.take_along_axis(NDArray,NDArray,int?)"/> (NumPy
/// <c>numpy.put_along_axis</c>, an advanced-assignment composition). All expected values come from
/// running NumPy 2.4.2. The op is IL-kernel-backed (an atomic validate-then-scatter pair of
/// whole-array strided odometers, <c>DirectILKernelGenerator.PutAlongAxis.cs</c>) and dtype-agnostic
/// via a byte-width-keyed element copy. Buckets:
/// <list type="bullet">
///   <item>argmax(keepdims) / argsort-driven in-place replacement; the docstring example.</item>
///   <item><c>axis=None</c>: C-contiguous writes back through the flat view; non-contiguous raises read-only.</item>
///   <item>Value broadcast (NOT cycled): full / column / (M,)-row / 0-d scalar / leading-1-strip / conflict.</item>
///   <item>J != M along the axis; negative indices (single wrap); duplicate indices (last write wins).</item>
///   <item>Index broadcasting on non-axis dims; source broadcasting on non-axis dims (last write wins).</item>
///   <item>Memory layouts: F-contiguous, transposed, negative-stride, sliced — all write through.</item>
///   <item>Value dtype cast (truncate toward zero; NaN/inf/overflow); COPY_IF_OVERLAP (values alias arr).</item>
///   <item>Per-dtype coverage on all 15 supported types.</item>
///   <item>Atomicity: an out-of-bounds index leaves <c>arr</c> completely untouched.</item>
///   <item>Error parity: axis / dtype / ndim / non-axis broadcast / value broadcast / read-only / OOB.</item>
/// </list>
/// </summary>
[TestClass]
public class PutAlongAxisTests
{
    private static NDArray A() => np.array(new int[,] { { 10, 30, 20 }, { 60, 40, 50 } });
    private static int[] Flat(NDArray a) => a.astype(NPTypeCode.Int32).ToArray<int>();
    private static long[] FlatL(NDArray a) => a.astype(NPTypeCode.Int64).ToArray<long>();

    // ===================================================================
    // Core semantics
    // ===================================================================

    [TestMethod]
    public void DocstringExample_ReplaceMaxWith99()
    {
        // NumPy docstring: put the argmax(keepdims) position to 99.
        var a = A();
        var ai = np.argmax(a, 1, keepdims: true);
        np.put_along_axis(a, ai, (NDArray)99, axis: 1);
        Flat(a).Should().Equal(10, 99, 20, 99, 40, 50);
    }

    [TestMethod]
    public void ArgsortScatter_RoundTripsToSortedValues_Axis1()
    {
        // put_along_axis(a, argsort(a), sort(a)) reconstructs a from its sorted values — the inverse
        // of take_along_axis(a, argsort(a)).
        var a = A();
        var order = np.argsort(a, 1);
        var sorted = np.sort(a, 1);
        np.put_along_axis(a, order, sorted, axis: 1);
        Flat(a).Should().Equal(10, 30, 20, 60, 40, 50);   // original restored
    }

    [TestMethod]
    public void ReturnsVoid_MutatesInPlace()
    {
        var a = A();
        np.put_along_axis(a, np.array(new int[,] { { 0 }, { 2 } }), (NDArray)(-1), axis: 1);
        Flat(a).Should().Equal(-1, 30, 20, 60, 40, -1);
    }

    // ===================================================================
    // axis = None
    // ===================================================================

    [TestMethod]
    public void AxisNone_Contiguous2D_WritesBackFlatCOrder()
    {
        var a = A();   // C-contiguous
        np.put_along_axis(a, np.array(new long[] { 0, 5 }), (NDArray)99, axis: null);
        Flat(a).Should().Equal(99, 30, 20, 60, 40, 99);
    }

    [TestMethod]
    public void AxisNone_Contiguous1D_WritesBack()
    {
        var a = np.array(new int[] { 0, 1, 2, 3, 4, 5 });
        np.put_along_axis(a, np.array(new long[] { 0, 5 }), (NDArray)99, axis: null);
        Flat(a).Should().Equal(99, 1, 2, 3, 4, 99);
    }

    [TestMethod]
    public void AxisNone_ZeroD_WritesBack()
    {
        var a = np.array(7);   // 0-d, contiguous
        np.put_along_axis(a, np.array(new long[] { 0 }), (NDArray)55, axis: null);
        ((int)a.GetValue()).Should().Be(55);
    }

    [TestMethod]
    public void AxisNone_NonContiguous_RaisesReadOnly()
    {
        // np.array(arr.flat) is a READ-ONLY copy for a non-contiguous source, so the assignment fails.
        var a = np.arange(6).reshape(2, 3).T;   // transposed non-contiguous
        Action act = () => np.put_along_axis(a, np.array(new long[] { 0, 1 }), (NDArray)9, axis: null);
        act.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
    }

    [TestMethod]
    public void AxisNone_NonContiguous_FloatIndices_DtypeErrorFirst()
    {
        // Dtype check precedes the read-only failure (NumPy _make_along_axis_idx order).
        var a = np.arange(6).reshape(2, 3).T;
        Action act = () => np.put_along_axis(a, np.array(new double[] { 0.0, 1.0 }), (NDArray)9, axis: null);
        act.Should().Throw<IndexError>().WithMessage("`indices` must be an integer array");
    }

    [TestMethod]
    public void AxisNone_NonOneDIndices_Raises()
    {
        var a = A();
        Action act = () => np.put_along_axis(a, np.array(new int[,] { { 0, 1 }, { 2, 0 } }), (NDArray)9, axis: null);
        act.Should().Throw<ValueError>().WithMessage("when axis=None, `indices` must have a single dimension.");
    }

    // ===================================================================
    // Value broadcasting (NOT cycling)
    // ===================================================================

    [TestMethod]
    public void Value_FullShape()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
                          np.array(new long[,] { { 7, 8, 9 }, { 1, 2, 3 } }), axis: 1);
        FlatL(a).Should().Equal(7, 8, 9, 3, 2, 1);
    }

    [TestMethod]
    public void Value_ScalarBroadcast()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }), (NDArray)42L, axis: 1);
        FlatL(a).Should().Equal(42, 42, 42, 42, 42, 42);
    }

    [TestMethod]
    public void Value_RowBroadcast_RightAligned()
    {
        // values (3,) broadcast to the (2,3) indexing result.
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
                          np.array(new long[] { 100, 200, 300 }), axis: 1);
        FlatL(a).Should().Equal(100, 200, 300, 300, 200, 100);
    }

    [TestMethod]
    public void Value_LeadingSizeOneDimStripped()
    {
        // values (1,2,3) broadcasts to the (2,3) result — the extra leading size-1 dim is stripped.
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
                          np.ones(new Shape(1, 2, 3), NPTypeCode.Int64), axis: 1);
        FlatL(a).Should().Equal(1, 1, 1, 1, 1, 1);
    }

    [TestMethod]
    public void Value_BroadcastConflict_Raises()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Int64);
        Action act = () => np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
                                             np.array(new long[] { 10, 20 }), axis: 1);
        act.Should().Throw<ValueError>()
           .WithMessage("shape mismatch: value array of shape (2,) could not be broadcast to indexing result of shape (2,3)");
    }

    [TestMethod]
    public void Value_ExtraLeadingNonUnit_Raises()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Int64);
        Action act = () => np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
                                             np.ones(new Shape(2, 2, 3), NPTypeCode.Int64), axis: 1);
        act.Should().Throw<ValueError>()
           .WithMessage("shape mismatch: value array of shape (2,2,3) could not be broadcast to indexing result of shape (2,3)");
    }

    // ===================================================================
    // Index families: J != M, negative wrap, duplicates
    // ===================================================================

    [TestMethod]
    public void JLessThanM()
    {
        var a = np.arange(6).reshape(2, 3);
        np.put_along_axis(a, np.array(new int[,] { { 0, 2 }, { 1, 0 } }),
                          np.array(new int[,] { { 7, 8 }, { 9, 1 } }), axis: 1);
        FlatL(a).Should().Equal(7, 1, 8, 1, 9, 5);
    }

    [TestMethod]
    public void JGreaterThanM_DuplicatesLastWins()
    {
        var a = np.arange(6).reshape(2, 3);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2, 0, 1 } }),
                          np.array(new int[,] { { 1, 2, 3, 4, 5 } }), axis: 1);
        FlatL(a).Should().Equal(4, 5, 3, 4, 5, 3);
    }

    [TestMethod]
    public void NegativeIndices_SingleWrap()
    {
        var a = np.arange(6).reshape(2, 3);
        np.put_along_axis(a, np.array(new int[,] { { -1 }, { -2 } }), (NDArray)99, axis: 1);
        FlatL(a).Should().Equal(0, 1, 99, 3, 99, 5);
    }

    [TestMethod]
    public void DuplicateIndices_LastWriteWins()
    {
        var a = np.zeros(new Shape(5), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new long[] { 0, 0, 0 }), np.array(new long[] { 1, 2, 3 }), axis: 0);
        FlatL(a).Should().Equal(3, 0, 0, 0, 0);
    }

    // ===================================================================
    // Broadcasting on non-axis dims
    // ===================================================================

    [TestMethod]
    public void IndexBroadcast_OverNonAxisDim()
    {
        // idx (1,4) broadcasts over the non-axis dim 0 to (3,4).
        var a = np.arange(12).reshape(3, 4);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2, 3 } }),
                          np.array(new int[,] { { -1, -2, -3, -4 }, { -5, -6, -7, -8 }, { -9, -10, -11, -12 } }), axis: 1);
        FlatL(a).Should().Equal(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12);
    }

    [TestMethod]
    public void SourceBroadcast_SizeOneDim_LastWriteWins()
    {
        // arr (1,3), idx (2,3): both iteration rows collapse onto arr row 0 — the LAST (row 1) wins.
        var a = np.zeros(new Shape(1, 3), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2 }, { 0, 1, 2 } }),
                          np.array(new long[,] { { 10, 20, 30 }, { 40, 50, 60 } }), axis: 1);
        FlatL(a).Should().Equal(40, 50, 60);
    }

    // ===================================================================
    // Memory layouts — all write through to the base
    // ===================================================================

    [TestMethod]
    public void Layout_FContiguous_WritesThrough()
    {
        var a = np.asfortranarray(np.arange(6).reshape(2, 3).astype(NPTypeCode.Int64));
        np.put_along_axis(a, np.array(new int[,] { { 0, 2 } }), np.array(new long[,] { { 77, 88 } }), axis: 1);
        FlatL(a).Should().Equal(77, 1, 88, 77, 4, 88);
    }

    [TestMethod]
    public void Layout_Transposed_WritesThrough()
    {
        var baseA = np.arange(24).reshape(2, 3, 4);
        var at = baseA.transpose(new int[] { 1, 0, 2 });   // (3,2,4) non-contiguous view
        np.put_along_axis(at, np.zeros(new Shape(3, 1, 4), NPTypeCode.Int64), (NDArray)(-7), axis: 1);
        // Sets at[:, 0, :] == baseA[:, :, :][... first plane] -> first 12 base elements become -7.
        FlatL(baseA).Should().StartWith(new long[] { -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7 });
    }

    [TestMethod]
    public void Layout_NegativeStride_WritesThrough()
    {
        var a = np.arange(6).reshape(2, 3);
        var rv = a[":, ::-1"];   // negative stride on axis 1
        np.put_along_axis(rv, np.array(new int[,] { { 0 }, { 1 } }), (NDArray)99, axis: 1);
        FlatL(a).Should().Equal(0, 1, 99, 3, 99, 5);
    }

    [TestMethod]
    public void Layout_SlicedStrided_WritesThrough()
    {
        var a = np.arange(20).reshape(4, 5);
        var asl = a["1:3, ::2"];   // (2,3) strided view over rows 1..2, cols 0,2,4
        np.put_along_axis(asl, np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
                          np.array(new long[,] { { 100, 101, 102 }, { 200, 201, 202 } }), axis: 1);
        // asl maps to base cols {0,2,4} of rows {1,2}: row1 cols->[100,101,102]; row2 cols reversed idx.
        ((long)a.GetValue(1, 0)).Should().Be(100);
        ((long)a.GetValue(1, 2)).Should().Be(101);
        ((long)a.GetValue(1, 4)).Should().Be(102);
        ((long)a.GetValue(2, 0)).Should().Be(202);
        ((long)a.GetValue(2, 2)).Should().Be(201);
        ((long)a.GetValue(2, 4)).Should().Be(200);
    }

    // ===================================================================
    // Value dtype casting
    // ===================================================================

    [TestMethod]
    public void ValueCast_FloatToInt_TruncatesTowardZero()
    {
        var a = np.zeros(new Shape(1, 4), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1, 2, 3 } }),
                          np.array(new double[,] { { 2.9, -2.9, 2.1, -2.1 } }), axis: 1);
        FlatL(a).Should().Equal(2, -2, 2, -2);
    }

    [TestMethod]
    public void ValueCast_NaNInf_ToIntSentinel()
    {
        // win-amd64 (MSVC) C-undefined cast: NaN/inf -> long.MinValue, matching NumPy on this host.
        var a = np.zeros(new Shape(1, 2), NPTypeCode.Int64);
        np.put_along_axis(a, np.array(new int[,] { { 0, 1 } }),
                          np.array(new double[] { double.NaN, double.PositiveInfinity }), axis: 1);
        FlatL(a).Should().Equal(long.MinValue, long.MinValue);
    }

    [TestMethod]
    public void ValueCast_Overflow_WrapsModularly()
    {
        var a = np.zeros(new Shape(1, 1), NPTypeCode.SByte);
        np.put_along_axis(a, np.array(new int[,] { { 0 } }), np.array(new int[] { 300 }), axis: 1);
        ((sbyte)a.GetValue(0, 0)).Should().Be(44);   // 300 mod 256 = 44
    }

    // ===================================================================
    // COPY_IF_OVERLAP — values aliasing arr
    // ===================================================================

    [TestMethod]
    public void Overlap_ValuesAliasArr_ReadsOriginal()
    {
        // put_along_axis(a, reversing-idx, a) reverses each row via a COPY of the original values.
        var a = np.arange(6).reshape(2, 3);
        np.put_along_axis(a, np.array(new int[,] { { 2, 1, 0 }, { 2, 1, 0 } }), a, axis: 1);
        FlatL(a).Should().Equal(2, 1, 0, 5, 4, 3);
    }

    // ===================================================================
    // Atomicity — OOB leaves arr untouched
    // ===================================================================

    [TestMethod]
    public void OutOfBounds_LeavesArrUntouched()
    {
        var a = np.arange(6).reshape(2, 3);
        Action act = () => np.put_along_axis(a, np.array(new int[,] { { 0, 9 }, { 1, 0 } }), (NDArray)99, axis: 1);
        act.Should().Throw<IndexError>().WithMessage("index 9 is out of bounds for axis 1 with size 3");
        FlatL(a).Should().Equal(0, 1, 2, 3, 4, 5);   // NOTHING written (a[0,0]=0 valid write did NOT land)
    }

    [TestMethod]
    public void NegativeOutOfBounds_ReportsOriginalValue()
    {
        var a = np.arange(6).reshape(2, 3);
        Action act = () => np.put_along_axis(a, np.array(new int[,] { { -4 } }), (NDArray)9, axis: 1);
        act.Should().Throw<IndexError>().WithMessage("index -4 is out of bounds for axis 1 with size 3");
    }

    // ===================================================================
    // Error parity (probed NumPy 2.4.2 order)
    // ===================================================================

    [TestMethod]
    public void Error_FloatIndices()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Double);
        Action act = () => np.put_along_axis(a, np.array(new double[,] { { 0.0, 1.0, 2.0 } }), (NDArray)9, axis: 1);
        act.Should().Throw<IndexError>().WithMessage("`indices` must be an integer array");
    }

    [TestMethod]
    public void Error_BoolIndices()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Double);
        Action act = () => np.put_along_axis(a, np.array(new bool[,] { { true, false, true } }), (NDArray)9, axis: 1);
        act.Should().Throw<IndexError>().WithMessage("`indices` must be an integer array");
    }

    [TestMethod]
    public void Error_NdimMismatch()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Double);
        Action act = () => np.put_along_axis(a, np.array(new long[] { 0, 1 }), (NDArray)9, axis: 1);
        act.Should().Throw<ValueError>().WithMessage("`indices` and `arr` must have the same number of dimensions");
    }

    [TestMethod]
    public void Error_AxisOutOfRange()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Double);
        Action hi = () => np.put_along_axis(a, np.array(new int[,] { { 0 } }), (NDArray)9, axis: 5);
        hi.Should().Throw<AxisError>().WithMessage("*axis 5 is out of bounds for array of dimension 2*");
        Action lo = () => np.put_along_axis(a, np.array(new int[,] { { 0 } }), (NDArray)9, axis: -3);
        lo.Should().Throw<AxisError>().WithMessage("*axis -3 is out of bounds for array of dimension 2*");
    }

    [TestMethod]
    public void Error_NonAxisBroadcastConflict()
    {
        var a = np.zeros(new Shape(2, 3), NPTypeCode.Double);
        Action act = () => np.put_along_axis(a, np.zeros(new Shape(3, 3), NPTypeCode.Int64), (NDArray)9, axis: 1);
        act.Should().Throw<IndexError>()
           .WithMessage("shape mismatch: indexing arrays could not be broadcast together with shapes (2,1) (3,3) ");
    }

    [TestMethod]
    public void Error_ReadOnlyDestination()
    {
        var ro = np.broadcast_to(np.zeros(new Shape(3), NPTypeCode.Double), new Shape(2, 3));   // read-only
        Action act = () => np.put_along_axis(ro, np.array(new int[,] { { 0 }, { 1 } }), (NDArray)9, axis: 1);
        act.Should().Throw<ValueError>().WithMessage("assignment destination is read-only");
    }

    [TestMethod]
    public void Error_EmptyAxis_OutOfBounds()
    {
        var a = np.zeros(new Shape(2, 0), NPTypeCode.Double);
        Action act = () => np.put_along_axis(a, np.zeros(new Shape(2, 1), NPTypeCode.Int64), (NDArray)9, axis: 1);
        act.Should().Throw<IndexError>().WithMessage("index 0 is out of bounds for axis 1 with size 0");
    }

    [TestMethod]
    public void Error_NullArguments()
    {
        var a = A();
        var idx = np.array(new int[,] { { 0 }, { 0 } });
        ((Action)(() => np.put_along_axis(null, idx, (NDArray)1, axis: 1))).Should().Throw<ArgumentNullException>();
        ((Action)(() => np.put_along_axis(a, null, (NDArray)1, axis: 1))).Should().Throw<ArgumentNullException>();
        ((Action)(() => np.put_along_axis(a, idx, null, axis: 1))).Should().Throw<ArgumentNullException>();
    }

    // ===================================================================
    // Empty
    // ===================================================================

    [TestMethod]
    public void Empty_ZeroJ_NoOp()
    {
        var a = np.arange(6).reshape(2, 3);
        np.put_along_axis(a, np.zeros(new Shape(2, 0), NPTypeCode.Int64), (NDArray)9, axis: 1);
        FlatL(a).Should().Equal(0, 1, 2, 3, 4, 5);   // untouched
    }

    // ===================================================================
    // Dtype coverage (all 15)
    // ===================================================================

    [TestMethod]
    public void Dtype_AllFifteen_ScatterSentinelIntoColumnOne()
    {
        // Put a sentinel into column 1 of a (1,3) zero array, per dtype; verify it lands.
        void Case(NPTypeCode tc, NDArray sentinel, long expected)
        {
            var a = np.zeros(new Shape(1, 3), tc);
            np.put_along_axis(a, np.array(new int[,] { { 1 } }), sentinel, axis: 1);
            a.astype(NPTypeCode.Int64).GetInt64(0, 1).Should().Be(expected);
        }

        Case(NPTypeCode.Boolean, (NDArray)true, 1);
        Case(NPTypeCode.Byte, (NDArray)(byte)200, 200);
        Case(NPTypeCode.SByte, (NDArray)(sbyte)(-5), -5);
        Case(NPTypeCode.Int16, (NDArray)(short)(-300), -300);
        Case(NPTypeCode.UInt16, (NDArray)(ushort)60000, 60000);
        Case(NPTypeCode.Int32, (NDArray)(-70000), -70000);
        Case(NPTypeCode.UInt32, (NDArray)(uint)4000000000, 4000000000);
        Case(NPTypeCode.Int64, (NDArray)(-5000000000L), -5000000000L);
        Case(NPTypeCode.UInt64, (NDArray)(ulong)9000000000000000000, 9000000000000000000L);
        Case(NPTypeCode.Char, (NDArray)'A', 65);

        // Float / decimal / complex — check by their own dtype (value-exact).
        var s = np.zeros(new Shape(1, 3), NPTypeCode.Single);
        np.put_along_axis(s, np.array(new int[,] { { 2 } }), (NDArray)3.5f, axis: 1);
        ((float)s.GetValue(0, 2)).Should().Be(3.5f);

        var db = np.zeros(new Shape(1, 3), NPTypeCode.Double);
        np.put_along_axis(db, np.array(new int[,] { { 2 } }), (NDArray)3.25, axis: 1);
        ((double)db.GetValue(0, 2)).Should().Be(3.25);

        var h = np.zeros(new Shape(1, 3), NPTypeCode.Half);
        np.put_along_axis(h, np.array(new int[,] { { 2 } }), (NDArray)(Half)2.5f, axis: 1);
        ((Half)h.GetValue(0, 2)).Should().Be((Half)2.5f);

        var m = np.zeros(new Shape(1, 3), NPTypeCode.Decimal);
        np.put_along_axis(m, np.array(new int[,] { { 2 } }), (NDArray)1.5m, axis: 1);
        ((decimal)m.GetValue(0, 2)).Should().Be(1.5m);

        var cx = np.zeros(new Shape(1, 3), NPTypeCode.Complex);
        np.put_along_axis(cx, np.array(new int[,] { { 2 } }), (NDArray)new Complex(1, 2), axis: 1);
        ((Complex)cx.GetValue(0, 2)).Should().Be(new Complex(1, 2));
    }
}
