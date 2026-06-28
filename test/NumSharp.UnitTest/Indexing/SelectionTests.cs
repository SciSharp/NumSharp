using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for the np.* selection family: <c>take</c>, <c>put</c>, <c>place</c>,
/// <c>extract</c>, <c>compress</c>. The first three are IL-kernel-backed and
/// dtype-agnostic via byte-level <c>cpblk</c>; <c>extract</c> and
/// <c>compress</c> compose <c>flatnonzero</c> + <c>take</c>. Test buckets:
/// <list type="bullet">
///   <item>Per-dtype coverage on all 15 supported types (one round-trip).</item>
///   <item>Axis variations for <c>take</c> (None / 0 / 1 / -1 / -2).</item>
///   <item>Mode variations (raise / wrap / clip) on <c>take</c> and <c>put</c>.</item>
///   <item>Shape preservation, 0-d input, empty inputs.</item>
///   <item>OOB raise diagnostics matching NumPy's "out of bounds" messages.</item>
///   <item>Cycling: put values shorter than indices, place vals shorter than mask trues.</item>
///   <item>Round-trip consistency: take ∘ put should restore original at indexed positions.</item>
///   <item>extract: bool/int/float condition, multi-dim ravel, size mismatch.</item>
///   <item>compress: axis None/0/1/-1, 1-D validation, truncation, out= dispatch.</item>
/// </list>
/// </summary>
[TestClass]
public class SelectionTests
{
    // =================================================================
    // np.take — per-dtype coverage
    // =================================================================

    [TestMethod]
    public void Take_Boolean_RoundTrip()
    {
        var a = np.array(new bool[] { false, true, false, true, false });
        var r = np.take(a, np.array(new int[] { 0, 1, 4 }));
        r.ToArray<bool>().Should().Equal(false, true, false);
    }

    [TestMethod]
    public void Take_Byte_RoundTrip()
    {
        var a = np.array(new byte[] { 1, 2, 3, 4, 5 });
        var r = np.take(a, np.array(new int[] { 0, 2, 4 }));
        r.ToArray<byte>().Should().Equal((byte)1, (byte)3, (byte)5);
    }

    [TestMethod]
    public void Take_SByte_RoundTrip()
    {
        var a = np.array(new sbyte[] { -1, 2, -3, 4, -5 });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<sbyte>().Should().Equal((sbyte)(-1), (sbyte)(-3));
    }

    [TestMethod]
    public void Take_Int16_RoundTrip()
    {
        var a = np.array(new short[] { 100, 200, 300, 400 });
        var r = np.take(a, np.array(new int[] { 1, 3 }));
        r.ToArray<short>().Should().Equal((short)200, (short)400);
    }

    [TestMethod]
    public void Take_UInt16_RoundTrip()
    {
        var a = np.array(new ushort[] { 100, 200, 300, 400 });
        var r = np.take(a, np.array(new int[] { 1, 3 }));
        r.ToArray<ushort>().Should().Equal((ushort)200, (ushort)400);
    }

    [TestMethod]
    public void Take_Int32_RoundTrip()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var r = np.take(a, np.array(new int[] { 0, 2, 4 }));
        r.ToArray<int>().Should().Equal(10, 30, 50);
    }

    [TestMethod]
    public void Take_UInt32_RoundTrip()
    {
        var a = np.array(new uint[] { 10, 20, 30, 40 });
        var r = np.take(a, np.array(new int[] { 3, 1, 0 }));
        r.ToArray<uint>().Should().Equal(40u, 20u, 10u);
    }

    [TestMethod]
    public void Take_Int64_RoundTrip()
    {
        var a = np.array(new long[] { 100, 200, 300 });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<long>().Should().Equal(100L, 300L);
    }

    [TestMethod]
    public void Take_UInt64_RoundTrip()
    {
        var a = np.array(new ulong[] { 100, 200, 300 });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<ulong>().Should().Equal(100UL, 300UL);
    }

    [TestMethod]
    public void Take_Char_RoundTrip()
    {
        var a = np.array(new char[] { 'a', 'b', 'c', 'd' });
        var r = np.take(a, np.array(new int[] { 1, 3 }));
        r.ToArray<char>().Should().Equal('b', 'd');
    }

    [TestMethod]
    public void Take_Half_RoundTrip()
    {
        var a = np.array(new Half[] { (Half)1.5, (Half)2.5, (Half)3.5 });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<Half>().Should().Equal((Half)1.5, (Half)3.5);
    }

    [TestMethod]
    public void Take_Single_RoundTrip()
    {
        var a = np.array(new float[] { 1.5f, 2.5f, 3.5f });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<float>().Should().Equal(1.5f, 3.5f);
    }

    [TestMethod]
    public void Take_Double_RoundTrip()
    {
        var a = np.array(new double[] { 1.5, 2.5, 3.5 });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<double>().Should().Equal(1.5, 3.5);
    }

    [TestMethod]
    public void Take_Decimal_RoundTrip()
    {
        var a = np.array(new decimal[] { 1.5m, 2.5m, 3.5m });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<decimal>().Should().Equal(1.5m, 3.5m);
    }

    [TestMethod]
    public void Take_Complex_RoundTrip()
    {
        var a = np.array(new Complex[]
        {
            new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)
        });
        var r = np.take(a, np.array(new int[] { 0, 2 }));
        r.ToArray<Complex>().Should().Equal(new Complex(1, 2), new Complex(5, 6));
    }

    // =================================================================
    // np.take — axis variations
    // =================================================================

    [TestMethod]
    public void Take_2D_Axis0_NumPyParity()
    {
        // a = [[10,20,30],[40,50,60]]; take rows 0,1,0 → (3, 3) shape.
        var a = np.array(new int[,] { { 10, 20, 30 }, { 40, 50, 60 } });
        var r = np.take(a, np.array(new int[] { 0, 1, 0 }), axis: 0);
        r.shape.Should().Equal(3, 3);
        for (int j = 0; j < 3; j++) r.GetInt32(0, j).Should().Be(10 + j * 10);
        for (int j = 0; j < 3; j++) r.GetInt32(1, j).Should().Be(40 + j * 10);
        for (int j = 0; j < 3; j++) r.GetInt32(2, j).Should().Be(10 + j * 10);
    }

    [TestMethod]
    public void Take_2D_Axis1_NumPyParity()
    {
        var a = np.array(new int[,] { { 10, 20, 30 }, { 40, 50, 60 } });
        var r = np.take(a, np.array(new int[] { 2, 1 }), axis: 1);
        r.shape.Should().Equal(2, 2);
        r.GetInt32(0, 0).Should().Be(30);
        r.GetInt32(0, 1).Should().Be(20);
        r.GetInt32(1, 0).Should().Be(60);
        r.GetInt32(1, 1).Should().Be(50);
    }

    [TestMethod]
    public void Take_2D_NegativeAxis_NumPyParity()
    {
        // axis=-1 equivalent to axis=1 for 2-D.
        var a = np.array(new int[,] { { 10, 20, 30 }, { 40, 50, 60 } });
        var r = np.take(a, np.array(new int[] { 2, 1 }), axis: -1);
        r.shape.Should().Equal(2, 2);
        r.GetInt32(0, 0).Should().Be(30);
        r.GetInt32(1, 1).Should().Be(50);
    }

    [TestMethod]
    public void Take_2D_AxisNone_FlattensSource()
    {
        var a = np.array(new int[,] { { 10, 20, 30 }, { 40, 50, 60 } });
        // C-order flat: [10,20,30,40,50,60]; take [0,3,5] → [10,40,60].
        var r = np.take(a, np.array(new int[] { 0, 3, 5 }));
        r.ToArray<int>().Should().Equal(10, 40, 60);
    }

    [TestMethod]
    public void Take_2D_2DIndices_ShapePreserved()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var r = np.take(a, np.array(new int[,] { { 0, 1 }, { 2, 3 } }));
        r.shape.Should().Equal(2, 2);
        r.GetInt32(0, 0).Should().Be(10);
        r.GetInt32(0, 1).Should().Be(20);
        r.GetInt32(1, 0).Should().Be(30);
        r.GetInt32(1, 1).Should().Be(40);
    }

    [TestMethod]
    public void Take_0d_Source_ScalarIdx_ReturnsScalar()
    {
        var a = NDArray.Scalar(5);
        var r = np.take(a, 0L);
        r.size.Should().Be(1);
        r.GetInt32(0).Should().Be(5);
    }

    [TestMethod]
    public void Take_0d_Source_1eltIdx_Returns1D()
    {
        var a = NDArray.Scalar(5);
        var r = np.take(a, np.array(new int[] { 0 }));
        r.shape.Should().Equal(1);
        r.GetInt32(0).Should().Be(5);
    }

    [TestMethod]
    public void Take_Empty_Indices_ReturnsEmpty()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var r = np.take(a, np.array(new int[0]));
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Take_OOB_Raise_Throws()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var act = () => np.take(a, np.array(new int[] { 10 }));
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    public void Take_Wrap_MultiPeriod()
    {
        // 10 in size=5 with wrap: 10 - 5 = 5 ≥ 5 → fallback to %: 10 % 5 = 0.
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var r = np.take(a, np.array(new int[] { 10 }), mode: "wrap");
        r.ToArray<int>().Should().Equal(10);
    }

    [TestMethod]
    public void Take_Clip_BothBounds()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var r = np.take(a, np.array(new int[] { 100, -100 }), mode: "clip");
        r.ToArray<int>().Should().Equal(50, 10);
    }

    [TestMethod]
    public void Take_NonContig_Source_MaterializesAndWorks()
    {
        // Reverse view via [::-1] is non-contig.
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var rev = a["::-1"];
        var r = np.take(rev, np.array(new int[] { 0, 2 }));
        r.ToArray<int>().Should().Equal(50, 30);
    }

    [TestMethod]
    public void Take_InvalidAxis_Throws()
    {
        var a = np.array(new int[] { 10, 20 });
        var act = () => np.take(a, np.array(new int[] { 0 }), axis: 5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ── out= parameter (NumPy parity) ────────────────────────────────

    [TestMethod]
    public void Take_OutParam_ReturnsSameReference_AndFillsValues()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var idx = np.array(new int[] { 0, 2, 4 });
        var outArr = np.zeros<int>(new int[] { 3 });
        var r = np.take(a, idx, @out: outArr);
        ReferenceEquals(r, outArr).Should().BeTrue("out= must return the supplied buffer");
        outArr.ToArray<int>().Should().Equal(10, 30, 50);
    }

    [TestMethod]
    public void Take_OutParam_DtypeCast_FillsWithCastedValues()
    {
        // NumPy out= permits unsafe writeback IF the source's dtype can be safely
        // cast back from out's dtype (i.e. can_cast(out, src, "safe") = True).
        // src=float64, out=int32 satisfies that direction (int32→float64 is safe),
        // so the writeback truncates values into int32.
        var a = np.array(new double[] { 10.7, 20.5, 30.0 });
        var outInt = np.zeros<int>(new int[] { 2 });
        np.take(a, np.array(new int[] { 0, 2 }), @out: outInt);
        outInt.ToArray<int>().Should().Equal(10, 30);
    }

    [TestMethod]
    public void Take_OutParam_ShapeMismatch_Throws()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var outBad = np.zeros<int>(new int[] { 5 });
        var act = () => np.take(a, np.array(new int[] { 0 }), @out: outBad);
        act.Should().Throw<ArgumentException>().WithMessage("*output array does not match*");
    }

    [TestMethod]
    public void Take_OutParam_2D_PreservesShape()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var idx = np.array(new int[,] { { 0, 2 }, { 4, 1 } });
        var outArr = np.zeros<int>(new int[] { 2, 2 });
        np.take(a, idx, @out: outArr);
        outArr.GetInt32(0, 0).Should().Be(10);
        outArr.GetInt32(0, 1).Should().Be(30);
        outArr.GetInt32(1, 0).Should().Be(50);
        outArr.GetInt32(1, 1).Should().Be(20);
    }

    // =================================================================
    // np.put — basic, broadcasting, modes
    // =================================================================

    [TestMethod]
    public void Put_Basic_ExactPairing()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        np.put(a, np.array(new int[] { 0, 2 }), np.array(new int[] { 100, 200 }));
        a.ToArray<int>().Should().Equal(100, 20, 200, 40, 50);
    }

    [TestMethod]
    public void Put_Cycle_ValuesShorter()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        np.put(a, np.array(new int[] { 0, 1, 2, 3 }), np.array(new int[] { 100, 200 }));
        a.ToArray<int>().Should().Equal(100, 200, 100, 200, 50);
    }

    [TestMethod]
    public void Put_Cycle_SingleValue()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        np.put(a, np.array(new int[] { 0, 2, 4 }), np.array(new int[] { 99 }));
        a.ToArray<int>().Should().Equal(99, 20, 99, 40, 99);
    }

    [TestMethod]
    public void Put_2D_FlatIndexing()
    {
        var a = np.array(new int[,] { { 10, 20 }, { 30, 40 } });
        np.put(a, np.array(new int[] { 0, 3 }), np.array(new int[] { 99, 88 }));
        a.GetInt32(0, 0).Should().Be(99);
        a.GetInt32(0, 1).Should().Be(20);
        a.GetInt32(1, 0).Should().Be(30);
        a.GetInt32(1, 1).Should().Be(88);
    }

    [TestMethod]
    public void Put_Wrap_MultiPeriod()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        np.put(a, np.array(new int[] { 5 }), np.array(new int[] { 99 }), mode: "wrap");
        a.ToArray<int>().Should().Equal(10, 20, 99);
    }

    [TestMethod]
    public void Put_Clip_Saturates()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        np.put(a, np.array(new int[] { 5, -10 }), np.array(new int[] { 99, 88 }), mode: "clip");
        a.ToArray<int>().Should().Equal(88, 20, 99);
    }

    [TestMethod]
    public void Put_Raise_OOBThrows()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var act = () => np.put(a, np.array(new int[] { 5 }), np.array(new int[] { 99 }));
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [TestMethod]
    public void Put_EmptyIndices_NoOp()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        np.put(a, np.array(new int[0]), np.array(new int[] { 99 }));
        a.ToArray<int>().Should().Equal(10, 20, 30);
    }

    [TestMethod]
    public void Put_EmptyValues_WithIndices_Throws()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var act = () => np.put(a, np.array(new int[] { 0 }), np.array(new int[0]));
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Put_EmptyArray_WithIndices_Throws()
    {
        var a = np.array(new int[0]);
        var act = () => np.put(a, np.array(new int[] { 0 }), np.array(new int[] { 99 }));
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [TestMethod]
    public void Put_DtypeCast_FloatIntoIntArray()
    {
        // values are cast to a's dtype.
        var a = np.array(new int[] { 0, 0, 0 });
        np.put(a, np.array(new int[] { 0, 2 }), np.array(new double[] { 1.5, 2.5 }));
        // Truncated to int.
        a.GetInt32(0).Should().Be(1);
        a.GetInt32(2).Should().Be(2);
    }

    [TestMethod]
    public void Put_DuplicateIndices_LastWriteWins()
    {
        // NumPy parity: when the same flat index appears multiple times in
        // indices, the value paired with the last occurrence wins.
        var a = np.array(new int[] { 10, 20, 30 });
        np.put(a, np.array(new int[] { 0, 0, 0 }), np.array(new int[] { 100, 200, 300 }));
        a.ToArray<int>().Should().Equal(300, 20, 30);
    }

    [TestMethod]
    public void Put_NonContig_Target_PropagatesToParent()
    {
        // NumPy WRITEBACKIFCOPY semantics: writes to a non-contig view propagate
        // back to the parent's storage at the view-translated positions.
        var a = np.array(new int[,]
        {
            { 0,  1,  2,  3,  4 },
            { 5,  6,  7,  8,  9 },
            { 10, 11, 12, 13, 14 },
            { 15, 16, 17, 18, 19 }
        });
        var aSlice = a["1::2, :"];   // rows 1, 3 — shape (2, 5), non-contig
        aSlice.Shape.IsContiguous.Should().BeFalse();

        np.put(aSlice, np.array(new int[] { 0, 5 }), np.array(new int[] { 100, 200 }));

        // View sees the writes
        aSlice.GetInt32(0, 0).Should().Be(100);
        aSlice.GetInt32(1, 0).Should().Be(200);
        // Parent sees them too — rows 1 and 3 of the original
        a.GetInt32(1, 0).Should().Be(100);
        a.GetInt32(3, 0).Should().Be(200);
        // Other parent rows untouched
        a.GetInt32(0, 0).Should().Be(0);
        a.GetInt32(2, 0).Should().Be(10);
    }

    // =================================================================
    // np.place — basic, broadcasting, edge cases
    // =================================================================

    [TestMethod]
    public void Place_Basic_ExactPairing()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new bool[] { true, false, true, false, true });
        np.place(a, mask, np.array(new int[] { 100, 200, 300 }));
        a.ToArray<int>().Should().Equal(100, 20, 200, 40, 300);
    }

    [TestMethod]
    public void Place_Cycle_SingleVal()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var mask = np.array(new bool[] { true, false, true, false, true });
        np.place(a, mask, np.array(new int[] { 99 }));
        a.ToArray<int>().Should().Equal(99, 20, 99, 40, 99);
    }

    [TestMethod]
    public void Place_Cycle_ValsShorterThanTrues()
    {
        // 4 trues, 2 vals → cycle [v0, v1, v0, v1].
        var a = np.array(new int[] { 1, 2, 3, 4, 5, 6 });
        var mask = np.array(new bool[] { true, false, true, true, false, true });
        np.place(a, mask, np.array(new int[] { 10, 20 }));
        a.ToArray<int>().Should().Equal(10, 2, 20, 10, 5, 20);
    }

    [TestMethod]
    public void Place_2D_FlatMaskWalk()
    {
        var a = np.array(new int[,] { { 10, 20 }, { 30, 40 } });
        // mask flat: [F, T, T, T] (where elements > 15)
        var mask = a > 15;
        np.place(a, mask, np.array(new int[] { 99, 88, 77 }));
        a.GetInt32(0, 0).Should().Be(10);
        a.GetInt32(0, 1).Should().Be(99);
        a.GetInt32(1, 0).Should().Be(88);
        a.GetInt32(1, 1).Should().Be(77);
    }

    [TestMethod]
    public void Place_NoTrues_NoOp()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var mask = np.array(new bool[] { false, false, false });
        np.place(a, mask, np.array(new int[] { 99 }));
        a.ToArray<int>().Should().Equal(10, 20, 30);
    }

    [TestMethod]
    public void Place_EmptyVals_WithTrues_Throws()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var mask = np.array(new bool[] { true, false, true });
        var act = () => np.place(a, mask, np.array(new int[0]));
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Place_MaskSizeMismatch_Throws()
    {
        var a = np.array(new int[] { 10, 20, 30 });
        var mask = np.array(new bool[] { true, false });   // wrong size
        var act = () => np.place(a, mask, np.array(new int[] { 99 }));
        act.Should().Throw<ArgumentException>();
    }

    [TestMethod]
    public void Place_IntMask_TreatedAsTruthy()
    {
        // NumPy coerces int masks to bool via non-zero → True.
        var a = np.array(new int[] { 10, 20, 30, 40 });
        var imask = np.array(new int[] { 0, 1, 2, 3 });   // first is False, rest truthy
        np.place(a, imask, np.array(new int[] { 99 }));
        a.ToArray<int>().Should().Equal(10, 99, 99, 99);
    }

    [TestMethod]
    public void Place_NonContig_Target_PropagatesToParent()
    {
        var a = np.array(new int[,]
        {
            { 0,  1,  2,  3,  4 },
            { 5,  6,  7,  8,  9 },
            { 10, 11, 12, 13, 14 },
            { 15, 16, 17, 18, 19 }
        });
        var aSlice = a["1::2, :"];   // (2, 5) non-contig view of rows 1, 3
        aSlice.Shape.IsContiguous.Should().BeFalse();

        np.place(aSlice, aSlice > 6, np.array(new int[] { 99 }));

        // After: slice values >6 replaced with 99. Slice = [[5,6,99,99,99],[99,99,99,99,99]].
        aSlice.GetInt32(0, 0).Should().Be(5);
        aSlice.GetInt32(0, 1).Should().Be(6);
        aSlice.GetInt32(0, 2).Should().Be(99);
        aSlice.GetInt32(1, 0).Should().Be(99);
        aSlice.GetInt32(1, 4).Should().Be(99);
        // Parent row 1 = [5, 6, 99, 99, 99]; row 3 = [99, 99, 99, 99, 99].
        a.GetInt32(1, 0).Should().Be(5);
        a.GetInt32(1, 2).Should().Be(99);
        a.GetInt32(3, 0).Should().Be(99);
        a.GetInt32(3, 4).Should().Be(99);
        // Untouched rows
        a.GetInt32(0, 0).Should().Be(0);
        a.GetInt32(2, 0).Should().Be(10);
    }

    [TestMethod]
    public void Place_0d_Arr_WritesScalar()
    {
        // NumPy accepts 0-d arrays — np.place(np.array(5), True, [99]) → 99.
        var a = NDArray.Scalar(5);
        np.place(a, NDArray.Scalar(true), np.array(new int[] { 99 }));
        a.GetInt32(0).Should().Be(99);
    }

    // =================================================================
    // Cross-validation: take ∘ put round-trip
    // =================================================================

    [TestMethod]
    public void TakePut_RoundTrip_RestoresValues()
    {
        var a = np.array(new int[] { 10, 20, 30, 40, 50 });
        var idx = np.array(new int[] { 0, 2, 4 });

        // Snapshot the indexed values, overwrite with 0, then put the snapshot back.
        var snapshot = np.take(a, idx);
        np.put(a, idx, np.array(new int[] { 0, 0, 0 }));
        a.ToArray<int>().Should().Equal(0, 20, 0, 40, 0);

        np.put(a, idx, snapshot);
        a.ToArray<int>().Should().Equal(10, 20, 30, 40, 50);
    }

    [TestMethod]
    public void Take_RandomConsistency_VsManualCompute()
    {
        // Random data + random indices, verify take matches manual flat indexing.
        var rng = new Random(42);
        int n = 100, m = 30;
        var data = new int[n];
        for (int i = 0; i < n; i++) data[i] = rng.Next(-1000, 1000);
        var indices = new int[m];
        for (int i = 0; i < m; i++) indices[i] = rng.Next(n);

        var a = np.array(data);
        var idx = np.array(indices);
        var r = np.take(a, idx);

        for (long i = 0; i < m; i++)
            r.GetInt32(i).Should().Be(data[indices[i]]);
    }

    // =================================================================
    // np.extract — composes flatnonzero + take(axis=None)
    // =================================================================

    [TestMethod]
    public void Extract_Basic_2DBoolCondition()
    {
        // Doc example: arr.ravel()[condition.ravel()]. np.arange defaults to int64.
        var arr = np.arange(12).reshape(3, 4);
        var cond = (arr % 3) == 0;
        var r = np.extract(cond, arr);
        r.Shape.Should().Be(new Shape(4));
        r.ToArray<long>().Should().Equal(0L, 3L, 6L, 9L);
    }

    [TestMethod]
    public void Extract_1DCondAgainst2DArr_Ravels()
    {
        var arr = np.arange(12).reshape(3, 4);
        var cond = np.array(new bool[] { false, true, false, true, true, false });
        var r = np.extract(cond, arr);
        // ravel(arr) = [0..11]; True at idx 1,3,4 → [1, 3, 4]
        r.ToArray<long>().Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Extract_ShorterCond_TruncatesByAlignment()
    {
        var cond = np.array(new bool[] { true, false, true });
        var arr = np.arange(10);
        var r = np.extract(cond, arr);
        r.ToArray<long>().Should().Equal(0L, 2L);
    }

    [TestMethod]
    public void Extract_0DSource()
    {
        var r = np.extract(np.array(new bool[] { true }), NDArray.Scalar(7));
        r.Shape.Should().Be(new Shape(1));
        r.GetInt32(0).Should().Be(7);
    }

    [TestMethod]
    public void Extract_AllFalse_EmptyResult()
    {
        var r = np.extract(np.array(new bool[] { false, false, false, false, false }), np.arange(5));
        r.size.Should().Be(0);
        r.ndim.Should().Be(1);
    }

    [TestMethod]
    public void Extract_IntCondition_TreatedAsNonzero()
    {
        // Negative & nonzero ints both count as True.
        var cond = np.array(new int[] { 1, 0, -3, 0, 5 });
        var r = np.extract(cond, np.arange(5));
        r.ToArray<long>().Should().Equal(0L, 2L, 4L);
    }

    [TestMethod]
    public void Extract_FloatCondition_NonzeroIsTrue()
    {
        var cond = np.array(new double[] { 0.0, 1.5, 0.0, 0.5, 0.0 });
        var r = np.extract(cond, np.arange(5));
        r.ToArray<long>().Should().Equal(1L, 3L);
    }

    [TestMethod]
    public void Extract_2DCondAnd2DArr_RavelsBoth()
    {
        var cond = np.array(new bool[,] { { true, false }, { true, true } });
        var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var r = np.extract(cond, arr);
        r.ndim.Should().Be(1);
        r.ToArray<int>().Should().Equal(1, 3, 4);
    }

    [TestMethod]
    public void Extract_EmptyCond_EmptyResult()
    {
        var r = np.extract(np.array(new bool[] { }), np.arange(5));
        r.size.Should().Be(0);
        r.ndim.Should().Be(1);
    }

    [TestMethod]
    public void Extract_LongerCondWithOOBTrue_Raises()
    {
        // Cond size 20 but arr size 5; True at idx 15 → OOB.
        var bigCond = new bool[20];
        bigCond[0] = true;
        bigCond[15] = true;
        var action = () => np.extract(np.array(bigCond), np.arange(5));
        action.Should().Throw<Exception>()
            .Where(e => e is IndexOutOfRangeException || e is ArgumentOutOfRangeException);
    }

    [TestMethod]
    public void Extract_LongerCondAllTruesInRange_OK()
    {
        // Cond longer than arr but all True positions are < arr.size → no error.
        var cond = new bool[10];
        cond[0] = true; cond[2] = true;
        var r = np.extract(np.array(cond), np.arange(5));
        r.ToArray<long>().Should().Equal(0L, 2L);
    }

    [TestMethod]
    public void Extract_NonContigSource_View()
    {
        var src = np.arange(20).reshape(4, 5);
        var view = src["::2, ::2"]; // shape (2, 3), non-contig
        var cond = np.array(new bool[] { true, false, true, false, true, true });
        var r = np.extract(cond, view);
        // ravel(view) = [0, 2, 4, 10, 12, 14]; trues at 0,2,4,5 → [0, 4, 12, 14]
        r.ToArray<long>().Should().Equal(0L, 4L, 12L, 14L);
    }

    [TestMethod]
    public void Extract_DtypePreservation_Float()
    {
        var r = np.extract(np.array(new bool[] { true, false }), np.array(new double[] { 1.5, 2.5 }));
        r.dtype.Should().Be(typeof(double));
        r.GetDouble(0).Should().Be(1.5);
    }

    [TestMethod]
    public void Extract_AllDtypes_Smoke()
    {
        // One round-trip on each of the 15 dtypes; relies on take's dtype-agnostic kernel.
        var cond = np.array(new bool[] { true, false, true });

        np.extract(cond, np.array(new bool[] { true, false, true })).ToArray<bool>().Should().Equal(true, true);
        np.extract(cond, np.array(new byte[] { 1, 2, 3 })).ToArray<byte>().Should().Equal((byte)1, (byte)3);
        np.extract(cond, np.array(new sbyte[] { -1, 2, -3 })).ToArray<sbyte>().Should().Equal((sbyte)(-1), (sbyte)(-3));
        np.extract(cond, np.array(new short[] { 10, 20, 30 })).ToArray<short>().Should().Equal((short)10, (short)30);
        np.extract(cond, np.array(new ushort[] { 10, 20, 30 })).ToArray<ushort>().Should().Equal((ushort)10, (ushort)30);
        np.extract(cond, np.array(new int[] { 100, 200, 300 })).ToArray<int>().Should().Equal(100, 300);
        np.extract(cond, np.array(new uint[] { 100, 200, 300 })).ToArray<uint>().Should().Equal(100u, 300u);
        np.extract(cond, np.array(new long[] { 1000, 2000, 3000 })).ToArray<long>().Should().Equal(1000L, 3000L);
        np.extract(cond, np.array(new ulong[] { 1000, 2000, 3000 })).ToArray<ulong>().Should().Equal(1000UL, 3000UL);
        np.extract(cond, np.array(new char[] { 'a', 'b', 'c' })).ToArray<char>().Should().Equal('a', 'c');
        np.extract(cond, np.array(new Half[] { (Half)1, (Half)2, (Half)3 })).ToArray<Half>().Should().Equal((Half)1, (Half)3);
        np.extract(cond, np.array(new float[] { 1f, 2f, 3f })).ToArray<float>().Should().Equal(1f, 3f);
        np.extract(cond, np.array(new double[] { 1.0, 2.0, 3.0 })).ToArray<double>().Should().Equal(1.0, 3.0);
        np.extract(cond, np.array(new decimal[] { 1m, 2m, 3m })).ToArray<decimal>().Should().Equal(1m, 3m);
        np.extract(cond, np.array(new Complex[] { new(1, 2), new(3, 4), new(5, 6) })).ToArray<Complex>()
            .Should().Equal(new Complex(1, 2), new Complex(5, 6));
    }

    [TestMethod]
    public void Extract_NullArgs_Throws()
    {
        var arr = np.arange(5);
        var cond = np.array(new bool[] { true });
        ((Action)(() => np.extract(null, arr))).Should().Throw<ArgumentNullException>();
        ((Action)(() => np.extract(cond, null))).Should().Throw<ArgumentNullException>();
    }

    // =================================================================
    // np.compress — validates 1-D, delegates to flatnonzero + take(axis)
    // =================================================================

    [TestMethod]
    public void Compress_Axis0_IntCondition()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new int[] { 0, 1, 0 }), a, axis: 0);
        r.Shape.Should().Be(new Shape(1, 2));
        np.ravel(r).ToArray<int>().Should().Equal(3, 4);
    }

    [TestMethod]
    public void Compress_Axis0_BoolCondition()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { false, true, true }), a, axis: 0);
        r.Shape.Should().Be(new Shape(2, 2));
        np.ravel(r).ToArray<int>().Should().Equal(3, 4, 5, 6);
    }

    [TestMethod]
    public void Compress_Axis1()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { false, true }), a, axis: 1);
        r.Shape.Should().Be(new Shape(3, 1));
        np.ravel(r).ToArray<int>().Should().Equal(2, 4, 6);
    }

    [TestMethod]
    public void Compress_AxisNone_FlattensFirst()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        // axis=None: ravel(a) = [1,2,3,4,5,6]; cond [F,T] of len 2 → True at idx 1 → [2].
        var r = np.compress(np.array(new bool[] { false, true }), a);
        r.Shape.Should().Be(new Shape(1));
        r.GetInt32(0).Should().Be(2);
    }

    [TestMethod]
    public void Compress_AxisNegative()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { false, true }), a, axis: -1);
        r.Shape.Should().Be(new Shape(3, 1));
        np.ravel(r).ToArray<int>().Should().Equal(2, 4, 6);
    }

    [TestMethod]
    public void Compress_ShorterCond_Truncates()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { true }), a, axis: 0);
        r.Shape.Should().Be(new Shape(1, 2));
        np.ravel(r).ToArray<int>().Should().Equal(1, 2);
    }

    [TestMethod]
    public void Compress_LongerCondWithOOBTrue_Raises()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        // axis=1 has size 2; cond has 4 Trues; index 2,3 OOB.
        var action = () => np.compress(np.array(new bool[] { true, true, true, true }), a, axis: 1);
        action.Should().Throw<Exception>()
            .Where(e => e is IndexOutOfRangeException || e is ArgumentOutOfRangeException);
    }

    [TestMethod]
    public void Compress_TwoDimCondition_Raises()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var action = () => np.compress(np.array(new bool[,] { { true, false }, { true, true } }), a, axis: 0);
        action.Should().Throw<ArgumentException>().WithMessage("*condition must be a 1-d array*");
    }

    [TestMethod]
    public void Compress_ZeroDimCondition_Raises()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var action = () => np.compress(NDArray.Scalar(true), a, axis: 0);
        action.Should().Throw<ArgumentException>().WithMessage("*condition must be a 1-d array*");
    }

    [TestMethod]
    public void Compress_FloatCondition()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new double[] { 0.0, 1.5, 0.0 }), a, axis: 0);
        r.Shape.Should().Be(new Shape(1, 2));
        np.ravel(r).ToArray<int>().Should().Equal(3, 4);
    }

    [TestMethod]
    public void Compress_EmptyCond_Axis0_RetainsOtherDims()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { }), a, axis: 0);
        r.Shape.Should().Be(new Shape(0, 2));
    }

    [TestMethod]
    public void Compress_EmptyCond_AxisNone()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { }), a);
        r.Shape.Should().Be(new Shape(0));
    }

    [TestMethod]
    public void Compress_AllFalse_Axis0()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var r = np.compress(np.array(new bool[] { false, false, false }), a, axis: 0);
        r.Shape.Should().Be(new Shape(0, 2));
    }

    [TestMethod]
    public void Compress_OutOfBoundsAxis_Raises()
    {
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
        var action = () => np.compress(np.array(new bool[] { true }), a, axis: -3);
        action.Should().Throw<ArgumentOutOfRangeException>();
    }

    [TestMethod]
    public void Compress_OutDispatch_ReturnsOutWithCorrectDtype()
    {
        // out.dtype must be safely castable to src.dtype (NumPy rule —
        // mirrors PyArray_TakeFrom's WRITEBACKIFCOPY scratch init via
        // PyArray_FromArray(out, src_dtype, ...)). Here src=int64, out=int32
        // → can_cast(int32, int64, "safe") = True, so this is allowed.
        var a = np.array(new long[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var outArr = np.zeros(new Shape(2, 2), NPTypeCode.Int32);
        var r = np.compress(np.array(new bool[] { true, false, true }), a, axis: 0, @out: outArr);
        ReferenceEquals(r, outArr).Should().BeTrue();
        r.dtype.Should().Be(typeof(int));
        np.ravel(r).ToArray<int>().Should().Equal(1, 2, 5, 6);
    }

    [TestMethod]
    public void Compress_OutDispatch_UnsafeCastDirection_Raises()
    {
        // out.dtype int64 cannot be safely cast to src.dtype int32 — NumPy
        // raises TypeError with this exact message.
        var a = np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var outArr = np.zeros(new Shape(2, 2), NPTypeCode.Int64);
        var action = () => np.compress(np.array(new bool[] { true, false, true }), a, axis: 0, @out: outArr);
        action.Should().Throw<TypeError>()
            .WithMessage("Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'");
    }

    [TestMethod]
    public void Compress_OutDispatch_FloatToInt_AllowedUnsafeWriteback()
    {
        // src=float64, out=int32 → can_cast(int32, float64, "safe") = True, so
        // NumPy permits this even though the writeback truncates. Values:
        // float64 src [0.5,1.5,...,8.5] → take rows 0,2 → [0.5,1.5,2.5,6.5,7.5,8.5]
        // writeback to int32 truncates toward zero: [0,1,2,6,7,8].
        var src = np.arange(9, NPTypeCode.Double).reshape(3, 3) + 0.5;
        var outArr = np.zeros(new Shape(2, 3), NPTypeCode.Int32);
        var r = np.compress(np.array(new bool[] { true, false, true }), src, axis: 0, @out: outArr);
        r.dtype.Should().Be(typeof(int));
        np.ravel(r).ToArray<int>().Should().Equal(0, 1, 2, 6, 7, 8);
    }

    [TestMethod]
    public void Compress_ZeroDimSource_AxisNone()
    {
        var r = np.compress(np.array(new bool[] { true }), NDArray.Scalar(5));
        r.Shape.Should().Be(new Shape(1));
        r.GetInt32(0).Should().Be(5);
    }

    [TestMethod]
    public void Compress_NullArgs_Throws()
    {
        var a = np.arange(5);
        var cond = np.array(new bool[] { true });
        ((Action)(() => np.compress(null, a))).Should().Throw<ArgumentNullException>();
        ((Action)(() => np.compress(cond, null))).Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void Compress_NonContigSource_GathersCorrectly()
    {
        // Sliced source along axis=0; compress should hit the WRITEBACKIFCOPY path
        // inside take (when needed) and produce correct results.
        var src = np.arange(24).reshape(6, 4);
        var view = src["::2"]; // shape (3, 4), non-contig
        var r = np.compress(np.array(new bool[] { true, false, true }), view, axis: 0);
        r.Shape.Should().Be(new Shape(2, 4));
        // src[::2] rows: [0..3], [8..11], [16..19]; pick rows 0,2 → [0..3, 16..19]
        np.ravel(r).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 16L, 17L, 18L, 19L);
    }

    [TestMethod]
    public void Extract_TransposedSource_RavelsLogicalOrder()
    {
        // Transposed view of (3,4) → (4,3); ravel walks logical C-order.
        // src.T.ravel() = [0,4,8, 1,5,9, 2,6,10, 3,7,11].
        var src = np.arange(12, NPTypeCode.Int32).reshape(3, 4).T;
        src.Shape.IsContiguous.Should().BeFalse();
        var cond = np.array(new bool[] { true, false, true, false, true, false, true, false, true, false, true, false });
        var r = np.extract(cond, src);
        r.ToArray<int>().Should().Equal(0, 8, 5, 2, 10, 7);
    }

    [TestMethod]
    public void Extract_NegativeStrideSource()
    {
        var src = np.arange(10, NPTypeCode.Int32)["::-1"];
        var cond = np.array(new bool[] { true, false, true, false, true, false, true, false, true, false });
        var r = np.extract(cond, src);
        // src reversed = [9,8,...,0]; keep every other → [9,7,5,3,1]
        r.ToArray<int>().Should().Equal(9, 7, 5, 3, 1);
    }

    [TestMethod]
    public void Extract_BroadcastedSource()
    {
        var src = np.broadcast_to(np.array(new int[] { 1, 2, 3 }), new Shape(4, 3));
        src.Shape.IsBroadcasted.Should().BeTrue();
        var cond = np.array(Enumerable.Repeat(true, 12).ToArray());
        var r = np.extract(cond, src);
        r.ToArray<int>().Should().Equal(1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3);
    }

    [TestMethod]
    public void Extract_NaNConditionIsTruthy()
    {
        // NumPy treats NaN as nonzero (truthy) in mask interpretation.
        var cond = np.array(new double[] { double.NaN, 0.0, double.NaN });
        var arr = np.array(new int[] { 10, 20, 30 });
        var r = np.extract(cond, arr);
        r.ToArray<int>().Should().Equal(10, 30);
    }

    [TestMethod]
    public void Extract_ComplexConditionZeroIsFalse()
    {
        // Complex 0+0j is False; non-zero real OR imag → True.
        var cond = np.array(new Complex[] { new(0, 0), new(1, 0), new(0, 1), new(0, 0) });
        var arr = np.array(new int[] { 10, 20, 30, 40 });
        var r = np.extract(cond, arr);
        r.ToArray<int>().Should().Equal(20, 30);
    }

    [TestMethod]
    public void Extract_NonContigConditionView()
    {
        // Condition is a strided view of a larger buffer.
        var bigCond = np.array(new bool[] { true, false, false, true, false, true, true, false });
        var view = bigCond["::2"]; // [T, F, F, T]
        var arr = np.array(new int[] { 10, 20, 30, 40 });
        var r = np.extract(view, arr);
        r.ToArray<int>().Should().Equal(10, 40);
    }

    [TestMethod]
    public void Extract_0DConditionTrue()
    {
        // 0-d True cond: ravel gives 1-element 1-D; nonzero gives [0]; take arr[0].
        var r = np.extract(NDArray.Scalar(true), np.array(new int[] { 10, 20, 30 }));
        r.Shape.Should().Be(new Shape(1));
        r.GetInt32(0).Should().Be(10);
    }

    [TestMethod]
    public void Extract_0DConditionFalse_Empty()
    {
        var r = np.extract(NDArray.Scalar(false), np.array(new int[] { 10, 20, 30 }));
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Compress_TransposedSource_Axis0()
    {
        // T view of (3,4) is (4,3) non-contig; compress axis=0 selects logical rows.
        var src = np.arange(12, NPTypeCode.Int32).reshape(3, 4).T;
        var r = np.compress(np.array(new bool[] { true, false, true, false }), src, axis: 0);
        r.Shape.Should().Be(new Shape(2, 3));
        np.ravel(r).ToArray<int>().Should().Equal(0, 4, 8, 2, 6, 10);
    }

    [TestMethod]
    public void Compress_TransposedSource_Axis1()
    {
        var src = np.arange(12, NPTypeCode.Int32).reshape(3, 4).T; // (4, 3)
        var r = np.compress(np.array(new bool[] { true, false, true }), src, axis: 1);
        r.Shape.Should().Be(new Shape(4, 2));
        np.ravel(r).ToArray<int>().Should().Equal(0, 8, 1, 9, 2, 10, 3, 11);
    }

    [TestMethod]
    public void Compress_NegativeStrideSource_Axis0()
    {
        var src = np.arange(20, NPTypeCode.Int32).reshape(4, 5)["::-1"];
        // src is reversed-row view: [[15..19],[10..14],[5..9],[0..4]]
        var r = np.compress(np.array(new bool[] { true, false, true, false }), src, axis: 0);
        r.Shape.Should().Be(new Shape(2, 5));
        np.ravel(r).ToArray<int>().Should().Equal(15, 16, 17, 18, 19, 5, 6, 7, 8, 9);
    }

    [TestMethod]
    public void Compress_BroadcastedSource()
    {
        var src = np.broadcast_to(np.arange(3, NPTypeCode.Int32), new Shape(4, 3));
        var r = np.compress(np.array(new bool[] { true, false, true, false }), src, axis: 0);
        r.Shape.Should().Be(new Shape(2, 3));
        np.ravel(r).ToArray<int>().Should().Equal(0, 1, 2, 0, 1, 2);
    }

    [TestMethod]
    public void Compress_NaNCondition()
    {
        var cond = np.array(new double[] { double.NaN, 0.0, double.NaN });
        var src = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.compress(cond, src, axis: 0);
        r.Shape.Should().Be(new Shape(2, 3));
        np.ravel(r).ToArray<int>().Should().Equal(0, 1, 2, 6, 7, 8);
    }

    [TestMethod]
    public void Compress_ComplexCondition()
    {
        var cond = np.array(new Complex[] { new(0, 0), new(1, 0), new(0, 1) });
        var src = np.arange(9, NPTypeCode.Int32).reshape(3, 3);
        var r = np.compress(cond, src, axis: 0);
        r.Shape.Should().Be(new Shape(2, 3));
        np.ravel(r).ToArray<int>().Should().Equal(3, 4, 5, 6, 7, 8);
    }

    [TestMethod]
    public void Compress_NonContigCondition()
    {
        // 1-D cond via slicing → strided cond, but still ndim==1.
        var bigCond = np.zeros(new Shape(20), NPTypeCode.Boolean);
        for (int i = 0; i < 20; i += 4) bigCond.SetByte((byte)1, i); // every 4th true
        var view = bigCond[":10:2"]; // size 5: [T, F, T, F, T]
        var src = np.arange(15, NPTypeCode.Int32).reshape(5, 3);
        var r = np.compress(view, src, axis: 0);
        r.Shape.Should().Be(new Shape(3, 3));
        np.ravel(r).ToArray<int>().Should().Equal(0, 1, 2, 6, 7, 8, 12, 13, 14);
    }

    [TestMethod]
    public void Compress_5DSource()
    {
        // 7-D was too large to construct easily here; use 5-D from probes.
        var src = np.arange(2 * 3 * 2 * 3 * 2, NPTypeCode.Int32).reshape(2, 3, 2, 3, 2);
        var r = np.compress(np.array(new bool[] { true, false }), src, axis: 2);
        r.Shape.Should().Be(new Shape(2, 3, 1, 3, 2));
    }

    [TestMethod]
    public void Compress_EmptyAxisSource_EmptyCond_PreservesShape()
    {
        // src is (3, 0, 4); empty cond is valid since len(cond) == axis dim (0).
        var src = np.zeros(new Shape(3, 0, 4), NPTypeCode.Int32);
        var r = np.compress(np.array(new bool[] { }), src, axis: 1);
        r.Shape.Should().Be(new Shape(3, 0, 4));
    }

    [TestMethod]
    public void Compress_EmptyAxisSource_NonEmptyCond_Raises()
    {
        // src is (3, 0, 4); cond [T] would need axis dim ≥ 1, but it's 0.
        var src = np.zeros(new Shape(3, 0, 4), NPTypeCode.Int32);
        var action = () => np.compress(np.array(new bool[] { true }), src, axis: 1);
        action.Should().Throw<Exception>()
            .Where(e => e is ArgumentException || e is IndexOutOfRangeException || e is ArgumentOutOfRangeException);
    }

    [TestMethod]
    public void Compress_AliasedCondAndSource_Independent()
    {
        // cond computed from src (so cond shares semantic content but separate buffer).
        var src = np.array(new int[] { 0, 1, 0, 2, 0, 3 });
        var cond = src > 0;
        var r = np.extract(cond, src);
        r.ToArray<int>().Should().Equal(1, 2, 3);
    }

    [TestMethod]
    public void Compress_AllDtypes_Smoke()
    {
        // Per-dtype gather along axis=0 from a (3,2) shape.
        var cond = np.array(new bool[] { false, true, true });

        np.compress(cond, np.array(new bool[,] { { false, true }, { true, false }, { true, true } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new byte[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new sbyte[,] { { -1, 2 }, { 3, -4 }, { -5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new short[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new ushort[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new uint[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new long[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new ulong[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new char[,] { { 'a', 'b' }, { 'c', 'd' }, { 'e', 'f' } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new Half[,] { { (Half)1, (Half)2 }, { (Half)3, (Half)4 }, { (Half)5, (Half)6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new decimal[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
        np.compress(cond, np.array(new Complex[,] { { new(1, 2), new(3, 4) }, { new(5, 6), new(7, 8) }, { new(9, 10), new(11, 12) } }), axis: 0)
            .Shape.Should().Be(new Shape(2, 2));
    }
}
