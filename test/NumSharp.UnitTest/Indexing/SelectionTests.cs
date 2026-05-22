using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Tests for the np.* selection family: <c>take</c>, <c>put</c>, <c>place</c>.
/// All three are IL-kernel-backed and dtype-agnostic via byte-level
/// <c>cpblk</c>. Test buckets:
/// <list type="bullet">
///   <item>Per-dtype coverage on all 15 supported types (one round-trip).</item>
///   <item>Axis variations for <c>take</c> (None / 0 / 1 / -1 / -2).</item>
///   <item>Mode variations (raise / wrap / clip) on <c>take</c> and <c>put</c>.</item>
///   <item>Shape preservation, 0-d input, empty inputs.</item>
///   <item>OOB raise diagnostics matching NumPy's "out of bounds" messages.</item>
///   <item>Cycling: put values shorter than indices, place vals shorter than mask trues.</item>
///   <item>Round-trip consistency: take ∘ put should restore original at indexed positions.</item>
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
}
