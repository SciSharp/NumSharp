using System;
using System.Numerics;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
///     Tests for np.argwhere (NumPy 2.4.2 parity).
///     NumPy semantics: returns the (N, ndim) int64 array of coordinates of non-zero
///     elements, traversed in C-order. Equivalent to np.transpose(np.nonzero(a))
///     except for the 0-d special case (truthy → (1,0), falsy → (0,0)).
/// </summary>
[TestClass]
public class ArgwhereTests
{
    private static void AssertRow(NDArray result, long row, params long[] expectedCoords)
    {
        for (int d = 0; d < expectedCoords.Length; d++)
            result.GetInt64(row, d).Should().Be(expectedCoords[d], $"row {row} col {d}");
    }

    #region 1D — basic indexing

    [TestMethod]
    public void Argwhere_1D_Predicate()
    {
        // NumPy: np.argwhere([3,1,4,1,5] > 3) == [[2],[4]]
        var a = np.array(new[] { 3, 1, 4, 1, 5 });
        var r = np.argwhere(a > 3);

        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        r.dtype.Should().Be(typeof(long));
        AssertRow(r, 0, 2);
        AssertRow(r, 1, 4);
    }

    [TestMethod]
    public void Argwhere_1D_AllNonZero()
    {
        // np.argwhere([1,2,3]) == [[0],[1],[2]]
        var r = np.argwhere(np.array(new[] { 1, 2, 3 }));
        r.shape.Should().BeEquivalentTo(new long[] { 3, 1 });
        AssertRow(r, 0, 0);
        AssertRow(r, 1, 1);
        AssertRow(r, 2, 2);
    }

    [TestMethod]
    public void Argwhere_1D_AllZero_ReturnsZeroRowsOneCol()
    {
        // np.argwhere(np.zeros(5)).shape == (0, 1)
        var r = np.argwhere(np.zeros(5));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 1 });
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Argwhere_Bool_1D()
    {
        var r = np.argwhere(np.array(new[] { true, false, true }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 0);
        AssertRow(r, 1, 2);
    }

    #endregion

    #region 2D / 3D — multi-dim

    [TestMethod]
    public void Argwhere_2D_Predicate()
    {
        // np.argwhere(np.arange(6).reshape(2,3) > 1) == [[0,2],[1,0],[1,1],[1,2]]
        var x = np.arange(6).reshape(2, 3);
        var r = np.argwhere(x > 1);

        r.shape.Should().BeEquivalentTo(new long[] { 4, 2 });
        AssertRow(r, 0, 0, 2);
        AssertRow(r, 1, 1, 0);
        AssertRow(r, 2, 1, 1);
        AssertRow(r, 3, 1, 2);
    }

    [TestMethod]
    public void Argwhere_3D_Mod()
    {
        // np.argwhere(np.arange(24).reshape(2,3,4) % 5 == 0)
        //   == [[0,0,0],[0,1,1],[0,2,2],[1,0,3],[1,2,0]]
        var y = np.arange(24).reshape(2, 3, 4);
        var r = np.argwhere(y % 5 == 0);

        r.shape.Should().BeEquivalentTo(new long[] { 5, 3 });
        AssertRow(r, 0, 0, 0, 0);
        AssertRow(r, 1, 0, 1, 1);
        AssertRow(r, 2, 0, 2, 2);
        AssertRow(r, 3, 1, 0, 3);
        AssertRow(r, 4, 1, 2, 0);
    }

    [TestMethod]
    public void Argwhere_4D()
    {
        // np.argwhere(np.arange(16).reshape(2,2,2,2) > 12) == [[1,1,0,1],[1,1,1,0],[1,1,1,1]]
        var y = np.arange(16).reshape(2, 2, 2, 2);
        var r = np.argwhere(y > 12);

        r.shape.Should().BeEquivalentTo(new long[] { 3, 4 });
        AssertRow(r, 0, 1, 1, 0, 1);
        AssertRow(r, 1, 1, 1, 1, 0);
        AssertRow(r, 2, 1, 1, 1, 1);
    }

    [TestMethod]
    public void Argwhere_2D_BoolMask()
    {
        // np.argwhere([[T,F,T],[F,T,F]]) == [[0,0],[0,2],[1,1]]
        var m = np.array(new[,] { { true, false, true }, { false, true, false } });
        var r = np.argwhere(m);

        r.shape.Should().BeEquivalentTo(new long[] { 3, 2 });
        AssertRow(r, 0, 0, 0);
        AssertRow(r, 1, 0, 2);
        AssertRow(r, 2, 1, 1);
    }

    #endregion

    #region 0-d (scalar) — special case

    [TestMethod]
    public void Argwhere_0D_Truthy_ReturnsOneZeroShape()
    {
        // np.argwhere(np.array(5)).shape == (1, 0)
        var r = np.argwhere(np.array(5));
        r.shape.Should().BeEquivalentTo(new long[] { 1, 0 });
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Argwhere_0D_Falsy_ReturnsZeroZeroShape()
    {
        // np.argwhere(np.array(0)).shape == (0, 0)
        var r = np.argwhere(np.array(0));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 0 });
        r.size.Should().Be(0);
    }

    [TestMethod]
    public void Argwhere_0D_Bool_True()
    {
        var r = np.argwhere(np.array(true));
        r.shape.Should().BeEquivalentTo(new long[] { 1, 0 });
    }

    [TestMethod]
    public void Argwhere_0D_Bool_False()
    {
        var r = np.argwhere(np.array(false));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 0 });
    }

    [TestMethod]
    public void Argwhere_0D_Float_Zero()
    {
        var r = np.argwhere(np.array(0.0));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 0 });
    }

    [TestMethod]
    public void Argwhere_0D_Float_NonZero()
    {
        var r = np.argwhere(np.array(3.14));
        r.shape.Should().BeEquivalentTo(new long[] { 1, 0 });
    }

    #endregion

    #region Empty input

    [TestMethod]
    public void Argwhere_EmptyInput_1D()
    {
        // np.argwhere(np.zeros(0)).shape == (0, 1)
        var r = np.argwhere(np.zeros(0));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 1 });
    }

    [TestMethod]
    public void Argwhere_EmptyInput_2D()
    {
        // np.argwhere(np.zeros((0,3))).shape == (0, 2)
        var r = np.argwhere(np.zeros(new Shape(0L, 3L)));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 2 });
    }

    [TestMethod]
    public void Argwhere_EmptyInput_3D()
    {
        // np.argwhere(np.zeros((2,0,4))).shape == (0, 3)
        var r = np.argwhere(np.zeros(new Shape(2L, 0L, 4L)));
        r.shape.Should().BeEquivalentTo(new long[] { 0, 3 });
    }

    #endregion

    #region Non-contiguous layouts

    [TestMethod]
    public void Argwhere_Sliced_2D()
    {
        // x[::2, ::2] is (2, 3): [[0,2,4],[10,12,14]]; > 5 → [[1,0],[1,1],[1,2]]
        var x = np.arange(20).reshape(4, 5);
        var sliced = x["::2, ::2"];
        sliced.Shape.IsContiguous.Should().BeFalse();

        var r = np.argwhere(sliced > 5);
        r.shape.Should().BeEquivalentTo(new long[] { 3, 2 });
        AssertRow(r, 0, 1, 0);
        AssertRow(r, 1, 1, 1);
        AssertRow(r, 2, 1, 2);
    }

    [TestMethod]
    public void Argwhere_Transposed()
    {
        var x = np.arange(20).reshape(4, 5);
        var t = x.T;
        t.Shape.IsContiguous.Should().BeFalse();

        // t > 10 has 9 non-zero positions (NumPy reference output)
        var r = np.argwhere(t > 10);
        r.shape.Should().BeEquivalentTo(new long[] { 9, 2 });
        AssertRow(r, 0, 0, 3);
        AssertRow(r, 1, 1, 2);
        AssertRow(r, 2, 1, 3);
        AssertRow(r, 3, 2, 2);
        AssertRow(r, 4, 2, 3);
        AssertRow(r, 5, 3, 2);
        AssertRow(r, 6, 3, 3);
        AssertRow(r, 7, 4, 2);
        AssertRow(r, 8, 4, 3);
    }

    [TestMethod]
    public void Argwhere_NegativeStride()
    {
        var x = np.arange(20).reshape(4, 5);
        var neg = x["::-1"];
        neg.Shape.IsContiguous.Should().BeFalse();

        var r = np.argwhere(neg > 10);
        r.shape.Should().BeEquivalentTo(new long[] { 9, 2 });
        AssertRow(r, 0, 0, 0);
        AssertRow(r, 1, 0, 1);
        AssertRow(r, 2, 0, 2);
        AssertRow(r, 3, 0, 3);
        AssertRow(r, 4, 0, 4);
        AssertRow(r, 5, 1, 1);
        AssertRow(r, 6, 1, 2);
        AssertRow(r, 7, 1, 3);
        AssertRow(r, 8, 1, 4);
    }

    [TestMethod]
    public void Argwhere_Broadcasted()
    {
        // np.broadcast_to([1,0,1], (3,3)) — broadcast view, IsBroadcasted=true.
        var b = np.broadcast_to(np.array(new[] { 1, 0, 1 }), new Shape(3, 3));
        b.Shape.IsBroadcasted.Should().BeTrue();

        var r = np.argwhere(b);
        r.shape.Should().BeEquivalentTo(new long[] { 6, 2 });
        AssertRow(r, 0, 0, 0);
        AssertRow(r, 1, 0, 2);
        AssertRow(r, 2, 1, 0);
        AssertRow(r, 3, 1, 2);
        AssertRow(r, 4, 2, 0);
        AssertRow(r, 5, 2, 2);
    }

    #endregion

    #region Dtype coverage

    [TestMethod]
    public void Argwhere_Dtype_SByte()
    {
        var r = np.argwhere(np.array(new sbyte[] { 0, 1, -1, 0 }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 2);
    }

    [TestMethod]
    public void Argwhere_Dtype_Byte()
    {
        var r = np.argwhere(np.array(new byte[] { 0, 200, 0, 1 }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 3);
    }

    [TestMethod]
    public void Argwhere_Dtype_Int16()
    {
        var r = np.argwhere(np.array(new short[] { 0, 7, 0, -3, 0 }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
    }

    [TestMethod]
    public void Argwhere_Dtype_UInt64()
    {
        var r = np.argwhere(np.array(new ulong[] { 0, 100, 0, ulong.MaxValue }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 3);
    }

    [TestMethod]
    public void Argwhere_Dtype_Single()
    {
        var r = np.argwhere(np.array(new[] { 0f, 1.5f, 0f, -0.1f }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
    }

    [TestMethod]
    public void Argwhere_Dtype_Double()
    {
        var r = np.argwhere(np.array(new[] { 1.0, 0.0, 3.5 }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 0);
        AssertRow(r, 1, 2);
    }

    [TestMethod]
    public void Argwhere_Dtype_Decimal()
    {
        var r = np.argwhere(np.array(new[] { 0m, 1m, 0m, 2m, 0m }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 3);
    }

    [TestMethod]
    public void Argwhere_Dtype_Half()
    {
        var r = np.argwhere(np.array(new[] { (Half)0, (Half)1, (Half)0, (Half)2 }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 3);
    }

    [TestMethod]
    public void Argwhere_Dtype_Char()
    {
        // '\0' is the zero value for char.
        var r = np.argwhere(np.array(new[] { '\0', 'a', '\0', 'b' }));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 3);
    }

    [TestMethod]
    public void Argwhere_Dtype_Complex()
    {
        // 0+0j is zero; everything else is non-zero.
        var c = new[]
        {
            new Complex(0, 0),
            new Complex(1, 0),
            new Complex(0, 1),
            new Complex(0, 0)
        };
        var r = np.argwhere(np.array(c));
        r.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
        AssertRow(r, 0, 1);
        AssertRow(r, 1, 2);
    }

    #endregion

    #region Result dtype + size

    [TestMethod]
    public void Argwhere_Result_DtypeIs_Int64()
    {
        var r = np.argwhere(np.array(new[] { 1, 0, 1 }));
        r.dtype.Should().Be(typeof(long));
        r.typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Argwhere_LargeAllOnes_1M_BoolMatchesCount()
    {
        // 1M-element bool all true — exercises the IL bit-scan + popcount kernels.
        var mask = np.ones(1_000_000, np.@bool);
        var r = np.argwhere(mask);
        r.shape.Should().BeEquivalentTo(new long[] { 1_000_000, 1 });
        r.GetInt64(0, 0).Should().Be(0);
        r.GetInt64(999_999, 0).Should().Be(999_999);
    }

    [TestMethod]
    public void Argwhere_LargeAllZeros_1M_BoolReturnsEmpty()
    {
        // All-false prescan short-circuit should take O(size) SIMD, no row alloc.
        var mask = np.zeros(1_000_000, np.@bool);
        var r = np.argwhere(mask);
        r.shape.Should().BeEquivalentTo(new long[] { 0, 1 });
    }

    [TestMethod]
    public void Argwhere_2D_SparsePattern_1M_BoolMatchesNumPy()
    {
        // Sparse: every 100th element non-zero.
        var n = 10_000L;
        var mask = np.zeros(new Shape(100, n), np.@bool);
        // Set diagonal-ish pattern: mask[i, i*100] = True for i in 0..99 (every 100 elems flat)
        for (long i = 0; i < 100; i++)
            mask.SetData(true, (int)i, 0);  // mask[i, 0] = True — 100 non-zeros total

        var r = np.argwhere(mask);
        r.shape.Should().BeEquivalentTo(new long[] { 100, 2 });
        for (long i = 0; i < 100; i++)
        {
            r.GetInt64(i, 0).Should().Be(i);
            r.GetInt64(i, 1).Should().Be(0);
        }
    }

    #endregion
}
