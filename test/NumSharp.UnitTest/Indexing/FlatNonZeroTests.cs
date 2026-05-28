using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Targeted tests for <c>np.flatnonzero</c> — equivalent to
/// <c>np.nonzero(np.ravel(a))[0]</c> but implemented directly against the
/// <c>ArgwhereCountKernel</c> + <c>ArgwhereFlatKernel</c> IL pair so we never
/// allocate per-axis NDArray columns we'd just discard.
///
/// <para>
/// Coverage mirrors <see cref="NonzeroIlRefactorTests"/>:
/// <list type="bullet">
///   <item>Per-dtype IL kernel cache slots (all 15 supported dtypes).</item>
///   <item>0-d truthy/falsy via <c>atleast_1d</c> recursion.</item>
///   <item>Multi-SIMD-chunk inputs to exercise the SIMD body, not just the
///         scalar tail.</item>
///   <item>Multi-dim inputs to verify C-order flattening (the flat index from
///         the IL scan IS the index into the raveled view).</item>
///   <item>Non-contig materialise path (transposed view, neg-stride slice,
///         2-D slice) to confirm the explicit <c>Dispose</c> on the
///         materialised intermediate keeps the buffer alive through the IL
///         scan even in Release-mode GC pressure.</item>
///   <item>Empty shapes and the C-order ravel of degenerate shapes.</item>
///   <item>Cross-validation against <c>np.nonzero(np.ravel(a))[0]</c> — they
///         must agree element-wise on any input.</item>
/// </list>
/// </para>
/// </summary>
[TestClass]
public class FlatNonZeroTests
{
    private static long[] ToLongs(NDArray<long> nd)
    {
        var buf = new long[nd.size];
        for (long i = 0; i < nd.size; i++)
            buf[i] = nd.GetAtIndex(i);
        return buf;
    }

    // ── All 15 dtypes — exercise every per-dtype IL kernel cache slot ───
    //
    // Same logical input (zero at positions 0, 2; non-zero at 1, 3, 4)
    // mapped through each dtype. The result must be [1, 3, 4] regardless
    // of dtype. Mirrors the NonzeroIlRefactorTests dtype matrix but for
    // the 1-D flatnonzero entry point.

    [TestMethod]
    public void FlatNonZero_Dtype_Boolean()
    {
        var a = np.array(new bool[] { false, true, false, true, true });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Byte()
    {
        var a = np.array(new byte[] { 0, 1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_SByte()
    {
        var a = np.array(new sbyte[] { 0, -1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Int16()
    {
        var a = np.array(new short[] { 0, -1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_UInt16()
    {
        var a = np.array(new ushort[] { 0, 1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Int32()
    {
        var a = np.array(new int[] { 0, -1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_UInt32()
    {
        var a = np.array(new uint[] { 0, 1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Int64()
    {
        var a = np.array(new long[] { 0, -1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_UInt64()
    {
        var a = np.array(new ulong[] { 0, 1, 0, 2, 3 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Char()
    {
        var a = np.array(new char[] { '\0', 'a', '\0', 'b', 'c' });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Half()
    {
        // Scalar IL path via Half.op_Inequality.
        var a = np.array(new Half[] { (Half)0, (Half)1, (Half)0, (Half)2, (Half)(-3) });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Single()
    {
        var a = np.array(new float[] { 0f, 1.5f, 0f, -2.5f, 3.0f });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Double()
    {
        var a = np.array(new double[] { 0d, 1.5, 0d, -2.5, 3.0 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Decimal()
    {
        // Scalar IL path via decimal.op_Inequality.
        var a = np.array(new decimal[] { 0m, 1.5m, 0m, -2.5m, 3.0m });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_Dtype_Complex()
    {
        // Scalar IL path via Complex.op_Inequality.
        var a = np.array(new Complex[]
        {
            Complex.Zero,
            new Complex(1, 0),
            Complex.Zero,
            new Complex(0, 1),
            new Complex(2, -3)
        });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 4L);
    }

    // ── Floating-point semantics ────────────────────────────────────────

    [TestMethod]
    public void FlatNonZero_Float_NaN_CountsAsNonZero()
    {
        // IEEE 754: NaN != 0.0 → NaN must be reported as a non-zero element.
        var a = np.array(new double[] { 0.0, double.NaN, 0.0, 1.5 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L);
    }

    [TestMethod]
    public void FlatNonZero_Float_NegativeZero_CountsAsZero()
    {
        // -0.0 == 0.0 numerically → must NOT appear in the result.
        var a = np.array(new double[] { -0.0, 0.0, 1.0 });
        ToLongs(np.flatnonzero(a)).Should().Equal(2L);
    }

    [TestMethod]
    public void FlatNonZero_Float_Infinity_CountsAsNonZero()
    {
        // ±Infinity is non-zero.
        var a = np.array(new double[] { 0.0, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 2L);
    }

    // ── 0-D special case — atleast_1d promotion + recursion ─────────────

    [TestMethod]
    public void FlatNonZero_0d_TruthyInt_ReturnsSingleZero()
    {
        var s = NDArray.Scalar(5);
        ToLongs(np.flatnonzero(s)).Should().Equal(0L);
    }

    [TestMethod]
    public void FlatNonZero_0d_FalsyInt_ReturnsEmpty()
    {
        var s = NDArray.Scalar(0);
        np.flatnonzero(s).size.Should().Be(0);
    }

    [TestMethod]
    public void FlatNonZero_0d_TruthyBool()
    {
        var s = NDArray.Scalar(true);
        ToLongs(np.flatnonzero(s)).Should().Equal(0L);
    }

    [TestMethod]
    public void FlatNonZero_0d_FalsyBool()
    {
        var s = NDArray.Scalar(false);
        np.flatnonzero(s).size.Should().Be(0);
    }

    [TestMethod]
    public void FlatNonZero_0d_TruthyDecimal_UsesScalarPath()
    {
        var s = NDArray.Scalar(1.5m);
        ToLongs(np.flatnonzero(s)).Should().Equal(0L);
    }

    [TestMethod]
    public void FlatNonZero_0d_NaN_IsTruthy()
    {
        var s = NDArray.Scalar(double.NaN);
        ToLongs(np.flatnonzero(s)).Should().Equal(0L);
    }

    // ── Multi-SIMD-chunk arrays — exercise the SIMD body, not just the tail ─

    [TestMethod]
    public void FlatNonZero_Int32_Large_AlternatingPattern_MatchesExpected()
    {
        int n = 1024;
        var data = new int[n];
        for (int i = 0; i < n; i++) data[i] = (i & 1) == 0 ? 0 : i;

        var r = np.flatnonzero(np.array(data));
        r.size.Should().Be(n / 2);
        for (long i = 0; i < r.size; i++)
            r.GetAtIndex(i).Should().Be(2 * i + 1);
    }

    [TestMethod]
    public void FlatNonZero_Byte_Large_AllNonZero_DenseSimdPath()
    {
        int n = 256;
        var data = new byte[n];
        for (int i = 0; i < n; i++) data[i] = (byte)((i % 255) + 1);

        var r = np.flatnonzero(np.array(data));
        r.size.Should().Be(n);
        for (long i = 0; i < n; i++)
            r.GetAtIndex(i).Should().Be(i);
    }

    [TestMethod]
    public void FlatNonZero_Bool_Large_AllFalse_CountZeroFastPath()
    {
        // All-zero → count kernel returns 0 → early return with empty array.
        var a = np.zeros<bool>(new int[] { 4096 });
        var r = np.flatnonzero(a);
        r.size.Should().Be(0);
        r.typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void FlatNonZero_Bool_Large_SparseEvery20_MatchesPattern()
    {
        int n = 400;
        var data = new bool[n];
        for (int i = 0; i < n; i += 20) data[i] = true;

        var got = ToLongs(np.flatnonzero(np.array(data)));
        var expected = Enumerable.Range(0, n / 20).Select(k => (long)(k * 20)).ToArray();
        got.Should().Equal(expected);
    }

    // ── Multi-dim inputs — verify C-order flattening ────────────────────

    [TestMethod]
    public void FlatNonZero_2D_FlattenedIndicesMatchRavelOrder()
    {
        // [[0,1,0],[2,0,3]] → ravel → [0,1,0,2,0,3] → nonzero positions [1,3,5]
        var a = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } });
        ToLongs(np.flatnonzero(a)).Should().Equal(1L, 3L, 5L);
    }

    [TestMethod]
    public void FlatNonZero_3D_FlattenedIndicesMatchRavelOrder()
    {
        // (2, 2, 3) all-non-zero filled with 1..12 → ravel order [1..12] → all positions.
        var a = np.array(Enumerable.Range(1, 12).ToArray()).reshape(2, 2, 3);
        var got = ToLongs(np.flatnonzero(a));
        got.Should().Equal(Enumerable.Range(0, 12).Select(i => (long)i).ToArray());
    }

    [TestMethod]
    public void FlatNonZero_NDim_4_SparseCornersOnly()
    {
        // (2,3,2,2) with non-zeros only at (0,0,0,0) and (1,2,1,1).
        // C-order flat indices: 0*12 + 0*4 + 0*2 + 0 = 0
        //                        1*12 + 2*4 + 1*2 + 1 = 23
        var a = np.zeros(new Shape(2, 3, 2, 2), NPTypeCode.Int32);
        a.SetInt32(7, 0, 0, 0, 0);
        a.SetInt32(9, 1, 2, 1, 1);

        var r = np.flatnonzero(a);
        ToLongs(r).Should().Equal(0L, 23L);
    }

    // ── Non-contig materialise path ─────────────────────────────────────

    [TestMethod]
    public void FlatNonZero_NonContig_Transposed_MatchesRavelOfMaterialized()
    {
        // src = [[0,1,0],[2,0,3]]; src.T = [[0,2],[1,0],[0,3]] (shape (3,2))
        // ravel(src.T) in C-order = [0, 2, 1, 0, 0, 3] → nonzero positions [1, 2, 5]
        var src = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } });
        var t = src.T;
        ToLongs(np.flatnonzero(t)).Should().Equal(1L, 2L, 5L);
    }

    [TestMethod]
    public void FlatNonZero_NonContig_NegStrideSlice_MatchesReversedContig()
    {
        // src = [1, 0, 2, 0, 3], src[::-1] = [3, 0, 2, 0, 1]
        // → nonzero flat indices [0, 2, 4]
        var src = np.array(new int[] { 1, 0, 2, 0, 3 });
        var rev = src["::-1"];
        ToLongs(np.flatnonzero(rev)).Should().Equal(0L, 2L, 4L);
    }

    [TestMethod]
    public void FlatNonZero_NonContig_2DSlice_FlattenedIndices()
    {
        // src (3,4) → slice [:2, 1:] (2,3) non-contig:
        //   [[1, 2, 0],
        //    [0, 6, 7]]
        // Ravel → [1, 2, 0, 0, 6, 7]. Non-zero flat indices: [0, 1, 4, 5].
        var src = np.array(new int[,] { { 0, 1, 2, 0 }, { 5, 0, 6, 7 }, { 8, 9, 10, 11 } });
        var v = src[":2, 1:"];
        ToLongs(np.flatnonzero(v)).Should().Equal(0L, 1L, 4L, 5L);
    }

    // ── Empty edge cases ────────────────────────────────────────────────

    [TestMethod]
    public void FlatNonZero_EmptyShape_0_3_ReturnsEmpty1D()
    {
        // size==0 → empty 1-D int64 array regardless of source ndim.
        var a = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var r = np.flatnonzero(a);
        r.size.Should().Be(0);
        r.ndim.Should().Be(1);
        r.typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void FlatNonZero_EmptyShape_2_0_4_ReturnsEmpty1D()
    {
        var a = np.zeros(new Shape(2, 0, 4), NPTypeCode.Int32);
        var r = np.flatnonzero(a);
        r.size.Should().Be(0);
        r.ndim.Should().Be(1);
        r.typecode.Should().Be(NPTypeCode.Int64);
    }

    // ── Result invariants ───────────────────────────────────────────────

    [TestMethod]
    public void FlatNonZero_Result_IsAlways1DInt64()
    {
        var inputs = new NDArray[]
        {
            np.array(new int[] { 1, 0, 2 }),
            np.array(new bool[] { true, false }),
            np.array(new double[] { 1.5, 0.0, double.NaN }),
            np.array(new decimal[] { 1m, 0m, 2m }),
            np.array(new Complex[] { Complex.Zero, new Complex(1, 0) }),
        };
        foreach (var nd in inputs)
        {
            var r = np.flatnonzero(nd);
            r.ndim.Should().Be(1, $"flatnonzero result must be 1-D for dtype {nd.dtype.Name}");
            r.typecode.Should().Be(NPTypeCode.Int64, $"flatnonzero result must be Int64 for dtype {nd.dtype.Name}");
        }
    }

    // ── Cross-validation against np.nonzero(np.ravel(a))[0] ─────────────
    //
    // The whole point of `np.flatnonzero` is to be the 1-D entry point that
    // matches the composition `np.nonzero(np.ravel(a))[0]`. Anything that
    // diverges between the two paths is a bug.

    [TestMethod]
    public void FlatNonZero_CrossValidate_Nonzero1D_RavelEquivalence()
    {
        var a = np.array(new int[] { 0, 5, -1, 0, 3, 0, 7 });
        var fnz = ToLongs(np.flatnonzero(a));
        var nz0 = ToLongs(np.nonzero(a)[0]);
        fnz.Should().Equal(nz0);
    }

    [TestMethod]
    public void FlatNonZero_CrossValidate_Nonzero2D_RavelEquivalence()
    {
        // np.nonzero returns per-axis arrays; the equivalent flat index is
        // i*cols + j. flatnonzero gives that flat index directly.
        var a = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } });
        var fnz = ToLongs(np.flatnonzero(a));

        var nz = np.nonzero(a);
        long cols = a.shape[1];
        var expected = new long[nz[0].size];
        for (long i = 0; i < nz[0].size; i++)
            expected[i] = nz[0].GetAtIndex(i) * cols + nz[1].GetAtIndex(i);

        fnz.Should().Equal(expected);
    }

    [TestMethod]
    public void FlatNonZero_CrossValidate_Nonzero3D_RavelEquivalence()
    {
        var data = new int[24];
        for (int i = 0; i < data.Length; i++) data[i] = (i % 3 == 0) ? 0 : i;
        var a = np.array(data).reshape(2, 3, 4);

        var fnz = ToLongs(np.flatnonzero(a));

        var nz = np.nonzero(a);
        long d1 = a.shape[1], d2 = a.shape[2];
        var expected = new long[nz[0].size];
        for (long i = 0; i < nz[0].size; i++)
        {
            expected[i] = nz[0].GetAtIndex(i) * d1 * d2
                        + nz[1].GetAtIndex(i) * d2
                        + nz[2].GetAtIndex(i);
        }

        fnz.Should().Equal(expected);
    }

    [TestMethod]
    public void FlatNonZero_IndexingRoundTrip_PreservesValues()
    {
        // NumPy doc example: x.ravel()[np.flatnonzero(x)] yields the non-zero
        // values in C-order. Validate that round-trip here.
        var x = np.array(new int[] { -2, -1, 0, 1, 0, 2 });
        var idx = np.flatnonzero(x);
        var raveled = np.ravel(x);
        var picked = new int[idx.size];
        for (long i = 0; i < idx.size; i++)
            picked[i] = raveled.GetInt32(idx.GetAtIndex(i));
        picked.Should().Equal(-2, -1, 1, 2);
    }
}
