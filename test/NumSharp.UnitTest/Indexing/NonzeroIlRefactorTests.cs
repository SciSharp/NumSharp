using System;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Indexing;

/// <summary>
/// Targeted tests for the <c>np.nonzero</c> IL refactor: the per-dtype IL kernels
/// (<c>ArgwhereCountKernel</c> + <c>ArgwhereFlatKernel</c>) plus the dtype-agnostic
/// per-dim expand kernel (<c>NonZeroPerDimKernel</c>) supersede the previous
/// <c>typeof(T) == typeof(bool)</c> dispatch branch and the
/// <c>List&lt;long&gt;</c>-based <c>NonZeroSimdHelper&lt;T&gt;</c> /
/// <c>FindNonZeroStridedHelper&lt;T&gt;</c> generic-T fallbacks.
///
/// <para>
/// The existing <c>NonzeroTests</c> / <c>NonzeroInt64Tests</c> /
/// <c>NonzeroEdgeCaseTests</c> already cover the happy path for the primitive
/// dtypes; the cases here focus on the surface area introduced (or de-risked)
/// by the refactor — every dtype's IL kernel, the 0-d branch, the multi-SIMD-chunk
/// SIMD body, the high-ndim carry chain in the expand kernel, the non-contig
/// materialize path, and cross-validation that argwhere and nonzero remain in
/// lock-step (same scan + same flat-index → coord conversion, transposed
/// output layouts).
/// </para>
/// </summary>
[TestClass]
public class NonzeroIlRefactorTests
{
    // ── Helpers ─────────────────────────────────────────────────────────

    private static long[] ColumnAsLongs(NDArray<long> nd)
    {
        var buf = new long[nd.size];
        for (long i = 0; i < nd.size; i++)
            buf[i] = nd.GetAtIndex(i);
        return buf;
    }

    private static long[][] ToColumns(NDArray<long>[] nz)
        => nz.Select(ColumnAsLongs).ToArray();

    // ── All 15 dtypes — exercise every per-dtype IL kernel cache slot ───
    //
    // Same logical input (zero at positions 0, 2; non-zero at 1, 3, 4)
    // mapped through each dtype. The result must be (array([1,3,4]),)
    // regardless of dtype.

    [TestMethod]
    public void Refactor_Dtype_Boolean()
    {
        var a = np.array(new bool[] { false, true, false, true, true });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Byte()
    {
        var a = np.array(new byte[] { 0, 1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_SByte()
    {
        // SByte was previously covered by the same generic-T fallback as int*;
        // the new per-dtype IL kernel uses Ldind_I1 (signed) so negatives are
        // correctly counted as non-zero.
        var a = np.array(new sbyte[] { 0, -1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Int16()
    {
        var a = np.array(new short[] { 0, -1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_UInt16()
    {
        var a = np.array(new ushort[] { 0, 1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Int32()
    {
        var a = np.array(new int[] { 0, -1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_UInt32()
    {
        var a = np.array(new uint[] { 0, 1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Int64()
    {
        var a = np.array(new long[] { 0, -1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_UInt64()
    {
        var a = np.array(new ulong[] { 0, 1, 0, 2, 3 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Char()
    {
        // Char reinterprets as ushort in the IL kernel (ArgwhereSimdElement).
        // '\0' counts as zero, all other code points count as non-zero.
        var a = np.array(new char[] { '\0', 'a', '\0', 'b', 'c' });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Half()
    {
        // Half has no Vector<Half> in the BCL — IL kernel falls back to the
        // scalar Half.op_Inequality path. NaN is non-zero, exact zero is zero.
        var a = np.array(new Half[] { (Half)0, (Half)1, (Half)0, (Half)2, (Half)(-3) });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Single()
    {
        var a = np.array(new float[] { 0f, 1.5f, 0f, -2.5f, 3.0f });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Double()
    {
        var a = np.array(new double[] { 0d, 1.5, 0d, -2.5, 3.0 });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Decimal()
    {
        // Decimal has no SIMD support — IL kernel falls back to the scalar
        // decimal.op_Inequality path. Tests that Ldobj + op_Inequality
        // wiring against default(decimal) is correct.
        var a = np.array(new decimal[] { 0m, 1.5m, 0m, -2.5m, 3.0m });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    [TestMethod]
    public void Refactor_Dtype_Complex()
    {
        // Complex has no SIMD support — IL kernel calls Complex.op_Inequality
        // against default(Complex). Both 0+0i and the Complex.Zero static
        // should count as zero; any non-zero real or imaginary part counts.
        var a = np.array(new Complex[]
        {
            Complex.Zero,
            new Complex(1, 0),
            Complex.Zero,
            new Complex(0, 1),
            new Complex(2, -3)
        });
        ColumnAsLongs(np.nonzero(a)[0]).Should().Equal(1L, 3L, 4L);
    }

    // ── 0-D special case — atleast_1d promotion + recursion ─────────────

    [TestMethod]
    public void Refactor_0d_TruthyInt_ReturnsSingleZero()
    {
        // NumPy 2.4 raises ValueError on bare 0-d nonzero, but its error
        // message recommends `np.atleast_1d(scalar).nonzero()` — which
        // returns `(array([0]),)` for any truthy 0-d input. We preserve
        // that semantic (the historical NumSharp behaviour).
        var s = NDArray.Scalar(5);
        var r = np.nonzero(s);
        r.Length.Should().Be(1);
        ColumnAsLongs(r[0]).Should().Equal(0L);
    }

    [TestMethod]
    public void Refactor_0d_FalsyInt_ReturnsEmpty()
    {
        var s = NDArray.Scalar(0);
        var r = np.nonzero(s);
        r.Length.Should().Be(1);
        r[0].size.Should().Be(0);
    }

    [TestMethod]
    public void Refactor_0d_TruthyBool()
    {
        var s = NDArray.Scalar(true);
        ColumnAsLongs(np.nonzero(s)[0]).Should().Equal(0L);
    }

    [TestMethod]
    public void Refactor_0d_FalsyBool()
    {
        var s = NDArray.Scalar(false);
        np.nonzero(s)[0].size.Should().Be(0);
    }

    [TestMethod]
    public void Refactor_0d_TruthyDecimal_UsesScalarPath()
    {
        // Decimal 0-d exercises the scalar op_Inequality branch.
        var s = NDArray.Scalar(1.5m);
        ColumnAsLongs(np.nonzero(s)[0]).Should().Equal(0L);
    }

    [TestMethod]
    public void Refactor_0d_NaN_IsTruthy()
    {
        // NaN ≠ 0.0 under IEEE 754 — must count as non-zero.
        var s = NDArray.Scalar(double.NaN);
        ColumnAsLongs(np.nonzero(s)[0]).Should().Equal(0L);
    }

    // ── Multi-SIMD-chunk arrays — exercise the SIMD body, not just the tail ─

    [TestMethod]
    public void Refactor_Int32_Large_AlternatingPattern_MatchesExpected()
    {
        // 1024 elements: even indices zero, odd indices non-zero. This forces
        // 32+ SIMD chunks through the count + scan loops on V256/V512.
        int n = 1024;
        var data = new int[n];
        for (int i = 0; i < n; i++) data[i] = (i & 1) == 0 ? 0 : i;

        var r = np.nonzero(np.array(data));
        r.Length.Should().Be(1);
        r[0].size.Should().Be(n / 2);
        for (long i = 0; i < r[0].size; i++)
            r[0].GetAtIndex(i).Should().Be(2 * i + 1);
    }

    [TestMethod]
    public void Refactor_Byte_Large_AllNonZero_DenseSimdPath()
    {
        // Dense path: every element non-zero. Stresses the bit-scan inner
        // loop (32 indices materialized per V256 chunk on byte).
        int n = 256;
        var data = new byte[n];
        for (int i = 0; i < n; i++) data[i] = (byte)((i % 255) + 1);

        var r = np.nonzero(np.array(data));
        r[0].size.Should().Be(n);
        for (long i = 0; i < n; i++)
            r[0].GetAtIndex(i).Should().Be(i);
    }

    [TestMethod]
    public void Refactor_Bool_Large_AllFalse_FastPath()
    {
        // All-zero mask. The count kernel must return 0 → early return
        // with `ndim` empty result arrays. No flat scan, no expand.
        var a = np.zeros<bool>(new int[] { 4096 });
        var r = np.nonzero(a);
        r.Length.Should().Be(1);
        r[0].size.Should().Be(0);
    }

    [TestMethod]
    public void Refactor_Bool_Large_SparseEvery20_MatchesPattern()
    {
        // Single non-zero per 20-element block — verifies the bit-scan
        // inner loop correctly extracts isolated set bits from chunks.
        int n = 400;
        var data = new bool[n];
        for (int i = 0; i < n; i += 20) data[i] = true;

        var r = np.nonzero(np.array(data));
        var got = ColumnAsLongs(r[0]);
        var expected = Enumerable.Range(0, n / 20).Select(k => (long)(k * 20)).ToArray();
        got.Should().Equal(expected);
    }

    // ── ndim ≥ 4 — exercise the carry chain in NonZeroPerDimKernel ──────

    [TestMethod]
    public void Refactor_NDim_4_AllNonZero_PerDimColumnsMatchArange()
    {
        // (2, 2, 2, 3) all-non-zero (filled with 1..24). The result columns
        // must enumerate the C-order coords (i, j, k, l) for i in 0..1,
        // j in 0..1, k in 0..1, l in 0..2.
        var data = Enumerable.Range(1, 24).ToArray();
        var a = np.array(data).reshape(2, 2, 2, 3);
        var r = np.nonzero(a);
        r.Length.Should().Be(4);

        // Expected per-dim columns (C-order traversal).
        long idx = 0;
        for (long i = 0; i < 2; i++)
            for (long j = 0; j < 2; j++)
                for (long k = 0; k < 2; k++)
                    for (long l = 0; l < 3; l++, idx++)
                    {
                        r[0].GetAtIndex(idx).Should().Be(i);
                        r[1].GetAtIndex(idx).Should().Be(j);
                        r[2].GetAtIndex(idx).Should().Be(k);
                        r[3].GetAtIndex(idx).Should().Be(l);
                    }
    }

    [TestMethod]
    public void Refactor_NDim_3_SparseCornersOnly_CarryChainExercise()
    {
        // (3,3,3) with non-zeros only at (0,0,0) and (2,2,2).
        // Forces a single large delta in the flat-index buffer that must
        // propagate through the full carry chain inside the IL kernel.
        var a = np.zeros(new Shape(3, 3, 3), NPTypeCode.Int32);
        a.SetInt32(7, 0, 0, 0);
        a.SetInt32(9, 2, 2, 2);

        var r = np.nonzero(a);
        r.Length.Should().Be(3);
        r[0].size.Should().Be(2);

        ColumnAsLongs(r[0]).Should().Equal(0L, 2L);
        ColumnAsLongs(r[1]).Should().Equal(0L, 2L);
        ColumnAsLongs(r[2]).Should().Equal(0L, 2L);
    }

    [TestMethod]
    public void Refactor_NDim_3_NonRectangularDims_RowDimDifferent()
    {
        // (2, 5, 3) — inner-most dim 3, outer 5, outer-most 2. Verifies that
        // dimStrides[d+1] * dims[d+1] is used correctly when dims differ.
        // We set just one element at coord (1, 3, 2).
        var a = np.zeros(new Shape(2, 5, 3), NPTypeCode.Int32);
        a.SetInt32(42, 1, 3, 2);

        var r = np.nonzero(a);
        r.Length.Should().Be(3);
        r[0].size.Should().Be(1);
        r[0].GetAtIndex(0).Should().Be(1L);
        r[1].GetAtIndex(0).Should().Be(3L);
        r[2].GetAtIndex(0).Should().Be(2L);
    }

    // ── Non-contig materialize path ─────────────────────────────────────

    [TestMethod]
    public void Refactor_NonContig_Transposed_MatchesContigOnSameData()
    {
        // [[0,1,0],[2,0,3]].T = [[0,2],[1,0],[0,3]] (shape (3,2))
        // The transposed array routes through np.ascontiguousarray → the
        // same IL kernels operate on a freshly-materialised C-contig copy.
        var src = np.array(new int[,] { { 0, 1, 0 }, { 2, 0, 3 } });
        var t = src.T;
        var r = np.nonzero(t);

        r.Length.Should().Be(2);
        ColumnAsLongs(r[0]).Should().Equal(0L, 1L, 2L);
        ColumnAsLongs(r[1]).Should().Equal(1L, 0L, 1L);
    }

    [TestMethod]
    public void Refactor_NonContig_NegStrideSlice_MatchesReversedContig()
    {
        // Reversed view via [::-1] has stride[-1] = -1 → non-contig.
        // Materialization yields the reversed sequence; nonzero on
        // [3, 0, 2, 0, 1] should give [0, 2, 4] (positions of 3, 2, 1).
        var src = np.array(new int[] { 1, 0, 2, 0, 3 });
        var rev = src["::-1"];
        var r = np.nonzero(rev);
        ColumnAsLongs(r[0]).Should().Equal(0L, 2L, 4L);
    }

    [TestMethod]
    public void Refactor_NonContig_2DSlice_RowsAndColumns()
    {
        // (3, 4) → slice [:2, 1:] yields a (2, 3) non-contig view.
        // The contig materialisation should give:
        //   [[1, 2, 0],
        //    [0, 6, 7]]
        // Non-zero coords: (0,0), (0,1), (1,1), (1,2).
        var src = np.array(new int[,] { { 0, 1, 2, 0 }, { 5, 0, 6, 7 }, { 8, 9, 10, 11 } });
        var v = src[":2, 1:"];
        var r = np.nonzero(v);
        r.Length.Should().Be(2);
        ColumnAsLongs(r[0]).Should().Equal(0L, 0L, 1L, 1L);
        ColumnAsLongs(r[1]).Should().Equal(0L, 1L, 1L, 2L);
    }

    // ── Empty edge cases ────────────────────────────────────────────────

    [TestMethod]
    public void Refactor_EmptyShape_0_3_ReturnsTwoEmptyArrays()
    {
        var a = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var r = np.nonzero(a);
        r.Length.Should().Be(2);
        r[0].size.Should().Be(0);
        r[1].size.Should().Be(0);
        r[0].typecode.Should().Be(NPTypeCode.Int64);
        r[1].typecode.Should().Be(NPTypeCode.Int64);
    }

    [TestMethod]
    public void Refactor_EmptyShape_2_0_4_ReturnsThreeEmptyArrays()
    {
        // size==0 returns one empty Int64 array per dim, preserving ndim.
        var a = np.zeros(new Shape(2, 0, 4), NPTypeCode.Int32);
        var r = np.nonzero(a);
        r.Length.Should().Be(3);
        for (int d = 0; d < 3; d++)
        {
            r[d].size.Should().Be(0);
            r[d].typecode.Should().Be(NPTypeCode.Int64);
        }
    }

    // ── Cross-validation with np.argwhere ───────────────────────────────
    //
    // argwhere(a) and nonzero(a) share the same Count/Flat IL kernels — only
    // the coord expand step differs (argwhere writes (count, ndim) row-major,
    // nonzero writes ndim per-dim columns). The two must therefore stay in
    // lock-step element-wise: argwhere(a)[i, d] == nonzero(a)[d][i].

    [TestMethod]
    public void Refactor_CrossValidate_Argwhere_2D()
    {
        var a = np.array(new int[,] { { 0, 1, 0, 2 }, { 3, 0, 4, 0 }, { 0, 5, 0, 6 } });
        var nz = np.nonzero(a);
        var aw = np.argwhere(a);

        long count = aw.shape[0];
        aw.shape[1].Should().Be(2);
        for (long i = 0; i < count; i++)
        {
            aw.GetInt64(i, 0).Should().Be(nz[0].GetAtIndex(i));
            aw.GetInt64(i, 1).Should().Be(nz[1].GetAtIndex(i));
        }
    }

    [TestMethod]
    public void Refactor_CrossValidate_Argwhere_3D_Dense()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var nz = np.nonzero(a);
        var aw = np.argwhere(a);

        long count = aw.shape[0];
        aw.shape[1].Should().Be(3);
        for (long i = 0; i < count; i++)
        {
            aw.GetInt64(i, 0).Should().Be(nz[0].GetAtIndex(i));
            aw.GetInt64(i, 1).Should().Be(nz[1].GetAtIndex(i));
            aw.GetInt64(i, 2).Should().Be(nz[2].GetAtIndex(i));
        }
    }

    [TestMethod]
    public void Refactor_CrossValidate_Argwhere_NDim_4_Sparse()
    {
        // 4-D sparse — most-stressful coord-expand path.
        var a = np.zeros(new Shape(2, 3, 4, 5), NPTypeCode.Int32);
        a.SetInt32(1, 0, 0, 0, 0);
        a.SetInt32(2, 1, 2, 3, 4);
        a.SetInt32(3, 0, 1, 2, 3);

        var nz = np.nonzero(a);
        var aw = np.argwhere(a);

        long count = aw.shape[0];
        count.Should().Be(3);
        aw.shape[1].Should().Be(4);

        for (long i = 0; i < count; i++)
            for (int d = 0; d < 4; d++)
                aw.GetInt64(i, d).Should().Be(nz[d].GetAtIndex(i));
    }

    // ── Result dtype invariant ──────────────────────────────────────────

    [TestMethod]
    public void Refactor_Result_IsAlwaysInt64()
    {
        // np.nonzero returns int64 indices regardless of input dtype.
        // (Matches NumPy's intp on 64-bit platforms.)
        Type[] sampleDtypes = {
            typeof(bool), typeof(byte), typeof(sbyte),
            typeof(short), typeof(ushort), typeof(int), typeof(uint),
            typeof(long), typeof(ulong), typeof(char),
            typeof(Half), typeof(float), typeof(double),
            typeof(decimal), typeof(Complex)
        };

        foreach (var t in sampleDtypes)
        {
            var npt = (NPTypeCode)Enum.Parse(typeof(NPTypeCode), t.Name);
            var a = np.ones(new Shape(3), npt);
            var r = np.nonzero(a);
            r.Length.Should().Be(1);
            r[0].typecode.Should().Be(NPTypeCode.Int64);
        }
    }

    // ── Indexing round-trip ─────────────────────────────────────────────

    [TestMethod]
    public void Refactor_IndexingRoundTrip_PreservesValues()
    {
        // a[nonzero(a)] should yield the non-zero values in C-order.
        var a = np.array(new int[,] { { 3, 0, 0 }, { 0, 4, 0 }, { 5, 6, 0 } });
        var idx = np.nonzero(a);
        var vals = a[idx];
        vals.size.Should().Be(4);
        vals.GetInt32(0).Should().Be(3);
        vals.GetInt32(1).Should().Be(4);
        vals.GetInt32(2).Should().Be(5);
        vals.GetInt32(3).Should().Be(6);
    }
}
