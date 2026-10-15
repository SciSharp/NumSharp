using System;
using System.Linq;

namespace NumSharp.Tests.Indexing;

/// <summary>
///     Tests for <c>np.FancyIndexAlongAxis</c> — NumPy's fancy index along one non-leading axis,
///     <c>a[(slice(None),) * axis + (indices,)]</c>, laid out as NumPy lays it out: a view of a fresh buffer whose first
///     axis is the indexed one (MapIter's buffer, transposed back), written in one pass by
///     <c>DirectILKernelGenerator.GetTakeLeadingKernel</c>. The values are <c>np.take(a, indices, axis)</c>'s, the
///     reference every case compares against byte for byte; the layout is NumPy's, pinned against NumPy 2.4.2 through
///     <c>Generator.permutation</c> in <c>RandomSampling/RandomResultOwnership.Test.cs</c>.
/// </summary>
/// <remarks>
///     The kernel is compiled per copy width — a 1/2/4/8/16-byte slab copies with typed MOVs in a 4x-unrolled loop, any
///     other slab size with <c>cpblk</c> — and walks the source in tiles of rows. The cases reach every width, every unroll
///     remainder, several tiles with a partial last one, empty operands, and every source layout (a non-C source is copied
///     to C order first).
/// </remarks>
[TestClass]
public class FancyIndexAlongAxisTests
{
    /// <summary>The 15 NumSharp dtypes.</summary>
    private static readonly NPTypeCode[] Dtypes =
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
        NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single,
        NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
    };

    /// <summary>
    ///     The strides NumPy's fancy index gives the result: the buffer <c>(m,) + shape without axis</c> is C-ordered, and
    ///     moving its first axis back to <paramref name="axis"/> permutes those strides.
    /// </summary>
    /// <param name="shape">The source dimensions.</param>
    /// <param name="axis">The indexed (non-leading) axis.</param>
    /// <param name="m">The number of indices.</param>
    /// <param name="itemsize">Bytes per element.</param>
    /// <returns>The result strides, in bytes.</returns>
    private static long[] NumPyStrides(long[] shape, int axis, long m, long itemsize)
    {
        var buffer = new long[shape.Length];
        buffer[0] = m;
        for (int d = 0, k = 1; d < shape.Length; d++)
            if (d != axis)
                buffer[k++] = shape[d];

        var bufferStrides = new long[shape.Length];
        long stride = itemsize;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            bufferStrides[d] = stride;
            stride *= buffer[d];
        }

        var strides = new long[shape.Length];
        strides[axis] = bufferStrides[0];
        for (int d = 0, k = 1; d < shape.Length; d++)
            if (d != axis)
                strides[d] = bufferStrides[k++];
        return strides;
    }

    /// <summary>
    ///     Gathers <paramref name="a"/> along <paramref name="axis"/> and checks the result: <c>np.take</c>'s shape, dtype and
    ///     bytes, and NumPy's layout — a writeable view with the permuted buffer strides.
    /// </summary>
    /// <param name="a">The source, any layout.</param>
    /// <param name="idx">The indices, each in range.</param>
    /// <param name="axis">The non-leading axis.</param>
    /// <param name="what">The case, named in failure messages.</param>
    private static void AssertGather(NDArray a, long[] idx, int axis, string what)
    {
        using var indices = np.array(idx);
        using var r = np.FancyIndexAlongAxis(a, indices, axis);
        using var expected = np.take(a, indices, axis);

        r.shape.Should().Equal(expected.shape, what);
        r.typecode.Should().Be(a.typecode, what);
        r.flags.owndata.Should().BeFalse($"{what}: NumPy's fancy index along a non-leading axis is a view");
        r.flags.writeable.Should().BeTrue(what);
        if (r.size == 0)
            return; // NumPy 2.x reports all-zero strides for an empty array: nothing layout-specific to compare.

        r.strides.Should().Equal(NumPyStrides(a.shape, axis, idx.Length, a.dtypesize), what);
        using var dense = np.ascontiguousarray(r);
        dense.Unsafe.ReadOnlyBytes().SequenceEqual(expected.Unsafe.ReadOnlyBytes())
            .Should().BeTrue($"{what}: the values are np.take's");
    }

    /// <summary>
    ///     Every dtype, along the middle axis (a 4-element slab: 4..64 bytes, typed MOVs up to 16 bytes, <c>cpblk</c> past
    ///     them) and the last axis (a one-element slab: the dtype's own width, 1..16 bytes).
    /// </summary>
    [TestMethod]
    public void EveryDtype_MatchesTake_InNumPysLayout()
    {
        foreach (var tc in Dtypes)
        {
            using var src = np.arange(2L * 3 * 4).astype(tc).reshape(2, 3, 4);
            AssertGather(src, new long[] { 2, 0, 1 }, 1, $"{tc} axis 1");
            AssertGather(src, new long[] { 3, 1, 0, 2, 1 }, 2, $"{tc} axis 2");
        }
    }

    /// <summary>
    ///     Outer extents 1..9 leave every remainder of the 4x-unrolled typed loop (0..3 rows after the unrolled part), and
    ///     the same extents through the <c>cpblk</c> loop (a 3-byte slab).
    /// </summary>
    [TestMethod]
    public void EveryUnrollRemainder_TypedAndCpblk()
    {
        for (long outer = 1; outer <= 9; outer++)
        {
            using var typed = np.arange(outer * 5).reshape(outer, 5);
            AssertGather(typed, new long[] { 4, 0, 3 }, 1, $"int64 ({outer}, 5)");

            using var block = np.arange(outer * 5 * 3).astype(np.int8).reshape(outer, 5, 3);
            AssertGather(block, new long[] { 4, 0, 3 }, 1, $"int8 ({outer}, 5, 3)");
        }
    }

    /// <summary>
    ///     The indices need not be a permutation: repeats, fewer or more indices than the axis holds, and a single index.
    /// </summary>
    [TestMethod]
    public void AnyInRangeIndexSequence()
    {
        using var a = np.arange(4L * 5).reshape(4, 5);
        AssertGather(a, new long[] { 2, 2, 0, 4, 1, 3, 3 }, 1, "7 indices with repeats into 5");
        AssertGather(a, new long[] { 3 }, 1, "one index");
        AssertGather(a, new long[] { 4, 0 }, 1, "two indices");
    }

    /// <summary>
    ///     A source of 2500 rows of 2400 bytes spans several 1 MB read tiles (436 rows each) with a partial last tile.
    /// </summary>
    [TestMethod]
    public void ManyTiles_WithAPartialLastTile()
    {
        using var a = np.arange(2500L * 600).astype(np.float32).reshape(2500, 600);
        var idx = Enumerable.Range(0, 600).Select(i => (long)(i * 7 % 600)).ToArray();
        AssertGather(a, idx, 1, "(2500, 600) float32");
    }

    /// <summary>
    ///     Empty operands: no indices, an empty outer axis, and an empty trailing axis — each NumPy's shape, a view.
    /// </summary>
    [TestMethod]
    public void EmptyOperands()
    {
        using var a = np.arange(12L).reshape(3, 4);
        AssertGather(a, Array.Empty<long>(), 1, "no indices");

        using var noRows = np.zeros(new Shape(0L, 5L));
        AssertGather(noRows, new long[] { 1, 2 }, 1, "an empty outer axis");

        using var noInner = np.zeros(new Shape(2L, 3L, 0L));
        AssertGather(noInner, new long[] { 2, 0 }, 1, "an empty trailing axis");
    }

    /// <summary>
    ///     Every source layout: F-ordered, transposed 3-D, column-stepped, reversed on both axes, an offset slice, and a
    ///     read-only broadcast (the result of a gather is a fresh buffer, so it is writeable even then).
    /// </summary>
    [TestMethod]
    public void EverySourceLayout()
    {
        using var c = np.arange(4L * 6).reshape(4, 6);
        using var f = np.asfortranarray(c);
        AssertGather(f, new long[] { 5, 0, 3 }, 1, "F-ordered");

        using var cube = np.arange(2L * 3 * 4).reshape(2, 3, 4);
        using var t = np.transpose(cube, new[] { 2, 0, 1 });
        AssertGather(t, new long[] { 1, 0 }, 1, "transposed 3-D");
        AssertGather(t, new long[] { 2, 0, 1 }, 2, "transposed 3-D, last axis");

        using var stepped = c[":, ::2"];
        AssertGather(stepped, new long[] { 2, 0, 1 }, 1, "column-stepped");

        using var reversed = c["::-1, ::-1"];
        AssertGather(reversed, new long[] { 5, 1, 2, 0 }, 1, "reversed");

        using var offset = c["1:, 2:"];
        AssertGather(offset, new long[] { 3, 0, 2 }, 1, "an offset slice");

        using var row = np.arange(5L);
        using var broadcast = np.broadcast_to(row, new Shape(4L, 5L));
        AssertGather(broadcast, new long[] { 4, 2, 0 }, 1, "a read-only broadcast");
    }

    /// <summary>
    ///     The result is a fresh buffer: writing into it leaves the source as it was.
    /// </summary>
    [TestMethod]
    public void WritesDoNotReachTheSource()
    {
        using var a = np.arange(12L).reshape(3, 4);
        using var indices = np.array(new long[] { 3, 1 });
        using var r = np.FancyIndexAlongAxis(a, indices, 1);
        r[":"] = 99L;
        r.ToArray<long>().Should().AllBeEquivalentTo(99L);
        a.ToArray<long>().Should().Equal(Enumerable.Range(0, 12).Select(i => (long)i));
    }

    /// <summary>
    ///     What the method does not serve raises <see cref="ArgumentOutOfRangeException"/>: axis 0 (use <c>np.take</c>), an
    ///     axis outside the array (it takes a normalized axis), and an index array that is not 1-D C-contiguous int64.
    /// </summary>
    [TestMethod]
    public void RejectsWhatItDoesNotServe()
    {
        using var a = np.arange(12L).reshape(3, 4);
        using var ok = np.array(new long[] { 1, 0 });

        foreach (int axis in new[] { 0, 2, -1 })
        {
            Action bad = () => np.FancyIndexAlongAxis(a, ok, axis);
            bad.Should().Throw<ArgumentOutOfRangeException>($"axis {axis} is not a non-leading axis of a 2-D array");
        }

        using var int32 = np.array(new[] { 1, 0 });
        using var twoD = np.array(new long[,] { { 1, 0 } });
        using var wide = np.arange(4L);
        using var strided = wide["::2"];
        foreach (var (name, indices) in new[] { ("int32", int32), ("2-D", twoD), ("strided", strided) })
        {
            Action bad = () => np.FancyIndexAlongAxis(a, indices, 1);
            bad.Should().Throw<ArgumentOutOfRangeException>($"{name} indices are not served");
        }
    }
}
