using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;

namespace NumSharp.UnitTest.Manipulation;

[TestClass]
public class NpyIterCopyTests : TestClass
{
    [TestMethod]
    public void Copyto_StridedDestination_SameDType()
    {
        var dst = np.zeros(8, np.int64);
        var view = dst["::2"];
        var src = np.array(new long[] { 10, 20, 30, 40 });

        np.copyto(view, src);

        dst.Should().BeOfValues(10L, 0L, 20L, 0L, 30L, 0L, 40L, 0L);
    }

    [TestMethod]
    public void Copyto_BroadcastSource_ToStridedDestination_SameDType()
    {
        var dst = np.zeros(new Shape(2, 6), np.int64);
        var view = dst[":, ::2"];
        var src = np.array(new long[] { 7, 8, 9 });

        np.copyto(view, src);

        var expected = np.array(new long[,]
        {
            { 7, 0, 8, 0, 9, 0 },
            { 7, 0, 8, 0, 9, 0 }
        });

        np.array_equal(dst, expected).Should().BeTrue();
    }

    [TestMethod]
    public void Copyto_TransposeView_SameDType()
    {
        var dst = np.zeros(new Shape(2, 3), np.int64);
        var src = np.array(new long[,]
        {
            { 0, 1 },
            { 2, 3 },
            { 4, 5 }
        });

        np.copyto(dst.T, src);

        var expected = np.array(new long[,]
        {
            { 0, 2, 4 },
            { 1, 3, 5 }
        });

        np.array_equal(dst, expected).Should().BeTrue();
    }

    [TestMethod]
    public void Copy_NonContiguousView_SameDType()
    {
        var src = np.arange(12).reshape(3, 4).T;

        var clone = src.copy();

        np.array_equal(clone, src).Should().BeTrue();
        clone.Shape.IsContiguous.Should().BeTrue();
    }

    [TestMethod]
    public void Copyto_BoolColumnSlice_ToBoolColumnSlice_SameDType()
    {
        var src = np.array(new bool[,]
        {
            { true, false },
            { false, true }
        });
        var dst = np.zeros(new Shape(2, 2), np.bool_);

        np.copyto(dst[":, :1"], src[":, 1:"]);

        var expected = np.array(new bool[,]
        {
            { false, false },
            { true, false }
        });

        np.array_equal(dst, expected).Should().BeTrue();
    }

    [TestMethod]
    public void Copyto_BroadcastSource_ToNegativeStrideDestination_SameDType()
    {
        var backing = np.zeros(new Shape(3, 4), np.int64);
        var dst = backing[":, ::-2"];
        var src = np.broadcast_to(np.array(new long[,] { { 10 }, { 20 }, { 30 } }), new Shape(3, 2));

        np.copyto(dst, src);

        var expectedBacking = np.array(new long[,]
        {
            { 0, 10, 0, 10 },
            { 0, 20, 0, 20 },
            { 0, 30, 0, 30 }
        });
        var expectedView = np.array(new long[,]
        {
            { 10, 10 },
            { 20, 20 },
            { 30, 30 }
        });

        np.array_equal(backing, expectedBacking).Should().BeTrue();
        np.array_equal(dst, expectedView).Should().BeTrue();
    }

    [TestMethod]
    public void Copyto_TransposedOffsetDestination_SameDType()
    {
        var backing = np.zeros(new Shape(4, 5), np.int64);
        var dst = backing.T["1:4, ::-1"];
        var src = np.array(new long[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        });

        np.copyto(dst, src);

        var expectedBacking = np.array(new long[,]
        {
            { 0, 4, 8, 12, 0 },
            { 0, 3, 7, 11, 0 },
            { 0, 2, 6, 10, 0 },
            { 0, 1, 5, 9, 0 }
        });

        np.array_equal(backing, expectedBacking).Should().BeTrue();
    }

    [TestMethod]
    public void Copyto_BoolChainedViews_SameDType()
    {
        var src = np.array(new bool[,]
        {
            { true, false, true, false },
            { false, true, false, true },
            { true, true, false, false }
        }).T["1:, ::-1"];
        var backing = np.zeros(new Shape(4, 3), np.bool_);
        var dst = backing["::-1, :"][":-1, :"];

        np.copyto(dst, src);

        var expectedBacking = np.array(new bool[,]
        {
            { false, false, false },
            { false, true, false },
            { false, false, true },
            { true, true, false }
        });

        np.array_equal(backing, expectedBacking).Should().BeTrue();
    }

    [TestMethod]
    public void Copy_BroadcastColumnView_MaterializesContiguousWritableCopy()
    {
        var src = np.broadcast_to(np.array(new long[,] { { 1 }, { 2 }, { 3 } }), new Shape(3, 4));

        var copy = src.copy();

        copy.Should().BeShaped(3, 4);
        copy.Should().BeOfValues(1L, 1L, 1L, 1L, 2L, 2L, 2L, 2L, 3L, 3L, 3L, 3L);
        copy.Shape.IsContiguous.Should().BeTrue();
        copy.Shape.IsBroadcasted.Should().BeFalse();
        copy.Shape.IsWriteable.Should().BeTrue();
    }

    [TestMethod]
    public void Copy_TransposedOffsetView_MaterializesExpectedOrder()
    {
        var src = np.arange(12).reshape(3, 4).T["1:, ::-1"];

        var copy = src.copy();

        copy.Should().BeShaped(3, 3);
        copy.Should().BeOfValues(9L, 5L, 1L, 10L, 6L, 2L, 11L, 7L, 3L);
        copy.Shape.IsContiguous.Should().BeTrue();
        copy.Shape.IsBroadcasted.Should().BeFalse();
    }
}
