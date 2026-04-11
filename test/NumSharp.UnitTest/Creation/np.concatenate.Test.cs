using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    public class np_concatenate_test : TestClass
    {
        [Test]
        public void Case1_Axis1()
        {
            var a = np.full(new Shape(3, 1, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 2, 3), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), 1);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, 0, :"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, 1, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, 2, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void Case1_Axis1_Cast()
        {
            var a = np.full(new Shape(3, 1, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 2, 3), 2, NPTypeCode.Double);
            var c = np.concatenate((a, b), 1);

            c.dtype.Should().Be<double>();
            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, 0, :"].flat.Cast<double>().Should().AllBeEquivalentTo(1);
            c[":, 1, :"].flat.Cast<double>().Should().AllBeEquivalentTo(2);
            c[":, 2, :"].flat.Cast<double>().Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void Case1_Axis0()
        {
            var a = np.full(new Shape(1, 3, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(2, 3, 3), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), 0);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c["0, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c["1, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c["2, :, :"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void Case1_Axis2()
        {
            var a = np.full(new Shape(3, 3, 1), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 3, 2), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), 2);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, :, 0"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, :, 1"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, :, 2"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void Case1_Axis_minus1()
        {
            var a = np.full(new Shape(3, 3, 1), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 3, 2), 2, NPTypeCode.Int32);
            var c = np.concatenate((a, b), -1);

            c.shape.Should().HaveCount(3).And.ContainInOrder(3, 3, 3);
            c.size.Should().Be(27);
            c[":, :, 0"].flat.Cast<int>().Should().AllBeEquivalentTo(1);
            c[":, :, 1"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
            c[":, :, 2"].flat.Cast<int>().Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void Case2_Axis1_3Arrays_Cast()
        {
            var a = np.full(new Shape(3, 1, 3), 1, NPTypeCode.Int32);
            var b = np.full(new Shape(3, 2, 3), 2, NPTypeCode.Decimal);
            var c = np.full(new Shape(3, 1, 3), 2, NPTypeCode.Byte);
            var d = np.concatenate((a, b, c), 1);
            d.dtype.Should().Be<decimal>();
            d.shape.Should().HaveCount(3).And.ContainInOrder(3, 4, 3);
            d.size.Should().Be(36);
            d[":, 0, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(1);
            d[":, 1, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(2);
            d[":, 2, :"].flat.Cast<decimal>().Should().AllBeEquivalentTo(2);
        }

        /// <summary>
        ///     hstack with column-broadcast arrays correctly preserves row data.
        ///     Verifies: broadcast_to column vectors, then hstack horizontally.
        ///     NumPy: np.hstack([np.broadcast_to([[1],[2]],(2,2)), np.broadcast_to([[3],[4]],(2,3))])
        ///            → [[1,1,3,3,3], [2,2,4,4,4]]
        /// </summary>
        [Test]
        public void Hstack_BroadcastColumnVectors()
        {
            var a = np.broadcast_to(np.array(new int[,] { { 1 }, { 2 } }), new Shape(2, 2));
            var b = np.broadcast_to(np.array(new int[,] { { 3 }, { 4 } }), new Shape(2, 3));
            var r = np.hstack(a, b);

            r.shape.Should().BeEquivalentTo(new long[] { 2, 5 });

            var expected = np.array(new int[,] { { 1, 1, 3, 3, 3 }, { 2, 2, 4, 4, 4 } });
            np.array_equal(r, expected).Should().BeTrue(
                "hstack with broadcast arrays should preserve row data correctly");
        }

        /// <summary>
        ///     vstack with sliced+broadcast array correctly reads all rows.
        ///     Verifies: slice column, broadcast, then vstack vertically.
        ///     NumPy: x = np.arange(6).reshape(3,2)
        ///            np.vstack([np.broadcast_to(x[:,0:1],(3,3)), [[10,20,30]]])
        ///            → [[0,0,0], [2,2,2], [4,4,4], [10,20,30]]
        /// </summary>
        [Test]
        public void Vstack_SlicedBroadcast()
        {
            var x = np.arange(6).reshape(3, 2);
            var col = x[":, 0:1"]; // (3,1): [[0],[2],[4]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));
            var other = np.array(new int[,] { { 10, 20, 30 } });

            var r = np.vstack(bcol, other);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 3 });

            r.GetInt64(0, 0).Should().Be(0L, "Row 0 should be [0,0,0]");
            r.GetInt64(1, 0).Should().Be(2L, "Row 1 should be [2,2,2]");
            r.GetInt64(2, 0).Should().Be(4L, "Row 2 should be [4,4,4]");
            r.GetInt64(3, 0).Should().Be(10L, "Row 3 should be [10,20,30]");
        }

        /// <summary>
        ///     concatenate axis=0 with sliced+broadcast correctly reads all data.
        ///     Verifies: slice non-first column, broadcast, then concatenate.
        ///     NumPy: x = np.arange(12).reshape(3,4)
        ///            np.concatenate([np.broadcast_to(x[:,1:2],(3,3)), np.ones((1,3),dtype=int)])
        ///            → [[1,1,1], [5,5,5], [9,9,9], [1,1,1]]
        /// </summary>
        [Test]
        public void Concatenate_SlicedBroadcast_Axis0()
        {
            var x = np.arange(12).reshape(3, 4);
            var col = x[":, 1:2"]; // (3,1): [[1],[5],[9]]
            var bcol = np.broadcast_to(col, new Shape(3, 3));
            var other = np.ones(new Shape(1, 3), np.int32);

            var r = np.concatenate(new NDArray[] { bcol, other }, axis: 0);
            r.shape.Should().BeEquivalentTo(new long[] { 4, 3 });

            r.GetInt64(0, 0).Should().Be(1L, "Row 0 should be [1,1,1]");
            r.GetInt64(1, 0).Should().Be(5L, "Row 1 should be [5,5,5]");
            r.GetInt64(2, 0).Should().Be(9L, "Row 2 should be [9,9,9]");
            r.GetInt64(3, 0).Should().Be(1L, "Row 3 should be [1,1,1]");
        }
    }
}
