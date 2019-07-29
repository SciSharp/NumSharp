using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using FluentAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class BroadcastTests
    {
        [TestMethod]
        public void BroadcastArrayTest()
        {
            var arr1 = new int[][] {new int[] {1, 2, 3}};
            NDArray nd1 = np.array(arr1);
            var arr2 = new int[][] {new int[] {4}, new int[] {5}};
            NDArray nd2 = np.array(arr2);

            var (b, c) = np.broadcast_arrays(nd1, nd2);

            Assert.IsTrue(b.GetInt32(0, 0) == 1.0);
            Assert.IsTrue(b.GetInt32(0, 1) == 2.0);
            Assert.IsTrue(b.GetInt32(0, 2) == 3.0);
            Assert.IsTrue(c.GetInt32(0, 0) == 4.0);
            Assert.IsTrue(c.GetInt32(0, 1) == 4.0);
            Assert.IsTrue(c.GetInt32(0, 2) == 4.0);
            Assert.IsTrue(b.size == 6);
            Assert.IsTrue(c.size == 6);
        }

        /// <summary>
        ///     Taken from https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        /// </summary>
        [TestMethod]
        public void basics_ResolveReturnShape()
        {
            Shape arrOne;

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(5, 5), Shape.Scalar);
            arrOne.dimensions.Should().ContainInOrder(5, 5);

            arrOne = DefaultEngine.ResolveReturnShape(Shape.Scalar, new Shape(5, 5));
            arrOne.dimensions.Should().ContainInOrder(5, 5);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(5, 5), new Shape(5, 5));
            arrOne.dimensions.Should().ContainInOrder(5, 5);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(256, 256, 3), new Shape(3));
            arrOne.dimensions.Should().ContainInOrder(256, 256, 3);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(8, 1, 6, 1), new Shape(7, 1, 5));
            arrOne.dimensions.Should().ContainInOrder(8, 7, 6, 5);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(5, 4), new Shape(1));
            arrOne.dimensions.Should().ContainInOrder(5, 4);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(5, 4), new Shape(4));
            arrOne.dimensions.Should().ContainInOrder(5, 4);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(15, 3, 5), new Shape(15, 1, 5));
            arrOne.dimensions.Should().ContainInOrder(15, 3, 5);

            arrOne = DefaultEngine.ResolveReturnShape(new Shape(15, 3, 5), new Shape(3, 1));
            arrOne.dimensions.Should().ContainInOrder(15, 3, 5);

            new Action(() => DefaultEngine.ResolveReturnShape(new Shape(3), new Shape(4))).Should().Throw<Exception>();
            new Action(() => DefaultEngine.ResolveReturnShape(new Shape(2, 1), new Shape(8, 4, 3))).Should().Throw<Exception>();
        }

        /// <summary>
        ///     Taken from https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        /// </summary>
        [TestMethod]
        public void basics_broadcasting()
        {
            (Shape LeftShape, Shape RightShape) ret;

            ret = DefaultEngine.Broadcast(new Shape(5, 5), Shape.Scalar);
            ret.LeftShape.dimensions.Should().ContainInOrder(5, 5);
            ret.RightShape.dimensions.Should().ContainInOrder(5, 5);
            ret.RightShape.strides.Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret.RightShape.GetOffset(i.i1, i.i2)).Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret.LeftShape.GetOffset(i.i1, i.i2))
                .Should().BeInAscendingOrder().And.Subject.First().Should().Be(0);

            ret = DefaultEngine.Broadcast(Shape.Scalar, new Shape(5, 5));
            ret.LeftShape.dimensions.Should().ContainInOrder(5, 5);
            ret.RightShape.dimensions.Should().ContainInOrder(5, 5);
            ret.LeftShape.strides.Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret.LeftShape.GetOffset(i.i1, i.i2)).Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret.RightShape.GetOffset(i.i1, i.i2))
                .Should().BeInAscendingOrder().And.Subject.First().Should().Be(0);

            ret = DefaultEngine.Broadcast(new Shape(5, 5), new Shape(5, 5));
            ret.LeftShape.dimensions.Should().ContainInOrder(5, 5);
            ret.RightShape.dimensions.Should().ContainInOrder(5, 5);
            ret.LeftShape.strides.Should().ContainInOrder(5, 1);
            ret.RightShape.strides.Should().ContainInOrder(5, 1);

            ret = DefaultEngine.Broadcast(new Shape(5, 5, 3), new Shape(5, 3));
            ret.LeftShape.dimensions.Should().ContainInOrder(5, 5, 3);
            ret.RightShape.dimensions.Should().ContainInOrder(5, 5, 3);

            ret = DefaultEngine.Broadcast(new Shape(256, 256, 3), new Shape(3));
            ret.LeftShape.dimensions.Should().ContainInOrder(256, 256, 3);
            ret.RightShape.dimensions.Should().ContainInOrder(256, 256, 3);

            ret = DefaultEngine.Broadcast(new Shape(8, 1, 6, 1), new Shape(7, 1, 5));
            ret.LeftShape.dimensions.Should().ContainInOrder(8, 7, 6, 5);
            ret.RightShape.dimensions.Should().ContainInOrder(8, 7, 6, 5);

            ret = DefaultEngine.Broadcast(new Shape(5, 4), new Shape(1));
            ret.LeftShape.dimensions.Should().ContainInOrder(5, 4);
            ret.RightShape.dimensions.Should().ContainInOrder(5, 4);
            indexes2(5, 4).Select(i => ret.LeftShape.GetOffset(i.i1, i.i2))
                .Should().BeInAscendingOrder().And.Subject.First().Should().Be(0);
            indexes2(5, 4).Select(i => ret.RightShape.GetOffset(i.i1, i.i2)).Should().AllBeEquivalentTo(0);


            ret = DefaultEngine.Broadcast(new Shape(5, 4), new Shape(4));
            ret.LeftShape.dimensions.Should().ContainInOrder(5, 4);
            ret.RightShape.dimensions.Should().ContainInOrder(5, 4);

            ret = DefaultEngine.Broadcast(new Shape(15, 3, 5), new Shape(15, 1, 5));
            ret.LeftShape.dimensions.Should().ContainInOrder(15, 3, 5);
            ret.RightShape.dimensions.Should().ContainInOrder(15, 3, 5);

            ret = DefaultEngine.Broadcast(new Shape(15, 3, 5), new Shape(3, 1));
            ret.LeftShape.dimensions.Should().ContainInOrder(15, 3, 5);
            ret.RightShape.dimensions.Should().ContainInOrder(15, 3, 5);

            new Action(() => DefaultEngine.ResolveReturnShape(new Shape(3), new Shape(4))).Should().Throw<Exception>();
            new Action(() => DefaultEngine.ResolveReturnShape(new Shape(2, 1), new Shape(8, 4, 3))).Should().Throw<Exception>();

        }

        /// <summary>
        ///     Taken from https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        /// </summary>
        [TestMethod]
        public void basics_broadcasting_narrays()
        {
            Shape[] ret;

            ret = DefaultEngine.Broadcast(new Shape[] {new Shape(5, 5), Shape.Scalar});
            ret[0].dimensions.Should().ContainInOrder(5, 5);
            ret[1].dimensions.Should().ContainInOrder(5, 5);
            ret[1].strides.Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret[1].GetOffset(i.i1, i.i2)).Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret[0].GetOffset(i.i1, i.i2))
                .Should().BeInAscendingOrder().And.Subject.First().Should().Be(0);

            ret = DefaultEngine.Broadcast(new Shape[]{Shape.Scalar, new Shape(5, 5)});
            ret[0].dimensions.Should().ContainInOrder(5, 5);
            ret[1].dimensions.Should().ContainInOrder(5, 5);
            ret[0].strides.Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret[0].GetOffset(i.i1, i.i2)).Should().AllBeEquivalentTo(0);
            indexes2(5, 5).Select(i => ret[1].GetOffset(i.i1, i.i2))
                .Should().BeInAscendingOrder().And.Subject.First().Should().Be(0);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(5, 5), new Shape(5, 5)});
            ret[0].dimensions.Should().ContainInOrder(5, 5);
            ret[1].dimensions.Should().ContainInOrder(5, 5);
            ret[0].strides.Should().ContainInOrder(5, 1);
            ret[1].strides.Should().ContainInOrder(5, 1);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(5, 5, 3), new Shape(5, 3)});
            ret[0].dimensions.Should().ContainInOrder(5, 5, 3);
            ret[1].dimensions.Should().ContainInOrder(5, 5, 3);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(256, 256, 3), new Shape(3)});
            ret[0].dimensions.Should().ContainInOrder(256, 256, 3);
            ret[1].dimensions.Should().ContainInOrder(256, 256, 3);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(8, 1, 6, 1), new Shape(7, 1, 5)});
            ret[0].dimensions.Should().ContainInOrder(8, 7, 6, 5);
            ret[1].dimensions.Should().ContainInOrder(8, 7, 6, 5);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(5, 4), new Shape(1)});
            ret[0].dimensions.Should().ContainInOrder(5, 4);
            ret[1].dimensions.Should().ContainInOrder(5, 4);
            indexes2(5, 4).Select(i => ret[0].GetOffset(i.i1, i.i2))
                .Should().BeInAscendingOrder().And.Subject.First().Should().Be(0);
            indexes2(5, 4).Select(i => ret[1].GetOffset(i.i1, i.i2)).Should().AllBeEquivalentTo(0);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(5, 4), new Shape(4)});
            ret[0].dimensions.Should().ContainInOrder(5, 4);
            ret[1].dimensions.Should().ContainInOrder(5, 4);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(15, 3, 5), new Shape(15, 1, 5)});
            ret[0].dimensions.Should().ContainInOrder(15, 3, 5);
            ret[1].dimensions.Should().ContainInOrder(15, 3, 5);

            ret = DefaultEngine.Broadcast(new Shape[]{new Shape(15, 3, 5), new Shape(3, 1)});
            ret[0].dimensions.Should().ContainInOrder(15, 3, 5);
            ret[1].dimensions.Should().ContainInOrder(15, 3, 5);

            new Action(() => DefaultEngine.ResolveReturnShape(new Shape(3), new Shape(4))).Should().Throw<Exception>();
            new Action(() => DefaultEngine.ResolveReturnShape(new Shape(2, 1), new Shape(8, 4, 3))).Should().Throw<Exception>();
        }

        [TestMethod]
        public void broadcast_accessing()
        {
            var left = _create_arange(5, 5);
            var right = _create_arange(5, 1);
            var x = DefaultEngine.Broadcast(left.Shape, right.Shape);
            left.Storage.SetShapeUnsafe(x.LeftShape);
            right.Storage.SetShapeUnsafe(x.RightShape);
            var ret = new NDArray(typeof(int), x.LeftShape);
            foreach ((int i1, int i2) in indexes2(5, 5))
            {
                ret.SetValue(left.GetInt32(i1, i2) + right.GetInt32(i1, i2), i1, i2);
                Console.WriteLine($"{ret.GetInt32(i1, i2)} = {left.GetInt32(i1, i2)} {right.GetInt32(i1, i2)}");
            }

            Console.WriteLine(ret.ToString(false));

            left = _create_arange(2, 5, 5);
            right = _create_arange(1, 1, 5);
            x = DefaultEngine.Broadcast(left.Shape, right.Shape);
            left.Storage.SetShapeUnsafe(x.LeftShape);
            right.Storage.SetShapeUnsafe(x.RightShape);
            ret = new NDArray(typeof(int), x.LeftShape);
            foreach ((int i1, int i2, int i3) in indexes3(2, 5, 5))
            {
                ret.SetValue(left.GetInt32(i1, i2, i3) + right.GetInt32(i1, i2, i3), i1, i2, i3);
                Console.WriteLine($"{ret.GetInt32(i1, i2, i3)} = {left.GetInt32(i1, i2)} {right.GetInt32(i1, i2, i3)}");
            }

            Console.WriteLine(ret.ToString(false));
        }

        NDArray _create_arange(Shape shape)
        {
            return np.arange(shape.size).reshape(ref shape);
        }

        NDArray _create_arange(params int[] dims)
        {
            var rshape = new Shape(dims);
            return np.arange(rshape.size).reshape(rshape);
        }

        IEnumerable<int> indexes(int len)
        {
            for (int i = 0; i < len; i++)
            {
                yield return i;
            }
        }

        IEnumerable<(int i1, int i2)> indexes2(int len1, int len2)
        {
            for (int i = 0; i < len1; i++)
            {
                for (int j = 0; j < len2; j++)
                {
                    yield return (i, j);
                }
            }
        }

        IEnumerable<(int i1, int i2, int i3)> indexes3(int len1, int len2, int len3)
        {
            for (int i = 0; i < len1; i++)
            {
                for (int j = 0; j < len2; j++)
                {
                    for (int k = 0; k < len3; k++)
                    {
                        yield return (i, j, k);
                    }
                }
            }
        }
    }
}
