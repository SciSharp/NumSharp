using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.View
{
    public class UnmanagedStorageReshapeViewTest : TestClass
    {

        [Test]
        public void ReshapeSlicedArray()
        {
            var t = new UnmanagedStorage(np.arange(20).GetData(), new Shape(2, 10));
            var view = t.GetView(":, 5:");
            Assert.AreEqual(new Shape(2, 5), view.Shape);
            AssertAreEqual(new int[] { 5, 6, 7, 8, 9, 15, 16, 17, 18, 19 }, view.ToArray<int>());
            view.Reshape(10);
            Assert.AreEqual(new Shape(10), view.Shape);
            new int[] { 5, 6, 7, 8, 9, 15, 16, 17, 18, 19 }.Should().BeEquivalentTo( view.ToArray<int>());
            new NDArray(view).ToString(flat: true).Should().Be("array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])");
        }

        [Test]
        public void ExpandDimensions()
        {
            //>>> np.arange(6)
            //array([0, 1, 2, 3, 4, 5])
            //>>> np.arange(6).reshape(2, 3)
            //array([[0, 1, 2],
            //       [3, 4, 5]])
            //>>> a=np.arange(6).reshape(2,3)
            //>>> b=a[:, 1:]
            //>>> b
            //array([[1, 2],
            //       [4, 5]])
            //>>> c=b.reshape(2,1,2,1)
            //>>> c
            //array([[[[1],
            //         [2]]],
            //
            //
            //       [[[4],
            //         [5]]]])
            var t = new UnmanagedStorage(np.arange(6).GetData(), new Shape(2, 3));
            t.Shape.IsSliced.Should().Be(false);
            var view = t.GetView(":, 1:");
            view.Shape.IsSliced.Should().Be(true);
            new NDArray(view).ToString(flat: true).Should().Be("array([[1, 2], [4, 5]])");
            view.Reshape(2, 1, 2, 1);
            view.Shape.IsSliced.Should().Be(true);
            AssertAreEqual(new int[] { 1, 2, 4, 5 }, view.ToArray<int>());

            // doing the same disecting with slicing which ToString would do
            view.GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 1, 2, });
            view.GetView("1").ToArray<int>().Should().BeEquivalentTo(new int[] { 4, 5, });
            view.GetView("0").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 1, 2 });
            view.GetView("0").GetView("0").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 1, });
            view.GetView("0").GetView("0").GetView("1").ToArray<int>().Should().BeEquivalentTo(new int[] { 2, });
            view.GetView("0").GetView("0").GetView("0").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 1, });
            view.GetView("0").GetView("0").GetView("1").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 2, });
            view.GetView("1").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 4, 5 });
            view.GetView("1").GetView("0").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 4, });
            view.GetView("1").GetView("0").GetView("1").ToArray<int>().Should().BeEquivalentTo(new int[] { 5, });
            view.GetView("1").GetView("0").GetView("0").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 4, });
            view.GetView("1").GetView("0").GetView("1").GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 5, });

            // this is to show that ToString works in principle:
            np.arange(4).reshape(2, 1, 2, 1).ToString(flat: true).Should().Be("array([[[[0], [1]]], [[[2], [3]]]])");
            var nd = new NDArray(view);
            // and now tostring of the reshaped
            nd.ToString(flat: true).Should().Be("array([[[[1], [2]]], [[[4], [5]]]])");
        }

        [Test]
        public void SliceReshapedSlicedArray()
        {
            var t = new UnmanagedStorage(np.arange(20).GetData(), new Shape(2, 10));
            var view = t.GetView(":, 5:");
            view.Reshape(10);
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(new int[] { 5, 6, 7, 8, 9, 15, 16, 17, 18, 19 }, view.ToArray<int>());
            var v1 = view.GetView("1::2");
            AssertAreEqual(new int[] { 6, 8, 15, 17, 19 }, v1.ToArray<int>());
            new NDArray(v1).ToString(flat: true).Should().Be("array([6, 8, 15, 17, 19])");
        }

        [Test]
        public void TheUltimateTest______SliceReshapedSlicedReshapedSlicedArray()
        {
            var t = new UnmanagedStorage(np.arange(20).GetData(), new Shape(20));
            var view = t.GetView("::-1");
            view.Reshape(5, 4);
            var v1=view.GetView(":, 1:-1");
            new NDArray(v1).ToString(flat: true).Should().Be("array([[18, 17], [14, 13], [10, 9], [6, 5], [2, 1]])");
            v1.Reshape(1,2,5);
            new NDArray(v1).ToString(flat: true).Should().Be("array([[[18, 17, 14, 13, 10], [9, 6, 5, 2, 1]]])");
            var v2 = v1.GetView(":, ::-1, ::-2");
            new NDArray(v2).ToString(flat: true).Should().Be("array([[[1, 5, 9], [10, 14, 18]]])");
            v2.Reshape(2, 3, 1);
            new NDArray(v2).ToString(flat: true).Should().Be("array([[[1], [5], [9]], [[10], [14], [18]]])");
            var v3 = v2.GetView(":,::-2, 0");
            new NDArray(v3).ToString(flat: true).Should().Be("array([[9, 1], [18, 10]])");
            v3.SetData(ArraySlice.FromArray(new int[]{ 99, 11, -18, -10}));
            new NDArray(v3).ToString(flat: true).Should().Be("array([[99, 11], [-18, -10]])");
            new NDArray(v2).ToString(flat: true).Should().Be("array([[[11], [5], [99]], [[-10], [14], [-18]]])");
            new NDArray(v1).ToString(flat: true).Should().Be("array([[[-18, 17, 14, 13, -10], [99, 6, 5, 2, 11]]])");
            new NDArray(t).ToString(flat: true).Should().Be("array([0, 11, 2, 3, 4, 5, 6, 7, 8, 99, -10, 11, 12, 13, 14, 15, 16, 17, -18, 19])");
        }

        [Test]
        public void ReshapeSlicedArray1()
        {
            //>>> a
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            var t = new UnmanagedStorage(np.arange(9).GetData(), new Shape(3, 3));
            var view = t.GetView(":, 1:");
            //>>> a[:, 1:]
            //array([[1, 2],
            //       [4, 5],
            //       [7, 8]])
            view.Should().BeOfValues(1,2,4,5,7,8).And.BeShaped(3, 2);
            view.Reshape(2,3);
            //>>> a[:, 1:].reshape(2,3)
            //array([[1, 2, 4],
            //       [5, 7, 8]])
            view.Should().BeOfValues(1, 2, 4, 5, 7, 8).And.BeShaped(2,3);
            view.GetValue(0, 0).Should().Be(1);
            view.GetValue(1, 0).Should().Be(5);
            view.GetValue(1, 1).Should().Be(7);
            view.GetValue(1, 2).Should().Be(8);
        }

    }
}


