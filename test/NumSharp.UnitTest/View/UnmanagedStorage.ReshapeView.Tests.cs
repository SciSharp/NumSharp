using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.View
{
    [TestClass]
    public class UnmanagedStorageReshapeViewTest : TestClass
    {

        [TestMethod]
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

        [TestMethod]
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
            var v=view.GetView("0").GetView("0").GetView("1");
            v.GetView("0").ToArray<int>().Should().BeEquivalentTo(new int[] { 2, });

            // this is to show that ToString works in principle:
            np.arange(4).reshape(2, 1, 2, 1).ToString(flat: true).Should().Be("array([[[[0], [1]]], [[[2], [3]]]])");
            var nd = new NDArray(view);
            // and now tostring of the reshaped
            nd.ToString(flat: true).Should().Be("array([[[[1], [2]]], [[[4], [5]]]])");
        }

        [TestMethod]
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
    }
}


