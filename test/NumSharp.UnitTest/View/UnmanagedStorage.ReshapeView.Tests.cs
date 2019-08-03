using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

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
            AssertAreEqual(new int[] { 5, 6, 7, 8, 9, 15, 16, 17, 18, 19 }, view.ToArray<int>());
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
            new NDArray( view).ToString(flat:true).Should().Be("array([[1, 2], [4, 5]])");
            view.Reshape(2,1,2,1);
            view.Shape.IsSliced.Should().Be(true);
            AssertAreEqual(new int[]{1, 2, 4, 5}, view.ToArray<int>());
            // this is to show that ToString works in principle:
            np.arange(4).reshape(2, 1, 2, 1).ToString(flat: true).Should().Be("array([[[[0], [1]]], [[[2], [3]]]])");
            var nd = new NDArray(view);
            // but here something goes wrong
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


