using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.View
{
    [TestClass]
    public class UnmanagedStorageTest : TestClass
    {
        [TestMethod]
        public void GetData_1D()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Assert.AreEqual(new Shape(10), data.Shape);
            // return identical view
            var view = data.GetView(":");
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(data.ToArray<int>(), view.ToArray<int>());
            view = data.GetView("-77:77");
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(data.ToArray<int>(), view.ToArray<int>());
            // return reduced view
            view = data.GetView("7:");
            Assert.AreEqual(new Shape(3), view.Shape);
            AssertAreEqual(new int[] { 7, 8, 9 }, view.ToArray<int>());
            view = data.GetView(":5");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 0, 1, 2, 3, 4 }, view.ToArray<int>());
            view = data.GetView("2:3");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new int[] { 2 }, view.ToArray<int>());
            // return stepped view
            view = data.GetView("::2");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 0, 2, 4, 6, 8 }, view.ToArray<int>());
            view = data.GetView("::3");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 0, 3, 6, 9 }, view.ToArray<int>());
            view = data.GetView("::77");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new[] { 0 }, view.ToArray<int>());
            view = data.GetView("-77:77:77");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new[] { 0 }, view.ToArray<int>());
            // negative step!
            view = data.GetView("::-1");
            Assert.AreEqual(new Shape(10), view.Shape);
            AssertAreEqual(data.ToArray<int>().OfType<int>().Reverse().ToArray(), view.ToArray<int>());
            view = data.GetView("::-2");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 9, 7, 5, 3, 1 }, view.ToArray<int>());
            view = data.GetView("::-3");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 9, 6, 3, 0 }, view.ToArray<int>());
            view = data.GetView("::-77");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new[] { 9 }, view.ToArray<int>());
            view = data.GetView("77:-77:-77");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new[] { 9 }, view.ToArray<int>());
        }

        [TestMethod]
        public void GetData_1D_Negative()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            // return reduced view
            var view = data.GetView("-7:");
            Assert.AreEqual(new Shape(7), view.Shape);
            AssertAreEqual(new int[] { 3, 4, 5, 6, 7, 8, 9 }, view.ToArray<int>());
            view = data.GetView(":-5");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 0, 1, 2, 3, 4 }, view.ToArray<int>());
            view = data.GetView(":-6");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 0, 1, 2, 3, }, view.ToArray<int>());
            view = data.GetView("-3:-2");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new int[] { 7 }, view.ToArray<int>());
            // negative step!
            view = data.GetView("-7::-1");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 3, 2, 1, 0 }, view.ToArray<int>());
            view = data.GetView(":-5:-1");
            Assert.AreEqual(new Shape(4), view.Shape);
            AssertAreEqual(new int[] { 9, 8, 7, 6 }, view.ToArray<int>());
            view = data.GetView(":-6:-1");
            Assert.AreEqual(new Shape(5), view.Shape);
            AssertAreEqual(new int[] { 9, 8, 7, 6, 5, }, view.ToArray<int>());
            view = data.GetView("-2:-3:-1");
            Assert.AreEqual(new Shape(1), view.Shape);
            AssertAreEqual(new int[] { 8 }, view.ToArray<int>());
        }

        [TestMethod]
        public void Indexing_1D()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            // return identical view
            var view = data.GetView(":");
            Assert.AreEqual(0, view.GetValue<int>(0));
            Assert.AreEqual(5, view.GetValue<int>(5));
            Assert.AreEqual(9, view.GetValue<int>(9));
            view = data.GetView("-77:77");
            Assert.AreEqual(0, view.GetValue<int>(0));
            Assert.AreEqual(5, view.GetValue<int>(5));
            Assert.AreEqual(9, view.GetValue<int>(9));
            // return reduced view
            view = data.GetView("7:");
            Assert.AreEqual(7, view.GetValue<int>(0));
            Assert.AreEqual(8, view.GetValue<int>(1));
            Assert.AreEqual(9, view.GetValue<int>(2));
            view = data.GetView(":5");
            Assert.AreEqual(0, view.GetValue<int>(0));
            Assert.AreEqual(1, view.GetValue<int>(1));
            Assert.AreEqual(2, view.GetValue<int>(2));
            Assert.AreEqual(3, view.GetValue<int>(3));
            Assert.AreEqual(4, view.GetValue<int>(4));
            view = data.GetView("2:3");
            Assert.AreEqual(2, view.GetValue<int>(0));
            // return stepped view
            view = data.GetView("::2");
            Assert.AreEqual(0, view.GetValue<int>(0));
            Assert.AreEqual(2, view.GetValue<int>(1));
            Assert.AreEqual(4, view.GetValue<int>(2));
            Assert.AreEqual(6, view.GetValue<int>(3));
            Assert.AreEqual(8, view.GetValue<int>(4));
            view = data.GetView("::3");
            Assert.AreEqual(0, view.GetValue<int>(0));
            Assert.AreEqual(3, view.GetValue<int>(1));
            Assert.AreEqual(6, view.GetValue<int>(2));
            Assert.AreEqual(9, view.GetValue<int>(3));
            view = data.GetView("-77:77:77");
            Assert.AreEqual(0, view.GetValue<int>(0));
            // negative step!
            view = data.GetView("::-1");
            Assert.AreEqual(9, view.GetValue<int>(0));
            Assert.AreEqual(4, view.GetValue<int>(5));
            Assert.AreEqual(0, view.GetValue<int>(9));
            view = data.GetView("::-2");
            Assert.AreEqual(9, view.GetValue<int>(0));
            Assert.AreEqual(7, view.GetValue<int>(1));
            Assert.AreEqual(5, view.GetValue<int>(2));
            Assert.AreEqual(3, view.GetValue<int>(3));
            Assert.AreEqual(1, view.GetValue<int>(4));
            view = data.GetView("::-3");
            Assert.AreEqual(9, view.GetValue<int>(0));
            Assert.AreEqual(6, view.GetValue<int>(1));
            Assert.AreEqual(3, view.GetValue<int>(2));
            Assert.AreEqual(0, view.GetValue<int>(3));
            view = data.GetView("77:-77:-77");
            Assert.AreEqual(9, view.GetValue<int>(0));
        }

        [TestMethod]
        public void NestedView_1D()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            // return identical view
            var identical = data.GetView(":");
            Assert.AreEqual(new Shape(10), identical.Shape);
            var view1 = identical.GetView("1:9");
            Assert.AreEqual(new Shape(8), view1.Shape);
            AssertAreEqual(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, }, view1.ToArray<int>());
            var view2 = view1.GetView("::-2");
            Assert.AreEqual(new Shape(4), view2.Shape);
            AssertAreEqual(new int[] { 8, 6, 4, 2, }, view2.ToArray<int>());
            var view3 = view2.GetView("::-3");
            Assert.AreEqual(new Shape(2), view3.Shape);
            AssertAreEqual(new int[] { 2, 8 }, view3.ToArray<int>());
            // all must see the same modifications, no matter if original or any view is modified
            // modify original
            data.SetData(ArraySlice.FromArray(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 }));
            var arr = view1.ToArray<int>();
            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, -8, }, view1.ToArray<int>());
            AssertAreEqual(new int[] { -8, -6, -4, -2, }, view2.ToArray<int>());
            AssertAreEqual(new int[] { -2, -8 }, view3.ToArray<int>());
            // modify views
            view1.SetData(88, 7);
            AssertAreEqual(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, 88, -9 }, data.ToArray<int>());
            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, 88, }, view1.ToArray<int>());
            AssertAreEqual(new int[] { 88, -6, -4, -2, }, view2.ToArray<int>());
            AssertAreEqual(new int[] { -2, 88 }, view3.ToArray<int>());
            view3.SetData(22, 0);
            AssertAreEqual(new int[] { 0, -1, 22, -3, -4, -5, -6, -7, 88, -9 }, data.ToArray<int>());
            AssertAreEqual(new int[] { -1, 22, -3, -4, -5, -6, -7, 88, }, view1.ToArray<int>());
            AssertAreEqual(new int[] { 88, -6, -4, 22, }, view2.ToArray<int>());
            AssertAreEqual(new int[] { 22, 88 }, view3.ToArray<int>());
        }

        [TestMethod]
        public void GetData_2D()
        {
            //>>> x = np.arange(9).reshape(3, 3)
            //>>> x
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            //>>> x[:]
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            //>>> x[1:]
            //array([[3, 4, 5],
            //       [6, 7, 8]])
            //>>> x[1:,:]
            //array([[3, 4, 5],
            //       [6, 7, 8]])
            //>>> x[:, 1:]
            //array([[1, 2],
            //       [4, 5],
            //       [7, 8]])
            //>>> x[1:2, 0:1]
            //array([[3]])
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
            data.Reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), data.Shape);
            // return identical view
            var view = data.GetView(":");
            Assert.AreEqual(new Shape(3, 3), view.Shape);
            AssertAreEqual(data.ToArray<int>(), view.ToArray<int>());
            // return reduced view
            view = data.GetView("1:");
            Assert.AreEqual(new Shape(2, 3), view.Shape);
            AssertAreEqual(new int[] { 3, 4, 5, 6, 7, 8 }, view.ToArray<int>());
            view = data.GetView("1:,:");
            Assert.AreEqual(new Shape(2, 3), view.Shape);
            AssertAreEqual(new int[] { 3, 4, 5, 6, 7, 8 }, view.ToArray<int>());

            view = data.GetView(":,1:");
            Assert.AreEqual(new Shape(3, 2), view.Shape);
            AssertAreEqual(new int[] { 1, 2, 4, 5, 7, 8 }, view.ToArray<int>());
            view = data.GetView("1:2, 0:1");
            Assert.AreEqual(new Shape(1, 1), view.Shape);
            AssertAreEqual(new int[] { 3 }, view.ToArray<int>());
            // return stepped view
            //>>> x
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            //>>> x[::2]
            //array([[0, 1, 2],
            //       [6, 7, 8]])
            //>>> x[::3]
            //array([[0, 1, 2]])
            //>>> x[::- 1]
            //array([[6, 7, 8],
            //       [3, 4, 5],
            //       [0, 1, 2]])
            //>>> x[::- 2]
            //array([[6, 7, 8],
            //       [0, 1, 2]])
            //>>> x[::- 3]
            //array([[6, 7, 8]])
            view = data.GetView("::2");
            Assert.AreEqual(new Shape(2, 3), view.Shape);
            AssertAreEqual(new int[] { 0, 1, 2, 6, 7, 8 }, view.ToArray<int>());
            view = data.GetView("::3");
            Assert.AreEqual(new Shape(1, 3), view.Shape);
            AssertAreEqual(new int[] { 0, 1, 2 }, view.ToArray<int>());
            view = data.GetView("::-1");
            Assert.AreEqual(new Shape(3, 3), view.Shape);
            AssertAreEqual(new int[] { 6, 7, 8, 3, 4, 5, 0, 1, 2, }, view.ToArray<int>());
            view = data.GetView("::-2");
            Assert.AreEqual(new Shape(2, 3), view.Shape);
            AssertAreEqual(new int[] { 6, 7, 8, 0, 1, 2, }, view.ToArray<int>());
            view = data.GetView("::-3");
            Assert.AreEqual(new Shape(1, 3), view.Shape);
            AssertAreEqual(new int[] { 6, 7, 8, }, view.ToArray<int>());
            // N-Dim Stepping
            //>>> x[::2,::2]
            //array([[0, 2],
            //       [6, 8]])
            //>>> x[::- 1,::- 2]
            //array([[8, 6],
            //       [5, 3],
            //       [2, 0]])
            view = data.GetView("::2, ::2");
            Assert.AreEqual(new Shape(2, 2), view.Shape);
            AssertAreEqual(new int[] { 0, 2, 6, 8 }, view.ToArray<int>());
            view = data.GetView("::-1, ::-2");
            Assert.AreEqual(new Shape(3, 2), view.Shape);
            AssertAreEqual(new int[] { 8, 6, 5, 3, 2, 0 }, view.ToArray<int>());
        }

        [TestMethod]
        public void NestedView_2D()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            data.Reshape(2, 10);
            //>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            //>>> x = x.reshape(2, 10)
            //>>> x
            //array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            //       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
            // return identical view
            var identical = data.GetView(":");
            Assert.AreEqual(new Shape(2, 10), identical.Shape);
            //>>> x[:, 1:9]
            //array([[1, 2, 3, 4, 5, 6, 7, 8],
            //       [1, 2, 3, 4, 5, 6, 7, 8]])
            var view1 = identical.GetView(":,1:9");
            Assert.AreEqual(new Shape(2, 8), view1.Shape);
            AssertAreEqual(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 }, view1.ToArray<int>());
            //>>> x[:, 1:9][:,::- 2]
            //array([[8, 6, 4, 2],
            //       [8, 6, 4, 2]])
            var view2 = view1.GetView(":,::-2");
            Assert.AreEqual(new Shape(2, 4), view2.Shape);
            AssertAreEqual(new int[] { 8, 6, 4, 2, 8, 6, 4, 2 }, view2.ToArray<int>());
            //>>> x[:, 1:9][:,::- 2][:,::- 3]
            //array([[2, 8],
            //       [2, 8]])
            var view3 = view2.GetView(":,::-3");
            Assert.AreEqual(new Shape(2, 2), view3.Shape);
            AssertAreEqual(new int[] { 2, 8, 2, 8 }, view3.ToArray<int>());
            // all must see the same modifications, no matter if original or any view is modified
            // modify original
            data.SetData(ArraySlice.FromArray(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 }));
            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, -8, -1, -2, -3, -4, -5, -6, -7, -8 }, view1.ToArray<int>());
            AssertAreEqual(new int[] { -8, -6, -4, -2, -8, -6, -4, -2 }, view2.ToArray<int>());
            AssertAreEqual(new int[] { -2, -8, -2, -8 }, view3.ToArray<int>());
            // modify views
            view1.SetData(88, 0, 7);
            view1.SetData(888, 1, 7);
            AssertAreEqual(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, 88, -9, 0, -1, -2, -3, -4, -5, -6, -7, 888, -9 },
                data.ToArray<int>());
            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, 88, -1, -2, -3, -4, -5, -6, -7, 888 },
                view1.ToArray<int>());
            AssertAreEqual(new int[] { 88, -6, -4, -2, 888, -6, -4, -2 }, view2.ToArray<int>());
            AssertAreEqual(new int[] { -2, 88, -2, 888 }, view3.ToArray<int>());
            view3.SetData(22, 0, 0);
            view3.SetData(222, 1, 0);
            AssertAreEqual(new int[] { 0, -1, 22, -3, -4, -5, -6, -7, 88, -9, 0, -1, 222, -3, -4, -5, -6, -7, 888, -9 },
                data.ToArray<int>());
            AssertAreEqual(new int[] { -1, 22, -3, -4, -5, -6, -7, 88, -1, 222, -3, -4, -5, -6, -7, 888 },
                view1.ToArray<int>());
            AssertAreEqual(new int[] { 88, -6, -4, 22, 888, -6, -4, 222 }, view2.ToArray<int>());
            AssertAreEqual(new int[] { 22, 88, 222, 888 }, view3.ToArray<int>());
        }

        [TestMethod]
        public void Reduce_1D_to_Scalar()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
            Assert.AreEqual(new Shape(10), data.Shape);
            // return scalar
            var view = data.GetView("7");
            Assert.AreEqual(Shape.Scalar, view.Shape);
            AssertAreEqual(new int[] { 7 }, view.ToArray<int>());
        }

        [TestMethod]
        public void Reduce_2D_to_1D_and_0D()
        {
            //>>> x = np.arange(9).reshape(3, 3)
            //>>> x
            //array([[0, 1, 2],
            //       [3, 4, 5],
            //       [6, 7, 8]])
            //>>> x[1]
            //array([3, 4, 5])
            //>>> x[:,1]
            //array([1, 4, 7])
            //>>> x[2, 2]
            //8
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
            data.Reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), data.Shape);
            // return identical view
            var view = data.GetView("1");
            Assert.AreEqual(new Shape(3), view.Shape);
            AssertAreEqual(new int[] { 3, 4, 5 }, view.ToArray<int>());
            // return reduced view
            view = data.GetView(":,1");
            Assert.AreEqual(new Shape(3), view.Shape);
            AssertAreEqual(new int[] { 1, 4, 7 }, view.ToArray<int>());

            view = data.GetView("2,2");
            Assert.AreEqual(Shape.Scalar, view.Shape);
            AssertAreEqual(new int[] { 8 }, view.ToArray<int>());
            // recursive dimensionality reduction
            view = data.GetView("2").GetView("2");
            Assert.AreEqual(Shape.Scalar, view.Shape);
            AssertAreEqual(new int[] { 8 }, view.ToArray<int>());
        }

        [TestMethod]
        public void NestedDimensionalityReduction()
        {
            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
            data.Reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), data.Shape);
            var view = data.GetView("2");
            //Assert.AreEqual(new Shape(3), view.Shape);
            //AssertAreEqual(new int[] { 6, 7, 8 }, view.ToArray<int>());
            var view1 = view.GetView("2");
            Assert.AreEqual(Shape.Scalar, view1.Shape);
            AssertAreEqual(new int[] { 8 }, view1.ToArray<int>());
            var view2 = view.GetView("1::-1");
            Assert.AreEqual(new Shape(2), view2.Shape);
            AssertAreEqual(new int[] { 7, 6 }, view2.ToArray<int>());
        }

        [TestMethod]
        public void Scalar_to_array()
        {
            var a = new UnmanagedStorage(17);
            AssertAreEqual(new int[] { 17 }, a.ToArray<int>());
        }

        [TestMethod]
        public void DimensionalityReduction4D_to_1D()
        {
            var t = new UnmanagedStorage(np.arange(30).GetData(), new Shape(2, 1, 3, 5));
            var view = t.GetView("0,0,:,0");
            Assert.AreEqual(new Shape(3), view.Shape);
            Assert.AreEqual(5, view.GetValue<int>(1));
            Assert.AreEqual(10, view.GetValue<int>(2));
            AssertAreEqual(new int[] { 0, 5, 10 }, view.ToArray<int>());
        }

        [Ignore("This is not implemented yet")]
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
        }



        //[TestMethod]
        //public void ToStringTest()
        //{
        //    var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        //    var view = new UnmanagedStorage(data);
        //    Console.WriteLine(view.ToString(flat: true));
        //    Assert.AreEqual("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", view.ToString(flat: true));
        //    data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
        //    data.Reshape(3, 3);
        //    view = new UnmanagedStorage(data);
        //    Console.WriteLine(view.ToString(flat: true));
        //    Assert.AreEqual("[[0, 1, 2], [3, 4, 5], [6, 7, 8]]", view.ToString(flat: true));
        //    data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });
        //    data.Reshape(2, 2, 2);
        //    view = new UnmanagedStorage(data);
        //    Console.WriteLine(view.ToString(flat: true));
        //    Assert.AreEqual("[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]", view.ToString(flat: true));
        //}

        //        [TestMethod]
        //        public void ToString_NonFlatTest()
        //        {
        //            var data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        //            var view = new UnmanagedStorage(data);
        //            Console.WriteLine(view);
        //            Assert.AreEqual("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", view.ToString(flat: false));
        //            data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
        //            data.Reshape(3, 3);
        //            view = new UnmanagedStorage(data);
        //            Console.WriteLine(view);
        //            Assert.AreEqual("[[0, 1, 2], \r\n" +
        //                            "[3, 4, 5], \r\n" +
        //                            "[6, 7, 8]]", view.ToString(flat: false));
        //            data = new UnmanagedStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });
        //            data.Reshape(2, 2, 2);
        //            view = new UnmanagedStorage(data);
        //            Console.WriteLine(view);
        //            Assert.AreEqual("[[[0, 1], \r\n" +
        //                            "[2, 3]], \r\n" +
        //                            "[[4, 5], \r\n" +
        //                            "[6, 7]]]", view.ToString(flat: false));
        //        }

    }
}


