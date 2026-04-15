//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
//using NumSharp.Backends;

//namespace NumSharp.UnitTest.View
//{
//    [TestClass]
//    public class ViewStorageTest : TestClass
//    {
//        [TestMethod]
//        public void GetData_1D()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            Assert.AreEqual(new Shape(10), data.Shape);
//            // return identical view
//            var view = new ViewStorage(data, ":");
//            Assert.AreEqual(new Shape(10), view.Shape);
//            AssertAreEqual(data.GetData(), view.GetData());
//            view = new ViewStorage(data, "-77:77");
//            Assert.AreEqual(new Shape(10), view.Shape);
//            AssertAreEqual(data.GetData(), view.GetData());
//            // return reduced view
//            view = new ViewStorage(data, "7:");
//            Assert.AreEqual(new Shape(3), view.Shape);
//            AssertAreEqual(new int[] { 7, 8, 9 }, view.GetData());
//            view = new ViewStorage(data, ":5");
//            Assert.AreEqual(new Shape(5), view.Shape);
//            AssertAreEqual(new int[] { 0, 1, 2, 3, 4 }, view.GetData());
//            view = new ViewStorage(data, "2:3");
//            Assert.AreEqual(new Shape(1), view.Shape);
//            AssertAreEqual(new int[] { 2 }, view.GetData());
//            // return stepped view
//            view = new ViewStorage(data, "::2");
//            Assert.AreEqual(new Shape(5), view.Shape);
//            AssertAreEqual(new int[] { 0, 2, 4, 6, 8 }, view.GetData());
//            view = new ViewStorage(data, "::3");
//            Assert.AreEqual(new Shape(4), view.Shape);
//            AssertAreEqual(new int[] { 0, 3, 6, 9 }, view.GetData());
//            view = new ViewStorage(data, "-77:77:77");
//            Assert.AreEqual(new Shape(1), view.Shape);
//            AssertAreEqual(new[] { 0 }, view.GetData());
//            // negative step!
//            view = new ViewStorage(data, "::-1");
//            Assert.AreEqual(new Shape(10), view.Shape);
//            AssertAreEqual(data.GetData().OfType<int>().Reverse().ToArray(), view.GetData());
//            view = new ViewStorage(data, "::-2");
//            Assert.AreEqual(new Shape(5), view.Shape);
//            AssertAreEqual(new int[] { 9, 7, 5, 3, 1 }, view.GetData());
//            view = new ViewStorage(data, "::-3");
//            Assert.AreEqual(new Shape(4), view.Shape);
//            AssertAreEqual(new int[] { 9, 6, 3, 0 }, view.GetData());
//            view = new ViewStorage(data, "-77:77:-77");
//            Assert.AreEqual(new Shape(1), view.Shape);
//            AssertAreEqual(new[] { 9 }, view.GetData());
//        }

//        [TestMethod]
//        public void Indexing_1D()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            // return identical view
//            var view = new ViewStorage(data, ":");
//            Assert.AreEqual(0, view.GetData<int>(0));
//            Assert.AreEqual(5, view.GetData<int>(5));
//            Assert.AreEqual(9, view.GetData<int>(9));
//            view = new ViewStorage(data, "-77:77");
//            Assert.AreEqual(0, view.GetData<int>(0));
//            Assert.AreEqual(5, view.GetData<int>(5));
//            Assert.AreEqual(9, view.GetData<int>(9));
//            // return reduced view
//            view = new ViewStorage(data, "7:");
//            Assert.AreEqual(7, view.GetData<int>(0));
//            Assert.AreEqual(8, view.GetData<int>(1));
//            Assert.AreEqual(9, view.GetData<int>(2));
//            view = new ViewStorage(data, ":5");
//            Assert.AreEqual(0, view.GetData<int>(0));
//            Assert.AreEqual(1, view.GetData<int>(1));
//            Assert.AreEqual(2, view.GetData<int>(2));
//            Assert.AreEqual(3, view.GetData<int>(3));
//            Assert.AreEqual(4, view.GetData<int>(4));
//            view = new ViewStorage(data, "2:3");
//            Assert.AreEqual(2, view.GetData<int>(0));
//            // return stepped view
//            view = new ViewStorage(data, "::2");
//            Assert.AreEqual(0, view.GetData<int>(0));
//            Assert.AreEqual(2, view.GetData<int>(1));
//            Assert.AreEqual(4, view.GetData<int>(2));
//            Assert.AreEqual(6, view.GetData<int>(3));
//            Assert.AreEqual(8, view.GetData<int>(4));
//            view = new ViewStorage(data, "::3");
//            Assert.AreEqual(0, view.GetData<int>(0));
//            Assert.AreEqual(3, view.GetData<int>(1));
//            Assert.AreEqual(6, view.GetData<int>(2));
//            Assert.AreEqual(9, view.GetData<int>(3));
//            view = new ViewStorage(data, "-77:77:77");
//            Assert.AreEqual(0, view.GetData<int>(0));
//            // negative step!
//            view = new ViewStorage(data, "::-1");
//            Assert.AreEqual(9, view.GetData<int>(0));
//            Assert.AreEqual(4, view.GetData<int>(5));
//            Assert.AreEqual(0, view.GetData<int>(9));
//            view = new ViewStorage(data, "::-2");
//            Assert.AreEqual(9, view.GetData<int>(0));
//            Assert.AreEqual(7, view.GetData<int>(1));
//            Assert.AreEqual(5, view.GetData<int>(2));
//            Assert.AreEqual(3, view.GetData<int>(3));
//            Assert.AreEqual(1, view.GetData<int>(4));
//            view = new ViewStorage(data, "::-3");
//            Assert.AreEqual(9, view.GetData<int>(0));
//            Assert.AreEqual(6, view.GetData<int>(1));
//            Assert.AreEqual(3, view.GetData<int>(2));
//            Assert.AreEqual(0, view.GetData<int>(3));
//            view = new ViewStorage(data, "-77:77:-77");
//            Assert.AreEqual(9, view.GetData<int>(0));
//        }

//        [TestMethod]
//        public void NestedView_1D()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            // return identical view
//            var identical = new ViewStorage(data, ":");
//            Assert.AreEqual(new Shape(10), identical.Shape);
//            var view1 = new ViewStorage(identical, "1:9");
//            Assert.AreEqual(new Shape(8), view1.Shape);
//            AssertAreEqual(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, }, view1.GetData());
//            var view2 = new ViewStorage(view1, "::-2");
//            Assert.AreEqual(new Shape(4), view2.Shape);
//            AssertAreEqual(new int[] { 8, 6, 4, 2, }, view2.GetData());
//            var view3 = new ViewStorage(view2, "::-3");
//            Assert.AreEqual(new Shape(2), view3.Shape);
//            AssertAreEqual(new int[] { 2, 8 }, view3.GetData());
//            // all must see the same modifications, no matter if original or any view is modified
//            // modify original
//            data.ReplaceData(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 });
//            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, -8, }, view1.GetData());
//            AssertAreEqual(new int[] { -8, -6, -4, -2, }, view2.GetData());
//            AssertAreEqual(new int[] { -2, -8 }, view3.GetData());
//            // modify views
//            view1.SetData(88, 7);
//            AssertAreEqual(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, 88, -9 }, data.GetData());
//            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, 88, }, view1.GetData());
//            AssertAreEqual(new int[] { 88, -6, -4, -2, }, view2.GetData());
//            AssertAreEqual(new int[] { -2, 88 }, view3.GetData());
//            view3.SetData(22, 0);
//            AssertAreEqual(new int[] { 0, -1, 22, -3, -4, -5, -6, -7, 88, -9 }, data.GetData());
//            AssertAreEqual(new int[] { -1, 22, -3, -4, -5, -6, -7, 88, }, view1.GetData());
//            AssertAreEqual(new int[] { 88, -6, -4, 22, }, view2.GetData());
//            AssertAreEqual(new int[] { 22, 88 }, view3.GetData());
//        }

//        [TestMethod]
//        public void GetData_2D()
//        {
//            //>>> x = np.arange(9).reshape(3, 3)
//            //>>> x
//            //array([[0, 1, 2],
//            //       [3, 4, 5],
//            //       [6, 7, 8]])
//            //>>> x[:]
//            //array([[0, 1, 2],
//            //       [3, 4, 5],
//            //       [6, 7, 8]])
//            //>>> x[1:]
//            //array([[3, 4, 5],
//            //       [6, 7, 8]])
//            //>>> x[1:,:]
//            //array([[3, 4, 5],
//            //       [6, 7, 8]])
//            //>>> x[:, 1:]
//            //array([[1, 2],
//            //       [4, 5],
//            //       [7, 8]])
//            //>>> x[1:2, 0:1]
//            //array([[3]])
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
//            data.Reshape(3, 3);
//            Assert.AreEqual(new Shape(3, 3), data.Shape);
//            // return identical view
//            var view = new ViewStorage(data, ":");
//            Assert.AreEqual(new Shape(3, 3), view.Shape);
//            AssertAreEqual(data.GetData(), view.GetData());
//            // return reduced view
//            view = new ViewStorage(data, "1:");
//            Assert.AreEqual(new Shape(2, 3), view.Shape);
//            AssertAreEqual(new int[] { 3, 4, 5, 6, 7, 8 }, view.GetData());
//            view = new ViewStorage(data, "1:,:");
//            Assert.AreEqual(new Shape(2, 3), view.Shape);
//            AssertAreEqual(new int[] { 3, 4, 5, 6, 7, 8 }, view.GetData());
//            view = new ViewStorage(data, ":,1:");
//            Assert.AreEqual(new Shape(3, 2), view.Shape);
//            AssertAreEqual(new int[] { 1, 2, 4, 5, 7, 8 }, view.GetData());
//            view = new ViewStorage(data, "1:2, 0:1");
//            Assert.AreEqual(new Shape(1, 1), view.Shape);
//            AssertAreEqual(new int[] { 3 }, view.GetData());
//            // return stepped view
//            //>>> x
//            //array([[0, 1, 2],
//            //       [3, 4, 5],
//            //       [6, 7, 8]])
//            //>>> x[::2]
//            //array([[0, 1, 2],
//            //       [6, 7, 8]])
//            //>>> x[::3]
//            //array([[0, 1, 2]])
//            //>>> x[::- 1]
//            //array([[6, 7, 8],
//            //       [3, 4, 5],
//            //       [0, 1, 2]])
//            //>>> x[::- 2]
//            //array([[6, 7, 8],
//            //       [0, 1, 2]])
//            //>>> x[::- 3]
//            //array([[6, 7, 8]])
//            view = new ViewStorage(data, "::2");
//            Assert.AreEqual(new Shape(2, 3), view.Shape);
//            AssertAreEqual(new int[] { 0, 1, 2, 6, 7, 8 }, view.GetData());
//            view = new ViewStorage(data, "::3");
//            Assert.AreEqual(new Shape(1, 3), view.Shape);
//            AssertAreEqual(new int[] { 0, 1, 2 }, view.GetData());
//            view = new ViewStorage(data, "::-1");
//            Assert.AreEqual(new Shape(3, 3), view.Shape);
//            AssertAreEqual(new int[] { 6, 7, 8, 3, 4, 5, 0, 1, 2, }, view.GetData());
//            view = new ViewStorage(data, "::-2");
//            Assert.AreEqual(new Shape(2, 3), view.Shape);
//            AssertAreEqual(new int[] { 6, 7, 8, 0, 1, 2, }, view.GetData());
//            view = new ViewStorage(data, "::-3");
//            Assert.AreEqual(new Shape(1, 3), view.Shape);
//            AssertAreEqual(new int[] { 6, 7, 8, }, view.GetData());
//            // N-Dim Stepping
//            //>>> x[::2,::2]
//            //array([[0, 2],
//            //       [6, 8]])
//            //>>> x[::- 1,::- 2]
//            //array([[8, 6],
//            //       [5, 3],
//            //       [2, 0]])
//            view = new ViewStorage(data, "::2, ::2");
//            Assert.AreEqual(new Shape(2, 2), view.Shape);
//            AssertAreEqual(new int[] { 0, 2, 6, 8 }, view.GetData());
//            view = new ViewStorage(data, "::-1, ::-2");
//            Assert.AreEqual(new Shape(3, 2), view.Shape);
//            AssertAreEqual(new int[] { 8, 6, 5, 3, 2, 0 }, view.GetData());
//        }

//        [TestMethod]
//        public void NestedView_2D()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            data.Reshape(2, 10);
//            //>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
//            //>>> x = x.reshape(2, 10)
//            //>>> x
//            //array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
//            //       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
//            // return identical view
//            var identical = new ViewStorage(data, ":");
//            Assert.AreEqual(new Shape(2, 10), identical.Shape);
//            //>>> x[:, 1:9]
//            //array([[1, 2, 3, 4, 5, 6, 7, 8],
//            //       [1, 2, 3, 4, 5, 6, 7, 8]])
//            var view1 = new ViewStorage(identical, ":,1:9");
//            Assert.AreEqual(new Shape(2, 8), view1.Shape);
//            AssertAreEqual(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 }, view1.GetData());
//            //>>> x[:, 1:9][:,::- 2]
//            //array([[8, 6, 4, 2],
//            //       [8, 6, 4, 2]])
//            var view2 = new ViewStorage(view1, ":,::-2");
//            Assert.AreEqual(new Shape(2, 4), view2.Shape);
//            AssertAreEqual(new int[] { 8, 6, 4, 2, 8, 6, 4, 2 }, view2.GetData());
//            //>>> x[:, 1:9][:,::- 2][:,::- 3]
//            //array([[2, 8],
//            //       [2, 8]])
//            var view3 = new ViewStorage(view2, ":,::-3");
//            Assert.AreEqual(new Shape(2, 2), view3.Shape);
//            AssertAreEqual(new int[] { 2, 8, 2, 8 }, view3.GetData());
//            // all must see the same modifications, no matter if original or any view is modified
//            // modify original
//            data.ReplaceData(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9 });
//            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, -8, -1, -2, -3, -4, -5, -6, -7, -8 }, view1.GetData());
//            AssertAreEqual(new int[] { -8, -6, -4, -2, -8, -6, -4, -2 }, view2.GetData());
//            AssertAreEqual(new int[] { -2, -8, -2, -8 }, view3.GetData());
//            // modify views
//            view1.SetData(88, 0, 7);
//            view1.SetData(888, 1, 7);
//            AssertAreEqual(new int[] { 0, -1, -2, -3, -4, -5, -6, -7, 88, -9, 0, -1, -2, -3, -4, -5, -6, -7, 888, -9 },
//                data.GetData());
//            AssertAreEqual(new int[] { -1, -2, -3, -4, -5, -6, -7, 88, -1, -2, -3, -4, -5, -6, -7, 888 },
//                view1.GetData());
//            AssertAreEqual(new int[] { 88, -6, -4, -2, 888, -6, -4, -2 }, view2.GetData());
//            AssertAreEqual(new int[] { -2, 88, -2, 888 }, view3.GetData());
//            view3.SetData(22, 0, 0);
//            view3.SetData(222, 1, 0);
//            AssertAreEqual(new int[] { 0, -1, 22, -3, -4, -5, -6, -7, 88, -9, 0, -1, 222, -3, -4, -5, -6, -7, 888, -9 },
//                data.GetData());
//            AssertAreEqual(new int[] { -1, 22, -3, -4, -5, -6, -7, 88, -1, 222, -3, -4, -5, -6, -7, 888 },
//                view1.GetData());
//            AssertAreEqual(new int[] { 88, -6, -4, 22, 888, -6, -4, 222 }, view2.GetData());
//            AssertAreEqual(new int[] { 22, 88, 222, 888 }, view3.GetData());
//        }

//        [TestMethod]
//        public void Reduce_1D_to_Scalar()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            Assert.AreEqual(new Shape(10), data.Shape);
//            // return scalar
//            var view = new ViewStorage(data, "7");
//            Assert.AreEqual(Shape.Scalar, view.Shape);
//            AssertAreEqual(new int[] { 7 }, view.GetData());
//        }

//        [TestMethod]
//        public void Reduce_2D_to_1D_and_0D()
//        {
//            //>>> x = np.arange(9).reshape(3, 3)
//            //>>> x
//            //array([[0, 1, 2],
//            //       [3, 4, 5],
//            //       [6, 7, 8]])
//            //>>> x[1]
//            //array([3, 4, 5])
//            //>>> x[:,1]
//            //array([1, 4, 7])
//            //>>> x[2, 2]
//            //8
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
//            data.Reshape(3, 3);
//            Assert.AreEqual(new Shape(3, 3), data.Shape);
//            // return identical view
//            var view = new ViewStorage(data, "1");
//            Assert.AreEqual(new Shape(3), view.Shape);
//            AssertAreEqual(new int[] { 3, 4, 5 }, view.GetData());
//            // return reduced view
//            view = new ViewStorage(data, ":,1");
//            Assert.AreEqual(new Shape(3), view.Shape);
//            AssertAreEqual(new int[] { 1, 4, 7 }, view.GetData());
//            view = new ViewStorage(data, "2,2");
//            Assert.AreEqual(Shape.Scalar, view.Shape);
//            AssertAreEqual(new int[] { 8 }, view.GetData());
//            // recursive dimensionality reduction
//            view = new ViewStorage(new ViewStorage(data, "2"), "2");
//            Assert.AreEqual(Shape.Scalar, view.Shape);
//            AssertAreEqual(new int[] { 8 }, view.GetData());
//        }

//        [TestMethod]
//        public void NestedDimensionalityReduction()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
//            data.Reshape(3, 3);
//            Assert.AreEqual(new Shape(3, 3), data.Shape);
//            var view = new ViewStorage(data, "2");
//            Assert.AreEqual(new Shape(3), view.Shape);
//            //AssertAreEqual(new int[] { 6, 7, 8 }, view.GetData());
//            var view1 = new ViewStorage(view, "2");
//            Assert.AreEqual(Shape.Scalar, view1.Shape);
//            AssertAreEqual(new int[] { 8 }, view1.GetData());
//            var view2 = new ViewStorage(view, ":2:-1");
//            Assert.AreEqual(new Shape(2), view2.Shape);
//            AssertAreEqual(new int[] { 7, 6 }, view2.GetData());
//        }

//        [TestMethod]
//        public void ToStringTest()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            var view = new ViewStorage(data);
//            Console.WriteLine(view.ToString(flat: true));
//            Assert.AreEqual("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", view.ToString(flat: true));
//            data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
//            data.Reshape(3, 3);
//            view = new ViewStorage(data);
//            Console.WriteLine(view.ToString(flat: true));
//            Assert.AreEqual("[[0, 1, 2], [3, 4, 5], [6, 7, 8]]", view.ToString(flat: true));
//            data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });
//            data.Reshape(2,2,2);
//            view = new ViewStorage(data);
//            Console.WriteLine(view.ToString(flat: true));
//            Assert.AreEqual("[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]", view.ToString(flat: true));
//        }

//        [TestMethod]
//        public void ToString_NonFlatTest()
//        {
//            var data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//            var view = new ViewStorage(data);
//            Console.WriteLine(view);
//            Assert.AreEqual("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", view.ToString(flat: false));
//            data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
//            data.Reshape(3, 3);
//            view = new ViewStorage(data);
//            Console.WriteLine(view);
//            Assert.AreEqual("[[0, 1, 2], \r\n" +
//                            "[3, 4, 5], \r\n" +
//                            "[6, 7, 8]]", view.ToString(flat: false));
//            data = new TypedArrayStorage(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 });
//            data.Reshape(2, 2, 2);
//            view = new ViewStorage(data);
//            Console.WriteLine(view);
//            Assert.AreEqual("[[[0, 1], \r\n" +
//                            "[2, 3]], \r\n" +
//                            "[[4, 5], \r\n" +
//                            "[6, 7]]]", view.ToString(flat: false));
//        }

//    }
//}


