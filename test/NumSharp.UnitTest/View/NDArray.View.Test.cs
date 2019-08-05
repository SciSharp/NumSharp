using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.View
{
    [TestClass]
    public class NDArrayViewTest : TestClass
    {
        [TestMethod]
        public void ValueTest()
        {
            var x = np.arange(3);
            var v = x.view();

            v[0] = 1;

            Assert.IsTrue((int)x[0] == (int)v[0]);
        }

        [TestMethod]
        public void GetData_1D()
        {
            // these slicing operations should be executed with Span<T> internally
            var data = np.arange(10);
            Assert.AreEqual(new Shape(10), new Shape(data.shape));
            // return identical view
            var view = data[":"];
            Assert.AreEqual(new Shape(10), new Shape(view.shape));
            AssertAreEqual(data.ToArray<int>(), view.ToArray<int>());
            view = data["-77:77"];
            Assert.AreEqual(new Shape(10), new Shape(view.shape));
            AssertAreEqual(data.ToArray<int>(), view.ToArray<int>());
            // return reduced view
            view = data["7:"];
            Assert.AreEqual(new Shape(3), new Shape(view.shape));
            AssertAreEqual(new int[] {7, 8, 9}, view.ToArray<int>());
            view = data[":5"];
            Assert.AreEqual(new Shape(5), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 1, 2, 3, 4}, view.ToArray<int>());
            view = data["2:3"];
            Assert.AreEqual(new Shape(1), new Shape(view.shape));
            AssertAreEqual(new int[] {2}, view.ToArray<int>());
        }

        [TestMethod]
        public void GetData_1D_Stepping()
        {
            // these slicing operations should be executed with Stepping internally
            var data = np.arange(10);
            // return stepped view
            var view = data["::2"];
            Assert.AreEqual(new Shape(5), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 2, 4, 6, 8}, view.ToArray<int>());
            view = data["::3"];
            Assert.AreEqual(new Shape(4), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 3, 6, 9}, view.ToArray<int>());
            view = data["-77:77:77"];
            Assert.AreEqual(new Shape(1), new Shape(view.shape));
            AssertAreEqual(new[] {0}, view.ToArray<int>());
            // negative step!
            view = data["::-1"];
            Assert.AreEqual(new Shape(10), new Shape(view.shape));
            AssertAreEqual(data.ToArray<int>().OfType<int>().Reverse().ToArray(), view.ToArray<int>());
            view = data["::-2"];
            Assert.AreEqual(new Shape(5), new Shape(view.shape));
            AssertAreEqual(new int[] {9, 7, 5, 3, 1}, view.ToArray<int>());
            view = data["::-3"];
            Assert.AreEqual(new Shape(4), new Shape(view.shape));
            AssertAreEqual(new int[] {9, 6, 3, 0}, view.ToArray<int>());
            view = data["77:-77:-77"];
            Assert.AreEqual(new Shape(1), new Shape(view.shape));
            AssertAreEqual(new[] {9}, view.ToArray<int>());
        }

        [TestMethod]
        public void Indexing_1D()
        {
            var data = np.arange(10);
            // return identical view
            var view = data[":"];
            Assert.AreEqual(0, (int)view[0]);
            Assert.AreEqual(5, (int)data[5]);
            Assert.AreEqual(9, (int)data[9]);
            var nd5 = view[5];
            Assert.AreEqual(5, (int)nd5);
            Assert.AreEqual(9, (int)view[9]);
            view = data["-77:77"];
            Assert.AreEqual(0, (int)view[0]);
            Assert.AreEqual(5, (int)view[5]);
            Assert.AreEqual(9, (int)view[9]);
            // return reduced view
            view = data["7:"];
            Assert.AreEqual(7, (int)view[0]);
            Assert.AreEqual(8, (int)view[1]);
            Assert.AreEqual(9, (int)view[2]);
            view = data[":5"];
            Assert.AreEqual(0, (int)view[0]);
            Assert.AreEqual(1, (int)view[1]);
            Assert.AreEqual(2, (int)view[2]);
            Assert.AreEqual(3, (int)view[3]);
            Assert.AreEqual(4, (int)view[4]);
            view = data["2:3"];
            Assert.AreEqual(2, (int)view[0]);
        }

        [TestMethod]
        public void Indexing_1D_Stepping()
        {
            var data = np.arange(10);
            // return stepped view
            var view = data["::2"];
            //Assert.AreEqual(0, (int)view[0]);
            Assert.AreEqual(2, (int)view[1]);
            Assert.AreEqual(4, (int)view[2]);
            Assert.AreEqual(6, (int)view[3]);
            Assert.AreEqual(8, (int)view[4]);
            view = data["::3"];
            Assert.AreEqual(0, (int)view[0]);
            Assert.AreEqual(3, (int)view[1]);
            Assert.AreEqual(6, (int)view[2]);
            Assert.AreEqual(9, (int)view[3]);
            view = data["-77:77:77"];
            Assert.AreEqual(0, (int)view[0]);
            // negative step!
            view = data["::-1"];
            Assert.AreEqual(9, (int)view[0]);
            Assert.AreEqual(4, (int)view[5]);
            Assert.AreEqual(0, (int)view[9]);
            view = data["::-2"];
            Assert.AreEqual(9, (int)view[0]);
            Assert.AreEqual(7, (int)view[1]);
            Assert.AreEqual(5, (int)view[2]);
            Assert.AreEqual(3, (int)view[3]);
            Assert.AreEqual(1, (int)view[4]);
            view = data["::-3"];
            Assert.AreEqual(9, (int)view[0]);
            Assert.AreEqual(6, (int)view[1]);
            Assert.AreEqual(3, (int)view[2]);
            Assert.AreEqual(0, (int)view[3]);
            view = data["77:-77:-77"];
            Assert.AreEqual(9, (int)view[0]);
        }

        [TestMethod]
        public void Shared_Data_1D()
        {
            var data = np.arange(10);
            // return identical view
            var view = data[":"];
            Assert.AreEqual(new Shape(10), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, view.ToArray<int>());
            data[9] = 99;
            Assert.AreEqual(99, (int)view[9]);
            view[1] = 11;
            Assert.AreEqual(11, (int)data[1]);
            data.SetData(new int[] {0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
            Assert.AreEqual(-1, (int)data[1]);
            Assert.AreEqual(-1, (int)view[1]);
        }

        [TestMethod]
        public void NestedView_1D()
        {
            var data = np.arange(10);
            // return identical view
            var identical = data[":"];
            Assert.AreEqual(new Shape(10), new Shape(identical.shape));
            var view1 = identical["1:9"];
            Assert.AreEqual(new Shape(8), new Shape(view1.shape));
            AssertAreEqual(new int[] {1, 2, 3, 4, 5, 6, 7, 8,}, view1.ToArray<int>());
            var view2 = view1["4:"];
            Assert.AreEqual(new Shape(4), new Shape(view2.shape));
            AssertAreEqual(new int[] {5, 6, 7, 8,}, view2.ToArray<int>());
            var view3 = view2[":2"];
            Assert.AreEqual(new Shape(2), new Shape(view3.shape));
            AssertAreEqual(new int[] {5, 6}, view3.ToArray<int>());
            // all must see the same modifications, no matter if original or any view is modified
            // modify original
            data.SetData(new int[] {0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
            AssertAreEqual(new int[] {-1, -2, -3, -4, -5, -6, -7, -8,}, view1.ToArray<int>());
            AssertAreEqual(new int[] {-5, -6, -7, -8,}, view2.ToArray<int>());
            AssertAreEqual(new int[] {-5, -6}, view3.ToArray<int>());
            // modify views
            view1.SetValue(55, 4);
            AssertAreEqual(new int[] {0, -1, -2, -3, -4, 55, -6, -7, -8, -9}, data.ToArray<int>());
            AssertAreEqual(new int[] {-1, -2, -3, -4, 55, -6, -7, -8,}, view1.ToArray<int>());
            AssertAreEqual(new int[] {55, -6, -7, -8,}, view2.ToArray<int>());
            AssertAreEqual(new int[] {55, -6}, view3.ToArray<int>());
            view3.SetValue(66, 1);
            AssertAreEqual(new int[] {0, -1, -2, -3, -4, 55, 66, -7, -8, -9}, data.ToArray<int>());
            AssertAreEqual(new int[] {-1, -2, -3, -4, 55, 66, -7, -8,}, view1.ToArray<int>());
            AssertAreEqual(new int[] {55, 66, -7, -8,}, view2.ToArray<int>());
            AssertAreEqual(new int[] {55, 66,}, view3.ToArray<int>());
        }

        [TestMethod]
        public void NestedView_1D_Stepping()
        {
            var data = np.arange(10);
            // return identical view
            var identical = data[":"];
            Assert.AreEqual(new Shape(10), new Shape(identical.shape));
            var view1 = identical["1:9"];
            Assert.AreEqual(new Shape(8), new Shape(view1.shape));
            AssertAreEqual(new int[] {1, 2, 3, 4, 5, 6, 7, 8,}, view1.ToArray<int>());
            var view2 = view1["::-2"];
            Assert.AreEqual(new Shape(4), new Shape(view2.shape));
            AssertAreEqual(new int[] {8, 6, 4, 2,}, view2.ToArray<int>());
            var view3 = view2["::-3"];
            Assert.AreEqual(new Shape(2), new Shape(view3.shape));
            AssertAreEqual(new int[] {2, 8}, view3.ToArray<int>());
            // all must see the same modifications, no matter if original or any view is modified
            // modify original
            data.SetData(new int[] {0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
            AssertAreEqual(new int[] {-1, -2, -3, -4, -5, -6, -7, -8,}, view1.ToArray<int>());
            AssertAreEqual(new int[] {-8, -6, -4, -2,}, view2.ToArray<int>());
            AssertAreEqual(new int[] {-2, -8}, view3.ToArray<int>());
            // modify views
            view1.SetValue(88, 7);
            AssertAreEqual(new int[] {0, -1, -2, -3, -4, -5, -6, -7, 88, -9}, data.ToArray<int>());
            AssertAreEqual(new int[] {-1, -2, -3, -4, -5, -6, -7, 88,}, view1.ToArray<int>());
            AssertAreEqual(new int[] {88, -6, -4, -2,}, view2.ToArray<int>());
            AssertAreEqual(new int[] {-2, 88}, view3.ToArray<int>());
            view3.SetValue(22, 0);
            AssertAreEqual(new int[] {0, -1, 22, -3, -4, -5, -6, -7, 88, -9}, data.ToArray<int>());
            AssertAreEqual(new int[] {-1, 22, -3, -4, -5, -6, -7, 88,}, view1.ToArray<int>());
            AssertAreEqual(new int[] {88, -6, -4, 22,}, view2.ToArray<int>());
            AssertAreEqual(new int[] {22, 88}, view3.ToArray<int>());
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
            var data = np.arange(9).reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(data.shape));
            // return identical view
            var view = data[":"];
            Assert.AreEqual(new Shape(3, 3), new Shape(view.shape));
            AssertAreEqual(data.ToArray<int>(), view.ToArray<int>());
            // return reduced view
            view = data["1:"];
            Assert.AreEqual(new Shape(2, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {3, 4, 5, 6, 7, 8}, view.ToArray<int>());
            view = data["1:,:"];
            Assert.AreEqual(new Shape(2, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {3, 4, 5, 6, 7, 8}, view.ToArray<int>());
        }

        [TestMethod]
        public void GetData_2D_Stepped()
        {
            var data = np.arange(9).reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(data.shape));
            var view = data[":,1:"];
            Assert.AreEqual(new Shape(3, 2), new Shape(view.shape));
            AssertAreEqual(new int[] {1, 2, 4, 5, 7, 8}, view.ToArray<int>());
            view = data["1:2, 0:1"];
            Assert.AreEqual(new Shape(1, 1), new Shape(view.shape));
            AssertAreEqual(new int[] {3}, view.ToArray<int>());
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
            view = data["::2"];
            Assert.AreEqual(new Shape(2, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 1, 2, 6, 7, 8}, view.ToArray<int>());
            view = data["::3"];
            Assert.AreEqual(new Shape(1, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 1, 2}, view.ToArray<int>());
            view = data["::-1"];
            Assert.AreEqual(new Shape(3, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {6, 7, 8, 3, 4, 5, 0, 1, 2,}, view.ToArray<int>());
            view = data["::-2"];
            Assert.AreEqual(new Shape(2, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {6, 7, 8, 0, 1, 2,}, view.ToArray<int>());
            view = data["::-3"];
            Assert.AreEqual(new Shape(1, 3), new Shape(view.shape));
            AssertAreEqual(new int[] {6, 7, 8,}, view.ToArray<int>());
            // N-Dim Stepping
            //>>> x[::2,::2]
            //array([[0, 2],
            //       [6, 8]])
            //>>> x[::- 1,::- 2]
            //array([[8, 6],
            //       [5, 3],
            //       [2, 0]])
            view = data["::2, ::2"];
            Assert.AreEqual(new Shape(2, 2), new Shape(view.shape));
            AssertAreEqual(new int[] {0, 2, 6, 8}, view.ToArray<int>());
            view = data["::-1, ::-2"];
            Assert.AreEqual(new Shape(3, 2), new Shape(view.shape));
            AssertAreEqual(new int[] {8, 6, 5, 3, 2, 0}, view.ToArray<int>());
        }

        [TestMethod]
        public void NestedView_2D()
        {
            var data = np.array(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
            data = data.reshape(2, 10);
            //>>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            //>>> x = x.reshape(2, 10)
            //>>> x
            //array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            //       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
            // return identical view
            var identical = data[":"];
            Assert.AreEqual(new Shape(2, 10), new Shape(identical.shape));
            //>>> x[:, 1:9]
            //array([[1, 2, 3, 4, 5, 6, 7, 8],
            //       [1, 2, 3, 4, 5, 6, 7, 8]])
            var view1 = identical[":,1:9"];
            Assert.AreEqual(new Shape(2, 8), new Shape(view1.shape));
            AssertAreEqual(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}, view1.ToArray<int>());
            //>>> x[:, 1:9][:,::- 2]
            //array([[8, 6, 4, 2],
            //       [8, 6, 4, 2]])
            var view2 = view1[":,::-2"];
            Assert.AreEqual(new Shape(2, 4), new Shape(view2.shape));
            AssertAreEqual(new int[] {8, 6, 4, 2, 8, 6, 4, 2}, view2.ToArray<int>());
            //>>> x[:, 1:9][:,::- 2][:,::- 3]
            //array([[2, 8],
            //       [2, 8]])
            var view3 = view2[":,::-3"];
            Assert.AreEqual(new Shape(2, 2), new Shape(view3.shape));
            AssertAreEqual(new int[] {2, 8, 2, 8}, view3.ToArray<int>());
            // all must see the same modifications, no matter if original or any view is modified
            // modify original
            data.SetData(new int[,] {{0, -1, -2, -3, -4, -5, -6, -7, -8, -9}, {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}});
            AssertAreEqual(new int[] {-1, -2, -3, -4, -5, -6, -7, -8, -1, -2, -3, -4, -5, -6, -7, -8}, view1.ToArray<int>());
            AssertAreEqual(new int[] {-8, -6, -4, -2, -8, -6, -4, -2}, view2.ToArray<int>());
            AssertAreEqual(new int[] {-2, -8, -2, -8}, view3.ToArray<int>());
            // modify views
            view1.SetValue(88, 0, 7);
            view1.SetValue(888, 1, 7);
            AssertAreEqual(new int[] {0, -1, -2, -3, -4, -5, -6, -7, 88, -9, 0, -1, -2, -3, -4, -5, -6, -7, 888, -9},
                data.ToArray<int>());
            AssertAreEqual(new int[] {-1, -2, -3, -4, -5, -6, -7, 88, -1, -2, -3, -4, -5, -6, -7, 888},
                view1.ToArray<int>());
            AssertAreEqual(new int[] {88, -6, -4, -2, 888, -6, -4, -2}, view2.ToArray<int>());
            AssertAreEqual(new int[] {-2, 88, -2, 888}, view3.ToArray<int>());
            view3.SetValue(22, 0, 0);
            view3.SetValue(222, 1, 0);
            AssertAreEqual(new int[] {0, -1, 22, -3, -4, -5, -6, -7, 88, -9, 0, -1, 222, -3, -4, -5, -6, -7, 888, -9},
                data.ToArray<int>());
            AssertAreEqual(new int[] {-1, 22, -3, -4, -5, -6, -7, 88, -1, 222, -3, -4, -5, -6, -7, 888},
                view1.ToArray<int>());
            AssertAreEqual(new int[] {88, -6, -4, 22, 888, -6, -4, 222}, view2.ToArray<int>());
            AssertAreEqual(new int[] {22, 88, 222, 888}, view3.ToArray<int>());
        }

        [TestMethod]
        public void Reduce_1D_to_Scalar()
        {
            var data = np.arange(10);
            Assert.AreEqual(new Shape(10), new Shape(data.shape));
            // return scalar
            var view = data["7"];
            Assert.AreEqual(Shape.Scalar, new Shape(view.shape));
            AssertAreEqual(new int[] {7}, view.ToArray<int>());
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
            //>>> x[2, 2]
            //8
            var data = np.arange(9).reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(data.shape));
            // return identical view
            var view = data["1"];
            Assert.AreEqual(new Shape(3), new Shape(view.shape));
            AssertAreEqual(new int[] {3, 4, 5}, view.ToArray<int>());
            // return reduced view
            view = data["2,2"];
            Assert.AreEqual(Shape.Scalar, new Shape(view.shape));
            AssertAreEqual(new int[] {8}, view.ToArray<int>());
            // recursive dimensionality reduction
            view = data["2"]["2"];
            Assert.AreEqual(Shape.Scalar, new Shape(view.shape));
            AssertAreEqual(new int[] {8}, view.ToArray<int>());
        }

        [TestMethod]
        public void Reduce_2D_to_1D_and_0D_Stepping()
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
            var data = np.arange(9).reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(data.shape));
            // return identical view
            var view = data["1"];
            Assert.AreEqual(new Shape(3), new Shape(view.shape));
            AssertAreEqual(new int[] {3, 4, 5}, view.ToArray<int>());
            // return reduced view
            view = data[":,1"];
            Assert.AreEqual(new Shape(3), new Shape(view.shape));
            AssertAreEqual(new int[] {1, 4, 7}, view.ToArray<int>());
            view = data["2,2"];
            Assert.AreEqual(Shape.Scalar, new Shape(view.shape));
            AssertAreEqual(new int[] {8}, view.ToArray<int>());
            // recursive dimensionality reduction
            view = data["2"]["2"];
            Assert.AreEqual(Shape.Scalar, new Shape(view.shape));
            AssertAreEqual(new int[] {8}, view.ToArray<int>());
        }

        [TestMethod]
        public void DimensionalityReduction4D_to_1D()
        {
            var t = np.arange(15).reshape(1, 1, 3, 5);
            var view = t[Slice.Index(0), Slice.Index(0), Slice.All, Slice.Index(0)];

            Assert.AreEqual(new Shape(3), view.Shape);
            AssertAreEqual(new int[] { 0, 5, 10 }, view.ToArray<int>());
            t = np.arange(30).reshape(2, 1, 3, 5);
            view = t[Slice.Index(0), Slice.Index(0), Slice.All, Slice.Index(0)];
            Assert.AreEqual(new Shape(3), view.Shape);
            AssertAreEqual(new int[] { 0, 5, 10 }, view.ToArray<int>());
        }

        [TestMethod]
        public void NestedDimensionalityReduction()
        {
            var data = np.arange(9).reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(data.shape));
            var view = data["2"];
            Assert.AreEqual(new Shape(3), new Shape(view.shape));
            AssertAreEqual(new int[] {6, 7, 8}, view.ToArray<int>());
            var view1 = view["2"];
            Assert.AreEqual(Shape.Scalar, new Shape(view1.shape));
            AssertAreEqual(new int[] {8}, view1.ToArray<int>());
            var view2 = view[":2:1"];
            Assert.AreEqual(new Shape(2), new Shape(view2.shape));
            AssertAreEqual(new int[] {6, 7}, view2.ToArray<int>());
        }

        [TestMethod]
        public void NestedDimensionalityReduction_Stepped()
        {
            var data = np.arange(9).reshape(3, 3);
            Assert.AreEqual(new Shape(3, 3), new Shape(data.shape));
            var view = data["2"];
            Assert.AreEqual(new Shape(3), new Shape(view.shape));
            AssertAreEqual(new int[] {6, 7, 8}, view.ToArray<int>());
            var view1 = view["2"];
            Assert.AreEqual(Shape.Scalar, new Shape(view1.shape));
            AssertAreEqual(new int[] {8}, view1.ToArray<int>());
            var view2 = view["1::-1"];
            Assert.AreEqual(new Shape(2), new Shape(view2.shape));
            AssertAreEqual(new int[] {7, 6}, view2.ToArray<int>());
        }

        [TestMethod]
        public void SlicingToScalar()
        {
            //numpy code:
            //lhs = np.full((6, 3, 3), 5, np.int32)
            //lhs = lhs[::2,:,:]
            //slice = lhs[1, 1, 2]
            //print(slice)
            //print(slice.shape)
            //print(slice.ndim == 0)

            //outputs:
            //5
            //()
            //True

            var lhs = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            lhs = lhs["::2,:,:"];
            var slice = lhs.Storage.GetData(1, 1, 2);
            slice.Count.Should().Be(1);
            slice.Shape.IsScalar.Should().BeTrue();
            slice.Shape.IsSliced.Should().BeTrue("Slicing should occurs only when lhs is already sliced.");
        }

        [TestMethod]
        public unsafe void SliceSelectsAll()
        {
            var lhs = np.full(5, (6, 3, 3), NPTypeCode.Int32);
            var sliced = lhs[":"];

            (lhs.Storage.Address == sliced.Storage.Address).Should().BeTrue("When slice selects all values, it shouldn't return a view but a new wrapper for Storage");
            sliced.Should().NotBeSliced();
            lhs.Should().Be(sliced);
        }

        [TestMethod]
        public void Multiply_2DSlice_By_1D()
        {
            /*
                # Values generated by NumSharp.UnitTest\Utilities\ndarray-generator.py
                a = np.arange(1, 81)
                a = a.reshape(5, 4, 4)
                b = a[:, :, 0]
                cSharp.asCode2D("b54", b)
                c = np.arange(1, 5)
                ret = b * c;
                cSharp.asCode2D("ret54", ret)
            */
            NDArray b54 = new NDArray(new Int32[] {
                1, 5, 9, 13,
                17, 21, 25, 29,
                33, 37, 41, 45,
                49, 53, 57, 61,
                65, 69, 73, 77
            }, new Shape(new int[] { 5, 4 }));
            NDArray ret54 = new NDArray(new Int32[] {
                1, 10, 27, 52,
                17, 42, 75, 116,
                33, 74, 123, 180,
                49, 106, 171, 244,
                65, 138, 219, 308
            }, new Shape(new int[] { 5, 4 }));
            var a = np.arange(1, 81);
            a = a.reshape(5, 4, 4);
            var b = a[":, :, 0"];
            Assert.AreEqual(b, b54);
            var c = np.arange(1, 5);
            var ret = b * c;
            Assert.AreEqual(ret54, ret);
        }

        [TestMethod]
        public void AllSlicesAreIndexes()
        {
            var a = np.arange(27).reshape(3, 3, 3);
            var ret = a[Slice.Index(0), Slice.Index(0), Slice.Index(0)];
            ret.Should().NotBeSliced().And.BeScalar(value: 0);
            a[Slice.Index(2), Slice.Index(2), Slice.Index(2)].Should().NotBeSliced().And.BeScalar(value: 26);

            ret = a[Slice.Index(0), Slice.Index(1)];
            ret.Should().NotBeSliced().And.BeShaped(3).And.BeOfValues(3,4,5);

            ret = a[Slice.Index(0), Slice.Index(1), Slice.All];
            ret.Should().BeSliced();
        }

        [TestMethod]
        public void SlicingWithNegativeIndex()
        {
            var a = np.arange(3 * 1 * 3 * 3).reshape((3, 1, 3, 3));

            var b = a["-1, :, 1, :"];
            b.ToString(flat: true).Should().Be("array([[21, 22, 23]])");
            //or b = a[Slice.Index(-1), Slice.All, Slice.Index(1), Slice.All];
            b.GetValue(0, 0).Should().Be(21); //offset computed inside is -6

            b.Should().BeShaped(1, 3).And.BeOfValues(21, 22, 23);
        }
    }
}
