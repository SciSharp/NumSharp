using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class IndexingTest
    {
        [TestMethod]
        public void IndexAccessorGetter()
        {
            var nd = np.arange(12).reshape(3, 4);

            Assert.IsTrue(nd.Data<int>(1, 1) == 5);
            Assert.IsTrue(nd.Data<int>(2, 0) == 8);
        }

        [TestMethod]
        public void NDArrayAccess()
        {
            var nd = np.arange(4).reshape(2, 2);

            var row1 = (nd[0] as NDArray).MakeGeneric<int>();
            Assert.AreEqual(row1[0], 0);
            Assert.AreEqual(row1[1], 1);
        }

        [TestMethod]
        public void NDArrayAccess3Dim()
        {
            NDArray nd = np.arange(1, 18, 1).reshape(3,3,2);
            var row1 = (nd[0] as NDArray).MakeGeneric<int>();
            Assert.AreEqual(row1[0,0], 1);
            Assert.AreEqual(row1[0,1], 2);
            Assert.AreEqual(row1[1,0], 3);
            Assert.AreEqual(row1[1,1], 4);
            Assert.AreEqual(row1[2,0], 5);
            Assert.AreEqual(row1[2,1], 6);


        }

        [TestMethod]
        public void IndexAccessorSetter()
        {
            var nd = np.arange(12).reshape(3, 4);

            Assert.IsTrue(nd.Data<int>(0, 3) == 3);
            Assert.IsTrue(nd.Data<int>(1, 3) == 7);

            // set value
            nd.SetData(10, 0, 0);
            Assert.IsTrue(nd.Data<int>(0, 0) == 10);
            Assert.IsTrue(nd.Data<int>(1, 3) == 7);
        }
        [TestMethod]
        public void BoolArray()
        {
            NDArray A = new double[] {1,2,3};

            NDArray booleanArr = new bool[] { false, false, true };

            A[booleanArr.MakeGeneric<bool>()] = 1;

            Assert.IsTrue(System.Linq.Enumerable.SequenceEqual(A.Data<double>(), new double[] { 1, 2, 1 }));

            A = new double[,] {{1,2,3},{4,5,6}};

            booleanArr = new bool[,] { { true, false, true }, { false, true, false } };

            A[booleanArr.MakeGeneric<bool>()] = -2;

            Assert.IsTrue( System.Linq.Enumerable.SequenceEqual(A.Data<double>(),new double[] {-2,2,-2,4, -2,6} ));

        }
        [TestMethod]
        public void Compare()
        {
            NDArray A = new double[,] {{1,2,3},{4,5,6}};

            var boolArr = A < 3;
            Assert.IsTrue(Enumerable.SequenceEqual(boolArr.Data<bool>(), new [] { true, true, false, false, false, false }));

            A[A < 3] = -2;
            Assert.IsTrue(Enumerable.SequenceEqual(A.Data<double>(), new double[] { -2, -2, 3, 4, 5, 6 }));

            var a = A[A == -2 | A > 5];

            Assert.IsTrue(Enumerable.SequenceEqual(a.Data<double>(), new double[] { -2, -2, 6 }));
        }

        [TestMethod]
        public void NDArrayByNDArray()
        {
            NDArray x = new double[] {1,2,3,4,5,6};

            NDArray index = new int[] {1,3,5};

            NDArray selected = x[index];

            double[] a = (System.Array) selected as double[];
            double[] b = {2,4,6};

            Assert.IsTrue(Enumerable.SequenceEqual(a,b)); 
        }

        [TestMethod]
        public void Filter1D()
        {
            var nd = np.array(new int[] { 3, 1, 1, 2, 3, 1 });
            var filter = np.array(new int[] { 0, 2, 5 });
            var result = nd[filter];

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 1, 1 }, result.Data<int>()));
        }

        [TestMethod]
        public void Filter2D()
        {
            var nd = np.array(new int[][]
            {
                new int[]{ 3, 1, 1, 2},
                new int[]{ 1, 2, 2, 3},
                new int[]{ 2, 1, 1, 3},
            });
            var filter = np.array(new int[] { 0, 2});
            var result = nd[filter];

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 1, 1, 2 }, (result[0] as NDArray).Data<int>()));
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 2, 1, 1, 3 }, (result[1] as NDArray).Data<int>()));
        }
    }
}
