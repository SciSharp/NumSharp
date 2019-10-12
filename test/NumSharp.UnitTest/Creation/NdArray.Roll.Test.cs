using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NdArrayRollTest
    {
        [Ignore("TODO: fix roll")]
        [TestMethod]
        public void Base1DTest()
        {
            NDArray nd1 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

            var nd2 = nd1.roll(2);

            double[] expNd1 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            double[] expNd2 = new double[] {9, 10, 1, 2, 3, 4, 5, 6, 7, 8};

            ArraySlice<double> nd1_ = nd1.GetData<double>();
            ArraySlice<double> nd2_ = nd2.GetData<double>();

            Assert.IsTrue(Enumerable.SequenceEqual(nd1_, expNd1));
            Assert.IsTrue(Enumerable.SequenceEqual(nd2_, expNd2));

            nd1 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

            nd2 = nd1.roll(-2);

            expNd1 = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            expNd2 = new double[] {3, 4, 5, 6, 7, 8, 9, 10, 1, 2};

            nd1_ = nd1.GetData<double>();
            nd2_ = nd2.GetData<double>(); 

            Assert.IsTrue(Enumerable.SequenceEqual(nd1_, expNd1));
            Assert.IsTrue(Enumerable.SequenceEqual(nd2_, expNd2));
        }

        [Ignore("TODO: fix roll")]
        [TestMethod]
        public void Base2DTest()
        {
            var nd1 = np.arange(10).reshape(2, 5);

            var nd2 = nd1.roll(2);

            Assert.IsTrue(Enumerable.SequenceEqual(nd1.shape, nd2.shape));

            var nd3 = nd1.roll(-2);

            Assert.IsTrue(Enumerable.SequenceEqual(nd1.shape, nd3.shape));
        }

        [TestMethod]
        public void RollWithAxis()
        {
            var x = np.arange(10);
            var x2 = x.reshape(2, 5);

            var x3 = x2.roll(1, 0);
            var x4 = x2.roll(1, 1);

            int[,] x3_ = (Array)x3 as int[,];
            int[,] x4_ = (Array)x4 as int[,];

            int[,] expX3 = {{5, 6, 7, 8, 9}, {0, 1, 2, 3, 4}};
            int[,] expX4 = {{4, 0, 1, 2, 3}, {9, 5, 6, 7, 8}};

            for (int idx = 0; idx < expX3.GetLength(0); idx++)
            for (int jdx = 0; jdx < expX3.GetLength(1); jdx++)
            {
                Assert.IsTrue(x3_[idx, jdx] == expX3[idx, jdx]);
                Assert.IsTrue(x4_[idx, jdx] == expX4[idx, jdx]);
            }
        }
    }
}
