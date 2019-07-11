using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayReShapeTest
    {
        [TestMethod]
        public void ReShape()
        {
            var nd = np.arange(6);
            var n1 = np.reshape(nd, 3, 2);
            var n = n1.MakeGeneric<int>();

            Assert.IsTrue(n[0, 0] == 0);
            Assert.IsTrue(n[1, 1] == 3);
            Assert.IsTrue(n[2, 1] == 5);

            n = np.reshape(np.arange(6), 2, 3, 1).MakeGeneric<int>();
            Assert.IsTrue(n[1, 1, 0] == 4);
            Assert.IsTrue(n[1, 2, 0] == 5);

            n = np.reshape(np.arange(12), 2, 3, 2).MakeGeneric<int>();
            Assert.IsTrue(n[0, 0, 1] == 1);
            Assert.IsTrue(n[1, 0, 1] == 7);
            Assert.IsTrue(n[1, 1, 0] == 8);

            n = np.reshape(np.arange(12), 3, 4).MakeGeneric<int>();
            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 0] == 8);

            n = np.reshape(n, 2, 6).MakeGeneric<int>();

            Assert.IsTrue(n[1, 0] == 6);
        }

        [TestMethod]
        public void PerformaceBitmapSimulation()
        {
            var npRealWorldBitmap = new NDArray(typeof(byte), new Shape(2531, 2081));

            Assert.IsTrue(npRealWorldBitmap.Data<byte>().Count == (2531 * 2081));
        }

        /// <summary>
        /// numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). 
        /// It simply means that it is an unknown dimension and we want numpy to figure it out. 
        /// And numpy will figure this by looking at the 'length of the array and remaining dimensions' and making sure it satisfies the above mentioned criteria
        /// </summary>
        [TestMethod]
        public void ReshapeNegative()
        {
            NDArray<int> nd;
            nd = np.arange(12).MakeGeneric<int>();
            nd.reshape(-1, 2);
            Assert.IsTrue(nd.shape[0] == 6);
            Assert.IsTrue(nd.shape[1] == 2);

            nd = np.arange(12).MakeGeneric<int>();
            ;
            nd.reshape(2, -1);
            Assert.IsTrue(nd.shape[0] == 2);
            Assert.IsTrue(nd.shape[1] == 6);

            nd = np.arange(12).MakeGeneric<int>();
            ;
            nd.reshape(1, 3, 4);
            nd.reshape(-1, 3);
            Assert.IsTrue(nd.shape[0] == 4);
            Assert.IsTrue(nd.shape[1] == 3);

            nd = np.arange(12).MakeGeneric<int>();
            ;
            nd.reshape(1, 3, 4);
            nd.reshape(3, -1);
            Assert.IsTrue(nd.shape[0] == 3);
            Assert.IsTrue(nd.shape[1] == 4);

            nd = np.arange(100 * 100 * 3).MakeGeneric<int>();
            ;
            nd.reshape(100, 100, 3);
            nd.reshape(-1, 3);
            Assert.IsTrue(nd.shape[0] == 10000);
            Assert.IsTrue(nd.shape[1] == 3);

            /*np.arange(15801033);
            np.reshape(2531, 2081, 3);
            np.reshape(-1, 3);
            Assert.IsTrue(np.shape[0] == 5267011);
            Assert.IsTrue(np.shape[1] == 3);*/
        }

        [TestMethod]
        public void ValueTest()
        {
            var x = np.arange(4).MakeGeneric<int>();
            var y = x.reshape(2, 2).MakeGeneric<int>();
            y[0, 1] = 8;
            Assert.AreEqual(x[1], y[0, 1]);
        }
    }
}
