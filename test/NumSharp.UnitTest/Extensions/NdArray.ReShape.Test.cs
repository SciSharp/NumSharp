using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayReShapeTest
    {
        [TestMethod]
        public void ReShape()
        {
            var np = new NumPy<int>();
            var n = np.reshape(np.arange(6), 3, 2);

            Assert.IsTrue(n[0, 0] == 0);
            Assert.IsTrue(n[1, 1] == 3);
            Assert.IsTrue(n[2, 1] == 5);
            // Assert.IsTrue(np2.ToString().Equals("array([[0, 1], [2, 3], [4, 5]])"));

            n = np.reshape(np.arange(6), 2, 3, 1);
            Assert.IsTrue(n[1, 1, 0] == 4);
            Assert.IsTrue(n[1, 2, 0] == 5);

            n = np.reshape(np.arange(12), 2, 3, 2);
            Assert.IsTrue(n[0, 0, 1] == 1);
            Assert.IsTrue(n[1, 0, 1] == 7);
            Assert.IsTrue(n[1, 1, 0] == 8);

            n = np.reshape(np.arange(12), 3, 4);
            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 0] == 8);

            n = np.reshape(n, 2, 6);
            Assert.IsTrue(n[1, 0] == 6);
        }

        [TestMethod]
        public void PerformaceBitmapSimulation()
        {
            var npRealWorldBitmap = new NDArray<byte>();
            //npRealWorldBitmap.ARange(2081 * 2531);
            npRealWorldBitmap.reshape(2531, 2081);
        }

        /// <summary>
        /// numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). 
        /// It simply means that it is an unknown dimension and we want numpy to figure it out. 
        /// And numpy will figure this by looking at the 'length of the array and remaining dimensions' and making sure it satisfies the above mentioned criteria
        /// </summary>
        [TestMethod]
        public void ReshapeNegative()
        {
            var np = new NDArray<int>();
            np.arange(12);
            np.reshape(-1, 2);
            Assert.IsTrue(np.Shape.Shapes[0] == 6);
            Assert.IsTrue(np.Shape.Shapes[1] == 2);

            np.arange(12);
            np.reshape(2, -1);
            Assert.IsTrue(np.Shape.Shapes[0] == 2);
            Assert.IsTrue(np.Shape.Shapes[1] == 6);

            np.arange(12);
            np.reshape(1, 3, 4);
            np.reshape(-1, 3);
            Assert.IsTrue(np.Shape.Shapes[0] == 4);
            Assert.IsTrue(np.Shape.Shapes[1] == 3);

            np.arange(12);
            np.reshape(1, 3, 4);
            np.reshape(3, -1);
            Assert.IsTrue(np.Shape.Shapes[0] == 3);
            Assert.IsTrue(np.Shape.Shapes[1] == 4);

            np.arange(100 * 100 * 3);
            np.reshape(100, 100, 3);
            np.reshape(-1, 3);
            Assert.IsTrue(np.Shape.Shapes[0] == 10000);
            Assert.IsTrue(np.Shape.Shapes[1] == 3);

            np.arange(15801033);
            np.reshape(2531, 2081, 3);
            np.reshape(-1, 3);
            Assert.IsTrue(np.Shape.Shapes[0] == 5267011);
            Assert.IsTrue(np.Shape.Shapes[1] == 3);
        }
    }
}
