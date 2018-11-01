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
            var np = new NDArray<int>();
            np.ARange(6).ReShape(3, 2);

            Assert.IsTrue(np[0, 0] == 0);
            Assert.IsTrue(np[1, 1] == 3);
            Assert.IsTrue(np[2, 1] == 5);
            // Assert.IsTrue(np2.ToString().Equals("array([[0, 1], [2, 3], [4, 5]])"));

            np.ARange(6).ReShape(2, 3, 1);
            Assert.IsTrue(np[1, 1, 0] == 4);
            Assert.IsTrue(np[1, 2, 0] == 5);

            np.ARange(12).ReShape(2, 3, 2);
            Assert.IsTrue(np[0, 0, 1] == 1);
            Assert.IsTrue(np[1, 0, 1] == 7);
            Assert.IsTrue(np[1, 1, 0] == 8);

            np.ARange(12).ReShape(3, 4);
            Assert.IsTrue(np[1, 1] == 5);
            Assert.IsTrue(np[2, 0] == 8);
        }

        [TestMethod]
        public void Performance()
        {
            var np = new NDArray<int>();
            np.ARange(1024 * 1024);
        }

        [TestMethod]
        public void PerformaceBitmapSimulation()
        {
            var npRealWorldBitmap = new NDArray<byte>();
            //npRealWorldBitmap.ARange(2081 * 2531);
            npRealWorldBitmap.ReShape(2531, 2081);
        }

        [TestMethod]
        public void ReshapeNegative()
        {
            var np = new NDArray<int>();
            np.ARange(12);
            np.ReShape(-1, 2);
            Assert.IsTrue(np.Shape[0] == 6);
            Assert.IsTrue(np.Shape[1] == 2);

            np.ARange(12);
            np.ReShape(2, -1);
            Assert.IsTrue(np.Shape[0] == 2);
            Assert.IsTrue(np.Shape[1] == 6);

            np.ARange(12);
            np.ReShape(1, 3, 4);
            np.ReShape(-1, 3);
            Assert.IsTrue(np.Shape[0] == 4);
            Assert.IsTrue(np.Shape[1] == 3);

            np.ARange(12);
            np.ReShape(1, 3, 4);
            np.ReShape(3, -1);
            Assert.IsTrue(np.Shape[0] == 3);
            Assert.IsTrue(np.Shape[1] == 4);

            np.ARange(100 * 100 * 3);
            np.ReShape(100, 100, 3);
            np.ReShape(-1, 3);
            Assert.IsTrue(np.Shape[0] == 10000);
            Assert.IsTrue(np.Shape[1] == 3);

            np.ARange(15801033);
            np.ReShape(2531, 2081, 3);
            np.ReShape(-1, 3);
            Assert.IsTrue(np.Shape[0] == 5267011);
            Assert.IsTrue(np.Shape[1] == 3);
        }
    }
}
