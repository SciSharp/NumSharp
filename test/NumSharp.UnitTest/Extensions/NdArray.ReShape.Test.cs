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
        }

        [TestMethod]
        public void Performance()
        {
            var np = new NDArray<int>();
            np.ARange(1024 * 1024);
        }
    }
}
