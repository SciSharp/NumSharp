using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class NDArrayAMinTest
    {
        [TestMethod]
        public void amin()
        {
            //no axis
            var n = np.arange(4).reshape(2, 2);
            var n1 = np.amin(n).MakeGeneric<int>();
            Assert.IsTrue(n1[0] == 0);

            //2D with axis
            n1 = np.amin(n, 0).MakeGeneric<int>();
            Assert.IsTrue(n1[0] == 0);
            Assert.IsTrue(n1[1] == 1);
            n1 = np.amin(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[0] == 0);
            Assert.IsTrue(n1[1] == 2);

            //3D
            n = np.arange(24).reshape(4, 3, 2);
            n1 = np.amin(n, 0).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 1] == 1);
            Assert.IsTrue(n1[2, 1] == 5);
            Assert.IsTrue(n1[1, 1] == 3);
            n1 = np.amin(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[1, 1] == 7);
            Assert.IsTrue(n1[2, 1] == 13);
            Assert.IsTrue(n1[3, 0] == 18);

            //4D
            n = np.arange(24).reshape(2, 3, 2, 2);
            n1 = np.amin(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 0, 1] == 1);
            Assert.IsTrue(n1[1, 0, 1] == 13);
            Assert.IsTrue(n1[1, 1, 1] == 15);
            n1 = np.amin(n, 3).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 1, 1] == 6);
            Assert.IsTrue(n1[1, 1, 1] == 18);
            Assert.IsTrue(n1[1, 2, 1] == 22);
        }
    }
}
