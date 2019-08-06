using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArrayAbsoluteTest
    {
        [TestMethod]
        public void absolute()
        {
            //2D
            var n = np.arange(-2, 2).reshape(2, 2);
            var n1 = np.abs(n).MakeGeneric<double>();

            Assert.IsTrue(n1[0, 0] == 2);
            Assert.IsTrue(n1[0, 1] == 1);
            Assert.IsTrue(n1[1, 0] == 0);
            Assert.IsTrue(n1[1, 1] == 1);

            //3D
            n = np.arange(-4, 4).reshape(2, 2, 2);
            n1 = np.abs(n).MakeGeneric<double>();
            Assert.IsTrue(n1[0, 0, 0] == 4);
            Assert.IsTrue(n1[0, 0, 1] == 3);
            Assert.IsTrue(n1[1, 0, 0] == 0);
            Assert.IsTrue(n1[1, 1, 1] == 3);

            //4D
            n = np.arange(-12, 12).reshape(2, 3, 2, 2);
            n1 = np.abs(n).MakeGeneric<double>();
            Assert.IsTrue(n1[0, 0, 0, 0] == 12);
            Assert.IsTrue(n1[0, 1, 0, 0] == 8);
            Assert.IsTrue(n1[1, 2, 1, 1] == 11);
            Assert.IsTrue(n1[1, 2, 0, 1] == 9);
        }
    }
}
