using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArrayAMinTest
    {
        [TestMethod]
        public void amin()
        {
            var np = new NumPy<double>();

            //no axis
            var n = np.arange(4).reshape(2, 2);
            var n1 = np.amin(n);
            Assert.IsTrue(n1[0] == 0);

            //2D with axis
            n1 = np.amin(n, 0);
            Assert.IsTrue(n1[0] == 0);
            Assert.IsTrue(n1[1] == 1);
            n1 = np.amin(n, 1);
            Assert.IsTrue(n1[0] == 0);
            Assert.IsTrue(n1[1] == 2);

            //3D
            n = np.arange(24).reshape(4, 3, 2);
            n1 = np.amin(n, 0);
            Assert.IsTrue(n1[0, 1] == 1);
            Assert.IsTrue(n1[2, 1] == 5);
            Assert.IsTrue(n1[1, 1] == 3);
            n1 = np.amin(n, 1);
            Assert.IsTrue(n1[1, 1] == 7);
            Assert.IsTrue(n1[2, 1] == 13);
            Assert.IsTrue(n1[3, 0] == 18);

            //4D
            n = np.arange(24).reshape(2, 3, 2, 2);
            n1 = np.amin(n, 1);
            Assert.IsTrue(n1[0, 0, 1] == 1);
            Assert.IsTrue(n1[1, 0, 1] == 13);
            Assert.IsTrue(n1[1, 1, 1] == 15);
            n1 = np.amin(n, 3);
            Assert.IsTrue(n1[0, 1, 1] == 6);
            Assert.IsTrue(n1[1, 1, 1] == 18);
            Assert.IsTrue(n1[1, 2, 1] == 22);

            var np2 = new NumPyWithDType();
            var nd2 = np2.arange(0, 4, 1, NDArrayWithDType.double8).reshape(2, 2);
            nd2 = nd2.AMin(0);
        }
    }
}
