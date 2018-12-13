using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core;
using np = NumSharp.Core.NumPy;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArrayAMinTest
    {
        [TestMethod]
        public void amin()
        {
            //no axis
            var n = np.arange(4).reshape(2, 2);
            var n1 = np.amin(n).MakeGeneric<double>();
            Assert.IsTrue(n1[0] == 0);

            //2D with axis
            n1 = np.amin(n, 0).MakeGeneric<double>();
            Assert.IsTrue(n1[0] == 0);
            Assert.IsTrue(n1[1] == 1);
            n1 = np.amin(n, 1).MakeGeneric<double>();
            Assert.IsTrue(n1[0] == 0);
            Assert.IsTrue(n1[1] == 2);

            //3D
            n = np.arange(24).reshape(4, 3, 2);
            n1 = np.amin(n, 0).MakeGeneric<double>();
            Assert.IsTrue(n1[0, 1] == 1);
            Assert.IsTrue(n1[2, 1] == 5);
            Assert.IsTrue(n1[1, 1] == 3);
            n1 = np.amin(n, 1).MakeGeneric<double>();
            Assert.IsTrue(n1[1, 1] == 7);
            Assert.IsTrue(n1[2, 1] == 13);
            Assert.IsTrue(n1[3, 0] == 18);

            //4D
            n = np.arange(24).reshape(2, 3, 2, 2);
            n1 = np.amin(n, 1).MakeGeneric<double>();
            Assert.IsTrue(n1[0, 0, 1] == 1);
            Assert.IsTrue(n1[1, 0, 1] == 13);
            Assert.IsTrue(n1[1, 1, 1] == 15);
            n1 = np.amin(n, 3).MakeGeneric<double>();
            Assert.IsTrue(n1[0, 1, 1] == 6);
            Assert.IsTrue(n1[1, 1, 1] == 18);
            Assert.IsTrue(n1[1, 2, 1] == 22);
        }

        [TestMethod]
        public void amin2()
        {
            //no axis
            var n = np.arange(4.0).reshape(2, 2);
            var n1 = np.amin(n);
            Assert.IsTrue((double)n1[0] == 0);

            //2D with axis
            n1 = np.amin(n, 0);
            Assert.IsTrue(n1[0].Equals(0.0));
            Assert.IsTrue(n1[1].Equals(1.0));
            n1 = np.amin(n, 1);
            Assert.IsTrue(n1[0].Equals(0.0));
            Assert.IsTrue(n1[1].Equals(2.0));

            //3D
            n = np.arange(24.0).reshape(4, 3, 2);
            n1 = np.amin(n, 0);
            Assert.IsTrue(n1[0, 1].Equals(1.0));
            Assert.IsTrue(n1[2, 1].Equals(5.0));
            Assert.IsTrue(n1[1, 1].Equals(3.0));
            n1 = np.amin(n, 1);
            Assert.IsTrue(n1[1, 1].Equals(7.0));
            Assert.IsTrue(n1[2, 1].Equals(13.0));
            Assert.IsTrue(n1[3, 0].Equals(18.0));

            //4D
            n = np.arange(24.0).reshape(2, 3, 2, 2);
            n1 = np.amin(n, 1);
            Assert.IsTrue(n1[0, 0, 1].Equals(1.0));
            Assert.IsTrue(n1[1, 0, 1].Equals(13.0));
            Assert.IsTrue(n1[1, 1, 1].Equals(15.0));
            n1 = np.amin(n, 3);
            Assert.IsTrue(n1[0, 1, 1].Equals(6.0));
            Assert.IsTrue(n1[1, 1, 1].Equals(18.0));
            Assert.IsTrue(n1[1, 2, 1].Equals(22.0));
        }
    }
}
