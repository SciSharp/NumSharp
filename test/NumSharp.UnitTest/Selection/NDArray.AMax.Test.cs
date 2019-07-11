using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.UnitTest.Selection
{
    [TestClass]
    public class NDArrayAMaxTest
    {
        [TestMethod]
        public void argmax12()
        {
            NDArray x = DataSample.Int32D12;

            int y0 = np.argmax(x);
            Assert.AreEqual(y0, 3);
        }

        [TestMethod]
        public void argmax4x3()
        {
            NDArray x = DataSample.Int32D4x3;

            var y0 = np.argmax(x, 0);
            Assert.IsTrue(Enumerable.SequenceEqual(y0.Data<int>(), new int[] {0, 3, 2}));

            var y1 = np.argmax(x, 1);
            Assert.IsTrue(Enumerable.SequenceEqual(y1.Data<int>(), new int[] {0, 1, 2, 1}));
        }

        [TestMethod]
        public void amax()
        {
            //default type
            var n = np.arange(0, 12, 0.1);
            double d1 = np.amax<double>(n);
            Assert.IsTrue(d1.Equals(11.9));

            //no axis
            n = np.arange(4).reshape(2, 2);
            var max = np.amax<int>(n);

            Assert.IsTrue(max == 3);

            //2D with axis
            var n1 = np.amax(n, 0).MakeGeneric<int>();
            Assert.IsTrue(n1[0] == 2);
            Assert.IsTrue(n1[1] == 3);
            n1 = np.amax(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[0] == 1);
            Assert.IsTrue(n1[1] == 3);

            //3D
            /*n = np.arange(24).reshape(4, 3, 2);
            n1 = np.amax(n, 0).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 1] == 19);
            Assert.IsTrue(n1[2, 1] == 23);
            Assert.IsTrue(n1[1, 1] == 21);
            n1 = np.amax(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[1, 1] == 11);
            Assert.IsTrue(n1[2, 1] == 17);
            Assert.IsTrue(n1[3, 0] == 22);

            //4D
            n = np.arange(24).reshape(2, 3, 2, 2);
            n1 = np.amax(n, 1).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 0, 1] == 9);
            Assert.IsTrue(n1[1, 0, 1] == 21);
            Assert.IsTrue(n1[1, 1, 1] == 23);
            n1 = np.amax(n, 3).MakeGeneric<int>();
            Assert.IsTrue(n1[0, 1, 1] == 7);
            Assert.IsTrue(n1[1, 1, 1] == 19);
            Assert.IsTrue(n1[1, 2, 1] == 23);*/
        }
    }
}
