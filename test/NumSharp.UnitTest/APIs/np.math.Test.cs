using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class ApiMathTest
    {
        [TestMethod]
        public void add()
        {
            var x = np.arange(3);
            var y = np.arange(3);
            var z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<int>(), new int[] { 0, 2, 4 }));

            x = np.arange(9);
            y = np.arange(9);
            z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<int>(), new int[] { 0, 2, 4, 6, 8, 10, 12, 14, 16 }));
        }

        [TestMethod]
        public void sum()
        {
            var data = new int[,] { { 0, 1 }, { 0, 5 } };

            int s1 = np.sum(data);
            Assert.AreEqual(s1, 6);

            var s2 = np.sum(data, axis: 0);
            Assert.AreEqual(s2, 6);
        }

        [TestMethod]
        public void test()
        {
            var a = np.zeros((2, 2, 3), np.int32);
            NDArray b = new int[,] { { 1 }, { 2 } };
            NDArray c = new int[,] { { 1, 2, 3 } };
            a[0] = b;
            a[1] = c;
        }
    }
}
