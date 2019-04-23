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
        public void sum2x2()
        {
            var data = new int[,] { { 0, 1 }, { 0, 5 } };

            int s = np.sum(data);
            Assert.AreEqual(s, 6);

            var s0 = np.sum(data, axis: 0);
            Assert.IsTrue(Enumerable.SequenceEqual(s0.shape, new int[] { 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s0.Data<int>(), new int[] { 0, 6 }));

            var s1 = np.sum(data, axis: 1);
            Assert.IsTrue(Enumerable.SequenceEqual(s1.shape, new int[] { 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s1.Data<int>(), new int[] { 1, 5 }));
        }

        [TestMethod]
         public void sum2x3x2()
        {
            var data = np.arange(12).reshape(2, 3, 2);

            int s = np.sum(data);
            Assert.AreEqual(s, 66);

            var s0 = np.sum(data, axis: 0);
            Assert.IsTrue(Enumerable.SequenceEqual(s0.shape, new int[] { 3, 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s0.Data<int>(), new int[] { 6, 8, 10, 12, 14, 16 }));

            var s1 = np.sum(data, axis: 1);
            Assert.IsTrue(Enumerable.SequenceEqual(s1.shape, new int[] { 2, 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s1.Data<int>(), new int[] { 6, 9, 24, 27 }));

            var s2 = np.sum(data, axis: 2);
            Assert.IsTrue(Enumerable.SequenceEqual(s2.shape, new int[] { 2, 3 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s2.Data<int>(), new int[] { 1, 5, 9, 13, 17, 21 }));
        }
    }
}
