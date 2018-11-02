using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayARangeTest
    {
        [TestMethod]
        public void arange()
        {
            var np = new NumPy<int>();

            var n = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 0, 1, 2 }));

            n = np.arange(3, 7);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 3, 4, 5, 6 }));

            n = np.arange(3, 7, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 3, 5 }));

            n = np.arange(0, 11, 3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 0, 3, 6, 9 }));
        }
    }
}
