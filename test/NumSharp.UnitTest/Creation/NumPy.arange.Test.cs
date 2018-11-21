using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NumPyArangeTest
    {
        [TestMethod]
        public void arange()
        {
            var np = new NumPy();

            var n = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] { 0, 1, 2 }));

            n = np.arange(3, 7);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] { 3, 4, 5, 6 }));

            n = np.arange(3, 7, 2, np.double8);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<double>(), new double[] { 3, 5 }));

            n = np.arange(0, 11, 3, np.int16);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] { 0, 3, 6, 9 }));
        }
    }
}
