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
    public class NumSharpArangeTest
    {
        [TestMethod]
        public void arange()
        {
            var np = new NumPyGeneric<int>();

            var n = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 0, 1, 2 }));

            n = np.arange(3, 7);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 3, 4, 5, 6 }));

            n = np.arange(3, 7, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 3, 5 }));

            n = np.arange(0, 11, 3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data, new int[] { 0, 3, 6, 9 }));

            var nd2 = new NDArray(Core.NumPy.int16);
            var nd3 = nd2.arange(3);
        }
    }
}
