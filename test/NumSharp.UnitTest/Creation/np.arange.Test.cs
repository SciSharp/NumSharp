using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumPyArangeTest
    {
        [TestMethod]
        public void arange()
        {
            var n = np.arange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<int>(), new int[] { 0, 1, 2 }));

            n = np.arange(3, 7);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<int>(), new int[] { 3, 4, 5, 6 }));

            n = np.arange(3.0, 7.0, 2.0);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<double>(), new double[] { 3, 5 }));

            n = np.arange(0, 11, 3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<int>(), new int[] { 0, 3, 6, 9 }));
        }
    }
}
