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
        public void ARange()
        {
            var np = new NDArray<int>();

            np.ARange(3);
            Assert.IsTrue(Enumerable.SequenceEqual(np.Data, new int[] { 0, 1, 2 }));

            np.ARange(7, 3);
            Assert.IsTrue(Enumerable.SequenceEqual(np.Data, new int[] { 3, 4, 5, 6 }));

            np.ARange(7, 3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(np.Data, new int[] { 3, 5 }));
        }
    }
}
