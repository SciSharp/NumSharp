using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumPyConcatenateTest
    {
        [TestMethod]
        public void axis0()
        {
            var a = new int[][] { new int[] { 1, 2 }, new int[] { 3, 4 } };
            var b = new int[][] { new int[] { 5, 6 } };
            var nd = np.concatenate(new int[][][] { a, b }, axis: 0);
            Assert.AreEqual(nd.shape[0], 3);
            Assert.AreEqual(nd.shape[1], 2);
        }
    }
}
