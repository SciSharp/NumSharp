using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.UnitTest.LinearAlgebra
{
    [TestClass]
    public class NpProdTest
    {
        [TestMethod]
        public void NoAxis()
        {
            int p = np.prod(new int[] { 1, 2, 3 });
            Assert.AreEqual(p, 6);

            p = np.prod(new int[][] { new int[] { 1, 2 }, new int[] { 3, 4 } });
            Assert.AreEqual(p, 24);
        }
    }
}
