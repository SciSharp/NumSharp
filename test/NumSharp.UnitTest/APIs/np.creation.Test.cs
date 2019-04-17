using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class ApiCreationTest
    {
        [TestMethod]
        public void arange()
        {
            var nd = np.arange(9);
            // Assert.IsTrue(Enumerable.SequenceEqual(nd.Data<int>(), new int[] { 0, 1, 2 }));
        }

        [TestMethod]
        public void ndarray()
        {
            var x = np.ndarray((2, 3), dtype: np.int32, order: "F");
            for (int i = 0; i < 6; i++)
                x.itemset(i, i + 1);
        }
    }
}
