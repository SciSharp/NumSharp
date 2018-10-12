using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayRandomPermutationTest
    {
        [TestMethod]
        public void Prmutation()
        {
            var np = new NdArray<int>();
            var rands = np.Random().Permutation(10);
            var results = rands.Data.Distinct();

            var str = String.Join(", ", results);

            Assert.IsTrue(results.Count() == 10);
            Assert.IsTrue(str != "0, 1, 2, 3, 4, 5, 6, 7, 8, 9");
        }
    }
}
