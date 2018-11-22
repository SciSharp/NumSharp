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
    public class NumPyZerosTest
    {
        [TestMethod]
        public void zero()
        {
            var np = new NumPy();

            var n = np.zeros(3);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<float>(), new float[] { 0, 0, 0 }));

            n = np.zeros<int>(2, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] { 0, 0, 0, 0 }));
        }
    }
}
