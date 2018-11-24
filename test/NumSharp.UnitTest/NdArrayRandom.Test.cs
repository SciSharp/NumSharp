using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayRandomTest
    {
        NumPy np = new NumPy();

        [TestMethod]
        public void randn()
        {
            var n = np.random.randn(5, 2);
        }

        [TestMethod]
        public void normal()
        {
            var n = np.random.normal(0, 1, 5);
        }
    }
}
