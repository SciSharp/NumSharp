using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayRandomTest
    {
        [TestMethod]
        public void randn()
        {
            var n = new NumSharp.Core.NumPyRandom().randn(5, 2);
        }

        [TestMethod]
        public void normal()
        {
            var n = new NumSharp.Core.NumPyRandom().normal(0, 1, 5);
        }
    }
}
