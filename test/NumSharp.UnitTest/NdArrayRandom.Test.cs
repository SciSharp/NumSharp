using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayRandomTest
    {
        [TestMethod]
        public void randn()
        {
            var n = new NumPyRandom().randn(5, 2);
        }

        [TestMethod]
        public void normal()
        {
            var n = new NumPyRandom().normal(0, 1, 5);
        }
    }
}
