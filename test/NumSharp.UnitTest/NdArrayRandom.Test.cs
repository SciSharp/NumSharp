using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayRandomTest
    {
        NumPy<double> np = new NumPy<double>();
        [TestMethod]
        public void randn()
        {
            var n = np.random.randn(5, 2);
        }

        public void normal()
        {
            var n = np.random.normal(0, 1, 5);
        }
    }
}
