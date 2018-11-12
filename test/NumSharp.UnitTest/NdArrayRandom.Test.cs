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
        [TestMethod]
        public void NDArrayRandom()
        {
            NumPy<double> np = new NumPy<double>();
            var n = np.random.randn(5, 2);
        }
    }
}
