using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArraySqrtTest
    {
        [TestMethod]
        public void SqrtTest()
        {
            var np = new NDArray<double>().ARange(3);
            np[0] = 1;
            np[1] = 4;
            np[2] = 9;
            Assert.IsTrue(Enumerable.SequenceEqual(np.Sqrt().Data, new double[] { 1, 2, 3 }));
        }
    }
}
