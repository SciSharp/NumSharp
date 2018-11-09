using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArrayStdTest
    {
        [TestMethod]
        public void StdTest()
        {
            var np = new NDArray<double>().ARange(4).reshape(2,2);

            // Assert.IsTrue(Enumerable.SequenceEqual(np.Std().Data, new double[] { 1.1180339887498949 }));
            // Assert.IsTrue(Enumerable.SequenceEqual(np.Std(0).Data, new double[] { 1, 1 }));
            // Assert.IsTrue(Enumerable.SequenceEqual(np.Std(1).Data, new double[] { 0.5, 3.5 }));
        }
    }
}
