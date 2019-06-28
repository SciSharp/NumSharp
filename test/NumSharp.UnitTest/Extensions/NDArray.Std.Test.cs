using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArrayStdTest
    {
        [Ignore]
        [TestMethod]
        public void StdTest()
        {
            var nd1 = np.arange(4).reshape(2,2).MakeGeneric<double>();

            Assert.IsTrue(Enumerable.SequenceEqual(nd1.std().Data<double>(), new double[] { 1.1180339887498949 }));
            Assert.IsTrue(Enumerable.SequenceEqual(nd1.std(0).Data<double>(), new double[] { 1, 1 }));
            Assert.IsTrue(Enumerable.SequenceEqual(nd1.std(1).Data<double>(), new double[] { 0.5, 0.5 }));
        }
    }
}
