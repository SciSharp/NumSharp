using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumPyZerosTest
    {
        [TestMethod]
        public void zero()
        {
            var n = np.zeros(3);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<double>(), new double[] { 0, 0, 0 }));
        }
    }
}
