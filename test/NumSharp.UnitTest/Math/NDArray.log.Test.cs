using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class LogTest
    {
        [TestMethod]
        public void Case1()
        {
            var np1 = np.array(new double[] {1, Math.E, Math.E * Math.E, 0}); // .MakeGeneric<double>();
            np.log(np1).Should().BeOfValues(0, 1, 2, double.NegativeInfinity);
        }
    }
}
