using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class LogTest
    {
        [TestMethod]
        public void Simple1DArray()
        {
            var np1 = np.array(new double[] {1, Math.E, Math.E*Math.E, 0}).MakeGeneric<double>();
            
            var np2 = np1.log().MakeGeneric<double>();
            
            Assert.IsTrue(np2[0] == 0);
            Assert.IsTrue(np2[1] == 1);
            Assert.IsTrue(np2[2] == 2);
            Assert.IsTrue(np2[3] == double.NegativeInfinity);
            
        }
    }
}
