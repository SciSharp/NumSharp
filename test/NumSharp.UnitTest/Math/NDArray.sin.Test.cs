using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class SinTest
    {
        [TestMethod]
        public void Simple1DArray()
        {
            var np1 = new NDArray<double>().array(new double[] {0, 30, 45, 60, 90}) * (Math.PI / 180);
            
            var np2 = np1.sin();
            
            Assert.IsTrue(np2[0] == 0);
            Assert.IsTrue(np2[1] < 0.501);
            Assert.IsTrue(np2[1] > 0.498);
            Assert.IsTrue(np2[2] < 0.708);
            Assert.IsTrue(np2[2] > 0.7069);
            Assert.IsTrue(np2[3] < 0.867);
            Assert.IsTrue(np2[3] > 0.8659);
            Assert.IsTrue(np2[4] == 1);
            
        }
    }
}
