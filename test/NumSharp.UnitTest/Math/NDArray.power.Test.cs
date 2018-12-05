using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class PowerTest
    {
        [TestMethod]
        public void PowerWithSingleValue()
        {
            var np = new NumPy().arange(3);
            
            var np1 = np.power(2).MakeGeneric<double>();
            
            Assert.IsTrue(np1[0] == 0);
            Assert.IsTrue(np1[1] == 1);
            Assert.IsTrue(np1[2] == 4);
        }
    }
}
