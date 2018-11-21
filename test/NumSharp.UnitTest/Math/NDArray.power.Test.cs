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
            var np = new NumPyGeneric<double>().arange(3);
            
            np = np.power(2);
            
            Assert.IsTrue(np[0] == 0);
            Assert.IsTrue(np[1] == 1);
            Assert.IsTrue(np[2] == 4);
        }
    }
}
