using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class PowerTest
    {
        [TestMethod]
        public void PowerWithSingleValue()
        {
            // np.arange returns int64 by default (NumPy 2.x)
            var nd = np.arange(3);

            var nd1 = np.power(nd, 2).MakeGeneric<long>();

            Assert.IsTrue(nd1[0] == 0);
            Assert.IsTrue(nd1[1] == 1);
            Assert.IsTrue(nd1[2] == 4);
        }
    }
}
