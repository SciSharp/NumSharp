using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class ApiMathTest
    {
        [TestMethod]
        public void add()
        {
            var x = np.arange(3);
            var y = np.arange(3);
            var z = np.add(x, y);
        }
    }
}
