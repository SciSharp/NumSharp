using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
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
            np.BackendEngine = BackendType.VectorT;
            var n = np.arange(3);
        }
    }
}
