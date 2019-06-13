using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NpRandomUniformTests : TestClass
    {
        [TestMethod]
        public void Basic()
        {
            var low = np.array(1d, 2d, 3d, 4d, 5d);
            var high = low + 1;
            var uniformed = np.random.uniform(low, high);
        }
    }
}
