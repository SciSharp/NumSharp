using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayRandomRandIntTest
    {
        [TestMethod]
        public void randint()
        {
            var np = new NumPy<int>();
            var a = np.random.randint(low: 0, high: 10, size: new Shape(5, 5));
            Assert.IsTrue(a.Data.Count(x => x < 10) == 25);
        }
    }
}
