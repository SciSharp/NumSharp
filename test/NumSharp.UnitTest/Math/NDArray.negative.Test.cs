using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class NDArrayNegativeTest
    {
        [TestMethod]
        public void NegateAllPositives()
        {
            NDArray nd = new[] {1, -2, 3.3f};
            nd = np.negative(nd);
            Assert.IsTrue(nd.Data<float>().All(v => v <= 0));
        }
    }
}
