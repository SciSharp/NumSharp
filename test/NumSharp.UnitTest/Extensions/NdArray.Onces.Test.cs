using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayOncesTest
    {
        [TestMethod]
        public void SimpleInt1D()
        {
            var np = new NDArray<int>().Onces(5);

            Assert.IsTrue(np.Data.Where(x => x==1).ToArray().Length == 5);
        }
        [TestMethod]
        public void SimpleInt2D()
        {
            var np = new NDArray<int>().Onces(5,5);

            Assert.IsTrue(np.Data.Where(x => x==1).ToArray().Length == 25);
        }
        [TestMethod]
        public void SimpleDouble3D()
        {
            var np = new NDArray<double>().Onces(5,5,5);

            Assert.IsTrue(np.Data.Where(x => x==1).ToArray().Length == 125);
        }
    }
}
