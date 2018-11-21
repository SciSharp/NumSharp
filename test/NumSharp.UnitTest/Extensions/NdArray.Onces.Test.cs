using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayOncesTest
    {
        [TestMethod]
        public void SimpleInt1D()
        {
            var np = new NumPy<int>().ones(new Shape(5));

            Assert.IsTrue(np.Data.Where(x => x==1).ToArray().Length == 5);
        }
        [TestMethod]
        public void SimpleInt2D()
        {
            var np = new NumPy<int>().ones(new Shape(5,5));

            Assert.IsTrue(np.Data.Where(x => x==1).ToArray().Length == 25);
        }
        [TestMethod]
        public void SimpleDouble3D()
        {
            var np = new NumPy<double>().ones(new Shape(5,5,5));

            Assert.IsTrue(np.Data.Where(x => x==1).ToArray().Length == 125);
        }
    }
}
