using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;


namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class np_ones_Test
    {
        [TestMethod]
        public void SimpleInt1D()
        {
            var np1 = np.ones(new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x==1).ToArray().Length == 5);
        }
        [TestMethod]
        public void SimpleInt2D()
        {
            var np1 = np.ones(new Shape(5,5));

            Assert.IsTrue(np1.Data<double>().Where(x => x==1).ToArray().Length == 25);
        }
        [TestMethod]
        public void SimpleDouble3D()
        {
            var np1 = np.ones(new Shape(5,5,5));

            Assert.IsTrue(np1.Data<double>().Where(x => x==1).ToArray().Length == 125);
        }
    }
}
