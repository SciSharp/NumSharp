using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;


namespace NumSharp.UnitTest.Extensions
{
    public class np_zeros_like_Test
    {
        [Test]
        public void SimpleInt1D()
        {
            // create same-shaped zeros from ones
            var np1 = np.zeros_like(np.ones(new Shape(5)));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 5);
        }

        [Test]
        public void SimpleInt2D()
        {
            // create same-shaped zeros from ones
            var np1 = np.zeros_like(np.ones(new Shape(5, 5)));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 25);
        }

        [Test]
        public void SimpleDouble3D()
        {
            // create same-shaped zeros from ones
            var np1 = np.zeros_like(np.ones(new Shape(5, 5, 5)));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 125);
        }
    }
}
