using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NDArraySqrtTest
    {
        [TestMethod]
        public void DoubleSqrtTest()
        {
            var np = new NDArrayGeneric<double>().arange(3);
            np[0] = 1;
            np[1] = 4;
            np[2] = 9;
            Assert.IsTrue(Enumerable.SequenceEqual(np.Sqrt().Data, new double[] { 1, 2, 3 }));
        }
        [TestMethod]
        public void ComplexSqrtTest()
        {
            var np = new NDArrayGeneric<Complex>();
            np.Data = new Complex[] {
                new Complex(4, 0),
                new Complex(-1, 0),
                new Complex(-3, 4)
            };

            var actual = np.Sqrt().Data;
            var expected = new Complex[3];
            expected[0] = new Complex(2, 0);
            expected[1] = new Complex(0, 1);
            expected[2] = new Complex(1, 2);

            Assert.IsTrue(Enumerable.SequenceEqual(actual, expected));
        }

    }
}
