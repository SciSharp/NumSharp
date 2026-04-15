using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class NDArraySqrtTest
    {
        [TestMethod]
        public void Case1()
        {
            var nd = np.array(new double[] {1, 4, 9});
            np.sqrt(nd).Should().BeOfType<double>().And.BeOfValues(1, 2, 3);
        }

        /*[TestMethod]
        public void ComplexSqrtTest()
        {
            var np = new NDArray(typeof(Complex),3);
            np.Storage.ReplaceData(new Complex[] {
                new Complex(4, 0),
                new Complex(-1, 0),
                new Complex(-3, 4)
            });

            var actual = np.sqrt().Storage.GetData<Complex>();
            var expected = new Complex[3];
            expected[0] = new Complex(2, 0);
            expected[1] = new Complex(0, 1);
            expected[2] = new Complex(1, 2);

            Assert.IsTrue(Enumerable.SequenceEqual(actual, expected));
        }*/
    }
}
