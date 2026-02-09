using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    public class NumPyZerosTest
    {
        [Test]
        public void zero()
        {
            var n = np.zeros(3);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<double>(), new double[] {0, 0, 0}));
        }

        [Test]
        public void Zeros2Dim()
        {
            var n = np.zeros(3, 2);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<double>(), new double[] {0, 0, 0, 0, 0, 0}));
        }

        [Test]
        public void Zeros1DimWithDtype()
        {
            var n = np.zeros(new Shape(3), np.int32);
            Assert.IsTrue(Enumerable.SequenceEqual(n.Data<int>(), new int[] {0, 0, 0}));
        }

        [Test]
        public void SimpleInt1D()
        {
            var np1 = np.zeros(new Shape(5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 5);
        }

        [Test]
        public void SimpleInt2D()
        {
            var np1 = np.zeros(new Shape(5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 25);
        }

        [Test]
        public void SimpleDouble3D()
        {
            var np1 = np.zeros(new Shape(5, 5, 5));

            Assert.IsTrue(np1.Data<double>().Where(x => x == 0).ToArray().Length == 125);
        }
    }
}
