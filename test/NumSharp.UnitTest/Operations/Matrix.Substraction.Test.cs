using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class MatrixSubstractionTest
    {

        [TestMethod]
        public void DoubleTwo2D_MatrixSubstraction()
        {
            var np1 = new Matrix<Double>("1 2 3;4 5 6;7 8 9");
            var np2 = new Matrix<Double>("1 2 3;4 5 6;7 8 9");

            var np3 = np1 - np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 }, np3.Data));
        }

        [TestMethod]
        public void ComplexTwo2D_MatrixSubstraction()
        {
            var np1 = new Matrix<Complex>();
            var np2 = new Matrix<Complex>();

            np1.Data = new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8) };
            np2.Data = new Complex[] { new Complex(8, 7), new Complex(6, 5), new Complex(4, 3), new Complex(2, 1) };

            var np3 = np1 - np2;

            var expArray = new Complex[] { new Complex(-7, -5), new Complex(-3, -1), new Complex(1, 3), new Complex(5, 7) };

            Assert.IsTrue(Enumerable.SequenceEqual(expArray, np3.Data));
        }
    }
}
