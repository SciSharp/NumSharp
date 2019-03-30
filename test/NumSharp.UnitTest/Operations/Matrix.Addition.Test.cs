using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class MatrixAdditionTest 
    {
        [TestMethod]
        public void DoubleTwo2D_MatrixAddition()
        {
            var np1 = new matrix("1 2 3;4 5 6;7 8 9");
            var np2 = new matrix("1 2 3;4 5 6;7 8 9");

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18 }, np3.Storage.GetData<double>()));
        }

        /*[TestMethod]
        public void ComplexTwo2D_MatrixAddition()
        {
            var np1 = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4) });
            var np2 = np.array(new Complex[] { new Complex(5, 6), new Complex(7, 8) });

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new Complex[] { new Complex(6, 8), new Complex(10, 12) }, np3.Storage.GetData<Complex>()));
        }*/
    }
}
