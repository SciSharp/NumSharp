using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class MatrixAdditionTest
    {
        
        [TestMethod]
        public void  DoubleTwo2D()
        {
            var np1 = new Matrix<Double>("1 2 3;4 5 6;7 8 9");
            var np2 = new Matrix<Double>("1 2 3;4 5 6;7 8 9");

            var np3 = np1 + np2;

            Assert.IsTrue( Enumerable.SequenceEqual(new double[]{2, 4, 6, 8, 10, 12, 14, 16, 18},np3.Data));
        }
        [TestMethod]
        public void  ComplexTwo2D()
        {
            var np1 = new NDArray<Complex>().Array(new Complex[]{new Complex(1,2),new Complex(3,4)});
            var np2 = new NDArray<Complex>().Array(new Complex[]{new Complex(5,6),new Complex(7,8)});

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new Complex[]{new Complex(6,8),new Complex(10,12)},np3.Data));
            
        }
        
    }
}