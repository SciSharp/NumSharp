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
    public class NDArrayAdditionTest
    {
        
        [TestMethod]
        public void  DoubleTwo1D()
        {
            var np1 = new NDArray<double>().Array(new double[]{1,2,3});
            var np2 = new NDArray<double>().Array(new double[]{2,3,4});

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[]{3,5,7},np3.Data));

        }
        [TestMethod]
        public void  ComplexTwo1D()
        {
            var np1 = new NDArray<Complex>().Array(new Complex[]{new Complex(1,2),new Complex(3,4)});
            var np2 = new NDArray<Complex>().Array(new Complex[]{new Complex(5,6),new Complex(7,8)});

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new Complex[]{new Complex(6,8),new Complex(10,12)},np3.Data));

        }
        [TestMethod]
        public void Double1DPlusOffset()
        {
            var np1 = new NDArray<double>().Array(new double[]{1,2,3});

            var np3 = np1 + 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[]{3,4,5},np3.Data));
        }
        [TestMethod]
        public void Complex1DPlusOffset()
        {
            var np1 = new NDArray<Complex>().Array(new Complex[]{new Complex(1,2),new Complex(3,4)});

            var np2 = np1 + new Complex(1,2);    

            Assert.IsTrue(Enumerable.SequenceEqual(new Complex[]{new Complex(2,4),new Complex(4,6)},np2.Data));
        }
        [TestMethod]
        public void Complex2DArray()
        {
            var np1 = new NDArray<Complex[]>().Array(new Complex[][]{new Complex[]{new Complex(4,1),new Complex(3,5)},new Complex[]{new Complex(0,-2),new Complex(-3,2)}} );
            var np2 = new NDArray<Complex[]>().Array(new Complex[][]{new Complex[]{new Complex(1,2),new Complex(3,4)},new Complex[]{new Complex(1,2),new Complex(3,4)}} );

            var np3 = np1 + np2;

            // expected 
            var np4 = new Complex[][]{new Complex[]{new Complex(5,3),new Complex(6,9)},new Complex[]{new Complex(1,0),new Complex(0,6)}};

            // avoid for for loop by compare elements via linq
            var allEqual = np3.Data.Select( (x,idx) => x.Select( (y,jdx) => y == np4[idx][jdx] ) ).SelectMany(x => x).ToArray();

            Assert.IsTrue(allEqual.All(x => x == true));

        }
        [TestMethod]
        public void Double2DArray()
        {
            var np1 = new NDArray<double[]>().Array(new double[][]{new double[]{1,2,3},new double[]{4,5,6}} );
            var np2 = new NDArray<double[]>().Array(new double[][]{new double[]{9,8,7},new double[]{6,5,4}} );

            var np3 = np1 + np2;

            // expected 
            var np4 = new NDArray<double[]>().Array(new double[][]{new double[]{10,10,10},new double[]{10,10,10}} );

            // avoid for for loop by compare elements via linq
            var allEqual = np3.Data.Select( (x,idx) => x.Select( (y,jdx) => y == np4[idx][jdx] ) ).SelectMany(x => x).ToArray();

            Assert.IsTrue(allEqual.All(x => x == true));

        }
    }
}