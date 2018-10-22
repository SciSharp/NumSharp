using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayDotTest
    {
        [TestMethod]
        public void DotTwo1DDouble()
        {
            var series1 = new NDArray<double>();
            series1.Data = new double[]{1, 2, 3};
            
            var series2 = new NDArray<double>();
            series2.Data = new double[]{0, 1, 0.5};

            var innerProduct = series1.Dot(series2);
            Assert.IsTrue(innerProduct[0] == 3.5);
        }
        [TestMethod]
        public void MatrixMultiply()
        {
            var matrix1 = new NDArray<NDArray<double>>().Array(new double[,] {{1,2},{3,4},{5,6} });
            var matrix2 = new NDArray<NDArray<double>>().Array(new double[,] {{7,8,9},{10,11,12}});

            var matrix3 = matrix1.Dot(matrix2);
            
            var expectedResult = new NDArray<NDArray<double>>().Array(new double[,]{{27,30,33}, {61,68,75},{95,106,117}});
        }
        [TestMethod]
        public void DotTwo1DComplex()
        {
            var series1 = new NDArray<Complex>().Array(new Complex[]{new Complex(0,2),new Complex(0,3)});
            
            var series2 = new NDArray<Complex>().Array(new Complex[]{new Complex(0,2),new Complex(0,3)});
        
            var series3 = series1.Dot(series2);

            Complex[] expectedResult = new Complex[]{new Complex(-13,0)};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data.ToArray(),expectedResult));
            
        }
        
    }
}