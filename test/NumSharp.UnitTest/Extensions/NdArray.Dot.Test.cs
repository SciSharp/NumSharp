using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp.UnitTest.Shared;
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
            var series1 = new NDArray_Legacy<double>();
            series1.Data = new double[]{1, 2, 3};
            
            var series2 = new NDArray_Legacy<double>();
            series2.Data = new double[]{0, 1, 0.5};

            var innerProduct = series1.Dot(series2);
            Assert.IsTrue(innerProduct[0] == 3.5);
        }
        [TestMethod]
        public void MatrixMultiplyDouble()
        {   
            NDArray_Legacy<double[]> matrix1 = new NDArray_Legacy<double[]>();
            matrix1.Data = ArrayHelper.CreateJaggedArrayByMatrix( new double[,] {{1,2},{3,4},{5,6}} ); 
            
            var matrix2 = new NDArray_Legacy<double[]>();
            matrix2.Data = ArrayHelper.CreateJaggedArrayByMatrix(new double[,] {{7,8,9},{10,11,12}});

            var matrix3 = matrix1.Dot(matrix2);
            
            Assert.IsTrue(ArrayHelper.CompareTwoJaggedArrays((double[][])matrix3.Data,ArrayHelper.CreateJaggedArrayByMatrix(new double[,]{{27,30,33}, {61,68,75},{95,106,117}}) ));
        }
        [TestMethod]
        public void MatrixMultiplyComplex()
        {   
            NDArray_Legacy<Complex[]> matrix1 = new NDArray_Legacy<Complex[]>();
            matrix1.Data = ArrayHelper.CreateJaggedArrayByMatrix( new Complex[,] {{new Complex(1,-1),new Complex(2,-2)},{3,4},{5,6}} ); 

            var matrix2 = new NDArray_Legacy<Complex[]>();
            matrix2.Data = ArrayHelper.CreateJaggedArrayByMatrix(new Complex[,] {{7,8,9},{new Complex(10,-10),11, new Complex(12,-12)}});

            var matrix3 = matrix1.Dot(matrix2);

            var expectedResult = new Complex[3,3];
            expectedResult[0,0] = new Complex(7,-47);
            expectedResult[0,1] = new Complex(30,-30);
            expectedResult[0,2] = new Complex(9,-57);
            expectedResult[1,0] = new Complex(61,-40);
            expectedResult[1,1] = new Complex(68,0);
            expectedResult[1,2] = new Complex(75,-48);
            expectedResult[2,0] = new Complex(95,-60);
            expectedResult[2,1] = new Complex(106,0);
            expectedResult[2,2] = new Complex(117,-72);
            
            Assert.IsTrue(ArrayHelper.CompareTwoJaggedArrays((Complex[][])matrix3.Data,ArrayHelper.CreateJaggedArrayByMatrix(expectedResult) ));
        }
        [TestMethod]
        public void DotTwo1DComplex()
        {
            var series1 = new NDArray_Legacy<Complex>().Array(new Complex[]{new Complex(0,2),new Complex(0,3)});
            
            var series2 = new NDArray_Legacy<Complex>().Array(new Complex[]{new Complex(0,2),new Complex(0,3)});
        
            var series3 = series1.Dot(series2);

            Complex[] expectedResult = new Complex[]{new Complex(-13,0)};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Data.ToArray(),expectedResult));
            
        }
        
    }
}