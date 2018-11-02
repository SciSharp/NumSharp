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
            var series1 = new NDArray<double>().ARange(4,1,1) ;
            
            var series2 = new NDArray<double>();
            series2.Data = new double[]{0, 1, 0.5};

            var innerProduct = series1.Dot(series2);
            Assert.IsTrue(innerProduct.Data[0] == 3.5);
        }
        [TestMethod]
        public void MatrixMultiplyDouble()
        {   
            NDArray<double> matrix1 = new NDArray<double>().ARange(7,1,1).ReShape(3,2);
            
            NDArray<double> matrix2 = new NDArray<double>().ARange(13,7,1).ReShape(2,3) ;
        
            var matrix3 = matrix1.Dot(matrix2);
            
            var matrix4 = new double[]{39,54,69,49,68,87,59,82,105};

            Assert.IsTrue(Enumerable.SequenceEqual(matrix3.Data,matrix4));
        }
        [TestMethod]
        public void MatrixMultiplyComplex()
        {   
            NDArray<Complex> matrix1 = new NDArray<Complex>();
            matrix1.Data = new Complex[] {new Complex(1,-1),new Complex(2,-2), new Complex(3,0),new Complex(4,0), 5, 6}; 
            matrix1.Shape = new int[] {3, 2};

            NDArray<Complex> matrix2 = new NDArray<Complex>();
            matrix2.Data = new Complex[] {7,8,9,new Complex(10,-10),11, new Complex(12,-12)};
            matrix2.Shape = new int[] {2,3};

            var matrix3 = matrix1.Dot(matrix2);

            var matrix4 = new Complex[9];
            matrix4[0] = new Complex(39,-7);
            matrix4[1] = new Complex(54,-14);
            matrix4[2] = new Complex(69,0);
            matrix4[3] = new Complex(49,-49);
            matrix4[4] = new Complex(68,-68);
            matrix4[5] = new Complex(87,-60);
            matrix4[6] = new Complex(59,-59);
            matrix4[7] = new Complex(82,-82);
            matrix4[8] = new Complex(105,-72);

            Assert.IsTrue(Enumerable.SequenceEqual(matrix4,matrix3.Data));
        }
        [TestMethod]
        public void DotTwo1DComplex()
        {
            var series1 = new NDArray<Complex>().Array(new Complex[]{new Complex(0,2),new Complex(0,3)});
            
            var series2 = new NDArray<Complex>().Array(new Complex[]{new Complex(0,2),new Complex(0,3)});
        
            //var series3 = series1.Dot(series2);

            //Complex[] expectedResult = new Complex[]{new Complex(-13,0)};

            //Assert.IsTrue(Enumerable.SequenceEqual(series3.Data.ToArray(),expectedResult));
            
        }
        
    }
}