using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp.UnitTest.Shared;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;

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
            
            var series1 = new NDArrayGeneric<double>().arange(4,1,1) ;
            
            var series2 = new NDArrayGeneric<double>();
            series2.Data = new double[]{0, 1, 0.5};
            series2.Shape = new Shape(3);

            var innerProduct = series1.dot(series2);
            Assert.IsTrue(innerProduct.Data[0] == 3.5);
        }
        [TestMethod]
        public void MatrixMutliplyDifferentDataLength()
        {
            var A = new NDArrayGeneric<double>().arange(6).reshape(3,2);

            var B = new NDArrayGeneric<double>().arange(14).reshape(2,7);

            var C = A.dot(B);

            Assert.IsTrue(Enumerable.SequenceEqual(new double[]{7,8,9,10,11,12,13,21,26,31,36,41,46,51,35,44,53,62,71,80,89},C.Data));

        }
        [TestMethod]
        public void MatrixMultiplyDouble()
        {   
            NDArrayGeneric<double> matrix1 = new NDArrayGeneric<double>().arange(7,1,1).reshape(3,2);
            
            Assert.IsTrue(matrix1[0,0] == 1);
            Assert.IsTrue(matrix1[0,1] == 2);
            Assert.IsTrue(matrix1[1,0] == 3);
            Assert.IsTrue(matrix1[2,0] == 5);

            NDArrayGeneric<double> matrix2 = new NDArrayGeneric<double>().arange(13,7,1).reshape(2,3) ;
        
            Assert.IsTrue(matrix2[0,0] == 7);
            Assert.IsTrue(matrix2[0,1] == 8);
            Assert.IsTrue(matrix2[1,0] == 10);
            Assert.IsTrue(matrix2[0,2] == 9);


            var matrix3 = matrix1.dot(matrix2);

            var matrix4 = new NDArrayGeneric<double>().Zeros(3,3);

            for (int idx = 0; idx < 3; idx++)
            {
                for (int jdx = 0; jdx < 3;jdx++)
                {
                    matrix4[idx,jdx] = 0;
                    for (int kdx = 0; kdx < 2;kdx++)
                    {
                        matrix4[idx,jdx] += matrix1[idx,kdx] * matrix2[kdx,jdx];
                    }
                }
            }

            Assert.IsTrue(matrix4[0,0] == matrix3[0,0]);
            Assert.IsTrue(matrix4[0,1] == matrix3[0,1]);
            Assert.IsTrue(matrix4[0,2] == matrix3[0,2]);
            Assert.IsTrue(matrix4[1,0] == matrix3[1,0]);
            Assert.IsTrue(matrix4[1,1] == matrix3[1,1]);
            Assert.IsTrue(matrix4[1,2] == matrix3[1,2]);
            Assert.IsTrue(matrix4[2,0] == matrix3[2,0]);
            Assert.IsTrue(matrix4[2,1] == matrix3[2,1]);
            Assert.IsTrue(matrix4[2,2] == matrix3[2,2]);

        }
        [TestMethod]
        public void MatrixMultiplyComplex()
        {   
            NDArrayGeneric<Complex> matrix1 = new NDArrayGeneric<Complex>();
            matrix1.Data = new Complex[] {new Complex(1,-1),new Complex(2,-2), new Complex(3,0),new Complex(4,0), 5, 6}; 
            matrix1.Shape = new Shape(new int[] { 3, 2 });

            NDArrayGeneric<Complex> matrix2 = new NDArrayGeneric<Complex>();
            matrix2.Data = new Complex[] {7,8,9,new Complex(10,-10),11, new Complex(12,-12)};
            matrix2.Shape = new Shape(new int[] { 2, 3 });

            var matrix3 = matrix1.dot(matrix2);

            var matrix4 = new Complex[9];
            matrix4[0] = new Complex(7,-47);
            matrix4[1] = new Complex(30,-30);
            matrix4[2] = new Complex(9,-57);
            matrix4[3] = new Complex(61,-40);
            matrix4[4] = new Complex(68,0);
            matrix4[5] = new Complex(75,-48);
            matrix4[6] = new Complex(95,-60);
            matrix4[7] = new Complex(106,0);
            matrix4[8] = new Complex(117,-72);

            Assert.IsTrue(Enumerable.SequenceEqual(matrix4,matrix3.Data));
        }
        [TestMethod]
        public void DotTwo1DComplex()
        {
            var series1 = new NumPyGeneric<Complex>().array(new Complex[]{new Complex(0,2),new Complex(0,3)});
            
            var series2 = new NumPyGeneric<Complex>().array(new Complex[]{new Complex(0,2),new Complex(0,3)});
        
            //var series3 = series1.Dot(series2);

            //Complex[] expectedResult = new Complex[]{new Complex(-13,0)};

            //Assert.IsTrue(Enumerable.SequenceEqual(series3.Data.ToArray(),expectedResult));
            
        }
        
    }
}