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
    public class NdArrayDotTest : TestBase
    {
        [TestMethod]
        public void DotTwo1Int()
        {
            var X = np.array(new int[][]
            {
                new int[] { 1, 1 },
                new int[] { 1, 2 },
                new int[] { 2, 2 },
                new int[] { 2, 3 }
            });

            var y = np.dot(X, np.array(new int[] { 2, 3 }));

            var yArray = y.Storage.GetData<int>();

            Assert.AreEqual(yArray[0], 5);
            Assert.AreEqual(yArray[1], 8);
            Assert.AreEqual(yArray[2], 10);
            Assert.AreEqual(yArray[3], 13);
        }

        [TestMethod]
        public void DotTwo1DDouble()
        {
            
            var series1 = np.arange(1.0,4.0,1.0) ;
            
            var series2 = new NDArray(typeof(int),new Shape(3));
            series2.Storage.SetData(new double[]{0, 1, 0.5});
            
            var innerProduct = series1.dot(series2);
            Assert.IsTrue(innerProduct.Storage.GetData<double>()[0] == 3.5);
        }
        [TestMethod]
        public void MatrixMutliplyDifferentDataLength()
        {
            var A = np.arange(6).reshape(3,2);

            var B = np.arange(14).reshape(2,7);

            var C = A.dot(B);

            Assert.IsTrue(Enumerable.SequenceEqual(new double[]{7,8,9,10,11,12,13,21,26,31,36,41,46,51,35,44,53,62,71,80,89},C.Storage.GetData<double>()));

        }
        [TestMethod]
        public void MatrixMultiplyDouble()
        {   
            var matrix1 = np.arange(1,7,1).reshape(3,2);
            
            var matrix2 = np.arange(7,13,1).reshape(2,3) ;
        
            var matrix3 = matrix1.dot(matrix2);
            
            var matrix4 = new int[3,3];

            for (int idx = 0; idx < 3; idx++)
            {
                for (int jdx = 0; jdx < 3;jdx++)
                {
                    matrix4[idx,jdx] = 0;
                    for (int kdx = 0; kdx < 2;kdx++)
                    {
                        matrix4[idx,jdx] += ((int)matrix1[idx,kdx] * (int) matrix2[kdx,jdx]);
                    }
                }
            }

            Assert.IsTrue(matrix4[0,0] == (int)matrix3[0,0]);
            Assert.IsTrue(matrix4[0,1] == (int)matrix3[0,1]);
            Assert.IsTrue(matrix4[0,2] == (int)matrix3[0,2]);
            Assert.IsTrue(matrix4[1,0] == (int)matrix3[1,0]);
            Assert.IsTrue(matrix4[1,1] == (int)matrix3[1,1]);
            Assert.IsTrue(matrix4[1,2] == (int)matrix3[1,2]);
            Assert.IsTrue(matrix4[2,0] == (int) matrix3[2,0]);
            Assert.IsTrue(matrix4[2,1] == (int) matrix3[2,1]);
            Assert.IsTrue(matrix4[2,2] == (int) matrix3[2,2]);
            
        }
        [TestMethod]
        public void MatrixMultiplyComplex()
        {   
            var matrix1 = new NDArray(typeof(Complex),3,2);
            matrix1.Storage.SetData(new Complex[] {new Complex(1,-1),new Complex(2,-2), new Complex(3,0),new Complex(4,0), 5, 6}); 
            
            var matrix2 = new NDArray(typeof(Complex),2,3);
            matrix2.Storage.SetData(new Complex[] {7,8,9,new Complex(10,-10),11, new Complex(12,-12)});
            
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

            Assert.IsTrue(Enumerable.SequenceEqual(matrix4,matrix3.Storage.GetData<Complex>()));
        }
        [TestMethod]
        public void DotTwo1DComplex()
        {
            var series1 = np.array(new Complex[]{new Complex(0,2),new Complex(0,3)});
            
            var series2 = np.array(new Complex[]{new Complex(0,2),new Complex(0,3)});
        
            var series3 = series1.dot(series2);

            Complex[] expectedResult = new Complex[]{new Complex(-13,0)};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Storage.GetData<Complex>(),expectedResult));
            
        }
        
    }
}