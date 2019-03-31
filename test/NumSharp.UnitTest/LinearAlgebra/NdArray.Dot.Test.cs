using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    /// Test concolve with standard example from 
    /// https://www.numpy.org/devdocs/reference/generated/numpy.convolve.html
    /// </summary>
    [TestClass]
    public class NdArrayDotTest 
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

            var Y = np.array(new int[] { 2, 3 });

            var y = np.dot(X, Y);

            var yArray = y.Storage.GetData<int>();

            Assert.AreEqual(yArray[0], 5);
            Assert.AreEqual(yArray[1], 8);
            Assert.AreEqual(yArray[2], 10);
            Assert.AreEqual(yArray[3], 13);
        }

        [TestMethod]
        public void DotTwoScalar()
        {
            int sca1 = 2;
            int sca2 = 3;
            int sca3 = np.dot(sca1, sca2);

            Assert.AreEqual(sca3, 6);
        }

        [TestMethod]
        public void DotTwo1DInt()
        {
            var nd1 = np.arange(3);
            var nd2 = np.arange(3, 6);

            int nd3 = np.dot(nd1, nd2);
            Assert.IsTrue(nd3 == 14);
        }

        //[TestMethod]
        public void MatrixMutliplyDifferentDataLength()
        {
            var A = np.arange(6).reshape(3,2);

            var B = np.arange(14).reshape(2,7);

            var C = np.dot(A, B);

            Assert.IsTrue(Enumerable.SequenceEqual(new double[]{7,8,9,10,11,12,13,21,26,31,36,41,46,51,35,44,53,62,71,80,89},C.Storage.GetData<double>()));

        }

        //[TestMethod]
        public void MatrixMultiplyDouble()
        {   
            var matrix1 = np.arange(1,7,1).reshape(3,2);
            
            var matrix2 = np.arange(7,13,1).reshape(2,3);
        
            var matrix3 = np.dot(matrix1, matrix2);
            
            var matrix4 = new int[3,3];

            for (int idx = 0; idx < 3; idx++)
            {
                for (int jdx = 0; jdx < 3;jdx++)
                {
                    matrix4[idx,jdx] = 0;
                    for (int kdx = 0; kdx < 2;kdx++)
                    {
                        matrix4[idx,jdx] += ((int)matrix1[idx,kdx] * (int)matrix2[kdx,jdx]);
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

        //[TestMethod]
        public void DotTwo1DComplex()
        {
            var series1 = np.array(new Complex[]{new Complex(0,2),new Complex(0,3)});
            
            var series2 = np.array(new Complex[]{new Complex(0,2),new Complex(0,3)});
        
            var series3 = np.dot(series1, series2);

            Complex[] expectedResult = new Complex[]{new Complex(-13,0)};

            Assert.IsTrue(Enumerable.SequenceEqual(series3.Storage.GetData<Complex>(),expectedResult));
        }
        
    }
}