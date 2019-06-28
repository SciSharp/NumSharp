using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Maths
{
    [TestClass]
    public class MatrixOperationTest 
    {
        [TestMethod]
        public void CheckToString()
        {
            var matrix = new matrix("1 2 3;4 5 6;7 8 9", np.float64);

            string matrixAsString = matrix.ToString();

            Assert.IsTrue(matrixAsString.Contains("matrix([[1, 2, 3]"));
            Assert.IsTrue(matrixAsString.Contains("[4, 5, 6]"));
            Assert.IsTrue(matrixAsString.Contains("[7, 8, 9]"));
        }

        [TestMethod]
        public void FloatSubtraction()
        {
            var np1 = np.array(new float[] { 3, 5, 7 });

            var np2 = 0f - np1;

            Assert.IsTrue(Enumerable.SequenceEqual(new float[] { -3, -5, -7 }, np2.Data<float>()));
        }

        [TestMethod]
        public void DoubleTwo2D_MatrixAddition()
        {
            var np1 = new matrix("1 2 3;4 5 6;7 8 9", np.float64);
            var np2 = new matrix("1 2 3;4 5 6;7 8 9", np.float64);

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 4, 6, 8, 10, 12, 14, 16, 18 }, np3.Data<double>()));
        }

        [TestMethod]
        public void DoubleTwo2D_MatrixSubstraction()
        {
            var np1 = new matrix("1 2 3;4 5 6;7 8 9", np.float64);
            var np2 = new matrix("1 2 3;4 5 6;7 8 9", np.float64);

            var np3 = np1 - np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 }, np3.Data<double>()));
        }

        [TestMethod]
        public void DoubleTwo1D_NDArrayAddition()
        {
            var nd1 = np.arange(1, 4);
            var nd2 = np.arange(2, 5);

            var nd3 = nd1 + nd2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 3, 5, 7 }, nd3.Data<int>()));
        }

        /*[TestMethod]
        public void ComplexTwo1D_NDArrayAddition()
        {
            var data1 = new Complex[] { new Complex(1, 2), new Complex(3, 4) };
            var data2 = new Complex[] { new Complex(5, 6), new Complex(7, 8) };

            var np1 = new NDArray(typeof(Complex),2);
            
            np1.Storage.ReplaceData(data1);

            var np2 = new NDArray(typeof(Complex),2);
            
            np2.Storage.ReplaceData(data2);

            var np3 = np1 + np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new Complex[] { new Complex(6, 8), new Complex(10, 12) }, np3.Storage.GetData<Complex>()));

        }*/

        [TestMethod]
        public void Double1DPlusOffset_NDArrayAddition()
        {
            var np1 = np.arange(4);

            var np3 = np1 + 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 2, 3, 4, 5 }, np3.Data<int>()));
        }

        /*[TestMethod]
        public void Complex1DPlusOffset_NDArrayAddition()
        {
            var np1 = new NDArray(typeof(Complex),2);
            np1.Storage.ReplaceData(new Complex[] { new Complex(1, 2), new Complex(3, 4) });
            
            var np2 = np1 + new Complex(1, 2);

            Assert.IsTrue(Enumerable.SequenceEqual(new Complex[] { new Complex(2, 4), new Complex(4, 6) }, np2.Storage.GetData<Complex>()));
        }

        [TestMethod]
        public void Complex2DArray_NDArrayAddition()
        {
            var np1 = new NDArray(typeof(Complex), new Shape(2, 2));
            np1.Storage.ReplaceData(new Complex[] { new Complex(4, 1), new Complex(3, 5), new Complex(0, -2), new Complex(-3, 2) });


            var np2 = new NDArray(typeof(Complex), new Shape(2, 2));
            np2.Storage.ReplaceData(new Complex[] { new Complex(1, 2), new Complex(3, 4), new Complex(1, 2), new Complex(3, 4) });
            
            var np3 = np1 + np2;

            // expected
            var np4 = new Complex[] { new Complex(5, 3), new Complex(6, 9), new Complex(1, 0), new Complex(0, 6) };

            Assert.IsTrue(Enumerable.SequenceEqual(np3.Storage.GetData<Complex>(), np4));
        }*/

        [TestMethod]
        public void Double2DArray_NDArrayAddition()
        {
            var np1 = new NDArray(typeof(double), new Shape(2, 3));
            np1.ReplaceData(new double[] { 1, 2, 3, 4, 5, 6 });

            var np2 = new NDArray(typeof(double), new Shape(2, 3));
            np2.ReplaceData(new double[] { 9, 8, 7, 6, 5, 4 });

            var np3 = np1 + np2;

            // expected
            var np4 = new double[] { 10, 10, 10, 10, 10, 10 };

            Assert.IsTrue(Enumerable.SequenceEqual(np3.Data<double>(), np4));
        }

        [TestMethod]
        public void DoubleTwo1D_NDArraySubtraction()
        {
            var np1 = np.array(new double[] { 3, 5, 7 });
            var np2 = np.array(new double[] { 1, 3, 4 });

            var np3 = np1 - np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 2, 3 }, np3.Data<double>()));
        }

        [TestMethod]
        public void Double1DPlusOffset_NDArraySubtraction()
        {
            var np1 = np.array(new double[] { 3, 5, 7 });

            var np2 = np1 - 3d;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 0, 2, 4 }, np2.Data<double>()));
        }

        [TestMethod]
        public void DoubleTwo1D_NDArrayMultiplication()
        {
            var np1 = np.arange(1, 4, 1).reshape(1, 3);
            var np2 = np.arange(2, 5, 1).reshape(1, 3);

            var np3 = np1 * np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 6, 12 }, np3.Data<double>()));
        }

        [TestMethod]
        public void Double1DPlusOffset_NDArrayMultiplication()
        {
            var np1 = new NDArray(new double[] { 1, 2, 3 });
            var np3 = np1 * 2d;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 4, 6 }, np3.Data<double>()));
        }
    }
}
