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
    public class NDArrayMultiplicationTest
    {
        [TestMethod]
        public void DoubleTwo1D_NDArrayMultiplication()
        {
            var np1 = new NDArray(typeof(double)).arange(4, 1, 1).reshape(1, 3);
            var np2 = new NDArray(typeof(double)).arange(5, 2, 1).reshape(1, 3);

            var np3 = np1 * np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 6, 12 }, np3.Storage.GetData<double>()));
        }

        [TestMethod]
        public void Double1DPlusOffset_NDArrayMultiplication()
        {
            var np1 = new NDArray(typeof(double),3);
            np1.Storage.SetData(new double[] { 1, 2, 3 });

            var np3 = np1 * 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new double[] { 2, 4, 6 }, np3.Storage.GetData<double>()));
        }
    }
}
