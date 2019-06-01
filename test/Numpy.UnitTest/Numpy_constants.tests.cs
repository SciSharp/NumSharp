using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Numpy;
using Numpy.Models;
using Assert = NUnit.Framework.Assert;

namespace Numpy.UnitTest
{
    [TestClass]
    public class NumpyConstants
    {
        [TestMethod]
        public void infTest()
        {
            //>>> np.inf
            //inf
            //>>> np.array([1]) / 0.
            //array([Inf])
            Console.WriteLine(np.inf);
            var x = np.array(1) / 0.0;
            Assert.AreEqual(np.array(np.inf), x);
            Assert.AreNotEqual(np.array(0f), x);
            Assert.AreEqual(float.PositiveInfinity, np.inf);
        }

        [TestMethod]
        public void ninfTest()
        {
            //>>> np.NINF
            //-inf
            //>>> np.log(0)
            //-inf
            Assert.AreEqual(-np.inf, np.NINF);
            Assert.AreEqual(-np.inf, (float) np.log((NDarray) 0));
        }

        [TestMethod]
        public void NZERO_Test()
        {
            //>>> np.NZERO
            //-0.0
            //>>> np.PZERO
            //0.0
            //>>>
            //>>> np.isfinite([np.NZERO])
            //array([True])
            //>>> np.isnan([np.NZERO])
            //array([False])
            //>>> np.isinf([np.NZERO])
            //array([False])
            Assert.AreEqual(-0.0f, np.NZERO);
            Assert.AreEqual(0.0f, np.PZERO);
            Assert.IsTrue((bool) np.isfinite((NDarray) np.NZERO));
            Assert.IsFalse((bool) np.isnan((NDarray) np.NZERO));
            Assert.IsFalse((bool) np.isinf((NDarray) np.NZERO));
        }

        [TestMethod]
        public void nanTest()
        {
            //>>> np.nan
            //nan
            //>>> np.log(-1)
            //nan
            //>>> np.log([-1, 1, 2])
            //array([NaN,  0.        ,  0.69314718])
            Assert.AreEqual(float.NaN, np.nan);
            Assert.AreEqual(np.nan, (float)np.log((NDarray)(-1)));
            Assert.AreEqual("array([       nan, 0.        , 0.69314718])", np.log(new[]{-1, 1, 2}).repr);
        }

        [TestMethod]
        public void newaxisTest()
        {
            //>>> newaxis is None
            //True
            //>>> x = np.arange(3)
            //>>> x
            //array([0, 1, 2])
            //>>> x[:, newaxis]
            //array([[0],
            //[1],
            //[2]])
            //>>> x[:, newaxis, newaxis]
            //array([[[0]],
            //[[1]],
            //[[2]]])
            //>>> x[:, newaxis] * x
            //array([[0, 0, 0],
            //[0, 1, 2],
            //[0, 2, 4]])
            //Outer product, same as outer(x, y) :

            //>>>
            //>>> y = np.arange(3, 6)
            //>>> x[:, newaxis] * y
            //array([[ 0,  0,  0],
            //[ 3,  4,  5],
            //[ 6,  8, 10]])
            //x[newaxis, :] is equivalent to x[newaxis] and x[None]:

            //>>>
            //>>> x[newaxis, :].shape
            //(1, 3)
            //>>> x[newaxis].shape
            //(1, 3)
            //>>> x[None].shape
            //(1, 3)
            //>>> x[:, newaxis].shape
            //(3, 1)
        }

    }
}
