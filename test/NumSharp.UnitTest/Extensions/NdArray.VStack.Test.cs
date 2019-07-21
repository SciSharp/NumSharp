using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html
    /// </summary>
    [TestClass]
    public class NdArrayVStackTest
    {
        [TestMethod]
        public void VStackNDArrays()
        {
            //1D
            var n1 = np.array(new double[] { 1, 2, 3 });
            var n2 = np.array(new double[] { 2, 3, 4 });

            var n = np.vstack<double>(n1, n2).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size));

            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[1, 2] == 4);

            //2D
            var n01 = np.array(new double[][] { new double[] { 1 }, new double[] { 2 }, new double[] { 3 } });
            var n02 = np.array(new double[][] { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });

            var n0 = np.vstack<double>(n01, n02).MakeGeneric<double>();

            Assert.IsTrue(n.size == (n1.size + n2.size));
            
            Assert.IsTrue(n0[0, 0] == 1);
            Assert.IsTrue(n0[1, 0] == 2);
            Assert.IsTrue(n0[2, 0] == 3);
            Assert.IsTrue(n0[3, 0] == 4);
            Assert.IsTrue(n0[4, 0] == 5);
            Assert.IsTrue(n0[5, 0] == 6);
        }
    }
}
