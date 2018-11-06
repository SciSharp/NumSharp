using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

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
            var np = new NumPy<double>();
            var n1 = np.array(new double[] { 1, 2, 3 });
            var n2 = np.array(new double[] { 2, 3, 4 });

            var n = np.vstack(n1, n2);

            Assert.IsTrue(n.Size == (n1.Size + n2.Size));
            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[1, 2] == 4);

            //2D
            n1 = np.array(new double[][] { new double[] { 1 }, new double[] { 2 }, new double[] { 3 } });
            n2 = np.array(new double[][] { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });

            n = np.vstack(n1, n2);

            Assert.IsTrue(n.Size == (n1.Size + n2.Size));
            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[2, 0] == 3);
            Assert.IsTrue(n[3, 0] == 4);
            Assert.IsTrue(n[4, 0] == 5);
            Assert.IsTrue(n[5, 0] == 6);
        }
    }
}
