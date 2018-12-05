using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Tests following https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.hstack.html
    /// </summary>
    [TestClass]
    public class NdArrayHStackTest
    {
        [TestMethod]
        public void HStackNDArrays()
        {
            //1D
            var np = new NumPyGeneric<double>();
            var n1 = np.array(new double[] { 1, 2, 3 });
            var n2 = np.array(new double[] { 2, 3, 4 });

            var n = np.hstack(n1, n2);

            Assert.IsTrue(n.Size == (n1.Size + n2.Size));
            Assert.IsTrue(n[0] == 1);
            Assert.IsTrue(n[1] == 2);
            Assert.IsTrue(n[3] == 2);
            Assert.IsTrue(n[5] == 4);

            //2D
            n1 = np.array(new double[][] { new double[] { 1 }, new double[] { 2 }, new double[] { 3 } });
            n2 = np.array(new double[][] { new double[] { 4 }, new double[] { 5 }, new double[] { 6 } });

            n = np.hstack(n1, n2);

            Assert.IsTrue(n.Size == (n1.Size + n2.Size));
            Assert.IsTrue(n[0, 0] == 1);
            Assert.IsTrue(n[1, 0] == 2);
            Assert.IsTrue(n[2, 0] == 3);
            Assert.IsTrue(n[0, 1] == 4);
            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 1] == 6);

            //3D
            n1 = np.array(new double[] {0,1,2,3,4,5,6,7,8,9,10,11} ).reshape(2, 3, 2);
            n2 = np.array(new double[] {0,1,2,3,4,5,6,7,8,9,10,11} ).reshape(2, 3, 2);
            n = np.hstack(n1, n2);

            Assert.IsTrue(n.Size == (n1.Size + n2.Size));
            Assert.IsTrue(n[0, 0, 0] == 0);
            Assert.IsTrue(n[0, 1, 0] == 2);
            Assert.IsTrue(n[0, 2, 1] == 5);

            //4D
            //n1 = np.arange(24 * 10000000).ReShape(20000, 30, 20, 2);
            //n2 = np.arange(24 * 10000000).ReShape(20000, 30, 20, 2);
            //n = np.hstack(n1, n2);

            //Assert.IsTrue(n.Size == (n1.Size + n2.Size));

        }
    }
}
