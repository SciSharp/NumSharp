using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumPyConcatenateTest
    {
        [TestMethod]
        public void axis0()
        {
            var a = new int[][] { new int[] { 1, 2 }, new int[] { 3, 4 } };
            var b = new int[][] { new int[] { 5, 6 } };

           var c = new float[][] { new float[] { 1.3f, 2.3f }, new float[] { 3.3f, 4.3f } };
           var d = new float[][] { new float[] { 5.3f, 5.3f } };

           var e = new double[][] { new double[] { 3.6, 8.7  }, new double[] { 5.7, 8.9 } };
           var f = new double[][] { new double[] { 5.3, 5.3 } };

            var nd = np.concatenate(new int[][][] { a, b }, axis: 0);
            Assert.AreEqual(nd.shape[0], 3);
            Assert.AreEqual(nd.shape[1], 2);
        }
    }
}
