using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NumPyBoradCcastArrayTest
    {
        [TestMethod]
        public void BroadcastArrayTest()
        {
            var arr1 = new int[][] {new int[] {1, 2, 3}};
            NDArray nd1 = np.array(arr1);
            var arr2 = new int[][] {new int[] {4}, new int[] {5}};
            NDArray nd2 = np.array(arr2);

            NDArray[] a = np.broadcast_arrays(nd1, nd2);
            NDArray b = a[0];
            NDArray c = a[1];

            Assert.IsTrue(b.Data<int>(0, 0) == 1.0);
            Assert.IsTrue(b.Data<int>(0, 1) == 2.0);
            Assert.IsTrue(b.Data<int>(0, 2) == 3.0);
            Assert.IsTrue(c.Data<int>(0, 0) == 4.0);
            Assert.IsTrue(c.Data<int>(0, 1) == 4.0);
            Assert.IsTrue(c.Data<int>(0, 2) == 4.0);
            Assert.IsTrue(b.size == 6);
            Assert.IsTrue(c.size == 6);
        }
    }
}
