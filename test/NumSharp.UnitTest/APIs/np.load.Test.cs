using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class NumpyLoad
    {
        [TestMethod]
        public void NumpyLoadTest()
        {
            int[] a = {1, 2, 3, 4, 5};
            byte[] mem = np.Save(a);

            int[] b = np.Load<int[]>(mem);
        }

        [TestMethod]
        public void NumpyLoad1DimTest()
        {
            int[] arr = np.Load<int[]>(@"data/1-dim-int32_4_comma_empty.npy");
            Assert.IsTrue(arr[0] == 0);
            Assert.IsTrue(arr[1] == 1);
            Assert.IsTrue(arr[2] == 2);
            Assert.IsTrue(arr[3] == 3);
        }
    }
}
