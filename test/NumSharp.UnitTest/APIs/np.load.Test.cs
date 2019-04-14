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
            int[] a = { 1, 2, 3, 4, 5 };
            byte[] mem = np.Save(a);

            int[] b = np.Load<int[]>(mem);
        }
    }
}
