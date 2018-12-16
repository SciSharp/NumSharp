using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NumPyZerosTest
    {
        [TestMethod]
        public void zero()
        {
            var n = np.zeros(3);

            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<float>(), new float[] { 0, 0, 0 }));

            Assert.IsTrue(Enumerable.SequenceEqual(n.Storage.GetData<int>(), new int[] { 0, 0, 0 }));
        }
    }
}
