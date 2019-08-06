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
    public class NdArrayArgSortTest
    {
        [TestMethod]
        public void OneDimension()
        {
            var x = np.array(new int[] {3, 1, 2});
            x = x.argsort<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] {1, 2, 0}, x.Data<int>()));
        }

        [TestMethod]
        public void TwoDimension()
        {
            var x = np.array(new int[] {3, 1, 2});
        }
    }
}
