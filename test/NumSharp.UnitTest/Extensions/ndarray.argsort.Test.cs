using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    /// <summary>
    /// Tests following https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    /// </summary>
    public class NdArrayArgSortTest
    {
        [Test]
        public void OneDimension()
        {
            var x = np.array(new int[] {3, 1, 2});
            x = x.argsort<int>();
            Assert.IsTrue(Enumerable.SequenceEqual(new int[] {1, 2, 0}, x.Data<int>()));
        }

        [Test]
        public void TwoDimension()
        {
            var x = np.array(new int[] {3, 1, 2});
        }
    }
}
