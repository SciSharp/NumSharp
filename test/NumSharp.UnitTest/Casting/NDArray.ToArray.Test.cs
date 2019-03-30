using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using NumSharp;
using System.Linq;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class CastingTester
    {
        [TestMethod]
        public void ToDotNetArray()
        {
            var oneDArray = np.arange(10.0);
            var oneDArrayDotNet = (double[]) oneDArray.ToMuliDimArray<double>();

            var twoDArray = np.arange(8.0).reshape(2,4);
            var twoDArrayDotNet = (double[,]) twoDArray.ToMuliDimArray<double>();
        }

        [TestMethod]
        public void ToByteArray()
        {
            var nd = np.array(new int[][]
            {
                new int[]{ 3, 1},
                new int[]{ 2, 1 }
            });

            var bytes = nd.ToByteArray();
            Assert.IsTrue(Enumerable.SequenceEqual(new byte[]
            {
                3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0
            }, bytes));
        }
    }
}
