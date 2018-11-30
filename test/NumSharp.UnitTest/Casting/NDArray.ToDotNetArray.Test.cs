using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class CastingTester
    {
        NumPy np = new NumPy();
        [TestMethod]
        public void ToDotNetArray()
        {
            var oneDArray = np.arange(10.0);
            var oneDArrayDotNet = (double[]) oneDArray.ToMuliDimArray<double>();

            var twoDArray = np.arange(8.0).reshape(2,4);
            var twoDArrayDotNet = (double[,]) twoDArray.ToMuliDimArray<double>();

        }
    }
}
