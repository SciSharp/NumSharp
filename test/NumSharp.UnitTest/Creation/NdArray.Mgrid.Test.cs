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
    public class NdArrayMGridTest
    {
        [TestMethod]
        public void BaseTest()
        {
            var X = np.arange(1, 11, 2).mgrid(np.arange(-12, -3, 3));

            NDArray x = X.Item1;
            NDArray y = X.Item2;

            
        }

    }

}