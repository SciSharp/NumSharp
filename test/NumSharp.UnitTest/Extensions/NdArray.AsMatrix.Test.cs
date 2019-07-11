using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayAsMatrixTest
    {
        [TestMethod]
        public void ConvertNDArrayNDArrayDouble()
        {
            var nd = np.arange(9).reshape(3, 3).MakeGeneric<int>();

            var npAsMatrix = nd.AsMatrix().MakeGeneric<int>();

            for (int idx = 0; idx < 3; idx++)
            {
                for (int jdx = 0; jdx < 3; jdx++)
                {
                    Assert.AreEqual(nd[idx, jdx], npAsMatrix[idx, jdx]);
                }
            }
        }
    }
}
