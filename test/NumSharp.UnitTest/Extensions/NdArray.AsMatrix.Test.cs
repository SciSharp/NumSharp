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
    public class NdArrayAsMatrixTest 
    {
        [TestMethod]
        public void ConvertNDArrayNDArrayDouble()
        {
            var nd = np.arange(9).reshape(3,3);

            var npAsMatrix = nd.AsMatrix();

            for (int idx = 0; idx < 3;idx++)
            {
                for (int jdx = 0; jdx < 3; jdx++)
                {
                    Assert.AreEqual(nd[idx, jdx], npAsMatrix[idx, jdx]);
                }
            }
        }
    }
}
