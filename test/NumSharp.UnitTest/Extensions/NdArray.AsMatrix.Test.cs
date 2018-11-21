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
            var np = new NDArrayGeneric<double>().arange(9).reshape(3,3);

            var npAsMatrix = np.AsMatrix();

            for (int idx = 0; idx < 3;idx++)
            {
                for(int jdx = 0; jdx < 3; jdx++)
                {
                    Assert.AreEqual(np[idx,jdx],npAsMatrix[idx,jdx]);
                }
            }
        }
    }
}
