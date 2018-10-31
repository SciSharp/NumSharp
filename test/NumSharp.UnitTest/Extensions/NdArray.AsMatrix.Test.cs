using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayAsMatrixTest
    {
        [TestMethod]
        public void ConvertNDArrayNDArrayDouble()
        {
            var np = new NDArray_Legacy<NDArray_Legacy<double>>().Array(new double[,]{{1,2,3},{4,5,6},{7,8,9}});

            var npAsMatrix = np.AsMatrix();

            for (int idx = 0; idx < 3;idx++)
            {
                for(int jdx = 0; jdx < 3; jdx++)
                {
                    Assert.AreEqual(np[idx][jdx],npAsMatrix.Data[idx,jdx]);
                }
            }
        }
    }
}
