using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using NumSharp.Core;

namespace NumSharp.UnitTest.Docs
{
    [TestClass]
    public class NDArrayCreationTester
    {
        [TestMethod]
        public void Dump()
        {
            var np = new NDArray(typeof(double));
            np.Storage.SetData(new double[] {1,2,3,4,5,6,7,8,9});
            np.Storage.Shape = new Shape(3,3);

            Assert.IsTrue(np.ndim == 2);    

            var npGen = np.MakeGeneric<double>();

            Assert.IsTrue(npGen[0,0] == 1);
            Assert.IsTrue(npGen[1,1] == 5);
            Assert.IsTrue(npGen[2,2] == 9);
        }
    }
}
