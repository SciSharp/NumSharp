using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Docs
{
    [TestClass]
    public class NDArrayCreationTester
    {
        [TestMethod]
        public void Dump()
        {
            var np = new NDArray<double>();
            np.Data = new double[] {1,2,3,4,5};
            np.Shape = new Shape(np.Data.Length);

            Assert.IsTrue(np.NDim == 1);            
        }
    }
}
