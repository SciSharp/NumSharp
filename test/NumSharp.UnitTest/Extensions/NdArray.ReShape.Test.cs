using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayReShapeTest
    {
        [TestMethod]
        public void ReShape()
        {
            var np = new NDArray<int>();
            var np2 = np.ARange(6).ReShape(3, 2);

            Assert.IsTrue(np2.ToString().Equals("array([[0, 1], [2, 3], [4, 5]])"));
        }
    }
}
