using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayZerosTest
    {
        [TestMethod]
        public void Zeros1Dim()
        {
            var np = new NDArray<int>();
            np.Zeros(3);

            // Assert.IsTrue(np.ToString().Equals("array([0, 0, 0])"));
        }

        [TestMethod]
        public void Zeros2Dim()
        {
            var np = new NDArray<int>();
            np.Zeros(3, 2);

            // Assert.IsTrue(np.ToString().Equals("array([[0, 0], [0, 0], [0, 0]])"));
        }
    }
}
