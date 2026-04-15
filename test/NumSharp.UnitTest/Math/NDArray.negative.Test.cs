using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using System.Numerics;
using NumSharp;

namespace NumSharp.UnitTest.Maths
{
    public class NDArrayNegativeTest
    {
        [TestMethod]
        public void Negative_FlipsSign()
        {
            // np.negative flips the sign of each element
            // Input:  [1, -2, 3.3]
            // Output: [-1, 2, -3.3]
            NDArray nd = new[] { 1f, -2f, 3.3f };
            nd = np.negative(nd);

            var data = nd.Data<float>();
            Assert.AreEqual(-1f, data[0], 1e-6f);
            Assert.AreEqual(2f, data[1], 1e-6f);   // -2 becomes 2
            Assert.AreEqual(-3.3f, data[2], 1e-5f);
        }

        [TestMethod]
        public void Negative_AllPositiveInput()
        {
            // When all inputs are positive, all outputs are negative
            NDArray nd = new[] { 1f, 2f, 3f };
            nd = np.negative(nd);
            Assert.IsTrue(nd.Data<float>().All(v => v < 0));
        }

        [TestMethod]
        public void Negative_AllNegativeInput()
        {
            // When all inputs are negative, all outputs are positive
            NDArray nd = new[] { -1f, -2f, -3f };
            nd = np.negative(nd);
            Assert.IsTrue(nd.Data<float>().All(v => v > 0));
        }
    }
}
