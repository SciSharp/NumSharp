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
    [TestClass]
    public class NDArrayPositiveTest
    {
        [TestMethod]
        public void Positive_IsIdentity()
        {
            // np.positive is an identity function - returns unchanged values
            // Input:  [1, -2, 3.3]
            // Output: [1, -2, 3.3]
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] { 1f, -2f, 3.3f });
            nd = np.positive(nd);

            var data = nd.Data<float>();
            Assert.AreEqual(1f, data[0], 1e-6f);
            Assert.AreEqual(-2f, data[1], 1e-6f);  // -2 stays -2
            Assert.AreEqual(3.3f, data[2], 1e-5f);
        }

        [TestMethod]
        public void Positive_PreservesNegatives()
        {
            // np.positive preserves negative values (identity function)
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] { -1f, -2f, -3f });
            nd = np.positive(nd);

            // All values should still be negative
            Assert.IsTrue(nd.Data<float>().All(v => v < 0));
        }

        [TestMethod]
        public void Positive_PreservesPositives()
        {
            // np.positive preserves positive values (identity function)
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] { 1f, 2f, 3f });
            nd = np.positive(nd);

            Assert.IsTrue(nd.Data<float>().All(v => v > 0));
        }
    }
}
