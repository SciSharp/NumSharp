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
    public class NDArrayNegateTest : TestClass
    {
        [TestMethod]
        public void NegateArray()
        {
            //initialization
            var nd = new NDArray(np.float32, 3);
            nd.ReplaceData(new float[] {1, -2, 3.3f});

            //perform test
            nd = -nd;

            //assertions
            var result = nd.Data<float>();
            AssertAreEqual(new float[] {-1, 2, -3.3f}, result);
        }

        [TestMethod]
        public void AddArray()
        {
            //initialization
            var nd = new NDArray(np.float32, 3);
            var input = new float[] {1, -2, 3.3f};
            nd.ReplaceData(input);

            //perform test
            nd = +nd;

            //assertions
            var result = nd.Data<float>();
            AssertAreEqual(input, result);
        }

        [TestMethod]
        public void NegateEmptyArray()
        {
            //initialization
            var nd1 = new NDArray(np.float32, 0);

            //perform test
            var nd2 = -nd1;

            //assertions
            Assert.IsTrue(nd1.size == nd2.size);
        }
    }
}
