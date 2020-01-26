using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayGreaterOrLessTest
    {
        [TestMethod]
        public void BoolTwoScalar_NDArray()
        {
            NDArray np1 = 1;
            NDArray np2 = 2;
            bool np3 = np1 > np2;
            Assert.IsTrue(!np3);

            np1 = 3;
            np2 = 2;
            np3 = np1 > np2;
            Assert.IsTrue(np3);

            np3 = np2 < np1;
            Assert.IsTrue(np3);
        }

        [TestMethod]
        public void BoolTwo1D_NDArray()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }, new Shape(4));
            var np2 = new NDArray(new[] { 4, 3, 2, 1 }, new Shape(4));

            var np3 = np1 > np2;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { false, false, true, true }, np3.Data<bool>()));

            np3 = np2 < np1;
            Assert.IsTrue(Enumerable.SequenceEqual(new[] { false, false, true, true }, np3.Data<bool>()));
        }
    }
}
