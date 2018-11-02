using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDArrayTest
    {
        [TestMethod]
        public void IndexAccessor()
        {
            var np = new NumPy<int>();
            var n = np.arange(12).reshape(3, 4);

            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 0] == 8);

            n = np.arange(12).reshape(2, 3, 2);
            var n1 = n.Vector(1);

            Assert.IsTrue(n1[1, 1] == 9);
            Assert.IsTrue(n1[2, 1] == 11);

            var n2 = n.Vector(1, 2);

            Assert.IsTrue(n2[0] == 10);
            Assert.IsTrue(n2[1] == 11);
        }
    }
}
