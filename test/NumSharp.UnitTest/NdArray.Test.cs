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
            var np = new NDArray<int>();
            np.ARange(12).ReShape(3, 4);

            // get test
            Assert.IsTrue(np[1, 1] == 5);
            Assert.IsTrue(np[2, 0] == 8);

            //Assert.IsTrue(np[1, 1] == 10);
        }
    }
}
