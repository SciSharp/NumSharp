using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Core.Extensions;
using System.Linq;
using NumSharp.Core;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class NDStorageTest
    {
        [TestMethod]
        public void Index()
        {
            var shape0 = new Shape(4,3);

            int idx0 = shape0.GetIndexInShape(2,1);
            int[] idx00 = shape0.GetDimIndexOutShape(idx0);

            Assert.IsTrue(idx00[0] == 2);
            Assert.IsTrue(idx00[1] == 1);

        }
    }
}
