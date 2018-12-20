using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Core;
using NumSharp.Core.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.Extensions
{
    [TestClass]
    public class NdArrayRandomRandIntTest
    {
        [TestMethod]
        public void randint()
        {
            var a = new NumSharp.Core.NumPyRandom().randint(low: 0, high: 10, shape: new Shape(5, 5));
            Assert.IsTrue(a.Storage.GetData<int>().Count(x => x < 10) == 25);
        }
    }
}
