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
    public class NdArrayRandomShuffleTest
    {
        [TestMethod]
        public void Base1DTest()
        {
            var nd = np.arange(10);

            nd[8] = 5;

            np.random.shuffle(nd);

            Assert.IsNotNull(nd);
        }
    }
}
