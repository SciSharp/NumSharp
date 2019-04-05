using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.View
{
    [TestClass]
    public class NDArrayViewTest
    {
        [TestMethod]
        public void ValueTest()
        {
            var x = np.arange(3);
            var v = x.view();

            v[0] = 1;

            Assert.AreEqual(x[0], v[0]);
        }
    }
}
