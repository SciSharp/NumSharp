using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class MultiplyTest : TestClass
    {
        [TestMethod]
        public void Int32MultiplyTest1()
        {
            var nd1 = np.arange(3);

            var nd2 = nd1 * 2;

            AssertAreEqual(new int[] {0, 2, 4}, nd2.Data<int>());
        }
        [TestMethod]
        public void Int32MultiplyTest2()
        {
            var nd1 = np.arange(3);

            var nd2 = nd1 * 2;

            AssertAreEqual(new int[] {0, 2, 4}, nd2.Data<int>());
        }
    }
}
