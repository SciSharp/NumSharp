using System;
using System.Runtime.CompilerServices;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;
using static NumSharp.Slice;

namespace NumSharp.UnitTest.Manipulation
{
    [TestClass]
    public class NdArrayToStringTest : TestClass
    {
        [TestMethod]
        public void ReShape()
        {
            var nd = np.arange(6);
            var n1 = np.reshape(nd, 3, 2);
            var n = n1.MakeGeneric<int>();

            Assert.IsTrue(n[0, 0] == 0);
            Assert.IsTrue(n[1, 1] == 3);
            Assert.IsTrue(n[2, 1] == 5);

            n = np.reshape(np.arange(6), 2, 3, 1).MakeGeneric<int>();
            Assert.IsTrue(n[1, 1, 0] == 4);
            Assert.IsTrue(n[1, 2, 0] == 5);

            n = np.reshape(np.arange(12), 2, 3, 2).MakeGeneric<int>();
            Assert.IsTrue(n[0, 0, 1] == 1);
            Assert.IsTrue(n[1, 0, 1] == 7);
            Assert.IsTrue(n[1, 1, 0] == 8);

            n = np.reshape(np.arange(12), 3, 4).MakeGeneric<int>();
            Assert.IsTrue(n[1, 1] == 5);
            Assert.IsTrue(n[2, 0] == 8);

            n = np.reshape(n, 2, 6).MakeGeneric<int>();

            Assert.IsTrue(n[1, 0] == 6);
        }
    }
}
