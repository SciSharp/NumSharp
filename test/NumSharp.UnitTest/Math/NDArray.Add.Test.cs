using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class AddTest : TestClass
    {
        [TestMethod]
        public void UInt8AddTest1()
        {
            var nd1 = np.arange(3).astype(np.uint8);

            var nd2 = nd1 + 2;

            AssertAreEqual(new byte[] {2, 3, 4}, nd2.Data<byte>());
        }

        [TestMethod]
        public void UInt16AddTest1()
        {
            var nd1 = np.arange(3).astype(np.uint16);

            var nd2 = nd1 + 2;

            AssertAreEqual(new ushort[] {2, 3, 4}, nd2.Data<ushort>());
        }
    }
}
