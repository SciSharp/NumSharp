using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Creation
{
    [TestClass]
    public class NpFromBufferTest
    {
        [TestMethod]
        public void ToInt32()
        {
            int[] ints = {100, 200, 300, 400, 500};
            byte[] bytes = new byte[ints.Length * sizeof(int)];
            Buffer.BlockCopy(ints, 0, bytes, 0, bytes.Length);

            var nd = np.frombuffer(bytes, np.int32);
            Assert.AreEqual(nd.GetInt32(0), 100);
            Assert.AreEqual(nd.GetInt32(4), 500);
        }
    }
}
