using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NumSharp.Extensions;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class astypeTests
    {
        [TestMethod]
        public void Upcasting()
        {
            var nd = np.ones(np.int32, 3, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array!=nd.Array);
            Assert.IsTrue(int64.Array==nd.Array);
            Assert.IsTrue(int64_copied.Array.GetType().GetElementType() == typeof(Int64));
            Assert.IsTrue(int64.Array.GetType().GetElementType() == typeof(Int64));
        }        
        
        [TestMethod]
        public void UpcastingByteToLong()
        {
            var nd = np.ones(np.uint8, 3, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array!=nd.Array);
            Assert.IsTrue(int64.Array==nd.Array);
            Assert.IsTrue(int64_copied.Array.GetType().GetElementType() == typeof(Int64));
            Assert.IsTrue(int64.Array.GetType().GetElementType() == typeof(Int64));
        }        
        
        [TestMethod]
        public void UpcastingCharsToLong()
        {
            var nd = np.ones(np.chars, 3, 3, 3);
            var int64_copied = nd.astype(np.int64, true);
            var int64 = nd.astype(np.int64, false);

            //test copying
            Assert.IsTrue(int64_copied.Array!=nd.Array);
            Assert.IsTrue(int64.Array==nd.Array);
            Assert.IsTrue(int64_copied.Array.GetType().GetElementType() == typeof(Int64));
            Assert.IsTrue(int64.Array.GetType().GetElementType() == typeof(Int64));
        }        
        
        [TestMethod]
        public void DowncastingIntToShort()
        {
            var nd = np.ones(np.int32, 3, 3, 3);
            var int16_copied = nd.astype(np.int16, true);
            var int16 = nd.astype(np.int16, false);

            //test copying
            Assert.IsTrue(int16_copied.Array!=nd.Array);
            Assert.IsTrue(int16.Array==nd.Array);
            Assert.IsTrue(int16_copied.Array.GetType().GetElementType() == typeof(Int16));
            Assert.IsTrue(int16.Array.GetType().GetElementType() == typeof(Int16));
        }        
        
        [TestMethod]
        public void DowncastingIntToUShort()
        {
            var nd = np.ones(np.int32, 3, 3, 3);
            var int16_copied = nd.astype(np.uint16, true);
            var int16 = nd.astype(np.uint16, false);

            //test copying
            Assert.IsTrue(int16_copied.Array!=nd.Array);
            Assert.IsTrue(int16.Array==nd.Array);
            Assert.IsTrue(int16_copied.Array.GetType().GetElementType() == typeof(UInt16));
            Assert.IsTrue(int16.Array.GetType().GetElementType() == typeof(UInt16));
        }

    }
}
