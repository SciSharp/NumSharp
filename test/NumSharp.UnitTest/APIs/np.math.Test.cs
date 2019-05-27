using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.UnitTest.APIs
{
    [TestClass]
    public class ApiMathTest
    {
        [TestMethod]
        public void AddInt32()
        {
            var x = np.arange(3);
            var y = np.arange(3);
            var z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<int>(), new int[] { 0, 2, 4 }));

            x = np.arange(9);
            y = np.arange(9);
            z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<int>(), new int[] { 0, 2, 4, 6, 8, 10, 12, 14, 16 }));
        }
        
        [TestMethod]
        public void DivideInt32()
        {
            var x = np.arange(1, 4);
            var y = np.arange(1, 4);
            var z = np.divide(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<int>(), new int[] { 1, 1, 1 }));

            x = np.arange(1, 10);
            y = np.arange(1, 10);
            z = np.divide(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<int>(), new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
        }

        [TestMethod]
        public void Sum2x2Int32()
        {
            var data = new int[,] { { 0, 1 }, { 0, 5 } };

            int s = np.sum(data);
            Assert.AreEqual(s, 6);

            var s0 = np.sum(data, axis: 0);
            Assert.IsTrue(Enumerable.SequenceEqual(s0.shape, new int[] { 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s0.Data<int>(), new int[] { 0, 6 }));

            var s1 = np.sum(data, axis: 1);
            Assert.IsTrue(Enumerable.SequenceEqual(s1.shape, new int[] { 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s1.Data<int>(), new int[] { 1, 5 }));

            var s2 = np.sum(data, axis: -1);
            Assert.IsTrue(Enumerable.SequenceEqual(s1.shape, new int[] { 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s1.Data<int>(), new int[] { 1, 5 }));
        }

        [TestMethod]
         public void Sum2x3x2Int32()
        {
            var data = np.arange(12).reshape(2, 3, 2);

            int s = np.sum(data);
            Assert.AreEqual(s, 66);

            var s0 = np.sum(data, axis: 0);
            Assert.IsTrue(Enumerable.SequenceEqual(s0.shape, new int[] { 3, 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s0.Data<int>(), new int[] { 6, 8, 10, 12, 14, 16 }));

            var s1 = np.sum(data, axis: 1);
            Assert.IsTrue(Enumerable.SequenceEqual(s1.shape, new int[] { 2, 2 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s1.Data<int>(), new int[] { 6, 9, 24, 27 }));

            var s2 = np.sum(data, axis: 2);
            Assert.IsTrue(Enumerable.SequenceEqual(s2.shape, new int[] { 2, 3 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s2.Data<int>(), new int[] { 1, 5, 9, 13, 17, 21 }));

            var s3 = np.sum(data, axis: -1);
            Assert.IsTrue(Enumerable.SequenceEqual(s2.shape, new int[] { 2, 3 }));
            Assert.IsTrue(Enumerable.SequenceEqual(s2.Data<int>(), new int[] { 1, 5, 9, 13, 17, 21 }));
        }

        [TestMethod]
        public void AddUInt8()
        {
            var x = np.arange(3).astype(np.uint8);
            var y = np.arange(3).astype(np.uint8);
            var z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<Byte>(), new Byte[] { 0, 2, 4 }));

            x = np.arange(9).astype(np.uint8);
            y = np.arange(9).astype(np.uint8);
            z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<Byte>(), new Byte[] { 0, 2, 4, 6, 8, 10, 12, 14, 16 }));
        }

        [TestMethod]
        public void DivideUInt8()
        {
            var x = np.arange(1, 4).astype(np.uint8);
            var y = np.arange(1, 4).astype(np.uint8);
            var z = np.divide(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<Byte>(), new Byte[] { 1, 1, 1 }));

            x = np.arange(1, 10).astype(np.uint8);
            y = np.arange(1, 10).astype(np.uint8);
            z = np.divide(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<Byte>(), new Byte[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
        }

        [TestMethod]
        public void AddUInt16()
        {
            var x = np.arange(3).astype(np.uint16);
            var y = np.arange(3).astype(np.uint16);
            var z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<UInt16>(), new UInt16[] { 0, 2, 4 }));

            x = np.arange(9).astype(np.uint16);
            y = np.arange(9).astype(np.uint16);
            z = np.add(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<UInt16>(), new UInt16[] { 0, 2, 4, 6, 8, 10, 12, 14, 16 }));
        }

        [TestMethod]
        public void DivideUInt16()
        {
            var x = np.arange(1, 4).astype(np.uint16);
            var y = np.arange(1, 4).astype(np.uint16);
            var z = np.divide(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<UInt16>(), new UInt16[] { 1, 1, 1 }));

            x = np.arange(1, 10).astype(np.uint16);
            y = np.arange(1, 10).astype(np.uint16);
            z = np.divide(x, y);
            Assert.IsTrue(Enumerable.SequenceEqual(z.Data<UInt16>(), new UInt16[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 }));
        }

    }
}
