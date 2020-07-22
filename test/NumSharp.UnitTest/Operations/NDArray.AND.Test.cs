using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using NumSharp;
using NumSharp.Generic;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayAndTest
    {

        [TestMethod]
        public void BoolTwo1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {true, true, false, false}, new Shape(4));
            var np2 = new NDArray(new[] {true, false, true, false}, new Shape(4));

            var np3 = np1 & np2;

            Assert.IsTrue(Enumerable.SequenceEqual(new[] {true, false, false, false}, np3.Data<bool>()));
        }

        [TestMethod]
        public void BoolTwo2D_NDArrayAND()
        {
            var np1 = new NDArray(typeof(bool), new Shape(2, 3));
            np1.ReplaceData(new bool[] {true, true, false, false, true, false});

            var np2 = new NDArray(typeof(bool), new Shape(2, 3));
            np2.ReplaceData(new bool[] {true, false, true, false, true, true});

            var np3 = np1 & np2;

            // expected
            var np4 = new bool[] {true, false, false, false, true, false};

            Assert.IsTrue(Enumerable.SequenceEqual(np3.Data<bool>(), np4));
        }

        [TestMethod]
        public void Byte1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (byte)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new byte[] {0, 2, 2, 0}, np3.Data<byte>()));
        }

        [TestMethod]
        public void Byte2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (byte)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new byte[] {0, 2, 2, 0, 0, 2}, np3.Data<byte>()));
        }

        [TestMethod]
        public void UShort1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (ushort)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ushort[] {0, 2, 2, 0}, np3.Data<ushort>()));
        }

        [TestMethod]
        public void UShort2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (ushort)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ushort[] {0, 2, 2, 0, 0, 2}, np3.Data<ushort>()));
        }

        [TestMethod]
        public void UInt1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (uint)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new uint[] {0, 2, 2, 0}, np3.Data<uint>()));
        }

        [TestMethod]
        public void UInt2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (uint)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new uint[] {0, 2, 2, 0, 0, 2}, np3.Data<uint>()));
        }

        [TestMethod]
        public void ULong1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (ulong)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ulong[] {0, 2, 2, 0}, np3.Data<ulong>()));
        }

        [TestMethod]
        public void ULong2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (ulong)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ulong[] {0, 2, 2, 0, 0, 2}, np3.Data<ulong>()));
        }

        [TestMethod]
        public void Char1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (char)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new char[] {(char)0, (char)2, (char)2, (char)0 }, np3.Data<char>()));
        }

        [TestMethod]
        public void Char2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (char)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new char[] { (char)0, (char)2, (char)2, (char)0, (char)0, (char)2 }, np3.Data<char>()));
        }

        [TestMethod]
        public void Short1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (short)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new short[] {0, 2, 2, 0}, np3.Data<short>()));
        }

        [TestMethod]
        public void Short2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (short)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new short[] {0, 2, 2, 0, 0, 2}, np3.Data<short>()));
        }

        [TestMethod]
        public void Int1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] {0, 2, 2, 0}, np3.Data<int>()));
        }

        [TestMethod]
        public void Int2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] {0, 2, 2, 0, 0, 2}, np3.Data<int>()));
        }

        [TestMethod]
        public void Long1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4}, new Shape(4));

            var np3 = np1 & (long)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new long[] {0, 2, 2, 0}, np3.Data<long>()));
        }

        [TestMethod]
        public void Long2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));

            var np3 = np1 & (long)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new long[] {0, 2, 2, 0, 0, 2}, np3.Data<long>()));
        }
    }
}
