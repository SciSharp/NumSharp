using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Operations
{
    [TestClass]
    public class NDArrayAndTest
    {
#pragma warning disable CA1034, CA1815, CA1819
        public struct DataEntry<T>
        {
            public string Name { get; }
            public T[] Data { get; }
            public Shape Shape { get; }

            public DataEntry(string name, T[] data, params int[] dims)
            {
                Name = name;
                Data = data;
                Shape = dims;
            }

            public override string ToString() => $"{Name}{Shape}";
        }
#pragma warning restore CA1815, CA1034, CA1819

        public static List<(string name, string data, int[] dims)> EntryMappings = new List<(string name, string data, int[] dims)>
        {
            ("1D1False", "1False", new [] { 1 }),
            ("1D1True", "1True", new [] { 1 }),
            ("1D4False", "4False", new [] { 4 }),
            ("1D4True", "4True", new [] { 4 }),
            ("10D1False", "10False", new [] { 10 }),
            ("10D1True", "10True", new [] { 10 }),
            ("2D11False", "1False", new [] { 1, 1 }),
            ("2D11True", "1True", new [] { 1, 1 }),
            ("2D14False", "4False", new [] { 1, 4 }),
            ("2D14True", "4True", new [] { 1, 4 }),
            ("2D41False", "4False", new [] { 4, 1 }),
            ("2D41True", "4True", new [] { 4, 1 }),
            ("2D44False", "16False", new [] { 4, 4 }),
            ("2D44True", "16True", new [] { 4, 4 }),
            ("3D111False", "1False", new [] { 1, 1, 1 }),
            ("3D111True", "1True", new [] { 1, 1, 1 }),
            ("3D122False", "4False", new [] { 1, 2, 2 }),
            ("3D122True", "4True", new [] { 1, 2, 2 }),
        };


        public static Dictionary<string, bool[]> BoolArrayData = new Dictionary<string, bool[]>
        {
            { "1False", Enumerable.Repeat(false, 1).ToArray() },
            { "1True", Enumerable.Repeat(true, 1).ToArray() },
            { "4False", Enumerable.Repeat(false, 4).ToArray() },
            { "4True", Enumerable.Repeat(true, 4).ToArray() },
            { "10False", Enumerable.Repeat(false, 10).ToArray() },
            { "10True", Enumerable.Repeat(true, 10).ToArray() },
            { "16False", Enumerable.Repeat(false, 16).ToArray() },
            { "16True", Enumerable.Repeat(true, 16).ToArray() },
        };

        public static Dictionary<string, DataEntry<bool>> BoolEntryData = EntryMappings.ToDictionary(
            x => x.name,
            x => new DataEntry<bool>(x.name, BoolArrayData[x.data], x.dims));

        [DataTestMethod]
        // [1] & [1]
        [DataRow("1D1True", "1D1True", "1True")]
        [DataRow("1D1True", "1D1False", "1False")]
        [DataRow("1D1False", "1D1True", "1False")]
        [DataRow("1D1False", "1D1False", "1False")]

        // [1] & [10]
        [DataRow("1D1True", "10D1True", "10True")]
        [DataRow("1D1True", "10D1False", "10False")]
        [DataRow("1D1False", "10D1True", "10False")]
        [DataRow("1D1False", "10D1False", "10False")]

        // [10] & [1]
        [DataRow("10D1True", "1D1True", "10True")]
        [DataRow("10D1True", "1D1False", "10False")]
        [DataRow("10D1False", "1D1True", "10False")]
        [DataRow("10D1False", "1D1False", "10False")]

        // [10] & [10]
        [DataRow("10D1True", "10D1True", "10True")]
        [DataRow("10D1True", "10D1False", "10False")]
        [DataRow("10D1False", "10D1True", "10False")]
        [DataRow("10D1False", "10D1False", "10False")]

        // [1, 1] & [1, 1]
        [DataRow("2D11True", "2D11True", "1True")]
        [DataRow("2D11True", "2D11False", "1False")]
        [DataRow("2D11False", "2D11True", "1False")]
        [DataRow("2D11False", "2D11False", "1False")]

        // [1, 1] & [1, 4]
        [DataRow("2D11True", "2D14True", "4True")]
        [DataRow("2D11True", "2D14False", "4False")]
        [DataRow("2D11False", "2D14True", "4False")]
        [DataRow("2D11False", "2D14False", "4False")]

        // [1, 4] & [1, 1]
        [DataRow("2D14True", "2D11True", "4True")]
        [DataRow("2D14True", "2D11False", "4False")]
        [DataRow("2D14False", "2D11True", "4False")]
        [DataRow("2D14False", "2D11False", "4False")]

        // [1, 4] & [4, 1]
        [DataRow("2D14True", "2D41True", "16True")]
        [DataRow("2D14True", "2D41False", "16False")]
        [DataRow("2D14False", "2D41True", "16False")]
        [DataRow("2D14False", "2D41False", "16False")]

        // [4, 1] & [1, 4]
        [DataRow("2D41True", "2D14True", "16True")]
        [DataRow("2D41True", "2D14False", "16False")]
        [DataRow("2D41False", "2D14True", "16False")]
        [DataRow("2D41False", "2D14False", "16False")]

        // [4, 4] & [4, 4]
        [DataRow("2D44True", "2D44True", "16True")]
        [DataRow("2D44True", "2D44False", "16False")]
        [DataRow("2D44False", "2D44True", "16False")]
        [DataRow("2D44False", "2D44False", "16False")]

        // [1, 1, 1] & [1, 2, 2]
        [DataRow("3D111True", "3D122True", "4True")]
        [DataRow("3D111True", "3D122False", "4False")]
        [DataRow("3D111False", "3D122True", "4False")]
        [DataRow("3D111False", "3D122False", "4False")]

        // [1, 2, 2] & [1, 2, 2]
        [DataRow("3D122True", "3D111True", "4True")]
        [DataRow("3D122True", "3D111False", "4False")]
        [DataRow("3D122False", "3D111True", "4False")]
        [DataRow("3D122False", "3D111False", "4False")]

        // [1] & [1, 4]
        [DataRow("1D1True", "2D14True", "4True")]
        [DataRow("1D1True", "2D14False", "4False")]
        [DataRow("1D1False", "2D14True", "4False")]
        [DataRow("1D1False", "2D14False", "4False")]

        // [1, 4] & [1]
        [DataRow("2D14True", "1D1True", "4True")]
        [DataRow("2D14True", "1D1False", "4False")]
        [DataRow("2D14False", "1D1True", "4False")]
        [DataRow("2D14False", "1D1False", "4False")]

        // [4] & [1, 4]
        [DataRow("1D4True", "2D14True", "4True")]
        [DataRow("1D4True", "2D14False", "4False")]
        [DataRow("1D4False", "2D14True", "4False")]
        [DataRow("1D4False", "2D14False", "4False")]

        // [1, 4] & [4]
        [DataRow("2D14True", "1D4True", "4True")]
        [DataRow("2D14True", "1D4False", "4False")]
        [DataRow("2D14False", "1D4True", "4False")]
        [DataRow("2D14False", "1D4False", "4False")]
        public void NDArray_ABoolANDABool(string lname, string rname, string ename)
        {
            var ldata = BoolEntryData[lname];
            var rdata = BoolEntryData[rname];
            var expected = BoolArrayData[ename];
            var np1 = new NDArray(ldata.Data, ldata.Shape);
            var np2 = new NDArray(rdata.Data, rdata.Shape);

            var np3 = np1 & np2;

            Assert.IsTrue(Enumerable.SequenceEqual(expected, np3.Data<bool>()));
        }

        public static readonly Type[] Types = new[]
        {
            typeof(bool),
            typeof(char), typeof(byte), typeof(short), typeof(ushort),
            typeof(int), typeof(uint), typeof(long), typeof(ulong),
            typeof(float), typeof(double),
        };

        [TestMethod]
        public void Byte1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (byte)x).ToArray(), new Shape(4));

            var np3 = np1 & (byte)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new byte[] { 0, 2, 2, 0 }, np3.Data<byte>()));
        }

        [TestMethod]
        public void Byte2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (byte)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (byte)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new byte[] { 0, 2, 2, 0, 0, 2 }, np3.Data<byte>()));
        }

        [TestMethod]
        public void UShort1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (ushort)x).ToArray(), new Shape(4));

            var np3 = np1 & (ushort)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ushort[] { 0, 2, 2, 0 }, np3.Data<ushort>()));
        }

        [TestMethod]
        public void UShort2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (ushort)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (ushort)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ushort[] { 0, 2, 2, 0, 0, 2 }, np3.Data<ushort>()));
        }

        [TestMethod]
        public void UInt1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (uint)x).ToArray(), new Shape(4));

            var np3 = np1 & (uint)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new uint[] { 0, 2, 2, 0 }, np3.Data<uint>()));
        }

        [TestMethod]
        public void UInt2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (uint)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (uint)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new uint[] { 0, 2, 2, 0, 0, 2 }, np3.Data<uint>()));
        }

        [TestMethod]
        public void ULong1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (ulong)x).ToArray(), new Shape(4));

            var np3 = np1 & (ulong)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ulong[] { 0, 2, 2, 0 }, np3.Data<ulong>()));
        }

        [TestMethod]
        public void ULong2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (ulong)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (ulong)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new ulong[] { 0, 2, 2, 0, 0, 2 }, np3.Data<ulong>()));
        }

        [TestMethod]
        public void Char1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (char)x).ToArray(), new Shape(4));

            var np3 = np1 & (char)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new char[] { (char)0, (char)2, (char)2, (char)0 }, np3.Data<char>()));
        }

        [TestMethod]
        public void Char2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (char)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (char)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new char[] { (char)0, (char)2, (char)2, (char)0, (char)0, (char)2 }, np3.Data<char>()));
        }

        [TestMethod]
        public void Short1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (short)x).ToArray(), new Shape(4));

            var np3 = np1 & (short)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new short[] { 0, 2, 2, 0 }, np3.Data<short>()));
        }

        [TestMethod]
        public void Short2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (short)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (short)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new short[] { 0, 2, 2, 0, 0, 2 }, np3.Data<short>()));
        }

        [TestMethod]
        public void Int1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (int)x).ToArray(), new Shape(4));

            var np3 = np1 & 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 2, 2, 0 }, np3.Data<int>()));
        }

        [TestMethod]
        public void Int2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (int)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & 2;

            Assert.IsTrue(Enumerable.SequenceEqual(new int[] { 0, 2, 2, 0, 0, 2 }, np3.Data<int>()));
        }

        [TestMethod]
        public void Long1D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4 }.Select(x => (long)x).ToArray(), new Shape(4));

            var np3 = np1 & (long)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new long[] { 0, 2, 2, 0 }, np3.Data<long>()));
        }

        [TestMethod]
        public void Long2D_NDArrayAND()
        {
            var np1 = new NDArray(new[] { 1, 2, 3, 4, 5, 6 }.Select(x => (long)x).ToArray(), new Shape(2, 3));

            var np3 = np1 & (long)2;

            Assert.IsTrue(Enumerable.SequenceEqual(new long[] { 0, 2, 2, 0, 0, 2 }, np3.Data<long>()));
        }
    }
}
