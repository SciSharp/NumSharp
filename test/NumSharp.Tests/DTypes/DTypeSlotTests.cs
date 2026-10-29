using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     The element slots of the np-layer dtype plan's P3, batch 1: <c>setitem</c> / <c>getitem</c> (NEP 42's slots 7 and 8)
    ///     and the legacy <c>PyArray_ArrFuncs</c> <c>copyswapn</c>, <c>fill</c> and <c>nonzero</c>, on every DType class
    ///     NumSharp.Core registers. The slots have no caller yet, so each one is held to the path it comes from (the flat
    ///     iterator's store, the element read, <c>ndarray.byteswap</c>, <c>np.arange</c>, <c>np.nonzero</c>, the text and
    ///     <c>StringDType</c> element codecs) and, where NumPy's own arithmetic or byte order decides the result, to the
    ///     bytes <c>numpy==2.4.2</c> produces (probed on win-amd64, CPython 3.12).
    /// </summary>
    [TestClass]
    public unsafe class DTypeSlotTests
    {
        // ------------------------------------------------------------------------------------------------------------
        // helpers
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     A zeroed block of native memory for one test, freed on dispose. The slots take raw element pointers, and a
        ///     native block (unlike a <c>fixed</c> array) can be used inside the lambdas the exception checks need.
        /// </summary>
        private sealed class NativeBuffer : IDisposable
        {
            /// <summary>The block's first byte.</summary>
            public readonly byte* P;

            /// <summary>Allocates <paramref name="length"/> zeroed bytes (at least one).</summary>
            /// <param name="length">The block's size in bytes.</param>
            public NativeBuffer(int length) => P = (byte*)NativeMemory.AllocZeroed((nuint)System.Math.Max(length, 1));

            /// <summary>Copies <paramref name="count"/> bytes from <paramref name="offset"/> into a managed array.</summary>
            /// <param name="offset">The first byte to copy.</param>
            /// <param name="count">The number of bytes.</param>
            /// <returns>The bytes.</returns>
            public byte[] Bytes(int offset, int count) => new ReadOnlySpan<byte>(P + offset, count).ToArray();

            /// <summary>Frees the block.</summary>
            public void Dispose() => NativeMemory.Free(P);
        }

        /// <summary>
        ///     The DType classes NumSharp.Core registers itself: other tests register probe classes in the same process, and
        ///     those have no element storage to test.
        /// </summary>
        /// <returns>The core classes, in registration order.</returns>
        private static DTypeMeta[] CoreClasses()
            => DTypeRegistry.All.Where(m => m.GetType().Assembly == typeof(DTypeMeta).Assembly).ToArray();

        /// <summary>Runs <paramref name="action"/> and returns what it threw.</summary>
        /// <param name="action">The call to run.</param>
        /// <returns>The exception, or null when the call returned.</returns>
        private static Exception Record(Action action)
        {
            try
            {
                action();
                return null;
            }
            catch (Exception e)
            {
                return e;
            }
        }

        /// <summary>A value as the assertion messages name it.</summary>
        /// <param name="value">The value in the C# boundary map.</param>
        /// <returns><c>None</c> for null, else the value's CLR type and text.</returns>
        private static string Describe(object value) => value is null ? "None" : $"{value.GetType().Name} {value}";

        /// <summary>
        ///     What the flat iterator's store raises for <paramref name="value"/>, the store the numeric <c>setitem</c> slot
        ///     comes from (NumSharp's <c>PyArray_Pack</c> port, then the element write).
        /// </summary>
        /// <param name="a">A one-element array of the class under test.</param>
        /// <param name="value">The value to store.</param>
        /// <returns>The pack error (np.flatiter rewords it as NumPy's <c>iter_ass_subscript</c> does, so the reworded
        ///     error's cause is returned), or null when the store succeeded.</returns>
        private static Exception FlatStore(NDArray a, object value)
        {
            var e = Record(() => a.flatiter[0] = value);
            return e is ValueError && e.Message == "Error setting single item of array." && e.InnerException != null
                ? e.InnerException
                : e;
        }

        /// <summary>Stores <paramref name="value"/> through <paramref name="d"/>'s <c>setitem</c> and reads it back.</summary>
        /// <param name="d">The descriptor.</param>
        /// <param name="value">The value to store.</param>
        /// <returns>The element <c>getitem</c> reads.</returns>
        private static object StoreRead(DType d, object value)
        {
            using var buf = new NativeBuffer(d.itemsize);
            d.Meta.setitem(d, value, buf.P, null);
            return d.Meta.getitem(d, buf.P, null);
        }

        /// <summary>The bytes <c>setitem</c> stores for <paramref name="value"/>, as lowercase hex.</summary>
        /// <param name="d">The descriptor.</param>
        /// <param name="value">The value to store.</param>
        /// <returns>The element's bytes in memory order.</returns>
        private static string StoredHex(DType d, object value)
        {
            using var buf = new NativeBuffer(d.itemsize);
            d.Meta.setitem(d, value, buf.P, null);
            return Convert.ToHexString(buf.Bytes(0, d.itemsize)).ToLowerInvariant();
        }

        /// <summary>Lowercase hex of a managed byte array.</summary>
        /// <param name="bytes">The bytes.</param>
        /// <returns>The hex text.</returns>
        private static string Hex(byte[] bytes) => Convert.ToHexString(bytes).ToLowerInvariant();

        // ------------------------------------------------------------------------------------------------------------
        // every class
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Every class NumSharp.Core registers defines the slots its storage needs: a class with element storage
        ///     overrides <c>setitem</c>, <c>getitem</c>, <c>copyswapn</c> and <c>nonzero</c>; only the numeric classes define
        ///     <c>fill</c> (NumPy's <c>S</c>, <c>U</c> and <c>StringDType</c> have no fill function); a class without storage
        ///     (the datetime pair until its storage lands, the three abstract Python scalar classes) overrides none. The count
        ///     is a tripwire: a new class must decide its slots before it is counted.
        /// </summary>
        [TestMethod]
        public void EveryClass_DefinesTheSlotsItsStorageNeeds()
        {
            var classes = CoreClasses();
            classes.Should().HaveCount(27,
                "19 numeric classes, S and U, the datetime pair, StringDType and three abstract Python scalar classes; a new class decides its slots, then updates this count");
            foreach (var m in classes)
            {
                var type = m.GetType();
                bool Overrides(string slot)
                    => type.GetMethod(slot, BindingFlags.Instance | BindingFlags.NonPublic)!.DeclaringType == type;
                foreach (var slot in new[] { "setitem", "getitem", "copyswapn", "nonzero" })
                    Overrides(slot).Should().Be(m.HasStorage, $"{m.Name}.{slot}");
                Overrides("fill").Should().Be(m is LegacyBuiltinDTypeMeta, $"{m.Name}.fill");
            }
        }

        /// <summary>
        ///     A class without element storage refuses every slot but <c>fill</c>, which answers NumPy's "no fill-function"
        ///     (false) and touches nothing.
        /// </summary>
        [TestMethod]
        public void Slots_OfAClassWithoutStorage_Refuse()
        {
            var without = CoreClasses().Where(m => !m.HasStorage).ToArray();
            without.Select(m => m.Name).Should().BeEquivalentTo(
                new[] { "DateTime64DType", "TimeDelta64DType", "_PyLongDType", "_PyFloatDType", "_PyComplexDType" });
            foreach (var m in without)
            {
                var d = m.default_descr();
                using var buf = new NativeBuffer(16);
                void Refuses(string slot, Action call)
                    => Record(call).Should().BeOfType<NotSupportedException>($"{m.Name}.{slot}")
                        .Which.Message.Should().Contain($"has no '{slot}' slot");
                Refuses("setitem", () => m.setitem(d, 1, buf.P, null));
                Refuses("getitem", () => m.getitem(d, buf.P, null));
                Refuses("copyswapn", () => m.copyswapn(d, buf.P, 8, null, 0, 1, true));
                Refuses("nonzero", () => m.nonzero(d, buf.P));
                m.fill(d, buf.P, 2).Should().BeFalse(m.Name);
                buf.Bytes(0, 16).Should().OnlyContain(b => b == 0, m.Name);
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // setitem / getitem
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The values the numeric store matrix stores: every kind of the C# boundary map, in and out of range.</summary>
        private static readonly object[] StoredValues =
        {
            0, 1, -1, 127, 128, 255, 256, -129, 70000, -70000, 1L << 40, long.MinValue, ulong.MaxValue,
            BigInteger.Pow(2, 64), -BigInteger.Pow(2, 70), 2.5, -2.5, 0.1f, 1e300, -0.0, double.NaN,
            double.PositiveInfinity, double.NegativeInfinity, (Half)1.5, 'A', 1.25m, new Complex(1, 2),
            new Complex(3, 0), true, false, null, "7", "1.5", "x", " 12 ",
        };

        /// <summary>
        ///     Every numeric class's <c>setitem</c> stores the bytes the flat iterator's store writes, and raises its error
        ///     (type and text) where the store raises; <c>getitem</c> reads what the element read returns (the same CLR type
        ///     and value).
        /// </summary>
        [TestMethod]
        public void SetItem_IsTheFlatIteratorsStore_GetItem_TheElementRead_ForEveryNumericClass()
        {
            foreach (var m in CoreClasses().OfType<LegacyBuiltinDTypeMeta>())
            {
                var d = m.default_descr();
                int size = m.ItemSize;
                foreach (var value in StoredValues)
                {
                    string what = $"{m.Name} <- {Describe(value)}";
                    using var a = np.zeros(new Shape(1), d);
                    var expected = FlatStore(a, value);
                    using var buf = new NativeBuffer(size);
                    var actual = Record(() => m.setitem(d, value, buf.P, null));
                    if (expected != null)
                    {
                        actual.Should().NotBeNull(what);
                        actual.GetType().Should().Be(expected.GetType(), what);
                        actual.Message.Should().Be(expected.Message, what);
                        continue;
                    }
                    actual.Should().BeNull(what);
                    buf.Bytes(0, size).Should().Equal(a.tobytes(), what);
                    var read = m.getitem(d, buf.P, null);
                    var element = a.GetAtIndex(0);
                    read.GetType().Should().Be(element.GetType(), what);
                    read.Should().Be(element, what);
                }
            }
        }

        /// <summary>
        ///     The numeric <c>setitem</c>'s conversions and refusals are NumPy's, text for text (<c>a[0] = value</c> on a
        ///     one-element array): an out-of-range Python int, None, a NaN and an infinity into an integer, a string, a complex
        ///     into a real class; a float truncated into an integer; an overflow to inf into float16; None as the positive quiet
        ///     NaN.
        /// </summary>
        [TestMethod]
        public void SetItem_ConvertsAndRefusesLikeNumPy()
        {
            void Throws<TException>(DType d, object value, string text) where TException : Exception
            {
                using var buf = new NativeBuffer(d.itemsize);
                Record(() => d.Meta.setitem(d, value, buf.P, null)).Should().BeOfType<TException>(Describe(value))
                    .Which.Message.Should().Be(text);
            }

            Throws<OverflowException>(np.int8, 1000, "Python integer 1000 out of bounds for int8");
            Throws<OverflowException>(np.int8, -129, "Python integer -129 out of bounds for int8");
            Throws<TypeError>(np.int8, null, "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            Throws<ValueError>(np.int8, double.NaN, "cannot convert float NaN to integer");
            Throws<OverflowException>(np.int8, double.PositiveInfinity, "cannot convert float infinity to integer");
            Throws<ValueError>(np.int8, "x", "invalid literal for int() with base 10: 'x'");
            Throws<TypeError>(np.int8, new Complex(1, 2), "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
            Throws<TypeError>(np.float16, new Complex(1, 0), "float() argument must be a string or a real number, not 'complex'");

            StoreRead(np.int8, 2.5).Should().Be((sbyte)2);
            StoreRead(np.float16, 1e5).Should().Be(Half.PositiveInfinity);
            StoreRead(np.float16, BigInteger.Pow(2, 70)).Should().Be(Half.PositiveInfinity);
            StoredHex(np.float16, null).Should().Be("007e");                   // 0x7e00
            StoredHex(np.float32, null).Should().Be("0000c07f");               // 0x7fc00000
            StoredHex(np.float64, null).Should().Be("000000000000f87f");       // 0x7ff8000000000000
            StoredHex(np.complex128, null).Should().Be("000000000000f87f000000000000f87f");
        }

        /// <summary>
        ///     A byte-swapped descriptor stores the swapped bytes (NumPy's <c>tobytes()</c> of a big-endian array) and reads
        ///     them back as the value, leaving the element as it is: the whole item swaps, a complex number's halves swap
        ///     apart; an explicit host order is native; a one-byte item has no order.
        /// </summary>
        [TestMethod]
        public void SetItem_GetItem_NonNativeDescriptor_StoreAndReadTheSwappedBytes()
        {
            void Swapped(string dtype, object value, string numpyHex)
            {
                var d = np.dtype(dtype);
                int size = d.itemsize;
                using var buf = new NativeBuffer(size);
                d.Meta.setitem(d, value, buf.P, null);
                Hex(buf.Bytes(0, size)).Should().Be(numpyHex, dtype);
                var read = d.Meta.getitem(d, buf.P, null);
                Hex(buf.Bytes(0, size)).Should().Be(numpyHex, $"{dtype}: getitem reads a swapped copy");
                // The value read back is the value a native descriptor of the class stores and reads.
                var native = StoreRead(d.Meta.default_descr(), value);
                read.GetType().Should().Be(native.GetType(), dtype);
                read.Should().Be(native, dtype);
            }

            Swapped(">i4", 0x01020304, "01020304");
            Swapped(">u2", 0xABCD, "abcd");
            Swapped(">f8", 1.5, "3ff8000000000000");
            Swapped(">c16", new Complex(1.5, -2), "3ff8000000000000c000000000000000");
            Swapped(">f2", 1.0, "3c00");
            Swapped(">f2", -0.0, "8000");
            Swapped("<i4", 0x01020304, "04030201");
            Swapped(">i1", -2, "fe");

            // the sign of a byte-swapped zero survives the read
            var be = np.dtype(">f2");
            using var z = new NativeBuffer(2);
            be.Meta.setitem(be, -0.0, z.P, null);
            BitConverter.HalfToUInt16Bits((Half)be.Meta.getitem(be, z.P, null)).Should().Be(0x8000);
        }

        /// <summary>
        ///     The <c>S</c> and <c>U</c> slots are the element codec every text store and read goes through
        ///     (<see cref="StringArraySlice.SetIndex(long, object)"/> / <see cref="StringArraySlice.GetIndex(long)"/>): the same
        ///     bytes or the same error for every value, and the same text read back.
        /// </summary>
        [TestMethod]
        public void TextSlots_AreTheElementCodec()
        {
            object[] values =
            {
                "xyzw", "ab", "", null, 12, 3.5, 'q', true, new byte[] { 104, 105 }, "a\0b", "ab\0", "é",
                new byte[] { 0xC3, 0xA9 }, new object[] { "a" },
            };
            foreach (var dtype in new[] { "S3", "U3", "S1", "U1" })
            {
                var d = np.dtype(dtype);
                foreach (var value in values)
                {
                    string what = $"{dtype} <- {Describe(value)}";
                    using var a = np.zeros(new Shape(1), d);
                    var cells = (StringArraySlice)a.Storage.InternalArray;
                    var expected = Record(() => cells.SetIndex(0, value));
                    using var buf = new NativeBuffer(d.itemsize);
                    var actual = Record(() => d.Meta.setitem(d, value, buf.P, null));
                    if (expected != null)
                    {
                        actual.Should().NotBeNull(what);
                        actual.GetType().Should().Be(expected.GetType(), what);
                        actual.Message.Should().Be(expected.Message, what);
                        continue;
                    }
                    actual.Should().BeNull(what);
                    buf.Bytes(0, d.itemsize).Should().Equal(a.tobytes(), what);
                    var read = d.Meta.getitem(d, buf.P, null);
                    var element = cells.GetIndex(0);
                    read.GetType().Should().Be(element.GetType(), what);
                    if (read is byte[] bytes)
                        bytes.Should().Equal((byte[])element, what);
                    else
                        read.Should().Be(element, what);
                }
            }
        }

        /// <summary>
        ///     NumPy's text stores and reads: None stores <c>'None'</c> truncated, a number its <c>str()</c>, a long text is cut
        ///     to the width; a read keeps an interior NUL and drops the trailing ones; a byte-swapped <c>U</c> stores each code
        ///     unit big-endian (NumPy's <c>tobytes()</c>) and reads the text back.
        /// </summary>
        [TestMethod]
        public void TextSlots_StoreAndReadLikeNumPy()
        {
            var s3 = np.dtype("S3");
            var u3 = np.dtype("U3");
            ((byte[])StoreRead(s3, null)).Should().Equal("Non"u8.ToArray());
            ((byte[])StoreRead(s3, 12)).Should().Equal("12"u8.ToArray());
            ((byte[])StoreRead(s3, "xyzw")).Should().Equal("xyz"u8.ToArray());
            StoreRead(u3, null).Should().Be("Non");
            StoreRead(u3, 12).Should().Be("12");
            StoreRead(u3, 3.5).Should().Be("3.5");
            StoreRead(u3, "xyzw").Should().Be("xyz");
            StoreRead(u3, "a\0b").Should().Be("a\0b");
            StoreRead(u3, "ab\0").Should().Be("ab");
            ((byte[])StoreRead(s3, new byte[] { (byte)'a', 0, (byte)'b' })).Should().Equal(new byte[] { (byte)'a', 0, (byte)'b' });
            ((byte[])StoreRead(s3, new byte[] { (byte)'a', (byte)'b', 0 })).Should().Equal("ab"u8.ToArray());

            var be = np.dtype(">U3");
            StoredHex(be, "abc").Should().Be("000000610000006200000063");
            StoreRead(be, "abc").Should().Be("abc");
        }

        // ------------------------------------------------------------------------------------------------------------
        // StringDType
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The StringDType descriptors the matrix runs over: no NA object, the three NA kinds, coercion off.</summary>
        /// <returns>The descriptors.</returns>
        private static DType[] StringDTypeKinds() => new[]
        {
            np.dtypes.StringDType(), np.dtypes.StringDType(na_object: null), np.dtypes.StringDType(na_object: double.NaN),
            np.dtypes.StringDType(na_object: "NA"), np.dtypes.StringDType(na_object: ""), np.dtypes.StringDType(coerce: false),
        };

        /// <summary>
        ///     The <c>StringDType</c> slots go through the buffer's string allocator, as
        ///     <see cref="VStringArraySlice.SetIndex(long, object)"/> / <see cref="VStringArraySlice.GetIndex(long)"/> do: the
        ///     same value read back (an NA cell reads the NA object) or the same error, for every NA kind and value, short and
        ///     arena strings both; <c>copyswapn</c> refuses (a cell holds a reference into its buffer's allocator) and
        ///     <c>fill</c> is NumPy's missing fill function.
        /// </summary>
        [TestMethod]
        public void StringDTypeSlots_GoThroughTheBuffersAllocator()
        {
            object[] values =
            {
                "", "a", "xyz", "a text longer than a short string's fifteen bytes", null, double.NaN, "NA", 5, 2.5, 'c', true,
            };
            foreach (var d in StringDTypeKinds())
            {
                foreach (var value in values)
                {
                    string what = $"{d} <- {Describe(value)}";
                    var twin = VStringArraySlice.Allocate(d, 3);
                    var cells = VStringArraySlice.Allocate(d, 3);
                    try
                    {
                        var expected = Record(() => twin.SetIndex(1, value));
                        byte* item = cells.Address + VStringDTypeMeta.SIZEOF_NPY_PACKED_STATIC_STRING;
                        var actual = Record(() => d.Meta.setitem(d, value, item, cells));
                        if (expected != null)
                        {
                            actual.Should().NotBeNull(what);
                            actual.GetType().Should().Be(expected.GetType(), what);
                            actual.Message.Should().Be(expected.Message, what);
                            continue;
                        }
                        actual.Should().BeNull(what);
                        d.Meta.getitem(d, item, cells).Should().Be(twin.GetIndex(1), what);
                        // the neighbouring cells are untouched
                        cells.GetIndex(0).Should().Be(twin.GetIndex(0), what);
                        cells.GetIndex(2).Should().Be(twin.GetIndex(2), what);
                    }
                    finally
                    {
                        twin.DangerousFree();
                        cells.DangerousFree();
                    }
                }

                using var buf = new NativeBuffer(32);
                Record(() => d.Meta.copyswapn(d, buf.P, 16, buf.P + 16, 16, 1, false)).Should().BeOfType<NotSupportedException>()
                    .Which.Message.Should().Contain("hold references");
                d.Meta.fill(d, buf.P, 3).Should().BeFalse();
            }
        }

        /// <summary>
        ///     NumPy's <c>StringDType</c> stores: a number is stored as its <c>str()</c> under coercion and refused with
        ///     NumPy's text without it; None without an NA object stores <c>'None'</c>.
        /// </summary>
        [TestMethod]
        public void StringDTypeSlots_StoreLikeNumPy()
        {
            object Store(DType d, object value)
            {
                var cells = VStringArraySlice.Allocate(d, 1);
                try
                {
                    d.Meta.setitem(d, value, cells.Address, cells);
                    return d.Meta.getitem(d, cells.Address, cells);
                }
                finally
                {
                    cells.DangerousFree();
                }
            }

            var t = np.dtypes.StringDType();
            Store(t, 5).Should().Be("5");
            Store(t, 2.5).Should().Be("2.5");
            Store(t, null).Should().Be("None");
            var strict = np.dtypes.StringDType(coerce: false);
            Record(() => Store(strict, 5)).Should().BeOfType<ValueError>()
                .Which.Message.Should().Be("StringDType only allows string data when string coercion is disabled.");
            Store(np.dtypes.StringDType(na_object: null), null).Should().BeNull();
        }

        /// <summary>
        ///     A <c>StringDType</c> element is stored and read only through the buffer it is a cell of: no buffer, a buffer of
        ///     another class, and a pointer outside the buffer's cells or between two cells are refused before anything is
        ///     written.
        /// </summary>
        [TestMethod]
        public void StringDTypeSlots_RefuseAnElementOutsideTheBuffer()
        {
            var d = np.dtypes.StringDType();
            var cells = VStringArraySlice.Allocate(d, 2);
            using var text = np.zeros(new Shape(2), np.dtype("U3"));
            try
            {
                void Refused(IArraySlice buffer, byte* item)
                {
                    long at = (long)item;
                    Record(() => d.Meta.setitem(d, "x", (byte*)at, buffer)).Should().BeOfType<ArgumentException>()
                        .Which.ParamName.Should().Be("buffer");
                    Record(() => d.Meta.getitem(d, (byte*)at, buffer)).Should().BeOfType<ArgumentException>()
                        .Which.ParamName.Should().Be("buffer");
                }

                Refused(null, cells.Address);
                Refused(text.Storage.InternalArray, cells.Address);
                Refused(cells, cells.Address + 8);
                Refused(cells, cells.Address - 16);
                Refused(cells, cells.Address + 32);
                cells.GetIndex(0).Should().Be("");
                cells.GetIndex(1).Should().Be("");
            }
            finally
            {
                cells.DangerousFree();
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // copyswapn
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>copyswapn</c> with a swap is <c>ndarray.byteswap</c> for every class with byte-ordered elements (the whole
        ///     item, a complex number's halves apart, a <c>U</c> element's 4-byte code units, an <c>S</c> element unchanged); a
        ///     second swap in place restores the source; without a swap it is a copy, and a source stride gathers.
        /// </summary>
        [TestMethod]
        public void CopySwapN_IsByteswap_ForEveryClassWithBytes()
        {
            const int n = 7;
            var descrs = CoreClasses().OfType<LegacyBuiltinDTypeMeta>().Select(m => m.default_descr())
                .Concat(new[] { "S3", "U3", "S1", "U1" }.Select(s => np.dtype(s)));
            foreach (var d in descrs)
            {
                int size = d.itemsize;
                string what = d.ToString(true);
                using var a = np.zeros(new Shape(n), d);
                byte* p = (byte*)a.Storage.Address;
                for (int k = 0; k < n * size; k++)
                    p[k] = (byte)(k * 37 + 11);
                var source = new ReadOnlySpan<byte>(p, n * size).ToArray();
                using var swapped = a.byteswap();
                var expected = swapped.tobytes();

                using var dst = new NativeBuffer(n * size);
                d.Meta.copyswapn(d, dst.P, size, p, size, n, swap: true);
                dst.Bytes(0, n * size).Should().Equal(expected, what);
                d.Meta.copyswapn(d, dst.P, size, null, 0, n, swap: true);
                dst.Bytes(0, n * size).Should().Equal(source, $"{what}: two swaps");

                int half = (n + 1) / 2;
                using var gathered = new NativeBuffer(half * size);
                d.Meta.copyswapn(d, gathered.P, size, p, 2 * size, half, swap: false);
                for (int i = 0; i < half; i++)
                    gathered.Bytes(i * size, size).Should().Equal(source.Skip(2 * i * size).Take(size), $"{what}[{2 * i}]");
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // fill
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Runs <c>fill</c> over NumPy's first two elements of an <c>np.arange</c> result and compares the whole run with
        ///     NumPy's (and, where given, with NumSharp's own <c>np.arange</c>).
        /// </summary>
        /// <typeparam name="T">The unit the expected values are written in (an element, or its raw bits).</typeparam>
        /// <param name="d">The descriptor.</param>
        /// <param name="numpy">NumPy's <c>np.arange</c> result, in units of <typeparamref name="T"/>.</param>
        /// <param name="arange">NumSharp's <c>np.arange</c> call for the same result, or null.</param>
        private static void FillMatches<T>(DType d, T[] numpy, Func<NDArray> arange = null) where T : unmanaged
        {
            int units = d.itemsize / sizeof(T);
            int n = numpy.Length / units;
            using var buf = new NativeBuffer(numpy.Length * sizeof(T));
            var b = (T*)buf.P;
            // np.arange stores start and start + step through setitem, then fills the rest.
            for (int k = 0; k < 2 * units; k++)
                b[k] = numpy[k];
            d.Meta.fill(d, buf.P, n).Should().BeTrue(d.ToString());
            new ReadOnlySpan<T>(b, numpy.Length).ToArray().Should().Equal(numpy, d.ToString());
            if (arange != null)
            {
                using var a = arange();
                a.tobytes().Should().Equal(buf.Bytes(0, numpy.Length * sizeof(T)), $"np.arange as {d}");
            }
        }

        /// <summary>
        ///     <c>fill</c> is NumPy's arithmetic, element for element: an integer class wraps through C's promotion (a 64-bit
        ///     product, truncated; uint64 modular), float32 and float64 compute in their own precision, float16 in float32 with
        ///     one rounding per element (a tie to even at 0.2998), complex its two parts apart. Expected values are
        ///     <c>np.arange</c>'s results in NumPy 2.4.2 (floats as raw bits); NumSharp's <c>np.arange</c> agrees on the
        ///     integer ramps.
        /// </summary>
        [TestMethod]
        public void Fill_IsNumPysArangeArithmetic()
        {
            FillMatches(np.int8, new sbyte[] { 100, 110, 120, -126, -116 }, () => np.arange(100, 150, 10, np.int8));
            FillMatches(np.uint8, new byte[] { 250, 252, 254, 0, 2 }, () => np.arange(250, 260, 2, np.uint8));
            FillMatches(np.int16, new short[] { 32760, 32765, -32766, -32761 }, () => np.arange(32760, 32780, 5, np.int16));
            FillMatches(np.uint16, new ushort[] { 65530, 65533, 0, 3 }, () => np.arange(65530, 65540, 3, np.uint16));
            FillMatches(np.int32, new[] { 2147483640, 2147483645, -2147483646, -2147483641 },
                () => np.arange(2147483640L, 2147483660L, 5, np.int32));
            FillMatches(np.uint32, new uint[] { 4294967290, 4294967295, 4, 9 }, () => np.arange(4294967290L, 4294967310L, 5, np.uint32));
            FillMatches(np.int64, new[] { 9223372036854775797, 9223372036854775802, 9223372036854775807, -9223372036854775804 });
            FillMatches(np.uint64, new ulong[] { 18446744073709551605, 18446744073709551610, 18446744073709551615, 4 });
            // np.arange(0.1, 0.55, 0.1, dtype=...): NumSharp's np.arange computes these ramps in double today, so only
            // NumPy's bytes are the reference here
            FillMatches(np.float32, new uint[] { 1036831949, 1045220557, 1050253722, 1053609165, 1056964608 });
            FillMatches(np.float16, new ushort[] { 11878, 12902, 13516, 13926, 14336 });
            FillMatches(np.float64, new ulong[]
            {
                4591870180066957722, 4596373779694328218, 4599075939470750516, 4600877379321698714, 4602678819172646912,
                4603579539098121011, 4604480259023595111, 4605380978949069210, 4606281698874543309,
            });
            // np.arange(0.1+0.2j, 0.1+1.2j, 0.1+0.1j): the real and imaginary progressions apart
            FillMatches(np.complex128, new ulong[]
            {
                4591870180066957722, 4596373779694328218, 4596373779694328218, 4599075939470750516, 4599075939470750516,
                4600877379321698715, 4600877379321698714, 4602678819172646913, 4602678819172646912, 4603579539098121012,
            });
            // NumSharp's decimal (no NumPy class) in exact decimal arithmetic, raising past its range
            FillMatches(np.@decimal, new[] { 0.1m, 0.3m, 0.5m, 0.7m, 0.9m });
            using var big = new NativeBuffer(3 * 16);
            ((decimal*)big.P)[0] = decimal.MaxValue - 1;
            ((decimal*)big.P)[1] = decimal.MaxValue;
            Record(() => np.@decimal.Meta.fill(np.@decimal, big.P, 3)).Should().BeOfType<OverflowException>();
        }

        /// <summary>
        ///     Every numeric class's <c>fill</c> continues the progression its first two elements start (1, 3, 5, …, each
        ///     element the bytes its own store writes), the bool class raises NumPy's <c>BOOL_fill</c> text, and the text
        ///     classes have no fill function.
        /// </summary>
        [TestMethod]
        public void Fill_EveryNumericClass_ContinuesTheProgression()
        {
            const int n = 6;
            foreach (var m in CoreClasses().OfType<LegacyBuiltinDTypeMeta>())
            {
                var d = m.default_descr();
                int size = m.ItemSize;
                using var buf = new NativeBuffer(n * size);
                m.setitem(d, 1, buf.P, null);
                m.setitem(d, 3, buf.P + size, null);
                if (m.TypeCode == NPTypeCode.Boolean)
                {
                    Record(() => m.fill(d, buf.P, n)).Should().BeOfType<TypeError>().Which.Message.Should()
                        .Be("arange() is only supported for booleans when the result has at most length 2.");
                    continue;
                }
                m.fill(d, buf.P, n).Should().BeTrue(m.Name);
                for (int i = 0; i < n; i++)
                    Hex(buf.Bytes(i * size, size)).Should().Be(StoredHex(d, 1 + 2 * i), $"{m.Name}[{i}]");
            }
            using var text = new NativeBuffer(16);
            foreach (var d in new[] { np.dtype("S3"), np.dtype("U3") })
                d.Meta.fill(d, text.P, 2).Should().BeFalse(d.ToString());
        }

        // ------------------------------------------------------------------------------------------------------------
        // nonzero
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>An array of zeros, signed zeros, NaNs, subnormals and other values of the class under test.</summary>
        /// <param name="m">The class.</param>
        /// <returns>A fresh contiguous array of the class's default descriptor.</returns>
        private static NDArray SpecialValues(LegacyBuiltinDTypeMeta m)
        {
            var d = m.default_descr();
            switch (m.TypeCode)
            {
                case NPTypeCode.Boolean:
                    return np.array(new[] { false, true, false, true });
                case NPTypeCode.Half:
                    return np.array(new ushort[] { 0x0000, 0x8000, 0x0001, 0x7c00, 0xfc00, 0x7e00, 0x8001 }).view(np.float16);
                case NPTypeCode.Single:
                    return np.array(new[] { 0f, -0f, float.NaN, float.Epsilon, 1f, float.NegativeInfinity }).astype(d);
                case NPTypeCode.Double:
                    return np.array(new[] { 0.0, -0.0, double.NaN, double.Epsilon, 1.0 }).astype(d);
                case NPTypeCode.Complex:
                    return np.array(new[]
                    {
                        Complex.Zero, new Complex(0, -0.0), new Complex(-0.0, 0), new Complex(0, 1), new Complex(double.NaN, 0),
                    }).astype(d);
                case NPTypeCode.Decimal:
                    // -0m and 0.0m have nonzero bytes (the sign, the scale) and are zero
                    return np.array(new[] { 0m, decimal.Negate(0m), 1m, 0.0m, -1.5m });
                default:
                    return np.array(new long[] { 0, 1, -1, 0, 256, 65536, 1L << 32, long.MinValue }).astype(d);
            }
        }

        /// <summary>The slot's answer for each element of <paramref name="a"/> beside <c>np.nonzero</c>'s.</summary>
        /// <param name="a">A fresh contiguous array.</param>
        /// <param name="what">The assertion's label.</param>
        /// <returns><c>np.nonzero</c>'s indices.</returns>
        private static long[] NonzeroMatches(NDArray a, string what)
        {
            var d = a.dtype;
            var indices = np.nonzero(a)[0].ToArray<long>();
            for (long i = 0; i < a.size; i++)
            {
                byte* item = (byte*)a.Storage.Address + i * d.itemsize;
                d.Meta.nonzero(d, item).Should().Be(indices.Contains(i), $"{what}[{i}]");
            }
            return indices;
        }

        /// <summary>
        ///     Every class's <c>nonzero</c> answers what <c>np.nonzero</c> answers, element for element (a NaN is nonzero, a
        ///     signed zero is not, a float16 by its bits, a complex number when either part is, a decimal by its value), and
        ///     NumPy 2.4.2 gives the same indices for the float, half, complex and text specials.
        /// </summary>
        [TestMethod]
        public void Nonzero_IsNpNonzero_ForEveryClass()
        {
            foreach (var m in CoreClasses().OfType<LegacyBuiltinDTypeMeta>())
            {
                using var a = SpecialValues(m);
                var indices = NonzeroMatches(a, m.Name);
                switch (m.TypeCode)
                {
                    case NPTypeCode.Half: indices.Should().Equal(new long[] { 2, 3, 4, 5, 6 }, m.Name); break;
                    case NPTypeCode.Single: indices.Should().Equal(new long[] { 2, 3, 4, 5 }, m.Name); break;
                    case NPTypeCode.Double: indices.Should().Equal(new long[] { 2, 3, 4 }, m.Name); break;
                    case NPTypeCode.Complex: indices.Should().Equal(new long[] { 3, 4 }, m.Name); break;
                    case NPTypeCode.Decimal: indices.Should().Equal(new long[] { 2, 4 }, m.Name); break;
                }
            }

            // np.array([b'', b'\x00', b'a'], dtype='S2'): the NUL byte pads to an all-zero element
            using var s = np.zeros(new Shape(3), np.dtype("S2"));
            var sCells = (StringArraySlice)s.Storage.InternalArray;
            sCells.SetIndex(0, new byte[0]);
            sCells.SetIndex(1, new byte[] { 0 });
            sCells.SetIndex(2, new byte[] { (byte)'a' });
            NonzeroMatches(s, "S2").Should().Equal(new long[] { 2 });
            using var u = np.array(new[] { "", "a", "\0b" }, np.dtype("U2"));
            NonzeroMatches(u, "U2").Should().Equal(new long[] { 1, 2 });
        }

        /// <summary>
        ///     A byte-swapped element is tested on its value: a big-endian <c>-0.0</c> (float64, float16, a complex number's
        ///     imaginary part) is zero although its bytes read natively are a nonzero subnormal.
        /// </summary>
        [TestMethod]
        public void Nonzero_NonNativeDescriptor_TestsTheValue()
        {
            bool Nonzero(string dtype, object value)
            {
                var d = np.dtype(dtype);
                using var buf = new NativeBuffer(d.itemsize);
                d.Meta.setitem(d, value, buf.P, null);
                return d.Meta.nonzero(d, buf.P);
            }

            Nonzero(">f8", -0.0).Should().BeFalse();
            Nonzero(">f8", 1.0).Should().BeTrue();
            Nonzero(">f2", -0.0).Should().BeFalse();
            Nonzero(">f2", 1.0).Should().BeTrue();
            Nonzero(">c16", new Complex(0, -0.0)).Should().BeFalse();
            Nonzero(">c16", new Complex(0, 1)).Should().BeTrue();
            Nonzero(">i4", 0).Should().BeFalse();
            Nonzero(">i4", 1).Should().BeTrue();
            Nonzero(">U3", "").Should().BeFalse();
            Nonzero(">U3", "a").Should().BeTrue();
        }

        /// <summary>
        ///     <c>StringDType</c>'s <c>nonzero</c> is NumPy's: a cell is nonzero when its text is not empty; an NA cell is
        ///     nonzero only for a NaN-like NA object (a string NA, even a non-empty one, and None are zero). The indices are
        ///     NumPy 2.4.2's for <c>np.nonzero(np.array(['', 'a', na, 'xyz'], dtype=StringDType(na_object=na)))</c>.
        /// </summary>
        [TestMethod]
        public void Nonzero_StringDType_IsNumPys()
        {
            (DType d, object na, long[] numpy)[] cases =
            {
                (np.dtypes.StringDType(na_object: null), null, new long[] { 1, 3 }),
                (np.dtypes.StringDType(na_object: double.NaN), double.NaN, new long[] { 1, 2, 3 }),
                (np.dtypes.StringDType(na_object: "NA"), "NA", new long[] { 1, 3 }),
                (np.dtypes.StringDType(na_object: ""), "", new long[] { 1, 3 }),
            };
            foreach (var (d, na, numpy) in cases)
            {
                using var a = np.array(new[] { "", "a", na, "xyz" }, d);
                NonzeroMatches(a, d.ToString()).Should().Equal(numpy, d.ToString());
            }
            using var plain = np.array(new[] { "", "a", "xyz", "a text longer than a short string's fifteen bytes" },
                np.dtypes.StringDType());
            NonzeroMatches(plain, "StringDType()").Should().Equal(new long[] { 1, 2, 3 });
        }
    }
}
