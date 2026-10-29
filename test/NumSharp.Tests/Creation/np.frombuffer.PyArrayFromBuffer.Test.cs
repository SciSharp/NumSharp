using System;
using System.Runtime.InteropServices;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     Every np.frombuffer overload runs NumPy's <c>PyArray_FromBuffer</c>: the descriptor checks, the buffer, the range
    ///     checks (the offset, then the count: the buffer-size rule only for a negative count, a count past the buffer
    ///     refused), then a 1-D view that does not own its data and is writeable exactly when the buffer is. Every expected
    ///     value, flag and text was probed against NumPy 2.4.2; the np-layer dtype plan's batch 6 replay of the snapshot found
    ///     each divergence pinned here (the numeric overloads were seven copies of the checks, with their own rules).
    /// </summary>
    [TestClass]
    public class np_frombuffer_PyArrayFromBuffer_Test
    {
        /// <summary>The bytes 0..<paramref name="n"/>-1, a C# <c>byte[]</c> (NumPy's <c>bytearray(range(n))</c>).</summary>
        /// <param name="n">The length.</param>
        /// <returns>A new array.</returns>
        private static byte[] Range(int n)
        {
            var b = new byte[n];
            for (int i = 0; i < n; i++)
                b[i] = (byte)i;
            return b;
        }

        /// <summary>
        ///     A count reads its items whatever bytes follow them: NumPy checks the buffer size against the item size only
        ///     for a negative count (NumSharp checked it first for every count).
        /// </summary>
        [TestMethod]
        public void Count_IgnoresTrailingPartialItem()
        {
            var b = Range(9);
            using var r = np.frombuffer(b, np.int32, count: 2);   // numpy: [50462976, 117835012]
            CollectionAssert.AreEqual(new[] { 50462976, 117835012 }, r.ToArray<int>());
            ((Action)(() => np.frombuffer(b, np.int32))).Should().Throw<ValueError>()
                .WithMessage("buffer size must be a multiple of element size");
            ((Action)(() => np.frombuffer(b, np.int32, count: 3))).Should().Throw<ValueError>()
                .WithMessage("buffer is smaller than requested size");
        }

        /// <summary>The range checks raise NumPy's ValueError with its texts (NumSharp raised ArgumentException).</summary>
        [TestMethod]
        public void RangeChecks_RaiseValueError()
        {
            var b = Range(9);
            var ex = Assert.ThrowsException<ValueError>(() => np.frombuffer(b, np.int32, offset: 10));
            ex.Message.Should().Be("offset must be non-negative and no greater than buffer length (9)");
            ex = Assert.ThrowsException<ValueError>(() => np.frombuffer(b, np.int32, offset: -1));
            ex.Message.Should().Be("offset must be non-negative and no greater than buffer length (9)");
        }

        /// <summary>
        ///     A view of a <c>byte[]</c> does not own its data and is writeable (NumPy: <c>owndata False</c> over a
        ///     <c>bytearray</c>); NumSharp's numeric result claimed OWNDATA. Writes reach the bytes.
        /// </summary>
        [TestMethod]
        public void ByteArrayView_DoesNotOwnItsData()
        {
            var b = new byte[8];
            using var r = np.frombuffer(b, np.int32);
            r.flags.owndata.Should().BeFalse();
            r.flags.writeable.Should().BeTrue();
            r[1] = 0x01020304;
            CollectionAssert.AreEqual(new byte[] { 0, 0, 0, 0, 4, 3, 2, 1 }, b);
        }

        /// <summary>
        ///     The segment IS the buffer (NumPy's frombuffer of a sliced memoryview): its items end at the segment's end,
        ///     and an offset counts from its start. NumSharp read to the end of the array behind it.
        /// </summary>
        [TestMethod]
        public void ArraySegmentAndMemory_BoundTheBuffer()
        {
            // numpy: big = bytearray(b'\xa5'*3 + bytes(range(1, 9)) + b'\xa5'*5); seg = memoryview(big)[3:11]
            var big = new byte[16];
            for (int i = 0; i < big.Length; i++)
                big[i] = 0xA5;
            for (int i = 0; i < 8; i++)
                big[3 + i] = (byte)(i + 1);
            var seg = new ArraySegment<byte>(big, 3, 8);
            using var r = np.frombuffer(seg, np.int32);           // numpy: [67305985, 134678021]
            CollectionAssert.AreEqual(new[] { 67305985, 134678021 }, r.ToArray<int>());
            r.flags.owndata.Should().BeFalse();
            ((Action)(() => np.frombuffer(new ArraySegment<byte>(big, 3, 9), np.int32))).Should().Throw<ValueError>()
                .WithMessage("buffer size must be a multiple of element size");

            var memory = new Memory<byte>(big, 3, 8);
            using var m = np.frombuffer(memory, np.int32, offset: 4); // numpy: [134678021]
            CollectionAssert.AreEqual(new[] { 134678021 }, m.ToArray<int>());
            ((Action)(() => np.frombuffer(memory, np.int32, offset: 9))).Should().Throw<ValueError>()
                .WithMessage("offset must be non-negative and no greater than buffer length (8)");
        }

        /// <summary>
        ///     A count past the memory is refused, never capped: the address and typed-array overloads capped it at the
        ///     available items. An address view without dispose does not own its data, an empty one included.
        /// </summary>
        [TestMethod]
        public unsafe void AddressAndTypedArray_RefuseACountPastTheBuffer()
        {
            IntPtr mem = Marshal.AllocHGlobal(8);
            try
            {
                ((Action)(() => np.frombuffer(mem, 8, np.int32, count: 3))).Should().Throw<ValueError>()
                    .WithMessage("buffer is smaller than requested size");
                using var r = np.frombuffer(mem, 8, np.int32);
                r.flags.owndata.Should().BeFalse();
                using var e = np.frombuffer(mem, 8, np.int32, count: 0);
                e.size.Should().Be(0);
                e.flags.owndata.Should().BeFalse();
            }
            finally
            {
                Marshal.FreeHGlobal(mem);
            }

            var ints = new[] { 1, 2 };
            ((Action)(() => np.frombuffer(ints, np.int32, count: 3))).Should().Throw<ValueError>()
                .WithMessage("buffer is smaller than requested size");
            using var t = np.frombuffer(ints, np.int16);
            CollectionAssert.AreEqual(new short[] { 1, 0, 2, 0 }, t.ToArray<short>());
            t.flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     A memoryview must be C-contiguous for every dtype (NumPy's <c>BufferError</c>); a non-native dtype reached
        ///     the bytes without the check, and the numeric route raised InvalidOperationException. Any dtype reads the
        ///     bytes of any source, whatever the item sizes (it failed for a U5 source read as int32).
        /// </summary>
        [TestMethod]
        public void MemoryView_IsReadAsCContiguousBytes()
        {
            using var f = np.asfortranarray(np.arange(12.0).reshape(3, 4));
            var ex = Assert.ThrowsException<BufferError>(() => np.frombuffer(f.data, np.dtype(">f8")));
            ex.Message.Should().Be("memoryview: underlying buffer is not C-contiguous");
            Assert.ThrowsException<BufferError>(() => np.frombuffer(f.data, np.int32));

            using var u = np.array(new[] { "ab", "c", "def" }).astype(np.dtype("U5"));
            using var codes = np.frombuffer(u.data, np.int32, count: 2);   // numpy: [97, 98]
            CollectionAssert.AreEqual(new[] { 97, 98 }, codes.ToArray<int>());
            ((Action)(() => np.frombuffer(u.data, np.float64))).Should().Throw<ValueError>()
                .WithMessage("buffer size must be a multiple of element size");
        }

        /// <summary>
        ///     A non-native dtype decodes the big-endian bytes on every overload (NumPy keeps '&gt;i2'; NumSharp stores
        ///     native). The address, Memory and typed-array overloads read the bytes as native (they dropped the byte
        ///     order with the descriptor).
        /// </summary>
        [TestMethod]
        public unsafe void NonNative_IsDecoded_OnEveryBufferKind()
        {
            var be = np.dtype(">i2");
            var bytes = new byte[] { 1, 2, 3, 4 };
            using var r = np.frombuffer(bytes, be);                                     // numpy: [258, 772]
            CollectionAssert.AreEqual(new short[] { 258, 772 }, r.ToArray<short>());
            using var m = np.frombuffer(new Memory<byte>(bytes), be);
            CollectionAssert.AreEqual(new short[] { 258, 772 }, m.ToArray<short>());
            using var t = np.frombuffer(new short[] { 0x0201, 0x0403 }, be);            // the same bytes, little-endian
            CollectionAssert.AreEqual(new short[] { 258, 772 }, t.ToArray<short>());
            fixed (byte* p = bytes)
            {
                using var a = np.frombuffer((IntPtr)p, 4, be);
                CollectionAssert.AreEqual(new short[] { 258, 772 }, a.ToArray<short>());
            }
        }
    }
}
