using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.frombuffer</c> and <c>np.ndarray(shape, buffer=...)</c> with a fixed-width string dtype (<c>S</c>/<c>U</c>),
    ///     against NumPy 2.4.2 (probed 2026-10-07). The string oracle tier (<c>strings.jsonl</c>, ops <c>frombuffer</c>,
    ///     <c>frombuffer_str</c> and <c>ndarray</c>) gates the byte[] matrix — values, checks in NumPy's order, flags; these
    ///     tests pin what the corpus cannot express: the other C# buffer kinds (span, segment, memory, typed array, pointer,
    ///     memoryview), lifetime and write-through, and the documented divergences.
    /// </summary>
    [TestClass]
    public class FromBufferStringTests
    {
        /// <summary>The ASCII bytes of <paramref name="text"/>: NumPy's <c>bytearray(b'...')</c>.</summary>
        /// <param name="text">ASCII text.</param>
        /// <returns>A new byte array.</returns>
        private static byte[] Ascii(string text) => Encoding.ASCII.GetBytes(text);

        // ---- the dtype-string overloads ---------------------------------------------------------------------------

        /// <summary>
        ///     Guards that the dtype-STRING overload reads <c>'c'</c> and <c>'S1'</c> as NumPy's ONE-byte strings. Its
        ///     parser used to map both to NumSharp's 2-byte <c>Char</c>, so <c>np.frombuffer(b'abcd', 'S1')</c> returned two
        ///     UTF-16 code units instead of NumPy's four bytes.
        /// </summary>
        [TestMethod]
        public void StringOverload_COrS1_IsOneByteStrings()
        {
            foreach (var spelling in new[] { "c", "S1", "|S1" })
            {
                using var a = np.frombuffer(Ascii("abcd"), spelling);
                a.dtype.str.Should().Be("|S1", spelling);
                a.ToString(true).Should().Be("array([b'a', b'b', b'c', b'd'], dtype='|S1')", spelling);
            }
        }

        /// <summary>Guards that a sized string spelling reaches the descriptor route through every string overload.</summary>
        [TestMethod]
        public void StringOverload_SizedSpellings_ReadWholeElements()
        {
            using var s3 = np.frombuffer(Ascii("abcdef"), "S3");
            s3.ToString(true).Should().Be("array([b'abc', b'def'], dtype='|S3')");

            using var u1 = np.frombuffer(Encoding.UTF32.GetBytes("hi"), "<U1");
            u1.ToString(true).Should().Be("array(['h', 'i'], dtype='<U1')");

            using var source = np.frombuffer(Ascii("abcdef"), np.uint8);
            using var viaView = np.frombuffer(source.data, "S2", count: 2, offset: 2);
            viaView.ToString(true).Should().Be("array([b'cd', b'ef'], dtype='|S2')");
        }

        // ---- the C# buffer kinds ----------------------------------------------------------------------------------

        /// <summary>
        ///     Guards the <see cref="ReadOnlySpan{T}"/> overload: the selected bytes are COPIED (a span cannot stay pinned), so
        ///     the result owns its data and a later change to the source does not reach it.
        /// </summary>
        [TestMethod]
        public void Span_CopiesIntoAnOwningArray()
        {
            var bytes = Ascii("abcdefghi");
            using var a = np.frombuffer((ReadOnlySpan<byte>)bytes, np.dtype("S3"), count: 2, offset: 3);
            a.ToString(true).Should().Be("array([b'def', b'ghi'], dtype='|S3')");
            a.flags.owndata.Should().BeTrue("a span is copied");

            bytes[3] = (byte)'X';
            a.ToString(true).Should().Be("array([b'def', b'ghi'], dtype='|S3')", "the copy does not see the source");
        }

        /// <summary>
        ///     Guards that an <see cref="ArraySegment{T}"/> is the WHOLE buffer: its elements end at the segment's end, as
        ///     NumPy's frombuffer of a sliced memoryview (<c>memoryview(b)[3:9]</c>) ends there.
        /// </summary>
        [TestMethod]
        public void Segment_EndsAtTheSegmentEnd()
        {
            var bytes = Ascii("abcdefghijkl");
            using var a = np.frombuffer(new ArraySegment<byte>(bytes, 3, 6), np.dtype("S3"));
            a.ToString(true).Should().Be("array([b'def', b'ghi'], dtype='|S3')");
            a.flags.owndata.Should().BeFalse("a view of the segment's array");

            // A count past the segment is NumPy's refusal, even though the array behind it has more bytes.
            Action tooMany = () => np.frombuffer(new ArraySegment<byte>(bytes, 3, 6), np.dtype("S3"), count: 3);
            tooMany.Should().Throw<ValueError>().WithMessage("buffer is smaller than requested size");
        }

        /// <summary>Guards the <see cref="Memory{T}"/> overload: a writeable view of the pinned memory, bounded by it.</summary>
        [TestMethod]
        public void Memory_IsAWriteThroughView()
        {
            var bytes = Ascii("abcdefgh");
            using var a = np.frombuffer(new Memory<byte>(bytes, 2, 4), np.dtype("S2"));
            a.ToString(true).Should().Be("array([b'cd', b'ef'], dtype='|S2')");

            a[0] = "XY";
            Encoding.ASCII.GetString(bytes).Should().Be("abXYefgh", "the result is a view of the memory");
        }

        /// <summary>
        ///     Guards a typed C# array as the buffer: its bytes read as string elements over the pinned array (NumPy:
        ///     <c>np.frombuffer(np.array([0x64636261, 0x68676665], np.int32), 'S4')</c> is <c>[b'abcd', b'efgh']</c>).
        /// </summary>
        [TestMethod]
        public void TypedArray_BytesReadAsStrings()
        {
            using var s4 = np.frombuffer(new[] { 0x64636261, 0x68676665 }, np.dtype("S4"));
            s4.ToString(true).Should().Be("array([b'abcd', b'efgh'], dtype='|S4')");
            s4.flags.owndata.Should().BeFalse();

            using var u1 = np.frombuffer(new[] { 97, 98 }, np.dtype("U1"));
            u1.ToString(true).Should().Be("array(['a', 'b'], dtype='<U1')");
        }

        /// <summary>
        ///     Guards unmanaged memory as the buffer: without a dispose action the result does not own its data (NumPy's array
        ///     over a foreign buffer); with one, releasing the result frees the memory exactly once.
        /// </summary>
        [TestMethod]
        public unsafe void Pointer_ViewOrOwnership()
        {
            var bytes = Ascii("abcdef");
            fixed (byte* p = bytes)
            {
                using var view = np.frombuffer((IntPtr)p, bytes.Length, np.dtype("S3"));
                view.ToString(true).Should().Be("array([b'abc', b'def'], dtype='|S3')");
                view.flags.owndata.Should().BeFalse("the memory is the caller's");
            }

            IntPtr mem = Marshal.AllocHGlobal(4);
            Marshal.Copy(Ascii("wxyz"), 0, mem, 4);
            int freed = 0;
            var owned = np.frombuffer(mem, 4, np.dtype("S2"), dispose: () => { freed++; Marshal.FreeHGlobal(mem); });
            owned.ToString(true).Should().Be("array([b'wx', b'yz'], dtype='|S2')");
            owned.flags.owndata.Should().BeTrue("ownership was transferred through the dispose action");
            owned.Dispose();
            freed.Should().Be(1, "releasing the last array frees the memory once");
        }

        /// <summary>
        ///     Guards NumPy's order for a memoryview over a read-only array: the unsized check first, then the result is
        ///     read-only for good (<c>setflags(write=True)</c> refuses).
        /// </summary>
        [TestMethod]
        public void MemoryView_ReadOnlySource_GivesAReadOnlyResult()
        {
            using var src = np.frombuffer(Ascii("abcdef"), np.uint8);
            src.setflags(write: false);
            using var a = np.frombuffer(src.data, np.dtype("S3"));
            a.ToString(true).Should().Be("array([b'abc', b'def'], dtype='|S3')");
            a.flags.writeable.Should().BeFalse();
            Action reenable = () => a.setflags(write: true);
            reenable.Should().Throw<Exception>("a read-only exporter's buffer stays read-only");

            Action unsized = () => np.frombuffer(src.data, np.dtype("S"));
            unsized.Should().Throw<ValueError>().WithMessage("itemsize cannot be zero in type");
        }

        /// <summary>
        ///     Guards that <see cref="NPTypeCode.String"/> (the string lane, which carries no width) is NumPy's unsized class
        ///     on every NPTypeCode overload: <c>np.frombuffer(b, np.str_)</c> is <c>ValueError: itemsize cannot be zero in
        ///     type</c>.
        /// </summary>
        [TestMethod]
        public unsafe void NPTypeCodeString_IsTheUnsizedRefusal()
        {
            var bytes = Ascii("abcdef");
            using var src = np.frombuffer(bytes, np.uint8);
            var calls = new Action[]
            {
                () => np.frombuffer(bytes, NPTypeCode.String),
                () => np.frombuffer(src.data, NPTypeCode.String),
                () => np.frombuffer((ReadOnlySpan<byte>)bytes, NPTypeCode.String),
                () => np.frombuffer(new ArraySegment<byte>(bytes), NPTypeCode.String),
                () => np.frombuffer(new Memory<byte>(bytes), NPTypeCode.String),
                () => np.frombuffer(new[] { 1, 2 }, NPTypeCode.String),
            };
            foreach (var call in calls)
                call.Should().Throw<ValueError>().WithMessage("itemsize cannot be zero in type");

            fixed (byte* p = bytes)
            {
                var address = (IntPtr)p;
                Action pointer = () => np.frombuffer(address, bytes.Length, NPTypeCode.String);
                pointer.Should().Throw<ValueError>().WithMessage("itemsize cannot be zero in type");
            }
        }

        // ---- documented divergences -------------------------------------------------------------------------------

        /// <summary>
        ///     Pins the big-endian <c>U</c> divergence: NumPy keeps <c>'&gt;U2'</c> as a view; NumSharp stores every array in
        ///     host order, so it decodes the code points into a native <c>'&lt;U2'</c> copy with the same values.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void BigEndianUnicode_DecodesIntoANativeCopy()
        {
            var bigEndian = new UTF32Encoding(bigEndian: true, byteOrderMark: false).GetBytes("abcd");
            using var a = np.frombuffer(bigEndian, np.dtype(">U2"));
            a.dtype.str.Should().Be("<U2", "NumPy: '>U2'");
            NDArray.AsStringArray(a).Should().Equal("ab", "cd");
            a.flags.owndata.Should().BeTrue("NumPy: a view (owndata False)");

            using var nd = np.ndarray(new Shape(2), np.dtype(">U2"), buffer: bigEndian);
            nd.dtype.str.Should().Be("<U2");
            NDArray.AsStringArray(nd).Should().Equal("ab", "cd");
        }

        /// <summary>
        ///     Pins the zero-width divergence: NumPy's <c>np.ndarray((2,), 'S')</c> builds an <c>S0</c> array (itemsize 0,
        ///     strides 0) and, over a buffer with the shape <c>(-1,)</c>, divides by the zero item size and crashes the
        ///     process. NumSharp's string buffers hold at least one character, so it refuses an unsized string dtype.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void Ndarray_UnsizedString_IsRefused()
        {
            Action noBuffer = () => np.ndarray(new Shape(2), np.dtype("S"));
            noBuffer.Should().Throw<NotSupportedException>().WithMessage("*zero-width*");
            Action overBuffer = () => np.ndarray(new Shape(2), np.dtype("U"), buffer: Ascii("abcdefgh"));
            overBuffer.Should().Throw<NotSupportedException>().WithMessage("*zero-width*");
        }

        /// <summary>
        ///     Guards np.ndarray's no-buffer branch with explicit strides for a string dtype: it allocates the descriptor's
        ///     bytes (the lane carries no width) under the requested byte strides, writeable, owning its data.
        /// </summary>
        [TestMethod]
        public void Ndarray_NoBuffer_StridedStringLayout()
        {
            using var a = np.ndarray(new Shape(2, 3), np.dtype("S2"), strides: new long[] { 2, 4 });
            a.dtype.str.Should().Be("|S2");
            a.strides.Should().Equal(2L, 4L);
            a.flags.owndata.Should().BeTrue();
            a.flags.writeable.Should().BeTrue();

            a[1, 2] = "zz";
            ((byte[])a.GetAtIndex(5)).Should().Equal((byte)'z', (byte)'z');
        }
    }
}
