using System;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for the self-contained code examples of <c>docs/website-src/docs/buffering.md</c>: which C#
    ///     routes copy a managed array and which share it, the byte and float reinterpretation of an <c>int[]</c>, a write
    ///     through a <c>frombuffer</c> view reaching the buffer, a <c>copy()</c> detaching, the copy a
    ///     <see cref="ReadOnlySpan{T}"/> needs, and an explicit byte order. The examples that need a file, a socket or a
    ///     native library are not run; their <c>np.frombuffer</c> overloads are the ones exercised here.
    /// </summary>
    [TestClass]
    public class BufferingDocTests
    {
        /// <summary>"Copy vs view": <c>np.array(int[])</c> copies, while <c>new NDArray(int[])</c> wraps the array, so a
        /// later write to the C# array shows through the second one only.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void ManagedArray_CopyVersusView()
        {
            int[] scores = { 85, 92, 78, 95, 88 };
            var copied = np.array(scores);
            var viewed = new NDArray(scores);
            scores[0] = 0;
            ((int)copied[0]).Should().Be(85, "np.array copies the data");
            ((int)viewed[0]).Should().Be(0, "new NDArray shares the array's memory");
        }

        /// <summary>"Reinterpreting typed arrays": the same 16 bytes of four ints read as bytes (little-endian
        /// <c>[1 0 0 0]</c> first) and as four floats.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void TypedArray_ReinterpretedAsBytesAndFloats()
        {
            int[] values = { 1, 2, 3, 4 };
            var asBytes = np.frombuffer<int>(values, typeof(byte));
            asBytes.size.Should().Be(16);
            if (BitConverter.IsLittleEndian)
                asBytes[":4"].ToArray<byte>().Should().Equal((byte)1, (byte)0, (byte)0, (byte)0);
            var asFloats = np.frombuffer<int>(values, typeof(float));
            asFloats.dtype.Should().Be(np.float32);
            asFloats.size.Should().Be(4);
        }

        /// <summary>"Writes go both ways" and "Making an independent copy": a store through the view rewrites the bytes
        /// (9999 is 0x270F: low byte 15, high byte 39), a buffer write shows through the view, and <c>copy()</c>
        /// detaches.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void View_WritesBothWaysAndCopyDetaches()
        {
            var buffer = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var shorts = np.frombuffer(buffer, typeof(short));
            shorts.SetInt16(9999, 0);
            if (BitConverter.IsLittleEndian)
            {
                buffer[0].Should().Be((byte)15);
                buffer[1].Should().Be((byte)39);
            }

            var bytes = new byte[] { 1, 2, 3, 4 };
            var view = np.frombuffer(bytes, typeof(byte));
            var independent = view.copy();
            bytes[0] = 99;
            ((byte)view[0]).Should().Be((byte)99, "shared");
            ((byte)independent[0]).Should().Be((byte)1, "an independent copy");
        }

        /// <summary>The <see cref="ReadOnlySpan{T}"/> overload copies (a span cannot be pinned for the array's lifetime),
        /// and a byte-order prefix decodes big-endian network data.</summary>
        /// <exception cref="Exception">The assertion framework's failure exception, on a divergence from the page.</exception>
        [TestMethod]
        public void SpanCopiesAndByteOrderDecodes()
        {
            ReadOnlySpan<byte> span = stackalloc byte[16];
            np.frombuffer(span, typeof(int)).ToArray<int>().Should().Equal(0, 0, 0, 0);

            var network = new byte[] { 0, 0, 0, 1, 0, 0, 1, 0 };
            np.frombuffer(network, ">i4").ToArray<int>().Should().Equal(1, 256);
            if (BitConverter.IsLittleEndian)
                np.frombuffer(network, "<i4").ToArray<int>().Should().Equal(16777216, 65536);
        }
    }
}
