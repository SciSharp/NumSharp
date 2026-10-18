using System;
using System.IO;
using System.Text;

namespace NumSharp.Tests.Backends.Unmanaged
{
    /// <summary>
    ///     The getters that box an element (<c>GetAtIndex(long)</c>, <c>GetValue(...)</c>, and through them <c>item()</c>,
    ///     text conversion, <c>tofile</c>, <c>np.strings.mod</c>'s values) read a float16 element's TWO bytes only: a float16
    ///     at the very end of mapped memory (<see cref="FencedMemory"/>: an inaccessible page right after it) is read without
    ///     touching the next page.
    /// </summary>
    /// <remarks>
    ///     The .NET 8 runtime's tier-0 code boxes a <see cref="Half"/> loaded through a pointer with a read past its two bytes
    ///     (reproduced with <c>VirtualAlloc</c> outside NumSharp: .NET 8.0.29 ends the process with "Internal CLR error",
    ///     .NET 10 and optimized code do not). <c>PercentFormatTests.Writers_OfAFencedBuffer_ReadOnlyItsElements</c> hit it
    ///     when run alone on net8.0. The getters now box a <see cref="Half"/> made from its 16-bit pattern. The failure is a
    ///     crash of the test host, and only while the getter is still tier-0 code, so this test catches a regression when it
    ///     runs early (alone, or before other tests warmed the getters).
    /// </remarks>
    [TestClass]
    public unsafe class HalfBoxingFenceTests
    {
        /// <summary>A float16 element ending at the inaccessible page, read by every boxing route.</summary>
        [TestMethod]
        public void HalfAtTheEndOfMappedMemory_IsReadAsTwoBytes()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("Fenced memory needs VirtualAlloc (Windows).");
            using var fence = new FencedMemory(pages: 1);
            foreach (int n in new[] { 1, 3 })
            {
                byte* p = fence.Place(2L * n, atEnd: true);
                for (int i = 0; i < n; i++)
                    ((ushort*)p)[i] = BitConverter.HalfToUInt16Bits((Half)(1.5 + i));
                var a = np.frombuffer((IntPtr)p, 2L * n, NPTypeCode.Half);
                Half last = (Half)(1.5 + n - 1);
                a.GetAtIndex(n - 1).Should().Be(last);
                a.Storage.GetValue(new long[] { n - 1 }).Should().Be(last);
                a.Storage.GetValue(new[] { n - 1 }).Should().Be(last);
                a.item(n - 1).Should().Be(last);
                a.ToString().Should().EndWith($"{(n == 1 ? "1.5" : "3.5")}]");
                using var ms = new MemoryStream();
                a.tofile(ms, ",", "%s");
                Encoding.ASCII.GetString(ms.ToArray()).Should().EndWith(n == 1 ? "1.5" : "3.5");
                using var r = np.strings.mod("%s", a);
                r.item(n - 1).Should().Be(n == 1 ? "1.5" : "3.5");
            }
        }
    }
}
