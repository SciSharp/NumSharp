using System;
using System.Runtime.InteropServices;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Native memory with an inaccessible page directly below its first byte and directly above its last one. A
    ///     read or a write one byte outside it ends the process with an access violation instead of going unseen.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why tests need it.</b> A kernel that READS past its operand gives right values (the extra bytes
    ///     are discarded), and sentinel margins only show writes. Such a read is a crash waiting for the one buffer
    ///     that ends where mapped memory ends: <c>np.zeros(163840, np.uint8)[1::2].copy()</c> ended the process until
    ///     2026-10-10, because the copy's second vector load read one byte past the view and a zeroed array of 128 KB
    ///     or more is its own region of pages. Place an operand against a fence with <see cref="Place"/> and the same
    ///     defect fails the test run, on every buffer size.</para>
    ///     <para><b>How.</b> The region is reserved whole and only its data pages are committed. A reserved page
    ///     that is not committed faults on any access, and nothing else can be mapped into it.</para>
    ///     <para><b>Where.</b> Windows only (<see cref="IsSupported"/>). A test that needs the fences reports
    ///     Inconclusive elsewhere. A violation ends the test host: the run reports "Test host process crashed" with
    ///     the stack of the offending kernel, not a failed assertion.</para>
    /// </remarks>
    internal sealed unsafe class FencedMemory : IDisposable
    {
        /// <summary>Bytes of one page (each fence is one page).</summary>
        public const int PageBytes = 4096;

        private const uint MemCommit = 0x1000, MemReserve = 0x2000, MemRelease = 0x8000, PageReadWrite = 0x04, PageNoAccess = 0x01;

        /// <summary>Reserves or commits pages (kernel32).</summary>
        /// <param name="address">Start of the range to commit, or zero to let the system choose a region.</param>
        /// <param name="size">Bytes.</param>
        /// <param name="allocationType">MEM_RESERVE or MEM_COMMIT.</param>
        /// <param name="protect">Page protection.</param>
        /// <returns>The range's base address, or zero on failure.</returns>
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr VirtualAlloc(IntPtr address, UIntPtr size, uint allocationType, uint protect);

        /// <summary>Releases a region reserved by <see cref="VirtualAlloc"/> (kernel32).</summary>
        /// <param name="address">The region's base address.</param>
        /// <param name="size">Zero with MEM_RELEASE.</param>
        /// <param name="freeType">MEM_RELEASE.</param>
        /// <returns>True on success.</returns>
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool VirtualFree(IntPtr address, UIntPtr size, uint freeType);

        /// <summary>True where the fences can be built.</summary>
        public static bool IsSupported => OperatingSystem.IsWindows();

        /// <summary>Base of the reserved region (the low fence).</summary>
        private readonly IntPtr _region;

        /// <summary>First accessible byte.</summary>
        public readonly byte* Start;

        /// <summary>One past the last accessible byte.</summary>
        public readonly byte* End;

        /// <summary>Accessible bytes.</summary>
        public long Bytes => End - Start;

        /// <summary>Reserves a fence page, <paramref name="pages"/> data pages and a fence page.</summary>
        /// <param name="pages">Number of accessible pages (at least 1).</param>
        /// <exception cref="PlatformNotSupportedException">The host is not Windows (see <see cref="IsSupported"/>).</exception>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="pages"/> is less than 1.</exception>
        /// <exception cref="OutOfMemoryException">The system refused the reservation or the commit.</exception>
        public FencedMemory(int pages)
        {
            if (!IsSupported)
                throw new PlatformNotSupportedException("FencedMemory needs VirtualAlloc (Windows).");
            if (pages < 1)
                throw new ArgumentOutOfRangeException(nameof(pages));
            _region = VirtualAlloc(IntPtr.Zero, (UIntPtr)((ulong)(pages + 2) * PageBytes), MemReserve, PageNoAccess);
            if (_region == IntPtr.Zero)
                throw new OutOfMemoryException("VirtualAlloc could not reserve the fenced region.");
            IntPtr data = VirtualAlloc(_region + PageBytes, (UIntPtr)((ulong)pages * PageBytes), MemCommit, PageReadWrite);
            if (data == IntPtr.Zero)
            {
                // Nothing was committed: give the reservation back before reporting.
                VirtualFree(_region, UIntPtr.Zero, MemRelease);
                throw new OutOfMemoryException("VirtualAlloc could not commit the fenced data pages.");
            }
            Start = (byte*)data;
            End = Start + (long)pages * PageBytes;
        }

        /// <summary>The lowest address of a block of <paramref name="span"/> bytes placed against a fence.</summary>
        /// <param name="span">Bytes of the block (at most <see cref="Bytes"/>).</param>
        /// <param name="atEnd">True: the block's last byte is the last accessible one. False: its first byte is the
        ///     first accessible one.</param>
        /// <returns>The block's lowest address. With <paramref name="atEnd"/> it is aligned like
        ///     <paramref name="span"/> (a block of whole elements ends on a page boundary, so it starts on an element
        ///     boundary).</returns>
        /// <exception cref="ArgumentOutOfRangeException">The block does not fit.</exception>
        public byte* Place(long span, bool atEnd)
        {
            if (span < 0 || span > Bytes)
                throw new ArgumentOutOfRangeException(nameof(span), $"{span} bytes do not fit {Bytes} fenced bytes.");
            return atEnd ? End - span : Start;
        }

        /// <summary>Releases the region. Pointers into it must not be used afterwards.</summary>
        public void Dispose() => VirtualFree(_region, UIntPtr.Zero, MemRelease);
    }
}
