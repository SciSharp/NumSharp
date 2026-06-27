using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Unmanaged.Pooling
{
    /// <summary>
    ///     OS virtual-memory allocation for large ZERO-INITIALIZED buffers
    ///     (the np.zeros fast path on Windows).
    ///
    ///     WHY THIS EXISTS
    ///     ---------------
    ///     On Windows, <see cref="NativeMemory.AllocZeroed"/> (CRT calloc)
    ///     routes mid-size requests (~256 KiB – 2 MiB) through the process heap
    ///     with <c>HEAP_ZERO_MEMORY</c>, which eagerly commits and memsets every
    ///     page up front — ~0.05 ms for an 800 KiB block even when the caller
    ///     never writes it. <c>VirtualAlloc(MEM_COMMIT)</c> instead returns
    ///     copy-on-write zero pages that the kernel only materialises on first
    ///     touch, so the same block costs ~0.002 ms. That lazy demand-zero is
    ///     exactly what NumPy's calloc gets for free from glibc's <c>mmap</c>
    ///     path; this class reproduces it explicitly for the Windows heap's
    ///     mid-size blind spot.
    ///
    ///     Only used at/above <see cref="ThresholdBytes"/>: below it the heap's
    ///     calloc is syscall-free and faster than a VirtualAlloc round-trip, and
    ///     above ~2 MiB the heap itself switches to VirtualAlloc so calloc is
    ///     already lazy — but VirtualAlloc is uniformly fast across that whole
    ///     range, so one threshold covers both the blind spot and the large tail.
    ///
    ///     On non-Windows, <see cref="IsSupported"/> is false and callers fall
    ///     back to calloc, whose glibc/macOS implementation already mmaps large
    ///     blocks lazily.
    /// </summary>
    internal static unsafe class OsVirtualMemory
    {
        /// <summary>
        ///     Byte size at/above which zeroed allocation prefers VirtualAlloc.
        ///     128 KiB: below this the heap's calloc is still cheap and a
        ///     VirtualAlloc syscall isn't worth it; at/above it the heap starts
        ///     eager-committing, which VirtualAlloc's demand-zero pages avoid.
        /// </summary>
        public const long ThresholdBytes = 128L * 1024;

        /// <summary>True when the VirtualAlloc fast path is available (Windows only).</summary>
        public static readonly bool IsSupported = OperatingSystem.IsWindows();

        private const uint MEM_COMMIT = 0x1000;
        private const uint MEM_RESERVE = 0x2000;
        private const uint MEM_RELEASE = 0x8000;
        private const uint PAGE_READWRITE = 0x04;
        private const uint PAGE_NOACCESS = 0x01;
        private const long PageSize = 4096;

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr VirtualAlloc(IntPtr lpAddress, UIntPtr dwSize, uint flAllocationType, uint flProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

        /// <summary>
        ///     Diagnostic page-heap allocator: returns <paramref name="bytes"/> of usable RW memory
        ///     whose LAST byte sits immediately before an inaccessible (PAGE_NOACCESS) guard page,
        ///     so any read/write one byte past the buffer faults INSTANTLY (AccessViolation at the
        ///     offending access) instead of silently corrupting an adjacent allocation. Used only by
        ///     <see cref="SizeBucketedBufferPool"/>'s opt-in NUMSHARP_GUARD_PAGES mode to localise an
        ///     out-of-bounds write to the exact code/site/case that performs it. The base of the
        ///     reserved region (for <see cref="FreeGuarded"/>) is returned via <paramref name="region"/>.
        ///     Returns <see cref="IntPtr.Zero"/> (and region Zero) on failure.
        /// </summary>
        public static IntPtr AllocGuarded(long bytes, out IntPtr region)
        {
            region = IntPtr.Zero;
            if (bytes <= 0)
                return IntPtr.Zero;

            long dataPages = ((bytes + PageSize - 1) / PageSize) * PageSize;   // usable rounded up to whole pages
            long total = dataPages + PageSize;                                 // + 1 guard page after the data

            IntPtr basep = VirtualAlloc(IntPtr.Zero, (UIntPtr)(nuint)total, MEM_RESERVE, PAGE_READWRITE);
            if (basep == IntPtr.Zero)
                return IntPtr.Zero;

            // Commit the data pages RW; commit the trailing page as NOACCESS (the guard).
            if (VirtualAlloc(basep, (UIntPtr)(nuint)dataPages, MEM_COMMIT, PAGE_READWRITE) == IntPtr.Zero ||
                VirtualAlloc((IntPtr)((byte*)basep + dataPages), (UIntPtr)(nuint)PageSize, MEM_COMMIT, PAGE_NOACCESS) == IntPtr.Zero)
            {
                VirtualFree(basep, UIntPtr.Zero, MEM_RELEASE);
                return IntPtr.Zero;
            }

            region = basep;
            // Right-align the usable buffer so byte [bytes] is the first byte of the guard page.
            return (IntPtr)((byte*)basep + (dataPages - bytes));
        }

        /// <summary>Release a region obtained from <see cref="AllocGuarded"/> (its <c>region</c> base).</summary>
        public static void FreeGuarded(IntPtr region)
        {
            if (region != IntPtr.Zero)
                VirtualFree(region, UIntPtr.Zero, MEM_RELEASE);
        }

        /// <summary>
        ///     Reserve + commit <paramref name="bytes"/> of zero-initialized
        ///     virtual memory (demand-zero pages). Returns <see cref="IntPtr.Zero"/>
        ///     if VirtualAlloc fails, so the caller can fall back to calloc.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static IntPtr Alloc(long bytes)
            => VirtualAlloc(IntPtr.Zero, (UIntPtr)(nuint)bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

        /// <summary>Release a region obtained from <see cref="Alloc"/> (MEM_RELEASE).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Free(IntPtr address)
            => VirtualFree(address, UIntPtr.Zero, MEM_RELEASE);
    }
}
