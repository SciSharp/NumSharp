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

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr VirtualAlloc(IntPtr lpAddress, UIntPtr dwSize, uint flAllocationType, uint flProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

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
