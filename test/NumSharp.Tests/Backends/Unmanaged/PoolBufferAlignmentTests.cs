using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Backends.Unmanaged
{
    /// <summary>
    ///     A buffer of one page or more starts on a 64-byte boundary
    ///     (<see cref="SizeBucketedBufferPool.AlignedMinBytes"/>, <see cref="SizeBucketedBufferPool.BufferAlignment"/>),
    ///     whichever way it reaches the caller: fresh, zeroed, reused from a bucket, outside the pool's size window,
    ///     or behind an array.
    /// </summary>
    /// <remarks>
    ///     The C allocator returns 16-byte aligned blocks. A buffer that starts in the middle of a 32-byte vector makes
    ///     every second vector load of a SIMD kernel cross a cache line, which cost 20 to 78 percent on arrays in the L2
    ///     cache; before 2026-10-10 each array won or lost that by its address. The pool allocates a block that is
    ///     80 bytes larger, hands out the first boundary a header past its start, and keeps the block's own pointer in
    ///     that header for the free.
    /// </remarks>
    [TestClass]
    public unsafe class PoolBufferAlignmentTests : TestClass
    {
        /// <summary>Sizes from one page up: at the threshold, off a power of two, large, and one outside the pooled window.</summary>
        private static readonly long[] AlignedSizes =
        {
            4096, 4097, 5000, 8000, 65536, 100_000, 131_071, 800_000, 1 << 20, 4_000_003,
            SizeBucketedBufferPool.MaxPoolableBytes + 1,
        };

        /// <summary>Whether <paramref name="p"/> is on a <see cref="SizeBucketedBufferPool.BufferAlignment"/>-byte boundary.</summary>
        /// <param name="p">The address.</param>
        /// <returns>True when it is.</returns>
        private static bool IsAligned(IntPtr p) => ((long)p & (SizeBucketedBufferPool.BufferAlignment - 1)) == 0;

        /// <summary>
        ///     <see cref="SizeBucketedBufferPool.Take"/> and <see cref="SizeBucketedBufferPool.TakeZeroed"/> give aligned
        ///     buffers at every size from one page up, the whole buffer can be written, a zeroed one is zero, and a
        ///     buffer taken again after its return is still aligned. Several buffers of one size are alive at once, so
        ///     the answer does not come from one lucky address.
        /// </summary>
        [TestMethod]
        public void TakeAndTakeZeroed_FromOnePageUp_AreAligned()
        {
            if (SizeBucketedBufferPool.GuardPagesEnabled)
                Assert.Inconclusive("The guard-page mode places a buffer's END at a page boundary instead.");

            foreach (long bytes in AlignedSizes)
            {
                var live = new List<IntPtr>();
                int count = bytes > (1 << 22) ? 2 : 6;
                for (int k = 0; k < count; k++)
                {
                    IntPtr p = SizeBucketedBufferPool.Take(bytes);
                    IsAligned(p).Should().BeTrue($"Take({bytes}) #{k} at {(long)p:X}");
                    // The first and the last byte belong to the buffer (the header lies in front of it).
                    ((byte*)p)[0] = 0xA5;
                    ((byte*)p)[bytes - 1] = 0x5A;
                    live.Add(p);
                }
                foreach (var p in live)
                {
                    ((byte*)p)[0].Should().Be(0xA5, $"{bytes}: another buffer's header overwrote this one's first byte");
                    ((byte*)p)[bytes - 1].Should().Be(0x5A);
                    SizeBucketedBufferPool.Return(p, bytes);
                }

                // Taken again: from the bucket when the size is pooled, fresh otherwise. Aligned either way.
                IntPtr again = SizeBucketedBufferPool.Take(bytes);
                IsAligned(again).Should().BeTrue($"Take({bytes}) after a return");
                SizeBucketedBufferPool.Return(again, bytes);

                IntPtr zeroed = SizeBucketedBufferPool.TakeZeroed(bytes);
                IsAligned(zeroed).Should().BeTrue($"TakeZeroed({bytes})");
                new ReadOnlySpan<byte>((void*)zeroed, (int)System.Math.Min(bytes, 1 << 20)).IndexOfAnyExcept((byte)0).Should().Be(-1, $"TakeZeroed({bytes}) is zero");
                ((byte*)zeroed)[bytes - 1].Should().Be(0);
                SizeBucketedBufferPool.Return(zeroed, bytes);
            }
        }

        /// <summary>
        ///     Below one page nothing changes: the buffer is the C allocator's own block, usable from its first to its
        ///     last byte and returned to the pool as before.
        /// </summary>
        [TestMethod]
        public void BelowOnePage_BuffersAreTheAllocatorsOwn()
        {
            foreach (long bytes in new long[] { 1, 8, 16, 100, 1000, 4000, 4095 })
            {
                IntPtr p = SizeBucketedBufferPool.Take(bytes);
                p.Should().NotBe(IntPtr.Zero);
                ((byte*)p)[0] = 1;
                ((byte*)p)[bytes - 1] = 2;
                SizeBucketedBufferPool.Return(p, bytes);

                IntPtr z = SizeBucketedBufferPool.TakeZeroed(bytes);
                new ReadOnlySpan<byte>((void*)z, (int)bytes).IndexOfAnyExcept((byte)0).Should().Be(-1);
                SizeBucketedBufferPool.Return(z, bytes);
            }
        }

        /// <summary>
        ///     Arrays of one page or more start on a 64-byte boundary however they are made: uninitialised, zeroed, a
        ///     range, the result of arithmetic, of a comparison, of a cast and of a copy.
        /// </summary>
        [TestMethod]
        public void Arrays_FromOnePageUp_StartOnACacheLine()
        {
            if (SizeBucketedBufferPool.GuardPagesEnabled)
                Assert.Inconclusive("The guard-page mode places a buffer's END at a page boundary instead.");

            foreach (long n in new long[] { 512, 1000, 12_500, 100_000 })      // float64: 4096 bytes and more
            {
                using var empty = np.empty(new Shape(n), NPTypeCode.Double);
                using var zeros = np.zeros(new Shape(n), NPTypeCode.Double);
                using var range = np.arange(n).astype(NPTypeCode.Double);
                using var sum = range + zeros;
                using var less = range < zeros;                              // n bytes: aligned from 4096 elements
                using var asSingle = range.astype(NPTypeCode.Single);       // 4n bytes
                using var copy = range["::-1"].copy();

                foreach (var (name, a) in new (string, NDArray)[] { ("empty", empty), ("zeros", zeros), ("arange", range), ("a + b", sum), ("copy", copy) })
                    IsAligned((IntPtr)a.Address).Should().BeTrue($"{name} of {n} float64 at {(long)a.Address:X}");
                if (n * 4 >= SizeBucketedBufferPool.AlignedMinBytes)
                    IsAligned((IntPtr)asSingle.Address).Should().BeTrue($"astype(float32) of {n}");
                if (n >= SizeBucketedBufferPool.AlignedMinBytes)
                    IsAligned((IntPtr)less.Address).Should().BeTrue($"a < b of {n}");
            }
        }

        /// <summary>
        ///     <see cref="SizeBucketedBufferPool.Clear"/> frees pooled aligned buffers through their header, and the
        ///     pool works afterwards.
        /// </summary>
        [TestMethod]
        public void Clear_FreesAlignedBuffers()
        {
            const long bytes = 24_000;
            var taken = new List<IntPtr>();
            for (int k = 0; k < 4; k++)
                taken.Add(SizeBucketedBufferPool.Take(bytes));
            foreach (var p in taken)
                SizeBucketedBufferPool.Return(p, bytes);

            SizeBucketedBufferPool.Clear();

            IntPtr after = SizeBucketedBufferPool.Take(bytes);
            if (!SizeBucketedBufferPool.GuardPagesEnabled)
                IsAligned(after).Should().BeTrue();
            ((byte*)after)[bytes - 1] = 9;
            SizeBucketedBufferPool.Return(after, bytes);
        }
    }
}
