using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     Pins the single-block iterator state allocation: NDIterRef allocates the state header
    ///     and an inline arena in ONE native block, carves the dimension/operand arrays from that
    ///     arena, and recycles the block through a bounded per-thread cache
    ///     (<see cref="NDIterRef.ReleaseStateBlock"/>). Measured: three-operand 1-D construction
    ///     161 ns → 93 ns; the three calloc/free pairs it replaced were 78 ns of that.
    ///
    ///     The correctness hazard of recycling is stale state: every array AllocateDimArrays
    ///     hands out must start zeroed (Coords, BaseOffsets — which FlipNegativeStrides
    ///     ACCUMULATES into — Buffers, BufStrides, ...). These tests build iterators of one shape
    ///     on a block that a very different iterator just released, so any byte that survives
    ///     recycling shows up as a wrong element.
    /// </summary>
    [TestClass]
    public unsafe class NDIterStateBlockTests
    {
        private static readonly NDIterPerOpFlags[] RO_WO = { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY };
        private static readonly NDIterPerOpFlags[] RO_RO_WO = { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY };
        private const NDIterGlobalFlags EXL = NDIterGlobalFlags.EXTERNAL_LOOP;
        private const NPY_ORDER KO = NPY_ORDER.NPY_KEEPORDER;
        private const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;

        private static void CopyF64(void** dp, long* st, long count, void* aux)
        {
            byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1];
            for (long i = 0; i < count; i++) { *(double*)po = *(double*)ps; ps += ss; po += so; }
        }

        private static void AddF64(void** dp, long* st, long count, void* aux)
        {
            byte* pa = (byte*)dp[0]; byte* pb = (byte*)dp[1]; byte* po = (byte*)dp[2];
            long sa = st[0], sb = st[1], so = st[2];
            for (long i = 0; i < count; i++) { *(double*)po = *(double*)pa + *(double*)pb; pa += sa; pb += sb; po += so; }
        }

        private static void DrainCache()
        {
            // Fill and release enough iterators that the per-thread cache is at capacity, so
            // the next construction is guaranteed to pop a RECYCLED block.
            var a = np.arange(4).astype(np.float64);
            for (int i = 0; i < NDIterRef.StateBlockCacheSlots + 2; i++)
            {
                using var it = NDIterRef.New(a);
            }
        }

        [TestMethod]
        public void Dispose_Parks_Block_And_Next_Construction_Pops_It()
        {
            var a = np.arange(8).astype(np.float64);

            { using var warm = NDIterRef.New(a); }
            int parked = NDIterRef.CachedStateBlockCount;
            Assert.IsTrue(parked >= 1, "Dispose must park the block in the per-thread cache");

            using (var it = NDIterRef.New(a))
            {
                Assert.AreEqual(parked - 1, NDIterRef.CachedStateBlockCount, "construction must pop a parked block");
                Assert.IsTrue(it.RawState->HasInlineArena, "a heap state must carry the inline arena");
            }

            Assert.AreEqual(parked, NDIterRef.CachedStateBlockCount, "dispose must park the block again");
        }

        [TestMethod]
        public void Cache_Is_Bounded()
        {
            var a = np.arange(8).astype(np.float64);
            const int live = NDIterRef.StateBlockCacheSlots * 3;

            // Drain what is parked, then hold more iterators alive than the cache can park.
            var states = new NDIterState*[live];
            for (int i = 0; i < live; i++)
            {
                var it = NDIterRef.New(a);
                states[i] = it.ReleaseState();
            }
            Assert.AreEqual(0, NDIterRef.CachedStateBlockCount);

            for (int i = 0; i < live; i++)
                NDIterRef.FreeState(states[i]);

            Assert.AreEqual(NDIterRef.StateBlockCacheSlots, NDIterRef.CachedStateBlockCount,
                "the cache must never hold more than StateBlockCacheSlots blocks");
        }

        [TestMethod]
        public void Recycled_Block_Hosts_A_Different_Iterator_Correctly()
        {
            // First tenant: a buffered CAST iterator (allocates buffers, sets BufStrides,
            // ArrayWritebackPtrs, ...), 1-D, two operands.
            var src32 = np.arange(1000).astype(np.float32);
            var dst64 = np.empty(new Shape(1000), np.float64);
            using (var it = NDIterRef.MultiNew(2, new[] { src32, dst64 },
                       NDIterGlobalFlags.BUFFERED | EXL | NDIterGlobalFlags.GROWINNER, KO, SAFE, RO_WO,
                       new[] { NPTypeCode.Double, NPTypeCode.Double }))
            {
                it.ForEach(CopyF64);
            }
            Assert.AreEqual(999.0, dst64.GetDouble(999));

            // Second tenant on the recycled block: 3-D, three operands, a broadcast operand,
            // strided output view — every array must start from zero for this to be right.
            var a = np.arange(24).astype(np.float64).reshape(2, 3, 4);
            var b = np.arange(4).astype(np.float64);                 // broadcasts over (2,3,4)
            var oBase = np.zeros(new Shape(2, 3, 8), np.float64);
            var o = oBase[":, :, ::2"];
            using (var it = NDIterRef.MultiNew(3, new[] { a, b, o }, EXL, KO, SAFE, RO_RO_WO))
            {
                it.ForEach(AddF64);
            }

            var expected = np.add(a, b);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        Assert.AreEqual(expected.GetDouble(i, j, k), o.GetDouble(i, j, k), $"[{i},{j},{k}]");
        }

        [TestMethod]
        public void Negative_Stride_Flip_On_Recycled_Block_Starts_From_Zero_BaseOffsets()
        {
            // FlipNegativeStrides ACCUMULATES into BaseOffsets (+=). A block that kept the
            // previous tenant's offsets would land the reversed view's base pointer off by
            // that much. Run a flipping iterator, release, then another flipping iterator of
            // a different size on the recycled block and check every element.
            var x = np.arange(64).astype(np.float64);
            var d1 = np.empty(new Shape(64), np.float64);
            using (var it = NDIterRef.MultiNew(2, new[] { x["::-1"], d1 }, EXL, KO, SAFE, RO_WO))
                it.ForEach(CopyF64);
            Assert.AreEqual(63.0, d1.GetDouble(0));

            var y = np.arange(10).astype(np.float64);
            var d2 = np.empty(new Shape(10), np.float64);
            using (var it = NDIterRef.MultiNew(2, new[] { y["::-1"], d2 }, EXL, KO, SAFE, RO_WO))
                it.ForEach(CopyF64);
            for (int i = 0; i < 10; i++)
                Assert.AreEqual(9.0 - i, d2.GetDouble(i), $"[{i}]");
        }

        [TestMethod]
        public void Arena_Overflow_Falls_Back_To_External_Blocks()
        {
            // 40 dimensions × 2 operands: 40*24 + 40*2*8 + 2*88 bytes ≈ 1.8 KB > the arena, so
            // AllocateDimArrays must allocate the separate blocks and free them on Dispose.
            var dims = new int[40];
            for (int i = 0; i < 38; i++) dims[i] = 1;
            dims[38] = 4; dims[39] = 4;
            var a = np.arange(16).astype(np.float64).reshape(new Shape(dims));
            var o = np.empty(new Shape(dims), np.float64);

            DrainCache();
            using (var it = NDIterRef.MultiNew(2, new[] { a, o }, EXL, KO, SAFE, RO_WO))
            {
                Assert.IsTrue(it.RawState->HasInlineArena);
                it.ForEach(CopyF64);
            }
            for (int i = 0; i < 16; i++)
                Assert.AreEqual((double)i, o.flat.GetDouble(i));

            // And the block is still recyclable afterwards (the arena was simply unused).
            using (var it = NDIterRef.New(a))
                Assert.IsTrue(it.RawState->HasInlineArena);
        }

        [TestMethod]
        public void Copy_Gets_Its_Own_Block()
        {
            var a = np.arange(12).astype(np.float64).reshape(3, 4);
            var o = np.empty(new Shape(3, 4), np.float64);
            using var it = NDIterRef.MultiNew(2, new[] { a, o }, EXL, KO, SAFE, RO_WO);
            using var copy = it.Copy();

            Assert.IsTrue(copy.RawState != it.RawState);
            Assert.IsTrue(copy.RawState->HasInlineArena);
            Assert.AreEqual(it.IterSize, copy.IterSize);

            copy.ForEach(CopyF64);
            for (int i = 0; i < 12; i++)
                Assert.AreEqual((double)i, o.flat.GetDouble(i));
        }

        [TestMethod]
        public void Detached_State_Freed_Through_FreeState_Is_Recycled()
        {
            var a = np.arange(8).astype(np.float64);
            var it = NDIterRef.New(a);
            var state = it.ReleaseState();
            it.Dispose();   // no-op: ownership was released

            int before = NDIterRef.CachedStateBlockCount;
            NDIterRef.FreeState(state);
            Assert.IsTrue(NDIterRef.CachedStateBlockCount == System.Math.Min(before + 1, NDIterRef.StateBlockCacheSlots));
        }

        [TestMethod]
        public void Production_Ufunc_Out_Route_Recycles_Across_Layouts()
        {
            // The production out= route constructs one iterator per call; alternate layouts
            // (contiguous / broadcast / strided / reversed) so consecutive calls reuse blocks
            // with different ndim/nop/flags and compare against the allocating route.
            var a = np.arange(60).astype(np.float64).reshape(3, 4, 5);
            var b = np.arange(5).astype(np.float64);
            var big = np.arange(120).astype(np.float64).reshape(3, 4, 10);
            var o = np.empty(new Shape(3, 4, 5), np.float64);

            for (int round = 0; round < 3; round++)
            {
                np.add(a, b, o);
                Assert.IsTrue(np.array_equal(o, np.add(a, b)), "broadcast");
                np.add(a, big[":, :, ::2"], o);
                Assert.IsTrue(np.array_equal(o, np.add(a, big[":, :, ::2"])), "strided");
                np.add(a, a["::-1"], o);
                Assert.IsTrue(np.array_equal(o, np.add(a, a["::-1"])), "reversed");
                np.sqrt(a, o);
                Assert.IsTrue(np.array_equal(o, np.sqrt(a)), "unary");
            }
        }
    }
}
