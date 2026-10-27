using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Backends.Iterators
{
    /// <summary>
    ///     Pins that NDIterRef's drivers run no inner loop over an EMPTY iteration, as NumPy's callers do (they test
    ///     <c>NpyIter_GetIterSize(iter) != 0</c> before <c>do { inner(...) } while (iternext(iter))</c>).
    /// </summary>
    /// <remarks>
    ///     The inner count of an empty iteration is still the innermost axis' extent when an OUTER axis is the empty one,
    ///     so before the guard a driver handed the kernel that many elements of operands holding none. np.average of an
    ///     empty <c>(11, 0, 14)</c> array along axis 2 with weights read and wrote 14 elements past its empty buffers
    ///     through the fused weighted-sum kernel: an access violation, or a heap corrupted for a later allocation to
    ///     find (found by np.average's random differential, the np-layer dtype plan's P2). Each test builds operands with
    ///     an empty outer axis and counts the kernel's calls; the np.average regression places the operand at a fenced
    ///     page so a single read past it ends the test host.
    /// </remarks>
    [TestClass]
    public unsafe class NDIterEmptyIterationTests
    {
        /// <summary>Kernel calls observed by the counting kernels (reset by each test).</summary>
        [ThreadStatic] private static int s_calls;

        /// <summary>A delegate kernel that only counts its calls (it touches no element, so a call cannot fault).</summary>
        /// <param name="dataptrs">The operand pointers (unused).</param>
        /// <param name="strides">The byte strides (unused).</param>
        /// <param name="count">The element count of the call (unused).</param>
        /// <param name="aux">The auxiliary pointer (unused).</param>
        private static void CountingKernel(void** dataptrs, long* strides, long count, void* aux) => s_calls++;

        /// <summary>A struct kernel that only counts its calls, for <see cref="NDIterRef.ExecuteGeneric{TKernel}"/>.</summary>
        private readonly struct CountingLoop : INDInnerLoop
        {
            /// <summary>Counts the call.</summary>
            /// <param name="dataptrs">The operand pointers (unused).</param>
            /// <param name="strides">The byte strides (unused).</param>
            /// <param name="count">The element count of the call (unused).</param>
            public void Execute(void** dataptrs, long* strides, long count) => s_calls++;
        }

        /// <summary>A reducing struct kernel that counts its calls and adds each call's count to the accumulator.</summary>
        private readonly struct CountingReduce : INDReducingInnerLoop<long>
        {
            /// <summary>Counts the call and accumulates its element count.</summary>
            /// <param name="dataptrs">The operand pointers (unused).</param>
            /// <param name="strides">The byte strides (unused).</param>
            /// <param name="count">The element count of the call.</param>
            /// <param name="accumulator">The running element count.</param>
            /// <returns>True: never stops early.</returns>
            public bool Execute(void** dataptrs, long* strides, long count, ref long accumulator)
            {
                s_calls++;
                accumulator += count;
                return true;
            }
        }

        /// <summary>The fused weighted sum's iterator over an empty operand reduced along its last axis: [a, w, num, scl]
        ///     with the outputs' op_axes -1 along the reduced axis, REDUCE_OK and EXTERNAL_LOOP.</summary>
        /// <param name="a">The operand.</param>
        /// <param name="w">The weights of the operand's shape.</param>
        /// <param name="num">The weighted sum (the kept axes).</param>
        /// <param name="scl">The weight sum (the kept axes).</param>
        /// <returns>The iterator (the caller disposes it).</returns>
        private static NDIterRef ReduceLastAxis(NDArray a, NDArray w, NDArray num, NDArray scl)
        {
            int[] ident = { 0, 1, 2 };
            int[] outAxes = { 0, 1, -1 };
            return NDIterRef.AdvancedNew(4, new[] { a, w, num, scl },
                NDIterGlobalFlags.REDUCE_OK | NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE, NDIterPerOpFlags.READWRITE },
                null, 3, new[] { ident, ident, outAxes, outAxes });
        }

        [TestMethod]
        public void ForEach_ReductionOverAnEmptyOuterAxis_MakesNoKernelCall()
        {
            using var a = np.zeros(new Shape(11, 0, 14), np.float64);
            using var w = np.ones(new Shape(11, 0, 14), np.float64);
            using var num = np.zeros(new Shape(11, 0), np.float64);
            using var scl = np.zeros(new Shape(11, 0), np.float64);
            using var it = ReduceLastAxis(a, w, num, scl);
            it.IterSize.Should().Be(0);
            s_calls = 0;
            it.ForEach(CountingKernel);
            s_calls.Should().Be(0, "an empty iteration has no element to hand a kernel");
        }

        [TestMethod]
        public void ForEach_ElementwiseOverAnEmptyLeadingAxis_MakesNoKernelCall()
        {
            foreach (var dims in new[] { new long[] { 0, 3, 4 }, new long[] { 3, 0, 4 }, new long[] { 3, 4, 0 } })
            {
                using var x = np.zeros(new Shape(dims), np.float64);
                using var o = np.zeros(new Shape(dims), np.float64);
                using var it = NDIterRef.MultiNew(2, new[] { x, o }, NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING, new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });
                s_calls = 0;
                it.ForEach(CountingKernel);
                s_calls.Should().Be(0, $"shape ({string.Join(", ", dims)}) is empty");
            }
        }

        [TestMethod]
        public void ExecuteGeneric_OverAnEmptyIteration_MakesNoKernelCall()
        {
            using var a = np.zeros(new Shape(11, 0, 14), np.float64);
            using var w = np.ones(new Shape(11, 0, 14), np.float64);
            using var num = np.zeros(new Shape(11, 0), np.float64);
            using var scl = np.zeros(new Shape(11, 0), np.float64);
            using var it = ReduceLastAxis(a, w, num, scl);
            s_calls = 0;
            it.ExecuteGeneric(new CountingLoop());
            s_calls.Should().Be(0);
        }

        [TestMethod]
        public void ExecuteReducing_OverAnEmptyIteration_ReturnsTheInitialValueWithoutACall()
        {
            using var a = np.zeros(new Shape(11, 0, 14), np.float64);
            using var w = np.ones(new Shape(11, 0, 14), np.float64);
            using var num = np.zeros(new Shape(11, 0), np.float64);
            using var scl = np.zeros(new Shape(11, 0), np.float64);
            using var it = ReduceLastAxis(a, w, num, scl);
            s_calls = 0;
            it.ExecuteReducing(new CountingReduce(), 7L).Should().Be(7L, "the reduction over no element is its start");
            s_calls.Should().Be(0);
        }

        [TestMethod]
        public void Average_EmptyOperandAlongANonEmptyAxis_ReadsNoElementAndIsNumPysEmptyResult()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs VirtualAlloc (Windows).");
            // One float64 whose last byte is the last accessible one, and two empty (11, 0, 14) views of it whose first
            // element would be the one PAST it: the address a kernel starts reading at is the high fence, so the first
            // read of an element that does not exist faults. (np.frombuffer of 0 bytes allocates a fresh empty array, so
            // the views are built over the one-element buffer instead.)
            using var fence = new FencedMemory(pages: 1);
            byte* last = fence.Place(sizeof(double), atEnd: true);
            using var cell = np.frombuffer((IntPtr)last, sizeof(double), np.float64);
            var pastTheEnd = new Shape(new long[] { 11, 0, 14 }, new long[] { 0, 0, 1 }, 1, 1);
            using var a = new NDArray(cell.Storage, pastTheEnd);
            using var w = new NDArray(cell.Storage, pastTheEnd);

            // NumPy 2.4.2: np.average(np.zeros((11, 0, 14)), axis=2, weights=np.ones((11, 0, 14)), returned=True) is a pair
            // of empty (11, 0) float64 arrays; axis 0 gives (0, 14); the tuple (0, 2) gives (0,).
            var (avg, scl) = np.average_returned(a, 2, w);
            using (avg)
            using (scl)
            {
                avg.shape.Should().Equal(11, 0);
                scl.shape.Should().Equal(11, 0);
                avg.dtype.Should().Be(np.float64);
            }

            var (avg0, scl0) = np.average_returned(a, 0, w);
            using (avg0)
            using (scl0)
                avg0.shape.Should().Equal(0, 14);

            using var avgTuple = np.average(a, new[] { 0, 2 }, w);
            avgTuple.shape.Should().Equal(0);
        }
    }
}
