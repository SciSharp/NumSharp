using System;
using System.Collections.Generic;
using System.Linq;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     np.nditer's BUFFERED iteration as the port of NumPy 2.4.2's buffering runs it (NDIterBufferManager.Buffering.cs):
    ///     the setup's chunking (the outer block a fill never crosses, the CONTIG rule), the edges NumPy's reduce double loop
    ///     has and the port keeps (a partial core lost after a seek, a reset that keeps a fill reaching past a new range end,
    ///     the setup remove_multi_index leaves behind), and the two places where NumPy is memory-unsafe and the port is not.
    ///     Every expected value was probed against NumPy 2.4.2 (win-amd64); the oracle's nditer_protocol tier gates the same
    ///     behaviors across layouts, dtypes and buffer sizes.
    /// </summary>
    [TestClass]
    public class np_nditer_BufferedReduce_Test
    {
        /// <summary>The (3, 4) int64 input every reduction here sums: <c>np.arange(12).reshape(3, 4)</c>.</summary>
        private static NDArray Input() => np.arange(12).reshape(3, 4);

        /// <summary>
        ///     NumPy's reduction idiom over <see cref="Input"/> into a zeroed int64 output: <c>for x, y in it: y[...] += x</c>.
        /// </summary>
        /// <param name="outputAxes">The output's op_axes: <c>[0, -1]</c> sums each row (the reduced axis is the inner one),
        ///     <c>[-1, 0]</c> each column (the reduced axis is the outer one).</param>
        /// <param name="cast">Iterate both operands as float64 (<c>casting='unsafe'</c>): both are then buffered.</param>
        /// <param name="position">Runs on the iterator before the walk (a seek or a range), or null.</param>
        /// <param name="flags">Extra global flags (<c>ranged</c> for a range).</param>
        /// <returns>The output's values after the walk and the number of elements it visited.</returns>
        private static (long[] Sums, int Visited) Accumulate(int[] outputAxes, bool cast, Action<np.NDIterator> position,
                                                             params string[] flags)
        {
            using var a = Input();
            using var output = np.zeros(new Shape(outputAxes[1] == -1 ? 3 : 4), np.int64);
            int visited = 0;
            using (var it = np.nditer(new[] { a, output }, new[] { "reduce_ok", "buffered" }.Concat(flags).ToArray(),
                       new[] { new[] { "readonly" }, new[] { "readwrite" } },
                       cast ? new DType[] { np.float64, np.float64 } : null, casting: cast ? "unsafe" : "safe",
                       op_axes: new[] { new[] { 0, 1 }, outputAxes }))
            {
                position?.Invoke(it);
                foreach (var v in it)
                {
                    np.add(v[1], v[0], @out: v[1]);
                    visited++;
                }
            }

            return (output.ToArray<long>(), visited);
        }

        // ------------------------------------------------------------------ NumPy's reduce-loop edges, kept

        /// <summary>
        ///     A seek into the middle of a core restarts the buffer there with a reduce outer size of 0, so the write-back of
        ///     that partial core transfers nothing: a BUFFERED (cast) output loses the partial row's sum (NumPy 2.4.2:
        ///     <c>[0, 22, 38]</c>), an unbuffered one is written in place and keeps it (<c>[5, 22, 38]</c>).
        /// </summary>
        [TestMethod]
        public void SeekInsideACore_BufferedOutputLosesThePartialCore()
        {
            Accumulate(new[] { 0, -1 }, cast: true, it => it.iterindex = 2).Sums.Should().Equal(0L, 22L, 38L);
            Accumulate(new[] { 0, -1 }, cast: false, it => it.iterindex = 2).Sums.Should().Equal(5L, 22L, 38L);
        }

        /// <summary>
        ///     A range set at the current position keeps the buffer NpyIter_Reset finds there (its early return compares only
        ///     the first core's end with the new range end), so the reduce fill made for the whole iteration runs to its end:
        ///     <c>iterrange = (0, 9)</c> visits all 12 elements and sums every row, in NumPy 2.4.2 and here, with and without a
        ///     cast (the write-back at the fill's end included).
        /// </summary>
        [TestMethod]
        public void RangeSetAtTheStart_KeepsTheFillReachingPastTheNewEnd()
        {
            foreach (var cast in new[] { false, true })
            {
                var (sums, visited) = Accumulate(new[] { 0, -1 }, cast, it => it.iterrange = (0, 9), "ranged");
                sums.Should().Equal(new[] { 6L, 22L, 38L }, $"cast={cast}");
                visited.Should().Be(12, $"cast={cast}");
            }
        }

        /// <summary>
        ///     remove_multi_index coalesces the axes but keeps the buffering setup chosen for the uncoalesced ones (NumPy's
        ///     NpyIter_RemoveMultiIndex never runs the setup again): the external loop enabled afterwards hands out chunks of
        ///     the old core, <c>[4, 4, 4]</c>, where an iterator built without a multi-index chunks the same walk
        ///     <c>[5, 5, 2]</c> (a (3, 4) array, float64 cast, buffer of 5 — NumPy 2.4.2's numbers).
        /// </summary>
        [TestMethod]
        public void RemoveMultiIndex_KeepsTheOldCore()
        {
            using var a = Input();
            using (var it = np.nditer(a, new[] { "buffered", "multi_index" }, op_dtypes: new[] { np.float64 }, buffersize: 5))
            {
                it.remove_multi_index();
                it.enable_external_loop();
                it.Select(v => v[0].size).Should().Equal(4L, 4L, 4L);
            }

            using (var it = np.nditer(a, new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.float64 }, buffersize: 5))
                it.Select(v => v[0].size).Should().Equal(5L, 5L, 2L);
        }

        /// <summary>
        ///     The stale outer extent NumPy reads after remove_multi_index merged the setup's outer axis away is what remained of
        ///     it at the range start: a (2, 3, 4) array (a core of one row, the outer axis the 3 rows, a buffer of 8) ranged
        ///     from index 8 — row 2 — keeps an extent of 1, so every later fill is one row, <c>[4, 4, 4, 4]</c>, where a range
        ///     from 4 keeps 2 rows per fill, <c>[8, 8, 4]</c> (NumPy 2.4.2's numbers).
        /// </summary>
        [TestMethod]
        public void RemoveMultiIndex_RangedStart_KeepsWhatRemainedOfTheOuterAxis()
        {
            using var b = np.arange(24).reshape(2, 3, 4);
            foreach (var (start, expected) in new[] { (8L, new[] { 4L, 4L, 4L, 4L }), (4L, new[] { 8L, 8L, 4L }) })
            {
                using var it = np.nditer(b, new[] { "buffered", "multi_index", "ranged" }, op_dtypes: new[] { np.float64 },
                    buffersize: 8);
                it.iterrange = (start, 24);
                it.remove_multi_index();
                it.enable_external_loop();
                it.Select(v => v[0].size).Should().Equal(expected, $"iterrange ({start}, 24)");
            }
        }

        /// <summary>
        ///     remove_multi_index over a buffered cast reduction whose merge is partial (an input summed over axis 1: the two
        ///     inner axes merge, the reduced one does not, so the setup's outer position names another axis): NumPy 2.4.2 keeps
        ///     the old setup and walks wrong elements — 36 of the 40 sums of a (2, 3, 4, 5) input stay 0
        ///     (<c>[60, 63, 66, 0, …, 240, 0, …]</c>), 91 of the 100 of a (5, 3, 4, 5) one. Following the old setup over the new
        ///     axes is not safe either: an earlier model wrote past the output, and a kept setup lets a fill of the (5, 3, 4, 5)
        ///     input that starts inside a block of the reduced axis step on into the next block, adding its first row into
        ///     this block's sums. NumSharp runs the setup again over the merged axes and gives the sums.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void RemoveMultiIndex_PartialMergeOfABufferedReduction_RerunsTheSetup()
        {
            foreach (var lead in new[] { 2, 5 })
            {
                using var x = np.arange(lead * 60).reshape(lead, 3, 4, 5);
                using var output = np.zeros(new Shape(lead, 4, 5), np.int64);
                using (var it = np.nditer(new[] { x, output }, new[] { "reduce_ok", "buffered", "multi_index" },
                           new[] { new[] { "readonly" }, new[] { "readwrite" } }, new DType[] { np.float64, np.float64 },
                           casting: "unsafe", op_axes: new[] { null, new[] { 0, -1, 1, 2 } }))
                {
                    it.remove_multi_index();
                    it.enable_external_loop();
                    foreach (var v in it)
                        np.add(v[1], v[0], @out: v[1]);
                }

                using var expected = np.sum(x, 1);
                output.flatten().ToArray<long>().Should().Equal(expected.flatten().ToArray<long>(), $"input ({lead}, 3, 4, 5)");
            }
        }

        // ------------------------------------------------------------------ the setup's chunking

        /// <summary>
        ///     A fill never crosses into the next block of the axes beyond the outer dimension: a (2, 3, 4) view that does not
        ///     coalesce chunks <c>[8, 4, 8, 4]</c> with a buffer of 10 (a core of 4, two rows per fill, then the block's last
        ///     row), and a (2, 3, 5) one <c>[12, 3, 12, 3]</c> with a buffer of 12 (NumPy 2.4.2's numbers).
        /// </summary>
        [TestMethod]
        public void BufferedChunks_NeverCrossTheOuterBlock()
        {
            using var b = np.arange(40).reshape(2, 4, 5);
            using (var it = np.nditer(b[":, :3, :4"], new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.float64 },
                       buffersize: 10))
                it.Select(v => v[0].size).Should().Equal(8L, 4L, 8L, 4L);
            using (var it = np.nditer(b[":, :3, :"], new[] { "buffered", "external_loop" }, op_dtypes: new[] { np.float64 },
                       buffersize: 12))
                it.Select(v => v[0].size).Should().Equal(12L, 3L, 12L, 3L);
        }

        /// <summary>
        ///     <c>contig</c> makes an operand whose inner stride is not its item size a CAST operand (NumPy's
        ///     <c>NPY_ITER_CONTIG</c> rule): a stepped view under <c>grow_inner</c> is one chunk of 5 without the flag and
        ///     buffer-sized chunks <c>[3, 2]</c> with it, each read through the contiguous buffer (stride 8); an unbuffered
        ///     iterator refuses such an operand with NumPy's TypeError.
        /// </summary>
        [TestMethod]
        public void Contig_BuffersANonContiguousOperand_AndRefusesWithoutBuffering()
        {
            using var a = np.arange(10);
            var v = a["::2"];
            using (var it = np.nditer(v, new[] { "buffered", "external_loop", "grow_inner" }, buffersize: 3))
                it.Select(c => c[0].size).Should().Equal(5L);
            using (var it = np.nditer(v, new[] { "buffered", "external_loop", "grow_inner" }, new[] { "readonly", "contig" },
                       buffersize: 3))
            {
                var chunks = it.Select(c => (c[0].size, c[0].strides[0])).ToList();
                chunks.Should().Equal((3L, 8L), (2L, 8L));
            }

            Action unbuffered = () => np.nditer(v, Array.Empty<string>(), new[] { "readonly", "contig" });
            unbuffered.Should().Throw<TypeError>().WithMessage(
                "Iterator operand required buffering, to be contiguous as requested, but buffering is not enabled");
        }

        // ------------------------------------------------------------------ NumPy's memory-unsafe edges, not kept

        /// <summary>
        ///     A range whose end lies inside the SECOND core of a reduce fill: NumPy 2.4.2 runs that fill (one core and a
        ///     part) as ONE inner loop past the core, so the part's elements land in the previous core's reduce element —
        ///     <c>iterrange = (2, 10)</c> over row sums gives <c>[5, 39, 0]</c> (row 2's 8 and 9 added into row 1), and with a
        ///     cast <c>[0, 39, 0]</c>. The port's inner loop never spans more than a core: the part is the next fill, which
        ///     in the reduce loop has an outer size of 0 — written in place without a cast (<c>[5, 22, 17]</c>, the right
        ///     sums), lost through a cast buffer like every other partial core (<c>[0, 22, 0]</c>).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void RangeEndInsideTheSecondCore_InnerReduction_NeverSpansACore()
        {
            Accumulate(new[] { 0, -1 }, cast: false, it => it.iterrange = (2, 10), "ranged").Sums.Should().Equal(5L, 22L, 17L);
            Accumulate(new[] { 0, -1 }, cast: true, it => it.iterrange = (2, 10), "ranged").Sums.Should().Equal(0L, 22L, 0L);
        }

        /// <summary>
        ///     The same range over COLUMN sums (the reduced axis is the outer one, the output's inner stride is 8 bytes): NumPy
        ///     2.4.2's inner loop runs past the core and past the end of the (4,) output, writing two elements beyond it — a heap
        ///     corruption the Windows heap reports as 0xC0000374, at a later allocation. The port writes nothing outside the
        ///     output and gives the column sums of elements 2 to 9, <c>[12, 14, 8, 10]</c>. Through a cast buffer both libraries
        ///     give <c>[4, 5, 6, 7]</c> (NumPy's overrun stays inside its buffer, whose extra elements are never written back;
        ///     the port loses the same partial cores).
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void RangeEndInsideTheSecondCore_OuterReduction_WritesNothingPastTheOutput()
        {
            Accumulate(new[] { -1, 0 }, cast: false, it => it.iterrange = (2, 10), "ranged").Sums.Should().Equal(12L, 14L, 8L, 10L);
            Accumulate(new[] { -1, 0 }, cast: true, it => it.iterrange = (2, 10), "ranged").Sums.Should().Equal(4L, 5L, 6L, 7L);
        }

        /// <summary>
        ///     <c>iternext()</c> on a <c>delay_bufalloc</c> iterator before the allocating reset: NumPy 2.4.2 copies into its
        ///     NULL buffer and the process ends (an access violation for a cast operand). The port moves the position and
        ///     copies nothing (the operand stays on its array, the element views refuse until the reset), so the walk after
        ///     the reset is the whole iteration, as without a cast, where NumPy survives: <c>True 1</c>, then
        ///     <c>[0.0 … 5.0]</c>.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void DelayedBuffers_IternextBeforeTheReset_MovesWithoutCopying()
        {
            using var a = np.arange(6);
            using var it = np.nditer(a, new[] { "buffered", "delay_bufalloc" }, op_dtypes: new[] { np.float64 });
            it.iternext().Should().BeTrue();
            it.iterindex.Should().Be(1L);
            it.reset();
            var values = new List<double>();
            foreach (var v in it)
                values.Add((double)v[0]);
            values.Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        }
    }
}
