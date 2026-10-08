using System;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — a BUFFERED reduction through np.nditer computes wrong values
    // =====================================================================
    //
    // Found 2026-10-08 while fixing the open-bugs rundown's bug 6 (np.nditer over strings), by a probe of NumPy's
    // documented reduction idiom:
    //
    //     it = np.nditer([a, None], flags=['reduce_ok', 'buffered', 'delay_bufalloc'],
    //                    op_flags=[['readonly'], ['readwrite', 'allocate']], op_axes=[None, [0, 1, -1]])
    //     with it:
    //         it.operands[1][...] = 0
    //         it.reset()
    //         for x, y in it:
    //             y[...] += x
    //
    // NumPy 2.4.2 returns the sum over the last axis, [[6, 22, 38], [54, 70, 86]] for a = arange(24).reshape(2, 3, 4),
    // at EVERY buffer size (0 / 3 / 5 / 7 / 13 probed, with and without op_dtypes=float64), with an allocated or a given
    // output. NumSharp's buffered REDUCE path is a homegrown double loop that predates the windowed buffering
    // (NDIter.SetupBufferedReduction / BufferedReduceIternext / CopyReduceBuffersToArrays): it supports one non-reduce
    // axis, writes a stale buffer element back over the accumulated output ([[0, 22, 38], ...] at the default buffer size
    // after the reset), and loses or misplaces partial sums when a buffer is smaller than the iteration
    // ([[12, 21, 30], [87, 57, 66]] at buffer size 3). The UNBUFFERED idiom is right (NDIterNumPyParityTests and the
    // nditer_protocol oracle tier's "reduce" section gate it).
    //
    // The fix is a port of NumPy 2.4.2's buffered-reduce design, a separate piece of work: npyiter_find_buffering_setup's
    // core / outer split (coresize, outerdim, per-operand BUFNEVER / single-stride / reduce flags, the inner and
    // reduce-outer strides), npyiter_copy_to_buffers / npyiter_copy_from_buffers with the reduce outer loop and the
    // reusable-buffer rule, npyiter_buffered_reduce_iternext, and npyiter_goto_iterindex's core offset. No internal
    // NumSharp caller builds a BUFFERED + REDUCE iterator (every reduction runs its own engine), so only np.nditer reaches
    // the legacy path.
    //
    // EXPECTED (when fixed): each test passes; remove its [OpenBugs] and add the buffered cases to the oracle's
    // nditer_protocol "reduce" section (gen_oracle.py gen_nditer_protocol), which leaves buffering out until then.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>The input every case reduces: <c>np.arange(24).reshape(2, 3, 4)</c> (int64).</summary>
        private static NDArray NditerReduceInput() => np.arange(24).reshape(2, 3, 4);

        /// <summary>NumPy's answer for every case: the sum over the last axis.</summary>
        private static readonly long[] NditerReduceExpected = { 6, 22, 38, 54, 70, 86 };

        /// <summary>
        ///     NumPy's documented idiom with the default buffer size: an allocated output, zeroed, then <c>reset()</c> (which
        ///     allocates the delayed buffers), then <c>y[...] += x</c> per element.
        /// </summary>
        /// <param name="bufferSize">The <c>buffersize</c> argument (0 = NumPy's default).</param>
        /// <param name="castToDouble">Iterate as float64 (<c>op_dtypes=[float64, float64]</c>, <c>casting='same_kind'</c>):
        ///     the input is then buffered with a cast.</param>
        /// <returns>The reduced output's values, flattened.</returns>
        private static double[] RunDelayedReduceIdiom(long bufferSize, bool castToDouble)
        {
            using var a = NditerReduceInput();
            DType[] dtypes = castToDouble ? new[] { np.float64, np.float64 } : null;
            using var it = np.nditer(new NDArray[] { a, null }, new[] { "reduce_ok", "buffered", "delay_bufalloc" },
                new[] { new[] { "readonly" }, new[] { "readwrite", "allocate" } }, dtypes,
                casting: castToDouble ? "same_kind" : "safe", op_axes: new[] { null, new[] { 0, 1, -1 } },
                buffersize: bufferSize);
            using var output = it.operands[1];
            np.copyto(output, 0, casting: "unsafe");
            it.reset();
            foreach (var v in it)
                np.add(v[1], v[0], @out: v[1]);

            using var flat = output.astype(np.float64).flatten();
            return flat.ToArray<double>();
        }

        /// <summary>The delayed-allocation idiom at the default buffer size (the whole iteration fits one buffer).</summary>
        [TestMethod, OpenBugs(IssueUrl = "nditer-buffered-reduce")]
        public void Nditer_BufferedReduce_DelayedIdiom_DefaultBufferSize_SumsTheLastAxis()
        {
            RunDelayedReduceIdiom(0, castToDouble: false).Should().Equal(Array.ConvertAll(NditerReduceExpected, v => (double)v));
        }

        /// <summary>
        ///     The same idiom with a buffer of 3 elements (shorter than the reduced axis): NumPy's iterator then walks the
        ///     reduced axis in chunks and still sums it.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "nditer-buffered-reduce")]
        public void Nditer_BufferedReduce_DelayedIdiom_SmallBuffer_SumsTheLastAxis()
        {
            RunDelayedReduceIdiom(3, castToDouble: false).Should().Equal(Array.ConvertAll(NditerReduceExpected, v => (double)v));
        }

        /// <summary>The idiom with a float64 loop over the int64 input (a buffered cast), buffer size 5.</summary>
        [TestMethod, OpenBugs(IssueUrl = "nditer-buffered-reduce")]
        public void Nditer_BufferedReduce_DelayedIdiom_CastLoop_SumsTheLastAxis()
        {
            RunDelayedReduceIdiom(5, castToDouble: true).Should().Equal(Array.ConvertAll(NditerReduceExpected, v => (double)v));
        }

        /// <summary>
        ///     A given zeroed output (no delayed allocation), buffer size 7: NumPy returns the same sums; NumSharp loses
        ///     partial sums at the buffer boundaries.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "nditer-buffered-reduce")]
        public void Nditer_BufferedReduce_GivenOutput_SmallBuffer_SumsTheLastAxis()
        {
            using var a = NditerReduceInput();
            using var output = np.zeros(new Shape(2, 3), np.int64);
            using (var it = np.nditer(new[] { a, output }, new[] { "reduce_ok", "buffered" },
                       new[] { new[] { "readonly" }, new[] { "readwrite" } }, op_axes: new[] { null, new[] { 0, 1, -1 } },
                       buffersize: 7))
            {
                foreach (var v in it)
                    np.add(v[1], v[0], @out: v[1]);
            }

            output.flatten().ToArray<long>().Should().Equal(NditerReduceExpected);
        }
    }
}
