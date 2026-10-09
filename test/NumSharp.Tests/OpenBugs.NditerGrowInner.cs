using System.Linq;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — np.nditer's grow_inner does not grow an operand that needs no buffer
    // =====================================================================
    //
    // Found 2026-10-09 by the np.nditer unit-test expansion (np.nditer.Protocol.Test.cs), probed against NumPy 2.4.2:
    //
    //     it = np.nditer(np.arange(10), flags=['buffered', 'external_loop', 'grow_inner'], buffersize=3)
    //     [len(c) for c in it]      # NumPy: [10]        NumSharp: [3, 3, 3, 1]
    //     it = np.nditer(np.zeros((100, 100)), flags=['buffered', 'external_loop', 'grow_inner'])
    //     [len(c) for c in it]      # NumPy: [10000]     NumSharp: [8192, 1808]
    //
    // NumPy's npyiter_find_buffering_setup (nditer_constr.c) caps the transfer size at the buffer size only when an operand
    // is really buffered or GROWINNER is off: "if (iterator_must_buffer || !(itflags & NPY_ITFLAG_GROWINNER)) best_size =
    // best_coresize * (maximum_size / best_coresize)". An operand without a cast whose strides collapse over the core is
    // BUFNEVER, so with grow_inner the whole core (here the whole coalesced iteration) is one inner loop. NumSharp's general
    // iterator sizes every window with NDIterBufferManager.ComputeTransferSize, which always caps at the buffer size: the
    // values are right, only the chunking differs (an external-loop caller sees more, shorter chunks). Without grow_inner
    // both libraries cap (np_nditer_Protocol_Test.Buffered_WithoutGrowInner_CapsEveryChunkAtTheBufferSize), and a cast
    // operand stays capped under grow_inner in both (GrowInner_CastOperand_KeepsBufferSizedChunks).
    //
    // The fix is the core-size search of npyiter_find_buffering_setup (the per-operand single-stride / cost model, the
    // grow_inner exemption), ported into the general iterator's window sizing. It changes the chunk sizes every buffered
    // NDIter caller sees, so it is its own piece of work; the reductions already run their own ports of that search.
    //
    // EXPECTED (when fixed): both tests pass; remove their [OpenBugs].
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     NumPy walks a contiguous, uncast operand under <c>grow_inner</c> in ONE chunk longer than the buffer:
        ///     <c>arange(10)</c> with a buffer of 3 is <c>[10]</c>.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "nditer-grow-inner")]
        public void Nditer_GrowInner_UncastOperand_IsOneChunk()
        {
            using var a = np.arange(10);
            using var it = np.nditer(a, new[] { "buffered", "external_loop", "grow_inner" }, buffersize: 3);
            it.Select(v => v[0].size).Should().Equal(10L);
        }

        /// <summary>
        ///     The same at NumPy's default buffer size: a contiguous <c>(100, 100)</c> array coalesces to one axis of 10000 and
        ///     walks it as ONE chunk, past the 8192-element buffer.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "nditer-grow-inner")]
        public void Nditer_GrowInner_ContiguousGrid_IsOneChunkPastTheBufferSize()
        {
            using var a = np.zeros(new Shape(100, 100));
            using var it = np.nditer(a, new[] { "buffered", "external_loop", "grow_inner" });
            it.Select(v => v[0].size).Should().Equal(10000L);
        }
    }
}
