using System.Linq;

namespace NumSharp.Tests
{
    // =====================================================================
    // FIXED (2026-10-14) — np.nditer's grow_inner did not grow an operand that needs no buffer (regression pins)
    // =====================================================================
    //
    // Found 2026-10-09 by the np.nditer unit-test expansion (np.nditer.Protocol.Test.cs), probed against NumPy 2.4.2:
    //
    //     it = np.nditer(np.arange(10), flags=['buffered', 'external_loop', 'grow_inner'], buffersize=3)
    //     [len(c) for c in it]      # NumPy: [10]        NumSharp was: [3, 3, 3, 1]
    //     it = np.nditer(np.zeros((100, 100)), flags=['buffered', 'external_loop', 'grow_inner'])
    //     [len(c) for c in it]      # NumPy: [10000]     NumSharp was: [8192, 1808]
    //
    // NumPy's npyiter_find_buffering_setup (nditer_constr.c) caps the transfer size at the buffer size only when an operand
    // is really buffered or GROWINNER is off: "if (iterator_must_buffer || !(itflags & NPY_ITFLAG_GROWINNER)) best_size =
    // best_coresize * (maximum_size / best_coresize)". An operand without a cast whose strides collapse over the core is
    // BUFNEVER, so with grow_inner the whole core (here the whole coalesced iteration) is one inner loop. NumSharp's general
    // iterator sized every window with NDIterBufferManager.ComputeTransferSize, which always capped at the buffer size.
    //
    // FIXED by the port of NumPy 2.4.2's buffering setup (NDIterBufferManager.FindBufferingSetup, strings phase 1 package
    // P1.7g): the cost model, the per-operand single-stride / BUFNEVER decision and the grow_inner exemption. Both tests pass;
    // the oracle's nditer_protocol tier records the chunks of every layout with and without grow_inner (its chunk / chunk2
    // sections). Without grow_inner both libraries cap (np_nditer_Protocol_Test.Buffered_WithoutGrowInner_CapsEveryChunkAtTheBufferSize),
    // and a cast operand stays capped under grow_inner in both (GrowInner_CastOperand_KeepsBufferSizedChunks).
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     NumPy walks a contiguous, uncast operand under <c>grow_inner</c> in ONE chunk longer than the buffer:
        ///     <c>arange(10)</c> with a buffer of 3 is <c>[10]</c>.
        /// </summary>
        [TestMethod]
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
        [TestMethod]
        public void Nditer_GrowInner_ContiguousGrid_IsOneChunkPastTheBufferSize()
        {
            using var a = np.zeros(new Shape(100, 100));
            using var it = np.nditer(a, new[] { "buffered", "external_loop", "grow_inner" });
            it.Select(v => v[0].size).Should().Equal(10000L);
        }
    }
}
