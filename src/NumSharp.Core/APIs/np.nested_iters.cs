using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Create nditers for use in nested loops (NumPy's <c>np.nested_iters</c>). Returns one
        ///     <see cref="NDIterator"/> per entry in <paramref name="axes"/>, outermost first, all
        ///     iterating the SAME operand buffer over different axis subsets. Advancing an outer
        ///     iterator re-bases every inner iterator to its new position — so a
        ///     <c>foreach (var _ in i) foreach (var _ in j) ...</c> walks the array in nested loops.
        /// </summary>
        /// <param name="op">The array to iterate over.</param>
        /// <param name="axes">
        ///     One integer list per nesting level; each is used as the <c>op_axes</c> for that level's
        ///     iterator. Must have at least 2 entries, and no axis may appear in more than one entry.
        /// </param>
        /// <param name="flags">Global iterator flags (see <see cref="np.nditer"/>).</param>
        /// <param name="op_flags">Per-operand flags.</param>
        /// <param name="op_dtypes">Per-operand iteration dtypes.</param>
        /// <param name="order">Iteration order ('C'/'F'/'A'/'K').</param>
        /// <param name="casting">Casting rule.</param>
        /// <param name="buffersize">Buffer size (applied to the innermost level only, as in NumPy).</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nested_iters.html
        ///     <para>
        ///     Parity: the levels are built as NumPy's <c>NpyIter_NestedIters</c> builds them — only the innermost level
        ///     is buffered (and keeps <c>external_loop</c>); with buffering, a requested dtype moves to that level
        ///     (the outer levels iterate the operand as it is); copies and allocated outputs are made by the first
        ///     level and shared by the rest. The value streams and multi-indices match NumPy 2.4.2, fixed-width strings
        ///     included (buffered width and class conversions on the inner level).
        ///     </para>
        /// </remarks>
        public static NDIterator[] nested_iters(
            NDArray op, int[][] axes,
            string[] flags = null, string[][] op_flags = null, DType[] op_dtypes = null,
            char order = 'K', string casting = "safe", long buffersize = 0)
            => nested_iters(new[] { op }, axes, flags, op_flags, op_dtypes, order, casting, buffersize);

        /// <inheritdoc cref="nested_iters(NDArray, int[][], string[], string[][], DType[], char, string, long)"/>
        public static NDIterator[] nested_iters(
            NDArray[] op, int[][] axes,
            string[] flags = null, string[][] op_flags = null, DType[] op_dtypes = null,
            char order = 'K', string casting = "safe", long buffersize = 0)
        {
            // --- axes validation (verbatim NumPy messages, NpyIter_NestedIters order) ---
            if (axes == null)
                throw new ValueError("axes must be a tuple of axis arrays");
            int nnest = axes.Length;
            if (nnest < 2)
                throw new ValueError("axes must have at least 2 entries for nested iteration");

            var used = new HashSet<int>();
            foreach (var item in axes)
            {
                if (item == null)
                    throw new ValueError("Each item in axes must be an integer tuple");
                foreach (var ax in item)
                {
                    // NumPy's pywrap guards the absolute bound here; the against-operand-ndim check is
                    // left to the iterator construction below, which reports NumPy's fuller message.
                    if (ax < 0)
                        throw new ValueError("An axis is out of bounds");
                    if (!used.Add(ax))
                        throw new ValueError("An axis is used more than once");
                }
            }

            if (op == null || op.Length == 0)
                throw new ArgumentException("Must provide at least one operand");
            int nop = op.Length;

            // Only the innermost level is buffered (and keeps external_loop): NpyIter_NestedIters splits flags_inner
            // from the outer levels' flags, and common_dtype is the first level's alone.
            var outerFlags = FilterFlags(flags, "external_loop", "buffered");
            var innerFlags = FilterFlags(flags, "common_dtype");
            bool buffered = flags != null && Array.IndexOf(flags, "buffered") >= 0;

            // NpyIter_NestedIters' per-operand split. Done only for well-formed per-operand arguments (a malformed one is
            // passed through, for NDIterator to raise NumPy's error about it):
            //  - ALLOCATE is dropped from an operand the caller passed (it marks exactly the allocated outputs);
            //  - the innermost level never copies or allocates (its operands are the first level's);
            //  - with buffering and no copy route, the REQUESTED dtype moves to the innermost (buffered) level — the outer
            //    levels iterate the operand as it is, so they need neither a cast nor its buffering — and the outer
            //    levels drop nbo / aligned / contig with it.
            string[][] outerOpFlags = op_flags;
            string[][] innerOpFlags = op_flags;
            DType[] outerDtypes = op_dtypes;
            DType[] innerDtypes = op_dtypes;
            bool[] allocatedOutput = new bool[nop];
            bool flagsWellFormed = op_flags == null || ((op_flags.Length == 1 || op_flags.Length == nop) &&
                                                         Array.TrueForAll(op_flags, f => f != null));
            if (flagsWellFormed && (op_dtypes == null || op_dtypes.Length == nop))
            {
                outerOpFlags = new string[nop][];
                innerOpFlags = new string[nop][];
                outerDtypes = op_dtypes == null ? null : (DType[])op_dtypes.Clone();
                // NumPy starts the innermost requests empty: without buffering it iterates the first level's operands
                // (their copies already hold the requested dtype).
                innerDtypes = op_dtypes == null ? null : new DType[nop];
                for (int i = 0; i < nop; i++)
                {
                    // npyiter_convert_ops: one list for every operand, NumPy's defaults when there is none.
                    var given = op_flags == null ? null : op_flags.Length == 1 ? op_flags[0] : op_flags[i];
                    var f = given != null
                        ? new List<string>(given)
                        : op[i] is null ? new List<string> { "writeonly", "allocate" } : new List<string> { "readonly" };
                    if (op[i] is not null)
                        f.RemoveAll(x => x == "allocate");
                    allocatedOutput[i] = f.Contains("allocate");

                    innerOpFlags[i] = f.FindAll(x => x != "copy" && x != "updateifcopy" && x != "allocate").ToArray();
                    if (buffered && !f.Contains("copy") && !f.Contains("updateifcopy") && !f.Contains("allocate"))
                    {
                        f.RemoveAll(x => x == "nbo" || x == "aligned" || x == "contig");
                        if (op_dtypes != null)
                        {
                            innerDtypes[i] = outerDtypes[i];
                            outerDtypes[i] = null;
                        }
                    }

                    outerOpFlags[i] = f.ToArray();
                }
            }

            var iters = new NDIterator[nnest];
            NDArray[] sharedOps = op;
            for (int inest = 0; inest < nnest; inest++)
            {
                // Every operand uses this level's axes as its op_axes entry — except an allocated output, which the
                // first level broadcasts naturally and every later level walks as a reduction (all axes -1).
                var opAxes = new int[nop][];
                for (int o = 0; o < nop; o++)
                {
                    if (allocatedOutput[o])
                    {
                        if (inest == 0)
                        {
                            opAxes[o] = null;
                        }
                        else
                        {
                            var negones = new int[axes[inest].Length];
                            Array.Fill(negones, -1);
                            opAxes[o] = negones;
                        }
                    }
                    else
                    {
                        opAxes[o] = axes[inest];
                    }
                }

                bool innermost = inest == nnest - 1;
                iters[inest] = new NDIterator(
                    sharedOps,
                    innermost ? innerFlags : outerFlags,
                    innermost ? innerOpFlags : outerOpFlags,
                    innermost ? innerDtypes : outerDtypes,
                    order, casting,
                    opAxes, null,
                    innermost ? buffersize : 0);

                // Share iter0's (possibly allocated/processed) operands with the rest, so every level
                // iterates the SAME buffer — the invariant the base-pointer re-base relies on. The later levels make no
                // copies of their own and resolve no common dtype (NumPy clears copy / updateifcopy / common_dtype).
                if (inest == 0)
                {
                    sharedOps = iters[0].operands;
                    outerFlags = FilterFlags(outerFlags, "common_dtype");
                    if (outerOpFlags != null && !ReferenceEquals(outerOpFlags, op_flags))
                        for (int o = 0; o < nop; o++)
                            outerOpFlags[o] = Array.FindAll(outerOpFlags[o], x => x != "copy" && x != "updateifcopy");
                }
            }

            // Switch all levels to advance-on-entry, link the chain, then run the initial re-base
            // cascade (NumPy's construction-time npyiter_resetbasepointers).
            for (int inest = 0; inest < nnest; inest++)
                iters[inest].SetupNested(inest < nnest - 1 ? iters[inest + 1] : null);
            iters[0].RebaseChildren();

            return iters;
        }

        private static string[] FilterFlags(string[] flags, params string[] remove)
        {
            if (flags == null)
                return null;
            var drop = new HashSet<string>(remove);
            var kept = new List<string>(flags.Length);
            foreach (var f in flags)
                if (!drop.Contains(f))
                    kept.Add(f);
            return kept.ToArray();
        }
    }
}
