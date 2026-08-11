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
        ///     Parity: the <c>multi_index</c> path — the documented primary use — is bit-exact with NumPy
        ///     2.4.2 across single/multiple operands, any axis order, and any nesting depth (coordinates
        ///     AND values pair identically). One KNOWN divergence: WITHOUT <c>multi_index</c>, the pure
        ///     value-stream TRAVERSAL ORDER can differ, because NumSharp's underlying <see cref="NDIterator"/>
        ///     currently reorders <c>op_axes</c> iteration by memory (F-like) regardless of the <c>order</c>
        ///     argument, where NumPy follows <c>order</c> (C/F/K). This is a pre-existing NDIter <c>op_axes</c>
        ///     limitation, not a property of the nesting itself; track <c>multi_index</c> for order parity.
        ///     </para>
        /// </remarks>
        public static NDIterator[] nested_iters(
            NDArray op, int[][] axes,
            string[] flags = null, string[][] op_flags = null, NPTypeCode[] op_dtypes = null,
            char order = 'K', string casting = "safe", long buffersize = 0)
            => nested_iters(new[] { op }, axes, flags, op_flags, op_dtypes, order, casting, buffersize);

        /// <inheritdoc cref="nested_iters(NDArray, int[][], string[], string[][], NPTypeCode[], char, string, long)"/>
        public static NDIterator[] nested_iters(
            NDArray[] op, int[][] axes,
            string[] flags = null, string[][] op_flags = null, NPTypeCode[] op_dtypes = null,
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

            // Outer levels drop external_loop/buffered; the innermost keeps them and gets the
            // buffersize — exactly as NpyIter_NestedIters splits flags_inner from flags.
            var outerFlags = FilterFlags(flags, "external_loop", "buffered");
            var innerFlags = FilterFlags(flags, "common_dtype");

            var iters = new NDIterator[nnest];
            NDArray[] sharedOps = op;
            for (int inest = 0; inest < nnest; inest++)
            {
                // Every operand uses this level's axes as its op_axes entry.
                var opAxes = new int[nop][];
                for (int o = 0; o < nop; o++)
                    opAxes[o] = axes[inest];

                bool innermost = inest == nnest - 1;
                iters[inest] = new NDIterator(
                    sharedOps,
                    innermost ? innerFlags : outerFlags,
                    op_flags, op_dtypes, order, casting,
                    opAxes, null,
                    innermost ? buffersize : 0);

                // Share iter0's (possibly allocated/processed) operands with the rest, so every level
                // iterates the SAME buffer — the invariant the base-pointer re-base relies on.
                if (inest == 0)
                    sharedOps = iters[0].operands;
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
