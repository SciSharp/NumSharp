using System;

namespace NumSharp
{
    public partial struct Shape
    {
        // Ports of NumPy's stride-permutation machinery (numpy/_core/src/multiarray/shape.c):
        // PyArray_CreateSortedStridePerm (single array) and PyArray_CreateMultiSortedStridePerm
        // (the multi-operand KEEPORDER vote PyArray_ConcatenateArrays lays its output out with).
        // Together they answer "which axis varies fastest in memory" — the full KEEPORDER
        // semantics that OrderResolver's binary C/F answer cannot express (a 3-D transpose keeps
        // its exact stride ORDER; a broadcast's stride-0 axes sort slowest). They live on Shape,
        // beside TryNocopyReshape, because they are pure dimension/stride computations — exactly
        // the concern this struct owns.
        //
        // Strides here are in ELEMENTS (NumSharp's convention; NumPy's are bytes). Every
        // comparison is a |stride| ordering WITHIN one array, so the unit scales out and the two
        // conventions produce identical permutations.

        /// <summary>
        ///     <c>PyArray_CreateSortedStridePerm</c>: axis indices sorted descending by |stride|,
        ///     equal magnitudes keeping their original axis order (NumPy's comparator tie-breaks
        ///     on the perm index — "C-order is the default in the face of ambiguity").
        /// </summary>
        internal static int[] SortedStridePerm(long[] strides)
        {
            int ndim = strides.Length;
            var perm = new int[ndim];
            for (int i = 0; i < ndim; i++)
                perm[i] = i;

            // Stable insertion sort (ndim is tiny); strict '<' keeps equal-|stride| axes in
            // original order, matching the qsort comparator's perm tie-break.
            for (int i = 1; i < ndim; i++)
            {
                int ax = perm[i];
                long s = Math.Abs(strides[ax]);
                int j = i - 1;
                while (j >= 0 && Math.Abs(strides[perm[j]]) < s)
                {
                    perm[j + 1] = perm[j];
                    j--;
                }

                perm[j + 1] = ax;
            }

            return perm;
        }

        /// <summary>
        ///     Build the element strides of a freshly-allocated dense array of
        ///     <paramref name="dims"/> laid out along <paramref name="perm"/> — the allocation
        ///     step NumPy runs after either perm sort (<c>PyArray_NewLikeArray</c> KEEPORDER,
        ///     <c>PyArray_ConcatenateArrays</c>): the axis the perm ranks LAST varies fastest.
        /// </summary>
        internal static long[] StridesForPerm(long[] dims, int[] perm)
        {
            var strides = new long[dims.Length];
            long s = 1;
            for (int idim = dims.Length - 1; idim >= 0; idim--)
            {
                int iperm = perm[idim];
                strides[iperm] = s;
                s *= dims[iperm];
            }

            return strides;
        }

        /// <summary>
        ///     <c>PyArray_CreateMultiSortedStridePerm</c>: the stable insertion sort with
        ///     per-pair multi-operand ambiguity voting (KEEPORDER exactly as the NpyIter
        ///     resolves it). A size-1 axis never votes in any operand; the vote scans ALL
        ///     operands per pair and C-order wins conflicts between operands (<c>shouldswap</c>
        ///     is cleared even after the comparison stopped being ambiguous, but only ever SET
        ///     while it still is — both quirks are load-bearing and ported verbatim).
        /// </summary>
        internal static int[] MultiSortedStridePerm(Shape[] shapes, int ndim)
        {
            var perm = new int[ndim];
            for (int i = 0; i < ndim; i++)
                perm[i] = i;

            for (int i0 = 1; i0 < ndim; i0++)
            {
                int ipos = i0;
                int axJ0 = perm[i0];

                for (int i1 = i0 - 1; i1 >= 0; i1--)
                {
                    bool ambig = true, shouldSwap = false;
                    int axJ1 = perm[i1];

                    for (int k = 0; k < shapes.Length; k++)
                    {
                        ref readonly var sh = ref shapes[k];
                        if (sh.dimensions[axJ0] != 1 && sh.dimensions[axJ1] != 1)
                        {
                            if (Math.Abs(sh.strides[axJ0]) <= Math.Abs(sh.strides[axJ1]))
                                shouldSwap = false; // C-order wins conflicts, even when already decided
                            else if (ambig)
                                shouldSwap = true;  // only set swap while it's still ambiguous

                            ambig = false;
                        }
                    }

                    // Unambiguous: either shift the insertion point to i1 or stop looking.
                    if (!ambig)
                    {
                        if (shouldSwap)
                            ipos = i1;
                        else
                            break;
                    }
                }

                if (ipos != i0)
                {
                    for (int i1 = i0; i1 > ipos; i1--)
                        perm[i1] = perm[i1 - 1];
                    perm[ipos] = axJ0;
                }
            }

            return perm;
        }

        /// <summary>
        ///     The KEEPORDER shape for a fresh OWNED allocation mirroring THIS shape's memory
        ///     order (NumPy <c>PyArray_NewLikeArray</c> with <c>NPY_KEEPORDER</c> — what
        ///     <c>copy(order='K')</c>/<c>astype(order='K')</c> allocate). C-/F-contiguous
        ///     sources come back as plain C/F shapes; a 3-D transpose keeps its exact stride
        ///     order (a neither-contiguous owned array), and a broadcast's stride-0 axes sort
        ///     slowest (probed 2.4.2: copying a <c>(4, 3)</c> row-broadcast with order='K'
        ///     yields byte strides <c>(8, 32)</c> — F-contiguous).
        /// </summary>
        internal readonly Shape KeepOrder()
        {
            var dims = (long[])dimensions.Clone();
            return new Shape(dims, StridesForPerm(dims, SortedStridePerm(strides)));
        }

        /// <summary>
        ///     The output shape <c>PyArray_ConcatenateArrays</c> allocates: dims =
        ///     <paramref name="outDims"/>, strides built along the multi-operand KEEPORDER vote
        ///     over <paramref name="inputs"/>. All-C inputs yield C; all-F (not also C) yield F;
        ///     stacked F operands — (1, m, n) expansions whose size-1 axis never votes — yield
        ///     the NEITHER-contiguous perm NumPy produces (probed 2.4.2:
        ///     <c>np.stack([f, f])</c> of F-contiguous (3, 4) → byte strides (96, 8, 24)).
        /// </summary>
        internal static Shape ConcatOutputShape(long[] outDims, Shape[] inputs)
        {
            // All-C fast path — parity-exact AND load-bearing for perf: when every input is
            // C-contiguous the vote's answer is the identity perm by construction (a C pair
            // compares no-swap; a size-1 axis pair is ambiguous, and ambiguity resolves to the
            // original C order). But the insertion sort cannot EXIT early through ambiguity, so
            // size-1-dominated shapes — np.r_'s uncapped ndmin=100000 padding — degrade it to
            // O(ndim²·narrays) (measured 12 s; NumPy never sees this only because it caps ndim
            // at 64). The flags check answers the same thing in O(ndim·narrays).
            bool allC = true;
            for (int k = 0; k < inputs.Length && allC; k++)
                allC = inputs[k].IsContiguous;

            var dims = (long[])outDims.Clone();
            if (allC)
                return new Shape(dims);

            return new Shape(dims, StridesForPerm(dims, MultiSortedStridePerm(inputs, dims.Length)));
        }

        /// <summary>
        ///     The layout NumPy's ufunc machinery allocates its output in — NpyIter's: the operands are broadcast over
        ///     <paramref name="dims"/> (<c>npyiter_fill_axisdata</c>), the iterator axes sorted by
        ///     <c>npyiter_find_best_axis_ordering</c>, and the output laid out dense along that order
        ///     (<c>npyiter_new_temp_array</c>). Use it wherever NumSharp reproduces the RESULT of a chain of NumPy ufunc
        ///     calls with one kernel: the result's strides (and so its C/F flags) are then NumPy's, not whatever order
        ///     the kernel happened to write in.
        /// </summary>
        /// <param name="dims">The broadcast output shape (the operands must broadcast to it; not checked). The returned
        ///     shape ADOPTS this array as its dimensions — pass a fresh one (or one nobody will ever write to): a Shape is
        ///     immutable only as long as the arrays it holds are.</param>
        /// <param name="operands">The ufunc's INPUT operands (0-d ones and NumPy scalars included: they never vote).
        ///     Each is read through its own dims/strides, right-aligned against <paramref name="dims"/>. An array
        ///     converts implicitly; a single operand needs no array (<c>new ReadOnlySpan&lt;Shape&gt;(in s)</c>).</param>
        /// <returns>A fresh, offset-0, dense shape of <paramref name="dims"/>.</returns>
        /// <remarks>
        ///     <para><b>How it differs from the other two votes in this file.</b> An operand's stride counts as 0 — it
        ///     abstains — on every axis where the broadcast extent is 1, the operand lacks the axis, its own extent is 1,
        ///     OR its stride really is 0 (an <c>np.broadcast_to</c> view): <c>npyiter_fill_axisdata</c> writes the
        ///     stride as is and the sort skips zero strides. <see cref="MultiSortedStridePerm"/> (concatenate's vote)
        ///     abstains only on extent-1 axes, so a broadcast view's stride-0 axis there sorts innermost, and
        ///     <see cref="KeepOrder"/> (<c>np.copy</c>'s order='K') sorts it innermost as well — which is why a copy of a
        ///     row-broadcast is F-contiguous while a ufunc over it (<c>np.broadcast_to(a, (3, 4)) + 0</c>) is C (probed
        ///     2.4.2).</para>
        ///     <para><b>The sort.</b> A stable insertion sort over the iterator's axes, innermost first (NpyIter keeps
        ///     its AXISDATA reversed from C order): an axis moves inward past another when, for the first operand that
        ///     sees both (neither stride 0), its |stride| is smaller; any LATER operand whose |strides| say otherwise
        ///     cancels the move ("C-order wins conflicts between operands"); a pair no operand sees is skipped and the
        ///     scan continues outward. Ported line for line — the conflict and skip rules are what make, e.g., a
        ///     transposed series broadcast against an F-ordered x come out with the series' axes outermost.</para>
        ///     <para><b>Allocation.</b> The innermost sorted axis gets stride 1, each further one the product of the
        ///     extents inside it — so an extent-1 axis gets whatever stride its sorted position gives it (NumPy's own
        ///     answer for it is path-dependent: its trivial-loop fast path fills C/F strides instead; flags never read
        ///     an extent-1 axis's stride). Strides are in ELEMENTS, NumSharp's convention; every comparison is within
        ///     one operand, so NumPy's byte strides sort identically.</para>
        ///     <para><b>Cost.</b> It runs on every call that lays out a non-C result (the polynomial evaluation and
        ///     mapdomain paths), so its only heap allocation is the strides array the result keeps: the vote table and
        ///     the permutation live on the stack up to <see cref="NpyIterStackLimit"/> entries, and the dims are adopted
        ///     rather than copied.</para>
        /// </remarks>
        internal static Shape NpyIterOutputShape(long[] dims, ReadOnlySpan<Shape> operands)
        {
            int nd = dims.Length;
            if (nd == 0)
                return NewScalar();
            int nop = operands.Length;

            // npyiter_fill_axisdata: the stride each operand contributes per ITERATOR axis i (array axis nd-1-i);
            // 0 = abstains (the four broadcast cases of the remarks). Stack-resident for any realistic rank × operand
            // count; the heap fallback keeps an absurd rank (NumSharp has no 64-dimension cap) from blowing the stack.
            int cells = nop * nd;
            Span<long> istr = cells <= NpyIterStackLimit ? stackalloc long[cells] : new long[cells];
            for (int iop = 0; iop < nop; iop++)
            {
                var od = operands[iop].dimensions;
                var os = operands[iop].strides;
                int ond = od.Length;
                for (int iax = 0; iax < nd; iax++)
                {
                    int ax = nd - 1 - iax;
                    int k = ax - (nd - ond);
                    istr[iop * nd + iax] = dims[ax] == 1 || k < 0 || od[k] == 1 ? 0 : os[k];
                }
            }

            // npyiter_find_best_axis_ordering: perm[i] = the iterator axis at sorted position i (innermost first).
            Span<int> perm = nd <= NpyIterStackLimit ? stackalloc int[nd] : new int[nd];
            for (int i = 0; i < nd; i++)
                perm[i] = i;
            for (int axI0 = 1; axI0 < nd; axI0++)
            {
                int axIpos = axI0;
                int axJ0 = perm[axI0];
                for (int axI1 = axI0 - 1; axI1 >= 0; axI1--)
                {
                    bool ambig = true, shouldSwap = false;
                    int axJ1 = perm[axI1];
                    for (int iop = 0; iop < nop; iop++)
                    {
                        long s0 = istr[iop * nd + axJ0], s1 = istr[iop * nd + axJ1];
                        if (s0 != 0 && s1 != 0)
                        {
                            // NumPy: "Set swap even if it's not ambiguous already, because in the case of conflicts
                            // between different operands, C-order wins" — and only SET it while still ambiguous.
                            if (Math.Abs(s1) <= Math.Abs(s0))
                                shouldSwap = false;
                            else if (ambig)
                                shouldSwap = true;
                            ambig = false;
                        }
                    }

                    // Unambiguous: move the insertion point or stop; ambiguous: keep scanning outward.
                    if (!ambig)
                    {
                        if (shouldSwap)
                            axIpos = axI1;
                        else
                            break;
                    }
                }

                if (axIpos != axI0)
                {
                    for (int axI1 = axI0; axI1 > axIpos; axI1--)
                        perm[axI1] = perm[axI1 - 1];
                    perm[axIpos] = axJ0;
                }
            }

            // npyiter_new_temp_array (no op_axes): dense strides, innermost sorted axis first.
            var strides = new long[nd];
            long stride = 1;
            for (int idim = 0; idim < nd; idim++)
            {
                int ax = nd - 1 - perm[idim];
                strides[ax] = stride;
                stride *= dims[ax];
            }

            return new Shape(dims, strides);
        }

        /// <summary>
        ///     Largest vote table (operands × rank) and permutation <see cref="NpyIterOutputShape"/> keeps on the stack —
        ///     256 longs = 2 KB, far above any ufunc's operand count times a realistic rank; beyond it the tables go to
        ///     the heap rather than risk the stack.
        /// </summary>
        private const int NpyIterStackLimit = 256;

        /// <summary>
        ///     The axes of a DENSE layout (no stride-0 axis of extent &gt; 1 — e.g. one <see cref="NpyIterOutputShape"/>
        ///     built) in memory order, outermost first: <c>np.transpose(a, perm)</c> of an array laid out this way is
        ///     C-contiguous, and a C-contiguous buffer of the transposed dims IS this layout's buffer — the identity
        ///     that lets a kernel which can only write C order produce this layout by writing the transposed view.
        /// </summary>
        /// <returns>The permutation, outermost axis first (length <see cref="NDim"/>).</returns>
        /// <remarks>
        ///     Ties — an extent-1 axis shares its stride with a neighbour — keep their original order (a stable sort).
        ///     An extent-1 axis is only ever read at index 0, so where it lands does not change any element's address:
        ///     every non-unit axis has a distinct stride in a dense layout, and their relative order is all that the
        ///     identity above needs.
        /// </remarks>
        internal int[] DenseAxisOrder()
        {
            int nd = dimensions.Length;
            var perm = new int[nd];
            for (int i = 0; i < nd; i++)
                perm[i] = i;
            // Stable insertion sort by stride, descending: ranks are tiny, and stability is what keeps ties put.
            for (int i = 1; i < nd; i++)
            {
                int a = perm[i];
                long s = strides[a];
                int j = i - 1;
                while (j >= 0 && strides[perm[j]] < s)
                {
                    perm[j + 1] = perm[j];
                    j--;
                }
                perm[j + 1] = a;
            }
            return perm;
        }
    }
}
