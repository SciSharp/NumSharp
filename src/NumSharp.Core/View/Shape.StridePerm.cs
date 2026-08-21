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
    }
}
