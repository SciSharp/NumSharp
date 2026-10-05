namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     NumPy's lightweight "raw array iteration" setup (<c>numpy/_core/src/multiarray/dtype_transfer.c</c>,
    ///     <c>PyArray_PrepareThreeRawArrayIter</c>): the axis order, stride signs and coalescing that NumPy's
    ///     assignment loops (<c>raw_array_wheremasked_assign_array</c> for <c>np.copyto(..., where=)</c>) walk
    ///     their operands in. Unlike <c>NpyIter</c> it looks only at the FIRST operand's strides.
    /// </summary>
    /// <remarks>
    ///     The order is observable only when operands overlap. NumPy's assignment copies an overlapping source
    ///     first unless the destination is 1-D with both runs in the same direction, so in practice it decides the
    ///     result of a 1-D overlapping copy — the reason this is a port and not "any fast order".
    /// </remarks>
    internal static unsafe class RawArrayIter
    {
        /// <summary>
        ///     Prepares three broadcast operands (A = the destination, B and C read alongside it) for a raw walk: the axes
        ///     sorted by A's absolute strides (the smallest becomes the innermost, index 0; ties keep the LATER axis
        ///     innermost), every axis on which A's stride is negative reversed for all three operands (their data pointers
        ///     moved to that axis's last element), and adjacent axes coalesced where all three operands allow it.
        /// </summary>
        /// <param name="ndim">The number of axes the operands were broadcast to.</param>
        /// <param name="shape">The broadcast shape (<paramref name="ndim"/> entries).</param>
        /// <param name="dataA">A's data pointer; moved by the reversal of negative-stride axes.</param>
        /// <param name="stridesA">A's BYTE strides.</param>
        /// <param name="dataB">B's data pointer; moved alongside A's.</param>
        /// <param name="stridesB">B's BYTE strides (0 on broadcast axes).</param>
        /// <param name="dataC">C's data pointer; moved alongside A's.</param>
        /// <param name="stridesC">C's BYTE strides (0 on broadcast axes).</param>
        /// <param name="outShape">Receives the prepared extents, innermost FIRST; room for <c>max(ndim, 1)</c> entries.</param>
        /// <param name="outStridesA">Receives A's prepared strides (non-negative), innermost first.</param>
        /// <param name="outStridesB">Receives B's prepared strides, innermost first.</param>
        /// <param name="outStridesC">Receives C's prepared strides, innermost first.</param>
        /// <returns>The prepared number of axes, at least 1. A 0-d walk is one axis of extent 1 with zero strides; an empty
        ///     operand is one axis of extent 0 (NumPy detects it during the reversal and stops there).</returns>
        /// <remarks>
        ///     A literal port, quirks included: NumPy sorts with <c>PyArray_CreateSortedStridePerm</c> (absolute strides
        ///     descending, ties by ascending axis) and reads the result back to front, so among equal strides the last
        ///     axis lands innermost; a size-1 axis is dropped during coalescing whatever its strides.
        /// </remarks>
        internal static int PrepareThree(int ndim, long* shape,
            ref byte* dataA, long* stridesA,
            ref byte* dataB, long* stridesB,
            ref byte* dataC, long* stridesC,
            long* outShape, long* outStridesA, long* outStridesB, long* outStridesC)
        {
            // Special case 0 and 1 dimensions.
            if (ndim == 0)
            {
                outShape[0] = 1;
                outStridesA[0] = 0;
                outStridesB[0] = 0;
                outStridesC[0] = 0;
                return 1;
            }

            if (ndim == 1)
            {
                long sa = stridesA[0], sb = stridesB[0], sc = stridesC[0], n = shape[0];
                outShape[0] = n;
                // "Always make a positive stride for the first operand"
                if (sa >= 0)
                {
                    outStridesA[0] = sa;
                    outStridesB[0] = sb;
                    outStridesC[0] = sc;
                }
                else
                {
                    dataA += sa * (n - 1);
                    dataB += sb * (n - 1);
                    dataC += sc * (n - 1);
                    outStridesA[0] = -sa;
                    outStridesB[0] = -sb;
                    outStridesC[0] = -sc;
                }
                return 1;
            }

            // Sort the axes based on the destination strides: PyArray_CreateSortedStridePerm's order is |stride|
            // descending with ties by ascending axis; an insertion sort is stable, so ties keep the axis order.
            int* perm = stackalloc int[ndim];
            for (int k = 0; k < ndim; k++)
                perm[k] = k;
            for (int k = 1; k < ndim; k++)
            {
                int axis = perm[k];
                long key = Abs(stridesA[axis]);
                int j = k - 1;
                while (j >= 0 && Abs(stridesA[perm[j]]) < key)
                {
                    perm[j + 1] = perm[j];
                    j--;
                }
                perm[j + 1] = axis;
            }

            // Read back to front: out axis i is the (ndim - 1 - i)-th of the sorted order, so axis 0 is innermost.
            for (int i = 0; i < ndim; i++)
            {
                int iperm = perm[ndim - i - 1];
                outShape[i] = shape[iperm];
                outStridesA[i] = stridesA[iperm];
                outStridesB[i] = stridesB[iperm];
                outStridesC[i] = stridesC[iperm];
            }

            // Reverse any negative strides of operand A.
            for (int i = 0; i < ndim; i++)
            {
                long sa = outStridesA[i], sb = outStridesB[i], sc = outStridesC[i], n = outShape[i];
                if (sa < 0)
                {
                    dataA += sa * (n - 1);
                    dataB += sb * (n - 1);
                    dataC += sc * (n - 1);
                    outStridesA[i] = -sa;
                    outStridesB[i] = -sb;
                    outStridesC[i] = -sc;
                }

                // Detect 0-size arrays here.
                if (n == 0)
                {
                    outShape[0] = 0;
                    outStridesA[0] = 0;
                    outStridesB[0] = 0;
                    outStridesC[0] = 0;
                    return 1;
                }
            }

            // Coalesce any dimensions where possible.
            int c = 0;
            for (int j = 1; j < ndim; j++)
            {
                if (outShape[c] == 1)
                {
                    // Drop axis c.
                    outShape[c] = outShape[j];
                    outStridesA[c] = outStridesA[j];
                    outStridesB[c] = outStridesB[j];
                    outStridesC[c] = outStridesC[j];
                }
                else if (outShape[j] == 1)
                {
                    // Drop axis j.
                }
                else if (outStridesA[c] * outShape[c] == outStridesA[j]
                         && outStridesB[c] * outShape[c] == outStridesB[j]
                         && outStridesC[c] * outShape[c] == outStridesC[j])
                {
                    // Coalesce axes c and j.
                    outShape[c] *= outShape[j];
                }
                else
                {
                    // Can't coalesce, go to next c.
                    c++;
                    outShape[c] = outShape[j];
                    outStridesA[c] = outStridesA[j];
                    outStridesB[c] = outStridesB[j];
                    outStridesC[c] = outStridesC[j];
                }
            }

            return c + 1;
        }

        /// <summary>The absolute value of a stride (a stride is never <see cref="long.MinValue"/>).</summary>
        /// <param name="v">The stride.</param>
        /// <returns>|<paramref name="v"/>|.</returns>
        private static long Abs(long v) => v < 0 ? -v : v;
    }
}
