using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Axis coalescing logic for NpyIter.
    /// Merges adjacent compatible axes to reduce iteration overhead.
    ///
    /// NUMSHARP DIVERGENCE: This implementation supports unlimited dimensions.
    /// Uses StridesNDim for stride array indexing (allocated based on actual ndim).
    /// </summary>
    public static unsafe class NpyIterCoalescing
    {
        /// <summary>
        /// Coalesce adjacent axes that have compatible strides for all operands.
        /// Reduces ndim, improving iteration efficiency.
        /// </summary>
        public static void CoalesceAxes(ref NpyIterState state)
        {
            if (state.NDim <= 1)
                return;

            int writeAxis = 0;
            int newNDim = 1;

            // Access dynamically allocated arrays directly (not fixed arrays)
            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;

            for (int readAxis = 0; readAxis < state.NDim - 1; readAxis++)
            {
                int nextAxis = readAxis + 1;
                long shape0 = shape[writeAxis];
                long shape1 = shape[nextAxis];

                // Check if all operands can be coalesced
                bool canCoalesce = true;

                for (int op = 0; op < state.NOp; op++)
                {
                    long stride0 = strides[op * stridesNDim + writeAxis];
                    long stride1 = strides[op * stridesNDim + nextAxis];

                    // NumPy's exact rule (nditer_api.c npyiter_coalesce_axes): a
                    // size-1 axis is only trivially absorbable when its stride is
                    // 0. Construction guarantees that invariant (fill_axisdata
                    // parity: any iterator axis with Shape[d]==1 gets stride 0 for
                    // every operand), which is what makes the merge rule below
                    // ("take stride1 when stride0==0") correct. The previous
                    // relaxed rule (any size-1 axis) combined with that merge rule
                    // let a size-1 axis carrying a nonzero stride win the merge,
                    // corrupting the iteration order (RemoveMultiIndex on
                    // transposed/sliced views walked the wrong elements).
                    bool opCanCoalesce =
                        (shape0 == 1 && stride0 == 0) ||
                        (shape1 == 1 && stride1 == 0) ||
                        (stride0 * shape0 == stride1);

                    if (!opCanCoalesce)
                    {
                        canCoalesce = false;
                        break;
                    }
                }

                if (canCoalesce)
                {
                    // Merge nextAxis into writeAxis
                    shape[writeAxis] *= shape1;

                    // Update strides (take non-zero stride)
                    for (int op = 0; op < state.NOp; op++)
                    {
                        int baseIdx = op * stridesNDim;
                        long stride0 = strides[baseIdx + writeAxis];
                        long stride1 = strides[baseIdx + nextAxis];

                        if (stride0 == 0)
                            strides[baseIdx + writeAxis] = stride1;
                    }
                }
                else
                {
                    // Move to next write position
                    writeAxis++;
                    if (writeAxis != nextAxis)
                    {
                        shape[writeAxis] = shape[nextAxis];

                        for (int op = 0; op < state.NOp; op++)
                        {
                            int baseIdx = op * stridesNDim;
                            strides[baseIdx + writeAxis] = strides[baseIdx + nextAxis];
                        }
                    }
                    newNDim++;
                }
            }

            // Update state
            state.NDim = newNDim;

            // Reset permutation to identity
            for (int d = 0; d < newNDim; d++)
                perm[d] = (sbyte)d;

            // Set IDENTPERM flag
            state.ItFlags |= (uint)NpyIterFlags.IDENTPERM;

            // Clear HASMULTIINDEX flag since coalescing invalidates original indices
            state.ItFlags &= ~(uint)NpyIterFlags.HASMULTIINDEX;

            // Update inner strides cache after dimension change
            state.UpdateInnerStrides();
        }

        /// <summary>
        /// Remove size-1 axes from the internal representation. Each contributes
        /// exactly one coordinate (always 0) and stride 0 (fill invariant), so
        /// removal never changes the element-visit sequence — it restores a
        /// meaningful innermost axis for EXLOOP/kernels and the buffer-manager
        /// linearity test. NumPy reaches the same state through its
        /// UNCONDITIONAL npyiter_coalesce_axes (the strict trivial branch
        /// absorbs every stride-0 size-1 axis); NumSharp's full coalesce is
        /// gated on all-operands-contiguous, so the non-coalesced branch calls
        /// this instead — without it, a trailing size-1 axis sits innermost and
        /// collapses EXLOOP to one-element inner loops ((N,1) strided views ran
        /// N kernel invocations of count 1).
        ///
        /// Must NOT be used when a multi-index or flat index is tracked — index
        /// reconstruction needs the original axis structure (NumPy likewise
        /// skips coalescing there).
        /// </summary>
        public static void RemoveUnitAxes(ref NpyIterState state)
        {
            if (state.NDim <= 1)
                return;

            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;

            int write = 0;
            for (int read = 0; read < state.NDim; read++)
            {
                if (shape[read] == 1)
                    continue;

                if (write != read)
                {
                    shape[write] = shape[read];
                    for (int op = 0; op < state.NOp; op++)
                    {
                        int baseIdx = op * stridesNDim;
                        strides[baseIdx + write] = strides[baseIdx + read];
                    }
                }
                write++;
            }

            if (write == state.NDim)
                return;  // no size-1 axes

            // Degenerate all-size-1 iteration: keep a single unit axis
            // (stride 0 already, per the fill invariant).
            if (write == 0)
            {
                shape[0] = 1;
                for (int op = 0; op < state.NOp; op++)
                    strides[op * stridesNDim] = 0;
                write = 1;
            }

            state.NDim = write;

            // Axis removal invalidates the original-axis mapping; reset to
            // identity exactly like CoalesceAxes (no index tracking is active
            // on this path).
            for (int d = 0; d < write; d++)
                perm[d] = (sbyte)d;
            state.ItFlags |= (uint)NpyIterFlags.IDENTPERM;

            state.UpdateInnerStrides();
        }

        /// <summary>
        /// Try to coalesce the inner dimension for better vectorization.
        /// Returns true if inner loop size increased.
        /// </summary>
        public static bool TryCoalesceInner(ref NpyIterState state)
        {
            if (state.NDim < 2)
                return false;

            int innerAxis = state.NDim - 1;
            int prevAxis = state.NDim - 2;

            var shape = state.Shape;
            var strides = state.Strides;
            int stridesNDim = state.StridesNDim;

            long innerShape = shape[innerAxis];
            long prevShape = shape[prevAxis];

            // Check if all operands allow coalescing these two axes
            for (int op = 0; op < state.NOp; op++)
            {
                int baseIdx = op * stridesNDim;
                long innerStride = strides[baseIdx + innerAxis];
                long prevStride = strides[baseIdx + prevAxis];

                // For contiguous inner loop, inner stride must be 1
                // and prev stride must be innerShape
                if (innerStride != 1 || prevStride != innerShape)
                    return false;
            }

            // Coalesce: merge prevAxis into innerAxis
            shape[innerAxis] = innerShape * prevShape;

            // Shift down outer axes
            for (int d = prevAxis; d < state.NDim - 2; d++)
            {
                shape[d] = shape[d + 1];
                for (int op = 0; op < state.NOp; op++)
                {
                    int baseIdx = op * stridesNDim;
                    strides[baseIdx + d] = strides[baseIdx + d + 1];
                }
            }

            state.NDim--;

            // Update inner strides cache after dimension change
            state.UpdateInnerStrides();
            return true;
        }

        /// <summary>
        /// Reorder axes for iteration based on the specified order.
        /// This is called BEFORE CoalesceAxes to enable full coalescing of contiguous arrays.
        ///
        /// Order semantics (matching NumPy):
        /// - C-order (NPY_CORDER): Last axis innermost (row-major logical order)
        ///   Forces axes to [n-1, n-2, ..., 0] order regardless of memory layout
        /// - F-order (NPY_FORTRANORDER): First axis innermost (column-major logical order)
        ///   Forces axes to [0, 1, ..., n-1] order regardless of memory layout
        /// - K-order (NPY_KEEPORDER): Follow memory layout (smallest stride innermost)
        ///   Sorts by stride to maximize cache efficiency
        /// - A-order (NPY_ANYORDER): Same as K-order
        ///
        /// The Perm array tracks the mapping: Perm[internal_axis] = original_axis
        /// This allows GetMultiIndex to return coordinates in the original axis order.
        /// </summary>
        /// <param name="state">Iterator state to modify</param>
        /// <param name="order">Iteration order</param>
        /// <param name="forCoalescing">If true, sort for coalescing (ascending).
        /// If false, sort for memory-order iteration with MULTI_INDEX (descending).
        /// Only affects K-order; C and F orders are deterministic.</param>
        public static void ReorderAxesForCoalescing(ref NpyIterState state, NPY_ORDER order, bool forCoalescing = true)
        {
            if (state.NDim <= 1)
                return;

            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;
            int ndim = state.NDim;

            // For C and F orders, we need deterministic axis ordering (not stride-based)
            // Note: In Advance(), axis NDim-1 is innermost (changes fastest)
            //
            // C-order (row-major): last axis changes fastest
            //   - Want original axis n-1 at internal position n-1 (innermost)
            //   - No reordering needed, identity permutation
            //
            // F-order (column-major): first axis changes fastest
            //   - Want original axis 0 at internal position n-1 (innermost)
            //   - Reverse axis order so internal = [n-1, n-2, ..., 0]
            //   - Perm = [n-1, n-2, ..., 0] (internal axis d = original axis n-1-d)
            if (order == NPY_ORDER.NPY_CORDER)
            {
                // C-order: no reordering needed, already identity
                state.ItFlags |= (uint)NpyIterFlags.IDENTPERM;
                return;
            }
            else if (order == NPY_ORDER.NPY_FORTRANORDER)
            {
                // F-order: reverse axis order so first axis is innermost
                ReverseAxes(ref state);
                state.ItFlags &= ~(uint)NpyIterFlags.IDENTPERM;
                return;
            }

            // K-order (KEEPORDER) and A-order (ANYORDER): sort by stride
            //
            // The sort order depends on whether coalescing will follow:
            // - forCoalescing=true (without MULTI_INDEX): ascending sort (smallest first)
            //   This allows the coalescing formula stride[i] * shape[i] == stride[i+1] to work.
            // - forCoalescing=false (with MULTI_INDEX): descending sort (largest first)
            //   This puts the smallest stride at position NDim-1, where Advance() starts,
            //   resulting in memory-order iteration.
            bool ascending = forCoalescing;  // Ascending for coalescing, descending for iteration

            // Simple insertion sort by minimum absolute stride across all operands
            // Using insertion sort for stability and good performance on nearly-sorted data
            // Key-strides scratch is hoisted out of the loop (CA2014: stackalloc in
            // a loop accumulates stack space per iteration — overflow risk for
            // high-dimensional iterators since NumSharp has no NPY_MAXDIMS cap).
            //
            // All-stride-0 axes (GetMinStride == 0; size-1 axes under the
            // fill_axisdata invariant) carry no ordering signal. In DESCENDING
            // (iteration-order) mode they must sort OUTERMOST, never innermost:
            // an innermost stride-0 axis collapses the inner loop to one element
            // and defeats the linearity test (forcing needless buffering), while
            // its position never changes the element-visit sequence (one
            // coordinate value). NumPy reaches the same outcome — its ambiguous
            // comparison keeps size-1 axes out of the way and its unconditional
            // coalesce absorbs them. In ASCENDING (pre-coalesce) mode key 0
            // sorting first is exactly right: the strict trivial branch absorbs
            // those axes immediately.
            var keyStrides = stackalloc long[state.NOp];
            for (int i = 1; i < ndim; i++)
            {
                long keyShape = shape[i];
                sbyte keyPerm = perm[i];

                // Gather key strides for all operands
                for (int op = 0; op < state.NOp; op++)
                    keyStrides[op] = strides[op * stridesNDim + i];

                long keyMinStride = GetMinStride(strides, state.NOp, i, stridesNDim);
                if (!ascending && keyMinStride == 0)
                    keyMinStride = long.MaxValue;

                int j = i - 1;
                while (j >= 0)
                {
                    long jMinStride = GetMinStride(strides, state.NOp, j, stridesNDim);
                    if (!ascending && jMinStride == 0)
                        jMinStride = long.MaxValue;

                    // Compare based on order (ascending = smallest first)
                    bool shouldShift = ascending
                        ? jMinStride > keyMinStride
                        : jMinStride < keyMinStride;

                    if (!shouldShift)
                        break;

                    // Shift element at j to j+1
                    shape[j + 1] = shape[j];
                    perm[j + 1] = perm[j];
                    for (int op = 0; op < state.NOp; op++)
                    {
                        int baseIdx = op * stridesNDim;
                        strides[baseIdx + j + 1] = strides[baseIdx + j];
                    }

                    j--;
                }

                // Insert key at j+1
                shape[j + 1] = keyShape;
                perm[j + 1] = keyPerm;
                for (int op = 0; op < state.NOp; op++)
                    strides[op * stridesNDim + j + 1] = keyStrides[op];
            }

            // Check if permutation is still identity
            bool isIdentity = true;
            for (int d = 0; d < ndim; d++)
            {
                if (perm[d] != d)
                {
                    isIdentity = false;
                    break;
                }
            }

            if (isIdentity)
                state.ItFlags |= (uint)NpyIterFlags.IDENTPERM;
            else
                state.ItFlags &= ~(uint)NpyIterFlags.IDENTPERM;
        }

        /// <summary>
        /// Reverse the axis order for C-order iteration.
        /// Internal order becomes [n-1, n-2, ..., 0].
        /// </summary>
        private static void ReverseAxes(ref NpyIterState state)
        {
            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;
            int ndim = state.NDim;

            // Reverse shape and perm
            for (int i = 0; i < ndim / 2; i++)
            {
                int j = ndim - 1 - i;

                // Swap shape
                (shape[i], shape[j]) = (shape[j], shape[i]);

                // Swap perm
                (perm[i], perm[j]) = (perm[j], perm[i]);

                // Swap strides for all operands
                for (int op = 0; op < state.NOp; op++)
                {
                    int baseIdx = op * stridesNDim;
                    (strides[baseIdx + i], strides[baseIdx + j]) = (strides[baseIdx + j], strides[baseIdx + i]);
                }
            }
        }

        /// <summary>
        /// Reorder axes for optimal memory access pattern.
        /// Prioritizes axes with stride=1 as innermost.
        /// </summary>
        [Obsolete("Use ReorderAxesForCoalescing with order parameter instead")]
        public static void ReorderAxes(ref NpyIterState state)
        {
            ReorderAxesForCoalescing(ref state, NPY_ORDER.NPY_KEEPORDER);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long GetMinStride(long* strides, int nop, int axis, int stridesNDim)
        {
            long min = long.MaxValue;
            for (int op = 0; op < nop; op++)
            {
                long stride = Math.Abs(strides[op * stridesNDim + axis]);
                if (stride > 0 && stride < min)
                    min = stride;
            }
            return min == long.MaxValue ? 0 : min;
        }

        /// <summary>
        /// Check if all operands are contiguous in the current internal axis order.
        /// This determines whether coalescing would preserve the iteration semantics
        /// for C/F order iteration.
        ///
        /// For coalescing to preserve iteration order, all operands must be contiguous
        /// such that stride[i] * shape[i] == stride[i+1] for adjacent axes.
        /// </summary>
        public static bool IsContiguousForCoalescing(ref NpyIterState state)
        {
            if (state.NDim <= 1)
                return true;  // Trivially contiguous

            var shape = state.Shape;
            var strides = state.Strides;
            int stridesNDim = state.StridesNDim;

            // Check each operand for contiguity in internal axis order
            for (int op = 0; op < state.NOp; op++)
            {
                int baseIdx = op * stridesNDim;

                // Check that stride[i] * shape[i] == stride[i+1] for all adjacent axes
                for (int i = 0; i < state.NDim - 1; i++)
                {
                    long stride0 = strides[baseIdx + i];
                    long shape0 = shape[i];
                    long stride1 = strides[baseIdx + i + 1];

                    // Handle broadcast dimensions (stride=0)
                    if (stride0 == 0 || stride1 == 0)
                        continue;  // Broadcast dims are always "contiguous" for coalescing

                    // Check contiguity: inner_stride * inner_shape == outer_stride
                    if (stride0 * shape0 != stride1)
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Flip axes with all-negative strides for memory-order iteration.
        ///
        /// NumPy's npyiter_flip_negative_strides():
        /// - For each axis, check if ALL operands have negative or zero strides
        /// - If so, negate the strides, adjust base pointers to start at the end,
        ///   and mark the axis as flipped in the Perm array (perm[d] = -1 - perm[d])
        /// - Sets NEGPERM flag and clears IDENTPERM
        ///
        /// This allows the iterator to traverse memory in ascending order even for
        /// reversed arrays, improving cache efficiency.
        /// </summary>
        /// <param name="state">Iterator state to modify</param>
        /// <returns>True if any axes were flipped</returns>
        public static bool FlipNegativeStrides(ref NpyIterState state)
        {
            if (state.NDim == 0)
                return false;

            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;
            int nop = state.NOp;
            bool anyFlipped = false;

            for (int axis = 0; axis < state.NDim; axis++)
            {
                // Check if ALL operands have negative or zero strides for this axis
                bool anyNegative = false;
                bool allNonPositive = true;

                for (int op = 0; op < nop; op++)
                {
                    long stride = strides[op * stridesNDim + axis];
                    if (stride < 0)
                    {
                        anyNegative = true;
                    }
                    else if (stride > 0)
                    {
                        allNonPositive = false;
                        break;
                    }
                    // stride == 0 is fine (broadcast dimension)
                }

                // Only flip if at least one stride is negative and none are positive
                if (anyNegative && allNonPositive)
                {
                    long shapeMinus1 = shape[axis] - 1;

                    // Flip strides and accumulate byte offset into BaseOffsets.
                    // NumPy nditer_constr.c:2579-2593 — baseoffsets records the cumulative
                    // offset from the array's origin to the iterator's start after flipping.
                    // This allows NpyIter_ResetBasePointers(baseptrs) to recompute
                    // resetdataptr[iop] = baseptrs[iop] + baseoffsets[iop].
                    for (int op = 0; op < nop; op++)
                    {
                        long stride = strides[op * stridesNDim + axis];
                        // Strides are SOURCE-array element strides and BaseOffsets/
                        // ResetDataPtrs/DataPtrs are byte positions in source-array
                        // memory, so the byte multiplier must be the SOURCE element
                        // size. ElementSizes is the buffer dtype size, which
                        // diverges under a buffered cast (bug (b) family: an int32
                        // source buffered as float64 doubled the flip offset and
                        // read past the array under K-order).
                        int elemSize = state.SrcElementSizes[op];
                        long byteOffset = shapeMinus1 * stride * elemSize;

                        // Track cumulative byte offset per-operand (negative because stride<0).
                        state.BaseOffsets[op] += byteOffset;

                        // Negate the stride
                        strides[op * stridesNDim + axis] = -stride;
                    }

                    // Mark axis as flipped in permutation
                    // perm[axis] = -1 - perm[axis] makes it negative
                    // Original axis = perm[axis] when >= 0, or -1 - perm[axis] when < 0
                    perm[axis] = (sbyte)(-1 - perm[axis]);

                    anyFlipped = true;
                }
            }

            if (anyFlipped)
            {
                // Propagate accumulated BaseOffsets into ResetDataPtrs and DataPtrs.
                // NumPy nditer_constr.c:2599-2605: "If any strides were flipped,
                // the base pointers were adjusted in the first AXISDATA, and need
                // to be copied to all the rest."
                for (int op = 0; op < nop; op++)
                {
                    state.ResetDataPtrs[op] += state.BaseOffsets[op];
                    state.DataPtrs[op] = state.ResetDataPtrs[op];
                }

                // Set NEGPERM flag and clear IDENTPERM
                state.ItFlags = (state.ItFlags | (uint)NpyIterFlags.NEGPERM) &
                                ~(uint)NpyIterFlags.IDENTPERM;
            }

            return anyFlipped;
        }
    }
}
