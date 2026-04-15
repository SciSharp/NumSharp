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
    internal static unsafe class NpyIterCoalescing
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

                    // Can coalesce if:
                    // - Either axis has shape 1 (trivial dimension)
                    // - Strides are compatible: stride0 * shape0 == stride1
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
        /// Reorder axes for optimal coalescing based on iteration order.
        /// This is called BEFORE CoalesceAxes to enable full coalescing of contiguous arrays.
        ///
        /// For C-order (row-major) iteration, axes are sorted so smallest strides come FIRST.
        /// This allows the coalescing formula stride[i]*shape[i]==stride[i+1] to work correctly.
        ///
        /// Example: C-contiguous (2,3,4) with strides [12,4,1]
        /// - Before reorder: (0,1) check: 12*2=24 != 4 → can't coalesce
        /// - After reorder to [4,3,2] with strides [1,4,12]:
        ///   (0,1) check: 1*4=4 == 4 ✓ → coalesce to [12,2], strides [1,12]
        ///   (0,1) check: 1*12=12 == 12 ✓ → coalesce to [24], strides [1]
        /// </summary>
        public static void ReorderAxesForCoalescing(ref NpyIterState state, NPY_ORDER order)
        {
            if (state.NDim <= 1)
                return;

            // KEEPORDER and ANYORDER: sort by stride to maximize coalescing
            // CORDER: sort ascending (smallest stride first = inner dimension first)
            // FORTRANORDER: sort descending (largest stride first)
            bool ascending = order != NPY_ORDER.NPY_FORTRANORDER;

            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;

            // Simple insertion sort by minimum absolute stride across all operands
            // Using insertion sort for stability and good performance on nearly-sorted data
            for (int i = 1; i < state.NDim; i++)
            {
                long keyShape = shape[i];
                sbyte keyPerm = perm[i];

                // Gather key strides for all operands
                var keyStrides = stackalloc long[state.NOp];
                for (int op = 0; op < state.NOp; op++)
                    keyStrides[op] = strides[op * stridesNDim + i];

                long keyMinStride = GetMinStride(strides, state.NOp, i, stridesNDim);

                int j = i - 1;
                while (j >= 0)
                {
                    long jMinStride = GetMinStride(strides, state.NOp, j, stridesNDim);

                    // Compare based on order
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

            // Mark that permutation may have changed
            state.ItFlags &= ~(uint)NpyIterFlags.IDENTPERM;
            state.ItFlags |= (uint)NpyIterFlags.NEGPERM;  // Indicate non-identity permutation
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
    }
}
