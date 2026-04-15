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
        /// Reorder axes for optimal memory access pattern.
        /// Prioritizes axes with stride=1 as innermost.
        /// </summary>
        public static void ReorderAxes(ref NpyIterState state)
        {
            if (state.NDim <= 1)
                return;

            var shape = state.Shape;
            var strides = state.Strides;
            var perm = state.Perm;
            int stridesNDim = state.StridesNDim;

            // Simple bubble sort by minimum stride (prefer contiguous axes as inner)
            for (int i = 0; i < state.NDim - 1; i++)
            {
                for (int j = 0; j < state.NDim - 1 - i; j++)
                {
                    long minStrideJ = GetMinStride(strides, state.NOp, j, stridesNDim);
                    long minStrideJ1 = GetMinStride(strides, state.NOp, j + 1, stridesNDim);

                    // Swap if j has larger minimum stride than j+1
                    // (we want smaller strides at higher indices = inner)
                    if (minStrideJ > minStrideJ1)
                    {
                        // Swap shapes
                        (shape[j], shape[j + 1]) = (shape[j + 1], shape[j]);

                        // Swap permutation
                        (perm[j], perm[j + 1]) = (perm[j + 1], perm[j]);

                        // Swap strides for all operands
                        for (int op = 0; op < state.NOp; op++)
                        {
                            int baseIdx = op * stridesNDim;
                            (strides[baseIdx + j], strides[baseIdx + j + 1]) =
                                (strides[baseIdx + j + 1], strides[baseIdx + j]);
                        }
                    }
                }
            }

            // Clear IDENTPERM if we reordered
            state.ItFlags &= ~(uint)NpyIterFlags.IDENTPERM;
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
