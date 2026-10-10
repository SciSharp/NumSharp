using NumSharp.Backends.Kernels;

// =============================================================================
// NDIter.Execution.ComparisonRun.cs — driving a ComparisonRun from the iterator
// =============================================================================
//
// The comparison ufunc routes (DefaultEngine.TryExecuteComparisonOpViaNDIter and
// ExecuteComparisonUfuncInto) build one iterator over [lhs, rhs, out]. Each inner
// loop chunk it hands out is one run per operand: a pointer, a byte stride and a
// count. A ComparisonRun turns such a chunk into a call of the vector comparison
// kernels whatever the strides and dtypes are (Backends/Kernels/ComparisonRun.cs).
//
// This file is the decision between that run and the per-element kernels the
// routes used before, which stay the better choice for narrow chunks.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public unsafe ref partial struct NDIterRef
    {
        /// <summary>
        ///     Runs a comparison over [lhs, rhs, out] through <paramref name="run"/>, one call per inner-loop chunk,
        ///     when that is the faster route for this iteration. Returns false, having touched nothing, when the
        ///     caller should keep its per-element route.
        /// </summary>
        /// <param name="run">The comparison of the two input dtypes (from <see cref="ComparisonRun.Get"/>).</param>
        /// <returns>True when the whole iteration was executed.</returns>
        /// <remarks>
        ///     <para>The iterator must have exactly three operands whose third is a Boolean output (or a buffered
        ///     output the iterator presents as Boolean); a <c>where=</c> iteration (four operands) is declined, its
        ///     mask runs are short and its per-element route is already ahead of NumPy's masked loop.</para>
        ///     <para>Unbuffered: the inner strides and the chunk length are the same for every chunk, so the decision
        ///     is made once, by <see cref="ComparisonRun.Prefers"/>. Narrow rows are declined on purpose: the caller's
        ///     route then reaches the 2-D block kernel, which loops the rows inside one call.</para>
        ///     <para>Buffered (an output of another dtype): the chunks are buffer windows, long except for a tiny
        ///     array, and the run handles a short one by itself, so it is always taken.</para>
        /// </remarks>
        internal bool TryExecuteComparisonRun(ComparisonRun run)
        {
            if (run is null || _state->NOp != 3)
                return false;

            if ((_state->ItFlags & (uint)NDIterFlags.BUFFER) == 0)
            {
                long* strides = GetInnerLoopByteStrides();
                if (!run.Prefers(strides[0], strides[1], strides[2], ResolveInnerLoopCount(), IsSingleInnerLoop()))
                    return false;
            }

            ForEach(run.InnerLoop);
            return true;
        }
    }
}
