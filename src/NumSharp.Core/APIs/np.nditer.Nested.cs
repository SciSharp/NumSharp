using System;

namespace NumSharp
{
    public static partial class np
    {
        // Nested-iteration support for np.nested_iters — the port of NumPy's nested_child machinery.
        // NumPy's own nditer object carries a `nested_child` field and re-bases inside npyiter_next /
        // npyiter_iternext / npyiter_reset (nditer_pywrap.c); NDIterator mirrors that here. Kept in this
        // partial (next to the np.nested_iters factory) so the shared np.nditer.cs holds only the fields
        // and three one-line guarded hooks. All members here are reachable ONLY in nested mode.
        public unsafe partial class NDIterator
        {
            /// <summary>
            ///     Enroll this iterator in a nested group: mark it nested and record the inner level (null for the innermost)
            ///     that this level re-bases on each advance and seek.
            /// </summary>
            /// <param name="child">The next inner level, or null for the innermost.</param>
            /// <remarks>
            ///     The started / finished flags are left as construction set them (NumPy's <c>NpyIter_NestedIters</c> sets
            ///     both for an empty level, neither otherwise); the initial re-base cascade then sets every child's.
            /// </remarks>
            internal void SetupNested(NDIterator child)
            {
                _nestedMode = true;
                _nestedChild = child;
            }

            /// <summary>
            ///     Re-base the whole child chain to the current position — NumPy's
            ///     <c>npyiter_resetbasepointers</c>: for each level, reset the child's operand base pointers
            ///     to this level's CURRENT data pointers (via the pre-existing
            ///     <see cref="NDIterRef.ResetBasePointers"/>) and rewind it to its start.
            /// </summary>
            internal void RebaseChildren()
            {
                var parent = this;
                while (parent._nestedChild != null)
                {
                    var child = parent._nestedChild;
                    int nop = parent.Borrow().NOp;
                    Span<IntPtr> baseptrs = stackalloc IntPtr[nop];
                    for (int op = 0; op < nop; op++)
                        baseptrs[op] = (IntPtr)parent._state->GetDataPtr(op);

                    var cit = child.Borrow();
                    cit.ResetBasePointers(baseptrs);
                    child._cachedNext = cit.PeekCachedIterNext();
                    // NpyIter_ResetBasePointers allocates delayed buffers; npyiter_resetbasepointers then restarts the child:
                    // finished (and started) at once when it has nothing to visit.
                    child._reduceDelayPending = false;
                    child._started = child._finished = cit.IterSize == 0;

                    parent = child;
                }
            }

            /// <summary>
            ///     MoveNext for a nested level — NumPy's <c>npyiter_next</c>, as the plain <see cref="MoveNext"/>: the first
            ///     pull publishes the start position WITHOUT advancing; every later pull advances (reusing
            ///     <see cref="iternext"/>, which cascades the re-base to the inner levels) then publishes. Advance-on-entry is
            ///     what keeps the in-body <see cref="multi_index"/> aligned and re-bases the child at the parent's CURRENT
            ///     position while the parent's body runs.
            /// </summary>
            /// <returns>True when a value was published; false at the end.</returns>
            private bool MoveNextNested()
            {
                // An empty level starts finished (construction / the parent's re-base set the flag).
                if (_state == null || _finished)
                {
                    Current = null;
                    return false;
                }

                if (_started)
                {
                    if (!iternext())    // advances, cascades the re-base, and sets _finished at the end
                    {
                        Current = null;
                        return false;
                    }
                }

                _started = true;
                Current = value;
                return true;
            }
        }
    }
}
