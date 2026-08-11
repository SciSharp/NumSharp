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
            ///     Enroll this iterator in a nested group: switch to advance-on-entry iteration and record
            ///     the inner level (null for the innermost) that this level re-bases on each advance.
            /// </summary>
            internal void SetupNested(NDIterator child)
            {
                _nestedMode = true;
                _started = false;
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
                    child._started = false;
                    child._exhausted = cit.IterSize == 0;

                    parent = child;
                }
            }

            /// <summary>
            ///     Advance-on-entry MoveNext (NumPy's <c>npyiter_next</c>): the first pull publishes the start
            ///     position WITHOUT advancing; every later pull advances (reusing <see cref="iternext"/>,
            ///     which cascades the re-base) then publishes. This is what NumSharp's default
            ///     publish-then-advance <see cref="MoveNext"/> cannot provide — it would misalign the in-body
            ///     <see cref="multi_index"/> and re-base the child one step early.
            /// </summary>
            private bool MoveNextNested()
            {
                if (_state == null || _exhausted)
                {
                    Current = null;
                    return false;
                }

                if (_started)
                {
                    if (!iternext())    // advances, cascades the re-base, and flags _exhausted at the end
                    {
                        Current = null;
                        return false;
                    }
                }
                else if (Borrow().IterSize == 0)   // empty from the start
                {
                    _exhausted = true;
                    Current = null;
                    return false;
                }

                _started = true;
                Current = value;
                return true;
            }
        }
    }
}
