// =============================================================================
// NDCharArray.TrivialLoop.cs — NumPy's trivial-loop copy of a chararray ufunc input (ufunc_object.c check_for_trivial_loop)
// =============================================================================
//
// Before a ufunc call runs its trivial loop, NumPy copies each input whose loop dtype differs from its own (or that is
// misaligned) into the loop's dtype when it is 0-d or 1-d with at most np.getbufsize() elements. The copy is
// PyArray_CastToType, which keeps the input's class: a chararray input copied into bool (the logical ufuncs) or into
// StringDType (beside a StringDType operand) meets chararray.__array_finalize__ and raises "Can only create a chararray
// from string data.". The first input that would need a copy but is larger refuses the trivial loop: no later input is
// copied, and the call computes through the iterator (a plain result).
//
// WHICH INPUTS ARE THE USER'S CHARARRAYS. Only an input of the USER's call can raise: a ufunc that a computation calls
// later on the same data converts with np.asarray first in NumPy. The woven subclass path hands a chararray input in as a
// plain view marked NDArray.IsTextUfuncInput (NDSubclass.PlainUfuncInput); so do the chararray members that compute their
// function's array themselves (center / ljust / rjust). The guard of the ONE ufunc call that reads the marks clears them.
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>check_for_trivial_loop</c> for a ufunc call over a chararray (ufunc_object.c): without a
        ///     <c>where=</c> mask, each input whose loop dtype differs from its own (or that is misaligned) is COPIED into the
        ///     loop's dtype when it is 0-d or 1-d with at most <c>np.getbufsize()</c> elements — and the copy
        ///     (<c>PyArray_CastToType</c>) keeps the subclass, so a chararray input copied into bool (the logical ufuncs) or
        ///     StringDType raises the finalize error. The first input that would need a copy but is larger refuses the trivial
        ///     loop: no later input is copied, and the call computes through the iterator (a plain result).
        /// </summary>
        /// <param name="x1">The first input.</param>
        /// <param name="loop1">The loop's descriptor for <paramref name="x1"/>.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <param name="loop2">The loop's descriptor for <paramref name="x2"/> (ignored when it is null).</param>
        /// <param name="where">The <c>where=</c> mask, or null: a mask selects the masked loop, which never copies. The C#
        ///     literal <c>true</c> (its array marked <see cref="NDArray.WasPythonBool"/>) is NumPy's <c>where=True</c>, which
        ///     <c>_wheremask_converter</c> reads as no mask; every other value — <c>false</c>, a 0-d <c>np.array(True)</c>,
        ///     any bool array — is a mask.</param>
        /// <returns>The finalize error, or null when the call goes on.</returns>
        /// <remarks>
        ///     Run it where NumPy runs the copy: after the loop was resolved and its output cast into <c>out=</c> checked, before
        ///     the broadcast (so <c>np.logical_and(c, empty)</c> raises the finalize error, not the broadcast's). It only reads
        ///     the inputs marked as the user's chararray ufunc inputs (<see cref="NDSubclass.PlainUfuncInput(NDArray)"/>, the
        ///     file header), and consumes the marks.
        /// </remarks>
        internal static ValueError TrivialLoopCopyError(NDArray x1, DType loop1, NDArray x2, DType loop2, NDArray where)
        {
            // No marked input: the call has no chararray input of the user's, or a guard already consumed the marks.
            if (!TakeUfuncInputMarks(x1, x2, null, out var marks))
                return null;
            // _wheremask_converter: only the Python object True is "no mask" (obj == Py_True); np.array(True) is a mask.
            if (where is not null && !(where.WasPythonBool && where.GetAtIndex<bool>(0)))
                return null;
            long bufsize = np.getbufsize();
            if (CopyStep(x1, loop1, marks, bufsize, out bool stop) is { } first)
                return first;
            if (stop || x2 is null)
                return null;
            return CopyStep(x2, loop2, marks, bufsize, out _);
        }

        /// <summary>
        ///     <see cref="TrivialLoopCopyError(NDArray, DType, NDArray, DType, NDArray)"/> for a ufunc of three leading inputs
        ///     and no <c>where=</c> (numpy.strings' internal ufuncs: <c>_center(a, width, fillchar)</c>,
        ///     <c>_replace(a, old, new, count)</c>): the inputs are checked IN ORDER, so a middle input that refuses the trivial
        ///     loop (an int32 <c>width</c> array longer than <c>np.getbufsize()</c>, cast to int64) leaves a chararray after
        ///     it uncopied.
        /// </summary>
        /// <param name="x1">The first input.</param>
        /// <param name="loop1">The loop's descriptor for <paramref name="x1"/>.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="loop2">The loop's descriptor for <paramref name="x2"/>.</param>
        /// <param name="x3">The third input.</param>
        /// <param name="loop3">The loop's descriptor for <paramref name="x3"/>.</param>
        /// <returns>The finalize error, or null when the call goes on.</returns>
        /// <remarks>
        ///     Inputs after the third need no check: they are numbers (<c>count</c>), never a chararray, and nothing follows
        ///     them that a refusal could protect. Run it at the same point as the two-input form; it consumes the marks.
        /// </remarks>
        internal static ValueError TrivialLoopCopyError(NDArray x1, DType loop1, NDArray x2, DType loop2, NDArray x3, DType loop3)
        {
            if (!TakeUfuncInputMarks(x1, x2, x3, out var marks))
                return null;
            long bufsize = np.getbufsize();
            if (CopyStep(x1, loop1, marks, bufsize, out bool stop) is { } first)
                return first;
            if (stop)
                return null;
            if (CopyStep(x2, loop2, marks, bufsize, out stop) is { } second)
                return second;
            if (stop)
                return null;
            return CopyStep(x3, loop3, marks, bufsize, out _);
        }

        /// <summary>
        ///     Reads and CLEARS the marks of the user's chararray ufunc inputs: the plain views carry
        ///     <see cref="NDArray.IsTextUfuncInput"/> (<see cref="NDSubclass.PlainUfuncInput(NDArray)"/>). Cleared on read, as
        ///     the guard of ONE ufunc call consumes them: a ufunc the computation calls later on the same views computes plain.
        /// </summary>
        /// <param name="x1">The first input of the guarded call, or null.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <param name="x3">The third input, or null.</param>
        /// <param name="marks">The marked inputs, for <see cref="UfuncInputMarks.Holds"/>.</param>
        /// <returns>False when no input is marked.</returns>
        private static bool TakeUfuncInputMarks(NDArray x1, NDArray x2, NDArray x3, out UfuncInputMarks marks)
        {
            NDArray m0 = x1 is not null && x1.IsTextUfuncInput ? x1 : null;
            NDArray m1 = x2 is not null && x2.IsTextUfuncInput ? x2 : null;
            NDArray m2 = x3 is not null && x3.IsTextUfuncInput ? x3 : null;
            marks = new UfuncInputMarks(m0, m1, m2);
            if (m0 is null && m1 is null && m2 is null)
                return false;
            if (m0 is not null)
                m0.IsTextUfuncInput = false;
            if (m1 is not null)
                m1.IsTextUfuncInput = false;
            if (m2 is not null)
                m2.IsTextUfuncInput = false;
            return true;
        }

        /// <summary>One input of <see cref="TrivialLoopCopyError(NDArray, DType, NDArray, DType, NDArray)"/>: copied,
        ///     refused, or not concerned.</summary>
        /// <param name="op">The input.</param>
        /// <param name="loop">Its loop descriptor.</param>
        /// <param name="marks">The marked chararray inputs of the call.</param>
        /// <param name="bufsize">NumPy's buffer size.</param>
        /// <param name="stop">Set when the input refuses the trivial loop (no later input is copied).</param>
        /// <returns>The finalize error when the copied input is a marked chararray and the loop dtype is not string data.</returns>
        private static ValueError CopyStep(NDArray op, DType loop, in UfuncInputMarks marks, long bufsize, out bool stop)
        {
            stop = false;
            // A cast between two different descriptors is never a view here (text to bool, text to StringDType), so it
            // needs a copy; a misaligned input needs one too (PyArray_ISALIGNED).
            bool cast = loop is not null && !op.dtype.Equals(loop);
            bool aligned = (op.Shape.Flags & ArrayFlags.ALIGNED) != 0 && op.Storage.IsDataAligned;
            if (!cast && aligned)
                return null;
            if (op.ndim == 0 || op.ndim == 1 && op.size <= bufsize)
            {
                // The copy is made as the input's class: a chararray copy into a non-string loop dtype is refused.
                if (cast && marks.Holds(op) && !AcceptsDType(loop))
                    return FinalizeError();
                return null;
            }
            stop = true;
            return null;
        }

        /// <summary>The user's chararray inputs of one guarded ufunc call (each the marked plain view, or null).</summary>
        internal readonly struct UfuncInputMarks
        {
            /// <summary>The first marked input, or null.</summary>
            private readonly NDArray _mark0;

            /// <summary>The second marked input, or null.</summary>
            private readonly NDArray _mark1;

            /// <summary>The third marked input, or null.</summary>
            private readonly NDArray _mark2;

            /// <summary>Records the marked inputs.</summary>
            /// <param name="mark0">The first.</param>
            /// <param name="mark1">The second.</param>
            /// <param name="mark2">The third.</param>
            internal UfuncInputMarks(NDArray mark0, NDArray mark1, NDArray mark2)
            {
                _mark0 = mark0;
                _mark1 = mark1;
                _mark2 = mark2;
            }

            /// <summary>Whether <paramref name="op"/> is one of the marked chararray inputs (by object identity: a copy or
            ///     another alias of the same chararray is not the user's operand).</summary>
            /// <param name="op">An input of the guarded call.</param>
            /// <returns>True for a marked input.</returns>
            internal bool Holds(NDArray op)
                => ReferenceEquals(op, _mark0) || ReferenceEquals(op, _mark1) || ReferenceEquals(op, _mark2);
        }
    }
}
