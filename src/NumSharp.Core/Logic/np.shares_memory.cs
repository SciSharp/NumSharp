using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Determine whether two arrays share ANY byte of memory — the EXACT answer, not a bound.
        /// </summary>
        /// <param name="a">First array. Any layout is accepted (contiguous, sliced, strided, transposed,
        ///     negative-stride, or broadcast view). A <c>null</c> reference has no memory and shares nothing.</param>
        /// <param name="b">Second array, under the same rules as <paramref name="a"/>.</param>
        /// <param name="max_work">
        ///     Effort cap on the overlap solver, mirroring NumPy's <c>max_work</c>:
        ///     <c>-1</c> (the default) and <c>-2</c> solve the collision EXACTLY however long it takes;
        ///     <c>0</c> checks only the byte bounds (so an undecided result becomes <see cref="TooHardError"/> here);
        ///     a positive value examines at most that many candidate solutions before giving up. Values below
        ///     <c>-2</c> are rejected.
        /// </param>
        /// <returns><c>true</c> only when at least one addressable byte is common to both arrays.</returns>
        /// <exception cref="ValueError"><paramref name="max_work"/> is less than <c>-2</c>.</exception>
        /// <exception cref="TooHardError">The solver hit the <paramref name="max_work"/> candidate cap
        ///     (or was restricted to a bounds check via <c>max_work: 0</c>) before it could decide exactly.</exception>
        /// <exception cref="OverflowException">The solver's integer arithmetic overflowed while computing overlap.</exception>
        /// <exception cref="RuntimeError">The solver reported an internal error (does not occur for valid inputs).</exception>
        /// <remarks>
        ///     This is the STRICT counterpart of <see cref="may_share_memory(NDArray,NDArray,long)"/>: two
        ///     interleaved views of one buffer (e.g. <c>base[::2]</c> and <c>base[1::2]</c>) have overlapping
        ///     byte bounds yet no common byte, so <c>shares_memory</c> returns <c>false</c> there while
        ///     <c>may_share_memory</c> returns <c>true</c>. Because the general problem is NP-hard, an
        ///     adversarial pair with a small default budget can exhaust the work cap and raise
        ///     <see cref="TooHardError"/> rather than silently guessing — pass a larger (or the default
        ///     unlimited) <paramref name="max_work"/> to force a definite answer. Mirrors
        ///     <c>numpy.shares_memory(a, b, /, max_work=-1)</c> (NumPy 2.4.2), which raises on the undecided
        ///     and overflow outcomes.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.shares_memory.html
        /// </remarks>
        public static bool shares_memory(NDArray a, NDArray b, long max_work = -1)
            // raiseExceptions: true — shares_memory promises an exact answer, so an undecided/overflow
            // outcome is surfaced as an exception rather than a conservative guess (NumPy: raise_exceptions=1).
            => SharesMemoryImpl(a, b, max_work, raiseExceptions: true);

        /// <summary>
        ///     Determine whether two arrays MIGHT share memory — a fast, CONSERVATIVE bounds check by default.
        /// </summary>
        /// <param name="a">First array (any layout; a <c>null</c> reference shares nothing).</param>
        /// <param name="b">Second array, under the same rules as <paramref name="a"/>.</param>
        /// <param name="max_work">
        ///     Effort cap on the overlap solver, mirroring NumPy's <c>max_work</c>:
        ///     <c>0</c> (the default) checks only the byte bounds — cheap and never raises;
        ///     <c>-1</c>/<c>-2</c> solve the collision exactly; a positive value caps the candidate search.
        ///     Values below <c>-2</c> are rejected.
        /// </param>
        /// <returns><c>true</c> when the arrays share memory OR the query cannot be decided within the cap —
        ///     a <c>false</c> is therefore a GUARANTEE of no sharing, while a <c>true</c> is only "possibly".</returns>
        /// <exception cref="ValueError"><paramref name="max_work"/> is less than <c>-2</c>.</exception>
        /// <exception cref="RuntimeError">The solver reported an internal error (does not occur for valid inputs).</exception>
        /// <remarks>
        ///     Unlike <see cref="shares_memory(NDArray,NDArray,long)"/> this NEVER raises for the undecided
        ///     or overflow outcomes: it resolves both to a conservative <c>true</c> ("might share"), which is
        ///     why the default <c>max_work: 0</c> is safe — a mere bounds overlap already answers "true"
        ///     without invoking the expensive exact solver. Use this as a cheap precondition (as the library
        ///     itself does before overlap-sensitive writes); reach for <c>shares_memory</c> only when a
        ///     precise yes/no is required. Mirrors <c>numpy.may_share_memory(a, b, /, max_work=0)</c>
        ///     (NumPy 2.4.2).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.may_share_memory.html
        /// </remarks>
        public static bool may_share_memory(NDArray a, NDArray b, long max_work = 0)
            // raiseExceptions: false — the promise here is only "might share", so an undecided/overflow
            // outcome is folded into a conservative true instead of an exception (NumPy: raise_exceptions=0).
            => SharesMemoryImpl(a, b, max_work, raiseExceptions: false);

        /// <summary>
        ///     Shared body of <see cref="shares_memory(NDArray,NDArray,long)"/> and
        ///     <see cref="may_share_memory(NDArray,NDArray,long)"/> — a port of NumPy's
        ///     <c>array_shares_memory_impl</c> (multiarraymodule.c). Runs the memory-overlap solver and
        ///     translates its verdict, with the two public entry points differing ONLY in
        ///     <paramref name="raiseExceptions"/> and their default <paramref name="max_work"/>.
        /// </summary>
        /// <param name="a">First array, or <c>null</c>.</param>
        /// <param name="b">Second array, or <c>null</c>.</param>
        /// <param name="max_work">Solver effort cap (already carries the caller's default).</param>
        /// <param name="raiseExceptions">When <c>true</c>, the undecided (<c>TooHard</c>) and <c>Overflow</c>
        ///     verdicts throw; when <c>false</c>, both collapse to a conservative <c>true</c>.</param>
        /// <returns>Whether the arrays overlap, per the semantics of the calling entry point.</returns>
        /// <exception cref="ValueError"><paramref name="max_work"/> is less than <c>-2</c>.</exception>
        /// <exception cref="TooHardError">Undecided within the cap and <paramref name="raiseExceptions"/> is set.</exception>
        /// <exception cref="OverflowException">Solver arithmetic overflowed and <paramref name="raiseExceptions"/> is set.</exception>
        /// <exception cref="RuntimeError">The solver reported an internal error.</exception>
        private static bool SharesMemoryImpl(NDArray a, NDArray b, long max_work, bool raiseExceptions)
        {
            // A null operand has no memory extent (NumPy coerces a non-array via FromAny into a fresh,
            // distinct array that shares nothing — the observable outcome is False for both entry points).
            if (a is null || b is null)
                return false;

            // Range guard BEFORE touching the solver, matching NumPy's order: -2/-1 = exact, 0 = bounds,
            // >0 = candidate cap; anything below -2 is not a meaningful mode.
            if (max_work < -2)
                throw new ValueError("Invalid value for max_work");

            MemOverlap result = NDMemOverlap.SolveMayShareMemory(a, b, max_work);
            switch (result)
            {
                case MemOverlap.No:
                    return false;
                case MemOverlap.Yes:
                    return true;
                case MemOverlap.Overflow:
                    // shares_memory promises exactness → surface the overflow; may_share_memory says "yes".
                    if (raiseExceptions)
                        throw new OverflowException("Integer overflow in computing overlap");
                    return true;
                case MemOverlap.TooHard:
                    // Budget exhausted (or bounds-only): the exact answer is unknown.
                    if (raiseExceptions)
                        throw new TooHardError("Exceeded max_work");
                    return true;
                default:
                    // MemOverlap.Error — an invalid problem the solver could not form. NumPy raises
                    // RuntimeError here regardless of raise_exceptions ("Doesn't happen usually").
                    throw new RuntimeError("Error in computing overlap");
            }
        }
    }
}
