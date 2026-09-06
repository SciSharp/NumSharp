using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the size of the buffer used in ufuncs (the default buffer size for buffered
        ///     NDIter/ufunc iteration on the calling thread).
        /// </summary>
        /// <returns>Size of the ufunc buffer, in elements. Defaults to 8192 (NumPy's <c>NPY_BUFSIZE</c>).</returns>
        /// <remarks>
        ///     Mirrors <c>numpy.getbufsize()</c>. The value is <b>thread-local</b>: it reflects the last
        ///     <see cref="setbufsize(long)"/> call on the current thread (or the 8192 default if none),
        ///     matching NumPy 2.x's context-local error/buffer state. The buffer size affects only how
        ///     buffered iteration is chunked internally — it never changes any computed result.
        ///     <example>
        ///     <code>
        ///     np.getbufsize();          // 8192
        ///     np.setbufsize(4096);      // returns 8192 (the previous size)
        ///     np.getbufsize();          // 4096
        ///     </code>
        ///     </example>
        /// </remarks>
        public static long getbufsize()
        {
            return NDIterBufferManager.CurrentBufferSize;
        }

        /// <summary>
        ///     Set the size of the buffer used in ufuncs (the default buffer size for buffered
        ///     NDIter/ufunc iteration on the calling thread) and return the previous size.
        /// </summary>
        /// <param name="size">
        ///     New buffer size in elements. Must be non-negative, at most 10,000,000, at least 5, and a
        ///     multiple of 16 — so the smallest accepted value is 16.
        /// </param>
        /// <returns>The buffer size in effect before this call.</returns>
        /// <exception cref="ValueError">
        ///     If <paramref name="size"/> is invalid. The message and the order in which the checks run
        ///     match NumPy verbatim: negative → <c>"buffer size must be non-negative"</c>; greater than
        ///     10,000,000 → <c>"Buffer size, {size}, is too big"</c>; less than 5 →
        ///     <c>"Buffer size, {size}, is too small"</c>; not a multiple of 16 →
        ///     <c>"Buffer size, {size}, is not a multiple of 16"</c>.
        /// </exception>
        /// <remarks>
        ///     Mirrors <c>numpy.setbufsize(size)</c>. The setting is <b>thread-local</b> (matching NumPy
        ///     2.x's context-local buffer state) and persists until changed again on the same thread; it
        ///     never affects other threads. Buffering is purely a performance/chunking knob, so changing
        ///     it leaves every computed result bit-for-bit identical.
        ///     <para>
        ///     Unlike NumPy — which accepts any Python int and raises <c>TypeError</c> for a
        ///     <c>float</c>/<c>bool</c> and <c>OverflowError</c> for a value beyond the platform integer —
        ///     C#'s type system already rejects a non-integer argument at compile time, so only the
        ///     value-range <see cref="ValueError"/>s remain reachable.
        ///     </para>
        /// </remarks>
        public static long setbufsize(long size)
        {
            // Validation order is NumPy's: setbufsize (Python) checks non-negativity, then _make_extobj
            // (C, extobj.c) checks too-big, too-small, and multiple-of-16 — in that sequence.
            if (size < 0)
                throw new ValueError("buffer size must be non-negative");
            if (size > NDIterBufferManager.MaxBufferSize)
                throw new ValueError($"Buffer size, {size}, is too big");
            if (size < 5)
                throw new ValueError($"Buffer size, {size}, is too small");
            if (size % 16 != 0)
                throw new ValueError($"Buffer size, {size}, is not a multiple of 16");

            long old = NDIterBufferManager.CurrentBufferSize;
            NDIterBufferManager.CurrentBufferSize = size;
            return old;
        }
    }
}
