using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Swap the bytes of the array elements — toggle between low-endian and big-endian data
        ///     representation. Mirrors NumPy's <c>ndarray.byteswap(inplace=False)</c>: the dtype is
        ///     unchanged and only the raw element bytes are reversed, so the reinterpreted values change.
        ///     A complex element has its real and imaginary parts swapped individually; 1-byte dtypes are
        ///     an in-place no-op (but <c>inplace=False</c> still returns a fresh copy).
        /// </summary>
        /// <param name="inplace">
        ///     When <c>true</c>, swap this array's data in place and return this same instance. When
        ///     <c>false</c> (default), return a byte-swapped copy and leave this array untouched.
        /// </param>
        /// <returns>The byte-swapped array (this instance when <paramref name="inplace"/>, else a copy).</returns>
        /// <exception cref="ValueError">
        ///     When <paramref name="inplace"/> is <c>true</c> and the array is not writeable (e.g. a broadcast view).
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     A NEP 55 <c>StringDType</c> array has no byte order to toggle, so the swap is the identity: <c>inplace</c>
        ///     returns this instance with its strings untouched (a read-only array still raises NumPy's
        ///     <see cref="ValueError"/>), and the default returns a copy in NumPy's <c>ANYORDER</c> layout (F only for an
        ///     F-contiguous-and-not-C source), each string duplicated into the copy's own allocator. NumPy 2.4.2
        ///     crashes the process on this call (owner's decision D14, <c>docs/plans/strings-phase1-completion.md</c>).
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.byteswap.html</para>
        /// </remarks>
        public NDArray byteswap(bool inplace = false) => UnmanagedStorage.Byteswap(this, inplace);
    }
}
