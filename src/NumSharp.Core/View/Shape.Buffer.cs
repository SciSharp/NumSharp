using System;

namespace NumSharp
{
    public partial struct Shape
    {
        // The stride arithmetic of NumPy's ndarray.__new__ (numpy/_core/src/multiarray/arrayobject.c: array_new), which
        // lays an array over a caller's BUFFER or over a fresh allocation from explicit byte strides. Everything here
        // works on raw dimension and BYTE-stride arrays, as NumPy does: the dimensions may still be negative (NumPy checks
        // the strides before it checks the dimensions), and the strides are the caller's, not yet converted to NumSharp's
        // element strides. np.ndarray (Creation/np.ndarray.cs) is the caller.
        //
        // Two arithmetics live side by side on purpose. CheckStrides and OffsetBoundsFromStrides are NumPy's own checks,
        // ported with C's 64-bit wrapping arithmetic, so a NumSharp call fails exactly where NumPy's does and with the same
        // error (probed: np.ndarray((2**61,), strides=(0,)) is the strides error, because 2**61 * 8 wraps to 0 bytes).
        // ExactByteExtent is the memory-safety check NumPy lacks: it runs after NumPy's checks pass and refuses any layout
        // whose elements would fall outside the buffer, which NumPy's wrapped or misordered checks let through.

        /// <summary>
        ///     The product of <paramref name="dims"/> with C's 64-bit wrapping arithmetic — NumPy's
        ///     <c>PyArray_MultiplyList</c>. A negative dimension makes the product negative, and an overflow wraps, exactly
        ///     as NumPy's own size checks see it.
        /// </summary>
        /// <param name="dims">The dimensions, possibly negative (not yet validated).</param>
        /// <returns>The wrapped product; 1 for no dimensions (a 0-d shape).</returns>
        /// <remarks>Use it only where NumPy's checks must be reproduced bit for bit. A size that addresses memory comes
        ///     from <see cref="ExactByteExtent"/> or from <see cref="AllocationGuard.CheckDimensions(long[], int)"/>, which
        ///     detect the overflow.</remarks>
        internal static long MultiplyListWrapping(long[] dims)
        {
            long product = 1;
            unchecked
            {
                for (int i = 0; i < dims.Length; i++)
                    product *= dims[i];
            }

            return product;
        }

        /// <summary>
        ///     The half-open byte range <c>[lower, upper)</c> that an array with <paramref name="dims"/> and
        ///     <paramref name="byteStrides"/> touches, relative to its first element — NumPy's
        ///     <c>offset_bounds_from_strides</c> (<c>numpy/_core/src/common/mem_overlap.c</c>), with C's wrapping arithmetic.
        /// </summary>
        /// <param name="itemSize">Bytes per element.</param>
        /// <param name="dims">The dimensions; a negative one is used as it is (NumPy checks the strides first).</param>
        /// <param name="byteStrides">The byte stride of every axis (same length as <paramref name="dims"/>).</param>
        /// <param name="lower">Receives the lowest touched byte offset (0 or negative).</param>
        /// <param name="upper">Receives one past the highest touched byte offset.</param>
        /// <remarks>
        ///     A zero dimension gives the empty range <c>[0, 0)</c>, whatever the strides. Each axis adds
        ///     <c>stride * (dim - 1)</c> to <paramref name="upper"/> when positive, else to <paramref name="lower"/>, and the
        ///     last element's <paramref name="itemSize"/> bytes close the range. The arithmetic wraps like NumPy's
        ///     <c>npy_intp</c>, so the caller sees NumPy's verdict for a huge stride or a negative dimension; use
        ///     <see cref="ExactByteExtent"/> for a range that addresses memory.
        /// </remarks>
        internal static void OffsetBoundsFromStrides(int itemSize, long[] dims, long[] byteStrides, out long lower, out long upper)
        {
            long lo = 0, hi = 0;
            unchecked
            {
                for (int i = 0; i < dims.Length; i++)
                {
                    if (dims[i] == 0)
                    {
                        // A zero-size array touches no memory: NumPy returns the empty range at once.
                        lower = 0;
                        upper = 0;
                        return;
                    }

                    long maxAxisOffset = byteStrides[i] * (dims[i] - 1);
                    if (maxAxisOffset > 0)
                        hi += maxAxisOffset;
                    else
                        lo += maxAxisOffset;
                }

                hi += itemSize;
            }

            lower = lo;
            upper = hi;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_CheckStrides</c> (<c>arrayobject.c</c>): can an array with these dimensions and byte strides,
        ///     starting <paramref name="offset"/> bytes into a buffer of <paramref name="numBytes"/> bytes, touch only that
        ///     buffer? With C's wrapping arithmetic, so the verdict is NumPy's for every input, including the inputs where
        ///     NumPy's arithmetic overflows.
        /// </summary>
        /// <param name="itemSize">Bytes per element.</param>
        /// <param name="dims">The dimensions, possibly negative.</param>
        /// <param name="numBytes">The buffer's length in bytes; 0 means "the array's own size",
        ///     <c>prod(dims) * itemSize</c>, which is what NumPy checks a fresh allocation against.</param>
        /// <param name="offset">The byte offset of the first element into the buffer (0 for a fresh allocation).</param>
        /// <param name="byteStrides">The byte stride of every axis.</param>
        /// <returns>True when NumPy accepts the strides; false when it raises
        ///     <c>ValueError: strides is incompatible with shape of requested array and size of buffer</c>.</returns>
        /// <remarks>
        ///     NumPy recomputes a zero <paramref name="numBytes"/> from the dimensions even when a real buffer of length 0 was
        ///     given, so an empty buffer passes as if it held the whole array. That and the wrapped arithmetic are why the
        ///     caller follows a passing check with <see cref="ExactByteExtent"/> before it addresses memory.
        /// </remarks>
        internal static bool CheckStrides(int itemSize, long[] dims, long numBytes, long offset, long[] byteStrides)
        {
            unchecked
            {
                if (numBytes == 0)
                    numBytes = MultiplyListWrapping(dims) * itemSize;

                long begin = -offset;
                long end = numBytes - offset;

                OffsetBoundsFromStrides(itemSize, dims, byteStrides, out long lower, out long upper);
                return !(upper > end || lower < begin);
            }
        }

        /// <summary>
        ///     The exact half-open byte range <c>[lower, upper)</c> an array touches relative to its first element: the same
        ///     rule as <see cref="OffsetBoundsFromStrides"/> computed without overflow, for the memory-safety check.
        /// </summary>
        /// <param name="itemSize">Bytes per element.</param>
        /// <param name="dims">The dimensions, already validated by <see cref="AllocationGuard.CheckDimensions(long[], int)"/>
        ///     (non-negative, and their byte product fits a <see cref="long"/>).</param>
        /// <param name="byteStrides">The byte stride of every axis.</param>
        /// <param name="lower">Receives the lowest touched byte offset (0 or negative), exactly.</param>
        /// <param name="upper">Receives one past the highest touched byte offset, exactly; 0 for a zero-size array.</param>
        /// <remarks>
        ///     <see cref="Int128"/> cannot overflow under that precondition: each term <c>|stride| * (dim - 1)</c> is below
        ///     <c>2^63 * dim</c>, and the dimensions longer than 1 multiply to less than 2^63, so they also SUM to less than
        ///     2^63; the whole range stays below 2^126.
        /// </remarks>
        internal static void ExactByteExtent(int itemSize, long[] dims, long[] byteStrides, out Int128 lower, out Int128 upper)
        {
            Int128 lo = 0, hi = 0;
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] == 0)
                {
                    lower = 0;
                    upper = 0;
                    return;
                }

                Int128 maxAxisOffset = (Int128)byteStrides[i] * (dims[i] - 1);
                if (maxAxisOffset > 0)
                    hi += maxAxisOffset;
                else
                    lo += maxAxisOffset;
            }

            lower = lo;
            upper = hi + itemSize;
        }

        /// <summary>
        ///     The byte strides NumPy gives an array laid out over a buffer from its dimensions alone — the strides half of
        ///     <c>_array_fill_strides</c> (<c>numpy/_core/src/multiarray/ctors.c</c>): C order, or F order when
        ///     <paramref name="fortran"/> is true.
        /// </summary>
        /// <param name="dims">The dimensions (non-negative).</param>
        /// <param name="itemSize">Bytes per element: the stride of the fastest axis.</param>
        /// <param name="fortran">True for column-major strides, false for row-major.</param>
        /// <returns>A new array of byte strides, one per axis.</returns>
        /// <remarks>
        ///     Unlike NumSharp's own C/F stride builders, a ZERO extent is not multiplied into the running product: NumPy
        ///     skips it (<c>if (dims[i]) itemsize *= dims[i]</c>), so <c>(3, 0)</c> of float64 has strides <c>(8, 8)</c> and
        ///     <c>(0, 3)</c> has <c>(24, 8)</c>. That is observable on an empty array laid over a buffer (probed:
        ///     <c>np.ndarray((3, 0), buffer=b'')</c>). An array NumPy ALLOCATES empty has all-zero strides instead (the
        ///     allocator zeroes them); the caller handles that case.
        /// </remarks>
        internal static long[] FillByteStrides(long[] dims, int itemSize, bool fortran)
        {
            var strides = new long[dims.Length];
            long running = itemSize;
            if (fortran)
            {
                for (int i = 0; i < dims.Length; i++)
                {
                    strides[i] = running;
                    if (dims[i] != 0)
                        running *= dims[i];
                }
            }
            else
            {
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    strides[i] = running;
                    if (dims[i] != 0)
                        running *= dims[i];
                }
            }

            return strides;
        }

        /// <summary>
        ///     Converts NumPy's BYTE strides to NumSharp's ELEMENT strides. It fails only when a stride that addresses
        ///     elements is not a whole number of elements.
        /// </summary>
        /// <param name="byteStrides">The byte stride of every axis.</param>
        /// <param name="dims">The dimensions (validated, non-negative). They decide which strides address elements.</param>
        /// <param name="itemSize">Bytes per element.</param>
        /// <param name="elementStrides">Receives a new array of element strides, or null when the conversion fails.</param>
        /// <param name="badAxis">Receives the first axis whose byte stride splits an element, or -1.</param>
        /// <returns>True when every stride that addresses elements converts exactly.</returns>
        /// <remarks>
        ///     <para>
        ///     NumSharp stores strides in elements, so a byte stride that splits an element (<c>strides=(6, 2)</c> for
        ///     int32, which NumPy accepts as an unaligned array) has no NumSharp representation, and the caller refuses
        ///     it. A byte OFFSET that splits an element is representable: the data pointer itself is unaligned, which
        ///     NumSharp's kernels handle (they use unaligned loads).
        ///     </para>
        ///     <para>
        ///     A stride that addresses no element is converted by truncation instead of refused: the stride of an axis
        ///     of extent 1 (index 0 only), and every stride of an empty array. NumPy accepts such strides freely
        ///     (its own tests build <c>shape=(10, 5, 1), strides=(8, 80, 4)</c> for float64), and every element address
        ///     stays exact. The cost is that <c>ndarray.strides</c> then reports the truncated value
        ///     (<c>0</c> for <c>4 // 8</c>) where NumPy reports the given one.
        ///     </para>
        /// </remarks>
        internal static bool TryElementStrides(long[] byteStrides, long[] dims, int itemSize, out long[] elementStrides, out int badAxis)
        {
            // An empty array addresses no element at all, so none of its strides is used.
            bool empty = false;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] == 0)
                    empty = true;

            var result = new long[byteStrides.Length];
            for (int i = 0; i < byteStrides.Length; i++)
            {
                long s = byteStrides[i];
                if (s % itemSize != 0 && !empty && dims[i] > 1)
                {
                    elementStrides = null;
                    badAxis = i;
                    return false;
                }

                // Exact for a whole number of elements; truncated (and harmless) for an unused stride.
                result[i] = s / itemSize;
            }

            elementStrides = result;
            badAxis = -1;
            return true;
        }
    }
}
