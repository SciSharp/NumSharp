using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array with the specified shape.
        ///     <para>
        ///     If the new array is larger than the original array, then the new array is filled
        ///     with repeated copies of <paramref name="a"/> (iterating over <paramref name="a"/>
        ///     in C-order, cycling back from the start). Note that this behavior is different from
        ///     <see cref="NDArray.resize(long[])"/> which fills with zeros instead.
        ///     </para>
        ///     <para>
        ///     NumPy's <c>np.resize</c> takes <c>new_shape</c> as a <b>single</b> argument (an int or a
        ///     sequence) — the multi-argument form <c>np.resize(a, 2, 3)</c> is a NumPy <c>TypeError</c>,
        ///     so it is not offered here. Pass the shape as an int, a value tuple, an array, or a
        ///     <see cref="Shape"/>; all resolve through <see cref="Shape"/>'s implicit conversions:
        ///     <c>np.resize(a, 6)</c>, <c>np.resize(a, (2, 3))</c>, <c>np.resize(a, new[]{2, 3})</c>,
        ///     <c>np.resize(a, new Shape(2, 3))</c>.
        ///     </para>
        /// </summary>
        /// <param name="a">Array to be resized.</param>
        /// <param name="new_shape">Shape of resized array (int / tuple / array / <see cref="Shape"/>).</param>
        /// <returns>
        ///     The new array is formed from the data in the old array, repeated if necessary to
        ///     fill out the required number of elements. The data are repeated iterating over the
        ///     array in C-order. Result is C-contiguous; dtype matches <paramref name="a"/>.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.resize.html</remarks>
        /// <exception cref="ArgumentNullException">If <paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentException">If any element of <paramref name="new_shape"/> is negative.</exception>
        public static unsafe NDArray resize(NDArray a, Shape new_shape)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            // Validate dimensions and compute the target element count. NumPy's np.resize
            // rejects negative dims with this exact message (distinct from ndarray.resize's
            // "negative dimensions not allowed").
            long newSize = 1;
            var newDims = new_shape.dimensions ?? Array.Empty<long>();
            for (int i = 0; i < newDims.Length; i++)
            {
                if (newDims[i] < 0)
                    throw new ArgumentException("all elements of `new_shape` must be non-negative");
                newSize *= newDims[i];
            }

            // The result is always C-contiguous (NumPy reshapes the concatenated 1-D read-out),
            // so normalize to C-order dims — independent of any strides carried by new_shape.
            var outShape = new Shape(newDims);

            // Flatten to a 1-D C-order read-out. ravel returns a view when a is already
            // contiguous, else a fresh contiguous copy — either way a dense run of `a.size`
            // elements starting at flat.Shape.offset. Disposed after the tiling copy.
            using var flat = ravel(a);
            long srcSize = flat.size;

            // First case must zero-fill (empty source has nothing to repeat); the second
            // would repeat zero times. NumPy: np.zeros_like(a, shape=new_shape).
            if (srcSize == 0 || newSize == 0)
                return zeros(outShape, a.typecode);

            // Allocate the exact-sized C-contiguous output and tile the source bytes into it.
            // This beats NumPy's concatenate((a,)*repeats)[:new_size] which over-allocates a
            // repeats*size buffer then slices — we allocate exactly new_size and fill by
            // doubling block copies (dtype-agnostic raw memory, O(new_size) bytes,
            // O(log(new_size/size)) memcpy calls).
            var result = new NDArray(a.typecode, outShape, fillZeros: false);

            long itemsize = a.dtypesize;
            byte* src = flat.Storage.Address + (long)flat.Shape.offset * itemsize;
            byte* dst = result.Storage.Address;
            long totalBytes = newSize * itemsize;
            long srcBytes = srcSize * itemsize;

            // Seed with the source (truncated when shrinking), then repeatedly double the
            // filled region from dst onto itself until the whole output is covered.
            long filled = Math.Min(srcBytes, totalBytes);
            Buffer.MemoryCopy(src, dst, totalBytes, filled);
            while (filled < totalBytes)
            {
                long chunk = Math.Min(filled, totalBytes - filled);
                Buffer.MemoryCopy(dst, dst + filled, totalBytes - filled, chunk);
                filled += chunk;
            }

            return result;
        }
    }
}
