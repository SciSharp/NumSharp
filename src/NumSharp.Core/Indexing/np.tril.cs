using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Lower triangle of an array — a copy of <paramref name="m"/> with all elements
        ///     <em>above</em> the <paramref name="k"/>-th diagonal zeroed.
        /// </summary>
        /// <param name="m">Input array. For arrays with <c>ndim &gt; 2</c>, <c>tril</c> applies to the final two axes.</param>
        /// <param name="k">Diagonal above which to zero elements. <c>k = 0</c> (the default) is the main diagonal, <c>k &lt; 0</c> is below it and <c>k &gt; 0</c> is above.</param>
        /// <returns>Lower triangle of <paramref name="m"/>, of the same dtype. Always a fresh, writeable, C-contiguous array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tril.html</remarks>
        public static NDArray tril(NDArray m, int k = 0)
            => Triangle(m, k, lower: true);

        /// <summary>
        ///     Upper triangle of an array — a copy of <paramref name="m"/> with all elements
        ///     <em>below</em> the <paramref name="k"/>-th diagonal zeroed.
        /// </summary>
        /// <param name="m">Input array. For arrays with <c>ndim &gt; 2</c>, <c>triu</c> applies to the final two axes.</param>
        /// <param name="k">Diagonal below which to zero elements. <c>k = 0</c> (the default) is the main diagonal, <c>k &lt; 0</c> is below it and <c>k &gt; 0</c> is above.</param>
        /// <returns>Upper triangle of <paramref name="m"/>, of the same dtype. Always a fresh, writeable, C-contiguous array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.triu.html</remarks>
        public static NDArray triu(NDArray m, int k = 0)
            => Triangle(m, k, lower: false);

        /// <summary>
        ///     Shared implementation of <see cref="tril"/> / <see cref="triu"/>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy is literally <c>where(tri(*m.shape[-2:], k=k, dtype=bool), m, zeros(1, m.dtype))</c>
        ///     for <c>tril</c> (and the complementary <c>k-1</c> mask for <c>triu</c>). Two
        ///     consequences of that spelling are load-bearing behaviour, reproduced here:
        ///     </para>
        ///     <list type="bullet">
        ///       <item>
        ///         <b>A 1-D input produces a 2-D result.</b> <c>m.shape[-2:]</c> of a
        ///         <c>(n,)</c> array is <c>(n,)</c>, so the mask is <c>tri(n)</c> → <c>(n, n)</c>
        ///         and the <c>where</c> broadcasts the row across it. Probed:
        ///         <c>np.tril([1,2,3])</c> → <c>[[1,0,0],[1,2,0],[1,2,3]]</c>.
        ///       </item>
        ///       <item>
        ///         <b>A 0-d input raises</b> — <c>m.shape[-2:]</c> is empty and NumPy leaks
        ///         <c>tri()</c>'s missing-argument <c>TypeError</c> straight to the caller. The
        ///         text is reproduced verbatim.
        ///       </item>
        ///     </list>
        ///     <para>
        ///     <b>Fast path.</b> Because the kept columns of every row form one contiguous run
        ///     (<c>[0, c_i)</c> for <c>tril</c>, <c>[lo_i, cols)</c> for <c>triu</c>), the mask
        ///     is redundant: allocate the result zeroed and blit just the kept run of each row.
        ///     That reads and writes only the retained triangle — roughly half the traffic of
        ///     NumPy's two full-size passes — with no mask temporary and no per-dtype code (the
        ///     copy is byte-wise). It applies whenever the source's last axis is unit-stride;
        ///     anything else (broadcast last axis, F-contiguous, <c>::2</c> column steps) falls
        ///     back to the literal <c>where</c> composition, which is always correct.
        ///     </para>
        /// </remarks>
        private static NDArray Triangle(NDArray m, int k, bool lower)
        {
            if (m is null) throw new ArgumentNullException(nameof(m));

            int ndim = m.ndim;
            if (ndim == 0)
                // NumPy leaks tri()'s own TypeError here: `tri(*m.shape[-2:])` gets no args.
                throw new ArgumentException("tri() missing 1 required positional argument: 'N'");

            // NumPy: tri(*m.shape[-2:], ...) — for 1-D that is tri(n) which squares up to (n, n).
            long rows = ndim == 1 ? m.shape[0] : m.shape[ndim - 2];
            long cols = ndim == 1 ? m.shape[0] : m.shape[ndim - 1];

            var fast = TryTriangleBlit(m, k, lower, cols);
            if (!(fast is null))
                return fast;

            // Literal NumPy composition — the mask's diagonal differs by one between the
            // two directions, and the operand order is what flips which half survives.
            var mask = np.tri(checked((int)rows), checked((int)cols), lower ? k : k - 1, NPTypeCode.Boolean);
            var zero = np.zeros(new Shape(1), m.typecode);
            return lower ? np.where(mask, m, zero) : np.where(mask, zero, m);
        }

        /// <summary>
        ///     Zero-allocate the result and copy only the retained column run of each row.
        ///     Returns <c>null</c> when the layout is not eligible, leaving the caller to use
        ///     the general <c>where</c> composition.
        /// </summary>
        private static unsafe NDArray TryTriangleBlit(NDArray m, int k, bool lower, long cols)
        {
            int ndim = m.ndim;

            // The 1-D case broadcasts a row across a square result — a different traversal
            // than "copy a run of each source row", so leave it to the composition path.
            if (ndim < 2)
                return null;

            var shape = m.Shape;

            // A unit-strided last axis is what makes the kept run contiguous in the SOURCE;
            // the destination is contiguous by construction. Broadcast (stride 0) and
            // column-stepped views fail this and fall back.
            if (shape.strides[ndim - 1] != 1)
                return null;

            long size = m.size;
            if (size == 0 || cols == 0)
                return np.zeros(new Shape((long[])m.shape.Clone()), m.typecode);

            int elemSize = m.typecode.SizeOf();
            bool writeOnce = PrefersWriteOnce(size * elemSize);

            // Below the threshold the buffer comes back dirty, so allocate uninitialised and
            // write every byte of every row exactly once (zeros, retained run, zeros). Above it
            // np.zeros rides free OS zero-pages and only the retained run is worth touching.
            var res = writeOnce
                ? new NDArray(m.typecode, new Shape((long[])m.shape.Clone()), false)
                : np.zeros(new Shape((long[])m.shape.Clone()), m.typecode);
            byte* src = m.Storage.Address;
            byte* dst = res.Storage.Address + res.Shape.Offset * elemSize;

            long numRows = size / cols;
            int outer = ndim - 1; // number of axes above the column axis

            // Odometer over every axis except the last: one iteration per output row.
            Span<long> coord = stackalloc long[outer];
            coord.Clear();

            for (long r = 0; r < numRows; r++)
            {
                long i = coord[outer - 1]; // row index within the trailing 2-D plane

                // Retained column run for this row.
                long lo, hi;
                if (lower)
                {
                    lo = 0;
                    hi = i + k + 1;
                    if (hi > cols) hi = cols;
                    if (hi < 0) hi = 0;
                }
                else
                {
                    lo = i + k;
                    if (lo < 0) lo = 0;
                    if (lo > cols) lo = cols;
                    hi = cols;
                }

                byte* dstRow = dst + r * cols * elemSize;

                if (writeOnce)
                    ZeroBytes(dstRow, lo * elemSize);

                if (hi > lo)
                {
                    // Source element offset of (coord..., lo) through the real strides.
                    long srcElem = shape.Offset;
                    for (int d = 0; d < outer; d++)
                        srcElem += coord[d] * shape.strides[d];
                    srcElem += lo;

                    long bytes = (hi - lo) * elemSize;
                    Buffer.MemoryCopy(
                        source: src + srcElem * elemSize,
                        destination: dstRow + lo * elemSize,
                        destinationSizeInBytes: bytes,
                        sourceBytesToCopy: bytes);
                }

                if (writeOnce)
                    ZeroBytes(dstRow + hi * elemSize, (cols - hi) * elemSize);

                // Advance the odometer.
                for (int d = outer - 1; d >= 0; d--)
                {
                    if (++coord[d] < shape.dimensions[d]) break;
                    coord[d] = 0;
                }
            }

            return res;
        }
    }
}
