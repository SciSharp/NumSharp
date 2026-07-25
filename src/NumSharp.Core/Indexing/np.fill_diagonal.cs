using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Fill the main diagonal of the given array of any dimensionality — <b>in place</b>.
        /// </summary>
        /// <param name="a">Array whose diagonal is to be filled; it is modified in place.</param>
        /// <param name="val">
        ///     Value(s) to write into the diagonal. A scalar is repeated; a sequence is raveled
        ///     and then <b>tiled cyclically</b> (and truncated) to the diagonal's length.
        /// </param>
        /// <param name="wrap">
        ///     For tall matrices the diagonal "wraps" after N columns and continues on the row
        ///     below. Off by default.
        /// </param>
        /// <exception cref="ArgumentException">
        ///     <c>array must be at least 2-d</c>, <c>All dimensions of input must be of equal length</c>
        ///     (ndim &gt; 2 with a non-hyper-cubic shape), or <c>underlying array is read-only</c>
        ///     — all verbatim NumPy <c>ValueError</c> texts.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html
        ///     <para>
        ///     <b>Values tile, they do not broadcast.</b> NumPy's body ends in
        ///     <c>a.flat[:end:step] = val</c>, and flat-iterator assignment repeats a short
        ///     right-hand side rather than raising: probed on a 6×6, <c>[1,2,3,4]</c> fills the
        ///     diagonal as <c>[1,2,3,4,1,2]</c>, while an over-long value list is truncated and
        ///     an <b>empty</b> one is a silent no-op. <see cref="resize(NDArray,int[])"/> already
        ///     implements exactly that cyclic-tile-or-truncate rule, so it supplies the values.
        ///     </para>
        ///     <para>
        ///     <b>No element loop.</b> NumPy addresses the diagonal as a strided slice of the
        ///     flat iterator; NumSharp resolves the same positions into <em>real strides</em> and
        ///     writes through ordinary aliased views. Without wrapping the targets are
        ///     <c>(i, i, …, i)</c>, i.e. one constant stride of <c>Σ strides</c> — a single view.
        ///     With wrapping on a tall matrix the flat position of target <c>i = q·cols + r</c>
        ///     is <c>q·(cols+1)·s0 + r·(s0+s1)</c>, so the run splits into <c>ceil(count/cols)</c>
        ///     equally-strided blocks — a handful of views, still no per-element addressing.
        ///     Because the targets are computed from strides rather than from memory order, this
        ///     works unchanged on transposed, sliced and otherwise non-contiguous arrays, which
        ///     it writes through exactly as NumPy does.
        ///     </para>
        /// </remarks>
        public static void fill_diagonal(NDArray a, object val, bool wrap = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            int ndim = a.ndim;
            if (ndim < 2)
                throw new ArgumentException("array must be at least 2-d");

            long step;
            long end;
            if (ndim == 2)
            {
                // Explicit, fast formula for the common case; rectangular arrays are accepted.
                long cols = a.shape[1];
                step = cols + 1;
                // Without wrapping, stop before a tall matrix's diagonal re-enters column 0.
                end = wrap ? a.size : cols * cols;
            }
            else
            {
                // For d > 2 the strided formula is only valid when all dimensions are equal.
                RequireEqualDimensions(a);

                step = 1;
                long cum = 1;
                for (int d = 0; d < ndim - 1; d++)
                {
                    cum *= a.shape[d];
                    step += cum;
                }

                end = a.size;
            }

            if (end > a.size) end = a.size;
            if (end <= 0 || step <= 0)
                return;

            long count = (end + step - 1) / step;
            if (count <= 0)
                return;

            if (!a.Shape.IsWriteable)
                throw new ArgumentException("underlying array is read-only");

            if (val is null)
                // NumPy's `None` reaches the dtype's converter: nan for a float array, TypeError
                // for an integral one. In C# a null `val` is a caller bug either way, and
                // silently writing nan/zero would hide it.
                throw new ArgumentNullException(nameof(val));

            // The overwhelmingly common call is a SCALAR fill. Building it straight at the
            // destination dtype skips asanyarray's dtype inference plus the ravel/astype pair,
            // and a stride-0 broadcast view then supplies `count` copies for free — the copy
            // machinery reads it just as happily as a materialised buffer.
            //
            // ONLY genuine scalars may take it: NDArray.Scalar goes through IConvertible, so a
            // C# array / List (NumPy's list-valued `val`, which must tile) would throw. Those —
            // along with Half and Complex, which are not IConvertible — take the general path.
            NDArray values;
            if (val is IConvertible && !(val is string))
            {
                values = np.broadcast_to(NDArray.Scalar(val, a.typecode), new Shape(count));
            }
            else
            {
                var source = val as NDArray ?? np.asanyarray(val);
                if (source.size == 0)
                    return; // NumPy: assigning an empty sequence leaves the diagonal untouched.

                var flat = source.ravel();
                if (flat.typecode != a.typecode)
                    flat = flat.astype(a.typecode);

                // Ravel + cyclic tile/truncate to exactly `count` values (np.resize's rule).
                values = flat.size == 1
                    ? np.broadcast_to(flat, new Shape(count))
                    : np.resize(flat, new[] {(int)count});
            }

            var shape = a.Shape;
            long bufferSize = shape.BufferSize;

            if (ndim > 2)
            {
                // Hyper-cubic: targets are (i, i, …, i) — one constant stride.
                long diagStride = 0;
                for (int d = 0; d < ndim; d++)
                    diagStride += shape.strides[d];

                WriteDiagonalBlock(a, values, 0, count, shape.Offset, diagStride, bufferSize);
                return;
            }

            long s0 = shape.strides[0];
            long s1 = shape.strides[1];
            long colCount = a.shape[1];
            long blockStride = s0 + s1;

            // Each block of `cols` targets marches down the diagonal; the next block restarts
            // `cols + 1` rows lower (that is the wrap). Without wrapping count <= cols, so
            // this degenerates to a single block.
            for (long written = 0; written < count; written += colCount)
            {
                long len = System.Math.Min(colCount, count - written);
                long q = written / colCount;
                long offset = shape.Offset + q * (colCount + 1) * s0;

                WriteDiagonalBlock(a, values, written, len, offset, blockStride, bufferSize);
            }
        }

        /// <summary>
        ///     Write <paramref name="len"/> values into an equally-strided run of
        ///     <paramref name="a"/>'s storage by aliasing it as a 1-D view.
        /// </summary>
        private static void WriteDiagonalBlock(
            NDArray a, NDArray values, long valueStart, long len, long offset, long stride, long bufferSize)
        {
            if (len <= 0) return;

            var targetShape = new Shape(new[] {len}, new[] {stride}, offset, bufferSize);
            var target = new NDArray(a.Storage.Alias(targetShape)) {TensorEngine = a.TensorEngine};

            var slice = valueStart == 0 && len == values.size
                ? values
                : values[$"{valueStart}:{valueStart + len}"];

            target.SetData(slice);
        }
    }
}
