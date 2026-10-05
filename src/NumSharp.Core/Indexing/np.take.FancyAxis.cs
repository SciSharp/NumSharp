using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The source bytes the fancy-axis gather keeps cache-resident per tile: about half of a P-core's 2 MB L2. A
        ///     sweep from 32 KB to 2 MB (2026-10-05, 12 shapes, float64/float32/int8) put 256 KB-1 MB within noise of each
        ///     other and ahead of 32-64 KB (too few rows per index run) and 2 MB (evictions).
        /// </summary>
        private const long FancyAxisReadWindowBytes = 1L << 20;

        /// <summary>
        ///     NumPy's fancy index along ONE non-leading axis, <c>a[(slice(None),) * axis + (indices,)]</c>, laid out as
        ///     NumPy lays it out: a VIEW of a fresh buffer whose first axis is the indexed one (NumPy's MapIter result,
        ///     transposed back by <c>PyArray_MapIterSwapAxes</c>) — <c>owndata</c> False, F-contiguous for 2-D input,
        ///     permuted strides beyond. <see cref="take(NDArray, NDArray, int?, NDArray, string)"/> returns the same
        ///     VALUES C-ordered (owning its data), which NumPy's indexing never returns for a non-leading axis.
        /// </summary>
        /// <param name="a">The source array, any layout (a non-C-contiguous source is copied to C order first, as
        ///     <c>np.take</c> does).</param>
        /// <param name="indices">A 1-D C-contiguous int64 index array whose every value is already in
        ///     <c>[0, a.shape[axis])</c>. NOT bounds-checked: an out-of-range value reads outside the source. The only
        ///     caller, <see cref="Generator.permutation(NDArray, int)"/>, passes a permutation of the axis.</param>
        /// <param name="axis">The normalized indexed axis, <c>1 &lt;= axis &lt; a.ndim</c>. (Axis 0 is NumPy's plain
        ///     <c>a[idx]</c>, C-ordered and owning — use <c>np.take</c>.)</param>
        /// <returns>The gathered array of shape <c>a.shape[:axis] + (indices.size,) + a.shape[axis+1:]</c>: a view whose
        ///     base buffer is laid out <c>(indices.size,) + a.shape[:axis] + a.shape[axis+1:]</c> in C order.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="axis"/> is not a non-leading axis of
        ///     <paramref name="a"/>, or <paramref name="indices"/> is not a 1-D C-contiguous int64 array.</exception>
        /// <exception cref="NotSupportedException">IL generation is switched off
        ///     (<see cref="DirectILKernelGenerator.Enabled"/> false): the fallback runs <c>np.take</c>, whose own IL kernel
        ///     is then unavailable too — the same exception <c>np.take</c> raises.</exception>
        /// <remarks>
        ///     <para>
        ///     The layout is observable (<c>flags.owndata</c>, the contiguity flags, <c>strides</c>), and NumPy produces it
        ///     whenever its own code indexes one non-leading axis (<c>Generator.permutation(x, axis)</c> is
        ///     <c>arr[tuple(slices)]</c>). The buffer is written in one pass by
        ///     <see cref="DirectILKernelGenerator.GetTakeLeadingKernel"/>, tiled so each tile of source rows stays
        ///     cache-resident while every index writes one contiguous run; building it from <c>np.take</c> instead costs a
        ///     transposing copy (or <c>np.take</c>'s own C-order copy of a moved view) — 1.5-5x slower, measured.
        ///     </para>
        ///     <para>
        ///     When the gather kernel is null (its emission failed), the same buffer is built by <c>np.take</c> over the
        ///     moved view: identical values and layout, slower. <c>np.take</c> runs an IL kernel of its own, so with IL
        ///     generation switched off this call raises <see cref="NotSupportedException"/>, as <c>np.take</c> (and the
        ///     <c>np.take</c> route this method replaced in <c>Generator.permutation</c>) does.
        ///     </para>
        /// </remarks>
        [NDScoped] // reclaims a C-order copy of a non-contiguous source and the buffer's own handle; the returned view keeps the buffer alive
        internal static unsafe NDArray FancyIndexAlongAxis(NDArray a, NDArray indices, int axis)
        {
            // Precondition checks are O(1); the per-value bounds are the caller's contract (see <param name="indices"/>).
            if (axis < 1 || axis >= a.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), axis, "FancyIndexAlongAxis serves a non-leading axis only.");
            if (indices.ndim != 1 || indices.typecode != NPTypeCode.Int64 || !indices.Shape.IsContiguous)
                throw new ArgumentOutOfRangeException(nameof(indices), "FancyIndexAlongAxis needs a 1-D C-contiguous int64 index array.");

            // The source's own dimension array, read only (a.shape would hand out a fresh copy per read).
            long[] aDims = a.Shape.dimensions;
            long outer = 1;
            for (int d = 0; d < axis; d++)
                outer *= aDims[d];
            long maxItem = aDims[axis];
            long inner = 1;
            for (int d = axis + 1; d < aDims.Length; d++)
                inner *= aDims[d];
            long innerBytes = inner * a.dtypesize;
            long m = indices.size;

            // NumPy's MapIter buffer: the indexed dimension first, then the remaining axes in their original order.
            var dims = new long[aDims.Length];
            dims[0] = m;
            for (int d = 0, k = 1; d < aDims.Length; d++)
                if (d != axis)
                    dims[k++] = aDims[d];

            NDArray buffer;
            var kernel = DirectILKernelGenerator.GetTakeLeadingKernel(DirectILKernelGenerator.CopyKindFor(innerBytes));
            if (kernel == null)
            {
                // No gather kernel (its emission failed, or IL generation is off — then take raises too): take over the
                // moved view yields exactly the buffer above (slower: take copies the non-contiguous moved view to C
                // order first).
                buffer = take(moveaxis(a, axis, 0), indices, axis: 0);
            }
            else
            {
                buffer = new NDArray(a.typecode, new Shape(dims), false);
                if (buffer.size > 0)
                {
                    // The kernel addresses the source as (outer, maxItem, inner) C-contiguous; any other layout is copied
                    // to C order first (the scope reclaims that copy).
                    var source = a.Shape.IsContiguous ? a : ascontiguousarray(a);

                    // Tile: enough source rows to fill the read window, at least one 64-byte line of contiguous writes per
                    // index, never more rows than exist.
                    long rowBytes = Math.Max(maxItem * innerBytes, 1);
                    long tileRows = Math.Max(1, FancyAxisReadWindowBytes / rowBytes);
                    tileRows = Math.Max(tileRows, (64 + innerBytes - 1) / Math.Max(innerBytes, 1));
                    tileRows = Math.Min(tileRows, outer);

                    byte* src = (byte*)source.Storage.Address + source.Shape.offset * source.dtypesize;
                    long* idx = (long*)indices.Storage.Address + indices.Shape.offset;
                    kernel(src, idx, m, outer, maxItem, innerBytes, tileRows, (byte*)buffer.Storage.Address);
                }
            }

            // NumPy returns the buffer transposed back into a's axis order: a view, never the buffer itself.
            return moveaxis(buffer, 0, axis);
        }
    }
}
