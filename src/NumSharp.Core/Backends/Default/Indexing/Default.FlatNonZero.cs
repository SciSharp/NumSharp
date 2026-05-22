using System;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.flatnonzero</c> — returns the int64 1-D array of flat indices of
        ///     non-zero elements in the raveled (C-order) input.
        ///     Equivalent to <c>np.nonzero(np.ravel(a))[0]</c>.
        ///
        ///     <para>
        ///     <b>Implementation:</b> two IL-emitted kernels keyed off the element type
        ///     (<see cref="ILKernelGenerator.GetArgwhereCountKernel"/>,
        ///     <see cref="ILKernelGenerator.GetArgwhereFlatKernel"/>). Same SIMD popcount
        ///     and bit-scan as <see cref="NonZero"/> / <see cref="Argwhere"/>, except we
        ///     never expand the flat indices into per-axis coords — the flat indices ARE
        ///     the result. No per-dim allocation, no expand kernel, no coord carry chain.
        ///     </para>
        ///
        ///     <para>
        ///     Routing:
        ///     <list type="bullet">
        ///       <item>0-d → promote via <c>atleast_1d</c>, recurse
        ///             (truthy→<c>[0]</c>, falsy→<c>[]</c>).</item>
        ///       <item>size == 0 → empty int64 1-D array.</item>
        ///       <item>Contiguous → IL count + IL flat-scan straight into the result buffer.</item>
        ///       <item>Non-contiguous → materialize via <c>ascontiguousarray</c> then
        ///             same path as contig (flat indices into the C-contig buffer ARE the
        ///             flat indices of the raveled non-contig input — that's what NumPy
        ///             returns).</item>
        ///     </list>
        ///     </para>
        ///
        ///     <para>
        ///     This is the cheapest of the three nonzero-family entry points: same SIMD
        ///     work as <see cref="NonZero"/>, no per-axis NDArray array allocations, no
        ///     expand IL kernel invocation. Multi-dim inputs cost the same as 1-D inputs
        ///     of equal size (modulo the non-contig materialization if needed).
        ///     </para>
        /// </summary>
        public override unsafe NDArray<long> FlatNonZero(NDArray nd)
        {
            // 0-d: NumPy 2.4 raises ValueError, but its own error message suggests
            // `np.atleast_1d(scalar).nonzero()`, which is what our NonZero already
            // does. Preserve symmetry with NonZero — recurse via atleast_1d.
            if (nd.ndim == 0)
                return FlatNonZero(np.atleast_1d(nd));

            long size = nd.size;

            // Empty input → empty int64 1-D result. Covers shape (0,), (0, 3), (2, 0, 4), …
            // NumPy's np.flatnonzero(np.zeros((2, 0, 4))) returns array([], dtype=int64).
            if (size == 0)
                return new NDArray<long>(0);

            // Materialize non-contig to C-contig. Mirror the ARC pattern from NonZero /
            // Argwhere — the `materialized` local is the GC root we explicitly Dispose
            // in finally so the source buffer survives the IL scan even if the JIT
            // decides `source` is dead after basePtr is computed.
            NDArray materialized = null;
            NDArray source = nd;
            if (!nd.Shape.IsContiguous)
            {
                materialized = np.ascontiguousarray(nd);
                source = materialized;
            }
            var sourceShape = source.Shape;

            byte* basePtr = (byte*)source.Storage.Address + sourceShape.offset * nd.dtypesize;

            var countKernel = ILKernelGenerator.GetArgwhereCountKernel(nd.dtype);
            var flatKernel = ILKernelGenerator.GetArgwhereFlatKernel(nd.dtype);
            if (countKernel == null || flatKernel == null)
                throw new NotSupportedException($"np.flatnonzero: no IL kernel available for {nd.dtype.Name}");

            try
            {
                long count = countKernel(basePtr, size);
                if (count == 0)
                    return new NDArray<long>(0);

                // Direct write into the result buffer — no temp, no expand step.
                // The flat indices into the C-contig source ARE the flat indices into
                // the raveled input (that's the definition of C-order ravel).
                var result = new NDArray<long>(count);
                flatKernel(basePtr, size, (long*)result.Address);
                return result;
            }
            finally
            {
                // ARC: see the matching block in NonZero / Argwhere for the full
                // explanation. Short version: source.Storage.Address is an unmanaged
                // pointer that doesn't keep `materialized` GC-alive, so without the
                // explicit Dispose() the new NDArray<long>(count) allocation above
                // can trigger a GC that frees the buffer behind basePtr mid-scan
                // (Release-mode AccessViolationException reproduced by the nonzero
                // bench harness; this method shares the same fix preventively).
                materialized?.Dispose();
            }
        }
    }
}
