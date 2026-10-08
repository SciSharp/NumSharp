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
        ///     Equivalent to <c>np.nonzero(np.ravel(a))[0]</c>, and returned as NumPy returns it: the single
        ///     column of nonzero's <c>(count, 1)</c> multi-index buffer — a unit-stride VIEW (OWNDATA false,
        ///     C- and F-contiguous) of that buffer, which the view keeps alive.
        ///
        ///     <para>
        ///     <b>Implementation:</b> two IL-emitted kernels keyed off the element type
        ///     (<see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereFlatKernel"/>). Same SIMD popcount
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
        ///       <item>size == 0 → empty int64 1-D view of a <c>(0, 1)</c> buffer.</item>
        ///       <item>Contiguous → IL count + IL flat-scan straight into the <c>(count, 1)</c> buffer.</item>
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
        /// <param name="nd">The array.</param>
        /// <returns>The indices (int64): a view of the <c>(count, 1)</c> buffer.</returns>
        /// <exception cref="StringComputationNotSupportedException">A non-empty string array: NumPy reads the strings' truth
        ///     values (the next phase of NumSharp's string support); an empty one gives an empty int64 result.</exception>
        /// <exception cref="NotSupportedException">The IL kernels cannot be emitted for the dtype.</exception>
        /// <remarks>
        ///     <b>BREAKING (NumPy parity):</b> the result used to be a separate owning array. As a view it reports
        ///     OWNDATA false and <c>@base</c> is the <c>(count, 1)</c> buffer; its strides are the dense unit stride, so
        ///     every reader that walks a C-contiguous array (take, fancy indexing, raw pointers) is unaffected.
        /// </remarks>
        public override unsafe NDArray<long> FlatNonZero(NDArray nd)
        {
            // Text: NumPy's flatnonzero is nonzero(ravel(a))[0], the strings' truth values (the next phase). An empty input
            // reads none: the bool stand-in gives the empty int64 result.
            if (Strings.NDStringUfuncs.ReduceStandIn("flatnonzero", nd) is { } standIn)
            {
                using (standIn)
                    return FlatNonZero(standIn);
            }
            // 0-d: NumPy 2.4 raises ValueError, but its own error message suggests
            // `np.atleast_1d(scalar).nonzero()`, which is what our NonZero already
            // does. Preserve symmetry with NonZero — recurse via atleast_1d.
            if (nd.ndim == 0)
            {
                var promoted = np.atleast_1d(nd);
                var flat = FlatNonZero(promoted);
                promoted.Dispose(); // 1-element view wrapper of the scalar's buffer
                return flat;
            }

            long size = nd.size;

            // Empty input → empty int64 1-D result. Covers shape (0,), (0, 3), (2, 0, 4), …
            // NumPy's np.flatnonzero(np.zeros((2, 0, 4))) returns array([], dtype=int64) — the column view of an empty
            // (0, 1) multi-index buffer, OWNDATA false like every other result.
            if (size == 0)
                return MakeEmptyNonZeroResult(1)[0];

            // Materialize non-contig to C-contig. Mirror the ARC pattern from NonZero /
            // Argwhere — the `materialized` local is the GC root we explicitly Dispose
            // in finally so the source buffer survives the IL scan even if the JIT
            // decides `source` is dead after basePtr is computed.
            NDArray materialized = null;
            NDArray source = nd;
            if (!nd.Shape.IsContiguous)
            {
                using var plain = NDMatrixBoundary.PlainFor(nd);   // an engine method computes plain: np.ascontiguousarray below must not apply its matrix rule
                materialized = np.ascontiguousarray(nd);
                source = materialized;
            }
            var sourceShape = source.Shape;

            byte* basePtr = (byte*)source.Storage.Address + sourceShape.offset * nd.dtypesize;

            var countKernel = DirectILKernelGenerator.GetArgwhereCountKernel(nd.dtype.type);
            var flatKernel = DirectILKernelGenerator.GetArgwhereFlatKernel(nd.dtype.type);
            if (countKernel == null || flatKernel == null)
                throw new NotSupportedException($"np.flatnonzero: no IL kernel available for {nd.dtype.type.Name}");

            try
            {
                long count = countKernel(basePtr, size);
                if (count == 0)
                    return MakeEmptyNonZeroResult(1)[0];

                // NumPy's PyArray_Nonzero of the 1-D ravel fills ONE (count, 1) multi-index buffer and np.flatnonzero
                // returns its column. Direct write into that buffer — no temp, no expand step: the flat indices into the
                // C-contig source ARE the flat indices into the raveled input (that's the definition of C-order ravel).
                var multiIndex = new NDArray(NPTypeCode.Int64, new Shape(count, 1), false);
                flatKernel(basePtr, size, (long*)multiIndex.Storage.Address);
                var column = MakeNonZeroColumnViews(multiIndex, count, 1)[0];
                // The column view holds its own counted reference on the buffer; release the buffer's own handle.
                multiIndex.Dispose();
                return column;
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
                // The contiguous route reads the CALLER's operand through basePtr; a temporary operand
                // (`np.argwhere(a > 0)`) is referenced only by this parameter, which the JIT may treat as dead
                // once the pointer is taken — keep it reachable until the kernels are done.
                GC.KeepAlive(nd);
            }
        }
    }
}
