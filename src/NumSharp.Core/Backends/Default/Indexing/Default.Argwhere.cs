using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.argwhere</c> — returns the (N, ndim) int64 array of coordinates of
        ///     non-zero elements, traversed in C-order. Equivalent to
        ///     <c>np.transpose(np.nonzero(a))</c> with NumPy's 0-d special case
        ///     (truthy → (1,0), falsy → (0,0)), and returned as NumPy returns it: <c>np.transpose</c> stacks
        ///     nonzero's index arrays into a fresh <c>(ndim, N)</c> C-order array and hands back its transpose — an
        ///     F-contiguous VIEW (OWNDATA false, element strides <c>(1, N)</c>) that keeps that array alive.
        ///
        ///     <para>
        ///     <b>Implementation:</b> three IL-emitted kernels keyed off the element type
        ///     (<see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereFlatKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetNonZeroPerDimKernel"/>). The runtime call
        ///     site has zero <c>typeof(T)</c> branches: it looks the kernels up in the
        ///     per-dtype <see cref="System.Collections.Concurrent.ConcurrentDictionary{Type,Object}"/>
        ///     cache and invokes them. Every loop (SIMD body, scalar tail, coord-expand
        ///     carry chain) lives inside the emitted IL.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Two-pass pre-size-then-fill:</b> a SIMD popcount sizes the <c>(ndim, N)</c> array
        ///     exactly; the SIMD bit-scan writes the flat indices straight into it — its only row for
        ///     ndim==1, its LAST row for ndim&gt;1 — and the per-dim expand kernel rewrites the rows in
        ///     place (row d = coordinate d of every element). No temporary buffer: the expand reads flat
        ///     index i before it writes column i of any row, and later steps read only flat[i+1..].
        ///     </para>
        ///
        ///     <para>
        ///     Routing:
        ///     <list type="bullet">
        ///       <item>0-d → NumPy's <c>argwhere(atleast_1d(a))[:, :0]</c>: shape <c>(1,0)</c> truthy /
        ///             <c>(0,0)</c> falsy, a view of the promoted result's buffer.</item>
        ///       <item>size == 0 → shape <c>(0, ndim)</c>, the transpose of an empty <c>(ndim, 0)</c> array.</item>
        ///       <item>Contiguous → IL count + IL scan + (ndim==1 ? direct write : in-place per-dim expand).</item>
        ///       <item>Non-contiguous → materialize via <c>ascontiguousarray</c> then
        ///             same path as contig.</item>
        ///     </list>
        ///     </para>
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>The coordinates (int64): a <c>(N, ndim)</c> view of the stacked <c>(ndim, N)</c> array.</returns>
        /// <exception cref="StringComputationNotSupportedException">A non-empty string array: NumPy reads the strings' truth
        ///     values (the next phase of NumSharp's string support); an empty one gives the (0, ndim) int64 result.</exception>
        /// <exception cref="NotSupportedException">The IL kernels cannot be emitted for the dtype.</exception>
        /// <remarks>
        ///     <b>BREAKING (NumPy parity):</b> the result used to be an owning C-contiguous <c>(N, ndim)</c> array. It is
        ///     now F-contiguous for ndim &gt;= 2 (row <c>i</c>'s coordinates are <c>N</c> elements apart in memory), so
        ///     code that reads its raw buffer row by row must go through the strides or take
        ///     <c>np.ascontiguousarray</c> first; element access, slicing and every np.* call read it correctly.
        /// </remarks>
        public override unsafe NDArray Argwhere(NDArray nd)
        {
            // Text: NumPy's argwhere reads the strings' truth values (the next phase; a 0-d input is atleast_1d'd first,
            // so its result shape depends on the value too). An empty input reads none: the bool stand-in gives the
            // (0, ndim) int64 result.
            if (Strings.NDStringUfuncs.ReduceStandIn("argwhere", nd) is { } standIn)
            {
                using (standIn)
                    return Argwhere(standIn);
            }
            // 0-d: NumPy's `argwhere(atleast_1d(a))[:, :0]` — the promoted (count, 1) result with its column sliced
            // away: a (count, 0) VIEW of the promoted result's buffer, keeping its strides (8, 8 for a truthy element).
            if (nd.ndim == 0)
            {
                var promoted = np.atleast_1d(nd);
                var full = ArgwhereContiguousOrMaterialize(promoted);
                promoted.Dispose(); // 1-element view wrapper of the scalar's buffer
                var fs = full.Shape;
                var sliced = new Shape(new long[] { fs.dimensions[0], 0 }, (long[])fs.strides.Clone(), fs.offset, fs.bufferSize);
                var view = new NDArray(full.Storage.Alias(sliced)) { TensorEngine = full.TensorEngine };
                full.Dispose(); // the sliced view holds its own counted reference on the buffer
                return view;
            }

            return ArgwhereContiguousOrMaterialize(nd);
        }

        /// <summary>
        ///     Single dispatch path for all contig and non-contig inputs. Non-contig is
        ///     materialized to a fresh C-contig buffer first so the same IL kernels apply
        ///     to every layout. Flat indices into the contig buffer map back to user-facing
        ///     coords through the shape dims (C-order traversal preserved).
        /// </summary>
        /// <param name="nd">A non-0-d array.</param>
        /// <returns>The <c>(N, ndim)</c> transposed view of the stacked <c>(ndim, N)</c> coordinate array.</returns>
        /// <exception cref="NotSupportedException">The IL kernels cannot be emitted for the dtype.</exception>
        private static unsafe NDArray ArgwhereContiguousOrMaterialize(NDArray nd)
        {
            int ndim = nd.ndim;
            long size = nd.size;

            // Empty input → shape (0, ndim), the transpose of an empty (ndim, 0) stack. Includes shape (0,3), (2,0,4), …
            if (size == 0)
                return ArgwhereTransposedView(new NDArray(NPTypeCode.Int64, new Shape(ndim, 0), false), 0, ndim);

            // Materialize non-contig to C-contig. For contig inputs we read from `nd`
            // directly (no copy); the local `materialized` is only set on the non-contig
            // branch and disposed in `finally` — see the ARC note at the bottom of the
            // method for why the explicit Dispose matters here.
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
                throw new NotSupportedException($"np.argwhere: no IL kernel available for {nd.dtype.type.Name}");

            try
            {
                long count = countKernel(basePtr, size);

                // NumPy's transpose(nonzero(a)): asarray stacks the ndim index arrays into a fresh (ndim, count) C-order
                // array — row d holds coordinate d of every non-zero element — and the result is its transpose.
                var stacked = new NDArray(NPTypeCode.Int64, new Shape(ndim, count), false);
                if (count == 0)
                    return ArgwhereTransposedView(stacked, 0, ndim);

                long* rows = (long*)stacked.Storage.Address;

                // ndim == 1: the (1, count) stack's only row IS the flat index list — scan straight into it.
                if (ndim == 1)
                {
                    flatKernel(basePtr, size, rows);
                    return ArgwhereTransposedView(stacked, count, ndim);
                }

                // ndim > 1: scan into the LAST row, then expand every row in place. The per-dim kernel reads flat[i]
                // before it writes column i of any row and keeps the previous flat index in a local, so overwriting
                // flat[i] (row ndim-1, column i) after the read is safe, and a later step reads only flat[i+1..].
                long* flat = rows + (ndim - 1) * count;
                flatKernel(basePtr, size, flat);

                var expandKernel = DirectILKernelGenerator.GetNonZeroPerDimKernel();
                if (expandKernel == null)
                    throw new NotSupportedException("np.argwhere: per-dim expand IL kernel unavailable");

                // Pre-compute dim strides for the expand kernel: dimStrides[ndim-1] = 1;
                // dimStrides[d] = dimStrides[d+1] * dims[d+1]. Cheap O(ndim) prologue. The row
                // addresses go to the kernel as pointer-sized integers (its long** reads them as pointers).
                Span<long> dimStrides = ndim <= 64 ? stackalloc long[ndim] : new long[ndim];
                Span<nint> outRows = ndim <= 64 ? stackalloc nint[ndim] : new nint[ndim];
                dimStrides[ndim - 1] = 1;
                for (int d = ndim - 2; d >= 0; d--)
                    dimStrides[d] = dimStrides[d + 1] * sourceShape.dimensions[d + 1];
                for (int d = 0; d < ndim; d++)
                    outRows[d] = (nint)(rows + d * count);

                fixed (long* dimsPtr = sourceShape.dimensions)
                fixed (long* dimStridesPtr = dimStrides)
                fixed (nint* outRowsPtr = outRows)
                    expandKernel(flat, count, dimsPtr, dimStridesPtr, ndim, (long**)outRowsPtr);

                return ArgwhereTransposedView(stacked, count, ndim);
            }
            finally
            {
                // ARC: source.Storage.Address is an unmanaged pointer that does NOT
                // keep `materialized` GC-alive. Under repeated argwhere() calls on a
                // non-contig input, the JIT can decide `materialized` is dead after
                // basePtr is computed — then the result-NDArray allocations below
                // trigger GC, the freshly materialized array's UnmanagedStorage gets
                // its refcount finalized to zero, and the buffer behind basePtr is
                // freed mid-IL-scan (the np.nonzero bench harness reproduces this
                // exact pattern as a Release-mode AccessViolationException; argwhere
                // shares the same fix preventively).
                //
                // Explicit Dispose() is the established ARC-release pattern in this
                // codebase (see commits 392529f2, 294d4329) — it forces the JIT to
                // keep `materialized` rooted until the call site here. For the contig
                // fast path materialized is null and Dispose is skipped.
                materialized?.Dispose();
                // The contiguous route reads the CALLER's operand through basePtr; a temporary operand
                // (`np.argwhere(a > 0)`) is referenced only by this parameter, which the JIT may treat as dead
                // once the pointer is taken — keep it reachable until the kernels are done.
                GC.KeepAlive(nd);
            }
        }

        /// <summary>
        ///     NumPy's <c>np.transpose</c> of the stacked <c>(ndim, count)</c> coordinate array: a <c>(count, ndim)</c>
        ///     view with element strides <c>(1, count)</c> — F-contiguous, OWNDATA false.
        /// </summary>
        /// <param name="stacked">The freshly filled <c>(ndim, count)</c> C-order array; its handle is released here (the
        ///     view holds its own counted reference on the buffer).</param>
        /// <param name="count">The number of non-zero elements.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <returns>The transposed view.</returns>
        private static NDArray ArgwhereTransposedView(NDArray stacked, long count, int ndim)
        {
            var transposed = new Shape(new long[] { count, ndim }, new long[] { 1, count }, 0, count * (long)ndim);
            var view = new NDArray(stacked.Storage.Alias(transposed)) { TensorEngine = stacked.TensorEngine };
            stacked.Dispose();
            return view;
        }
    }
}
