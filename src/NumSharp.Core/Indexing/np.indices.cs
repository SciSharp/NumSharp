using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        // ─────────────────────────────────────────────────────────────────────────────
        //  np.indices — dense form
        //
        //  Three dimension spellings are offered so the API is friendly to the whole
        //  house shape vocabulary, exactly as np.zeros/np.ones/np.empty do: an int[]
        //  (back-compat + the shortest literal), a long[] (THE house shape type —
        //  NDArray.shape, Shape.Dimensions and every computed size are long, so a
        //  caller must not down-cast their own array's shape to reach np.indices),
        //  and a Shape (accepts an NDArray's Shape directly, and lets a C# value tuple
        //  read like Python: np.indices((2, 3))). All three funnel into the long[]
        //  core so a grid dimension is never silently truncated to 32 bits.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Return an array representing the indices of a grid — dense form
        ///     (<c>int[]</c> spelling). For the sparse form, see <see cref="indices_sparse(int[], DType)"/>.
        /// </summary>
        /// <param name="dimensions">Shape of the grid, as a 32-bit array. Widened to the
        ///     long-native core; prefer the <see cref="indices(long[], DType)"/> overload
        ///     when you already hold a <see cref="long"/> shape (e.g. <c>arr.shape</c>).</param>
        /// <param name="dtype">Element type of the result — any dtype spelling
        ///     (<see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/>)
        ///     converts implicitly. Default (null) is <see cref="long"/> / <see cref="int64"/>
        ///     (NumPy's <c>intp</c>).</param>
        /// <returns>Single dense array of shape <c>(len(dimensions), *dimensions)</c>; the
        ///     d-th sub-array holds the d-th coordinate of each output position.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dimensions"/> is null.</exception>
        /// <exception cref="ArgumentException">A dimension is negative, or the grid size
        ///     overflows the maximum representable element count.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.indices.html</remarks>
        public static NDArray indices(int[] dimensions, DType dtype = null)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));
            // Widen to the long-native core — int[] does not convert to long[] by array
            // covariance, so this overload exists only to carry the 32-bit spelling in.
            return indices(WidenDimensions(dimensions), dtype);
        }

        /// <summary>
        ///     Return an array representing the indices of a grid — dense form, taking the
        ///     house <see cref="long"/> shape type. This is the overload to prefer: an
        ///     <see cref="NDArray"/>'s own <c>shape</c> is <c>long[]</c>, so it binds here
        ///     with no down-cast (the reason this overload exists — the former <c>int[]</c>-only
        ///     surface forced callers to truncate a long shape to reach <c>np.indices</c>).
        /// </summary>
        /// <param name="dimensions">Shape of the grid. An empty array yields NumPy's
        ///     <c>(0,)</c> result. Elements are not mutated.</param>
        /// <param name="dtype">Element type of the result — any dtype spelling converts
        ///     implicitly. Default (null) is <see cref="long"/> / <see cref="int64"/>.</param>
        /// <returns>Single dense array of shape <c>(len(dimensions), *dimensions)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dimensions"/> is null.</exception>
        /// <exception cref="ArgumentException">A dimension is negative, or the grid size
        ///     overflows the maximum representable element count.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.indices.html</remarks>
        [NDScoped]
        public static NDArray indices(long[] dimensions, DType dtype = null)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));
            var typeCode = dtype?.GetTypeCode() ?? NPTypeCode.Int64;

            // Special case: empty dimensions tuple → numpy returns shape (0,).
            if (dimensions.Length == 0)
                return new NDArray(typeCode, new Shape(0), false);

            return BuildDenseIndices(dimensions, typeCode);
        }

        /// <summary>
        ///     Return an array representing the indices of a grid — dense form, taking a
        ///     <see cref="Shape"/>. Accepts an <see cref="NDArray.Shape"/> directly and lets a
        ///     C# value tuple read like Python (<c>np.indices((2, 3))</c>, via <see cref="Shape"/>'s
        ///     implicit tuple conversions). Only the dimension SIZES are used — any strides/offset
        ///     the shape carries are ignored, since <c>indices</c> describes a grid of these sizes.
        /// </summary>
        /// <param name="dimensions">Grid shape. A 0-dimensional shape yields NumPy's <c>(0,)</c> result.</param>
        /// <param name="dtype">Element type of the result — any dtype spelling converts
        ///     implicitly. Default (null) is <see cref="long"/> / <see cref="int64"/>.</param>
        /// <returns>Single dense array of shape <c>(ndim, *dimensions)</c>.</returns>
        /// <exception cref="ArgumentException">A dimension is negative, or the grid size
        ///     overflows the maximum representable element count.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        ///     <para>
        ///     Because a scalar converts implicitly to a 1-D <see cref="Shape"/>, <c>np.indices(5)</c>
        ///     is accepted here as the 1-D grid <c>(1, 5)</c> — a NumSharp convenience; NumPy's
        ///     <c>np.indices(5)</c> raises (an int is not a sequence). Spell a 1-D grid as
        ///     <c>np.indices((5,))</c> or <c>np.indices(new long[]{5})</c> to stay verbatim.
        ///     </para>
        /// </remarks>
        public static NDArray indices(Shape dimensions, DType dtype = null)
        {
            // A Shape's Dimensions is the live long[] of sizes (null for a default(Shape)).
            return indices(dimensions.Dimensions ?? Array.Empty<long>(), dtype);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  np.indices — sparse form (NumPy's sparse=True)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Sparse counterpart to <see cref="indices(int[], DType)"/> — returns a tuple of
        ///     broadcast-shaped arrays where each axis-d array has shape
        ///     <c>(1, …, 1, dimensions[d], 1, …, 1)</c>. Equivalent to NumPy's
        ///     <c>np.indices(dimensions, sparse=True)</c>. <c>int[]</c> spelling; widened to the
        ///     long-native core.
        /// </summary>
        /// <param name="dimensions">Shape of the grid. Empty → an empty tuple.</param>
        /// <param name="dtype">Element type of each array — any dtype spelling converts
        ///     implicitly. Default (null) is <see cref="long"/> / <see cref="int64"/>.</param>
        /// <returns>One 1-along-axis-d array per dimension.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dimensions"/> is null.</exception>
        /// <exception cref="ArgumentException">A dimension is negative.</exception>
        /// <remarks>
        ///     NumPy's <c>sparse=True</c> mode is exposed as a separate method because C# can't
        ///     change return type based on a parameter the way Python's dynamic typing does, so
        ///     splitting the API is clearer than throwing from a shared signature.
        /// </remarks>
        public static NDArray[] indices_sparse(int[] dimensions, DType dtype = null)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));
            return indices_sparse(WidenDimensions(dimensions), dtype);
        }

        /// <summary>
        ///     Sparse grid indices taking the house <see cref="long"/> shape type — the overload to
        ///     prefer when you hold a <c>long[]</c> shape (e.g. <c>arr.shape</c>), so no down-cast is
        ///     needed. See <see cref="indices(long[], DType)"/> for the dense twin.
        /// </summary>
        /// <param name="dimensions">Shape of the grid. Empty → an empty tuple.</param>
        /// <param name="dtype">Element type of each array — any dtype spelling converts
        ///     implicitly. Default (null) is <see cref="long"/> / <see cref="int64"/>.</param>
        /// <returns>One <c>(1, …, 1, dimensions[d], 1, …, 1)</c> array per dimension.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dimensions"/> is null.</exception>
        /// <exception cref="ArgumentException">A dimension is negative.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.indices.html</remarks>
        [NDScoped]
        public static NDArray[] indices_sparse(long[] dimensions, DType dtype = null)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));

            if (dimensions.Length == 0)
                return Array.Empty<NDArray>();

            return BuildSparseIndicesAsArray(dimensions, dtype?.GetTypeCode() ?? NPTypeCode.Int64);
        }

        /// <summary>
        ///     Sparse grid indices taking a <see cref="Shape"/> — accepts an
        ///     <see cref="NDArray.Shape"/> or a Python-style tuple (<c>np.indices_sparse((2, 3))</c>).
        ///     Only the dimension SIZES are read. See <see cref="indices(Shape, DType)"/> for the
        ///     scalar-convenience note.
        /// </summary>
        /// <param name="dimensions">Grid shape. A 0-dimensional shape → an empty tuple.</param>
        /// <param name="dtype">Element type of each array — any dtype spelling converts
        ///     implicitly. Default (null) is <see cref="long"/> / <see cref="int64"/>.</param>
        /// <returns>One array per dimension.</returns>
        /// <exception cref="ArgumentException">A dimension is negative.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.indices.html</remarks>
        public static NDArray[] indices_sparse(Shape dimensions, DType dtype = null)
        {
            return indices_sparse(dimensions.Dimensions ?? Array.Empty<long>(), dtype);
        }

        // ─────────────────────────────────────────────────────────────────────────────
        //  Implementation
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Widens a 32-bit dimension list to the long-native shape type. Kept private —
        ///     int[] is only a convenience spelling on the public surface.
        /// </summary>
        /// <param name="dimensions">The 32-bit dimensions (already null-checked by the caller).</param>
        /// <returns>A fresh <c>long[]</c> holding the same values.</returns>
        private static long[] WidenDimensions(int[] dimensions)
        {
            var widened = new long[dimensions.Length];
            for (int i = 0; i < dimensions.Length; i++)
                widened[i] = dimensions[i];
            return widened;
        }

        /// <summary>
        ///     Fills the dense <c>(ndim, *dims)</c> coordinate grid. Allocates the buffer as
        ///     Int64 (the fill kernel's native dtype) and casts to <paramref name="dtype"/> only
        ///     when it differs, so the common intp request pays no copy.
        /// </summary>
        /// <param name="dimensions">Grid sizes — length ≥ 1 (the empty case is handled by the
        ///     public overload before reaching here). Read-only; never mutated.</param>
        /// <param name="dtype">Result element type.</param>
        /// <returns>The dense coordinate grid.</returns>
        /// <exception cref="ArgumentException">A dimension is negative, or the running product of
        ///     sizes overflows <see cref="long"/> (the grid can't be represented).</exception>
        /// <exception cref="NotSupportedException">The IL fill kernel is unavailable (e.g. AOT
        ///     with dynamic codegen disabled).</exception>
        private static unsafe NDArray BuildDenseIndices(long[] dimensions, NPTypeCode dtype)
        {
            int ndim = dimensions.Length;

            // Compute prod = total elements per slab and validate non-negative dims. The product
            // is long-native, so a dimension that only fits in 64 bits is honoured rather than
            // truncated — and its overflow is caught (a wrapped byte/element count reads as a
            // small, silently-wrong allocation otherwise).
            long prod = 1;
            for (int d = 0; d < ndim; d++)
            {
                long dim = dimensions[d];
                if (dim < 0)
                    throw new ArgumentException(
                        $"negative dimensions are not allowed (got dim[{d}] = {dim}).",
                        nameof(dimensions));
                long next = unchecked(prod * dim);
                // A non-zero dim whose multiply doesn't divide back out has wrapped int64.
                if (dim != 0 && next / dim != prod)
                    throw new ArgumentException(
                        "invalid dimensions: array size larger than the maximum possible size.",
                        nameof(dimensions));
                prod = next;
            }

            // Result shape: (ndim, *dimensions).
            var resultShape = new long[ndim + 1];
            resultShape[0] = ndim;
            for (int d = 0; d < ndim; d++) resultShape[d + 1] = dimensions[d];

            // Allocate as Int64 first (the kernel's native dtype). If the caller asked for a
            // different dtype, astype afterwards — the dense-fill IL kernel only emits long stores.
            var int64Result = new NDArray(NPTypeCode.Int64, new Shape(resultShape), false);

            // Fast-exit for shapes with size 0 — the buffer is already zero-allocated and no
            // slab fill is needed.
            if (prod == 0)
                return dtype == NPTypeCode.Int64 ? int64Result : int64Result.astype(dtype);

            // Compute dimStrides: dimStrides[ndim-1] = 1; dimStrides[d] = dimStrides[d+1] * dims[d+1].
            Span<long> dimStrides = stackalloc long[ndim];
            dimStrides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                dimStrides[d] = dimStrides[d + 1] * dimensions[d + 1];

            var kernel = DirectILKernelGenerator.GetIndicesKernel();
            if (kernel == null)
                throw new NotSupportedException("np.indices: IL kernel unavailable");

            long* resPtr = (long*)int64Result.Storage.Address;
            // `dimensions` is already long[], so the kernel's dims pointer pins it directly —
            // no per-call copy into a scratch span is needed (the int[] path widened once, up front).
            fixed (long* dsPtr = dimStrides)
            fixed (long* dPtr = dimensions)
            {
                kernel(resPtr, dsPtr, dPtr, ndim, prod);
            }

            return dtype == NPTypeCode.Int64 ? int64Result : int64Result.astype(dtype);
        }

        /// <summary>
        ///     Builds the sparse form: one <c>(1, …, 1, dim, 1, …, 1)</c> arange per axis. Each
        ///     axis is an <c>arange(0, dim)</c> reshaped to broadcast along its own dimension.
        /// </summary>
        /// <param name="dimensions">Grid sizes — length ≥ 1 (empty handled by the public overload).
        ///     Read-only; never mutated.</param>
        /// <param name="dtype">Result element type for each array.</param>
        /// <returns>One broadcast-shaped array per dimension.</returns>
        /// <exception cref="ArgumentException">A dimension is negative.</exception>
        private static NDArray[] BuildSparseIndicesAsArray(long[] dimensions, NPTypeCode dtype)
        {
            int ndim = dimensions.Length;
            var result = new NDArray[ndim];

            for (int d = 0; d < ndim; d++)
            {
                long dim = dimensions[d];
                if (dim < 0)
                    throw new ArgumentException(
                        $"negative dimensions are not allowed (got dim[{d}] = {dim}).",
                        nameof(dimensions));

                // axis-d array has shape (1, ..., 1, dim, 1, ..., 1) with `dim` along axis d.
                var axisShape = new long[ndim];
                for (int k = 0; k < ndim; k++) axisShape[k] = (k == d) ? dim : 1;

                // Build via arange then reshape. arange(0L, dim) picks the long overload, so a
                // 64-bit dimension is honoured rather than truncated.
                var arr = np.arange(0L, dim).reshape(axisShape);
                if (arr.GetTypeCode != dtype)
                    arr = arr.astype(dtype);
                result[d] = arr;
            }

            return result;
        }
    }
}
