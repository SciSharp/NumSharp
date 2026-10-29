using System;

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
        /// <exception cref="ValueError">A dimension is negative (<c>negative dimensions are not allowed</c>), or the grid's
        ///     byte count overflows (<c>array is too big; ...</c>): NumPy's allocation of the result, which comes before any
        ///     arange.</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses, even for an empty grid: a text dtype
        ///     (<c>arange() not supported for inputs with DType ...</c>), or bool with a dimension longer than two
        ///     (<c>arange() is only supported for booleans when the result has at most length 2.</c>).</exception>
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
        /// <exception cref="ValueError">A dimension is negative (<c>negative dimensions are not allowed</c>), or the grid's
        ///     byte count overflows (<c>array is too big; ...</c>): NumPy's allocation of the result, which comes before any
        ///     arange.</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses, even for an empty grid: a text dtype
        ///     (<c>arange() not supported for inputs with DType ...</c>), or bool with a dimension longer than two
        ///     (<c>arange() is only supported for booleans when the result has at most length 2.</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.indices.html</remarks>
        [NDScoped]
        public static NDArray indices(long[] dimensions, DType dtype = null)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));
            return NDArray.Indices(dimensions, dtype);
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
        /// <exception cref="ValueError">A dimension is negative (<c>negative dimensions are not allowed</c>), or the grid's
        ///     byte count overflows (<c>array is too big; ...</c>): NumPy's allocation of the result, which comes before any
        ///     arange.</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses, even for an empty grid: a text dtype
        ///     (<c>arange() not supported for inputs with DType ...</c>), or bool with a dimension longer than two
        ///     (<c>arange() is only supported for booleans when the result has at most length 2.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        ///     <para>
        ///     Because a scalar converts implicitly to a 1-D <see cref="Shape"/>, <c>np.indices(5)</c>
        ///     is accepted here as the 1-D grid <c>(1, 5)</c> — a NumSharp convenience; NumPy's
        ///     <c>np.indices(5)</c> raises (an int is not a sequence). C# has no 1-element tuple,
        ///     so spell a 1-D grid as <c>np.indices([5])</c> or <c>np.indices(new long[]{5})</c>.
        ///     </para>
        /// </remarks>
        public static NDArray indices(Shape dimensions, DType dtype = null)
        {
            // A Shape's Dimensions is the live long[] of sizes (null for a default(Shape)).
            return indices(dimensions.dimensions ?? Array.Empty<long>(), dtype);
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
        /// <returns>One 1-along-axis-d array per dimension, each a view of its arange (<c>flags.owndata</c> False, as
        ///     NumPy's reshape result is); a negative dimension's axis is 0 long.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dimensions"/> is null.</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses: a text dtype (<c>arange() not supported
        ///     for inputs with DType ...</c>), whatever the length, or bool with a dimension longer than two.</exception>
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
        /// <returns>One <c>(1, …, 1, dimensions[d], 1, …, 1)</c> array per dimension, each a view of its arange
        ///     (<c>flags.owndata</c> False).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dimensions"/> is null.</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses: a text dtype (<c>arange() not supported
        ///     for inputs with DType ...</c>), whatever the length, or bool with a dimension longer than two.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        ///     <para>
        ///     NumPy's sparse grid allocates nothing but its aranges, so a negative dimension is not refused: its arange is
        ///     empty and the reshape reads the negative entry as the unknown dimension (<c>(2, -3)</c> gives a <c>(2, 1)</c>
        ///     and a <c>(1, 0)</c> array), where the dense form raises.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray[] indices_sparse(long[] dimensions, DType dtype = null)
        {
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));
            return NDArray.IndicesSparse(dimensions, dtype);
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
        /// <returns>One array per dimension, each a view of its arange; a negative dimension's axis is 0 long.</returns>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses: a text dtype (<c>arange() not supported
        ///     for inputs with DType ...</c>), whatever the length, or bool with a dimension longer than two.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.indices.html</remarks>
        public static NDArray[] indices_sparse(Shape dimensions, DType dtype = null)
        {
            return indices_sparse(dimensions.dimensions ?? Array.Empty<long>(), dtype);
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
    }
}
