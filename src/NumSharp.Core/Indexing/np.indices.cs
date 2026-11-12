using System;

namespace NumSharp
{
    public static partial class np
    {
        // ─────────────────────────────────────────────────────────────────────────────
        //  np.indices — dense form
        //
        //  NumPy's one signature, indices(dimensions, dtype=int, sparse=False), over one
        //  Shape: an int[] (the shortest literal), a long[] (NDArray.shape, the house shape
        //  type, so a caller never down-casts its own shape), a tuple (np.indices((2, 3)))
        //  and a Shape all convert to it, and only the dimension sizes are read. The int[]
        //  and long[] overloads went with API-parity batch 8 (2026-11-12). sparse=True keeps
        //  its own name, np.indices_sparse below: its result is a tuple of arrays.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Return an array representing the indices of a grid — NumPy's dense <c>np.indices(dimensions, dtype=int)</c>.
        ///     For the sparse form (NumPy's <c>sparse=True</c>), see <see cref="indices_sparse(Shape, DType)"/>.
        /// </summary>
        /// <param name="dimensions">The shape of the grid: a tuple (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> (an
        ///     array's <c>shape</c>) or a <see cref="Shape"/>, all converting implicitly. Only the dimension SIZES are read —
        ///     any strides / offset a <see cref="Shape"/> carries are ignored, since <c>indices</c> describes a grid of these
        ///     sizes. A 0-dimensional shape yields NumPy's <c>(0,)</c> result.</param>
        /// <param name="dtype">Element type of the result — any dtype spelling (<see cref="Type"/>, <see cref="NPTypeCode"/>,
        ///     dtype string or <see cref="DType"/>) converts implicitly. Default (null) is <see cref="long"/> /
        ///     <see cref="int64"/> (NumPy's <c>int</c>, the platform's intp).</param>
        /// <returns>Single dense array of shape <c>(len(dimensions), *dimensions)</c>; the d-th sub-array holds the d-th
        ///     coordinate of each output position.</returns>
        /// <exception cref="ValueError">A dimension is negative (<c>negative dimensions are not allowed</c>), or the grid's
        ///     byte count overflows (<c>array is too big; ...</c>): NumPy's allocation of the result, which comes before any
        ///     arange.</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses, even for an empty grid: a text dtype
        ///     (<c>arange() not supported for inputs with DType ...</c>), or bool with a dimension longer than two
        ///     (<c>arange() is only supported for booleans when the result has at most length 2.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        ///     <para>
        ///     Because a scalar converts implicitly to a 1-D <see cref="Shape"/>, <c>np.indices(5)</c> is accepted as the 1-D
        ///     grid <c>(1, 5)</c> — a NumSharp convenience (a <see cref="Shape"/> cannot tell <c>5</c> from <c>(5,)</c>);
        ///     NumPy's <c>np.indices(5)</c> raises (<c>TypeError: 'int' object is not iterable</c>). C# has no 1-element
        ///     tuple, so a 1-D grid is <c>np.indices(new long[] { 5 })</c> or <c>np.indices(5)</c>.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray indices(Shape dimensions, DType dtype = null)
            // A Shape's dimensions are the live long[] of sizes (null for a default(Shape)); the array-object home reads them
            // without writing.
            => NDArray.indices(dimensions.dimensions ?? Array.Empty<long>(), dtype);

        // ─────────────────────────────────────────────────────────────────────────────
        //  np.indices — sparse form (NumPy's sparse=True)
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Sparse counterpart to <see cref="indices(Shape, DType)"/> — returns a tuple of
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
        ///     needed. See <see cref="indices(Shape, DType)"/> for the dense twin.
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
            return NDArray.indices_sparse(dimensions, dtype);
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
