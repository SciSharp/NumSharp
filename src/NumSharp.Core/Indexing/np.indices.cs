using System;
using System.Globalization;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        // ─────────────────────────────────────────────────────────────────────────────
        //  np.indices
        //
        //  NumPy's one signature, indices(dimensions, dtype=int, sparse=False), over one
        //  Shape: an int[] (the shortest literal), a long[] (NDArray.shape, the house shape
        //  type, so a caller never down-casts its own shape), a tuple (np.indices((2, 3)))
        //  and a Shape all convert to it, and only the dimension sizes are read. The int[]
        //  and long[] overloads went with API-parity batch 8 (2026-11-12); sparse=True came
        //  with batch 11 (the owner's decision 3) through IndicesResult, which replaced the
        //  NumSharp-only np.indices_sparse.
        // ─────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     The result of <see cref="indices(Shape, DType, bool)"/> — NumPy's dense grid OR sparse tuple as one type
        ///     (np.unique's <see cref="UniqueResult"/> pattern). Dense: <see cref="values"/> is the grid of shape
        ///     <c>(len(dimensions), *dimensions)</c>; sparse: <see cref="values"/> is null and the result holds one
        ///     broadcast-shaped array per axis.
        /// <para>
        ///     Both read like NumPy's: it indexes and deconstructs per axis (<c>var (i, j) = np.indices((2, 3));</c> works for
        ///     either form, as Python's tuple unpacking iterates a dense grid's first axis), converts implicitly to
        ///     <see cref="NDArray"/><c>[]</c> (the per-axis arrays), and a dense result converts implicitly to the grid's
        ///     <see cref="NDArray"/>.
        /// </para>
        /// </summary>
        /// <remarks>
        ///     <para>API-parity batch 11 (the owner's decision 3): it replaces np.indices' NDArray return and the
        ///     NumSharp-only <c>np.indices_sparse</c>. The NDArray members callers read on a dense grid forward to
        ///     <see cref="values"/> (shape, size, dtype, the typed getters, <c>ToArray&lt;T&gt;</c>, <c>array_equal</c>,
        ///     <c>reshape</c>, <c>astype</c>, <c>flags</c>).</para>
        ///     <para>It is <see cref="IDisposable"/> as the NDArray it replaced was: <c>using var g = np.indices(...);</c>
        ///     releases the grid, or every per-axis array of a sparse result.</para>
        ///     <para>Footguns: converting a SPARSE result to <see cref="NDArray"/> raises (NumPy's tuple of differently
        ///     shaped arrays has no array form); an extension method of <see cref="NDArray"/> does not see through the
        ///     implicit conversion - call it on <see cref="values"/>; it is not enumerable (convert it to
        ///     <see cref="NDArray"/><c>[]</c> for <c>foreach</c> or LINQ over the per-axis arrays).</para>
        /// </remarks>
        public readonly struct IndicesResult : INDArrayCarrier, IDisposable
        {
            // The sparse form's per-axis arrays (null for the dense form).
            private readonly NDArray[] _sparse;

            /// <summary>The dense grid of shape <c>(len(dimensions), *dimensions)</c>, or <c>null</c> for a sparse
            ///     result.</summary>
            public NDArray values { get; }

            /// <summary>Wraps a dense grid.</summary>
            /// <param name="grid">The grid.</param>
            internal IndicesResult(NDArray grid)
            {
                values = grid;
                _sparse = null;
            }

            /// <summary>Wraps a sparse result's per-axis arrays.</summary>
            /// <param name="sparse">One array per axis (an empty array for no dimension).</param>
            internal IndicesResult(NDArray[] sparse)
            {
                values = null;
                _sparse = sparse;
            }

            /// <summary>Whether this is NumPy's <c>sparse=True</c> tuple (no dense grid).</summary>
            public bool sparse => _sparse is not null;

            /// <summary>The number of per-axis arrays: <c>len(dimensions)</c> (a dense grid's first extent).</summary>
            public int Length => _sparse is not null ? _sparse.Length : (int)values.Shape.dimensions[0];

            /// <summary>The <paramref name="index"/>-th axis' coordinates: a dense grid's <c>grid[index]</c> (a view) or the
            ///     sparse tuple's <c>index</c>-th array.</summary>
            /// <param name="index">The axis, 0-based (a negative index counts from the end in the dense form).</param>
            /// <returns>The coordinates of that axis.</returns>
            /// <exception cref="IndexOutOfRangeException">The index is out of range (sparse).</exception>
            /// <exception cref="IndexError">The index is out of range (dense, NumPy's indexing text).</exception>
            public NDArray this[int index] => _sparse is not null ? _sparse[index] : values[index];

            /// <summary>The per-axis arrays as an <see cref="NDArray"/><c>[]</c>: the sparse tuple, or a dense grid's
            ///     first-axis views (NumPy's <c>tuple(grid)</c>).</summary>
            /// <returns>A new array of the per-axis arrays.</returns>
            public NDArray[] ToArray()
            {
                if (_sparse is not null)
                    return (NDArray[])_sparse.Clone();
                var parts = new NDArray[Length];
                for (int i = 0; i < parts.Length; i++)
                    parts[i] = values[i];
                return parts;
            }

            // ---- The dense grid's NDArray surface (delegates to values) ----

            /// <summary>The dense grid's shape (a sparse result has none: see <see cref="Dense"/>).</summary>
            public long[] shape => Dense.shape;

            /// <summary>The dense grid's <see cref="NumSharp.Shape"/>.</summary>
            public Shape Shape => Dense.Shape;

            /// <summary>The dense grid's rank: <c>1 + len(dimensions)</c>.</summary>
            public int ndim => Dense.ndim;

            /// <summary>The dense grid's element count.</summary>
            public long size => Dense.size;

            /// <summary>The dense grid's dtype.</summary>
            public DType dtype => _sparse is not null && _sparse.Length > 0 ? _sparse[0].dtype : Dense.dtype;

            /// <summary>The dense grid's storage lane.</summary>
            public NPTypeCode typecode => dtype.typecode;

            /// <summary>The dense grid's flags.</summary>
            public NDArrayFlags flags => Dense.flags;

            /// <summary>True if the dense grid equals <paramref name="rhs"/> element-wise (same shape).</summary>
            /// <param name="rhs">The array to compare with.</param>
            /// <returns>Whether the shapes and elements are equal.</returns>
            public bool array_equal(NDArray rhs) => Dense.array_equal(rhs);

            /// <summary>The dense grid's elements as a new <typeparamref name="T"/>[] in C order.</summary>
            /// <typeparam name="T">The element type, as <see cref="NDArray.ToArray{T}"/> reads it.</typeparam>
            /// <returns>A new managed array.</returns>
            public T[] ToArray<T>() where T : unmanaged => Dense.ToArray<T>();

            /// <summary>The dense grid's element at the given coordinates, as a long.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element.</returns>
            public long GetInt64(params long[] indices) => Dense.GetInt64(indices);

            /// <summary>The dense grid's element at the given coordinates, as an int.</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element.</returns>
            public int GetInt32(params long[] indices) => Dense.GetInt32(indices);

            /// <summary>The dense grid's element at the given coordinates, as a double (a <c>dtype=float</c> grid's own
            ///     type; any other dtype's element is converted).</summary>
            /// <param name="indices">The coordinates.</param>
            /// <returns>The element.</returns>
            public double GetDouble(params long[] indices) => Dense.GetDouble(indices);

            /// <summary>The dense grid with a new shape (<see cref="NDArray.reshape(long[])"/>).</summary>
            /// <param name="shape">The new dimensions.</param>
            /// <returns>The reshaped grid.</returns>
            /// <exception cref="IncorrectShapeException">The size does not match.</exception>
            public NDArray reshape(params long[] shape) => Dense.reshape(shape);

            /// <summary>The dense grid cast to <paramref name="dtype"/>
            ///     (<see cref="NDArray.astype(DType, char, string, bool)"/>); a grid built in its final dtype
            ///     (<c>np.indices(dims, dtype)</c>) skips the extra copy.</summary>
            /// <param name="dtype">The target dtype.</param>
            /// <param name="order">The memory order of a copy ('K' keeps the grid's).</param>
            /// <param name="casting">The casting rule ('unsafe' by default, as NumPy's astype).</param>
            /// <param name="copy">Whether to copy when no cast is needed.</param>
            /// <returns>The cast grid.</returns>
            /// <exception cref="InvalidCastException">The result is sparse, or the cast is refused under
            ///     <paramref name="casting"/>.</exception>
            public NDArray astype(DType dtype, char order = 'K', string casting = "unsafe", bool copy = true)
                => Dense.astype(dtype, order, casting, copy);

            /// <summary>The dense grid, or NumPy's refusal of a sparse tuple where an array is expected.</summary>
            /// <exception cref="InvalidCastException">The result is sparse.</exception>
            private NDArray Dense => values ?? throw new InvalidCastException(
                "np.indices(..., sparse=True) is a tuple of arrays of different shapes, not an array: index it, deconstruct it or convert it to NDArray[].");

            /// <summary>The dense grid (NumPy's single-array return).</summary>
            /// <param name="result">The result.</param>
            /// <returns>The grid.</returns>
            /// <exception cref="InvalidCastException">The result is sparse (a tuple of differently shaped arrays).</exception>
            public static implicit operator NDArray(IndicesResult result) => result.Dense;

            /// <summary>The per-axis arrays (<see cref="ToArray()"/>).</summary>
            /// <param name="result">The result.</param>
            /// <returns>A new array of the per-axis arrays.</returns>
            public static implicit operator NDArray[](IndicesResult result) => result.ToArray();

            /// <summary>Deconstructs a 2-D grid's coordinates: <c>var (i, j) = np.indices((2, 3))</c>.</summary>
            /// <param name="first">Axis 0's coordinates.</param>
            /// <param name="second">Axis 1's coordinates.</param>
            /// <exception cref="InvalidOperationException">The grid is not 2-D (Python's unpacking ValueError text).</exception>
            public void Deconstruct(out NDArray first, out NDArray second)
            {
                EnsureArity(2);
                first = this[0];
                second = this[1];
            }

            /// <summary>Deconstructs a 3-D grid's coordinates.</summary>
            /// <param name="first">Axis 0's coordinates.</param>
            /// <param name="second">Axis 1's coordinates.</param>
            /// <param name="third">Axis 2's coordinates.</param>
            /// <exception cref="InvalidOperationException">The grid is not 3-D.</exception>
            public void Deconstruct(out NDArray first, out NDArray second, out NDArray third)
            {
                EnsureArity(3);
                first = this[0];
                second = this[1];
                third = this[2];
            }

            /// <summary>Deconstructs a 4-D grid's coordinates.</summary>
            /// <param name="first">Axis 0's coordinates.</param>
            /// <param name="second">Axis 1's coordinates.</param>
            /// <param name="third">Axis 2's coordinates.</param>
            /// <param name="fourth">Axis 3's coordinates.</param>
            /// <exception cref="InvalidOperationException">The grid is not 4-D.</exception>
            public void Deconstruct(out NDArray first, out NDArray second, out NDArray third, out NDArray fourth)
            {
                EnsureArity(4);
                first = this[0];
                second = this[1];
                third = this[2];
                fourth = this[3];
            }

            /// <summary>Refuses a deconstruction into another number of arrays than the grid has axes.</summary>
            /// <param name="n">The deconstruction's arity.</param>
            /// <exception cref="InvalidOperationException">Python's <c>too many values to unpack (expected n)</c> /
            ///     <c>not enough values to unpack (expected n, got m)</c>.</exception>
            private void EnsureArity(int n)
            {
                int length = Length;
                if (length != n)
                    throw new InvalidOperationException(length > n
                        ? string.Create(CultureInfo.InvariantCulture, $"too many values to unpack (expected {n})")
                        : string.Create(CultureInfo.InvariantCulture, $"not enough values to unpack (expected {n}, got {length})"));
            }

            /// <summary>Releases the dense grid, or every per-axis array of a sparse result (the NDArray it replaced was
            ///     disposable the same way). Footgun: an array taken from a SPARSE result (the indexer, a deconstruction,
            ///     <see cref="ToArray()"/>) is the same object, so it is released too; a dense grid's per-axis arrays are
            ///     views that hold their own reference on the grid's buffer and stay readable until released
            ///     themselves.</summary>
            public void Dispose()
            {
                values?.Dispose();
                if (_sparse is not null)
                    foreach (var part in _sparse)
                        part?.Dispose();
            }

            // Yields the grid or every per-axis array: a [NDScoped] caller returning this result keeps them alive.
            void INDArrayCarrier.YieldTo(NDScope scope)
            {
                scope.Returns(values);
                if (_sparse is not null)
                    foreach (var part in _sparse)
                        scope.Returns(part);
            }
        }

        /// <summary>
        ///     Return the indices of a grid — NumPy's <c>np.indices(dimensions, dtype=int, sparse=False)</c>: the dense grid
        ///     of shape <c>(len(dimensions), *dimensions)</c> whose <c>d</c>-th sub-array holds the <c>d</c>-th coordinate
        ///     of each position, or with <paramref name="sparse"/> one broadcast-shaped array per axis, axis <c>d</c>'s of
        ///     shape <c>(1, …, 1, dimensions[d], 1, …, 1)</c>.
        /// </summary>
        /// <param name="dimensions">The shape of the grid: a tuple (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> (an
        ///     array's <c>shape</c>), a C# collection expression or a <see cref="Shape"/>, all converting implicitly. Only the
        ///     dimension SIZES are read — any strides / offset a <see cref="Shape"/> carries are ignored. A 0-dimensional
        ///     shape yields NumPy's <c>(0,)</c> dense result, or an empty sparse tuple.</param>
        /// <param name="dtype">Element type of the result — any dtype spelling (<see cref="Type"/>, <see cref="NPTypeCode"/>,
        ///     dtype string or <see cref="DType"/>) converts implicitly. Default (null) is <see cref="long"/> /
        ///     <see cref="int64"/> (NumPy's <c>int</c>, the platform's intp).</param>
        /// <param name="sparse">NumPy's <c>sparse</c>: return the per-axis arrays instead of the dense grid (each a VIEW of
        ///     its arange, <c>flags.owndata</c> False as NumPy's reshape result is).</param>
        /// <returns>An <see cref="IndicesResult"/>: the dense grid (it converts implicitly to <see cref="NDArray"/>) or the
        ///     sparse tuple; either deconstructs and indexes per axis.</returns>
        /// <exception cref="ValueError">Dense: a dimension is negative (<c>negative dimensions are not allowed</c>), or the
        ///     grid's byte count overflows (<c>array is too big; ...</c>) - NumPy's allocation of the result, which comes
        ///     before any arange. NumPy's sparse grid allocates nothing but its aranges, so a negative dimension is not
        ///     refused there: its arange is empty and the reshape reads the negative entry as the unknown dimension
        ///     (<c>(2, -3)</c> gives a <c>(2, 1)</c> and a <c>(1, 0)</c> array).</exception>
        /// <exception cref="TypeError">An arange of <paramref name="dtype"/> refuses, even for an empty grid: a text dtype
        ///     (<c>arange() not supported for inputs with DType ...</c>), or bool with a dimension longer than two
        ///     (<c>arange() is only supported for booleans when the result has at most length 2.</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.indices.html
        ///     <para>
        ///     Because a scalar converts implicitly to a 1-D <see cref="Shape"/>, <c>np.indices(5)</c> is accepted as the 1-D
        ///     grid <c>(1, 5)</c> — a NumSharp convenience (a <see cref="Shape"/> cannot tell <c>5</c> from <c>(5,)</c>);
        ///     NumPy's <c>np.indices(5)</c> raises (<c>TypeError: 'int' object is not iterable</c>). C# has no 1-element
        ///     tuple, so a 1-D grid is <c>np.indices([5])</c>, <c>np.indices(new long[] { 5 })</c> or <c>np.indices(5)</c>.
        ///     </para>
        ///     <para>API-parity batch 11 (the owner's decision 3): NumPy's <c>sparse</c> and the <see cref="IndicesResult"/>
        ///     return; the NumSharp-only <c>np.indices_sparse</c> (three overloads) went.</para>
        /// </remarks>
        [NDScoped]
        public static IndicesResult indices(Shape dimensions, DType dtype = null, bool sparse = false)
        {
            // A Shape's dimensions are the live long[] of sizes (null for a default(Shape)); the array-object home reads them
            // without writing.
            long[] dims = dimensions.dimensions ?? Array.Empty<long>();
            return sparse
                ? new IndicesResult(NDArray.indices_sparse(dims, dtype))
                : new IndicesResult(NDArray.indices(dims, dtype));
        }
    }
}
