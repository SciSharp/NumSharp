using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The Python functions of NumPy's lib/_twodim_base_impl.py over the array object (Appendix D's NDArray.TwoDimBase.cs):
    // np.eye, `m = zeros((N, M), dtype=dtype, order=order); m[:M-k].flat[i::M+1] = 1`, and np.diag / np.diagflat, whose
    // 1-D branch is `res = zeros((n, n), v.dtype); res[:n-k].flat[i::n+1] = v`. They are the array object's own members
    // (the np-layer dtype plan's D1: functions that create arrays), and np.eye, np.identity, np.diag and np.diagflat are
    // facades over them.
    //
    // The file sits beside its facades (Creation/) rather than under Backends/ because its statements still call np.*
    // functions (np.zeros, np.diagonal, np.copyto), which ratchet R2 forbids under Backends/ (plan C3).
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>np.eye</c> (<c>_twodim_base_impl.py</c>), the body of
        ///     <see cref="np.eye(int, int?, int, DType, char, string)"/> once the device was checked: an (N, M) array of zeros
        ///     with the <paramref name="k"/>-th diagonal set to one, in C or F order.
        /// </summary>
        /// <param name="N">The number of rows.</param>
        /// <param name="M">The number of columns, or null for <paramref name="N"/>.</param>
        /// <param name="k">The diagonal: 0 the main one, a positive value an upper one, a negative value a lower one. A
        ///     diagonal outside the matrix leaves it all zeros.</param>
        /// <param name="dtype">The dtype, or null for float64.</param>
        /// <param name="order"><c>'C'</c> or <c>'F'</c>, in either case.</param>
        /// <returns>A new (N, M) array: one on the diagonal, zero elsewhere.</returns>
        /// <exception cref="ValueError">An order other than C or F (<c>'A'</c> and <c>'K'</c>: <c>only 'C' or 'F' order is
        ///     permitted</c>, as NumPy's zeros refuses them), checked first; <paramref name="N"/> or the column count is
        ///     negative (<c>negative dimensions are not allowed</c>, the allocation's text); or a dtype whose <c>setitem</c>
        ///     refuses the Python int 1 (a <c>StringDType(coerce=False)</c>), raised after the zeros are allocated and before
        ///     any diagonal element is written.</exception>
        /// <remarks>An F-order result is the C result copied (<c>m.copy('F')</c>), so it costs a second allocation.</remarks>
        internal static unsafe NDArray Eye(int N, int? M, int k, DType dtype, char order)
        {
            int cols = M ?? N;
            // np.zeros' own order of checks once np.eye checked the order's letter and the device: A and K are refused
            // (only C and F lay out a new array), then a negative dimension by the allocation (AllocationGuard's NumPy text,
            // "negative dimensions are not allowed", for N or M alike).
            char physical = OrderResolver.Resolve(order);

            var resolvedType = dtype ?? DType.Double;
            var m = np.zeros(Shape.Matrix(N, cols), resolvedType);
            if (N == 0 || cols == 0)
                return physical == 'F' ? m.copy('F') : m;

            // Diagonal element count: rows where 0 <= i < N and 0 <= i+k < cols
            int rowStart = Math.Max(0, -k);
            int rowEnd = Math.Min(N, cols - k);
            if (rowEnd <= rowStart)
                return physical == 'F' ? m.copy('F') : m;

            // NumPy's `m[:M-k].flat[i::M+1] = 1`: the flat iterator's assignment converts the Python int 1 ONCE through the
            // class's setitem (PyArray_Pack: True, 1, 1.0, 1+0j, '1' for text; a StringDType(coerce=False) refuses it, as
            // NumPy's does) and copies that element into the other diagonal slots. Element (i, i+k) of the fresh C-ordered
            // (N, cols) array is flat element i*cols + (i+k).
            var descr = m.dtype;
            var meta = descr.Meta;
            long itemsize = descr.itemsize;
            byte* data = m.Storage.Address;
            IArraySlice buffer = m.Storage.InternalArray;
            byte* first = data + ((long)rowStart * cols + (rowStart + k)) * itemsize;
            try
            {
                meta.setitem(descr, 1L, first, buffer);
                // A class whose items hold references (PyDataType_REFCHK: StringDType, whose cell points into the array's
                // allocator) makes its copies through the class too, as NumPy's copy loop for it allocates a string per
                // element: each slot is stored by setitem. Every other class copies the converted element's bytes.
                bool references = meta.HasReferences;
                for (int i = rowStart + 1; i < rowEnd; i++)
                {
                    byte* slot = data + ((long)i * cols + (i + k)) * itemsize;
                    if (references)
                        meta.setitem(descr, 1L, slot, buffer);
                    else
                        Buffer.MemoryCopy(first, slot, itemsize, itemsize);
                }
            }
            catch
            {
                // The class refused the one (a StringDType(coerce=False)): the zeros never reach the caller, as NumPy drops
                // its m when the assignment raises.
                m.Dispose();
                throw;
            }

            return physical == 'F' ? m.copy('F') : m;
        }

        /// <summary>
        ///     NumPy's <c>np.diag</c> (<c>_twodim_base_impl.py</c>), the body of <see cref="np.diag"/> once its subclass
        ///     rules ran and the operand was checked for null: a 1-D operand is placed on the <paramref name="k"/>-th diagonal
        ///     of a new square matrix, a 2-D operand's <paramref name="k"/>-th diagonal is read.
        /// </summary>
        /// <param name="v">The operand, 1-D or 2-D.</param>
        /// <param name="k">The diagonal: 0 the main one, a positive value an upper one, a negative value a lower one.</param>
        /// <returns>For a 1-D operand a new <c>(n, n)</c> array, <c>n = v.size + |k|</c>; for a 2-D operand a read-only view
        ///     of its diagonal (<see cref="np.diagonal(NDArray, int, int, int)"/>).</returns>
        /// <exception cref="ArgumentException">The operand is not 1- or 2-dimensional (<c>Input must be 1- or 2-d.</c>, NumPy's
        ///     ValueError), or the matrix of the 1-D branch would have more than <see cref="int.MaxValue"/> rows.</exception>
        [NDScoped]
        internal static NDArray Diag(NDArray v, int k)
        {
            switch (v.ndim)
            {
                case 1:
                    return DiagonalEmbed(v, k);
                case 2:
                    return np.diagonal(v, k);
                default:
                    // NumPy: raise ValueError("Input must be 1- or 2-d.")
                    throw new ArgumentException("Input must be 1- or 2-d.");
            }
        }

        /// <summary>
        ///     NumPy's <c>np.diagflat</c> (<c>_twodim_base_impl.py</c>), the body of <see cref="np.diagflat"/> once its
        ///     subclass rules ran and the operand was checked for null: the operand raveled in C order, placed on the
        ///     <paramref name="k"/>-th diagonal of a new square matrix.
        /// </summary>
        /// <param name="v">The operand, of any rank (a 0-d one gives a <c>(1, 1)</c> matrix).</param>
        /// <param name="k">The diagonal: 0 the main one, a positive value an upper one, a negative value a lower one.</param>
        /// <returns>A new <c>(n, n)</c> array, <c>n = v.size + |k|</c>.</returns>
        /// <exception cref="ArgumentException">The matrix would have more than <see cref="int.MaxValue"/> rows.</exception>
        [NDScoped]
        internal static NDArray DiagFlat(NDArray v, int k) => DiagonalEmbed(v.ravel(), k);

        /// <summary>
        ///     Shared back-end of <see cref="np.diag"/>'s 1-D branch and <see cref="np.diagflat"/>:
        ///     allocate an <c>(n, n)</c> zero matrix and drop <paramref name="v1d"/> onto its
        ///     <paramref name="k"/>-th diagonal.
        /// </summary>
        /// <remarks>
        ///     NumPy writes the values through a strided flat slice
        ///     (<c>res.flat[i::n+1] = v</c>). NumSharp does the structurally identical thing but
        ///     without an element loop: it <see cref="Backends.Unmanaged.UnmanagedStorage.Alias(Shape)"/>es
        ///     a <b>writeable</b> length-<c>v.size</c>, stride-<c>(n+1)</c> view onto the fresh
        ///     zero buffer and lets the ordinary copy machinery fill it. The view is internal and
        ///     never escapes — it is a write target, not a result — which is why it does not carry
        ///     <see cref="np.diagonal(NDArray, int, int, int)"/>'s read-only contract.
        /// </remarks>
        [NDScopedCovered] // only callers: [NDScoped] Diag / DiagFlat (the AOT-fallback alias view is scope-reclaimed)
        private static NDArray DiagonalEmbed(NDArray v1d, int k)
        {
            long s = v1d.size;
            long absK = System.Math.Abs((long)k);
            long n = s + absK;

            if (n > int.MaxValue)
                throw new ArgumentException(
                    $"array is too big; `arr.size * arr.itemsize` is larger than the maximum possible size (n={n}).");

            var res = np.zeros(Shape.Matrix((int)n, (int)n), v1d.dtype); // descriptor: a string's zero is '' at v's width
            if (s == 0)
                return res;

            // Flat index of the first element of the k-th diagonal, then a constant
            // stride of n+1 walks it — the same arithmetic NumPy's flat-slice uses.
            long start = k >= 0 ? k : absK * n;

            // The descriptor answers which inputs cannot take the raw strided-store kernel below: the ones whose elements
            // hold references (NumPy's PyDataType_REFCHK; StringDType, whose cells point into the array's allocator), as
            // np.fill_diagonal's and the item-selection family's HasReferences decisions do.
            if (v1d.dtype.Meta.HasReferences)
            {
                // A StringDType input: the diagonal of MARKS receives v's element numbers, and the marked elements of the
                // fresh '' matrix receive the strings (never the raw strided-store kernel below).
                Backends.Strings.NDVStringMove.ScatterVia(res, v1d, (marks, index) =>
                {
                    using var diagonal = new NDArray(marks.Storage.Alias(new Shape(new[] { s }, new[] { n + 1 }, start, n * n)));
                    np.copyto(diagonal, index);
                });
                return res;
            }

            // Scatter v1d straight down the diagonal via the IL strided-store kernel — no
            // aliased view, and none of NDIter.Copy's per-strided-write overhead (measured on a
            // 3162² construct: 3.5 ms via SetData vs 2.4 ms via the kernel; the win grows with the
            // diagonal length). v1d is 1-D, so its single stride (0-d guarded to 0) addresses any
            // layout, including a strided/negative-stride input.
            long srcStride = v1d.ndim >= 1 ? v1d.Shape.strides[0] : 0;
            if (!NDArray.TryDiagonalScatter(res, start, n + 1, v1d, v1d.Shape.offset, srcStride, s))
            {
                // Fallback: alias the diagonal as a 1-D view and route through the copy machinery.
                var diagShape = new Shape(new[] {s}, new[] {n + 1}, start, res.Shape.BufferSize);
                var target = new NDArray(res.Storage.Alias(diagShape)) {TensorEngine = res.TensorEngine};
                target.SetData(v1d);
            }

            return res;
        }
    }
}
