using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Generic;

namespace NumSharp
{
    // ---------------------------------------------------------------------------------------------------------------------
    // The Python functions of NumPy's lib/_twodim_base_impl.py over the array object (Appendix D's NDArray.TwoDimBase.cs):
    // np.eye, `m = zeros((N, M), dtype=dtype, order=order); m[:M-k].flat[i::M+1] = 1`, and np.diag / np.diagflat, whose
    // 1-D branch is `res = zeros((n, n), v.dtype); res[:n-k].flat[i::n+1] = v`; np.tri with np.tril / np.triu (and the
    // write-once ZeroBytes they share), and np.tril_indices / np.triu_indices with their _from forms. They are the array
    // object's own members (the np-layer dtype plan's D1: functions that create arrays), and np.eye, np.identity, np.diag,
    // np.diagflat, np.tri, np.tril, np.triu and the four index functions are facades over them.
    //
    // The plan's phase P6 made its np.* calls (np.zeros, np.diagonal, np.copyto) the members their facades dispatch to
    // and moved the file from beside its facades (Creation/) to Backends/, beside the array object's class (plan C3).
    // P8 moved the triangle family here with its calls converted the same way (np.zeros / np.ones / np.arange / np.where /
    // np.tri). np.mask_indices stays a facade: its mask_func's result goes through np.not_equal and np.nonzero, whose class
    // rules NumPy's `a != 0` and `nonzero(a)` apply to the result's class, and these plain statements would not.
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
        internal static unsafe NDArray eye(int N, int? M, int k, DType dtype, char order)
        {
            int cols = M ?? N;
            // np.zeros' own order of checks once np.eye checked the order's letter and the device: A and K are refused
            // (only C and F lay out a new array), then a negative dimension by the allocation (AllocationGuard's NumPy text,
            // "negative dimensions are not allowed", for N or M alike).
            char physical = OrderResolver.Resolve(order);

            var resolvedType = dtype ?? DType.Double;
            var m = new NDArray(resolvedType, Shape.Matrix(N, cols), true);
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
        internal static NDArray diag(NDArray v, int k)
        {
            switch (v.ndim)
            {
                case 1:
                    return DiagonalEmbed(v, k);
                case 2:
                    // np.diagonal(v, k)'s statement: np.diag's facade has applied the class rules already.
                    return diagonal(v, k, 0, 1);
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
        internal static NDArray diagflat(NDArray v, int k) => DiagonalEmbed(v.ravel(), k);

        /// <summary>
        ///     Shared back-end of <see cref="np.diag"/>'s 1-D branch and <see cref="np.diagflat"/>:
        ///     allocate an <c>(n, n)</c> zero matrix and drop <paramref name="v1d"/> onto its
        ///     <paramref name="k"/>-th diagonal.
        /// </summary>
        /// <remarks>
        ///     NumPy writes the values through a strided flat slice
        ///     (<c>res.flat[i::n+1] = v</c>). NumSharp does the structurally identical thing but
        ///     without an element loop: it <see cref="UnmanagedStorage.Alias(Shape)"/>es
        ///     a <b>writeable</b> length-<c>v.size</c>, stride-<c>(n+1)</c> view onto the fresh
        ///     zero buffer and lets the ordinary copy machinery fill it. The view is internal and
        ///     never escapes — it is a write target, not a result — which is why it does not carry
        ///     <see cref="np.diagonal(NDArray, int, int, int)"/>'s read-only contract.
        /// </remarks>
        [NDScopedCovered] // only callers: [NDScoped] diag / diagflat (the AOT-fallback alias view is scope-reclaimed)
        private static NDArray DiagonalEmbed(NDArray v1d, int k)
        {
            long s = v1d.size;
            long absK = System.Math.Abs((long)k);
            long n = s + absK;

            if (n > int.MaxValue)
                throw new ArgumentException(
                    $"array is too big; `arr.size * arr.itemsize` is larger than the maximum possible size (n={n}).");

            var res = new NDArray(v1d.dtype, Shape.Matrix((int)n, (int)n), true); // descriptor: a string's zero is '' at v's width
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
                    UnmanagedStorage.Copyto(diagonal, index, NPY_CASTING.NPY_SAME_KIND_CASTING, null);
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

        // ---------------------------- np.tri ----------------------------

        /// <summary>
        ///     NumPy's <c>np.tri</c> (<c>_twodim_base_impl.py</c>), the body of
        ///     <see cref="np.tri(int, int?, int, DType)"/> once the dtype was read as its storage lane: an <c>(N, M)</c> array
        ///     with ones at and below the <paramref name="k"/>-th diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">The number of rows; a negative count is an empty axis (NumPy's <c>arange</c> of it), not an
        ///     error.</param>
        /// <param name="M">The number of columns, or null for <paramref name="N"/>; negative is an empty axis too.</param>
        /// <param name="k">The diagonal at and below which the array is filled: 0 the main one, a positive value above it,
        ///     a negative value below it.</param>
        /// <param name="dtype">The storage lane of the result (the facade's dtype, float64 by default).</param>
        /// <returns>A new, C-contiguous, writeable <c>(N, M)</c> array.</returns>
        /// <remarks>
        ///     <para>
        ///     NumPy computes this as <c>greater_equal.outer(arange(N), arange(-k, M-k))</c>
        ///     followed by an <c>astype</c> — two full N×M passes plus a bool temporary.
        ///     Because row <c>i</c> is exactly <c>c_i = clamp(i + k + 1, 0, M)</c> ones followed
        ///     by zeros, NumSharp instead allocates the (already zeroed) result and blits a
        ///     prefix of a single pre-built ones row into each row. That is one
        ///     <see cref="Buffer.MemoryCopy(void*, void*, long, long)"/> per row over an untouched zero tail — no
        ///     comparison kernel, no mask temporary, and completely dtype-agnostic (the copy
        ///     only ever sees bytes).
        ///     </para>
        ///     <para>
        ///     Scoped by <c>[NDScoped]</c>: the pre-built ones row is released on return; the result is yielded. The
        ///     allocation follows <see cref="SizeBucketedBufferPool.PrefersWriteOnce"/>: below its threshold a pooled
        ///     buffer comes back dirty, so it is allocated uninitialised and every byte is written once.
        ///     </para>
        /// </remarks>
        [NDScoped /* the ones row the rows are blitted from is released; the result is yielded */]
        internal static unsafe NDArray tri(int N, int? M, int k, NPTypeCode dtype)
        {
            // NumPy builds the row/column axes with arange, so a negative extent simply
            // produces an empty axis rather than an error.
            int rows = N < 0 ? 0 : N;
            int cols = (M ?? N) < 0 ? 0 : (M ?? N);

            // np.zeros(shape, dtype)'s statement: a zero-filled allocation (NPTypeCode converts to its DType).
            if (rows == 0 || cols == 0)
                return new NDArray(dtype, Shape.Matrix(rows, cols), true);

            int elemSize = dtype.SizeOf();
            bool writeOnce = SizeBucketedBufferPool.PrefersWriteOnce((long)rows * cols * elemSize);

            // Below the threshold the buffer comes back dirty, so pre-zeroing it would be a
            // wasted pass — allocate uninitialised and write every byte exactly once (ones
            // prefix, then zero suffix). Above it np.zeros is free, so only the ones prefix
            // is written. See SizeBucketedBufferPool.PrefersWriteOnce.
            var res = writeOnce
                ? new NDArray(dtype, Shape.Matrix(rows, cols), false)
                : new NDArray(dtype, Shape.Matrix(rows, cols), true);

            // One row of ones, blitted as a prefix into every output row (np.ones' statement: the array object's ones).
            var onesRow = ones(new Shape(cols), dtype);
            byte* src = onesRow.Storage.Address + onesRow.Shape.Offset * elemSize;
            byte* dst = res.Storage.Address + res.Shape.Offset * elemSize;
            long rowBytes = (long)cols * elemSize;

            for (long i = 0; i < rows; i++)
            {
                // Columns j <= i + k are kept => the first clamp(i + k + 1, 0, cols) entries.
                long keep = i + k + 1;
                if (keep < 0) keep = 0;
                else if (keep > cols) keep = cols;

                byte* row = dst + i * rowBytes;
                long onesBytes = keep * elemSize;

                if (onesBytes > 0)
                    Buffer.MemoryCopy(src, row, onesBytes, onesBytes);

                if (writeOnce)
                    ZeroBytes(row + onesBytes, rowBytes - onesBytes);
            }

            return res;
        }

        /// <summary>
        ///     Zero <paramref name="count"/> bytes at <paramref name="p"/>. Shared by the
        ///     write-once triangular fills (<see cref="tri(int, int?, int, NPTypeCode)"/>,
        ///     <see cref="tril"/>/<see cref="triu"/>), which allocate uninitialised and must
        ///     clear the dropped span themselves, and by numpy.polynomial's companion-matrix roots. Chunked because
        ///     <see cref="Span{T}"/> is int-length while a single row can in principle exceed 2 GB.
        /// </summary>
        /// <param name="p">The first byte to clear; the caller keeps the memory behind it alive and writable.</param>
        /// <param name="count">The number of bytes; zero or negative clears nothing.</param>
        internal static unsafe void ZeroBytes(byte* p, long count)
        {
            while (count > 0)
            {
                int chunk = count > int.MaxValue ? int.MaxValue : (int)count;
                new Span<byte>(p, chunk).Clear();
                p += chunk;
                count -= chunk;
            }
        }

        // ---------------------------- np.tril / np.triu ----------------------------

        /// <summary>
        ///     NumPy's <c>np.tril</c> (<c>_twodim_base_impl.py</c>), the body of <see cref="np.tril"/> inside its matrix
        ///     region: a copy of <paramref name="m"/> with the elements above the <paramref name="k"/>-th diagonal of its last
        ///     two axes zeroed.
        /// </summary>
        /// <param name="m">The array (a 1-d array is broadcast as the rows of a square matrix, as NumPy's
        ///     <c>tri(*m.shape[-2:])</c> of one dimension is).</param>
        /// <param name="k">The diagonal above which elements are zeroed.</param>
        /// <returns>A new, writeable, C-contiguous array of <paramref name="m"/>'s dtype (2-d for a 1-d operand).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="m"/> is 0-d (NumPy leaks <c>tri()</c>'s TypeError text:
        ///     <c>tri() missing 1 required positional argument: 'N'</c>).</exception>
        /// <exception cref="OverflowException">The last two axes exceed <see cref="int.MaxValue"/> on the general
        ///     <c>where</c> path.</exception>
        internal static NDArray tril(NDArray m, int k) => Triangle(m, k, lower: true);

        /// <summary>
        ///     NumPy's <c>np.triu</c> (<c>_twodim_base_impl.py</c>), the body of <see cref="np.triu"/> inside its matrix
        ///     region: a copy of <paramref name="m"/> with the elements below the <paramref name="k"/>-th diagonal of its last
        ///     two axes zeroed.
        /// </summary>
        /// <param name="m">The array (a 1-d array is broadcast as the rows of a square matrix).</param>
        /// <param name="k">The diagonal below which elements are zeroed.</param>
        /// <returns>A new, writeable, C-contiguous array of <paramref name="m"/>'s dtype (2-d for a 1-d operand).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="m"/> is 0-d (<c>tri() missing 1 required positional
        ///     argument: 'N'</c>).</exception>
        /// <exception cref="OverflowException">The last two axes exceed <see cref="int.MaxValue"/> on the general
        ///     <c>where</c> path.</exception>
        internal static NDArray triu(NDArray m, int k) => Triangle(m, k, lower: false);

        /// <summary>
        ///     Shared implementation of <see cref="tril"/> / <see cref="triu"/>.
        /// </summary>
        /// <param name="m">The array.</param>
        /// <param name="k">The diagonal.</param>
        /// <param name="lower">True for tril (keep at and below the diagonal), false for triu (keep at and above).</param>
        /// <returns>A new array (see <see cref="tril"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="m"/> is 0-d.</exception>
        /// <exception cref="OverflowException">The last two axes exceed <see cref="int.MaxValue"/> on the general
        ///     path.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy is literally <c>where(tri(*m.shape[-2:], k=k, dtype=bool), m, zeros(1, m.dtype))</c>
        ///     for <c>tril</c> (and the complementary <c>k-1</c> mask for <c>triu</c>). Two
        ///     consequences of that spelling are load-bearing behaviour, reproduced here:
        ///     </para>
        ///     <list type="bullet">
        ///       <item>
        ///         <b>A 1-D input produces a 2-D result.</b> <c>m.shape[-2:]</c> of a
        ///         <c>(n,)</c> array is <c>(n,)</c>, so the mask is <c>tri(n)</c> → <c>(n, n)</c>
        ///         and the <c>where</c> broadcasts the row across it. Probed:
        ///         <c>np.tril([1,2,3])</c> → <c>[[1,0,0],[1,2,0],[1,2,3]]</c>.
        ///       </item>
        ///       <item>
        ///         <b>A 0-d input raises</b> — <c>m.shape[-2:]</c> is empty and NumPy leaks
        ///         <c>tri()</c>'s missing-argument <c>TypeError</c> straight to the caller. The
        ///         text is reproduced verbatim.
        ///       </item>
        ///     </list>
        ///     <para>
        ///     <b>Fast path.</b> Because the kept columns of every row form one contiguous run
        ///     (<c>[0, c_i)</c> for <c>tril</c>, <c>[lo_i, cols)</c> for <c>triu</c>), the mask
        ///     is redundant: allocate the result zeroed and blit just the kept run of each row.
        ///     That reads and writes only the retained triangle — roughly half the traffic of
        ///     NumPy's two full-size passes — with no mask temporary and no per-dtype code (the
        ///     copy is byte-wise). It applies whenever the source's last axis is unit-stride;
        ///     anything else (broadcast last axis, F-contiguous, <c>::2</c> column steps) falls
        ///     back to the literal <c>where</c> composition, which is always correct.
        ///     </para>
        ///     <para>Scoped by <c>[NDScoped]</c>: the general path's mask and zero are released; the result is
        ///     yielded.</para>
        /// </remarks>
        [NDScoped /* the general path's tri mask and its one-element zero are released; the result is yielded */]
        private static NDArray Triangle(NDArray m, int k, bool lower)
        {
            if (m is null) throw new ArgumentNullException(nameof(m));

            int ndim = m.ndim;
            if (ndim == 0)
                // NumPy leaks tri()'s own TypeError here: `tri(*m.shape[-2:])` gets no args.
                throw new ArgumentException("tri() missing 1 required positional argument: 'N'");

            // NumPy: tri(*m.shape[-2:], ...) — for 1-D that is tri(n) which squares up to (n, n).
            long rows = ndim == 1 ? m.Shape.dimensions[0] : m.Shape.dimensions[ndim - 2];
            long cols = ndim == 1 ? m.Shape.dimensions[0] : m.Shape.dimensions[ndim - 1];

            var fast = TryTriangleBlit(m, k, lower, cols);
            if (!(fast is null))
                return fast;

            // Literal NumPy composition — the mask's diagonal differs by one between the
            // two directions, and the operand order is what flips which half survives. The calls are the members np.tri,
            // np.zeros and np.where dispatch to (np.tril's region already holds a matrix operand plain).
            var mask = tri(checked((int)rows), checked((int)cols), lower ? k : k - 1, NPTypeCode.Boolean);
            var zero = new NDArray(m.dtype, new Shape(1), true); // the descriptor: a string's zero is '' at m's width
            return lower ? where(mask, m, zero) : where(mask, zero, m);
        }

        /// <summary>
        ///     Zero-allocate the result and copy only the retained column run of each row.
        ///     Returns <c>null</c> when the layout is not eligible, leaving the caller to use
        ///     the general <c>where</c> composition.
        /// </summary>
        /// <param name="m">The array (not null, at least 1-d).</param>
        /// <param name="k">The diagonal.</param>
        /// <param name="lower">True for tril, false for triu.</param>
        /// <param name="cols">The length of <paramref name="m"/>'s last axis.</param>
        /// <returns>A new array of <paramref name="m"/>'s shape and dtype, or null for a StringDType operand, a 1-d one, or
        ///     a last axis that is not unit-stride.</returns>
        private static unsafe NDArray TryTriangleBlit(NDArray m, int k, bool lower, long cols)
        {
            int ndim = m.ndim;

            // A StringDType input takes NumPy's where composition (its StringDType gather duplicates the strings); the
            // blit below copies elements as raw bytes.
            if (m.Storage.IsVString)
                return null;

            // The 1-D case broadcasts a row across a square result — a different traversal
            // than "copy a run of each source row", so leave it to the composition path.
            if (ndim < 2)
                return null;

            var shape = m.Shape;

            // A unit-strided last axis is what makes the kept run contiguous in the SOURCE;
            // the destination is contiguous by construction. Broadcast (stride 0) and
            // column-stepped views fail this and fall back.
            if (shape.strides[ndim - 1] != 1)
                return null;

            // np.zeros(shape, dtype)'s statement below: a zero-filled allocation of m's descriptor.
            long size = m.size;
            if (size == 0 || cols == 0)
                return new NDArray(m.dtype, Shape.Adopt((long[])m.Shape.dimensions.Clone()), true);

            int elemSize = m.dtypesize; // the descriptor's itemsize (a string's width), not the lane's
            bool writeOnce = SizeBucketedBufferPool.PrefersWriteOnce(size * elemSize);

            // Below the threshold the buffer comes back dirty, so allocate uninitialised and
            // write every byte of every row exactly once (zeros, retained run, zeros). Above it
            // np.zeros rides free OS zero-pages and only the retained run is worth touching.
            var res = writeOnce
                ? new NDArray(m.dtype, Shape.Adopt((long[])m.Shape.dimensions.Clone()), false)
                : new NDArray(m.dtype, Shape.Adopt((long[])m.Shape.dimensions.Clone()), true);
            byte* src = m.Storage.Address;
            byte* dst = res.Storage.Address + res.Shape.Offset * elemSize;

            long numRows = size / cols;
            int outer = ndim - 1; // number of axes above the column axis

            // Odometer over every axis except the last: one iteration per output row.
            Span<long> coord = stackalloc long[outer];
            coord.Clear();

            for (long r = 0; r < numRows; r++)
            {
                long i = coord[outer - 1]; // row index within the trailing 2-D plane

                // Retained column run for this row.
                long lo, hi;
                if (lower)
                {
                    lo = 0;
                    hi = i + k + 1;
                    if (hi > cols) hi = cols;
                    if (hi < 0) hi = 0;
                }
                else
                {
                    lo = i + k;
                    if (lo < 0) lo = 0;
                    if (lo > cols) lo = cols;
                    hi = cols;
                }

                byte* dstRow = dst + r * cols * elemSize;

                if (writeOnce)
                    ZeroBytes(dstRow, lo * elemSize);

                if (hi > lo)
                {
                    // Source element offset of (coord..., lo) through the real strides.
                    long srcElem = shape.Offset;
                    for (int d = 0; d < outer; d++)
                        srcElem += coord[d] * shape.strides[d];
                    srcElem += lo;

                    long bytes = (hi - lo) * elemSize;
                    Buffer.MemoryCopy(
                        source: src + srcElem * elemSize,
                        destination: dstRow + lo * elemSize,
                        destinationSizeInBytes: bytes,
                        sourceBytesToCopy: bytes);
                }

                if (writeOnce)
                    ZeroBytes(dstRow + hi * elemSize, (cols - hi) * elemSize);

                // Advance the odometer.
                for (int d = outer - 1; d >= 0; d--)
                {
                    if (++coord[d] < shape.dimensions[d]) break;
                    coord[d] = 0;
                }
            }

            return res;
        }

        // ---------------------------- np.tril_indices / np.triu_indices ----------------------------

        /// <summary>
        ///     NumPy's <c>np.tril_indices</c> (<c>_twodim_base_impl.py</c>), the body of <see cref="np.tril_indices"/>: the
        ///     row and column indices of the lower triangle of an <c>(n, m)</c> array, at and below the
        ///     <paramref name="k"/>-th diagonal, in C (row-major) order.
        /// </summary>
        /// <param name="n">The row count; negative is an empty axis.</param>
        /// <param name="k">The diagonal offset.</param>
        /// <param name="m">The column count, or null for <paramref name="n"/>; negative is an empty axis.</param>
        /// <returns>Two new owning int64 arrays of the same length (OWNDATA True, as NumPy's boolean-mask gathers
        ///     are).</returns>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the shared column ladder is released; the two results are yielded.</remarks>
        [NDScoped /* the shared 0..m-1 column ladder is released; both index arrays are yielded */]
        internal static NDArray<long>[] tril_indices(int n, int k, int? m) => TriangleIndices(n, k, m, lower: true);

        /// <summary>
        ///     NumPy's <c>np.tril_indices_from</c>: <see cref="tril_indices"/> of a 2-d array's shape.
        /// </summary>
        /// <param name="arr">The 2-d array whose shape supplies the dimensions.</param>
        /// <param name="k">The diagonal offset.</param>
        /// <returns>The row and column indices (see <see cref="tril_indices"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="arr"/> is not 2-d (<c>input array must be 2-d</c>, NumPy's
        ///     ValueError text).</exception>
        /// <exception cref="OverflowException">A dimension exceeds <see cref="int.MaxValue"/>.</exception>
        internal static NDArray<long>[] tril_indices_from(NDArray arr, int k)
        {
            RequireTwoDimensional(arr);
            return tril_indices(checked((int)arr.Shape.dimensions[arr.ndim - 2]), k, checked((int)arr.Shape.dimensions[arr.ndim - 1]));
        }

        /// <summary>
        ///     NumPy's <c>np.triu_indices</c> (<c>_twodim_base_impl.py</c>), the body of <see cref="np.triu_indices"/>: the
        ///     row and column indices of the upper triangle of an <c>(n, m)</c> array, at and above the
        ///     <paramref name="k"/>-th diagonal, in C (row-major) order.
        /// </summary>
        /// <param name="n">The row count; negative is an empty axis.</param>
        /// <param name="k">The diagonal offset.</param>
        /// <param name="m">The column count, or null for <paramref name="n"/>; negative is an empty axis.</param>
        /// <returns>Two new owning int64 arrays of the same length (OWNDATA True, as NumPy's boolean-mask gathers
        ///     are).</returns>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the shared column ladder is released; the two results are yielded.</remarks>
        [NDScoped /* the shared 0..m-1 column ladder is released; both index arrays are yielded */]
        internal static NDArray<long>[] triu_indices(int n, int k, int? m) => TriangleIndices(n, k, m, lower: false);

        /// <summary>
        ///     NumPy's <c>np.triu_indices_from</c>: <see cref="triu_indices"/> of a 2-d array's shape.
        /// </summary>
        /// <param name="arr">The 2-d array whose shape supplies the dimensions.</param>
        /// <param name="k">The diagonal offset.</param>
        /// <returns>The row and column indices (see <see cref="triu_indices"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="arr"/> is not 2-d (<c>input array must be 2-d</c>, NumPy's
        ///     ValueError text).</exception>
        /// <exception cref="OverflowException">A dimension exceeds <see cref="int.MaxValue"/>.</exception>
        internal static NDArray<long>[] triu_indices_from(NDArray arr, int k)
        {
            RequireTwoDimensional(arr);
            return triu_indices(checked((int)arr.Shape.dimensions[arr.ndim - 2]), k, checked((int)arr.Shape.dimensions[arr.ndim - 1]));
        }

        /// <summary>
        ///     The _from forms' argument check: NumPy's <c>if arr.ndim != 2: raise ValueError("input array must be 2-d")</c>,
        ///     after NumSharp's null check.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="arr"/> is not 2-d.</exception>
        private static void RequireTwoDimensional(NDArray arr)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (arr.ndim != 2)
                throw new ArgumentException("input array must be 2-d");
        }

        /// <summary>
        ///     Shared generator behind <see cref="tril_indices"/> / <see cref="triu_indices"/>.
        /// </summary>
        /// <param name="n">The row count; negative is an empty axis.</param>
        /// <param name="k">The diagonal offset.</param>
        /// <param name="m">The column count, or null for <paramref name="n"/>.</param>
        /// <param name="lower">True for the lower triangle, false for the upper one.</param>
        /// <returns>The row and column index arrays, new and owning.</returns>
        /// <remarks>
        ///     <para>
        ///     NumPy materialises an <c>n×m</c> boolean mask via <see cref="tri"/>, broadcasts
        ///     both sparse coordinate grids to that shape, and runs two boolean gathers:
        ///     <c>tuple(broadcast_to(inds, tri_.shape)[tri_] for inds in indices(tri_.shape, sparse=True))</c>.
        ///     That is <c>O(n·m)</c> work plus three temporaries even when the answer is short
        ///     (a <c>k = -n</c> query returns nothing after scanning the whole mask).
        ///     </para>
        ///     <para>
        ///     The triangle is <em>separable</em>, so none of that is necessary: the kept columns
        ///     of row <c>i</c> are the single contiguous run <c>[0, c_i)</c> (lower) or
        ///     <c>[lo_i, m)</c> (upper). NumSharp sums those run lengths to size the output
        ///     exactly (the house pre-size-then-fill pattern), then emits each run with bulk
        ///     operations only — a <see cref="Span{T}.Fill"/> of the constant row index and a
        ///     <see cref="Buffer.MemoryCopy(void*, void*, long, long)"/> from a shared <c>0..m-1</c> iota for the columns.
        ///     Cost is <c>O(output)</c>, never <c>O(n·m)</c>.
        ///     </para>
        /// </remarks>
        [NDScopedCovered /* only reached via the [NDScoped] tril_indices / triu_indices members (the iota ladder is scope-reclaimed) */]
        private static unsafe NDArray<long>[] TriangleIndices(int n, int k, int? m, bool lower)
        {
            // NumPy reaches these extents through tri()/arange(), where a negative count is an
            // empty axis rather than an error.
            long rows = n < 0 ? 0 : n;
            long cols = (m ?? n) < 0 ? 0 : (m ?? n);

            // Pass 1 — exact size.
            long total = 0;
            for (long i = 0; i < rows; i++)
                total += RunLength(i, k, cols, lower);

            // OWNING arrays, as NumPy's boolean-mask gathers are (they were MakeGeneric aliases of their own buffers,
            // which reported OWNDATA False — the open-bugs rundown's bug 8 sweep).
            var rowsResult = new NDArray<long>(new Shape(total), false);
            var colsResult = new NDArray<long>(new Shape(total), false);
            var result = new[] {rowsResult, colsResult};

            if (total == 0)
                return result;

            // Shared 0..cols-1 column ladder; every row's column run is a slice of it (np.arange(cols)'s statement).
            var iota = UnmanagedStorage.ArangeObj(0L, cols, 1L, null);
            long* iotaPtr = (long*)iota.Storage.Address + iota.Shape.Offset;
            long* rp = (long*)rowsResult.Storage.Address;
            long* cp = (long*)colsResult.Storage.Address;

            // Pass 2 — emit, bulk-wise.
            long w = 0;
            for (long i = 0; i < rows; i++)
            {
                long lo = lower ? 0 : Clamp(i + k, 0, cols);
                long hi = lower ? Clamp(i + k + 1, 0, cols) : cols;
                long len = hi - lo;
                if (len <= 0) continue;

                new Span<long>(rp + w, (int)len).Fill(i);

                long bytes = len * sizeof(long);
                Buffer.MemoryCopy(
                    source: iotaPtr + lo,
                    destination: cp + w,
                    destinationSizeInBytes: bytes,
                    sourceBytesToCopy: bytes);

                w += len;
            }

            return result;
        }

        /// <summary>Number of retained columns in row <paramref name="i"/>.</summary>
        /// <param name="i">The row.</param>
        /// <param name="k">The diagonal offset.</param>
        /// <param name="cols">The column count.</param>
        /// <param name="lower">True for the lower triangle.</param>
        /// <returns>The run length, <c>0 .. cols</c>.</returns>
        private static long RunLength(long i, int k, long cols, bool lower)
        {
            if (lower)
                return Clamp(i + k + 1, 0, cols);

            return cols - Clamp(i + k, 0, cols);
        }

        /// <summary>Clamps <paramref name="value"/> into <c>[min, max]</c>.</summary>
        /// <param name="value">The value.</param>
        /// <param name="min">The lower bound.</param>
        /// <param name="max">The upper bound (not below <paramref name="min"/>).</param>
        /// <returns>The clamped value.</returns>
        private static long Clamp(long value, long min, long max)
            => value < min ? min : (value > max ? max : value);
    }
}
