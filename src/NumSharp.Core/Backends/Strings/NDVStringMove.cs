using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The movement routes of NEP 55 <c>StringDType</c> arrays that are not plain copies: GATHERS (<c>np.take</c>, fancy and
    ///     boolean indexing, <c>np.where</c>, <c>np.repeat</c>, <c>np.choose</c>, …) and SCATTERS (index and mask assignment,
    ///     <c>np.put</c>, <c>np.place</c>, <c>np.putmask</c>, <c>np.copyto(where=)</c>, …). Each route lets NumSharp's numeric
    ///     machinery decide WHICH elements move — with NumPy's validation, broadcasting, error texts and duplicate rules for the
    ///     same shapes — and then moves the strings through the allocators, so no cell ever lands in a buffer whose allocator
    ///     does not own it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Gathers run on cells, then re-home them.</b> The operation runs on the source's CELL VIEW
    ///     (<see cref="UnmanagedStorage.VStringCellView"/>, the 16-byte cells typed as <see cref="System.Numerics.Complex"/>), so
    ///     the fast numeric 16-byte kernels pick the elements. The result is a TEMPORARY array of raw cells that still refer to
    ///     the source's allocator (an arena offset or a heap pointer is meaningful there, and a short string is self-contained).
    ///     <see cref="Rehome"/> then duplicates every cell into a fresh <c>StringDType</c> buffer with its own allocator
    ///     (NumPy's <c>NpyString_dup</c>), reading each text through the SOURCE allocator. The temporary is never a
    ///     <c>StringDType</c> buffer, so no clear loop ever frees a string it does not own, and the source is alive and unchanged
    ///     for the whole operation (single-threaded, and both allocators are locked during the re-home).
    ///     </para>
    ///     <para>
    ///     <b>Scatters run on marks.</b> A scatter cannot run on cells: the old strings of the overwritten elements must be freed
    ///     and the new ones duplicated into the destination's allocator. So the numeric operation runs on MARKS instead — an
    ///     <c>int64</c> array of the destination's shape filled with -1 — and writes, for each selected element, the flat index
    ///     of the value element NumPy assigns there (the numeric route's broadcast, value rules, cyclic repetition, duplicates:
    ///     the last write wins). <see cref="ScatterMarks"/> then walks the marks once and, for every element whose mark is not
    ///     -1, frees the old string and duplicates the value's (<c>free_and_copy</c>).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDVStringMove
    {
        /// <summary>The operation NumPy names in the <c>MemoryError</c> texts of a string copy.</summary>
        private const string Location = "string to string cast";

        /// <summary>
        ///     The cell view of a <c>StringDType</c> array as an <see cref="NDArray"/> the caller disposes: the same cells and
        ///     layout, typed as <see cref="System.Numerics.Complex"/>. Its reference count forwards to the cells, so the array
        ///     keeps the buffer alive while it lives.
        /// </summary>
        /// <param name="t">A <c>StringDType</c> array.</param>
        /// <returns>The cell array (its elements must never be stored into another <c>StringDType</c> buffer raw).</returns>
        /// <exception cref="InvalidOperationException"><paramref name="t"/> is not a <c>StringDType</c> array.</exception>
        internal static NDArray CellArray(NDArray t)
            => new NDArray(t.Storage.VStringCellView()) { TensorEngine = t.TensorEngine };

        /// <summary>
        ///     A single-source gather: <paramref name="select"/> runs the movement on the cell view of <paramref name="source"/>,
        ///     and the cells it returns are duplicated into a fresh <c>StringDType</c> array of <paramref name="descr"/>.
        /// </summary>
        /// <param name="source">The <c>StringDType</c> array the elements come from.</param>
        /// <param name="select">The movement, applied to the cell array (a take, an index, a repeat, …). It may raise the
        ///     numeric route's errors, which are NumPy's errors for the same shapes; it must not compute on the values.</param>
        /// <param name="descr">The result descriptor, or null for <paramref name="source"/>'s.</param>
        /// <returns>
        ///     The gathered array. When <paramref name="select"/> returns a VIEW of the cell array (a basic selection), the
        ///     result is the matching view of <paramref name="source"/> (sharing its buffer and allocator), as NumPy's view.
        /// </returns>
        /// <exception cref="OutOfMemoryException">NumPy's <c>Failed to allocate string in string to string cast</c>.</exception>
        internal static NDArray Gather(NDArray source, Func<NDArray, NDArray> select, DType descr = null)
        {
            using var cells = CellArray(source);
            var raw = select(cells);
            try
            {
                if (TryViewOfSource(raw, cells, source, out var view))
                    return view;
                return Rehome(raw, source.Storage.VStringSlice, descr ?? source.dtype);
            }
            finally
            {
                if (!ReferenceEquals(raw, cells))
                    raw.Dispose();
            }
        }

        /// <summary>
        ///     When a selection returned a view of the cell array itself (no copy), the same view of the <c>StringDType</c>
        ///     source: equal dimensions and strides, the offset re-based from the cell array's buffer to the source's.
        /// </summary>
        /// <param name="raw">The selection's result.</param>
        /// <param name="cells">The cell array the selection ran on.</param>
        /// <param name="source">The <c>StringDType</c> source.</param>
        /// <param name="view">Receives the source view, or null.</param>
        /// <returns>True when <paramref name="raw"/> addresses the cell array's buffer (a view), false for a fresh result.</returns>
        private static bool TryViewOfSource(NDArray raw, NDArray cells, NDArray source, out NDArray view)
        {
            view = null;
            var src = source.Storage;
            long count = src.VStringSlice.Count;
            byte* bufferStart = src.Address, bufferEnd = src.Address + count * VStringAllocator.CellSize;
            byte* rawStart = raw.Storage.Address;
            // A fresh result is a different buffer; a view (or the cell array itself) lies inside the source's cells.
            if (rawStart < bufferStart || rawStart >= bufferEnd || (rawStart - bufferStart) % VStringAllocator.CellSize != 0)
                return false;
            var rs = raw.Shape;
            long baseCells = (rawStart - bufferStart) / VStringAllocator.CellSize;
            var shape = new Shape((long[])rs.dimensions.Clone(), (long[])rs.strides.Clone(), rs.offset + baseCells, count);
            if (!rs.IsWriteable && shape.IsWriteable)
                shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            view = new NDArray(src.Alias(shape)) { TensorEngine = source.TensorEngine };
            return true;
        }

        /// <summary>
        ///     Duplicates raw cells — copies of <paramref name="source"/>'s cells, still referring to its allocator — into a
        ///     fresh <c>StringDType</c> array of <paramref name="descr"/> with its own allocator (NumPy's <c>string_to_string</c>
        ///     rules: an NA cell of a descriptor with an NA object becomes that object's <c>str()</c> in a descriptor without
        ///     one, everything else is duplicated).
        /// </summary>
        /// <param name="raw">The raw cells (a <see cref="System.Numerics.Complex"/> array, any layout). Not disposed.</param>
        /// <param name="source">The slice whose allocator the raw cells refer to.</param>
        /// <param name="descr">The result descriptor.</param>
        /// <returns>The new array: <paramref name="raw"/>'s layout when it is a fresh dense array, else C order; when
        ///     <paramref name="raw"/> is a VIEW of a temporary (a reshape of a fresh buffer, <c>np.resize</c>'s
        ///     <c>concatenate(...)[:n].reshape(shape)</c>), the same view over a fresh <c>StringDType</c> buffer of the
        ///     temporary's size (<see cref="RehomeView"/>), so OWNDATA, strides and offset are the numeric route's.</returns>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static NDArray Rehome(NDArray raw, VStringArraySlice source, DType descr)
        {
            if (!raw.flags.owndata && !raw.Shape.IsBroadcasted)
                return RehomeView(raw, source, descr);
            var result = new NDArray(descr, ResultLayout(raw.Shape), true);
            if (result.size == 0)
                return result;
            try
            {
                using var dst = CellArray(result);
                var outAllocator = result.Storage.VStringSlice.Allocator;
                byte[] naName = NDVStringCopy.NAReplacement(source, result.Storage.VStringSlice);
                using var iter = NDIterRef.MultiNew(
                    2, new[] { raw, dst },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });
                VStringAllocator.AcquirePair(source.Allocator, outAllocator);
                try
                {
                    var inAllocator = source.Allocator;
                    iter.ForEach((dataptrs, strides, n, aux) =>
                        NDVStringCopy.CopyRun((byte*)dataptrs[0], strides[0], (byte*)dataptrs[1], strides[1], n, inAllocator, outAllocator, naName));
                }
                finally
                {
                    VStringAllocator.ReleasePair(source.Allocator, outAllocator);
                }
                return result;
            }
            catch
            {
                result.Dispose(); // the half-filled result owns the strings duplicated so far: its clear loop frees them
                throw;
            }
        }

        /// <summary>
        ///     <see cref="Rehome"/> for raw cells that are a VIEW of a temporary buffer: a fresh <c>StringDType</c> buffer of the
        ///     temporary's size (all cells empty) receives a duplicate of every cell the view reaches, at the same position, and
        ///     the result is the same view of it — a non-owning array, as NumPy's result is a view of its own temporary
        ///     (<c>np.resize</c> returns <c>concatenate(...)[:new_size].reshape(new_shape)</c>, OWNDATA false).
        /// </summary>
        /// <param name="raw">The raw cells: a non-owning, non-broadcast view of a temporary. Not disposed.</param>
        /// <param name="source">The slice whose allocator the raw cells refer to.</param>
        /// <param name="descr">The result descriptor.</param>
        /// <returns>The view over the new buffer (the buffer stays alive through the view's reference).</returns>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        /// <remarks>
        ///     Cells of the temporary that the view does not reach are left all-zero (the empty string), never copied: a
        ///     temporary may hold cells nobody wrote, and a cell of pool noise would read as an arena offset or a heap pointer.
        /// </remarks>
        private static NDArray RehomeView(NDArray raw, VStringArraySlice source, DType descr)
        {
            var rs = raw.Storage;
            long count = rs.InternalArray.Count;
            // The view's first element as a cell index of the temporary's buffer: a contiguous slice re-seats Address (offset
            // 0), a strided view keeps the buffer's address and a non-zero offset; the sum is right for both.
            long first = ((byte*)rs.Address - (byte*)rs.InternalArray.Address) / VStringAllocator.CellSize + raw.Shape.offset;
            var baseArray = new NDArray(descr, new Shape(new[] { count }), true);
            NDArray view = null;
            try
            {
                var shape = new Shape((long[])raw.Shape.dimensions.Clone(), (long[])raw.Shape.strides.Clone(), first, count);
                if (!raw.Shape.IsWriteable && shape.IsWriteable)
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                view = new NDArray(baseArray.Storage.Alias(shape)) { TensorEngine = raw.TensorEngine };
                if (view.size > 0)
                {
                    using var dst = CellArray(view);
                    var outAllocator = baseArray.Storage.VStringSlice.Allocator;
                    byte[] naName = NDVStringCopy.NAReplacement(source, baseArray.Storage.VStringSlice);
                    using var iter = NDIterRef.MultiNew(
                        2, new[] { raw, dst },
                        NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                        new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });
                    VStringAllocator.AcquirePair(source.Allocator, outAllocator);
                    try
                    {
                        var inAllocator = source.Allocator;
                        iter.ForEach((dataptrs, strides, n, aux) =>
                            NDVStringCopy.CopyRun((byte*)dataptrs[0], strides[0], (byte*)dataptrs[1], strides[1], n, inAllocator, outAllocator, naName));
                    }
                    finally
                    {
                        VStringAllocator.ReleasePair(source.Allocator, outAllocator);
                    }
                }
                return view;
            }
            catch
            {
                // The view holds its own reference on the buffer: drop it too, so the buffer's clear loop frees the strings
                // duplicated so far.
                view?.Dispose();
                throw;
            }
            finally
            {
                baseArray.Dispose(); // the view (when returned) keeps the buffer alive through its own reference
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_Ravel</c> copy of a <c>StringDType</c> array — the flat copy <c>tobytes()</c> takes of a buffer
        ///     that is not laid out in the requested order: a fresh buffer with the same dimensions, contiguous in
        ///     <paramref name="order"/>, whose strings are duplicated walking that order (C: row-major, F: column-major). The walk
        ///     decides each arena string's OFFSET in the copy (allocations are appended), which the copy's cells show.
        /// </summary>
        /// <param name="t">A <c>StringDType</c> array of any layout.</param>
        /// <param name="order"><c>'F'</c> for a column-major copy and walk; anything else row-major.</param>
        /// <returns>The new array (its own buffer and allocator); the caller disposes it.</returns>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static NDArray Flatten(NDArray t, char order)
        {
            bool fortran = order == 'F' || order == 'f';
            var result = new NDArray(t.dtype, Shape.Adopt((long[])t.Shape.dimensions.Clone(), fortran ? 'F' : 'C'), true);
            if (result.size == 0)
                return result;
            try
            {
                var source = t.Storage.VStringSlice;
                var outSlice = result.Storage.VStringSlice;
                using var src = CellArray(t);
                using var dst = CellArray(result);
                // An explicit C / F iteration order (not KEEPORDER): PyArray_CopyAsFlat walks the source in the requested
                // order, so the duplicates are appended to the new arena in that order.
                using var iter = NDIterRef.MultiNew(
                    2, new[] { src, dst },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                    fortran ? NPY_ORDER.NPY_FORTRANORDER : NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_NO_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });
                VStringAllocator.AcquirePair(source.Allocator, outSlice.Allocator);
                try
                {
                    var inAllocator = source.Allocator;
                    var outAllocator = outSlice.Allocator;
                    iter.ForEach((dataptrs, strides, n, aux) =>
                        NDVStringCopy.CopyRun((byte*)dataptrs[0], strides[0], (byte*)dataptrs[1], strides[1], n, inAllocator, outAllocator, null));
                }
                finally
                {
                    VStringAllocator.ReleasePair(source.Allocator, outSlice.Allocator);
                }
                return result;
            }
            catch
            {
                result.Dispose(); // the half-filled copy owns the strings duplicated so far: its clear loop frees them
                throw;
            }
        }

        /// <summary>
        ///     A multi-source gather (<c>np.where</c>, <c>np.choose</c>, <c>np.select</c>): <paramref name="raw"/> holds raw
        ///     cells picked from several sources, and <paramref name="which"/> (an <c>int64</c> array of the same dimensions)
        ///     names the source of each element; every cell is duplicated through its own source's allocator.
        /// </summary>
        /// <param name="raw">The raw cells (a <see cref="System.Numerics.Complex"/> array, any layout). Not disposed.</param>
        /// <param name="which">The source index of each element (<c>int64</c>, <paramref name="raw"/>'s dimensions). Not disposed.</param>
        /// <param name="sources">The source slices, indexed by <paramref name="which"/>.</param>
        /// <param name="descr">The result descriptor.</param>
        /// <param name="layout">The result's layout (NumPy's iterator output layout of the caller's operands), or null for
        ///     <paramref name="raw"/>'s layout when it is a fresh dense array, else C order.</param>
        /// <returns>The new array.</returns>
        /// <exception cref="ArgumentException"><paramref name="which"/> is not <c>int64</c>.</exception>
        /// <exception cref="IndexOutOfRangeException">A source index outside <paramref name="sources"/> (the caller's bug).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static NDArray RehomeMulti(NDArray raw, NDArray which, VStringArraySlice[] sources, DType descr, Shape? layout = null)
        {
            if (which.typecode != NPTypeCode.Int64)
                throw new ArgumentException("The source index array of a StringDType gather must be int64.", nameof(which));
            var result = new NDArray(descr, layout ?? ResultLayout(raw.Shape), true);
            if (result.size == 0)
                return result;
            try
            {
                using var dst = CellArray(result);
                var outSlice = result.Storage.VStringSlice;
                var outAllocator = outSlice.Allocator;
                var inAllocators = new VStringAllocator[sources.Length];
                var naNames = new byte[sources.Length][];
                for (int j = 0; j < sources.Length; j++)
                {
                    inAllocators[j] = sources[j].Allocator;
                    naNames[j] = NDVStringCopy.NAReplacement(sources[j], outSlice);
                }
                var locked = new List<VStringAllocator>(inAllocators) { outAllocator };
                using var iter = NDIterRef.MultiNew(
                    3, new[] { raw, which, dst },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY });
                VStringAllocator.AcquireAll(locked);
                try
                {
                    iter.ForEach((dataptrs, strides, n, aux) =>
                    {
                        byte* s = (byte*)dataptrs[0], w = (byte*)dataptrs[1], d = (byte*)dataptrs[2];
                        long ss = strides[0], ws = strides[1], ds = strides[2];
                        for (long i = 0; i < n; i++, s += ss, w += ws, d += ds)
                        {
                            long j = *(long*)w;
                            NDVStringCopy.CopyRun(s, 0, d, 0, 1, inAllocators[j], outAllocator, naNames[j]);
                        }
                    });
                }
                finally
                {
                    VStringAllocator.ReleaseAll(locked);
                }
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     <c>np.where(condition, x, y)</c> when <paramref name="x"/> or <paramref name="y"/> is a <c>StringDType</c> array —
        ///     NumPy's <c>PyArray_Where</c>: the result descriptor is <c>result_type(x, y)</c> (a <c>U</c> operand joins a
        ///     <c>StringDType</c> one; <c>S</c> and numbers raise NumPy's promotion error; two <c>StringDType</c> descriptors take
        ///     their common instance), the condition is read by its truth values, and each element is duplicated from the operand
        ///     the condition picks.
        /// </summary>
        /// <param name="condition">The condition (bool, numbers, or text read by its truth value).</param>
        /// <param name="x">Values where the condition holds.</param>
        /// <param name="y">Values elsewhere.</param>
        /// <param name="dims">The broadcast dimensions (the caller validated the broadcast).</param>
        /// <returns>The new <c>StringDType</c> array, in NumPy's iterator output layout for (condition, x, y).</returns>
        /// <exception cref="DTypePromotionError"><paramref name="x"/> and <paramref name="y"/> have no common DType.</exception>
        /// <exception cref="TypeError">Two <c>StringDType</c> descriptors with different NA objects (no common instance).</exception>
        /// <exception cref="System.Exception">The conversion of a <c>U</c> operand or of a text condition.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's string <c>MemoryError</c> texts.</exception>
        /// <remarks>
        ///     The selection runs twice through the numeric <c>np.where</c>: once on the operands' cell views (it picks the raw
        ///     cells, in NumPy's result layout) and once on two <c>int64</c> scalars (it records which operand each element came
        ///     from); <see cref="RehomeMulti"/> then duplicates every cell through its own operand's allocator. An empty result
        ///     converts nothing (NumPy's iterator runs no loop): it is an empty array of the result descriptor.
        /// </remarks>
        internal static NDArray Where(NDArray condition, NDArray x, NDArray y, long[] dims)
        {
            var descr = DTypePromotion.ResultType(x, y);
            var layout = Shape.NpyIterOutputShape((long[])dims.Clone(), new[] { condition.Shape, x.Shape, y.Shape });
            if (layout.size == 0)
                return new NDArray(descr, layout, true);

            NDArray cond = null, xt = null, yt = null;
            try
            {
                // NumPy's iterator casts the condition to bool (a StringDType condition through its bool cast, a fixed-width
                // one by its truth values) and the values to the result descriptor (a U operand becomes StringDType; a
                // StringDType operand is read as it is — its NA cells keep their meaning in the common instance).
                cond = condition.typecode == NPTypeCode.Boolean ? condition.view()
                    : condition.Storage.IsFixedString ? NDStringTruth.Truth(condition, NDStringTruth.Flavor.Cast)
                    : condition.astype(NPTypeCode.Boolean);
                xt = x.Storage.IsVString ? x : x.astype(descr, true, 'K', "unsafe");
                yt = y.Storage.IsVString ? y : y.astype(descr, true, 'K', "unsafe");

                using var xc = CellArray(xt);
                using var yc = CellArray(yt);
                using var raw = np.where(cond, xc, yc);
                using var zero = NDArray.Scalar(0L);
                using var one = NDArray.Scalar(1L);
                using var which = np.where(cond, zero, one);
                // The result takes NumPy's layout for (condition, x, y) — the condition votes too (np.tril's mask over an F
                // operand gives the strides NumPy's iterator chooses, not the operand's F order).
                return RehomeMulti(raw, which, new[] { xt.Storage.VStringSlice, yt.Storage.VStringSlice }, descr, layout);
            }
            finally
            {
                cond?.Dispose();
                if (xt is not null && !ReferenceEquals(xt, x))
                    xt.Dispose();
                if (yt is not null && !ReferenceEquals(yt, y))
                    yt.Dispose();
            }
        }

        /// <summary>
        ///     <c>ndarray.fill(value)</c> for a <c>StringDType</c> array (any layout, a view included): NumPy's
        ///     <c>PyArray_FillWithScalar</c>, which packs the value ONCE through <c>stringdtype_setitem</c> (the NA object packs
        ///     an NA element, anything else is its <c>str()</c> - a list or an array too - refused under <c>coerce=False</c>
        ///     unless it is text) and then writes it into every element (each long text gets its own allocation).
        /// </summary>
        /// <param name="dst">The writeable <c>StringDType</c> array.</param>
        /// <param name="value">Any C# value standing in for a Python object (null is <c>None</c>).</param>
        /// <exception cref="ValueError">A non-text value under <c>coerce=False</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's setitem <c>MemoryError</c> texts.</exception>
        /// <remarks>The value is converted even when the array is empty (NumPy packs it before it looks at the size).</remarks>
        internal static void Fill(NDArray dst, object value)
        {
            var slice = dst.Storage.VStringSlice;
            bool isNA = slice.IsNAValue(value);
            byte[] utf8 = isNA ? null : NDUtf8.Encode(VStringArraySlice.AsPyString(value, slice.Descr.VString.Coerce));
            if (dst.size == 0)
                return;
            var allocator = slice.Allocator;
            using var cells = CellArray(dst);
            using var iter = NDIterRef.MultiNew(
                1, new[] { cells },
                NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                new[] { NDIterPerOpFlags.READWRITE });
            allocator.Acquire();
            try
            {
                iter.ForEach((dataptrs, strides, n, aux) =>
                {
                    byte* d = (byte*)dataptrs[0];
                    long ds = strides[0];
                    for (long i = 0; i < n; i++, d += ds)
                    {
                        if (isNA)
                        {
                            if (allocator.PackNull(d) < 0)
                                throw new OutOfMemoryException("Failed to pack null string during StringDType setitem");
                            continue;
                        }
                        fixed (byte* b = utf8)
                            if (allocator.Pack(d, b, utf8.Length) < 0)
                                throw new OutOfMemoryException("Failed to pack string during StringDType setitem");
                    }
                });
            }
            finally
            {
                allocator.Release();
            }
        }

        /// <summary>
        ///     The values of a STORE into a <c>StringDType</c> array that NumPy converts with <c>PyArray_FromAny(values,
        ///     dtype=StringDType, FORCECAST)</c> — <c>np.put</c>, <c>np.putmask</c>, <c>np.place</c>, <c>np.fill_diagonal</c>,
        ///     <c>np.insert</c>, <c>np.put_along_axis</c>: Python objects (a C# literal, a boxed number, a string, a
        ///     <c>byte[]</c>, null for <c>None</c>, every element of a C# array or collection — NumPy's list) are stored by
        ///     <c>stringdtype_setitem</c> (the NA object packs NA, anything else is its <c>str()</c>, refused under
        ///     <c>coerce=False</c>; a text keeps its trailing NULs), an ndarray is CAST (allowed under <c>coerce=False</c>;
        ///     another <c>StringDType</c> descriptor follows the NA rules).
        /// </summary>
        /// <param name="value">The values as the caller received them.</param>
        /// <param name="descr">The destination's <c>StringDType</c> descriptor.</param>
        /// <returns>An array of <paramref name="descr"/>: <paramref name="value"/> itself when it already is one (an
        ///     <see cref="NDArray"/> of an equal descriptor), else a NEW array the caller disposes (or creates inside an
        ///     <c>[NDScoped]</c> member).</returns>
        /// <exception cref="ValueError">A non-text Python object under <c>coerce=False</c>, or a ragged sequence (NumPy's
        ///     inhomogeneous-shape text).</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        /// <exception cref="Exception">The cast's errors for an ndarray value.</exception>
        internal static NDArray StoreValue(object value, DType descr)
        {
            switch (value)
            {
                case NDArray nd:
                {
                    // A C# literal's 0-d array is the Python scalar it stands for (setitem); any other array is cast.
                    var packed = NDArray.PackForVStringStore(nd, descr);
                    if (!ReferenceEquals(packed, nd))
                        return packed;
                    return nd.Storage.IsVString && nd.dtype.Equals(descr) ? nd : nd.astype(descr, true, 'K', "unsafe");
                }
                case null:
                    return NDArray.PackForVStringStore(null, descr);
                case string:
                case byte[]:
                    return StoreScalar(value, descr);
                case Array arr when arr.Rank > 1 || IsTypedElement(arr.GetType().GetElementType()):
                {
                    // A rectangular C# array (or a 1-D array of typed elements): NumPy's list of Python objects, each element
                    // stored by setitem in C order (a typed numeric array is a list of Python numbers here, so coerce=False
                    // refuses it as NumPy refuses [5]).
                    var dims = new long[arr.Rank];
                    for (int d = 0; d < dims.Length; d++)
                        dims[d] = arr.GetLongLength(d);
                    var result = new NDArray(descr, Shape.Adopt(dims), true);
                    try
                    {
                        var slice = result.Storage.VStringSlice;
                        long i = 0;
                        foreach (var element in arr) // a multi-dimensional Array enumerates in row-major (C) order
                            slice.SetIndex(i++, element);
                    }
                    catch
                    {
                        result.Dispose();
                        throw;
                    }
                    return result;
                }
                case Array objects:
                    // An object[] or a jagged array (Python's nested lists): np.array's discovery with the requested
                    // descriptor stores every leaf by setitem and raises NumPy's ragged-shape error.
                    return np.array(objects, descr);
                case System.Collections.IEnumerable sequence:
                    return np.array(System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Cast<object>(sequence)), descr);
                default:
                    // A boxed number, bool or any other single object: one setitem.
                    return StoreScalar(value, descr);
            }
        }

        /// <summary>Whether a C# array's element type makes it a flat list of Python scalars (not objects, not nested arrays).</summary>
        /// <param name="elementType">The array's element type.</param>
        /// <returns>True for a typed element (string, number, bool, char, …).</returns>
        private static bool IsTypedElement(Type elementType) => elementType != typeof(object) && !elementType.IsArray;

        /// <summary>One Python object stored into a new 0-d <c>StringDType</c> array by <c>stringdtype_setitem</c>.</summary>
        /// <param name="value">The value (null is <c>None</c>).</param>
        /// <param name="descr">The <c>StringDType</c> descriptor.</param>
        /// <returns>The new 0-d array (the caller disposes it).</returns>
        /// <exception cref="ValueError">A non-text value under <c>coerce=False</c>.</exception>
        /// <exception cref="UnicodeDecodeError"><c>bytes</c> that are not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text with a lone surrogate.</exception>
        internal static NDArray StoreScalar(object value, DType descr)
        {
            var one = new NDArray(descr, Shape.Scalar, true);
            try
            {
                one.Storage.VStringSlice.SetIndex(0, value);
            }
            catch
            {
                one.Dispose();
                throw;
            }
            return one;
        }

        /// <summary>
        ///     <c>np.full(shape, fill_value, dtype)</c> / <c>np.full_like</c> / <c>np.ones</c> for a <c>StringDType</c> result:
        ///     NumPy's <c>a = empty(shape, dtype); copyto(a, fill_value, casting='unsafe')</c>. The fill value becomes an array
        ///     (a C# string a 0-d <c>U</c>, a <c>byte[]</c> a 0-d <c>S</c>, a number a numeric 0-d, a list its array) and is CAST
        ///     into every element (a number's <c>str()</c> even under <c>coerce=False</c>); <c>None</c> is NumPy's object array,
        ///     stored by setitem (the NA object, or <c>'None'</c>).
        /// </summary>
        /// <param name="shape">The result's dimensions (a fresh dense layout is kept, a view's layout re-laid out in C order).</param>
        /// <param name="fillValue">The fill value (null is <c>None</c>).</param>
        /// <param name="descr">The <c>StringDType</c> descriptor.</param>
        /// <returns>The filled array.</returns>
        /// <exception cref="ValueError"><c>None</c> under <c>coerce=False</c>, or a fill value that does not broadcast.</exception>
        /// <exception cref="System.Exception">The cast's own errors (a non-UTF-8 <c>bytes</c> value).</exception>
        internal static NDArray Full(Shape shape, object fillValue, DType descr)
        {
            var layout = shape.IsEmpty ? Shape.Adopt(Array.Empty<long>()) : shape;
            var result = new NDArray(descr, layout, true);
            try
            {
                if (fillValue is null)
                {
                    Fill(result, null);
                    return result;
                }
                if (fillValue is NDArray nd)
                {
                    np.copyto(result, nd, "unsafe");
                    return result;
                }
                using var value = fillValue is string || fillValue is byte[] || fillValue is char
                    ? NDStringArrays.FromScalar(fillValue, null)
                    : np.asanyarray(fillValue);
                np.copyto(result, value, "unsafe");
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     The elements of a <c>StringDType</c> array as C# strings in logical C order (any layout): each text, and for an NA
        ///     element the descriptor's NA object when it is a string or <c>None</c> (null).
        /// </summary>
        /// <param name="a">A <c>StringDType</c> array.</param>
        /// <returns>The values.</returns>
        /// <exception cref="InvalidCastException">An NA element whose NA object is neither a string nor <c>None</c> (a float
        ///     NaN has no string form here; <c>tolist()</c> returns it).</exception>
        internal static string[] ToStrings(NDArray a)
        {
            var result = new string[a.size];
            for (long i = 0; i < result.LongLength; i++)
            {
                var v = a.Storage.GetAtIndex(i);
                result[i] = v switch
                {
                    null => null,
                    string s => s,
                    _ => throw new InvalidCastException(
                        $"Element {i} of the {a.dtype} array is the NA object {NDPyObject.Repr(v)}, which is not a string; read it with tolist()."),
                };
            }
            return result;
        }

        /// <summary>
        ///     The flat index of every element of <paramref name="values"/> in C order, as an <c>int64</c> array of its shape —
        ///     the value operand a scatter hands to the numeric route in place of the strings.
        /// </summary>
        /// <param name="values">The value array (any dtype; only its shape is read).</param>
        /// <returns>A fresh C-contiguous <c>int64</c> array the caller disposes.</returns>
        internal static NDArray ValueIndex(NDArray values)
        {
            var dims = (long[])values.Shape.dimensions.Clone();
            using var flat = np.arange(values.size);
            return flat.reshape(Shape.Adopt(dims)).copy();
        }

        /// <summary>
        ///     Fresh MARKS for a scatter into <paramref name="dst"/>: an <c>int64</c> array of its dimensions, C order, every
        ///     element -1 ("not written").
        /// </summary>
        /// <param name="dst">The destination (only its shape is read).</param>
        /// <returns>The marks the caller disposes.</returns>
        internal static NDArray NewMarks(NDArray dst)
            => np.full(new Shape((long[])dst.Shape.dimensions.Clone()), -1L, NPTypeCode.Int64);

        /// <summary>
        ///     Applies a scatter decided on MARKS: for every element of <paramref name="dst"/> whose mark is not -1, the old string
        ///     is freed and the value element named by the mark (a flat C-order index into <paramref name="values"/>) is
        ///     duplicated into it (<c>free_and_copy</c>; an NA value stays NA, or becomes the NA object's <c>str()</c> in a
        ///     descriptor without one).
        /// </summary>
        /// <param name="dst">The writeable <c>StringDType</c> destination.</param>
        /// <param name="marks">The marks (<c>int64</c>, <paramref name="dst"/>'s dimensions, any layout). Not disposed.</param>
        /// <param name="values">The <c>StringDType</c> values the marks index (any layout; copied first when it is not
        ///     C-contiguous or when it may share memory with <paramref name="dst"/>, as NumPy copies an overlapping value).</param>
        /// <exception cref="ValueError"><paramref name="dst"/> is read-only.</exception>
        /// <exception cref="ArgumentException"><paramref name="marks"/> is not <c>int64</c>.</exception>
        /// <exception cref="IndexOutOfRangeException">A mark names no value element (the caller's marks are wrong).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static void ScatterMarks(NDArray dst, NDArray marks, NDArray values)
        {
            if (marks.typecode != NPTypeCode.Int64)
                throw new ArgumentException("The marks of a StringDType scatter must be int64.", nameof(marks));
            // NumPy's text for a read-only destination (NumSharpException.ThrowIfNotWriteable's).
            if (!dst.Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");
            if (dst.size == 0)
                return;

            // The marks index the values in C order, so the values must be C-contiguous; a value that may share memory with
            // the destination is copied first (its cells could be freed by an earlier write of this scatter).
            var dstSlice = dst.Storage.VStringSlice;
            bool overlap = ReferenceEquals(values.Storage.VStringSlice.Allocator, dstSlice.Allocator)
                           && NDMemOverlap.StoragesMayShareMemory(values.Storage, dst.Storage);
            NDArray vals = overlap || !values.Shape.IsContiguous ? values.copy('C') : values;
            try
            {
                var valSlice = vals.Storage.VStringSlice;
                byte* valBase = vals.Storage.Address + vals.Shape.offset * VStringAllocator.CellSize;
                long valCount = vals.size;
                var inAllocator = valSlice.Allocator;
                var outAllocator = dstSlice.Allocator;
                byte[] naName = NDVStringCopy.NAReplacement(valSlice, dstSlice);

                using var dcells = CellArray(dst);
                using var iter = NDIterRef.MultiNew(
                    2, new[] { marks, dcells },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK | NDIterGlobalFlags.REFS_OK,
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE });
                VStringAllocator.AcquirePair(inAllocator, outAllocator);
                try
                {
                    iter.ForEach((dataptrs, strides, n, aux) =>
                    {
                        byte* m = (byte*)dataptrs[0], d = (byte*)dataptrs[1];
                        long ms = strides[0], ds = strides[1];
                        for (long i = 0; i < n; i++, m += ms, d += ds)
                        {
                            long k = *(long*)m;
                            if (k < 0)
                                continue;
                            if ((ulong)k >= (ulong)valCount)
                                throw new IndexOutOfRangeException($"A scatter mark {k} is outside the {valCount} value elements.");
                            NDVStringCopy.CopyRun(valBase + k * VStringAllocator.CellSize, 0, d, 0, 1, inAllocator, outAllocator, naName);
                        }
                    });
                }
                finally
                {
                    VStringAllocator.ReleasePair(inAllocator, outAllocator);
                }
            }
            finally
            {
                if (!ReferenceEquals(vals, values))
                    vals.Dispose();
            }
        }

        /// <summary>
        ///     A scatter decided by the numeric route: <paramref name="op"/> runs the SAME operation (a put, a place, a masked
        ///     copy, …) on fresh marks with the value's element numbers as its values, after the caller ran NumPy's checks and
        ///     converted the value to the destination's descriptor; the marked elements then receive the strings.
        /// </summary>
        /// <param name="dst">The writeable <c>StringDType</c> destination.</param>
        /// <param name="values">The <c>StringDType</c> values the operation reads (already in the destination's descriptor).</param>
        /// <param name="op">The operation on (marks, value element numbers); it may raise its own errors.</param>
        /// <param name="keepPartialOnError">True when NumPy writes as it goes and an error leaves the elements written before
        ///     it (a put in raise mode into a contiguous target that overlaps nothing): the marks written before the error are
        ///     scattered, then the error propagates. False: an error leaves the destination unchanged.</param>
        /// <exception cref="System.Exception">Whatever <paramref name="op"/> raises.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's string <c>MemoryError</c> texts.</exception>
        internal static void ScatterVia(NDArray dst, NDArray values, Action<NDArray, NDArray> op, bool keepPartialOnError = false)
        {
            using var marks = NewMarks(dst);
            using var index = ValueIndex(values);
            try
            {
                op(marks, index);
            }
            catch
            {
                if (keepPartialOnError)
                    ScatterMarks(dst, marks, values);
                throw;
            }
            ScatterMarks(dst, marks, values);
        }

        /// <summary>
        ///     The layout of a gather's result: the raw result's own layout when it is a fresh dense array (offset 0, its buffer
        ///     exactly its elements — C, F or a permutation), else C order of its dimensions.
        /// </summary>
        /// <param name="raw">The raw result's shape.</param>
        /// <returns>The shape to allocate the <c>StringDType</c> result with.</returns>
        private static Shape ResultLayout(Shape raw)
        {
            var dims = (long[])raw.dimensions.Clone();
            if (raw.offset == 0 && raw.bufferSize == raw.size && !raw.IsBroadcasted && (raw.IsContiguous || raw.IsFContiguous))
                return raw.IsContiguous ? Shape.Adopt(dims) : Shape.Adopt(dims, 'F');
            return Shape.Adopt(dims);
        }
    }
}
