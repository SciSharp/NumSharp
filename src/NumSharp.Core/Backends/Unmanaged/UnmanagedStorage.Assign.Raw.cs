using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's raw_array_assign_array and raw_array_wheremasked_assign_array (array_assign_array.c): the walk that hands a
    // cast's strided loop one innermost run at a time, in NumPy's order. Every assignment route that converts element by
    // element runs it - the casts between text and numbers, the StringDType casts and the StringDType copy core -, so the
    // order a conversion error reports, the elements written before it and the order StringDType strings reach an arena
    // are NumPy's.
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
    {
        /// <summary>
        ///     NumPy's <c>raw_array_assign_array</c> (with <paramref name="mask"/> null) and
        ///     <c>raw_array_wheremasked_assign_array</c>: the axes sorted by the DESTINATION's absolute strides, an axis with a
        ///     negative destination stride walked from its end, adjacent axes coalesced (<c>PyArray_PrepareTwoRawArrayIter</c> /
        ///     <c>PrepareThree</c>, here <see cref="RawArrayIter.PrepareThree"/>), and <paramref name="loop"/> called once per
        ///     innermost run - once per run of SELECTED elements under a mask (NumPy's masked transfer,
        ///     <c>_strided_masked_wrapper_transfer_function</c>, calls the unmasked loop per such run).
        /// </summary>
        /// <param name="dst">The destination (writeable, its shape already checked against <paramref name="src"/>'s and
        ///     <paramref name="mask"/>'s by the caller, which also copied an overlapping source first).</param>
        /// <param name="src">The source, broadcastable to <paramref name="dst"/>. A source of higher rank than the destination
        ///     has extent-1 leading axes (the caller's broadcast check proved it); they are dropped, as NumPy's assignment
        ///     strips them.</param>
        /// <param name="mask">A bool mask broadcastable to <paramref name="dst"/>, or null for every element.</param>
        /// <param name="loop">The strided loop of one run, NumPy's <c>PyArrayMethod_StridedLoop</c> shape: <c>dataptrs[0]</c>
        ///     / <c>strides[0]</c> the source's first element and byte stride, <c>dataptrs[1]</c> / <c>strides[1]</c> the
        ///     destination's, <c>count</c> the run's length; <c>auxdata</c> is null (a NumSharp loop captures what it
        ///     needs).</param>
        /// <exception cref="Exception">Whatever <paramref name="loop"/> raises (a conversion error), at the run NumPy's walk
        ///     reaches it in; the runs before it are written.</exception>
        /// <remarks>
        ///     <para>
        ///     The walk order is observable: a conversion error names the first failing element in this order, the elements
        ///     before it are already converted, and a <c>StringDType</c> destination appends each duplicated arena string in
        ///     this order, so its cells' offsets (<c>tobytes()</c>) are NumPy's only when the walk is NumPy's.
        ///     </para>
        ///     <para>
        ///     Without a mask the third operand of the prepared iteration repeats the source: it coalesces exactly when the
        ///     source does, so the walk is NumPy's two-operand one.
        ///     </para>
        /// </remarks>
        internal static unsafe void RawArrayAssignArray(UnmanagedStorage dst, UnmanagedStorage src, UnmanagedStorage mask,
            NDInnerLoopFunc loop)
        {
            var ds = dst.Shape;
            int ndim = ds.NDim;
            int rank = Math.Max(ndim, 1);
            long* shape = stackalloc long[rank];
            long* sa = stackalloc long[rank];
            long* sb = stackalloc long[rank];
            long* sm = stackalloc long[rank];
            long* oShape = stackalloc long[rank];
            long* oa = stackalloc long[rank];
            long* ob = stackalloc long[rank];
            long* om = stackalloc long[rank];

            // Byte strides: a string's width and a StringDType cell's 16 bytes are the descriptor's item size (ItemBytes),
            // never the lane's.
            int dItem = dst.ItemBytes, sItem = src.ItemBytes;
            for (int d = 0; d < ndim; d++)
            {
                shape[d] = ds.dimensions[d];
                sa[d] = ds.strides[d] * dItem;
            }
            BroadcastRawStrides(src.Shape, ndim, sItem, sb);
            byte* a = dst.Address + ds.offset * dItem;
            byte* b = src.Address + src.Shape.offset * sItem;
            byte* m;
            if (mask is null)
            {
                // No mask: the third operand repeats the source (it coalesces exactly when the source does).
                for (int d = 0; d < ndim; d++)
                    sm[d] = sb[d];
                m = b;
            }
            else
            {
                BroadcastRawStrides(mask.Shape, ndim, 1, sm);
                m = mask.Address + mask.Shape.offset;
            }

            int n = RawArrayIter.PrepareThree(ndim, shape, ref a, sa, ref b, sb, ref m, sm, oShape, oa, ob, om);
            if (oShape[0] == 0)
                return;

            RawRunOuter(n, oShape, oa, ob, om, a, b, mask is null ? null : m, loop);
            // The walk reads raw addresses after the storages' last managed use: keep their buffers alive until it ends.
            GC.KeepAlive(dst);
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
        }

        /// <summary>
        ///     The byte strides of an operand broadcast to <paramref name="ndim"/> axes: right-aligned, a missing or length-1
        ///     axis broadcasting with stride 0; a higher-rank operand's extra leading axes (extent 1) dropped.
        /// </summary>
        /// <param name="s">The operand's shape.</param>
        /// <param name="ndim">The destination's rank.</param>
        /// <param name="item">The operand's element size in bytes.</param>
        /// <param name="strides">Receives <paramref name="ndim"/> byte strides.</param>
        private static unsafe void BroadcastRawStrides(Shape s, int ndim, int item, long* strides)
        {
            int lead = ndim - s.NDim;
            for (int d = 0; d < ndim; d++)
            {
                int sd = d - lead;
                strides[d] = sd < 0 || s.dimensions[sd] == 1 ? 0 : s.strides[sd] * item;
            }
        }

        /// <summary>
        ///     Walks the outer axes (1..<paramref name="n"/>-1, axis 1 fastest) of a prepared raw iteration and calls
        ///     <paramref name="loop"/> for each innermost run - for each run of SELECTED elements when a mask is given - in
        ///     NumPy's order.
        /// </summary>
        /// <param name="n">The prepared number of axes.</param>
        /// <param name="shape">The prepared extents, innermost first.</param>
        /// <param name="da">The destination's prepared byte strides.</param>
        /// <param name="db">The source's prepared byte strides.</param>
        /// <param name="dm">The mask's prepared byte strides (ignored without a mask).</param>
        /// <param name="a">The destination's first element (after the reversal of negative axes).</param>
        /// <param name="b">The source's first element (moved alongside).</param>
        /// <param name="m">The mask's first element, or null for no mask.</param>
        /// <param name="loop">The per-run strided loop.</param>
        private static unsafe void RawRunOuter(int n, long* shape, long* da, long* db, long* dm, byte* a, byte* b, byte* m,
            NDInnerLoopFunc loop)
        {
            long* coord = stackalloc long[n];
            for (int k = 0; k < n; k++)
                coord[k] = 0;
            // The loop's operand pointers and strides: [0] the source, [1] the destination (NumPy's cast loop order). The
            // inner strides are fixed for the whole walk; the pointers are rewritten per run.
            void** ptrs = stackalloc void*[2];
            long* strides = stackalloc long[2];
            strides[0] = db[0];
            strides[1] = da[0];
            long inner = shape[0];
            while (true)
            {
                if (m == null)
                {
                    ptrs[0] = b;
                    ptrs[1] = a;
                    loop(ptrs, strides, inner, null);
                }
                else
                {
                    // npy_memchr pairs: skip a run of masked-out elements, convert the run of selected ones after it.
                    long i = 0, ms = dm[0];
                    while (i < inner)
                    {
                        while (i < inner && m[i * ms] == 0)
                            i++;
                        long start = i;
                        while (i < inner && m[i * ms] != 0)
                            i++;
                        if (i > start)
                        {
                            ptrs[0] = b + start * db[0];
                            ptrs[1] = a + start * da[0];
                            loop(ptrs, strides, i - start, null);
                        }
                    }
                }

                int axis = 1;
                for (; axis < n; axis++)
                {
                    a += da[axis];
                    b += db[axis];
                    if (m != null)
                        m += dm[axis];
                    if (++coord[axis] < shape[axis])
                        break;
                    a -= da[axis] * shape[axis];
                    b -= db[axis] * shape[axis];
                    if (m != null)
                        m -= dm[axis] * shape[axis];
                    coord[axis] = 0;
                }
                if (axis >= n)
                    return;
            }
        }
    }
}
