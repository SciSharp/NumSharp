using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     NumPy's six comparison ufuncs over two fixed-width string arrays of ONE class (<c>S</c> with <c>S</c>, <c>U</c>
    ///     with <c>U</c>, any widths): <c>string_comparison_loop</c> of <c>string_ufuncs.cpp</c>, run by NDIter over the
    ///     IL text kernels (<see cref="ILKernelGenerator.GetTextCompareKernel"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Where it sits.</b> <see cref="NDStringUfuncs.BinaryGuard"/> runs every check NumPy makes before its loop —
    ///     <c>out=</c> / <c>where=</c> parsing, the <c>dtype=</c> loop, the operand classes, the output cast into
    ///     <c>out=</c> (same_kind) and the broadcast — and hands a call that computes to <see cref="Execute"/>. Nothing here
    ///     re-checks those.
    ///     </para>
    ///     <para>
    ///     <b>The result.</b> A fresh result is allocated in NumPy's ufunc output layout (<see cref="Shape.NpyIterOutputShape"/>
    ///     over the inputs and the mask: an F-ordered pair gives an F result). A provided <c>out=</c> is written and returned
    ///     itself; a non-bool <c>out=</c> receives the bool results through the iterator's buffered cast (NumPy's output
    ///     cast: <c>True</c> → 1). A <c>where=</c> mask leaves the masked-off elements of <c>out=</c> untouched (those of a
    ///     fresh result are uninitialized, as in NumPy).
    ///     </para>
    ///     <para>
    ///     <b>Routes.</b> A call whose operands walk one linear order — each input either a single element or of the
    ///     result's dimensions, all C-contiguous or all F-contiguous, no mask, a bool result, no overlap with <c>out=</c> —
    ///     calls the kernel ONCE over the whole result (NumPy's trivial loop; no iterator is built). Everything else is an
    ///     NDIter pass with <c>EXTERNAL_LOOP</c> and <c>COPY_IF_OVERLAP</c>: broadcasting, strides, F / transposed / reversed
    ///     views and masks are the iterator's, and the kernel's broadcast stages read a stride-0 comparand once per call.
    ///     </para>
    /// </remarks>
    internal static class NDStringComparison
    {
        /// <summary>The per-operand flags of an unmasked call: the inputs read, the output written, never broadcast.</summary>
        private static readonly NDIterPerOpFlags[] s_flags =
        {
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
        };

        /// <summary>
        ///     The per-operand flags of a masked call. The output is READWRITE (NumPy's masked ufunc loop treats a passed-in
        ///     out as read too, so the masked-off elements keep their contents through a buffered cast) and WRITEMASKED by
        ///     the trailing ARRAYMASK operand, which <see cref="NDIterRef.ForEach"/> turns into one kernel call per mask run.
        /// </summary>
        private static readonly NDIterPerOpFlags[] s_maskedFlags =
        {
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEMASKED | NDIterPerOpFlags.NO_BROADCAST |
            NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK,
        };

        /// <summary>
        ///     Whether <paramref name="ufunc"/> is one of the six comparison ufuncs this class computes.
        /// </summary>
        /// <param name="ufunc">NumPy's ufunc name.</param>
        /// <returns>True for <c>equal</c>, <c>not_equal</c>, <c>less</c>, <c>less_equal</c>, <c>greater</c>,
        ///     <c>greater_equal</c>.</returns>
        internal static bool IsComparison(string ufunc)
            => ufunc is "equal" or "not_equal" or "less" or "less_equal" or "greater" or "greater_equal";

        /// <summary>
        ///     Computes <c>ufunc(x1, x2, out=out, where=where)</c> over two string arrays of one class, after
        ///     <see cref="NDStringUfuncs.BinaryGuard"/> accepted the call.
        /// </summary>
        /// <param name="ufunc">One of the six comparison names (<see cref="IsComparison"/>).</param>
        /// <param name="x1">The first operand (<c>S</c> or <c>U</c>).</param>
        /// <param name="x2">The second operand, of the same class.</param>
        /// <param name="out">The <c>out=</c> operand, or null (its cast and shape were checked by the guard).</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <returns><paramref name="out"/> itself, or a fresh bool array of the broadcast shape.</returns>
        /// <exception cref="ArgumentException"><paramref name="ufunc"/> is not a comparison, or the operands are not two
        ///     strings of one class (a caller bug: the guard routes only those here).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text <paramref name="out"/> (the guard keeps those
        ///     deferred: NumPy formats the bools as <c>'True'</c> / <c>'False'</c>, the number-to-text cast of a later
        ///     subphase; a direct caller reaches the iterator's buffered cast, which refuses it).</exception>
        internal static unsafe NDArray Execute(string ufunc, NDArray x1, NDArray x2, NDArray @out, NDArray where)
        {
            if (!NDStringUfuncs.ComparesText(x1.dtype, x2.dtype))
                throw new ArgumentException($"'{ufunc}' compares two strings of one class, not {x1.dtype} and {x2.dtype}.");
            // greater / greater_equal are less / less_equal of the swapped pair: the kernel reads operand slot 1 as A.
            (TextCompareOp op, bool swap) = ufunc switch
            {
                "equal" => (TextCompareOp.Equal, false),
                "not_equal" => (TextCompareOp.NotEqual, false),
                "less" => (TextCompareOp.Less, false),
                "less_equal" => (TextCompareOp.LessEqual, false),
                "greater" => (TextCompareOp.Less, true),
                "greater_equal" => (TextCompareOp.LessEqual, true),
                _ => throw new ArgumentException($"'{ufunc}' is not a comparison ufunc.", nameof(ufunc)),
            };
            int unit = x1.dtype.kind == 'U' ? 4 : 1;
            int w1 = x1.dtype.itemsize, w2 = x2.dtype.itemsize;
            var kernel = GetKernel(op, unit, swap ? w2 : w1, swap ? w1 : w2, swap ? 1 : 0);

            var dims = NDStringUfuncs.BroadcastDims(x1, x2, @out, where);
            NDArray result = @out ?? new NDArray(NPTypeCode.Boolean, ResultShape(dims, x1, x2, where), false);
            try
            {
                if (result.size == 0)
                    return result;
                if (!TryTrivial(kernel, x1, x2, result, where, @out is not null))
                    RunIterator(kernel, x1, x2, result, where);
                return result;
            }
            catch
            {
                // A fresh result never escapes a failed call; a caller's out= stays the caller's.
                if (@out is null)
                    result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     The kernel for one comparison: the IL kernel, or (no dynamic code) the managed element loop that computes the
        ///     same NumPy semantics one unit at a time.
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's width in bytes.</param>
        /// <param name="widthB">B's width in bytes.</param>
        /// <param name="aIndex">The operand slot A is read from.</param>
        /// <returns>The kernel (never null).</returns>
        internal static NDInnerLoopFunc GetKernel(TextCompareOp op, int unit, int widthA, int widthB, int aIndex)
            => ILKernelGenerator.GetTextCompareKernel(op, unit, widthA, widthB, aIndex)
               ?? ManagedKernel(op, unit, widthA, widthB, aIndex);

        /// <summary>
        ///     The layout of a fresh result: NumPy's NpyIter output allocation over the two inputs and the mask (each votes
        ///     with the axes it really strides; C order wins ties), so the result's flags and strides are NumPy's.
        /// </summary>
        /// <param name="dims">The broadcast dimensions (adopted by the returned shape — a fresh array).</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>A dense, offset-0 shape of <paramref name="dims"/>.</returns>
        private static Shape ResultShape(long[] dims, NDArray x1, NDArray x2, NDArray where)
        {
            if (dims.Length == 0)
                return Shape.NewScalar();
            if (where is null)
                return Shape.NpyIterOutputShape(dims, new[] { x1.Shape, x2.Shape });
            return Shape.NpyIterOutputShape(dims, new[] { x1.Shape, x2.Shape, where.Shape });
        }

        /// <summary>
        ///     NumPy's trivial loop: when every operand walks the SAME linear order, one kernel call covers the whole result
        ///     and no iterator is built. Taken when there is no mask, the result is bool, and each input is either one element
        ///     (read with stride 0) or has the result's dimensions in the result's contiguous order (both C, or both F) — or,
        ///     for a 0-d / 1-D result, any 1-D input at its own byte stride (NumPy's <c>PyArray_TRIVIALLY_ITERABLE</c> takes
        ///     every 1-D operand, strided and reversed views included).
        /// </summary>
        /// <param name="kernel">The comparison kernel.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="result">The output (fresh or <c>out=</c>).</param>
        /// <param name="where">The mask, or null.</param>
        /// <param name="callerOut">True when <paramref name="result"/> is the caller's <c>out=</c>, which might overlap an
        ///     input's memory; such a call takes the iterator (its COPY_IF_OVERLAP makes the overlap safe).</param>
        /// <returns>True when the call was computed here.</returns>
        private static unsafe bool TryTrivial(NDInnerLoopFunc kernel, NDArray x1, NDArray x2, NDArray result, NDArray where,
                                              bool callerOut)
        {
            if (where is not null || result.typecode != NPTypeCode.Boolean)
                return false;
            var rs = result.Shape;
            void** ptrs = stackalloc void*[3];
            long* strides = stackalloc long[3];
            if (rs.NDim <= 1)
            {
                // PyArray_TRIVIALLY_ITERABLE accepts any 1-D operand: a strided or reversed 1-D view walks with its own
                // byte stride (negative included), a single element with stride 0 — no iterator for np.less(u[::2], v[::2]).
                if (!Linear1D(x1, rs) || !Linear1D(x2, rs))
                    return false;
                strides[0] = Stride1D(x1);
                strides[1] = Stride1D(x2);
                strides[2] = rs.NDim == 0 || result.size == 1 ? 0 : rs.strides[0];
            }
            else
            {
                bool c = rs.IsContiguous, f = rs.IsFContiguous;
                if (!c && !f)
                    return false;
                if (!LinearOperand(x1, rs, ref c, ref f) || !LinearOperand(x2, rs, ref c, ref f) || !(c || f))
                    return false;
                strides[0] = x1.size == 1 ? 0 : x1.dtype.itemsize;
                strides[1] = x2.size == 1 ? 0 : x2.dtype.itemsize;
                strides[2] = 1;
            }
            if (callerOut && (MayOverlap(result, x1) || MayOverlap(result, x2)))
                return false;

            ptrs[0] = ElementPointer(x1);
            ptrs[1] = ElementPointer(x2);
            ptrs[2] = ElementPointer(result);
            kernel(ptrs, strides, result.size, null);
            GC.KeepAlive(x1);
            GC.KeepAlive(x2);
            GC.KeepAlive(result);
            return true;
        }

        /// <summary>
        ///     Whether an input can be walked linearly next to the result: a single element always (stride 0), otherwise it
        ///     must have the result's dimensions and share one of the result's contiguous orders (the flags narrow to the
        ///     orders every operand so far shares).
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The result's shape.</param>
        /// <param name="c">In: the result (and earlier operands) are C-contiguous; out: and this one too.</param>
        /// <param name="f">In / out: the same for F order.</param>
        /// <returns>False when the input needs the iterator.</returns>
        private static bool LinearOperand(NDArray x, Shape rs, ref bool c, ref bool f)
        {
            if (x.size == 1)
                return true;
            var xs = x.Shape;
            if (!SameDimensions(xs.dimensions, rs.dimensions))
                return false;
            c &= xs.IsContiguous;
            f &= xs.IsFContiguous;
            return c || f;
        }

        /// <summary>
        ///     Whether an input can be walked linearly next to a 0-d or 1-D result: a single element (stride 0), or a 1-D
        ///     operand of the result's length at any stride.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The result's shape (0-d or 1-D).</param>
        /// <returns>False when the input needs the iterator (more dimensions than one with several elements).</returns>
        private static bool Linear1D(NDArray x, Shape rs)
            => x.size == 1 || x.ndim == 1 && rs.NDim == 1 && x.Shape.dimensions[0] == rs.dimensions[0];

        /// <summary>The byte stride a <see cref="Linear1D"/> input walks with: 0 for a single element, else its own.</summary>
        /// <param name="x">The input.</param>
        /// <returns>The byte stride (negative for a reversed view).</returns>
        private static long Stride1D(NDArray x) => x.size == 1 ? 0 : x.Shape.strides[0] * (long)x.dtype.itemsize;

        /// <summary>Whether two dimension lists are equal (same rank, same extents).</summary>
        /// <param name="a">The first dimensions (null for 0-d).</param>
        /// <param name="b">The second dimensions (null for 0-d).</param>
        /// <returns>True when equal.</returns>
        private static bool SameDimensions(long[] a, long[] b)
        {
            a ??= Array.Empty<long>();
            b ??= Array.Empty<long>();
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     Whether two arrays' memory ranges may intersect (a bounds test, NumPy's <c>MAY_SHARE_BOUNDS</c>): an
        ///     <c>out=</c> that may overlap an input is left to the iterator's COPY_IF_OVERLAP.
        /// </summary>
        /// <param name="a">The output.</param>
        /// <param name="b">An input.</param>
        /// <returns>True when the byte extents intersect.</returns>
        private static bool MayOverlap(NDArray a, NDArray b) => NDMemOverlap.SolveMayShareMemory(a, b, maxWork: 0) != MemOverlap.No;

        /// <summary>The address of an array's first element (its storage address plus its byte offset).</summary>
        /// <param name="a">The array (strings or bool).</param>
        /// <returns>The element pointer.</returns>
        private static unsafe byte* ElementPointer(NDArray a)
            => (byte*)a.Address + a.Shape.offset * NDIterRef.ElementBytes(a);

        /// <summary>
        ///     The NDIter pass: <c>EXTERNAL_LOOP</c> (the kernel gets whole inner rows), <c>COPY_IF_OVERLAP</c>, and for a
        ///     non-bool <c>out=</c> the buffered cast of the bool loop output into it. The string operands travel with their
        ///     own descriptors (<c>opDescrs</c>), which size their elements.
        /// </summary>
        /// <param name="kernel">The comparison kernel.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="result">The output.</param>
        /// <param name="where">The mask, or null.</param>
        /// <exception cref="StringComputationNotSupportedException">The buffered cast into a text <c>out=</c>.</exception>
        private static unsafe void RunIterator(NDInnerLoopFunc kernel, NDArray x1, NDArray x2, NDArray result, NDArray where)
        {
            bool cast = result.typecode != NPTypeCode.Boolean;
            var flags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            if (cast)
                flags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER | NDIterGlobalFlags.DELAY_BUFALLOC;
            // The loop's output descriptor is bool; a null entry keeps an operand's own descriptor (the string widths).
            var descrs = where is null ? new DType[3] : new DType[4];
            descrs[2] = DType.Boolean;
            var ops = where is null ? new[] { x1, x2, result } : new[] { x1, x2, result, where };
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, flags, NPY_ORDER.NPY_KEEPORDER,
                cast ? NPY_CASTING.NPY_UNSAFE_CASTING : NPY_CASTING.NPY_SAFE_CASTING,
                where is null ? s_flags : s_maskedFlags, opDescrs: descrs);
            iter.ForEach(kernel);
        }

        // =============================================================================================================
        // The managed kernel (hosts without dynamic code)
        // =============================================================================================================

        /// <summary>
        ///     The comparison as a managed <see cref="NDInnerLoopFunc"/>: NumPy's <c>Buffer::strcmp</c> one unit at a time.
        ///     Used only where the IL kernels cannot be emitted (a NativeAOT host, or IL generation disabled); the IL kernels
        ///     are tested against it.
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <param name="widthA">A's width in bytes.</param>
        /// <param name="widthB">B's width in bytes.</param>
        /// <param name="aIndex">The operand slot A is read from.</param>
        /// <returns>The kernel.</returns>
        internal static unsafe NDInnerLoopFunc ManagedKernel(TextCompareOp op, int unit, int widthA, int widthB, int aIndex)
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* pa = (byte*)dataptrs[aIndex], pb = (byte*)dataptrs[1 - aIndex], po = (byte*)dataptrs[2];
                long sa = strides[aIndex], sb = strides[1 - aIndex], so = strides[2];
                for (long i = 0; i < count; i++, pa += sa, pb += sb, po += so)
                {
                    int cmp = CompareElements(pa, widthA, pb, widthB, unit);
                    bool r = op switch
                    {
                        TextCompareOp.Equal => cmp == 0,
                        TextCompareOp.NotEqual => cmp != 0,
                        TextCompareOp.Less => cmp < 0,
                        _ => cmp <= 0,
                    };
                    *po = r ? (byte)1 : (byte)0;
                }
            };

        /// <summary>
        ///     NumPy's <c>Buffer&lt;enc&gt;::strcmp</c> without rstrip: the common prefix unit by unit (unsigned), then a
        ///     nonzero unit in the longer element's tail makes it the greater.
        /// </summary>
        /// <param name="a">The first element.</param>
        /// <param name="widthA">Its width in bytes.</param>
        /// <param name="b">The second element.</param>
        /// <param name="widthB">Its width in bytes.</param>
        /// <param name="unit">1 for <c>S</c>, 4 for <c>U</c>.</param>
        /// <returns>-1, 0 or 1.</returns>
        internal static unsafe int CompareElements(byte* a, int widthA, byte* b, int widthB, int unit)
        {
            int common = Math.Min(widthA, widthB);
            for (int o = 0; o < common; o += unit)
            {
                uint ua = unit == 1 ? a[o] : Unsafe.ReadUnaligned<uint>(a + o);
                uint ub = unit == 1 ? b[o] : Unsafe.ReadUnaligned<uint>(b + o);
                if (ua != ub)
                    return ua < ub ? -1 : 1;
            }
            for (int o = common; o < widthA; o++)
                if (a[o] != 0)
                    return 1;
            for (int o = common; o < widthB; o++)
                if (b[o] != 0)
                    return -1;
            return 0;
        }
    }
}
