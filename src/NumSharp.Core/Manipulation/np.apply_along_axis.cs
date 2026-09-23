using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        //  np.apply_along_axis — apply a function to 1-D slices along an axis.
        //
        //  Port of NumPy 2.x numpy.apply_along_axis (numpy/lib/_shape_base_impl.py). NumPy's own
        //  structure, followed here move for move:
        //
        //      axis = normalize_axis_index(axis, arr.ndim)
        //      inarr_view = transpose(arr, [..axis-1, axis+1.., axis])   # iteration axis last
        //      inds = ndindex(inarr_view.shape[:-1])                     # the non-axis dims
        //      res  = asanyarray(func1d(inarr_view[ind0], ...))          # first slice fixes shape+dtype
        //      buff = zeros(inarr_view.shape[:-1] + res.shape, res.dtype)
        //      for ind: buff[ind] = func1d(inarr_view[ind], ...)         # each write is contiguous
        //      out  = transpose(buff, buff_permute)                      # move the new dims into `axis`
        //
        //  There is no data-parallel kernel to emit here: func1d is arbitrary user code invoked once
        //  per 1-D slice, so this is a composition over the existing view/iterator machinery
        //  (transpose views, ndindex, GetData slices, copyto writes). The C# delegate call is orders
        //  of magnitude cheaper than NumPy's per-slice Python call, so even this straight composition
        //  outruns NumPy comfortably — the callback dispatch, not the plumbing, is the cost.
        // =====================================================================

        /// <summary>
        ///     Applies <paramref name="func1d"/> to each 1-D slice of <paramref name="arr"/> taken
        ///     along <paramref name="axis"/>, then reassembles the results. The FIRST slice's result
        ///     fixes the output dtype and the block of new dimensions that REPLACE <paramref name="axis"/>:
        ///     a scalar-returning func drops the axis (one fewer dimension), a 1-D-returning func
        ///     keeps the rank, a higher-D func inserts those dimensions in the axis' place. So the
        ///     output shape is <c>arr.shape[:axis] + func_result.shape + arr.shape[axis+1:]</c>.
        /// </summary>
        /// <param name="func1d">
        ///     Applied to each 1-D slice; must accept a 1-D <see cref="NDArray"/> and return an
        ///     <see cref="NDArray"/> (a scalar result is a 0-d array). Called once per slice in
        ///     logical C-order over the non-axis dimensions. Every call is expected to return the same
        ///     shape and dtype as the first — a differing result is broadcast and unsafe-cast into the
        ///     first result's slot (NumPy's <c>buff[ind] = res</c> semantics), so a shape that cannot
        ///     broadcast raises.
        /// </param>
        /// <param name="axis">
        ///     Axis of <paramref name="arr"/> to slice along; negative counts from the end. The slice
        ///     handed to <paramref name="func1d"/> is one-dimensional with length <c>arr.shape[axis]</c>.
        /// </param>
        /// <param name="arr">Input array, of any rank (≥ 1) and any memory layout.</param>
        /// <returns>
        ///     A freshly allocated array holding the reassembled results, with <paramref name="func1d"/>'s
        ///     output dtype. The result is a transposed view over a fresh contiguous buffer (as in
        ///     NumPy), so it owns its data.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="func1d"/> or <paramref name="arr"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="arr"/>'s rank (reports the original axis).</exception>
        /// <exception cref="ValueError">
        ///     Any non-axis (iteration) dimension is zero — <c>"Cannot apply_along_axis when any
        ///     iteration dimensions are 0"</c>, verbatim with NumPy — because there is then no first
        ///     slice to determine the result shape.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html
        ///     <para><b>Ownership.</b> Every array <paramref name="func1d"/> returns — and every slice handed to
        ///     it — stays the CALLER's: this method copies each result into its own buffer and never disposes
        ///     one, because a callback may return an array it keeps using (the slice itself, a held constant).
        ///     A <paramref name="func1d"/> that ALLOCATES a fresh result per slice therefore leaves those to a
        ///     GC — run the call inside an <see cref="NDScope"/> (yielding the result) when deterministic
        ///     release matters. This method's own buffer and views are released before it returns.</para>
        /// </remarks>
        public static NDArray apply_along_axis(Func<NDArray, NDArray> func1d, int axis, NDArray arr)
        {
            if (func1d is null)
                throw new ArgumentNullException(nameof(func1d));
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));

            int nd = arr.ndim;

            // normalize_axis_index: AxisError reports the ORIGINAL axis value (before wrapping).
            int ax = axis >= 0 ? axis : nd + axis;
            if (ax < 0 || ax >= nd)
                throw new AxisError(axis, nd);

            // Move the iteration axis to the end so each 1-D slice is the trailing dimension:
            // perm = [0..ax-1, ax+1..nd-1, ax]. transpose returns a view (no data movement).
            var perm = new int[nd];
            int p = 0;
            for (int i = 0; i < nd; i++)
                if (i != ax)
                    perm[p++] = i;
            perm[nd - 1] = ax;
            NDArray inarr = np.transpose(arr, perm);

            // The iteration space is every dimension except the last (the sliced axis). For a 1-D
            // input this is the empty shape, over which ndindex yields exactly one (empty) coordinate.
            long[] inDims = inarr.Shape.dimensions;
            int leadingLen = nd - 1;
            var leading = new long[leadingLen];
            Array.Copy(inDims, 0, leading, 0, leadingLen);

            // Drive ndindex manually: the FIRST coordinate is consumed before the buffer exists (its
            // func result fixes the buffer's shape+dtype), the rest fill the buffer. When a leading
            // dimension is 0 the sequence is empty — there is no first slice — which is exactly the
            // condition NumPy rejects.
            IEnumerator<long[]> inds = np.ndindex(leading).GetEnumerator();
            if (!inds.MoveNext())
                throw new ValueError("Cannot apply_along_axis when any iteration dimensions are 0");

            long[] ind0 = inds.Current;
            NDArray res0 = InvokeFunc1d(func1d, inarr, ind0);

            int resNd = res0.ndim;
            long[] resDims = res0.Shape.dimensions ?? Array.Empty<long>();

            // Buffer laid out so each write is contiguous: leading dims first, result dims trailing.
            // Allocated at the FIRST result's dtype (a DType — NumPy's zeros_like(res, shape=...)).
            var bufDims = new long[leadingLen + resNd];
            Array.Copy(leading, 0, bufDims, 0, leadingLen);
            Array.Copy(resDims, 0, bufDims, leadingLen, resNd);
            NDArray buff = np.zeros(new Shape(bufDims), res0.dtype);

            // Because ndindex walks the leading dims in C-order and `buff` is C-contiguous, the k-th
            // slice's slot is exactly the k-th contiguous block of `buff`'s bytes — NumPy's "each
            // write is contiguous" layout. When a result matches that slot verbatim (same dtype,
            // same shape, contiguous) we blit its bytes straight into the block, skipping the
            // per-slice destination view AND copyto's broadcast/cast setup (the dominant cost for a
            // tiny slice with a fast func); anything else falls back to the view+copyto path, which
            // reproduces NumPy's broadcast-on-assign and cast for a differing result. The span form
            // needs a 32-bit byte count, so it is gated on the whole buffer fitting int.MaxValue.
            long resBytes = res0.size * buff.dtypesize;
            bool useSpanWrite = buff.nbytes <= int.MaxValue;
            Span<byte> buffBytes = useSpanWrite ? buff.Unsafe.Bytes() : default;

            WriteSlot(buff, buffBytes, useSpanWrite, 0, ind0, res0, resDims, resBytes);
            long k = 1;
            while (inds.MoveNext())
            {
                long[] ind = inds.Current;
                WriteSlot(buff, buffBytes, useSpanWrite, k, ind, InvokeFunc1d(func1d, inarr, ind), resDims, resBytes);
                k++;
            }

            // Permute the buffer so the result dims land back at `axis`:
            //   buff_permute = [0:axis] + [buffNd-resNd : buffNd] + [axis : buffNd-resNd]
            // yielding shape arr.shape[:axis] + res.shape + arr.shape[axis+1:].
            int buffNd = leadingLen + resNd;
            var buffPermute = new int[buffNd];
            int q = 0;
            for (int i = 0; i < ax; i++)
                buffPermute[q++] = i;
            for (int i = buffNd - resNd; i < buffNd; i++)
                buffPermute[q++] = i;
            for (int i = ax; i < buffNd - resNd; i++)
                buffPermute[q++] = i;

            // Ownership: this method may release ONLY what it created and never handed to func1d — the
            // `buff` wrapper (the returned transpose view holds its own reference, so the buffer lives on)
            // and the `inarr` view (each slice was re-wrapped with its own reference). It must NOT scope or
            // dispose func1d's results or the slices passed to it: a callback may return or retain arrays
            // the caller still owns (a held array, the slice itself), so their lifetime stays the caller's.
            // Without the release, every call stranded `buff`'s reference — one pooled buffer until a GC.
            var result = np.transpose(buff, buffPermute);
            buff.Dispose();
            inarr.Dispose();
            return result;
        }

        /// <summary>
        ///     The <c>*args</c> overload — forwards extra positional arguments to every
        ///     <paramref name="func1d"/> call, mirroring NumPy's <c>apply_along_axis(func1d, axis,
        ///     arr, *args, **kwargs)</c>. Wraps the two-argument delegate into the single-argument
        ///     form; a call with no extra arguments passes an empty array, exactly as NumPy passes an
        ///     empty <c>*args</c>.
        /// </summary>
        /// <param name="func1d">Applied as <c>func1d(slice, args)</c> to each 1-D slice.</param>
        /// <param name="axis">Axis to slice along; negative counts from the end.</param>
        /// <param name="arr">Input array.</param>
        /// <param name="args">Extra positional arguments forwarded to every <paramref name="func1d"/> call.</param>
        /// <returns>See the primary overload.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="func1d"/> or <paramref name="arr"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="arr"/>'s rank.</exception>
        /// <exception cref="ValueError">Any iteration dimension is zero.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html</remarks>
        public static NDArray apply_along_axis(Func<NDArray, object[], NDArray> func1d, int axis, NDArray arr, params object[] args)
        {
            if (func1d is null)
                throw new ArgumentNullException(nameof(func1d));

            // args is never null for a params call (it is at worst an empty array), so the captured
            // closure hands func1d the same object every slice — no per-call allocation.
            return apply_along_axis(slice => func1d(slice, args), axis, arr);
        }

        /// <summary>
        ///     Invokes <paramref name="func1d"/> on the 1-D slice at leading coordinate
        ///     <paramref name="ind"/> and guards a null result (NumPy would build a 0-d object array
        ///     from <c>None</c>, a dtype NumSharp has no analog for — a null here is a caller bug, so
        ///     it is surfaced rather than allowed to NRE deep in the buffer allocation).
        /// </summary>
        /// <param name="func1d">The per-slice function.</param>
        /// <param name="inarr">The axis-last transposed view of the input.</param>
        /// <param name="ind">Leading coordinate (over the non-axis dims); empty for a 1-D input.</param>
        /// <returns>The func's result as an <see cref="NDArray"/>.</returns>
        /// <exception cref="ValueError"><paramref name="func1d"/> returned null.</exception>
        private static NDArray InvokeFunc1d(Func<NDArray, NDArray> func1d, NDArray inarr, long[] ind)
        {
            // GetData(ind) with (ndim-1) coordinates returns the trailing 1-D slice (a writeable view
            // sharing storage — read-only use here); an empty `ind` (1-D input) returns the whole 1-D
            // array. This is NumPy's inarr_view[ind + (Ellipsis,)] without the 0-d-decay hazard, since
            // GetData never collapses to a scalar.
            NDArray slice = inarr.GetData(ind);
            NDArray res = func1d(slice);
            if (res is null)
                throw new ValueError("apply_along_axis: func1d returned null");
            return res;
        }

        /// <summary>
        ///     Writes one func result into the buffer's k-th contiguous slot. FAST PATH (when
        ///     <paramref name="useSpanWrite"/> and the result matches the slot verbatim — same dtype
        ///     as the buffer, same shape as the first result, C-contiguous): blit the result's raw
        ///     bytes into the block at <c>k * resBytes</c>, no destination view, no cast/broadcast
        ///     setup. FALLBACK (any other result — a differing shape, dtype, or non-contiguous layout,
        ///     or a buffer too large for a 32-bit span): the trailing result-shaped sub-view
        ///     <c>buff.GetData(ind)</c> plus <see cref="copyto"/>, which reproduces NumPy's
        ///     <c>buff[ind] = res</c> broadcast-on-assign and unsafe cast exactly.
        /// </summary>
        /// <param name="buff">The contiguous output buffer (leading dims + result dims).</param>
        /// <param name="buffBytes">The whole buffer as bytes (valid only when <paramref name="useSpanWrite"/>).</param>
        /// <param name="useSpanWrite">True if the buffer fits a 32-bit span, enabling the blit fast path.</param>
        /// <param name="k">The linear slot index (C-order over the leading dims) — the fast-path block offset.</param>
        /// <param name="ind">Leading coordinate; empty targets the whole buffer (1-D input). Used only by the fallback.</param>
        /// <param name="res">The func result to store.</param>
        /// <param name="resDims">The first result's dimensions — the slot shape the fast path requires a verbatim match to.</param>
        /// <param name="resBytes">Bytes per slot (<c>res0.size * buff.dtypesize</c>).</param>
        private static void WriteSlot(NDArray buff, Span<byte> buffBytes, bool useSpanWrite, long k,
            long[] ind, NDArray res, long[] resDims, long resBytes)
        {
            // Blit only when the result IS the slot: matching the first result's dtype and shape means
            // no cast and no broadcast, and C-contiguity means its bytes are already in slot order.
            if (useSpanWrite && res.dtype == buff.dtype && res.Shape.IsContiguous && SameDims(res.Shape.dimensions, resDims))
            {
                res.Unsafe.ReadOnlyBytes().CopyTo(buffBytes.Slice((int)(k * resBytes), (int)resBytes));
                return;
            }

            NDArray dst = buff.GetData(ind);
            np.copyto(dst, res, casting: "unsafe");
        }

        /// <summary>
        ///     Compares two dimension arrays for exact equality (rank and every extent). Used to
        ///     decide whether a func result matches the buffer slot verbatim — a mismatch routes to
        ///     the broadcast/cast fallback so a differing result behaves like NumPy's item assignment.
        /// </summary>
        /// <param name="a">First dimension array (a result's dimensions; may be null for a 0-d result).</param>
        /// <param name="b">Second dimension array (the first result's dimensions).</param>
        /// <returns>True if both are the same rank with identical extents (two 0-d/empty are equal).</returns>
        private static bool SameDims(long[] a, long[] b)
        {
            int an = a?.Length ?? 0;
            int bn = b?.Length ?? 0;
            if (an != bn)
                return false;
            for (int i = 0; i < an; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }
    }
}
