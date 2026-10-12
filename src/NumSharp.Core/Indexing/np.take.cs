using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Take elements from an array along an axis. Equivalent to fancy
        ///     indexing along the specified axis.
        /// </summary>
        /// <param name="a">Source array.</param>
        /// <param name="indices">Integer array of indices to take (any layout; read in C order).</param>
        /// <param name="axis">
        ///     Axis along which to take. <c>null</c> (default) flattens <paramref name="a"/>
        ///     and treats <paramref name="indices"/> as flat indices. A 0-d <paramref name="a"/> is flattened too, so its
        ///     axis must be valid for one dimension (0 or -1).
        /// </param>
        /// <param name="out">
        ///     Optional destination array. When supplied, its shape must match the
        ///     natural take output; values are cast to <paramref name="out"/>'s dtype
        ///     via <see cref="np.copyto"/> with unsafe casting and the method returns
        ///     <paramref name="out"/> itself. When <c>null</c> (default), a fresh
        ///     array is allocated with <paramref name="a"/>'s dtype.
        /// </param>
        /// <param name="mode">
        ///     Boundary mode: <c>"raise"</c> (default — throw on OOB), <c>"wrap"</c>
        ///     (modulo with sign correction), or <c>"clip"</c> (saturate); <c>null</c> is NumPy's <c>None</c> (raise).
        /// </param>
        /// <returns>
        ///     New array with shape:
        ///     <list type="bullet">
        ///       <item><c>axis=None</c>: same as <paramref name="indices"/>.</item>
        ///       <item><c>axis=k</c>: <c>a.shape[:k] + indices.shape + a.shape[k+1:]</c>.</item>
        ///     </list>
        ///     Dtype matches <paramref name="a"/> (or <paramref name="out"/>'s dtype
        ///     when <paramref name="out"/> is supplied).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="indices"/> is null.</exception>
        /// <exception cref="ValueError">A bad <paramref name="mode"/> (<see cref="ClipModeConverter"/>), checked first;
        ///     an <paramref name="out"/> of another shape (<c>output array does not match result of ndarray.take</c>) or a
        ///     read-only one (<c>WRITEBACKIFCOPY base is read-only</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">The index dtype does not cast to int64 under <c>same_kind</c> (float, complex),
        ///     an empty index array included; or <paramref name="out"/>'s dtype does not cast to <paramref name="a"/>'s
        ///     under <c>safe</c>.</exception>
        /// <exception cref="IndexError">A non-empty result from an empty axis (<c>cannot do a non-empty take from an empty
        ///     axes.</c>), or an index out of bounds under <c>mode="raise"</c> (<c>index 9 is out of bounds for axis 0 with
        ///     size 4</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.take.html
        ///     <para>The work is <see cref="TakeFrom"/>, NumPy's <c>PyArray_TakeFrom</c>, which also fixes the order of the
        ///     checks above.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // reclaims the int64 index copy, the C-order source copy and the natural-result temp on the out= path
        public static NDArray take(NDArray a, NDArray indices, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, indices, axis, @out, mode), static s => take(s.a, s.indices, s.axis, s.@out, s.mode), @out);
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            // array_take converts the mode (and parses the axis) with its arguments, before PyArray_TakeFrom runs.
            int modeInt = ClipModeConverter(mode);
            return TakeFrom(a, indices, axis, @out, modeInt);
        }

        /// <summary>
        ///     Scalar convenience overload — take a single element by flat index.
        /// </summary>
        /// <param name="a">Source array.</param>
        /// <param name="index">The index (a 0-d index: the result has the shape of the taken element).</param>
        /// <param name="axis">As in <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</param>
        /// <param name="out">As in <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</param>
        /// <param name="mode">As in <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</param>
        /// <returns>The taken element(s).</returns>
        /// <exception cref="ValueError">See <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</exception>
        /// <exception cref="AxisError">See <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</exception>
        /// <exception cref="IndexError">See <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>.</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // the 0-d index wrapper is built in THIS frame, before the array overload's scope opens
        public static NDArray take(NDArray a, long index, int? axis = null, NDArray @out = null, string mode = "raise")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, index, axis, @out, mode), static s => take(s.a, s.index, s.axis, s.@out, s.mode), @out);
            // Wrap the scalar in a 0-d NDArray so the array overload's shape-preserving
            // logic emits a 0-d result (NumPy semantic for scalar input).
            var idxArr = NDArray.Scalar(index);
            return take(a, idxArr, axis, @out, mode);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_TakeFrom</c> (<c>item_selection.c</c>), statement for statement: the axis check, the index
        ///     cast, the result shape, the <paramref name="out"/> checks, the empty-axis check and the gather.
        /// </summary>
        /// <param name="a">The source array, any layout (a non-contiguous source is first copied to C order).</param>
        /// <param name="indices">The integer index array; a C-contiguous int32/int64 one is read in place.</param>
        /// <param name="axis">The caller's axis, or null for a flat take.</param>
        /// <param name="out">The caller's destination, or null.</param>
        /// <param name="mode">The parsed out-of-bounds policy (<see cref="ClipModeConverter"/>).</param>
        /// <returns>The taken values: <paramref name="out"/> itself when given, else a new C-contiguous array of
        ///     <paramref name="a"/>'s dtype.</returns>
        /// <exception cref="AxisError">An out-of-range <paramref name="axis"/>.</exception>
        /// <exception cref="TypeError">An index dtype that does not cast to int64 under <c>same_kind</c>, or an
        ///     <paramref name="out"/> dtype that does not cast to <paramref name="a"/>'s under <c>safe</c>.</exception>
        /// <exception cref="ValueError">An <paramref name="out"/> of another shape, or a read-only one.</exception>
        /// <exception cref="IndexError">A non-empty result from an empty axis, or an out-of-bounds index under
        ///     raise.</exception>
        /// <exception cref="NotSupportedException">No IL kernel exists for the element width.</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's order (probed 2.4.2): <c>PyArray_CheckAxis</c> first — <c>axis=None</c> and a 0-d source are
        ///     raveled, so a 0-d source takes axis 0 or -1 only (<c>np.take(np.array(5), [0], axis=1)</c> is the AxisError
        ///     "axis 1 is out of bounds for array of dimension 1") —; the index cast
        ///     (<c>PyArray_FromAny(indices, intp, SAME_KIND_CASTING)</c>, an empty float array too); the <c>out</c> shape,
        ///     its safe cast and its writeability (<c>PyArray_FromArray(out, a.dtype, WRITEBACKIFCOPY)</c>); then
        ///     <c>max_item == 0 &amp;&amp; size != 0</c>; then the gather, whose raise loop checks every index of every
        ///     outer position even when the slabs are empty (<c>np.take(np.zeros((3, 0, 0)), [0], axis=1)</c> is "index 0
        ///     is out of bounds for axis 1 with size 0", while an empty OUTER block checks nothing).
        ///     </para>
        ///     <para>
        ///     Covered by <c>[NDScopedCovered]</c>: the only callers are the <c>[NDScoped]</c> <c>take</c> overloads, so the
        ///     C-order copy of a non-contiguous source, the int64 copy of the indices and the natural result of an
        ///     <paramref name="out"/> call are released by that caller's scope, exceptions included. The caller's
        ///     <paramref name="a"/>, <paramref name="indices"/> and <paramref name="out"/> were built before any scope
        ///     opened, so an operand used as-is is never disposed.
        ///     </para>
        /// </remarks>
        [NDScopedCovered]
        private static NDArray TakeFrom(NDArray a, NDArray indices, int? axis, NDArray @out, int mode)
        {
            // PyArray_CheckAxis(self, &axis, CARRAY_RO): axis=None ravels the source, and so does a 0-d source (NumPy
            // ravels it to one dimension before it checks a given axis against that ONE dimension).
            bool ravel = axis is null || a.ndim == 0;
            int ax = 0;
            if (axis is not null)
            {
                int n = ravel ? 1 : a.ndim;
                ax = axis.Value;
                if (ax < -n || ax >= n)
                    throw new AxisError(axis.Value, n);
                if (ax < 0)
                    ax += n;
            }

            // PyArray_FromAny(indices0, intp, SAME_KIND_CASTING | DEFAULT): the cast rule is checked before the out
            // array and before the empty-axis check. A contiguous int32 array is read in place; any other cast is made in
            // C order (the kernels read the indices linearly).
            var idx64 = CastIndicesToInt64(indices, "same_kind", out _, out bool idx32);

            // The result shape: indices.shape for a raveled source, else a.shape[:axis] + indices.shape +
            // a.shape[axis+1:]; outer x indices x inner is the gather's factorisation around the axis.
            long outerSize = 1, maxItem, innerCount = 1;
            long[] outDims;
            if (ravel)
            {
                maxItem = a.size;
                outDims = (long[])indices.Shape.dimensions.Clone();
            }
            else
            {
                var dims = a.Shape.dimensions;
                for (int d = 0; d < ax; d++) outerSize *= dims[d];
                maxItem = dims[ax];
                for (int d = ax + 1; d < a.ndim; d++) innerCount *= dims[d];

                outDims = new long[a.ndim + indices.ndim - 1];
                int o = 0;
                for (int d = 0; d < ax; d++) outDims[o++] = dims[d];
                for (int d = 0; d < indices.ndim; d++) outDims[o++] = indices.Shape.dimensions[d];
                for (int d = ax + 1; d < a.ndim; d++) outDims[o++] = dims[d];
            }

            if (@out is not null)
                ValidateTakeOut(a, @out, outDims);

            if (a.Storage.IsVString)
            {
                // A StringDType source, after every check NumPy runs before its gather: the same take runs on the source's
                // cell view (it repeats those checks and raises the empty-axis and bounds errors with this array's shapes),
                // and the picked cells are duplicated into a fresh buffer (NDVStringMove). Its out= gets the strings through
                // the StringDType copy core (np.copyto, unsafe — the cast check above already ran).
                var taken = NDVStringMove.Gather(a, cells => TakeFrom(cells, indices, axis, null, mode));
                if (@out is null)
                    return taken;
                np.copyto(@out, taken, casting: "unsafe");
                return @out;
            }

            var result = new NDArray(a.dtype, Shape.Adopt(outDims), false); // the descriptor: a string's width
            long indicesCount = indices.size;

            // "Index error, since that is the usual error for raise mode" — only when something would be written.
            if (maxItem == 0 && result.size != 0)
                throw new IndexError("cannot do a non-empty take from an empty axes.");

            // npy_fasttake's raise loop checks indices[j] for every outer position i even when a slab is empty (chunk 0),
            // so an empty result still raises for an out-of-bounds index unless the outer block or the index list is
            // empty. Wrap and clip modes write nothing there and never fail.
            if (outerSize != 0 && indicesCount != 0)
            {
                if (result.size != 0)
                {
                    // Materialise a non-contiguous source to C order so the kernel can walk it linearly.
                    NDArray source = a.Shape.IsContiguous ? a : np.ascontiguousarray(a);
                    ExecuteTakeKernel(source, idx64, result,
                        outerSize: outerSize, indicesCount: indicesCount,
                        maxItem: maxItem, innerSize: innerCount * a.dtypesize, mode: mode, idx32: idx32,
                        axis: ravel ? null : ax);   // axis form: the raise-mode IndexError names this axis
                }
                else if (mode == ClipModeRaise)
                {
                    ThrowIfTakeIndexOutOfBounds(idx64, idx32, indicesCount, maxItem, ravel ? 0 : ax);
                }
            }

            if (@out is null)
                return result;

            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }

        /// <summary>
        ///     <c>PyArray_TakeFrom</c>'s three checks of a caller's <paramref name="out"/>, in NumPy's order: the shape (dims
        ///     only, any layout), the safe cast of <paramref name="out"/>'s dtype to <paramref name="a"/>'s
        ///     (<c>PyArray_FromArray(out, a.dtype, CARRAY | WRITEBACKIFCOPY)</c> casts the result back into it), and the
        ///     writeability of the write-back base.
        /// </summary>
        /// <param name="a">The source, whose dtype the gather writes.</param>
        /// <param name="out">The caller's destination.</param>
        /// <param name="outDims">The natural result's dimensions.</param>
        /// <exception cref="ValueError">Another shape (<c>output array does not match result of ndarray.take</c>), or a
        ///     read-only <paramref name="out"/> (<c>WRITEBACKIFCOPY base is read-only</c>).</exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast to <paramref name="a"/>'s under
        ///     <c>safe</c>.</exception>
        private static void ValidateTakeOut(NDArray a, NDArray @out, long[] outDims)
        {
            // PyArray_CompareLists(PyArray_DIMS(out), shape, nd): the dimensions alone, NumPy's text verbatim.
            var dims = @out.Shape.dimensions ?? Array.Empty<long>();
            bool sameShape = dims.Length == outDims.Length;
            for (int i = 0; sameShape && i < dims.Length; i++)
                sameShape = dims[i] == outDims[i];
            if (!sameShape)
                throw new ValueError("output array does not match result of ndarray.take");

            if (a.typecode == NPTypeCode.String || @out.typecode == NPTypeCode.String)
            {
                // A string's width is part of its dtype: the rule is checked between DESCRIPTORS (a <U2 out of a <U4
                // take is fine — <U2 → <U4 is safe — and the unsafe write-back truncates, as NumPy's does).
                if (!@out.HasSameElementType(a) && !DTypeCasting.CanCastTypeTo(@out.dtype, a.dtype, NumSharp.Backends.Iteration.NPY_CASTING.NPY_SAFE_CASTING))
                    throw new TypeError(
                        $"Cannot cast array data from {@out.dtype.ToString(true)} to {a.dtype.ToString(true)} according to the rule 'safe'");
            }
            else if (@out.GetTypeCode != a.GetTypeCode &&
                !np.can_cast(@out.GetTypeCode, a.GetTypeCode, "safe"))
            {
                throw new TypeError(
                    $"Cannot cast array data from dtype('{@out.GetTypeCode.AsNumpyDtypeName()}') to dtype('{a.GetTypeCode.AsNumpyDtypeName()}') according to the rule 'safe'");
            }

            // A read-only out refuses LAST of the three (probed 2.4.2: shape, then castability, then this) — with take's
            // own text: the writeability check of PyArray_FromArray(..., WRITEBACKIFCOPY) names the writeback base, not
            // the "output array" the elementwise ufuncs name.
            NumSharpException.ThrowIfNotWriteable(@out.Shape, "WRITEBACKIFCOPY base");
        }

        /// <summary>
        ///     The bounds check of <c>npy_fasttake</c>'s raise loop for a take whose slabs are empty: every index is still
        ///     checked against the axis length, as NumPy checks it before it copies the (empty) slab.
        /// </summary>
        /// <param name="idx64">The C-contiguous indices (int64, or int32 when <paramref name="idx32"/>).</param>
        /// <param name="idx32">True when <paramref name="idx64"/> holds int32 values.</param>
        /// <param name="count">The number of indices.</param>
        /// <param name="maxItem">The axis length.</param>
        /// <param name="axis">The axis NumPy names (0 for a flat take).</param>
        /// <exception cref="IndexError">The first index outside <c>[-maxItem, maxItem)</c>, NumPy's text.</exception>
        private static unsafe void ThrowIfTakeIndexOutOfBounds(NDArray idx64, bool idx32, long count, long maxItem, int axis)
        {
            long* p64 = (long*)idx64.Storage.Address + idx64.Shape.offset;
            int* p32 = (int*)idx64.Storage.Address + idx64.Shape.offset;
            for (long j = 0; j < count; j++)
            {
                long v = idx32 ? p32[j] : p64[j];
                // check_and_adjust_index: one negative wrap, then [0, maxItem).
                long adjusted = v < 0 ? v + maxItem : v;
                if (adjusted < 0 || adjusted >= maxItem)
                    throw new IndexError($"index {v} is out of bounds for axis {axis} with size {maxItem}");
            }
        }

        /// <summary>
        ///     Gathered-region footprint above which the take kernel emits software prefetch. Sized just
        ///     over typical per-core L2 (a 400 KB / 100 K-element source stays cache-resident and must
        ///     take the lean, prefetch-free kernel); a 40 MB / 10 M source clears it and gets prefetch.
        /// </summary>
        private const long IndexPrefetchThresholdBytes = 2L * 1024 * 1024;

        /// <summary>
        ///     Run the compiled take gather over <paramref name="source"/> into
        ///     <paramref name="result"/>, choosing between the lean flat kernel and the general
        ///     outer×inner kernel, and — under mode 'raise' — converting a kernel-reported
        ///     out-of-bounds index into NumPy's exact IndexError wording.
        /// </summary>
        /// <param name="source">The gathered array (read through raw pointer + offset).</param>
        /// <param name="idx64">The contiguous index array (int64 storage; reinterpreted as int32
        ///     values when <paramref name="idx32"/> is true).</param>
        /// <param name="result">Preallocated C-contiguous destination the kernel fills.</param>
        /// <param name="outerSize">Product of the dims before the take axis (1 for a flat take).</param>
        /// <param name="indicesCount">Number of indices gathered per outer position.</param>
        /// <param name="maxItem">The take axis's extent — the bound a raise-mode index must satisfy
        ///     after one negative wrap.</param>
        /// <param name="innerSize">Bytes per gathered slab (element bytes × trailing dims).</param>
        /// <param name="mode">0 = raise, 1 = wrap, 2 = clip (NumPy's mode order).</param>
        /// <param name="idx32">True when the index buffer holds int32 values read in place.</param>
        /// <param name="axis">The take axis as the CALLER spelled it, or null for a flat
        ///     (axis=None) take — load-bearing for the error text only: NumPy's raise-mode message
        ///     names the axis ("for axis 0 with size 8") on an axis take but drops the clause
        ///     entirely ("for size 8") on a flat one.</param>
        /// <exception cref="NotSupportedException">No IL kernel exists for this element width
        ///     (dynamic codegen unavailable).</exception>
        /// <exception cref="IndexError">Mode 'raise' met an index still out of
        ///     [0, <paramref name="maxItem"/>) after the single negative wrap — NumPy's IndexError, message per
        ///     <paramref name="axis"/> above, and nothing has been partially written that the
        ///     caller exposes (the result buffer is discarded on throw).</exception>
        private static unsafe void ExecuteTakeKernel(
            NDArray source, NDArray idx64, NDArray result,
            long outerSize, long indicesCount, long maxItem, long innerSize, int mode, bool idx32 = false,
            int? axis = null)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(innerSize);
            // Software prefetch pays off only when the randomly-gathered region misses cache; below that
            // threshold it is pure per-element overhead (measured ~1.6x slower at 100K). Gate on the
            // gathered footprint = maxItem slabs of innerSize bytes.
            bool prefetch = maxItem * innerSize > IndexPrefetchThresholdBytes;

            byte* srcPtr = (byte*)source.Storage.Address + source.Shape.offset * source.dtypesize;
            // idx32: the index buffer holds int32 values; the kernels' `long*` parameter is a reinterpret.
            long* idxPtr = idx32
                ? (long*)((int*)idx64.Storage.Address + idx64.Shape.offset)
                : (long*)idx64.Storage.Address + idx64.Shape.offset;
            byte* dstPtr = (byte*)result.Storage.Address;

            long status;
            long expected = outerSize * indicesCount;
            // The lean flat gather (DirectILKernelGenerator.GatherFlat.cs) serves the single-slab-per-
            // index RAISE case — axis=None, or axis=0 with a primitive-width trailing slab — with a
            // compile-time element width and no per-element mode dispatch; wrap/clip and multi-outer
            // gathers keep the general kernel.
            var flat = (mode == 0 && outerSize == 1 && copyKind != 0)
                ? DirectILKernelGenerator.GetTakeFlatKernel(copyKind, idx32, prefetch)
                : null;
            if (flat != null)
            {
                status = flat(srcPtr, idxPtr, indicesCount, maxItem, dstPtr);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetTakeKernel(copyKind, prefetch, idx32);
                if (kernel == null)
                    throw new NotSupportedException("np.take: IL kernel unavailable");
                status = kernel(srcPtr, idxPtr, indicesCount, outerSize,
                                maxItem, innerSize, mode, dstPtr);
            }
            if (status < expected)
            {
                long failPair = status;
                long badJ = failPair % indicesCount;
                long badVal = idx32 ? ((int*)idxPtr)[badJ] : idxPtr[badJ];
                // NumPy's IndexError text, verbatim (probed 2.4.2): the flat (axis=None) take runs on the raveled
                // array, so it names axis 0 too ("index 7 is out of bounds for axis 0 with size 5"). The
                // axis-less "for size {n}" spelling belongs to a.flat[i], not to take.
                throw new IndexError($"index {badVal} is out of bounds for axis {axis ?? 0} with size {maxItem}");
            }
        }

        /// <summary>
        ///     Casts <paramref name="indices"/> to contig int64 if needed. Sets
        ///     <paramref name="owned"/> to true when the returned array is a fresh
        ///     allocation that the caller must Dispose. The result shape matches
        ///     the input shape but always has contig strides.
        ///     <para>
        ///     NumPy converts an index array to <c>intp</c> under a fixed casting rule and raises
        ///     <see cref="TypeError"/> when it cannot — an integer or boolean index array is fine, but a
        ///     float/complex one is rejected rather than silently truncated. The rule differs by caller:
        ///     <c>take</c> uses <c>"same_kind"</c> (which also permits <c>uint64</c>) while <c>put</c>
        ///     uses <c>"safe"</c> (which rejects <c>uint64</c> too), and each leaks its own rule name in
        ///     the message — so <paramref name="castingRule"/> is threaded through verbatim.
        ///     </para>
        /// </summary>
        private static NDArray CastIndicesToInt64(NDArray indices, string castingRule, out bool owned)
            => CastIndicesToInt64(indices, castingRule, out owned, out _);

        /// <summary>
        ///     As <see cref="CastIndicesToInt64(NDArray, string, out bool)"/>, but a C-contiguous
        ///     <b>int32</b> index array is returned AS-IS with <paramref name="idx32"/> set: the take/put
        ///     kernels read int32 indices in place (their <c>idx32</c> variants), so the common C# index
        ///     (<c>np.array(new int[] {…})</c> is int32) no longer pays a widening copy — measured 27 % of
        ///     <c>np.take(a, idx32)</c> at 100K.
        /// </summary>
        private static NDArray CastIndicesToInt64(NDArray indices, string castingRule, out bool owned, out bool idx32)
        {
            var tc = indices.GetTypeCode;
            idx32 = false;

            if (tc == NPTypeCode.Int64 && indices.Shape.IsContiguous)
            {
                owned = false;
                return indices;
            }

            if (tc == NPTypeCode.Int32 && indices.Shape.IsContiguous)
            {
                owned = false;
                idx32 = true;
                return indices;
            }

            if (tc == NPTypeCode.Int64)
            {
                var c = np.ascontiguousarray(indices);
                owned = !ReferenceEquals(c, indices);
                return c;
            }

            // Text never reaches int64 under take's same_kind or put's safe rule; the descriptor names its width (the
            // storage lane alone cannot answer a cast question).
            if (tc == NPTypeCode.String)
                throw new TypeError(
                    $"Cannot cast array data from {indices.dtype.ToString(true)} to dtype('int64') according to the rule '{castingRule}'");
            if (!np.can_cast(tc, NPTypeCode.Int64, castingRule))
                throw new TypeError(
                    $"Cannot cast array data from dtype('{tc.AsNumpyDtypeName()}') to dtype('int64') according to the rule '{castingRule}'");

            // C order: the take / put kernels read the indices linearly, and astype's default 'K' keeps an F-order
            // input F-ordered — np.take(src, int16 F-order [[3, 2], [1, 0]]) read [3, 1, 2, 0] (NumPy: [3, 2, 1, 0]).
            owned = true;
            return indices.astype(NPTypeCode.Int64, copy: true, order: 'C');
        }
    }
}
