using System;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    // ---------------------------------------------------------------------------------------------------------------------
    // NumPy's item_selection.c (PyArray_TakeFrom, PyArray_PutTo, PyArray_PutMask, PyArray_Repeat, PyArray_Choose,
    // PyArray_Compress), compiled_base.c's arr_place and fromnumeric.py's resize: the memory work under np.take, np.put,
    // np.putmask, np.place, np.repeat, np.choose, np.compress and np.resize and their ndarray methods. They are the array
    // object's own members (the np-layer dtype plan's D1): positions select and assign elements, nothing computes a
    // value, so no TensorEngine member is involved and nothing overrides them. The np.* functions are facades over them:
    // they keep the class boundaries (numpy.matrix, numpy.char.chararray), the C# null checks and the conversions of
    // their C# argument forms.
    //
    // The file sits beside its facades rather than under Backends/Unmanaged/ because its statements still call np.*
    // functions (np.copyto, np.ascontiguousarray, np.can_cast, np.flatnonzero, ...), which ratchet R2 forbids under
    // Backends/ (plan C3); P6 moves it down once those calls are gone. The IL kernels stay in Backends/Kernels/.
    // ---------------------------------------------------------------------------------------------------------------------
    public partial class UnmanagedStorage
    {
        // ---- np.take: PyArray_TakeFrom ----------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_TakeFrom</c> (<c>item_selection.c</c>), statement for statement: the axis check, the index
        ///     cast, the result shape, the <paramref name="out"/> checks, the empty-axis check and the gather.
        /// </summary>
        /// <param name="a">The source array, any layout (a non-contiguous source is first copied to C order).</param>
        /// <param name="indices">The integer index array; a C-contiguous int32/int64 one is read in place.</param>
        /// <param name="axis">The caller's axis, or null for a flat take.</param>
        /// <param name="out">The caller's destination, or null.</param>
        /// <param name="mode">The parsed out-of-bounds policy (<see cref="np.ClipmodeConverter"/>).</param>
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
        internal static NDArray TakeFrom(NDArray a, NDArray indices, int? axis, NDArray @out, int mode)
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

            // npy_fasttake's needs_refcounting, PyDataType_REFCHK(PyArray_DESCR(self)): the items of a reference-holding
            // dtype (StringDType is the only one) are copied by the dtype, never as raw bytes.
            if (a.dtype.Meta.HasReferences)
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

            // The descriptor allocates (a string's width). The gather writes every element before the result is returned (an
            // out-of-bounds index raises and the scope reclaims the partial result), so a U result skips its zero fill.
            var result = NDArray.ForOverwrite(a.dtype, Shape.Adopt(outDims));
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
                else if (mode == np.ClipModeRaise)
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
            // A constant slab width up to MaxFixedCopyBytes (every primitive element, a string element) is baked into the
            // kernel; a wider slab keeps the runtime-sized cpblk.
            int copyKind = DirectILKernelGenerator.FixedCopyKindFor(innerSize);
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

        // ---- np.put: PyArray_PutTo --------------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_PutTo</c> (<c>item_selection.c</c>) behind <c>ndarray.put</c> and <c>np.put</c>, with
        ///     <c>array_put</c>'s clip-mode conversion, statement for statement. numpy.ma's <c>put</c> calls it for the data
        ///     and for the mask (<c>self._data.put</c> / <c>m.put</c>).
        /// </summary>
        /// <param name="a">The target, written in place.</param>
        /// <param name="indices">The flat indices: any integer or bool array that casts to int64 under <c>safe</c>, any
        ///     layout (read in C order).</param>
        /// <param name="values">The values: an <see cref="NDArray"/> (a C# literal's marked 0-d array is packed as the
        ///     Python scalar it stands for), a boxed C# number (packed by its kind), or anything
        ///     <see cref="NDArray.AsStoreValue"/> converts. Read in C order and cycled.</param>
        /// <param name="mode">The clip mode string (<see cref="np.ClipmodeConverter"/>).</param>
        /// <exception cref="ValueError">A bad <paramref name="mode"/>; then a read-only <paramref name="a"/>.</exception>
        /// <exception cref="TypeError">An index dtype that does not cast to int64 under <c>safe</c>.</exception>
        /// <exception cref="IndexError">A non-empty <paramref name="indices"/> into an empty <paramref name="a"/>, or an
        ///     out-of-bounds index under raise.</exception>
        /// <exception cref="OverflowException">A C# literal out of <paramref name="a"/>'s range.</exception>
        /// <exception cref="NotSupportedException">No IL put kernel exists for the element width.</exception>
        /// <remarks>
        ///     <para>
        ///     The order is NumPy's: the mode (parsed with <c>array_put</c>'s arguments), the writeability, the index cast
        ///     (<c>PyArray_ContiguousFromAny(indices, NPY_INTP)</c>, so an empty float index array is the TypeError too), the
        ///     empty-target IndexError, the value conversion (<c>PyArray_FromAny(values, a.dtype, DEFAULT | FORCECAST)</c>: a
        ///     literal is range-checked even when nothing is written), and only then the no-op of an empty value array.
        ///     </para>
        ///     <para>
        ///     NumPy writes into a copy of the target (<c>NPY_ARRAY_ENSURECOPY | NPY_ARRAY_WRITEBACKIFCOPY</c>) when the
        ///     target is not C-contiguous or when it may share memory with the converted values or indices
        ///     (<c>arrays_overlap</c>, a bounds check: <see cref="NDMemOverlap.SolveMayShareMemory"/> with
        ///     <c>maxWork: 0</c>), so a cyclic read of values that alias the target sees the ORIGINAL elements
        ///     (<c>np.put(a, [1, 2, 3], a[0:3])</c> shifts, <c>np.put(a, a, v)</c> reads the original indices). The copy is
        ///     written back only on success, as NumPy discards it on an error; a contiguous target without overlap is
        ///     written in place, so the indices before an out-of-bounds one are written (NumPy's raise loop writes as it
        ///     goes). An int32 index array NumSharp reads in place (NumPy copies it to intp) counts as overlap when it may
        ///     share the target's memory, which gives NumPy's answer: its values are read before any write.
        ///     </para>
        ///     <para>
        ///     Covered by <c>[NDScopedCovered]</c>: its callers are the <c>[NDScoped]</c> <c>np.put</c> overloads and
        ///     numpy.ma's <c>put</c>, which opens a scope of its own; every converted copy and the scratch target are
        ///     released by that scope, which also keeps them reachable while the kernel reads them through raw pointers.
        ///     </para>
        /// </remarks>
        [NDScopedCovered]
        internal static void PutTo(NDArray a, NDArray indices, object values, string mode)
        {
            // array_put converts `mode` with PyArray_ClipmodeConverter while it parses its arguments: a bad mode is
            // reported before PyArray_PutTo looks at the target.
            int modeInt = np.ClipmodeConverter(mode);

            // PyArray_FailUnlessWriteable(self, "put: output array") is PyArray_PutTo's first check (probed 2.4.2:
            // put(ro, [], []) and put(ro, [oob], v) report read-only).
            NumSharpException.ThrowIfNotWriteable(a.Shape, "put: output array");

            // PyArray_ContiguousFromAny(indices0, NPY_INTP): the 'safe' cast is checked for every index array, an empty
            // one too, before the empty target and the values (probed: an empty float index array raises TypeError).
            var idx64 = CastIndicesToInt64(indices, "safe", out _, out bool idx32);
            long ni = indices.size;
            if (ni > 0 && a.size == 0)
                throw new IndexError("cannot replace elements of an empty array");

            // PyArray_FromAny(values0, a.dtype, DEFAULT | FORCECAST), then `nv <= 0` returns without writing: a literal is
            // packed (range-checked) even when nothing is written (np.put(int8, [], 1000) raises).
            var vals = PutValuesLikeNumPy(values, a);
            if (vals.size == 0 || ni == 0)
                return;

            // overlap = arrays_overlap(self, values) || arrays_overlap(self, indices); with a non-contiguous target NumPy
            // also writes into a C-contiguous copy and writes it back.
            bool overlaps =
                NDMemOverlap.SolveMayShareMemory(a, vals, maxWork: 0) != MemOverlap.No ||
                NDMemOverlap.SolveMayShareMemory(a, idx64, maxWork: 0) != MemOverlap.No;
            // PyArray_PutTo's has_references, PyDataType_REFCHK(PyArray_DESCR(self)): a reference-holding target
            // (StringDType) receives its items through the dtype's copy, never as raw bytes.
            if (a.dtype.Meta.HasReferences)
            {
                // A StringDType target: the same put runs on marks (the values' element numbers, cycled by the numeric put
                // with its mode), and the marked elements receive the strings. NumPy writes a contiguous target that
                // overlaps nothing as it goes, so the elements before an out-of-bounds index stay written; a copy (a
                // non-contiguous target, or an overlap) is discarded on the error.
                Backends.Strings.NDVStringMove.ScatterVia(a, vals, (marks, index) => PutTo(marks, indices, index, mode),
                    keepPartialOnError: !overlaps && a.Shape.IsContiguous);
                return;
            }
            if (overlaps || !a.Shape.IsContiguous)
            {
                // A contiguous target is copied with copy() (ascontiguousarray would hand back the target itself); the
                // scratch is released by the caller's scope. A throw from the kernel skips the write-back: NumPy's
                // PyArray_DiscardWritebackIfCopy, so nothing reaches `a`.
                var scratch = a.Shape.IsContiguous ? a.copy() : np.ascontiguousarray(a);
                PutKernel(scratch, idx64, idx32, ni, vals, modeInt);
                np.copyto(a, scratch, casting: "unsafe");
                return;
            }

            PutKernel(a, idx64, idx32, ni, vals, modeInt);
        }

        /// <summary>
        ///     <c>PyArray_PutTo</c>'s value conversion, <c>PyArray_FromAny(values0, a.dtype, NPY_ARRAY_DEFAULT |
        ///     NPY_ARRAY_FORCECAST)</c>: the values as a C-contiguous array of <paramref name="a"/>'s element type.
        /// </summary>
        /// <param name="values">An <see cref="NDArray"/> or any value <see cref="NDArray.AsStoreValue"/> accepts.</param>
        /// <param name="a">The target, whose dtype the values take.</param>
        /// <returns>The caller's array when it already is C-contiguous of <paramref name="a"/>'s element type, else a new
        ///     array (released by the caller's scope).</returns>
        /// <exception cref="OverflowException">A C# literal out of <paramref name="a"/>'s range.</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer array.</exception>
        /// <exception cref="TypeError">A complex literal into a real array.</exception>
        /// <remarks>
        ///     The cast is made in C ORDER: the put kernels read the converted values linearly, and <c>astype</c>'s default
        ///     <c>'K'</c> keeps an F-order input F-ordered, which put an F-order int32 array's values into an int64 target
        ///     in memory order (<c>[10, 12, 11, 13]</c> where NumPy writes <c>[10, 11, 12, 13]</c>). The same-element-type
        ///     route takes <c>ascontiguousarray</c>, already C order.
        /// </remarks>
        [NDScopedCovered]
        private static NDArray PutValuesLikeNumPy(object values, NDArray a)
        {
            // Python's None: PyArray_FromAny(None, a.dtype, FORCECAST) is one element of None's setitem value in a's
            // descriptor (any dtype: NaN, nan+nanj, False, 'None' truncated, a StringDType NA object; int(None)'s TypeError).
            if (values is null)
                return NDArray.NoneForStore(a.dtype);

            // A StringDType target: Python objects (literals, boxed values, a list's elements) are stored by setitem, an
            // ndarray is cast (NDVStringMove.StoreValue).
            if (a.Storage.IsVString)
                return Backends.Strings.NDVStringMove.StoreValue(values, a.dtype);

            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar and is packed with PyArray_Pack's range
            // checks; a boxed C# number is packed by its kind; any other value is converted, for the cast below.
            var v = values is NDArray nd
                ? NDArray.PackForStore(nd, a.typecode, pyArrayPackRoute: false)
                : NDArray.AsStoreValue(values, a.typecode);

            // Same element type (for strings: same class AND width) reads the values as they are; anything else is
            // converted to a's dtype first (NumPy's FORCECAST: a string is truncated / NUL-padded, S <-> U checked).
            if (v.HasSameElementType(a))
                return v.Shape.IsContiguous ? v : np.ascontiguousarray(v);
            return v.astype(a.dtype, copy: true, order: 'C');
        }

        /// <summary>
        ///     The contiguous core of <see cref="PutTo"/>: runs the IL scatter kernel straight into <paramref name="a"/>'s
        ///     buffer, cycling <paramref name="vals"/>.
        /// </summary>
        /// <param name="a">The C-contiguous target (the caller's array, or the scratch copy).</param>
        /// <param name="idx64">The C-contiguous indices: int64, or int32 read in place when <paramref name="idx32"/>.</param>
        /// <param name="idx32">True when <paramref name="idx64"/> holds int32 values.</param>
        /// <param name="ni">The number of indices.</param>
        /// <param name="vals">The C-contiguous values of <paramref name="a"/>'s element type, at least one.</param>
        /// <param name="mode">The parsed boundary mode (<see cref="np.ClipModeRaise"/>, <see cref="np.ClipModeWrap"/>,
        ///     <see cref="np.ClipModeClip"/>).</param>
        /// <exception cref="NotSupportedException">No IL put kernel exists for the element width.</exception>
        /// <exception cref="IndexError">An index is out of bounds under mode raise (NumPy's text; the indices before it
        ///     have been written).</exception>
        private static unsafe void PutKernel(NDArray a, NDArray idx64, bool idx32, long ni, NDArray vals, int mode)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(a.dtypesize);

            byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            // idx32: the index buffer holds int32 values; the kernels' `long*` parameter is a reinterpret.
            long* idxPtr = idx32
                ? (long*)((int*)idx64.Storage.Address + idx64.Shape.offset)
                : (long*)idx64.Storage.Address + idx64.Shape.offset;
            byte* valsPtr = (byte*)vals.Storage.Address + vals.Shape.offset * vals.dtypesize;

            long status;
            // The lean flat scatter (DirectILKernelGenerator.GatherFlat.cs) serves RAISE mode with a
            // compile-time element width; wrap/clip keep the general kernel.
            var flat = mode == np.ClipModeRaise ? DirectILKernelGenerator.GetPutFlatKernel(copyKind, idx32) : null;
            if (flat != null)
            {
                status = flat(dstPtr, idxPtr, ni, valsPtr, vals.size, a.size);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetPutKernel(copyKind, idx32);
                if (kernel == null)
                    throw new NotSupportedException("np.put: IL kernel unavailable");
                status = kernel(dstPtr, idxPtr, ni,
                                valsPtr, vals.size,
                                a.size, a.dtypesize, mode);
            }
            if (status < ni)
            {
                long badI = status;
                long badVal = idx32 ? ((int*)idxPtr)[badI] : idxPtr[badI];
                // NumPy's check_and_adjust_index text (an IndexError), the ORIGINAL value of the bad index.
                throw new IndexError($"index {badVal} is out of bounds for axis 0 with size {a.size}");
            }
        }

        // ---- np.putmask: PyArray_PutMask --------------------------------------------------------------------------------

        /// <summary>
        ///     The body of both <c>np.putmask</c> forms (NumPy's <c>PyArray_PutMask</c>, in its order).
        /// </summary>
        /// <param name="a">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="a"/>'s size.</param>
        /// <param name="values">The values (an ndarray, or the array discovered from a list); <c>null</c> is Python's
        ///     <c>None</c>, converted where NumPy converts values (after the mask's size check).</param>
        /// <param name="sequence">True when <paramref name="values"/> came from a list: no <c>'safe'</c> rule.</param>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="mask"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="values"/> is null (<c>None</c>) and <paramref name="a"/> is an
        ///     integer (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">See <see cref="np.putmask(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="InvalidCastException">An ndarray of values that does not cast 'safely'.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">See <see cref="np.putmask(NDArray, NDArray, NDArray)"/>.</exception>
        [NDScoped]
        internal static void PutMaskCore(NDArray a, NDArray mask, NDArray values, bool sequence)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, mask, values);   // in place: the helpers below see a matrix as a plain array
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (mask is null) throw new ArgumentNullException(nameof(mask));

            // Read-only target reported FIRST — NumPy checks FailUnlessWriteable right after the
            // is-an-array check, before the mask-size test and before the nv==0 no-op (probed 2.4.2:
            // putmask(ro, all-false-mask, v) and putmask(ro, mask, []) both raise read-only). The
            // contiguous kernel writes a.Storage.Address directly; the non-contiguous path is guarded
            // via np.copyto below.
            NumSharpException.ThrowIfNotWriteable(a.Shape, "putmask: output array");

            // PyArray_FROM_OTF(mask, NPY_BOOL, CARRAY | FORCECAST) comes next, before the size check: a text mask becomes
            // its truth values here (NumPy's string-to-bool cast; a U unit above U+10FFFF raises SystemError before any
            // later error). The fresh truth array is reclaimed with the scope and never overlaps `a`.
            if (mask.Storage.IsVString)
                mask = mask.astype(NPTypeCode.Boolean, copy: true, order: 'C'); // StringDType's bool cast (NA rules)
            else if (mask.typecode == NPTypeCode.String)
                mask = Backends.Strings.NDStringTruth.Truth(mask, Backends.Strings.NDStringTruth.Flavor.Cast, 'C');

            if (mask.size != a.size)
                throw new ValueError("putmask: mask and data must be the same size");

            // Python's None: PyArray_FromAny(None, a.dtype, CARRAY) is ONE element of None's setitem value in a's descriptor,
            // converted here, after the size check and whatever the mask holds (an integer array raises int(None)'s TypeError
            // even under an all-false mask). The 0-d value is this scope's and is used like any value of a's element type.
            if (values is null)
                values = NDArray.NoneForStore(a.dtype);

            // PyArray_FromAny(values, a.dtype, CARRAY): the 'safe' rule for an array of values (a dtype check, so empty
            // values and an empty target raise too) and every value's conversion, before nv is read.
            NDArray converted = PlaceValuesLikeNumPy(values, a, sequence);

            // Empty values ⇒ no-op. NumPy: nv <= 0 returns None without writing (probed 2.4.2).
            // Checked after writeable + mask-size + the conversion (NumPy's order), before the contig copy so an
            // empty-values call never allocates a scratch.
            if (values.size == 0)
                return;

            // Empty a ⇒ empty mask ⇒ no-op (the loop runs zero times in NumPy).
            if (a.size == 0)
                return;

            // The values the loop reads (a converted copy, or the caller's array of a's element type).
            var source = converted ?? values;

            // NumPy WRITEBACKIFCOPY / ENSURECOPY semantics (PyArray_PutMask): copy `a` when it is
            // non-contiguous OR may share memory with `values`/`mask`. NumPy's `arrays_overlap` is
            // `solve_may_share_memory(..., NPY_MAY_SHARE_BOUNDS)` == `SolveMayShareMemory(maxWork:0)`
            // (a cheap bounds-only check: `!= No` means "may share"). The fresh scratch cannot alias
            // the read operands, so a write into it never disturbs a later cyclic read of `values`
            // that aliases `a` — e.g. `putmask(a, mask, a[2:8])`, where the contiguous fast path
            // would write a[i] and then read it back as the wrapped `values[i % nv]` and diverge.
            // The kernel writes the scratch in C-order flat (matching NumPy's copy-and-writeback);
            // np.copyto pushes the result back through `a`'s strides to the parent storage. The
            // recursive call sees a fresh scratch that overlaps neither operand, so it takes the
            // direct path (no re-copy, no recursion).
            // NumPy's arrays_overlap runs on the CONVERTED values (a fresh conversion never overlaps `a`).
            bool overlaps =
                NDMemOverlap.SolveMayShareMemory(a, source, maxWork: 0) != MemOverlap.No ||
                NDMemOverlap.SolveMayShareMemory(a, mask, maxWork: 0) != MemOverlap.No;
            if (overlaps || !a.Shape.IsContiguous)
            {
                // C-contiguous a (the overlap case) → a.copy() is a fresh C-order buffer;
                // any non-contiguous layout → ascontiguousarray makes the C-order copy.
                // The recursive call opens its own nested scope; the scratch itself is released by this one. It re-runs
                // the checks on the already converted values (the same element type: nothing left to fail but the mask's).
                var scratch = a.Shape.IsContiguous ? a.copy() : np.ascontiguousarray(a);
                PutMaskCore(scratch, mask, source, sequence);
                np.copyto(a, scratch, casting: "unsafe");
                return;
            }

            // Cast mask to contig bool (a text mask already is: its truth array, made above). It is raveled in C-order, aligning flat position i of the mask with flat
            // position i of a (shapes need not match, only sizes — NumPy checks size, not shape). A converted copy is
            // built inside the scope and released with it; an operand used as-is is the caller's and is never disposed.
            NDArray maskCast;
            if (mask.GetTypeCode == NPTypeCode.Boolean && mask.Shape.IsContiguous)
            {
                maskCast = mask;
            }
            else
            {
                // C order: the kernel walks the mask linearly beside the target's C-order positions, and astype's
                // default 'K' keeps an F-order mask F-ordered (an int8 F-order [[1, 1], [0, 0]] selected positions 0 and
                // 2 where NumPy selects 0 and 1).
                maskCast = mask.GetTypeCode == NPTypeCode.Boolean
                    ? np.ascontiguousarray(mask)
                    : mask.astype(NPTypeCode.Boolean, copy: true, order: 'C');
            }

            // The values in a's element type, C-contiguous (the conversion above already is).
            NDArray valsCast = source.Shape.IsContiguous ? source : np.ascontiguousarray(source);
            // PyArray_PutMask's PyDataType_REFCHK(PyArray_DESCR(self)): a reference-holding target (StringDType) receives
            // its items through the dtype's copy, never as raw bytes.
            if (a.dtype.Meta.HasReferences)
            {
                // A StringDType target: the same putmask runs on marks (the values' element numbers, cycled by position)
                // and the marked elements receive the strings.
                var mk = maskCast;
                Backends.Strings.NDVStringMove.ScatterVia(a, valsCast, (marks, index) => PutMaskCore(marks, mk, index, false));
                return;
            }
            ExecutePutMask(a, maskCast, valsCast);
        }

        private static unsafe void ExecutePutMask(NDArray a, NDArray maskBool, NDArray valsCast)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(a.dtypesize);
            var kernel = DirectILKernelGenerator.GetPutMaskKernel(copyKind);
            if (kernel == null)
                throw new NotSupportedException("np.putmask: IL kernel unavailable");

            byte* dstPtr = (byte*)a.Storage.Address + a.Shape.offset * a.dtypesize;
            byte* maskPtr = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            kernel(dstPtr, maskPtr, a.size, valsPtr, valsCast.size, a.dtypesize);
        }

        // ---- np.place: compiled_base.c's arr_place ----------------------------------------------------------------------

        /// <summary>
        ///     The body of both <c>np.place</c> forms (NumPy's <c>arr_place</c>, in its order).
        /// </summary>
        /// <param name="arr">Target array (modified in place).</param>
        /// <param name="mask">Boolean mask of <paramref name="arr"/>'s size.</param>
        /// <param name="vals">The values (an ndarray, or the array discovered from a list); <c>null</c> is Python's
        ///     <c>None</c>, converted where NumPy converts values (after the mask's size check).</param>
        /// <param name="sequence">True when <paramref name="vals"/> came from a list: NumPy converts it into the target's
        ///     dtype without the <c>'safe'</c> rule.</param>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="mask"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="vals"/> is null (<c>None</c>) and <paramref name="arr"/> is an
        ///     integer (or <c>decimal</c>) array.</exception>
        /// <exception cref="ValueError">See <see cref="np.place(NDArray, NDArray, NDArray)"/>.</exception>
        /// <exception cref="InvalidCastException">An ndarray of values that does not cast 'safely'.</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> array.</exception>
        /// <exception cref="ValueError">See <see cref="np.place(NDArray, NDArray, NDArray)"/>.</exception>
        [NDScoped]
        internal static void PlaceCore(NDArray arr, NDArray mask, NDArray vals, bool sequence)
        {
            using var plain = NDMatrixBoundary.PlainFor(arr, mask, vals);   // in place: the helpers below see a matrix as a plain array
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (mask is null) throw new ArgumentNullException(nameof(mask));

            // Read-only target reported first (NumPy uses WRITEBACKIFCOPY; probed 2.4.2:
            // place(ro, all-false-mask, v) still raises read-only). The contiguous kernel
            // (ExecutePlace) writes arr.Storage.Address directly; the non-contiguous path is
            // already guarded via np.copyto.
            NumSharpException.ThrowIfNotWriteable(arr.Shape, "WRITEBACKIFCOPY base");

            // PyArray_FROM_OTF(mask, NPY_BOOL, CARRAY | FORCECAST) comes next, before the size check: a text mask becomes
            // its truth values here (NumPy's string-to-bool cast; a U unit above U+10FFFF raises SystemError before any
            // later error). The fresh truth array is reclaimed with the scope.
            if (mask.Storage.IsVString)
                mask = mask.astype(NPTypeCode.Boolean, copy: true, order: 'C'); // StringDType's bool cast (NA rules)
            else if (mask.typecode == NPTypeCode.String)
                mask = Backends.Strings.NDStringTruth.Truth(mask, Backends.Strings.NDStringTruth.Flavor.Cast, 'C');

            if (mask.size != arr.size)
                throw new ValueError("place: mask and data must be the same size");

            // Python's None: PyArray_FromAny(None, arr.dtype, CARRAY) is ONE element of None's setitem value, converted here
            // whatever the mask holds (an integer array raises int(None)'s TypeError even under an all-false mask). The 0-d
            // value is this scope's.
            if (vals is null)
                vals = NDArray.NoneForStore(arr.dtype);

            // NumPy converts the values before it looks at the mask's contents — the casting rule (a dtype check, so an
            // empty target raises too) and every value's conversion.
            NDArray converted = PlaceValuesLikeNumPy(vals, arr, sequence);

            // Empty arr ⇒ empty mask ⇒ no-op.
            if (arr.size == 0)
                return;

            // NumPy WRITEBACKIFCOPY semantics for non-contig views — see np.put for
            // the same pattern. ascontiguousarray clones the view into a contig
            // scratch; the kernel writes into the scratch; np.copyto pushes the
            // changes back through the view's strides to the parent storage.
            if (!arr.Shape.IsContiguous)
            {
                // The recursive call opens its own nested scope; the scratch itself is released by this one. It re-runs
                // the checks on the already converted values (the same dtype: nothing left to fail but the mask's).
                var scratch = np.ascontiguousarray(arr);
                PlaceCore(scratch, mask, converted ?? vals, sequence);
                np.copyto(arr, scratch, casting: "unsafe");
                return;
            }

            // Cast mask to contig bool (a text mask already is: its truth array, made above); cast vals to contig arr.dtype. A converted copy is built inside the
            // scope and released with it; an operand used as-is is the caller's and is never disposed.
            NDArray maskCast;
            if (mask.GetTypeCode == NPTypeCode.Boolean && mask.Shape.IsContiguous)
            {
                maskCast = mask;
            }
            else
            {
                // C order: the kernel walks the mask linearly beside the target's C-order positions, and astype's
                // default 'K' keeps an F-order mask F-ordered (an int8 F-order [[1, 1], [0, 0]] selected positions 0 and
                // 2 where NumPy selects 0 and 1).
                maskCast = mask.GetTypeCode == NPTypeCode.Boolean
                    ? np.ascontiguousarray(mask)
                    : mask.astype(NPTypeCode.Boolean, copy: true, order: 'C');
            }

            if (vals.size == 0 && AnyTrueMask(maskCast))
                throw new ValueError("Cannot insert from an empty array!");

            if (vals.size == 0)
                return;   // mask all-false and empty vals — NumPy treats as no-op

            NDArray valsCast = converted ?? (vals.Shape.IsContiguous ? vals : np.ascontiguousarray(vals));
            // The reference-holding route PyArray_PutMask takes (PyDataType_REFCHK of the target's descriptor). NumPy's
            // arr_place has none: it copies each item with the dtype's ArrFuncs copyswap, which StringDType lacks.
            if (arr.dtype.Meta.HasReferences)
            {
                // A StringDType target: the same place runs on marks (the values' element numbers, cycled over the True
                // positions) and the marked elements receive the strings. (NumPy 2.4.2 crashes the process here.)
                var mk = maskCast;
                Backends.Strings.NDVStringMove.ScatterVia(arr, valsCast, (marks, index) => PlaceCore(marks, mk, index, false));
                return;
            }
            ExecutePlace(arr, maskCast, valsCast);
        }

        /// <summary>
        ///     The value conversion of <see cref="np.place(NDArray, NDArray, NDArray)"/> and
        ///     <see cref="np.putmask(NDArray, NDArray, NDArray)"/> — NumPy's <c>PyArray_FromAny(values, target.dtype, 0, 0,
        ///     NPY_ARRAY_CARRAY)</c>: an ARRAY of values must convert under the <c>'safe'</c> rule (a dtype check, made
        ///     whatever the sizes); a C# literal's 0-d array (<see cref="NDArray.WasPythonLiteral"/>, NumPy's Python scalar)
        ///     is PACKED with PyArray_Pack's range checks; any other 0-d value — NumSharp's stand-in for a NumPy scalar,
        ///     which NumPy also converts without the rule — and a list (a C# array) are converted unconditionally; then
        ///     every value is converted.
        /// </summary>
        /// <param name="values">The caller's values.</param>
        /// <param name="target">The array written into (its dtype is the conversion's target).</param>
        /// <param name="sequence">True when <paramref name="values"/> was discovered from a list: no <c>'safe'</c> rule, and
        ///     a list of strings parsed into a numeric target quotes each Python str (<c>'x'</c>) in its error.</param>
        /// <returns>The converted values (C-contiguous, the target's dtype), or null when <paramref name="values"/> already
        ///     holds the target's element type.</returns>
        /// <exception cref="InvalidCastException">An array of values does not cast 'safely' (NumPy's TypeError text).</exception>
        /// <exception cref="OverflowException">A C# literal out of the target's integer range (NumPy's OverflowError).</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer target.</exception>
        /// <exception cref="TypeError">A complex literal into a real target.</exception>
        /// <exception cref="ValueError">A text value that does not parse into a numeric target (NumPy's <c>int()</c> /
        ///     <c>float()</c> / <c>complex()</c> texts).</exception>
        /// <exception cref="UnicodeEncodeError">Text values that are not ASCII into an <c>S</c> target.</exception>
        /// <exception cref="UnicodeDecodeError">Byte values that are not ASCII into a <c>U</c> target.</exception>
        /// <remarks>An ARRAY of text values never casts safely to a number, and numbers cast safely to text only when the
        ///     text is wide enough to hold any of them (<c>int64 → &lt;U21</c>): both come from the rule check.</remarks>
        private static NDArray PlaceValuesLikeNumPy(NDArray values, NDArray target, bool sequence)
        {
            // A C# literal into a StringDType target is NumPy's Python scalar: stored by setitem (the NA object, its str(),
            // refused under coerce=False), not cast.
            if (target.Storage.IsVString && (values.WasPythonLiteral || values.WasPythonStr || values.WasPythonBool))
                return NDArray.PackForVStringStore(values, target.dtype);
            // A C# literal (NDArray.WasPythonLiteral / WasPythonStr) is NumPy's Python scalar: PyArray_FromAny(values, dtype,
            // CARRAY) packs it with PyArray_Pack's range checks (np.putmask(int8, m, 1000) / np.place(int8, m, 1000) raise
            // "Python integer 1000 out of bounds for int8"; a str is parsed, its error quoting the str) instead of applying
            // an array's 'safe' rule or cast. A literal of the target's own dtype is exact and is used as is.
            if ((values.WasPythonLiteral || values.WasPythonStr) && target.typecode != NPTypeCode.String)
            {
                var packed = NDArray.PackForStore(values, target.typecode, pyArrayPackRoute: false);
                return ReferenceEquals(packed, values) ? null : packed;
            }

            // PyArray_FromArray (an ndarray) checks the rule; a list is built in the target's dtype directly.
            if (!sequence && values.ndim != 0
                && !DTypeCasting.CanCastTypeTo(values.dtype, target.dtype, NPY_CASTING.NPY_SAFE_CASTING))
                throw Backends.Strings.NDStringCasting.InvalidCast(values.dtype, target.dtype, NPY_CASTING.NPY_SAFE_CASTING, scalar: false);

            if (values.HasSameElementType(target))
                return null;

            // Into StringDType (or from it): the StringDType casts (text, numbers' str(), the NA rules), in C order.
            if (target.Storage.IsVString || values.Storage.IsVString)
                return values.astype(target.dtype, copy: true, order: 'C');

            bool textValues = values.typecode == NPTypeCode.String, textTarget = target.typecode == NPTypeCode.String;
            if (textValues != textTarget)
            {
                // Text into numbers (a 0-d text array, or a list of strings — an ndarray of text failed the 'safe' rule
                // above) is parsed, numbers into text formatted, every value now, before the mask is read.
                var result = NDArray.ForOverwrite(target.dtype, Shape.Adopt((long[])values.Shape.dimensions.Clone()));
                try
                {
                    if (result.size > 0)
                        Backends.Strings.NDTextConversion.Convert(result.Storage, values.Storage,
                            sequence ? Backends.Strings.TextReprStyle.PythonObject : Backends.Strings.TextReprStyle.ArrayElement);
                    return result;
                }
                catch
                {
                    result.Dispose();
                    throw;
                }
            }
            // Number to number (unsafe for a scalar, the rule already checked for an array), or text to text (truncated /
            // NUL-padded, S <-> U ASCII-checked): every value converted now, as NumPy does before reading the mask. In C
            // order: the place / putmask kernels read the values linearly (PyArray_FromAny's CARRAY), and astype's default
            // 'K' kept an F-order input F-ordered (np.place's int32 F-order values were written in memory order).
            return values.astype(target.dtype, copy: true, order: 'C');
        }

        private static unsafe bool AnyTrueMask(NDArray maskBool)
        {
            // Walk the contig bool buffer linearly. mask.size > 0 by the time we get here.
            byte* p = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            long n = maskBool.size;
            for (long i = 0; i < n; i++)
                if (p[i] != 0) return true;
            return false;
        }

        private static unsafe void ExecutePlace(NDArray arr, NDArray maskBool, NDArray valsCast)
        {
            int copyKind = DirectILKernelGenerator.CopyKindFor(arr.dtypesize);
            var kernel = DirectILKernelGenerator.GetPlaceKernel(copyKind);
            if (kernel == null)
                throw new NotSupportedException("np.place: IL kernel unavailable");

            byte* dstPtr = (byte*)arr.Storage.Address + arr.Shape.offset * arr.dtypesize;
            byte* maskPtr = (byte*)maskBool.Storage.Address + maskBool.Shape.offset;
            byte* valsPtr = (byte*)valsCast.Storage.Address + valsCast.Shape.offset * valsCast.dtypesize;

            kernel(dstPtr, maskPtr, arr.size, valsPtr, valsCast.size, arr.dtypesize);
        }

        // ---- np.repeat: PyArray_Repeat ----------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_Repeat</c> (<c>item_selection.c</c>) for a scalar count: the count's sign, the
        ///     <c>StringDType</c> route, then the flat or the along-axis tiling. The body of
        ///     <see cref="np.repeat(NDArray, long, int?)"/> once its class boundaries and its null check have run.
        /// </summary>
        /// <param name="a">The source array, not null; any layout (the tiling reads a C-contiguous copy or view).</param>
        /// <param name="repeats">How many times each element repeats.</param>
        /// <param name="axis">The axis to repeat along, or null to repeat the flattened array.</param>
        /// <returns>A new C-contiguous array of <paramref name="a"/>'s dtype: 1-D of <c>a.size * repeats</c> elements
        ///     for a null <paramref name="axis"/>, else <paramref name="a"/>'s shape with the axis multiplied.</returns>
        /// <exception cref="ArgumentException"><paramref name="repeats"/> is negative.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range (a 0-d source takes 0 or -1).</exception>
        /// <remarks>
        ///     The <c>[NDScoped]</c> <c>np.repeat</c> overloads are its callers: their scope releases the raveled and
        ///     contiguous copies the tiling reads, and it roots <paramref name="a"/> while the kernels read it through raw
        ///     pointers.
        /// </remarks>
        internal static NDArray Repeat(NDArray a, long repeats, int? axis)
        {
            if (repeats < 0)
                throw new ArgumentException("repeats may not contain negative values");
            // PyArray_Repeat's needs_custom_copy, PyDataType_REFCHK(PyArray_DESCR(ret)) (ret has a's descriptor). A
            // StringDType source: the same repeat runs on its cell view (NumPy's checks, this array's shapes) and the cells
            // are duplicated (NDVStringMove) — never the raw byte kernels below.
            if (a.dtype.Meta.HasReferences)
                return Backends.Strings.NDVStringMove.Gather(a, cells => np.repeat(cells, repeats, axis));

            if (axis is null)
                return RepeatScalarFlat(a, repeats);

            return RepeatScalarAlongAxis(a, repeats, axis.Value);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_Repeat</c> (<c>item_selection.c</c>) for an array of counts: the
        ///     <c>StringDType</c> route, the counts' conversion (<c>PyArray_ContiguousFromAny(repeats, NPY_INTP, 0, 1)</c>:
        ///     the rank first, then the <c>'safe'</c> cast), then the flat or the along-axis tiling. The body of
        ///     <see cref="np.repeat(NDArray, NDArray, int?)"/> once its class boundaries and its null checks have run.
        /// </summary>
        /// <param name="a">The source array, not null; any layout.</param>
        /// <param name="repeats">The counts, not null: a 0-d or one-element array (broadcast to every element) or a 1-D
        ///     array as long as the flattened source or the axis.</param>
        /// <param name="axis">The axis to repeat along, or null to repeat the flattened array.</param>
        /// <returns>A new C-contiguous array of <paramref name="a"/>'s dtype.</returns>
        /// <exception cref="ValueError"><paramref name="repeats"/> has more than one dimension.</exception>
        /// <exception cref="TypeError"><paramref name="repeats"/> does not cast <c>'safe'</c>ly to int64.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">The counts' length does not match, or a count is negative.</exception>
        /// <remarks>
        ///     The <c>[NDScoped]</c> <c>np.repeat</c> overload is its caller: its scope releases the raveled copies and
        ///     roots the operands while the kernels read them through raw pointers.
        /// </remarks>
        internal static NDArray Repeat(NDArray a, NDArray repeats, int? axis)
        {
            // PyArray_Repeat's needs_custom_copy, PyDataType_REFCHK(PyArray_DESCR(ret)) (ret has a's descriptor). A
            // StringDType source: the same repeat runs on its cell view (the repeats' checks and the axis check are the
            // numeric route's, in NumPy's order) and the cells are duplicated (NDVStringMove).
            if (a.dtype.Meta.HasReferences)
                return Backends.Strings.NDVStringMove.Gather(a, cells => np.repeat(cells, repeats, axis));

            // NumPy parity: the repeats' rank is checked before its dtype (PyArray_FromAny's depth check, then the cast).
            if (repeats.ndim > 1)
                throw new ValueError("object too deep for desired array");
            // ...and must cast safely to int64 — reject float/complex/uint64/decimal/text (a 0-d repeats is a "scalar").
            if (!IsSafeToInt64(repeats.GetTypeCode))
                throw new TypeError(
                    $"Cannot cast {(repeats.ndim == 0 ? "scalar" : "array data")} from {repeats.dtype.ToString(true)} to " +
                    "dtype('int64') according to the rule 'safe'");

            if (axis is null)
                return RepeatPerElementFlat(a, repeats);

            return RepeatPerElementAlongAxis(a, repeats, axis.Value);
        }

        // ============== axis=None (flatten) paths ==============

        private static unsafe NDArray RepeatScalarFlat(NDArray a, long repeats)
        {
            if (a.size == 0 || repeats == 0)
                return new NDArray(a.dtype, Shape.Vector(0));

            // ravel() returns a C-contig view (no copy when already contig) or a fresh contig copy.
            NDArray src = a.ravel();
            long total = src.size * repeats;
            var ret = new NDArray(a.dtype, Shape.Vector(total));

            // Degenerate 3-loop: n_outer=1, n=size, chunk=elsize.
            var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(a.dtypesize);
            kernel(
                src: RepeatData(src),
                dst: (byte*)ret.Address,
                n_outer: 1,
                n: src.size,
                count: repeats);

            return ret;
        }

        private static unsafe NDArray RepeatPerElementFlat(NDArray a, NDArray repeats)
        {
            NDArray src = a.ravel();
            NDArray repFlat = repeats.ravel();

            // NumPy: scalar (0-d) or size-1 repeats broadcasts to a.size; anything else must match exactly.
            bool broadcast = repFlat.size == 1 || repeats.ndim == 0;
            if (!broadcast && src.size != repFlat.size)
                throw new ArgumentException(
                    $"operands could not be broadcast together with shape ({src.size},) ({repFlat.size},)");

            if (src.size == 0)
                return new NDArray(a.dtype, Shape.Vector(0));

            long[] counts;
            long broadcastVal;
            long total = ComputeCounts(repFlat, broadcast, src.size, out counts, out broadcastVal);

            if (total == 0)
                return new NDArray(a.dtype, Shape.Vector(0));

            var ret = new NDArray(a.dtype, Shape.Vector(total));

            if (broadcast)
            {
                var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(a.dtypesize);
                kernel(
                    src: RepeatData(src),
                    dst: (byte*)ret.Address,
                    n_outer: 1,
                    n: src.size,
                    count: broadcastVal);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetRepeatPerJKernel(a.dtypesize);
                fixed (long* pCounts = counts)
                {
                    kernel(
                        src: RepeatData(src),
                        dst: (byte*)ret.Address,
                        n_outer: 1,
                        n: src.size,
                        counts: pCounts);
                }
            }

            return ret;
        }

        // ============== axis-aware paths ==============

        private static unsafe NDArray RepeatScalarAlongAxis(NDArray a, long repeats, int axis)
        {
            int ndim = a.ndim;

            // NumPy: 0-d input with axis=0/-1 is silently promoted to a 1-d, size-1 array (PyArray_CheckAxis ravels it), so
            // any other axis is reported against that rank of 1.
            if (ndim == 0)
            {
                if (axis != 0 && axis != -1)
                    throw new AxisError(axis, 1);
                a = a.reshape(1);
                ndim = 1;
            }

            int normalizedAxis = NormalizeAxis(axis, ndim);

            // NumPy's PyArray_CheckAxis(... CARRAY) makes the operand C-contig — the
            // chunked memcpy reads a logically-rectangular slab of inner dims.
            NDArray src = a.Shape.IsContiguous ? a : np.ascontiguousarray(a);

            long[] inDims = src.Shape.dimensions;
            long n = inDims[normalizedAxis];
            long total = n * repeats;

            long[] outDims = (long[])inDims.Clone();
            outDims[normalizedAxis] = total;
            var ret = new NDArray(a.dtype, Shape.Adopt(outDims));

            if (src.size == 0 || total == 0)
                return ret;

            ComputeAxisGeometry(inDims, normalizedAxis, out long n_outer, out long nel);
            int chunkBytes = checked((int)(nel * a.dtypesize));

            var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(chunkBytes);
            kernel(
                src: RepeatData(src),
                dst: (byte*)ret.Address,
                n_outer: n_outer,
                n: n,
                count: repeats);

            return ret;
        }

        private static unsafe NDArray RepeatPerElementAlongAxis(NDArray a, NDArray repeats, int axis)
        {
            int ndim = a.ndim;
            // A 0-d input ravels to 1-D first (PyArray_CheckAxis): axis 0 / -1 fit, any other is reported against rank 1.
            if (ndim == 0)
            {
                if (axis != 0 && axis != -1)
                    throw new AxisError(axis, 1);
                a = a.reshape(1);
                ndim = 1;
            }

            int normalizedAxis = NormalizeAxis(axis, ndim);

            NDArray src = a.Shape.IsContiguous ? a : np.ascontiguousarray(a);
            NDArray repFlat = repeats.ravel();

            long[] inDims = src.Shape.dimensions;
            long n = inDims[normalizedAxis];

            // NumPy parity: scalar (0-d) or size-1 repeats broadcasts along the axis; otherwise the
            // size must match the axis length exactly.
            bool broadcast = repFlat.size == 1 || repeats.ndim == 0;
            if (!broadcast && repFlat.size != n)
                throw new ArgumentException(
                    $"operands could not be broadcast together with shape ({n},) ({repFlat.size},)");

            long[] counts;
            long broadcastVal;
            long total = ComputeCounts(repFlat, broadcast, n, out counts, out broadcastVal);

            long[] outDims = (long[])inDims.Clone();
            outDims[normalizedAxis] = total;
            var ret = new NDArray(a.dtype, Shape.Adopt(outDims));

            if (src.size == 0 || total == 0)
                return ret;

            ComputeAxisGeometry(inDims, normalizedAxis, out long n_outer, out long nel);
            int chunkBytes = checked((int)(nel * a.dtypesize));

            if (broadcast)
            {
                var kernel = DirectILKernelGenerator.GetRepeatBroadcastKernel(chunkBytes);
                kernel(
                    src: RepeatData(src),
                    dst: (byte*)ret.Address,
                    n_outer: n_outer,
                    n: n,
                    count: broadcastVal);
            }
            else
            {
                var kernel = DirectILKernelGenerator.GetRepeatPerJKernel(chunkBytes);
                fixed (long* pCounts = counts)
                {
                    kernel(
                        src: RepeatData(src),
                        dst: (byte*)ret.Address,
                        n_outer: n_outer,
                        n: n,
                        counts: pCounts);
                }
            }

            return ret;
        }

        // ============== helpers ==============

        /// <summary>
        ///     The address of <paramref name="x"/>'s first element: the storage base plus the view's offset. The kernels'
        ///     source is a C-contiguous array, but a contiguous VIEW can still start past its storage base (a row of a
        ///     strided parent, a 0-d view <c>a[3, ...]</c>), so the storage base alone would read the wrong elements.
        /// </summary>
        /// <param name="x">A C-contiguous source array.</param>
        /// <returns>The byte address of its element 0.</returns>
        private static unsafe byte* RepeatData(NDArray x) => (byte*)x.Storage.Address + x.Shape.offset * x.dtypesize;

        private static int NormalizeAxis(int axis, int ndim)
        {
            int original = axis;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new AxisError(original, ndim);
            return axis;
        }

        private static void ComputeAxisGeometry(long[] dims, int axis, out long n_outer, out long nel)
        {
            n_outer = 1;
            for (int i = 0; i < axis; i++) n_outer *= dims[i];
            nel = 1;
            for (int i = axis + 1; i < dims.Length; i++) nel *= dims[i];
        }

        // Materializes the per-j repeat counts as a long[] and validates non-negative.
        // Returns the total output size along the axis.
        private static long ComputeCounts(NDArray repFlat, bool broadcast, long n, out long[] counts, out long broadcastVal)
        {
            if (broadcast)
            {
                broadcastVal = Converts.ToInt64(repFlat.GetAtIndex(0));
                if (broadcastVal < 0)
                    throw new ArgumentException("repeats may not contain negative values");
                counts = null;
                return broadcastVal * n;
            }

            broadcastVal = 0;
            counts = new long[n];
            long total = 0;
            for (long j = 0; j < n; j++)
            {
                long c = Converts.ToInt64(repFlat.GetAtIndex(j));
                if (c < 0)
                    throw new ArgumentException("repeats may not contain negative values");
                counts[j] = c;
                total += c;
            }
            return total;
        }

        /// <summary>
        ///     NumPy "safe" casting check for the repeats dtype (target int64).
        ///     Integers that fit in int64 + boolean pass; uint64/float/complex/decimal reject.
        /// </summary>
        private static bool IsSafeToInt64(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        // ---- np.choose: PyArray_Choose ----------------------------------------------------------------------------------

        /// <summary>
        ///     The shared body of the three <c>choose</c> overloads: classifies each choice (NEP50 weak scalar or
        ///     strong array), resolves the common dtype, casts the choices and the index array, broadcasts them to
        ///     the result shape and runs the IL gather kernel.
        /// </summary>
        /// <param name="a">The index array.</param>
        /// <param name="choices">The choices: <see cref="NDArray"/>s, boxed C# scalars, or array-likes.</param>
        /// <param name="out">Optional destination (shape must equal the result shape).</param>
        /// <param name="mode">The clip mode string, parsed by NumPy's rules.</param>
        /// <returns>The fresh result, or <paramref name="out"/> after the result is copied into it.</returns>
        /// <exception cref="ArgumentNullException">A choice entry is null.</exception>
        /// <exception cref="IncorrectShapeException">The choices and <paramref name="a"/> do not broadcast together:
        ///     NumPy's multi-iterator text (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch
        ///     is between arg 0 with shape (0, 3) and arg 2 with shape (0,).</c>), the choices numbered first and the
        ///     index array last.</exception>
        /// <exception cref="ValueError">No choices, an out-of-range index under raise, a bad clip mode, or a
        ///     read-only <paramref name="out"/>.</exception>
        /// <exception cref="TypeError">The index array cannot be safe-cast to int64, or <paramref name="out"/> has
        ///     the wrong shape.</exception>
        /// <remarks>
        ///     Covered by <c>[NDScopedCovered]</c>: its only callers are the three <c>[NDScoped]</c> <c>choose</c>
        ///     overloads, so every array built here except the returned result is released by the caller's scope,
        ///     exceptions included. The scope also keeps the int64 index copy reachable while the kernel reads it
        ///     through raw pointers.
        /// </remarks>
        [NDScopedCovered]
        internal static unsafe NDArray ChooseCore(NDArray a, object[] choices, NDArray @out, string mode)
        {
            int modeInt = ParseChooseMode(mode);

            int n = choices.Length;
            if (n == 0)
                throw new ValueError("0-length sequence.");

            // ── 1) classify each choice + resolve the NEP50 result dtype ──────────────────
            // Mirrors NumPy's ConvertToCommonType: mark python scalars weak, then result_type.
            var mats = new NDArray[n];
            bool anyStrong = false;
            bool anyBeyondInt64 = false;
            bool anyString = false; // a fixed-width string choice: the common dtype is resolved by DESCRIPTOR below
            int weakRank = 0; // 0 none, 1 int, 2 float, 3 complex
            NPTypeCode strong = NPTypeCode.Empty;

            for (int i = 0; i < n; i++)
            {
                object item = choices[i];
                NDArray mat;
                switch (item)
                {
                    case null:
                        // NumPy would build an object array (a dtype NumSharp lacks); a null choice is a bug.
                        throw new ArgumentNullException($"choices[{i}]", "choices entries must not be null.");

                    case NDArray lit when lit.WasPythonLiteral:
                        // A C# number literal that reached choose as its implicit 0-d array (a literal inside an NDArray[]
                        // choices list): NumPy's weak Python scalar, exactly as the boxed literal below — an int8 choice
                        // beside the literal 1000 stays int8 (wrapping to -24), where a 0-d ARRAY of 1000 promotes.
                        mat = lit;
                        weakRank = Math.Max(weakRank, WeakRankOfLiteral(lit.typecode));
                        if (lit.typecode == NPTypeCode.UInt64 && lit.GetAtIndex<ulong>(0) > long.MaxValue) anyBeyondInt64 = true;
                        break;

                    case NDArray arr:
                        mat = arr;
                        if (arr.typecode == NPTypeCode.String)
                            anyString = true;
                        else if (!anyString)
                            strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, arr.GetTypeCode) : arr.GetTypeCode;
                        anyStrong = true;
                        break;

                    // Weak (NEP50 "python literal") scalars — the eight C# integer primitives, the two
                    // binary floats and Complex. They adopt the other choices' dtype rather than forcing
                    // promotion by their own width (int8 choice + 1000 stays int8, wrapping 1000 → -24).
                    case sbyte or byte or short or ushort or int or uint or long or ulong:
                        mat = np.asanyarray(item);
                        weakRank = Math.Max(weakRank, 1);
                        if (item is ulong u && u > long.MaxValue) anyBeyondInt64 = true;
                        break;

                    case float or double:
                        mat = np.asanyarray(item);
                        weakRank = Math.Max(weakRank, 2);
                        break;

                    case Complex:
                        mat = np.asanyarray(item);
                        weakRank = Math.Max(weakRank, 3);
                        break;

                    default:
                        // Strong by NumPy's rule `type(x) in (int,float,complex) else asarray(x)`:
                        // bool, char, Half, decimal, C# arrays and collections take the asarray branch.
                        mat = np.asanyarray(item);
                        if (mat.typecode == NPTypeCode.String)
                            anyString = true;
                        else if (!anyString)
                            strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, mat.GetTypeCode) : mat.GetTypeCode;
                        anyStrong = true;
                        break;
                }

                mats[i] = mat;
            }

            // Fixed-width strings: NumPy's ConvertToCommonType over DESCRIPTORS — the widest choice's width, S + U
            // giving U; a Python number next to a string has no common dtype (TypeError). A numeric choice array is
            // FORMATTED into that descriptor, which cannot fail — so NumPy's later checks (the index cast, the broadcast,
            // out=, the index range) still decide, and the formatted elements are read where the result takes them (see
            // step 6). The gather kernels copy elemBytes bytes, any width.
            DType stringDescr = null;
            if (anyString)
            {
                // Mixed text and numbers promote to a string descriptor (a numeric choice widens it to its dtype's text
                // width); the numeric choices are formatted into it below.
                var operands = new object[n];
                for (int i = 0; i < n; i++)
                    operands[i] = choices[i] is NDArray nd ? nd.Nep50Operand : choices[i] is sbyte or byte or short or ushort or int or uint or long or ulong or float or double or Complex ? choices[i] : mats[i];
                stringDescr = DTypePromotion.ResultType(operands);

                // NumPy's PyArray_ConvertToCommonType converts every choice to the common descriptor with
                // PyArray_FromArray(…, NPY_ARRAY_CARRAY), which requires a SAFE cast (a Python scalar is force-cast, but a
                // Python number never reaches a string descriptor: the promotion above refused it). S -> U, U2 -> U5 and a
                // number -> its text width are safe; U -> StringDType is NOT (np.can_cast('U2', StringDType(), 'safe') is
                // False), so a U or str choice beside a StringDType one raises — the first refused choice in order.
                for (int i = 0; i < n; i++)
                {
                    if (!DTypeCasting.CanCastTypeTo(mats[i].dtype, stringDescr, NPY_CASTING.NPY_SAFE_CASTING))
                        throw new TypeError(
                            $"Cannot cast {(mats[i].ndim == 0 ? "scalar" : "array data")} from {mats[i].dtype.ToString(true)} " +
                            $"to {stringDescr.ToString(true)} according to the rule 'safe'");
                }
            }

            NPTypeCode dtype = anyString ? NPTypeCode.String : ResolveChooseDtype(anyStrong, strong, weakRank, anyBeyondInt64);
            int elemBytes = anyString ? (int)stringDescr.itemsize : InfoOf.GetSize(dtype);

            // ── 2) cast each choice to the common dtype (copy only when needed) ───────────
            // A numeric choice of a text result is formatted here (each number's str(), truncated to the result width):
            // NumPy's PyArray_ConvertToCommonType converts every choice before the gather, and formatting cannot fail, so
            // when it happens is unobservable.
            var castChoices = new NDArray[n];
            for (int i = 0; i < n; i++)
            {
                if (anyString)
                    castChoices[i] = mats[i].dtype.Meta == stringDescr.Meta && mats[i].dtype.itemsize == stringDescr.itemsize
                        ? mats[i]
                        : mats[i].astype(stringDescr, copy: false);
                else
                    castChoices[i] = mats[i].GetTypeCode == dtype ? mats[i] : mats[i].astype(dtype, copy: false);
            }

            // ── 3) index → int64 under the 'safe' rule ────────────────────────────────────
            // idx64 is EITHER `a` itself (already int64 — the caller's, built before any scope opened, so never
            // disposed) OR a fresh int64 astype built under the caller's scope, which releases it — and its
            // broadcast view below — on return or on the out-of-range throw.
            NDArray idx64 = CastChooseIndexToInt64(a);

            // ── 4) common broadcast shape of (all choices + index) ────────────────────────
            // (a numeric choice of a text result broadcasts with its OWN shape — its placeholder is 0-d)
            var shapes = new Shape[n + 1];
            for (int i = 0; i < n; i++) shapes[i] = castChoices[i].Shape;
            shapes[n] = idx64.Shape;
            // PyArray_Choose's multi-iterator broadcasts the choices and then the index ("index array at the end"), so a
            // conflict is NumPy's multi-iterator text naming those positions ("... Mismatch is between arg 0 with shape
            // (0, 3) and arg 2 with shape (0,)."): np.broadcast_shapes raises exactly that, in the same argument order.
            np.broadcast_shapes(shapes);
            Shape common = Shape.Broadcast(shapes)[0];
            long[] resultDims = common.dimensions;
            int ndim = resultDims.Length;

            // ── 5) validate `out` (shape must EQUAL the result shape; must be writeable) ───
            if (@out is not null)
            {
                if (!ChooseDimsEqual(@out.Shape, resultDims))
                    throw new TypeError("choose: invalid shape for output array.");
                if (!@out.Shape.IsWriteable)
                    throw new ValueError("output array is read-only");
            }

            // ── 5b) a StringDType result: the cells move through the allocators ──────────
            // The same choose runs on the choices' cell views (it picks the raw cells; the index range and the mode are the
            // numeric route's, checked again with the same shapes) and on the choice numbers 0..n-1 (which choice each
            // element came from); every cell is then duplicated through its own choice's allocator. (NumPy 2.4.2 copies the
            // raw cells of a StringDType choose, so strings longer than 15 bytes from any choice but the result's own
            // descriptor read garbage there; this gives the values NumPy's choose means.)
            // PyArray_Choose's PyDataType_REFCHK(dtype), on the common descriptor: a reference-holding result
            // (StringDType) is assembled through the dtype's copy.
            DType resultDescr = stringDescr ?? (DType)dtype;
            if (resultDescr.Meta.HasReferences)
            {
                var cellChoices = new object[n];
                var numbers = new object[n];
                var slices = new VStringArraySlice[n];
                for (int i = 0; i < n; i++)
                {
                    cellChoices[i] = Backends.Strings.NDVStringMove.CellArray(castChoices[i]);
                    numbers[i] = NDArray.Scalar((long)i);
                    slices[i] = castChoices[i].Storage.VStringSlice;
                }
                var raw = ChooseCore(idx64, cellChoices, null, mode);
                var which = ChooseCore(idx64, numbers, null, mode);
                var chosen = Backends.Strings.NDVStringMove.RehomeMulti(raw, which, slices, resultDescr);
                return FinishChoose(chosen, @out);
            }

            // ── 6) allocate the fresh C-contiguous result (common dtype) ──────────────────
            var result = new NDArray(resultDescr, Shape.Adopt((long[])resultDims.Clone()), false);
            long totalSize = result.size;

            if (totalSize == 0)
                return FinishChoose(result, @out);   // empty — nothing to gather

            // ── 7) broadcast every choice and the index to the common shape ───────────────
            var chViews = new NDArray[n];
            bool allContig = true;
            for (int i = 0; i < n; i++)
            {
                var v = np.broadcast_to(castChoices[i], common);
                chViews[i] = v;
                if (!v.Shape.IsContiguous) allContig = false;
            }
            var idxView = np.broadcast_to(idx64, common);
            if (!idxView.Shape.IsContiguous) allContig = false;

            // ── 8) run the IL kernel (flat when every operand is C-contiguous, else strided) ─
            long status = allContig
                ? RunChooseFlat(chViews, idxView, result, elemBytes, modeInt, n, totalSize)
                : RunChooseStrided(chViews, idxView, result, elemBytes, modeInt, n, totalSize, resultDims, ndim);

            if (status < totalSize)
                throw new ValueError("invalid entry in choice array");

            return FinishChoose(result, @out);
        }

        /// <summary>
        ///     Run the flat choose kernel: every operand is C-contiguous at the result shape, so element
        ///     <c>flat</c> is at <c>base + flat*elem</c> in every choice and <c>idx[flat]</c> in the index.
        /// </summary>
        private static unsafe long RunChooseFlat(NDArray[] chViews, NDArray idxView, NDArray result, int elemBytes, int modeInt, int n, long totalSize)
        {
            var kernel = DirectILKernelGenerator.GetChooseFlatKernel(elemBytes, modeInt);
            if (kernel is null)
                throw new NotSupportedException("np.choose: IL kernel unavailable");

            byte** choiceBases = stackalloc byte*[n];
            for (int i = 0; i < n; i++)
                choiceBases[i] = (byte*)chViews[i].Storage.Address + chViews[i].Shape.offset * elemBytes;

            byte* idxBase = (byte*)idxView.Storage.Address + idxView.Shape.offset * 8;
            byte* dstBase = (byte*)result.Storage.Address;

            return kernel(choiceBases, idxBase, dstBase, totalSize, n);
        }

        /// <summary>
        ///     Run the strided choose kernel: each operand is read through its own per-dimension BYTE
        ///     strides (0 for a broadcast dim), so any layout — broadcast/scalar/strided/transposed/
        ///     negative-stride/sliced — is gathered in place with no materialisation.
        /// </summary>
        private static unsafe long RunChooseStrided(NDArray[] chViews, NDArray idxView, NDArray result, int elemBytes, int modeInt, int n, long totalSize, long[] resultDims, int ndim)
        {
            var kernel = DirectILKernelGenerator.GetChooseStridedKernel(elemBytes, modeInt);
            if (kernel is null)
                throw new NotSupportedException("np.choose: IL kernel unavailable");

            var choiceStrides = new long[n * ndim];
            for (int i = 0; i < n; i++)
            {
                var st = chViews[i].Shape.strides;
                for (int d = 0; d < ndim; d++)
                    choiceStrides[i * ndim + d] = st[d] * elemBytes;
            }

            var idxStrides = new long[ndim];
            var idxSt = idxView.Shape.strides;
            for (int d = 0; d < ndim; d++)
                idxStrides[d] = idxSt[d] * 8;

            byte** choiceBases = stackalloc byte*[n];
            for (int i = 0; i < n; i++)
                choiceBases[i] = (byte*)chViews[i].Storage.Address + chViews[i].Shape.offset * elemBytes;

            byte* idxBase = (byte*)idxView.Storage.Address + idxView.Shape.offset * 8;
            byte* dstBase = (byte*)result.Storage.Address;

            fixed (long* pChoiceStrides = choiceStrides)
            fixed (long* pIdxStrides = idxStrides)
            fixed (long* pShape = resultDims)
            {
                return kernel(choiceBases, pChoiceStrides, idxBase, pIdxStrides, dstBase, pShape, ndim, totalSize, n);
            }
        }

        /// <summary>
        ///     Deliver the computed result: when <paramref name="out"/> is supplied, unsafe-cast the fresh
        ///     result into it (NumPy's <c>PyArray_CopyInto</c>) and return <paramref name="out"/>;
        ///     otherwise return the fresh result. Computing into a temp first is what gives NumPy's
        ///     guarantee that a <c>mode="raise"</c> failure never leaves <paramref name="out"/> partially
        ///     written.
        /// </summary>
        private static NDArray FinishChoose(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;
            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }

        /// <summary>True when <paramref name="shape"/>'s dimensions equal <paramref name="dims"/> exactly.</summary>
        private static bool ChooseDimsEqual(Shape shape, long[] dims)
        {
            var sd = shape.dimensions;
            if (sd.Length != dims.Length) return false;
            for (int i = 0; i < dims.Length; i++)
                if (sd[i] != dims[i]) return false;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>result_type(*choices)</c> folded with NEP50 weak-scalar rules — the same
        ///     resolution <see cref="np.select"/> uses for its choices (minus the default slot).
        /// </summary>
        private static NPTypeCode ResolveChooseDtype(bool anyStrong, NPTypeCode strong, int weakRank, bool anyBeyondInt64)
        {
            if (!anyStrong)
            {
                // All weak literals — the NEP50 defaults; a weak int past int64 lifts to uint64.
                return weakRank switch
                {
                    3 => NPTypeCode.Complex,
                    2 => NPTypeCode.Double,
                    _ => anyBeyondInt64 ? NPTypeCode.UInt64 : NPTypeCode.Int64,
                };
            }

            return weakRank switch
            {
                0 => strong,
                1 => strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong,
                2 => NDExprTypeRules.IsFloatKind(strong) || strong == NPTypeCode.Decimal || strong == NPTypeCode.Complex
                    ? strong
                    : NPTypeCode.Double,
                _ => NPTypeCode.Complex,
            };
        }

        /// <summary>
        ///     The NEP 50 weak rank of a C# number literal's 0-d array (<see cref="NDArray.WasPythonLiteral"/>): 1 for a Python
        ///     <c>int</c> (an integer primitive), 2 for a <c>float</c> (<c>float</c> / <c>double</c>), 3 for a <c>complex</c>.
        /// </summary>
        /// <param name="literal">The literal array's dtype.</param>
        /// <returns>The rank <see cref="ResolveChooseDtype"/> reads.</returns>
        private static int WeakRankOfLiteral(NPTypeCode literal) => literal switch
        {
            NPTypeCode.Complex => 3,
            NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half => 2,
            _ => 1,
        };

        /// <summary>
        ///     Convert the index array to <c>int64</c> (NumPy's <c>PyArray_FROM_OT(ip, NPY_INTP)</c>) under
        ///     the <c>"safe"</c> casting rule. Integer and boolean indices pass; <c>uint64</c>, float,
        ///     complex and decimal are rejected with NumPy's verbatim message. An already-<c>int64</c>
        ///     array keeps its layout (the kernel reads it through its strides).
        /// </summary>
        private static NDArray CastChooseIndexToInt64(NDArray a)
        {
            var tc = a.GetTypeCode;
            if (tc == NPTypeCode.Int64)
                return a;

            // Text never reaches int64 under 'safe'; the descriptor names its width.
            if (tc == NPTypeCode.String)
                throw new TypeError(
                    $"Cannot cast array data from {a.dtype.ToString(true)} to dtype('int64') according to the rule 'safe'");
            if (!np.can_cast(tc, NPTypeCode.Int64, "safe"))
                throw new TypeError(
                    $"Cannot cast array data from dtype('{tc.AsNumpyDtypeName()}') to dtype('int64') according to the rule 'safe'");

            return a.astype(NPTypeCode.Int64);
        }

        /// <summary>
        ///     Parse the clip mode with NumPy's <c>PyArray_ClipmodeConverter</c> (<see cref="np.ClipmodeConverter"/>, the
        ///     parser <c>np.take</c>, <c>np.put</c> and <c>np.ravel_multi_index</c> share) and map it to the choose
        ///     kernel's constants.
        /// </summary>
        /// <param name="mode"><c>"raise"</c>, <c>"wrap"</c>, <c>"clip"</c>, or <c>null</c> (NumPy's <c>None</c>:
        ///     raise).</param>
        /// <returns>The <see cref="DirectILKernelGenerator"/> <c>ChooseMode*</c> constant.</returns>
        /// <exception cref="ValueError">See <see cref="np.ClipmodeConverter"/>: a misspelled mode (<c>Use one of 'clip',
        ///     'raise', or 'wrap' for clip mode</c>) or any other string (<c>clipmode must be one of 'clip', 'raise', or
        ///     'wrap' (got 'bogus')</c>).</exception>
        /// <remarks>The mapping is spelled out, not assumed: the shared numbering and the choose kernel's are equal
        ///     today, and a switch keeps a later renumbering of either from silently selecting the wrong mode.</remarks>
        private static int ParseChooseMode(string mode) => np.ClipmodeConverter(mode) switch
        {
            np.ClipModeWrap => DirectILKernelGenerator.ChooseModeWrap,
            np.ClipModeClip => DirectILKernelGenerator.ChooseModeClip,
            _ => DirectILKernelGenerator.ChooseModeRaise,
        };

        // ---- np.compress: PyArray_Compress ------------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_Compress</c> (<c>item_selection.c</c>): the condition's rank, the axis check
        ///     <c>PyArray_TakeFrom</c> makes next, then the fused mask-driven gather or NumPy's own chain,
        ///     <c>np.flatnonzero(condition)</c> into <see cref="TakeFrom"/> in raise mode. The body of
        ///     <see cref="np.compress(NDArray, NDArray, int?, NDArray)"/> once its class boundaries and null checks have run.
        /// </summary>
        /// <param name="condition">The condition, not null: a 1-D array of any dtype, read by truth value.</param>
        /// <param name="a">The source array, not null; any layout.</param>
        /// <param name="axis">The axis to select along, or null to select from the flattened array.</param>
        /// <param name="out">The caller's destination, or null.</param>
        /// <returns>The selected slices: <paramref name="out"/> itself when given, else a new array of
        ///     <paramref name="a"/>'s dtype.</returns>
        /// <exception cref="ValueError"><paramref name="condition"/> is not 1-D, or <paramref name="out"/> is of
        ///     another shape or read-only.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <c>max(a.ndim, 1)</c>.</exception>
        /// <exception cref="TypeError"><paramref name="out"/>'s dtype does not cast <c>'safe'</c>ly.</exception>
        /// <exception cref="IndexError">A true entry of <paramref name="condition"/> past the end of the axis.</exception>
        /// <remarks>
        ///     The flat indices <c>np.flatnonzero</c> returns are disposed here, exceptions included; the fused path
        ///     allocates only its result.
        /// </remarks>
        internal static NDArray Compress(NDArray condition, NDArray a, int? axis, NDArray @out)
        {
            // NumPy hard-requires 1-D condition; 0-D and 2-D+ both fail here (PyArray_Compress' ValueError).
            if (condition.ndim != 1)
                throw new ValueError(
                    "condition must be a 1-d array",
                    nameof(condition));

            // PyArray_TakeFrom's PyArray_CheckAxis, which NumPy reaches next: the axis addresses rank max(ndim, 1) (a 0-d
            // input ravels first), checked before either path reads anything — the fast path treated a 0-d input as one
            // element for ANY axis.
            if (axis is not null)
                Backends.DefaultEngine.check_and_adjust_axis(Math.Max(a.ndim, 1), axis.Value);

            // Fast path: bool cond, contig src, no out= override (so result dtype
            // == src dtype). Wires straight into the fused gather kernel.
            if (TryCompressFast(condition, a, axis, @out, out var fast))
                return fast;

            // Generic path: flatnonzero handles non-bool / strided cond; take
            // handles non-contig src, out= dispatch with safe-cast validation,
            // and OOB-True via RAISE mode.
            var indices = np.flatnonzero(condition);
            try
            {
                return np.take(a, indices, axis, @out);
            }
            finally
            {
                indices.Dispose();
            }
        }

        /// <summary>
        ///     Fused fast path for <c>np.compress</c>. Computes outer/inner
        ///     factorisation around the requested axis and runs the
        ///     <see cref="FilterAxisKernel"/> in one pass. Returns <c>false</c>
        ///     when preconditions aren't met (bool cond, contig src, etc.) so
        ///     the caller falls through to the flatnonzero+take chain.
        /// </summary>
        private static unsafe bool TryCompressFast(
            NDArray condition, NDArray a, int? axis, NDArray @out, out NDArray result)
        {
            result = null;
            // A reference-holding source (PyDataType_REFCHK; StringDType) takes the generic path (flatnonzero + np.take,
            // whose StringDType gather duplicates the cells); this kernel copies elements as raw bytes.
            if (a.dtype.Meta.HasReferences) return false;
            if (condition.GetTypeCode != NPTypeCode.Boolean) return false;
            if (!condition.Shape.IsContiguous) return false;
            if (!a.Shape.IsContiguous) return false;

            // out= path needs the take-level out dispatch (shape check + safe-cast).
            // Skip fast path when out is supplied with mismatched dtype; for matching
            // dtype we'd still need shape validation, so just defer to generic path.
            if (!(@out is null)) return false;

            var countKernel = DirectILKernelGenerator.GetArgwhereCountKernel(typeof(bool));
            // innerSize bucketing is decided once we know the axis factorisation.
            // For axis=k with innerCount>1 the slab spans multiple elements; we
            // pass innerCount*elemBytes — typically large, lands in the bulk
            // cpblk variant. For axis=None / 1-D src / axis=last the inner is
            // just one element (typed copy variant — significantly faster).
            if (countKernel == null) return false;

            // Compute outerSize × axisSize × innerSize factorisation around axis.
            // axis=null flattens a (treat as 1-D of a.size).
            // 0-d source with axis=null → 1-element 1-D pseudo-axis (handled below).
            long outerSize, axisSize, innerCount;
            long[] outDims;

            if (a.ndim == 0)
            {
                // 0-d treated as 1-element 1-D; behaviour identical to axis=None.
                outerSize = 1;
                axisSize = 1;
                innerCount = 1;
                outDims = new long[1] { 0 }; // placeholder; will overwrite with count below
            }
            else if (axis is null)
            {
                outerSize = 1;
                axisSize = a.size;
                innerCount = 1;
                outDims = new long[1] { 0 };
            }
            else
            {
                int ax = Backends.DefaultEngine.check_and_adjust_axis(a.ndim, axis.Value);

                outerSize = 1;
                for (int d = 0; d < ax; d++) outerSize *= a.Shape.dimensions[d];
                axisSize = a.Shape.dimensions[ax];
                innerCount = 1;
                for (int d = ax + 1; d < a.ndim; d++) innerCount *= a.Shape.dimensions[d];

                // Output shape: a.shape with axis dim replaced by N (count).
                outDims = new long[a.ndim];
                for (int d = 0; d < a.ndim; d++) outDims[d] = a.Shape.dimensions[d];
            }

            long elemBytes = a.dtypesize;
            long innerBytes = innerCount * elemBytes;
            long maskSize = condition.size;
            long effectiveScan = Math.Min(maskSize, axisSize);
            byte* maskPtr = (byte*)condition.Storage.Address + condition.Shape.offset;

            // OOB-True detection: cond longer than axisSize and any True past
            // axisSize would index OOB on `a` → mirror NumPy's take(nonzero(cond), axis) IndexError: the
            // FIRST out-of-bounds index is the first True at or past the axis end, named with its axis
            // (axis 0 for the raveled axis=None / 0-d forms); an empty axis is take's "empty axes" text.
            if (maskSize > axisSize)
            {
                var tail = new ReadOnlySpan<byte>(maskPtr + axisSize, checked((int)(maskSize - axisSize)));
                int firstTrue = tail.IndexOfAnyExcept((byte)0);
                if (firstTrue >= 0)
                {
                    if (axisSize == 0)
                        throw new IndexOutOfRangeException("cannot do a non-empty take from an empty axes.");
                    int axisNumber = a.ndim == 0 || axis is null ? 0 : (axis.Value < 0 ? axis.Value + a.ndim : axis.Value);
                    throw new IndexOutOfRangeException(
                        $"index {axisSize + firstTrue} is out of bounds for axis {axisNumber} with size {axisSize}");
                }
            }

            // Pass 1: popcount mask over the effective scan range.
            long count = countKernel(maskPtr, effectiveScan);

            // Build output shape: axis dim → count. For axis=None / 0-d, shape is (count,).
            if (a.ndim == 0 || axis is null)
                outDims = new long[1] { count };
            else
            {
                int ax = axis.Value < 0 ? axis.Value + a.ndim : axis.Value;
                outDims[ax] = count;
            }

            result = new NDArray(a.dtype, Shape.Adopt(outDims), false); // descriptor: a string's width

            if (count == 0 || outerSize == 0 || innerCount == 0)
                return true;

            // Source pointer with the (always-contig here) offset; 0-d source has
            // no offset stride contribution.
            byte* srcPtr = (byte*)a.Storage.Address + a.Shape.offset * elemBytes;
            byte* dstPtr = (byte*)result.Storage.Address;

            long srcOuterStride = axisSize * innerBytes;
            long dstOuterStride = count * innerBytes;

            // Pick the typed-or-bulk variant by innerBytes — single-element
            // slabs (innerCount==1) lift to the typed Ldind/Stind path.
            var filterKernel = DirectILKernelGenerator.GetFilterAxisKernel(innerBytes);
            if (filterKernel == null) return false;

            long written = filterKernel(
                srcPtr, maskPtr, effectiveScan,
                outerSize, srcOuterStride, dstOuterStride,
                innerBytes,
                dstPtr);

            if (written != count)
                throw new InvalidOperationException(
                    $"compress: filter kernel wrote {written}, expected {count}");

            return true;
        }

        // ---- np.resize: fromnumeric.py's resize (the data repeated; ndarray.resize is Resize) ---------------------------

        /// <summary>
        ///     The statements of NumPy's <c>numpy.resize</c> (<c>_core/fromnumeric.py</c>): the dimensions checked, the
        ///     source raveled, an empty source or result zero-filled (<c>np.zeros_like(a, shape=new_shape)</c>), and
        ///     otherwise the raveled source repeated into the new size. NumPy concatenates <c>ceil(new_size / a.size)</c>
        ///     copies and slices them; this allocates exactly the new size and fills it by doubling byte copies. The body
        ///     of <see cref="np.resize(NDArray, Shape)"/> once its class boundaries and null check have run.
        /// </summary>
        /// <param name="a">The source array, not null; any layout (read in C order).</param>
        /// <param name="new_shape">The new shape: only its dimensions are read.</param>
        /// <returns>A C-contiguous array of <paramref name="a"/>'s dtype: NumPy's view of a fresh buffer
        ///     (<c>flags.owndata</c> false), or for an empty source or result an owning array of zeros.</returns>
        /// <exception cref="ArgumentException">A dimension of <paramref name="new_shape"/> is negative.</exception>
        /// <remarks>
        ///     <c>np.resize</c> is not <c>ndarray.resize</c>: that one is <see cref="Resize"/> (NumPy's
        ///     <c>PyArray_Resize</c>), which works in place and fills a grown array with zeros.
        /// </remarks>
        internal static unsafe NDArray ResizeRepeating(NDArray a, Shape new_shape)
        {
            // Validate dimensions and compute the target element count. NumPy's np.resize
            // rejects negative dims with this exact message (distinct from ndarray.resize's
            // "negative dimensions not allowed").
            long newSize = 1;
            var newDims = new_shape.dimensions ?? Array.Empty<long>();
            for (int i = 0; i < newDims.Length; i++)
            {
                if (newDims[i] < 0)
                    throw new ArgumentException("all elements of `new_shape` must be non-negative");
                newSize *= newDims[i];
            }

            // The result is always C-contiguous (NumPy reshapes the concatenated 1-D read-out),
            // so normalize to C-order dims — independent of any strides carried by new_shape.
            var outShape = Shape.Adopt(newDims);

            // A reference-holding source (PyDataType_REFCHK; StringDType): the same tiling runs on its cell view and the
            // cells are duplicated (NDVStringMove), as NumPy's concatenate copies them through the dtype; an empty source or
            // result is NumPy's zeros_like ('' elements).
            if (a.dtype.Meta.HasReferences)
            {
                if (a.size == 0 || newSize == 0)
                    return np.zeros(outShape, a.dtype);
                return Backends.Strings.NDVStringMove.Gather(a, cells => np.resize(cells, new_shape));
            }

            // Flatten to a 1-D C-order read-out. ravel returns a view when a is already
            // contiguous, else a fresh contiguous copy — either way a dense run of `a.size`
            // elements starting at flat.Shape.offset. Disposed after the tiling copy.
            using var flat = np.ravel(a);
            long srcSize = flat.size;

            // First case must zero-fill (empty source has nothing to repeat); the second
            // would repeat zero times. NumPy: np.zeros_like(a, shape=new_shape).
            if (srcSize == 0 || newSize == 0)
                return np.zeros(outShape, a.dtype); // the descriptor: a string result is NUL-filled at a's width

            // Allocate the exact-sized C-contiguous output and tile the source bytes into it.
            // This beats NumPy's concatenate((a,)*repeats)[:new_size] which over-allocates a
            // repeats*size buffer then slices — we allocate exactly new_size and fill by
            // doubling block copies (dtype-agnostic raw memory, O(new_size) bytes,
            // O(log(new_size/size)) memcpy calls).
            var result = new NDArray(a.dtype, outShape, fillZeros: false);

            long itemsize = a.dtypesize;
            byte* src = flat.Storage.Address + (long)flat.Shape.offset * itemsize;
            byte* dst = result.Storage.Address;
            long totalBytes = newSize * itemsize;
            long srcBytes = srcSize * itemsize;

            // Seed with the source (truncated when shrinking), then repeatedly double the
            // filled region from dst onto itself until the whole output is covered.
            long filled = Math.Min(srcBytes, totalBytes);
            Buffer.MemoryCopy(src, dst, totalBytes, filled);
            while (filled < totalBytes)
            {
                long chunk = Math.Min(filled, totalBytes - filled);
                Buffer.MemoryCopy(dst, dst + filled, totalBytes - filled, chunk);
                filled += chunk;
            }

            // NumPy returns reshape(concatenate(...)[:new_size], new_shape): a view, not the buffer's owner. The view
            // takes its own reference on the buffer, so the owning wrapper is released at once.
            var view = new NDArray(result.Storage.Alias(result.Shape)) { TensorEngine = a.TensorEngine };
            result.Dispose();
            return view;
        }
    }
}
