using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.pad ==============================
        // Pad an N-D array along every axis. Eleven string modes + user callable.
        //
        // NumPy 2.4.2 reference: numpy/lib/_arraypad_impl.py
        //
        // Architecture mirrors NumPy's _arraypad_impl.py — the helper functions
        // (_AsPairs, _PadSimple, _SliceAtAxis, _ViewRoi, _SetPadArea, _GetEdges,
        // _GetLinearRamps, _GetStats, _SetReflectBoth, _SetWrapBoth) have a 1:1
        // correspondence with their underscore-prefixed Python counterparts.
        // This is intentional — the corner-propagation logic that turns iterative
        // axis-padding into N-D padding is load-bearing and brittle, so the port
        // tracks the upstream structure exactly.
        //
        // Performance: every bulk operation routes through existing IL kernels —
        //   * np.empty for allocation (single pointer alloc)
        //   * IArraySlice.Fill for uniform-value PadSimple (InitBlockUnaligned/unrolled scalar)
        //   * padded[band] = src routes through NDIter.Copy → StridedCastKernel
        //     (cpblk contig path, per-row memcpy strided path, "convert once +
        //      memcpy" broadcast fast path for 1-thick stripe and scalar fills)
        //   * stat modes use existing axis-reduction IL kernels (np.max/min/mean/median)
        //
        // No new IL emission is required for any mode.

        public delegate void PadFunc(NDArray vector, (int before, int after) padWidth, int axis, object kwargs);

        // ---------------------------- public overloads ----------------------------

        /// <summary>
        ///     Pad an array. Scalar <paramref name="pad_width"/> applies <c>(pad_width, pad_width)</c>
        ///     to every axis.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.pad.html</remarks>
        public static NDArray pad(NDArray array, int pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadImpl(array, (object)pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        public static NDArray pad(NDArray array, int[] pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        public static NDArray pad(NDArray array, int[,] pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        public static NDArray pad(NDArray array, (int before, int after) pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        /// <summary>
        ///     Pad an array by a per-axis dict (NumPy 2.3+): each key names an axis (negative counts from the end), each
        ///     value pads it by an integer on both sides or by a <c>(before, after)</c> tuple; an axis without a key is not
        ///     padded.
        /// </summary>
        /// <param name="array">The array to pad.</param>
        /// <param name="pad_width">The per-axis widths, read in the dictionary's order (a later key for the same axis wins).</param>
        /// <param name="mode">The pad mode, as for the other overloads.</param>
        /// <param name="constant_values">The fill of <c>'constant'</c>.</param>
        /// <param name="end_values">The ends of <c>'linear_ramp'</c>.</param>
        /// <param name="stat_length">The window of the statistic modes.</param>
        /// <param name="reflect_type"><c>'even'</c> or <c>'odd'</c> for <c>'reflect'</c> / <c>'symmetric'</c>.</param>
        /// <returns>The padded array.</returns>
        /// <exception cref="AssertionError">A value that is neither an integer nor a 2-tuple of integers (NumPy's
        ///     <c>typing.assert_never</c>; a typed C# array is an ndarray, which NumPy refuses too).</exception>
        /// <exception cref="IndexError">A key outside <c>[-ndim, ndim)</c> ("list assignment index out of range", NumPy's
        ///     list assignment - not an AxisError). Every key of a 0-d array is one.</exception>
        /// <exception cref="TypeError">The widths are no integer table: a 0-d array's empty dict, every axis given bools,
        ///     or a width outside int64.</exception>
        /// <exception cref="ValueError">A negative width (checked after every key, as NumPy's <c>_as_pairs</c> does).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.pad.html</remarks>
        public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);
        }

        // Callable mode (overloads matching each pad_width shape).
        public static NDArray pad(NDArray array, int pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadCallableImpl(array, (object)pad_width, mode, kwargs);
        }

        public static NDArray pad(NDArray array, int[] pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadCallableImpl(array, pad_width, mode, kwargs);
        }

        public static NDArray pad(NDArray array, int[,] pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadCallableImpl(array, pad_width, mode, kwargs);
        }

        public static NDArray pad(NDArray array, (int before, int after) pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadCallableImpl(array, pad_width, mode, kwargs);
        }

        /// <summary>
        ///     Pad an array by a per-axis dict with a callable mode: the dict is read as in the string-mode dict overload,
        ///     then <paramref name="mode"/> fills each 1-D vector's pad area.
        /// </summary>
        /// <param name="array">The array to pad.</param>
        /// <param name="pad_width">The per-axis widths, read in the dictionary's order (a later key for the same axis wins).</param>
        /// <param name="mode">The callable that writes each vector's pad area in place.</param>
        /// <param name="kwargs">Passed to <paramref name="mode"/> unchanged.</param>
        /// <returns>The padded array.</returns>
        /// <exception cref="AssertionError">A value that is neither an integer nor a 2-tuple of integers.</exception>
        /// <exception cref="IndexError">A key outside <c>[-ndim, ndim)</c> ("list assignment index out of range").</exception>
        /// <exception cref="TypeError">The widths are no integer table (a 0-d array's empty dict, every axis given bools,
        ///     a width outside int64).</exception>
        /// <exception cref="ValueError">A negative width.</exception>
        public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, PadFunc mode, object kwargs = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(array);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return PadCallableImpl(array, pad_width, mode, kwargs);
        }

        // ---------------------------- top-level dispatcher ----------------------------

        // Scoped here (not per-overload): every value-mode pad overload funnels through this
        // dispatcher, so one ambient scope reclaims the whole mode-body call tree (the
        // `2 * edge` temps in _SetReflectBoth, BuildLinearRamp's reshape, …) for all of them.
        // PadCallableImpl is deliberately NOT scoped: its tree disposes everything it creates,
        // and an ambient scope there would reach into the user's PadFunc callback and dispose
        // arrays the callback creates for its own use.
        [NDScoped]
        private static NDArray PadImpl(NDArray array, object pad_width, string mode,
            object constant_values, object end_values, object stat_length, string reflect_type)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (mode is null) throw new ArgumentNullException(nameof(mode));

            int ndim = Math.Max(array.ndim, 1); // 0-D treated as 1-D for pad_width normalization
            // A dict pad_width runs NumPy's dict branch first, against the array's TRUE rank (a 0-d array's list has no
            // slot, so every key is an IndexError there); the (ndim, 2) table it builds then goes through _as_pairs
            // like any other pad_width.
            if (pad_width is IDictionary<int, object> widths)
                pad_width = DictPadWidth(widths, array.ndim);
            long[,] padPairs = _AsPairs(pad_width, ndim, asIndex: true);

            ValidateModeKwargs(mode, constant_values, end_values, stat_length, reflect_type);

            // No-op fast path: every pad value is zero. Still allocates a new array per NumPy. NumPy itself has no such
            // shortcut — its mode bodies run for every axis, widths of 0 included — so the copy is taken only where no
            // mode body can raise (ZeroWidthIsCopy): linear_ramp's linspace over text, the stat modes' reductions over
            // text or with stat_length=0, and a constant_values array that does not broadcast all raise in NumPy.
            if (AllPairsZero(padPairs) && ZeroWidthIsCopy(mode, constant_values))
                return array.copy();

            // Empty-axis guard: only 'constant' and 'empty' may extend an empty axis. Past it NumPy returns _pad_simple's
            // array as it is ("_pad_simple already returned the correct result"): no other mode body runs over an empty
            // array, so none of their checks (a text linear_ramp's linspace, a text stat) can raise there.
            if (array.size == 0 && mode != "constant" && mode != "empty")
            {
                for (int ax = 0; ax < array.ndim; ax++)
                {
                    if (array.Shape.dimensions[ax] == 0 && (padPairs[ax, 0] != 0 || padPairs[ax, 1] != 0))
                        throw new ArgumentException(
                            $"can't extend empty axis {ax} using modes other than 'constant' or 'empty'");
                }
                if (mode is "edge" or "wrap" or "reflect" or "symmetric" or "maximum" or "minimum" or "mean" or "median"
                    or "linear_ramp")
                    return _PadSimple(array, padPairs, fillValue: null).padded;
            }

            switch (mode)
            {
                case "constant":
                    return PadConstant(array, padPairs, constant_values);
                case "edge":
                    return PadEdge(array, padPairs);
                case "empty":
                    return _PadSimple(array, padPairs, fillValue: null).padded;
                case "wrap":
                    return PadWrap(array, padPairs);
                case "reflect":
                    return PadReflectOrSymmetric(array, padPairs, reflect_type ?? "even", includeEdge: false);
                case "symmetric":
                    return PadReflectOrSymmetric(array, padPairs, reflect_type ?? "even", includeEdge: true);
                case "maximum":
                case "minimum":
                case "mean":
                case "median":
                    return PadStat(array, padPairs, stat_length, mode);
                case "linear_ramp":
                    return PadLinearRamp(array, padPairs, end_values);
                default:
                    throw new ArgumentException($"mode '{mode}' is not supported", nameof(mode));
            }
        }

        private static NDArray PadCallableImpl(NDArray array, object pad_width, PadFunc func, object kwargs)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (func is null) throw new ArgumentNullException(nameof(func));
            int ndim = Math.Max(array.ndim, 1);
            // The dict branch first, as in PadImpl (NumPy's pad runs it before it looks at the mode).
            if (pad_width is IDictionary<int, object> widths)
                pad_width = DictPadWidth(widths, array.ndim);
            long[,] padPairs = _AsPairs(pad_width, ndim, asIndex: true);
            if (AllPairsZero(padPairs)) return array.copy();

            // NumPy: zero-fill the padded buffer, then visit each 1-D vector along
            // each axis (via ndindex over view.shape[:-1]) and let user code mutate.
            var (padded, _) = _PadSimple(array, padPairs, fillValue: BoxedZero(array.GetTypeCode));
            if (array.ndim == 0) return padded;

            for (int axis = 0; axis < padded.ndim; axis++)
            {
                // Move pad axis to the end so we can iterate the outer (ndim-1)
                // dims and pass each 1-D vector by reference to the user function.
                var view = np.moveaxis(padded, axis, padded.ndim - 1);
                int outerNdim = view.ndim - 1;
                long axisLen = view.Shape.dimensions[outerNdim];
                long left = padPairs[axis, 0];
                long right = padPairs[axis, 1];

                if (outerNdim == 0)
                {
                    // 1-D array: a single vector — pass the whole view.
                    func(view, ((int)left, (int)right), axis, kwargs);
                    view.Dispose();
                    continue;
                }

                // ndindex over view.shape[:-1]: iterate every multi-index, slice
                // the 1-D vector at that position, call user-func, dispose.
                var coords = new long[outerNdim];
                long totalOuter = 1;
                for (int i = 0; i < outerNdim; i++) totalOuter *= view.Shape.dimensions[i];
                for (long idx = 0; idx < totalOuter; idx++)
                {
                    var slices = new Slice[view.ndim];
                    for (int i = 0; i < outerNdim; i++) slices[i] = Slice.Index(coords[i]);
                    slices[outerNdim] = Slice.All;
                    var vec = view[slices];
                    try { func(vec, ((int)left, (int)right), axis, kwargs); }
                    finally { vec.Dispose(); }

                    // Advance coords (row-major).
                    for (int d = outerNdim - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        if (coords[d] < view.Shape.dimensions[d]) break;
                        coords[d] = 0;
                    }
                }
                view.Dispose();
            }
            return padded;
        }

        // ---------------------------- mode bodies ----------------------------

        private static NDArray PadConstant(NDArray array, long[,] padPairs, object constantValues)
        {
            // Text takes NumPy's assignment of each constant into the string array (np.copyto, unsafe): a str / bytes
            // constant is stored as text, a number would first be converted to its text.
            if (array.typecode == NPTypeCode.String)
                return PadConstantText(array, padPairs, constantValues);

            int ndim = array.ndim;
            // Normalize constant_values to a (ndim, 2) box-array. NumPy default is 0.
            object cv = constantValues ?? BoxedZero(array.GetTypeCode);
            object[,] valuePairs = _AsPairsValues(cv, ndim);

            // NumPy strategy: allocate uninitialized, center-copy the original, then
            // fill each axis's pad band via _SetPadArea. The corner-propagation logic
            // in _ViewRoi ensures corners are written by axis 0 and not re-overwritten.
            //
            // Note: a "uniform-fill-whole-buffer" shortcut is tempting (one Fill of N
            // elements vs ~2*N_pad_band tiny writes), but for typical pad sizes
            // (pad << array size) the whole-buffer fill is the dominant cost and
            // makes constant mode slower than NumPy. Skip the shortcut entirely.
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);
            for (int axis = 0; axis < ndim; axis++)
            {
                var roi = _ViewRoi(padded, originalSlice, axis);
                long left = padPairs[axis, 0];
                long right = padPairs[axis, 1];
                object lv = CastBoxToDType(valuePairs[axis, 0], array.GetTypeCode);
                object rv = CastBoxToDType(valuePairs[axis, 1], array.GetTypeCode);
                _SetPadArea(roi, axis, (left, right), (lv, rv));
                roi.Dispose();
            }
            return padded;
        }

        /// <summary>
        ///     <c>np.pad(..., mode='constant')</c> of a fixed-width string array: NumPy's same steps — allocate, copy the
        ///     original into the center, then assign each axis's before / after constant into its pad band, the bands of
        ///     earlier axes covering the corners. Each constant is assigned as text: a <see cref="string"/> or <c>byte[]</c>
        ///     is encoded once into a 0-d string array and broadcast by the string copy route, which cuts it to the array's
        ///     width (<c>'zzzzz'</c> into <c>&lt;U3</c> is <c>'zzz'</c>) and converts <c>S ↔ U</c> with the ASCII checks; a
        ///     number (the default constant is 0) is written as its <c>str()</c> (<c>np.pad(u, 1)</c> pads with <c>'0'</c>).
        /// </summary>
        /// <param name="array">The string array to pad.</param>
        /// <param name="padPairs">The normalized (before, after) widths per axis.</param>
        /// <param name="constantValues">The constant(s) in any of np.pad's spellings, or null for NumPy's default 0.</param>
        /// <returns>The padded array (the input's dtype, a fresh buffer).</returns>
        /// <exception cref="UnicodeEncodeError">A non-ASCII str constant into an <c>S</c> array.</exception>
        /// <exception cref="UnicodeDecodeError">A non-ASCII bytes constant into a <c>U</c> array.</exception>
        private static NDArray PadConstantText(NDArray array, long[,] padPairs, object constantValues)
        {
            int ndim = array.ndim;
            // NumPy's default constant is the number 0 (np.array(0), an int64 scalar), stored in a string array as the
            // text '0' — its str().
            constantValues ??= 0L;

            // NumPy's _as_pairs runs over the TRUE rank: a 0-d input validates the values' shape against (0, 2) and then
            // has no axis whose constant would be converted.
            object[,] valuePairs = _AsPairsValues(constantValues, ndim);

            // Encode every constant up front, before any write, so a refused value leaves no half-built result behind
            // (NumPy validates the values while converting them, before it allocates).
            var encoded = new NDArray[ndim, 2];
            for (int axis = 0; axis < encoded.GetLength(0); axis++)
                for (int side = 0; side < 2; side++)
                {
                    object v = valuePairs[axis, side];
                    // Text stays text; a number becomes its 0-d array, which the area assignment writes as its str()
                    // (NumPy's _as_pairs makes every constant a NumPy scalar through np.array).
                    encoded[axis, side] = v is NDArray given ? given
                        : v is string or byte[] or char ? Backends.Strings.NDStringArrays.FromScalar(v, null)
                        : v is null ? Backends.Strings.NDStringArrays.FromScalar(null, array.dtype)
                        : NDArray.Scalar(v);
                }

            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);
            for (int axis = 0; axis < ndim; axis++)
            {
                var roi = _ViewRoi(padded, originalSlice, axis);
                _SetPadArea(roi, axis, (padPairs[axis, 0], padPairs[axis, 1]), (encoded[axis, 0], encoded[axis, 1]));
                roi.Dispose();
            }
            return padded;
        }

        private static NDArray PadEdge(NDArray array, long[,] padPairs)
        {
            int ndim = array.ndim;
            if (ndim == 0)
                return array.copy(); // 0-D has no axis to pad
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);
            for (int axis = 0; axis < ndim; axis++)
            {
                long left = padPairs[axis, 0];
                long right = padPairs[axis, 1];
                if (left == 0 && right == 0) continue;
                var roi = _ViewRoi(padded, originalSlice, axis);
                var (leftEdge, rightEdge) = _GetEdges(roi, axis, (left, right));
                try
                {
                    _SetPadArea(roi, axis, (left, right), (leftEdge, rightEdge));
                }
                finally
                {
                    leftEdge.Dispose();
                    rightEdge.Dispose();
                    roi.Dispose();
                }
            }
            return padded;
        }

        private static NDArray PadWrap(NDArray array, long[,] padPairs)
        {
            int ndim = array.ndim;
            if (ndim == 0) return array.copy();
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);
            for (int axis = 0; axis < ndim; axis++)
            {
                long left = padPairs[axis, 0], right = padPairs[axis, 1];
                if (left == 0 && right == 0) continue;
                var roi = _ViewRoi(padded, originalSlice, axis);
                // Period = current valid length on this axis = roi.shape[axis] - left - right.
                // Captured BEFORE the iterative shrink loop because the valid area never grows
                // for wrap — only the pad band shrinks.
                long originalPeriod = roi.Shape.dimensions[axis] - right - left;
                try
                {
                    while (left > 0 || right > 0)
                    {
                        (left, right) = _SetWrapBoth(roi, axis, (left, right), originalPeriod);
                    }
                }
                finally { roi.Dispose(); }
            }
            return padded;
        }

        private static NDArray PadReflectOrSymmetric(NDArray array, long[,] padPairs, string method, bool includeEdge)
        {
            int ndim = array.ndim;
            if (ndim == 0) return array.copy();
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);
            for (int axis = 0; axis < ndim; axis++)
            {
                long left = padPairs[axis, 0], right = padPairs[axis, 1];
                if (left == 0 && right == 0) continue;

                // Singleton-axis fallback: legacy NumPy behavior — extending a length-1
                // axis with reflect/symmetric falls back to edge padding (there's nothing
                // to reflect). Operates on the full padded (not roi) per upstream code.
                if (array.Shape.dimensions[axis] == 1)
                {
                    var (leftEdge, rightEdge) = _GetEdges(padded, axis, (left, right));
                    try { _SetPadArea(padded, axis, (left, right), (leftEdge, rightEdge)); }
                    finally { leftEdge.Dispose(); rightEdge.Dispose(); }
                    continue;
                }

                var roi = _ViewRoi(padded, originalSlice, axis);
                long originalPeriod = array.Shape.dimensions[axis];
                try
                {
                    while (left > 0 || right > 0)
                    {
                        (left, right) = _SetReflectBoth(roi, axis, (left, right), method, originalPeriod, includeEdge);
                    }
                }
                finally { roi.Dispose(); }
            }
            return padded;
        }

        /// <summary>
        ///     One iteration of reflect/symmetric padding along <paramref name="axis"/>.
        ///     Returns the residual <c>(left, right)</c> pad amounts — when both are 0
        ///     the loop terminates; otherwise the caller iterates with the residuals,
        ///     each time reading from the (now-larger) valid region.
        ///
        ///     Mirrors <c>_set_reflect_both</c> in <c>numpy/lib/_arraypad_impl.py</c>.
        /// </summary>
        [NDScopedCovered] // only caller: PadReflectOrSymmetric ← [NDScoped] PadImpl (the `2 * edge` odd-mode temp)
        private static (long left, long right) _SetReflectBoth(
            NDArray padded, int axis, (long left, long right) width, string method, long originalPeriod, bool includeEdge)
        {
            long leftPad = width.left, rightPad = width.right;
            long oldLength = padded.Shape.dimensions[axis] - rightPad - leftPad;
            long edgeOffset;
            if (includeEdge)
            {
                // Symmetric: period must be multiple of original to avoid wrapping a partial section.
                oldLength = oldLength / originalPeriod * originalPeriod;
                edgeOffset = 1;
            }
            else
            {
                // Reflect: period multiple of (original - 1) (edges are shared between cycles).
                oldLength = ((oldLength - 1) / (originalPeriod - 1)) * (originalPeriod - 1) + 1;
                edgeOffset = 0;
                oldLength -= 1; // edge omitted from the chunk
            }

            if (leftPad > 0)
            {
                long chunkLength = Math.Min(oldLength, leftPad);
                long stop = leftPad - edgeOffset;
                long start = stop + chunkLength;
                var leftSlices = _SliceAtAxis(new Slice(start, stop, -1), axis, padded.ndim);
                NDArray leftChunk = padded[leftSlices];
                NDArray odd = null;
                try
                {
                    if (method == "odd")
                    {
                        var edgeSlices = _SliceAtAxis(new Slice(leftPad, leftPad + 1), axis, padded.ndim);
                        using var edge = padded[edgeSlices];
                        odd = 2 * edge - leftChunk;
                    }

                    long padStart = leftPad - chunkLength;
                    long padStop = leftPad;
                    var padSlices = _SliceAtAxis(new Slice(padStart, padStop), axis, padded.ndim);
                    AssignSliceValue(padded, padSlices, odd ?? leftChunk);
                }
                finally { leftChunk.Dispose(); odd?.Dispose(); }
                leftPad -= chunkLength;
            }

            if (rightPad > 0)
            {
                long chunkLength = Math.Min(oldLength, rightPad);
                long start = -rightPad + edgeOffset - 2;
                long stop = start - chunkLength;
                var rightSlices = _SliceAtAxis(new Slice(start, stop, -1), axis, padded.ndim);
                NDArray rightChunk = padded[rightSlices];
                NDArray odd = null;
                try
                {
                    if (method == "odd")
                    {
                        var edgeSlices = _SliceAtAxis(new Slice(-rightPad - 1, -rightPad), axis, padded.ndim);
                        using var edge = padded[edgeSlices];
                        odd = 2 * edge - rightChunk;
                    }

                    long padStart = padded.Shape.dimensions[axis] - rightPad;
                    long padStop = padStart + chunkLength;
                    var padSlices = _SliceAtAxis(new Slice(padStart, padStop), axis, padded.ndim);
                    AssignSliceValue(padded, padSlices, odd ?? rightChunk);
                }
                finally { rightChunk.Dispose(); odd?.Dispose(); }
                rightPad -= chunkLength;
            }

            return (leftPad, rightPad);
        }

        /// <summary>
        ///     One iteration of wrap padding along <paramref name="axis"/>. Returns
        ///     residual <c>(left, right)</c> pad amounts so the caller can iterate
        ///     when the requested pad exceeds <paramref name="originalPeriod"/>.
        ///
        ///     Mirrors <c>_set_wrap_both</c> in <c>numpy/lib/_arraypad_impl.py</c>.
        /// </summary>
        private static (long left, long right) _SetWrapBoth(
            NDArray padded, int axis, (long left, long right) width, long originalPeriod)
        {
            long leftPad = width.left, rightPad = width.right;
            long period = padded.Shape.dimensions[axis] - rightPad - leftPad;
            period = period / originalPeriod * originalPeriod; // ensure multiple of original

            long newLeftPad = 0, newRightPad = 0;

            if (leftPad > 0)
            {
                long sliceEnd = leftPad + period;
                long sliceStart = sliceEnd - Math.Min(period, leftPad);
                var rightSlices = _SliceAtAxis(new Slice(sliceStart, sliceEnd), axis, padded.ndim);
                NDArray rightChunk = padded[rightSlices];
                try
                {
                    Slice[] padSlices;
                    if (leftPad > period)
                    {
                        padSlices = _SliceAtAxis(new Slice(leftPad - period, leftPad), axis, padded.ndim);
                        newLeftPad = leftPad - period;
                    }
                    else
                    {
                        padSlices = _SliceAtAxis(new Slice(null, leftPad), axis, padded.ndim);
                    }
                    AssignSliceValue(padded, padSlices, rightChunk);
                }
                finally { rightChunk.Dispose(); }
            }

            if (rightPad > 0)
            {
                long sliceStart = -rightPad - period;
                long sliceEnd = sliceStart + Math.Min(period, rightPad);
                var leftSlices = _SliceAtAxis(new Slice(sliceStart, sliceEnd), axis, padded.ndim);
                NDArray leftChunk = padded[leftSlices];
                try
                {
                    Slice[] padSlices;
                    if (rightPad > period)
                    {
                        padSlices = _SliceAtAxis(new Slice(-rightPad, -rightPad + period), axis, padded.ndim);
                        newRightPad = rightPad - period;
                    }
                    else
                    {
                        padSlices = _SliceAtAxis(new Slice(-rightPad, null), axis, padded.ndim);
                    }
                    AssignSliceValue(padded, padSlices, leftChunk);
                }
                finally { leftChunk.Dispose(); }
            }

            return (newLeftPad, newRightPad);
        }

        private static NDArray PadStat(NDArray array, long[,] padPairs, object statLength, string mode)
        {
            int ndim = array.ndim;
            // -1 sentinel == "use full valid axis length". NumPy's _as_pairs(stat_length, ndim, as_index=True) runs over the
            // TRUE rank before any axis is visited, so a 0-d input still refuses a negative or badly shaped stat_length.
            long[,] lengthPairs = _AsPairs(statLength, ndim, asIndex: true);
            if (ndim == 0) return array.copy();
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);

            for (int axis = 0; axis < ndim; axis++)
            {
                long left = padPairs[axis, 0], right = padPairs[axis, 1];
                // NumPy computes each axis' statistics whatever its widths (_get_stats runs for every axis); a zero-width
                // axis is skipped only where that computation cannot raise — a text input's reduction refuses (or, for
                // StringDType maximum / minimum, computes an unused value), and stat_length=0 refuses maximum / minimum.
                if (left == 0 && right == 0 && !ZeroWidthStatCanRaise(array, lengthPairs, axis, mode)) continue;
                var roi = _ViewRoi(padded, originalSlice, axis);
                try
                {
                    var (lStat, rStat) = _GetStats(roi, axis, (left, right),
                        (lengthPairs[axis, 0], lengthPairs[axis, 1]), mode);
                    try
                    {
                        _SetPadArea(roi, axis, (left, right), (lStat, rStat));
                    }
                    finally
                    {
                        lStat.Dispose();
                        if (!ReferenceEquals(rStat, lStat)) rStat.Dispose();
                    }
                }
                finally { roi.Dispose(); }
            }
            return padded;
        }

        /// <summary>
        ///     Computes the per-side statistic. Mirrors NumPy's <c>_get_stats</c>.
        ///     Returns <c>(left, right)</c> where both arrays have axis dim = 1
        ///     and other dims match <paramref name="padded"/>. When the
        ///     <paramref name="lengthPair"/> entries are equal and span the full
        ///     valid axis length, the same array is returned for both sides
        ///     (caller must check reference-equality before disposing twice).
        /// </summary>
        private static (NDArray left, NDArray right) _GetStats(
            NDArray padded, int axis, (long left, long right) width,
            (long left, long right) lengthPair, string mode)
        {
            long leftIndex = width.left;
            long rightIndex = padded.Shape.dimensions[axis] - width.right;
            long maxLength = rightIndex - leftIndex;

            long leftLength = lengthPair.left;
            long rightLength = lengthPair.right;
            if (leftLength < 0 || maxLength < leftLength) leftLength = maxLength;
            if (rightLength < 0 || maxLength < rightLength) rightLength = maxLength;

            if ((leftLength == 0 || rightLength == 0) && (mode == "maximum" || mode == "minimum"))
                throw new ArgumentException("stat_length of 0 yields no value for padding");

            // Left stat
            var leftSlices = _SliceAtAxis(new Slice(leftIndex, leftIndex + leftLength), axis, padded.ndim);
            NDArray leftStat;
            using (var leftChunk = padded[leftSlices])
            {
                leftStat = StatReduce(leftChunk, axis, mode);
                MaybeRoundCast(ref leftStat, padded);
            }

            if (leftLength == rightLength && leftLength == maxLength)
                return (leftStat, leftStat); // identical → reuse

            // Right stat
            var rightSlices = _SliceAtAxis(new Slice(rightIndex - rightLength, rightIndex), axis, padded.ndim);
            NDArray rightStat;
            using (var rightChunk = padded[rightSlices])
            {
                rightStat = StatReduce(rightChunk, axis, mode);
                MaybeRoundCast(ref rightStat, padded);
            }

            return (leftStat, rightStat);
        }

        /// <summary>
        ///     Whether the statistics of a ZERO-width axis can raise, so that <see cref="PadStat"/> must compute them as
        ///     NumPy's <c>_get_stats</c> does even though nothing is padded along that axis.
        /// </summary>
        /// <param name="array">The input (non-empty: an empty input never reaches the stat modes).</param>
        /// <param name="lengthPairs">The <c>stat_length</c> pairs (-1 for NumPy's <c>None</c>).</param>
        /// <param name="axis">The axis.</param>
        /// <param name="mode">The stat mode.</param>
        /// <returns>True for a text input (its reduction refuses, or computes an unused <c>StringDType</c> extreme) and
        ///     for a zero <c>stat_length</c> under <c>maximum</c> / <c>minimum</c> (<c>stat_length of 0 yields no value for
        ///     padding</c>); false when the statistics are numbers that nothing reads.</returns>
        private static bool ZeroWidthStatCanRaise(NDArray array, long[,] lengthPairs, int axis, string mode)
            => array.typecode == NPTypeCode.String
               || (mode is "maximum" or "minimum") && (lengthPairs[axis, 0] == 0 || lengthPairs[axis, 1] == 0);

        /// <summary>
        ///     One side's statistic of a stat-mode pad: NumPy's <c>stat_func(chunk, axis=axis, keepdims=True)</c> with
        ///     <c>np.amax</c> / <c>np.amin</c> / <c>np.mean</c> / <c>np.median</c>.
        /// </summary>
        /// <param name="chunk">The <c>stat_length</c> slice next to the pad area.</param>
        /// <param name="axis">The padded axis.</param>
        /// <param name="mode"><c>maximum</c>, <c>minimum</c>, <c>mean</c> or <c>median</c>.</param>
        /// <returns>The statistic with the axis kept at length 1 (a 0-d result from the 1-D numeric fast path broadcasts
        ///     the same way).</returns>
        /// <exception cref="TypeError">A text chunk whose reduction NumPy refuses (its texts, from the reductions).</exception>
        /// <exception cref="NotSupportedException">An unknown mode (the caller validated it).</exception>
        /// <remarks>Numbers take two shortcuts that give NumPy's values: a flat reduction for a 1-D chunk, and the mean as
        ///     <c>sum / n</c> (see the body); text takes NumPy's statement as it is.</remarks>
        private static NDArray StatReduce(NDArray chunk, int axis, string mode)
        {
            // Text takes NumPy's statement exactly (np.amax / np.amin / np.mean / np.median with the axis and
            // keepdims=True): the numeric shortcuts below rewrite the call (a flat reduction, the mean as sum / n), which
            // changes the answer for text — np.mean of StringDType text reaches the divide's no-loop refusal, a flat one
            // the str / int refusal, the mean rewritten as sum / n yet another text (probed against 2.4.2).
            if (chunk.typecode == NPTypeCode.String)
                return mode switch
                {
                    "maximum" => np.amax(chunk, axis: axis, keepdims: true),
                    "minimum" => np.amin(chunk, axis: axis, keepdims: true),
                    "mean" => np.mean(chunk, axis, true),
                    "median" => np.median(chunk, axis: axis, keepdims: true),
                    _ => throw new NotSupportedException($"stat mode '{mode}'"),
                };

            // 1D fast path: no-axis variants are 100×+ faster than axis+keepdims
            // for np.mean (DefaultEngine.Mean axis-path perf bug).  Returns 0-D,
            // broadcasts identically.
            if (chunk.ndim == 1 && axis == 0)
            {
                switch (mode)
                {
                    case "maximum": return np.amax(chunk);
                    case "minimum": return np.amin(chunk);
                    case "mean":    return np.mean(chunk);
                    case "median":  return np.median(chunk);
                }
            }

            // N-D path. Two pre-emptive workarounds for DefaultEngine bugs:
            //
            //  1. np.mean(arr, axis, keepdims) on N-D arrays is ~100× slower
            //     than np.sum(arr, axis, keepdims) (DefaultEngine.Mean axis-path
            //     reflection-loop bug). Compute as sum/N instead.
            //
            //  2. Axis reductions on sliced/non-contig 3D+ views are ~20× slower
            //     than on contiguous arrays (the strided reduction inner loop
            //     does mod/div per element for outer-axis reductions). The 2D
            //     non-contig case has near-identical perf to contig, so the
            //     materialisation is only worth it from ndim ≥ 3.
            NDArray src = chunk;
            bool ownsSrc = false;
            if (chunk.ndim >= 3 && !chunk.Shape.IsContiguous)
            {
                src = chunk.copy();
                ownsSrc = true;
            }
            try
            {
                switch (mode)
                {
                    case "maximum": return np.amax(src, axis: axis, keepdims: true);
                    case "minimum": return np.amin(src, axis: axis, keepdims: true);
                    case "mean":
                    {
                        using var sum = np.sum(src, axis: axis, keepdims: true);
                        long n = src.Shape.dimensions[axis];
                        // Promote sum to float for the divide if it isn't already.
                        if (IsIntegerDtype(sum.GetTypeCode))
                        {
                            using var asF = sum.astype(typeof(double));
                            using var divisor = NDArray.Scalar((double)n, NPTypeCode.Double);
                            return asF / divisor;
                        }
                        using var divisorF = NDArray.Scalar((double)n, sum.GetTypeCode);
                        return sum / divisorF;
                    }
                    case "median":  return np.median(src, axis: axis, keepdims: true);
                    default: throw new NotSupportedException($"stat mode '{mode}'");
                }
            }
            finally { if (ownsSrc) src.Dispose(); }
        }

        /// <summary>
        ///     NumPy <c>_round_if_needed</c>: when the destination dtype is integer,
        ///     round to nearest (banker's), then cast back to integer dtype. Skips
        ///     the cast when stat already returned the destination dtype (max/min).
        /// </summary>
        private static void MaybeRoundCast(ref NDArray stat, NDArray padded)
        {
            if (!IsIntegerDtype(padded.GetTypeCode)) return;
            if (stat.GetTypeCode == padded.GetTypeCode) return;
            // Round to nearest-even then cast.
            var rounded = np.around(stat);
            var castBack = rounded.astype(padded.dtype);
            stat.Dispose();
            if (!ReferenceEquals(rounded, castBack)) rounded.Dispose();
            stat = castBack;
        }

        private static bool IsIntegerDtype(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return true;
                default:
                    return false;
            }
        }

        // ---------------------------- linear_ramp ----------------------------

        /// <summary>
        ///     NumPy's <c>mode='linear_ramp'</c>: each pad area is <c>np.linspace(end_value, edge, num=width,
        ///     endpoint=False, dtype=padded.dtype)</c> along its axis (the right one reversed).
        /// </summary>
        /// <param name="array">The input (non-empty: an empty input returned before the mode body).</param>
        /// <param name="padPairs">The (before, after) widths per axis.</param>
        /// <param name="endValues">The <c>end_values</c> argument (null for NumPy's default 0).</param>
        /// <returns>The padded array; a copy for a 0-d input, which has no axis to pad.</returns>
        /// <exception cref="DTypePromotionError">Text on either side of a linspace (the input or an end value): NumPy's
        ///     <c>result_type(start, stop, float)</c> refusal (<see cref="ThrowIfLinearRampText"/>).</exception>
        /// <exception cref="TypeError">A fixed-width text input with an end value whose class promotes with text (a NumPy
        ///     float or complex scalar): linspace's <c>np.subtract</c> has no text loop.</exception>
        /// <exception cref="ValueError"><paramref name="endValues"/> does not broadcast to (ndim, 2).</exception>
        private static NDArray PadLinearRamp(NDArray array, long[,] padPairs, object endValues)
        {
            int ndim = array.ndim;
            // NumPy's _as_pairs(end_values, ndim) runs first, over the TRUE rank: a 0-d input validates the values' shape
            // against (0, 2) and then has no axis to pad.
            object[,] endPairs = _AsPairsValues(endValues ?? 0, ndim);
            if (ndim == 0) return array.copy();
            // NumPy runs np.linspace(start=end_value, stop=edge, num=width, dtype=padded.dtype) for every axis and side,
            // widths of 0 included, and text on either side refuses there. Checked before any ramp is computed: the
            // padded array of a call that raises is discarded in NumPy too, so the order of the work is unobservable.
            ThrowIfLinearRampText(array, endValues, ndim);
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);

            for (int axis = 0; axis < ndim; axis++)
            {
                long left = padPairs[axis, 0], right = padPairs[axis, 1];
                if (left == 0 && right == 0) continue;

                var roi = _ViewRoi(padded, originalSlice, axis);
                try
                {
                    var (leftEdge, rightEdge) = _GetEdges(roi, axis, (left, right));
                    try
                    {
                        // Build coefficients (width,) reshaped to broadcast across non-axis dims.
                        if (left > 0)
                        {
                            using var ramp = BuildLinearRamp(leftEdge, axis, left, endPairs[axis, 0], padded.GetTypeCode, reverse: false);
                            var padSlices = _SliceAtAxis(new Slice(0, left), axis, padded.ndim);
                            AssignSliceValue(roi, padSlices, ramp);
                        }
                        if (right > 0)
                        {
                            using var ramp = BuildLinearRamp(rightEdge, axis, right, endPairs[axis, 1], padded.GetTypeCode, reverse: true);
                            long axisLen = roi.Shape.dimensions[axis];
                            var padSlices = _SliceAtAxis(new Slice(axisLen - right, axisLen), axis, padded.ndim);
                            AssignSliceValue(roi, padSlices, ramp);
                        }
                    }
                    finally { leftEdge.Dispose(); rightEdge.Dispose(); }
                }
                finally { roi.Dispose(); }
            }
            return padded;
        }

        /// <summary>
        ///     The refusal of NumPy's per-axis <c>np.linspace(start=end_value, stop=edge, num=width, endpoint=False,
        ///     dtype=padded.dtype)</c> when text is on either side. <c>linspace</c> first promotes
        ///     <c>_array_converter(start, stop).result_type(ensure_inexact=True)</c> — the start's class, the edge's class
        ///     and the abstract Python float, in that order — and then computes <c>np.subtract(stop, start,
        ///     dtype=type(result))</c>; with text one of the two always refuses. Every start comes from ONE
        ///     <c>np.array(end_values)</c>, so every start has the same class and the first linspace (axis 0, left) decides.
        /// </summary>
        /// <param name="array">The input; its descriptor is every edge's (the edges are slices of the padded array).</param>
        /// <param name="endValues">The <c>end_values</c> argument as given (null for NumPy's default 0).</param>
        /// <param name="ndim">The input's rank (at least 1); kept for the order of the checks it documents.</param>
        /// <exception cref="DTypePromotionError">The promotion refuses: a text input meets an integer, bool, text or Python
        ///     number start, a StringDType input any number, a numeric input a text start (NumPy's text, raised by
        ///     <see cref="DTypePromotion.ResultType"/>).</exception>
        /// <exception cref="TypeError">The promotion succeeds into a text dtype (a NumPy float or complex start with an
        ///     S / U input promotes to U32 / S32 / U64) and the subtract finds no text loop: <c>ufunc 'subtract' did not
        ///     contain a loop with signature matching types (&lt;class 'numpy.dtypes.StrDType'&gt;, &lt;class
        ///     'numpy.dtypes.Float64DType'&gt;) -&gt; &lt;class 'numpy.dtypes.StrDType'&gt;</c>.</exception>
        /// <remarks>
        ///     NumPy's <c>_as_pairs(end_values, ndim)</c> makes the starts: an <c>np.array(end_values)</c> of one or two
        ///     elements (not shaped (2, 1)) hands out its elements as NumPy scalars — STRONG operands of the array's dtype,
        ///     so the default <c>0</c> is an <c>np.int64</c> and the pair <c>(1, 2.5)</c> two <c>np.float64</c>s — while a
        ///     larger array goes through <c>broadcast_to(...).tolist()</c>, whose Python ints, floats and complex numbers
        ///     are WEAK (<c>_PyLongDType</c>, <c>_PyFloatDType</c>, <c>_PyComplexDType</c>); a Python bool stays
        ///     <c>BoolDType</c> and a str <c>StrDType</c>. Probed against 2.4.2: every refusal's DType list names exactly
        ///     these classes, and a zero width refuses too (the linspace runs with <c>num=0</c>).
        /// </remarks>
        private static void ThrowIfLinearRampText(NDArray array, object endValues, int ndim)
        {
            var (startArray, strong) = LinearRampEndValuesArray(endValues);
            bool textInput = array.typecode == NPTypeCode.String;
            bool textStart = startArray.Meta is StringDTypeMeta || startArray.Meta == DTypeRegistry.VString;
            // Numbers on both sides: every linspace computes (BuildLinearRamp is its arithmetic).
            if (!textInput && !textStart || ndim == 0)
                return;
            // The start operand as linspace's promotion sees it: the NumPy scalar's descriptor, or the Python value's
            // class (an abstract one for a weak int / float / complex).
            object start = strong ? startArray : WeakStartClass(startArray);
            // result_type(start, stop, ensure_inexact): raises NumPy's DTypePromotionError when no common DType exists.
            DType promoted = DTypePromotion.ResultType(start, array.dtype, DTypeRegistry.PyFloat);
            string startClass = start is DType d ? d.Meta.ToString() : start.ToString();
            // The promotion found a (text) dtype; np.subtract(stop, start, dtype=type(promoted)) has no loop for it.
            throw new TypeError(
                $"ufunc 'subtract' did not contain a loop with signature matching types ({array.dtype.Meta}, {startClass}) -> {promoted.Meta}");
        }

        /// <summary>
        ///     The dtype of NumPy's <c>np.array(end_values)</c> inside <c>_as_pairs</c>, and whether its elements reach
        ///     <c>np.linspace</c> as NumPy scalars (strong) rather than as the Python values of <c>tolist()</c> (weak).
        /// </summary>
        /// <param name="endValues">The argument as given: null (NumPy's default <c>0</c>), a C# scalar (a Python scalar of
        ///     the boundary map), a string or <c>byte[]</c>, a two-value tuple, a C# array, an <c>object[,]</c>, a
        ///     dictionary (a NumSharp form) or an <see cref="NDArray"/>.</param>
        /// <returns>The array's descriptor (its CLASS is what the promotion reads) and the strength: true for one element,
        ///     or two elements not shaped (2, 1) — NumPy's <c>x[0]</c> / <c>x[1]</c> shortcuts.</returns>
        private static (DType Descr, bool Strong) LinearRampEndValuesArray(object endValues)
        {
            switch (endValues)
            {
                case null:
                    return (np.int64, true);
                case NDArray nd:
                    return (nd.dtype, nd.size == 1 || nd.size == 2 && !(nd.ndim == 2 && nd.Shape.dimensions[0] == 2));
                case string or char:
                    return (np.dtype("U"), true);
                case byte[]:
                    return (np.dtype("S"), true);
                case object[,] o2:
                {
                    var values = new List<object>(o2.Length);
                    foreach (var v in o2)
                        values.Add(v);
                    bool column = o2.GetLength(0) == 2 && o2.GetLength(1) == 1;
                    return (CoercedEndValuesDescr(values), o2.Length == 1 || o2.Length == 2 && !column);
                }
                case object[] o1:
                    return (CoercedEndValuesDescr(o1), o1.Length <= 2);
                case int[] ia:
                    return (np.int64, ia.Length <= 2);
                case long[] la:
                    return (np.int64, la.Length <= 2);
                case double[] da:
                    return (np.float64, da.Length <= 2);
                case IDictionary dict:
                {
                    // A NumSharp form (NumPy refuses a dict here): its values are read like the tolist() path's.
                    var values = new List<object>();
                    foreach (DictionaryEntry kv in dict)
                    {
                        var (b, a) = ExtractValuePair(kv.Value);
                        values.Add(b);
                        values.Add(a);
                    }
                    return (CoercedEndValuesDescr(values), false);
                }
                default:
                    return TryExtractPair(endValues, out object before, out object after)
                        ? (CoercedEndValuesDescr(new[] { before, after }), true)
                        : (CoercedEndValuesDescr(new[] { endValues }), true);
            }
        }

        /// <summary>
        ///     The dtype NumPy's array coercion gives a list of Python scalars: any str makes a str array (the numbers
        ///     become their text), else any bytes a bytes array, else complex, float, int and bool in that order (bool
        ///     alone stays bool; bool with ints is int).
        /// </summary>
        /// <param name="values">The C# values (the boundary map's Python scalars; an <see cref="NDArray"/> element counts
        ///     as its dtype's kind).</param>
        /// <returns>The coerced descriptor (unsized for text: only the class is read).</returns>
        private static DType CoercedEndValuesDescr(IEnumerable<object> values)
        {
            bool anyStr = false, anyBytes = false, anyComplex = false, anyFloat = false, anyInt = false, allBool = true;
            bool anyHugeUnsigned = false, anyNegative = false;
            foreach (var v in values)
            {
                allBool &= v is bool;
                switch (v)
                {
                    case string or char: anyStr = true; break;
                    case byte[]: anyBytes = true; break;
                    case System.Numerics.Complex: anyComplex = true; break;
                    case float or double or decimal or Half: anyFloat = true; break;
                    case bool: break;
                    case ulong ul: anyInt = true; anyHugeUnsigned |= ul > long.MaxValue; break;
                    case sbyte or byte or short or ushort or int or uint or long or System.Numerics.BigInteger:
                        anyInt = true;
                        anyNegative |= Convert.ToDouble(v) < 0;
                        break;
                    case NDArray nd:
                        if (nd.typecode == NPTypeCode.String) anyStr = true;
                        else if (nd.typecode == NPTypeCode.Complex) anyComplex = true;
                        else if (nd.typecode is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half or NPTypeCode.Decimal) anyFloat = true;
                        else if (nd.typecode != NPTypeCode.Boolean) anyInt = true;
                        allBool &= nd.typecode == NPTypeCode.Boolean;
                        break;
                    default: anyFloat = true; break;
                }
            }
            if (anyStr) return np.dtype("U");
            if (anyBytes) return np.dtype("S");
            if (anyComplex) return np.complex128;
            if (anyFloat) return np.float64;
            if (anyInt) return anyHugeUnsigned ? (anyNegative ? np.float64 : np.uint64) : np.int64;
            return allBool ? np.@bool : np.float64;
        }

        /// <summary>
        ///     The class linspace's promotion sees for a WEAK start (a Python value of <c>tolist()</c>): the abstract NEP 50
        ///     class of a Python int, float or complex; a Python bool's <c>BoolDType</c>; a str's or bytes' own class.
        /// </summary>
        /// <param name="arrayDescr">The descriptor of <c>np.array(end_values)</c>.</param>
        /// <returns>A <see cref="DTypeMeta"/> (abstract) or the <see cref="DType"/> itself.</returns>
        private static object WeakStartClass(DType arrayDescr)
        {
            var tc = arrayDescr.typecode;
            if (tc.IsInteger())
                return DTypeRegistry.PyLong;
            if (tc is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half or NPTypeCode.Decimal)
                return DTypeRegistry.PyFloat;
            if (tc == NPTypeCode.Complex)
                return DTypeRegistry.PyComplex;
            return arrayDescr;
        }

        /// <summary>
        ///     Build a linear ramp of length <paramref name="width"/> along
        ///     <paramref name="axis"/>, blending <paramref name="edge"/> (a
        ///     1-thick slice with axis-dim=1) toward scalar
        ///     <paramref name="endValue"/>.  Uses a scalar <c>np.linspace(0,1)</c>
        ///     reshaped to broadcast along non-axis dims, then computes the
        ///     ramp via existing SIMD-accelerated binary ops.
        ///
        ///     Mirrors NumPy <c>_get_linear_ramps</c> (which uses a vectorized
        ///     <c>np.linspace</c> with array bounds; ours is scalar so we
        ///     express the same formula as one broadcast multiply + add).
        /// </summary>
        [NDScopedCovered] // only caller: PadLinearRamp ← [NDScoped] PadImpl (coefReshaped view is scope-reclaimed)
        private static NDArray BuildLinearRamp(NDArray edge, int axis, long width, object endValue,
            NPTypeCode dtype, bool reverse)
        {
            // Coefficient vector: np.linspace(0, 1, width, endpoint=False) has shape (width,)
            // ramp[k] = end + coef[k] * (edge - end)  →  end at k=0, edge - step at k=width-1
            // For the right side we want the reversed ramp (edge-step nearest original,
            // end at far end) which means coef reversed along axis.
            NDArray coef = np.linspace(0.0, 1.0, width, endpoint: false);
            if (reverse)
            {
                // Reverse along the only axis (axis 0 of the (width,) coef).
                var rc = coef[new Slice(null, null, -1)];
                coef.Dispose();
                coef = rc.copy(); // materialize so subsequent reshape/broadcast see contig stride
                rc.Dispose();
            }

            // Reshape coef to (1, ..., width, ..., 1) at axis position.
            int ndim = edge.ndim;
            var coefShape = new long[ndim];
            for (int i = 0; i < ndim; i++) coefShape[i] = i == axis ? width : 1L;
            var coefReshaped = coef.reshape(Shape.Adopt(coefShape));

            // Arithmetic dtype selection — must preserve the source dtype's
            // structure (especially Complex's imaginary component).  For integer
            // dtypes we promote to double for precision and cast back at the
            // end (matching NumPy's linspace(dtype=int) truncate-cast semantics).
            // For floating / complex source we compute in the source dtype so
            // imaginary parts and float32 precision are preserved.
            bool isInteger = IsIntegerDtype(dtype);
            NPTypeCode workType = isInteger ? NPTypeCode.Double : dtype;
            using var endNd = NDArray.Scalar(
                NumSharp.Utilities.Converts.ChangeType(endValue, workType), workType);

            NDArray edgeWork = edge.GetTypeCode == workType ? edge : edge.astype(NumSharp.NPTypeCodeExtensions.AsType(workType));
            try
            {
                // ramp = end + coef * (edge - end)
                using var diff = edgeWork - endNd;
                using var scaled = coefReshaped * diff;
                NDArray ramp = scaled + endNd;
                coef.Dispose();
                if (!ReferenceEquals(edgeWork, edge)) edgeWork.Dispose();

                // NumPy linear_ramp uses np.linspace(..., dtype=padded.dtype).  For an
                // INTEGER destination dtype np.linspace floors toward -inf before casting
                // — NOT C-style truncation toward zero.  e.g. linspace(0, -3, 2, F) has
                // samples [0, -1.5] and yields [0, -2] (floor), not [0, -1] (truncate).
                // Float / complex destinations keep the fractional value (no floor).
                if (ramp.GetTypeCode != dtype)
                {
                    NDArray toCast = ramp;
                    if (isInteger)
                    {
                        var floored = np.floor(ramp);
                        ramp.Dispose();
                        toCast = floored;
                    }
                    var cast = toCast.astype(NumSharp.NPTypeCodeExtensions.AsType(dtype));
                    if (!ReferenceEquals(toCast, cast)) toCast.Dispose();
                    return cast;
                }
                return ramp;
            }
            catch
            {
                if (!ReferenceEquals(edgeWork, edge)) edgeWork.Dispose();
                throw;
            }
        }

        // ---------------------------- helpers (1:1 with NumPy) ----------------------------

        /// <summary>
        ///     Allocate <c>np.empty</c> with shape <c>arr.shape + 2*pad</c>, optionally
        ///     fill with <paramref name="fillValue"/>, then copy <paramref name="array"/>
        ///     into the center.  Returns <c>(padded, originalAreaSlice)</c> where
        ///     <c>originalAreaSlice</c> identifies the unpadded region for later
        ///     <see cref="_ViewRoi"/> calls.
        /// </summary>
        private static (NDArray padded, Slice[] originalSlice) _PadSimple(
            NDArray array, long[,] padPairs, object fillValue)
        {
            int ndim = array.ndim;
            // NumPy: 0-D input → newShape is empty tuple, padded == array.copy()
            if (ndim == 0)
            {
                var clone = array.copy();
                if (fillValue != null)
                    clone.SetAtIndex(CastBoxToDType(fillValue, array.GetTypeCode), 0);
                return (clone, Array.Empty<Slice>());
            }

            var newDims = new long[ndim];
            var originalSlice = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
            {
                long size = array.Shape.dimensions[i];
                long left = padPairs[i, 0];
                long right = padPairs[i, 1];
                newDims[i] = left + size + right;
                originalSlice[i] = new Slice(left, left + size);
            }

            // NumPy: order = 'F' if array.flags.fnc else 'C'  (F-contig and NOT also C).
            char order = (array.Shape.IsFContiguous && !array.Shape.IsContiguous) ? 'F' : 'C';
            var padded = np.empty(Shape.Adopt(newDims, order), array.dtype);

            if (fillValue != null)
            {
                // Whole-buffer fill via the underlying IArraySlice.Fill — uses
                // InitBlockUnaligned for byte-sized dtypes, 8x-unrolled scalar
                // otherwise. Routes around the per-axis broadcast loop.
                padded.Storage.InternalArray.Fill(CastBoxToDType(fillValue, array.GetTypeCode));
            }

            // Copy original values into the center.  padded[originalSlice] = array
            // routes through NDIter.Copy → IL StridedCastKernel (per-row memcpy).
            padded[originalSlice] = array;
            return (padded, originalSlice);
        }

        /// <summary>
        ///     Returns a view of <paramref name="padded"/> for axis-<paramref name="axis"/>
        ///     processing. Axes ≤ <paramref name="axis"/> see the full padded extent;
        ///     axes &gt; <paramref name="axis"/> are clipped to the original region —
        ///     prevents the iterative axis pass from re-overwriting corners already
        ///     set by earlier axes' pad bands.
        /// </summary>
        private static NDArray _ViewRoi(NDArray padded, Slice[] originalSlice, int axis)
        {
            int ndim = padded.ndim;
            var slices = new Slice[ndim];
            // axes [0..axis] keep the full padded range
            for (int i = 0; i <= axis; i++) slices[i] = Slice.All;
            // axes (axis..ndim-1] clip to the original (un-corner-padded) region
            for (int i = axis + 1; i < ndim; i++) slices[i] = originalSlice[i];
            return padded[slices];
        }

        /// <summary>
        ///     Writes <paramref name="leftValue"/> into the left pad band and
        ///     <paramref name="rightValue"/> into the right pad band of
        ///     <paramref name="padded"/> along <paramref name="axis"/>. Values may
        ///     be scalars (broadcast-fill) or NDArrays (broadcast across non-axis dims).
        /// </summary>
        private static void _SetPadArea(NDArray padded, int axis, (long left, long right) width,
            (object left, object right) values)
        {
            if (width.left > 0)
            {
                var slices = _SliceAtAxis(new Slice(0, width.left), axis, padded.ndim);
                AssignSliceValue(padded, slices, values.left);
            }
            if (width.right > 0)
            {
                long axisLen = padded.Shape.dimensions[axis];
                var slices = _SliceAtAxis(new Slice(axisLen - width.right, axisLen), axis, padded.ndim);
                AssignSliceValue(padded, slices, values.right);
            }
        }

        /// <summary>
        ///     Returns the 1-thick left/right edge slices of the valid region in
        ///     <paramref name="padded"/> along <paramref name="axis"/>.  Width
        ///     pair identifies where the valid region begins/ends; both returned
        ///     views have axis-dim = 1 (other dims preserved).
        /// </summary>
        private static (NDArray left, NDArray right) _GetEdges(NDArray padded, int axis, (long left, long right) width)
        {
            int ndim = padded.ndim;
            long axisLen = padded.Shape.dimensions[axis];
            long leftIdx = width.left;
            long rightIdx = axisLen - width.right;

            var leftSlices = _SliceAtAxis(new Slice(leftIdx, leftIdx + 1), axis, ndim);
            var rightSlices = _SliceAtAxis(new Slice(rightIdx - 1, rightIdx), axis, ndim);
            return (padded[leftSlices], padded[rightSlices]);
        }

        /// <summary>
        ///     Build a <see cref="Slice"/> array of length <paramref name="ndim"/>
        ///     with <paramref name="sl"/> at position <paramref name="axis"/> and
        ///     <see cref="Slice.All"/> elsewhere.  Mirrors NumPy's
        ///     <c>(slice(None),) * axis + (sl,) + (...,)</c>.
        /// </summary>
        private static Slice[] _SliceAtAxis(Slice sl, int axis, int ndim)
        {
            var slices = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
                slices[i] = i == axis ? sl : Slice.All;
            return slices;
        }

        // ---------------------------- pair normalisation ----------------------------

        /// <summary>
        ///     NumPy's dict branch of <c>np.pad</c> (NumPy 2.3+), statement for statement: <c>seq = [(0, 0)] * array.ndim</c>;
        ///     for each item in the dictionary's order, a <c>match</c> of the width (an int, or a 2-tuple of ints; anything
        ///     else reaches <c>typing.assert_never</c>) and then the Python list assignment <c>seq[axis] = …</c>; then
        ///     <c>np.asarray(seq)</c>'s integer-dtype test.
        /// </summary>
        /// <param name="widths">The per-axis widths: an integer pads both sides, a (before, after) tuple each side. Keys
        ///     count from the end when negative, as Python list indices do.</param>
        /// <param name="ndim">The padded array's TRUE rank (0 for a 0-d array, whose list has no slot at all).</param>
        /// <returns>The <c>(ndim, 2)</c> width table, for <see cref="_AsPairs"/> to refuse negative widths and broadcast.</returns>
        /// <exception cref="AssertionError">A width that is neither an integer nor a 2-tuple of integers: NumPy's
        ///     <c>typing.assert_never</c>, "Expected code to be unreachable, but got: 1.5". Checked per item BEFORE that
        ///     item's axis, as NumPy matches before it assigns.</exception>
        /// <exception cref="IndexError">An axis outside <c>[-ndim, ndim)</c>: Python's "list assignment index out of range".
        ///     The key is a list index and is never normalized, so NumPy raises no AxisError here.</exception>
        /// <exception cref="TypeError">"`pad_width` must be of integral type.": <c>np.asarray(seq)</c> is not an integer
        ///     array - the empty list of a 0-d array (float64), every row assigned Python bools (bool), or a width outside
        ///     int64 (uint64 or object).</exception>
        /// <remarks>
        ///     <para>
        ///     A negative width is NOT refused here: NumPy refuses it afterwards, in <c>_as_pairs</c>, so an out-of-range axis
        ///     LATER in the dict wins over a negative width earlier in it (<c>{0: -1, 5: 1}</c> is the IndexError). The old
        ///     dict path checked the axis first and the widths per item, and raised the base <see cref="ArgumentException"/>
        ///     for all of them; it still serves <c>stat_length</c>, where a dict is a NumSharp extension (NumPy refuses it).
        ///     </para>
        ///     <para>
        ///     The house map: a C# integer primitive or <see cref="System.Numerics.BigInteger"/> is a Python int, a
        ///     <see cref="bool"/> a Python bool (an int subclass: it pads by 0 or 1), a two-item <see cref="ValueTuple{T1,T2}"/>
        ///     a tuple. A typed C# array is an ndarray and a Half / char / decimal a NumPy scalar; NumPy's match accepts
        ///     neither, so both are the AssertionError (NumSharp used to accept a two-element <c>int[]</c> / <c>long[]</c>).
        ///     </para>
        /// </remarks>
        private static long[,] DictPadWidth(IDictionary<int, object> widths, int ndim)
        {
            var seq = new long[ndim, 2];
            // np.asarray(seq)'s dtype: bool only when EVERY row holds bools (an untouched row is the ints (0, 0)), not an
            // integer type when ANY row holds a value past int64. Tracked per row: a later key can overwrite a row.
            var boolRow = new bool[ndim];
            var hugeRow = new bool[ndim];
            foreach (var kv in widths)
            {
                if (!TryPadWidthItem(kv.Value, out long before, out long after, out bool isBool, out bool huge))
                    throw new AssertionError($"Expected code to be unreachable, but got: {PolyIndexArgument.Repr(kv.Value)}");
                int axis = kv.Key;
                if (axis < -ndim || axis >= ndim)
                    throw new IndexError("list assignment index out of range");
                if (axis < 0)
                    axis += ndim;
                seq[axis, 0] = before;
                seq[axis, 1] = after;
                boolRow[axis] = isBool;
                hugeRow[axis] = huge;
            }

            bool allBool = ndim > 0;
            bool anyHuge = false;
            for (int d = 0; d < ndim; d++)
            {
                allBool &= boolRow[d];
                anyHuge |= hugeRow[d];
            }
            if (ndim == 0 || allBool || anyHuge)
                throw new TypeError("`pad_width` must be of integral type.");
            return seq;
        }

        /// <summary>
        ///     The width patterns of NumPy's dict branch: <c>case int(both)</c> (one Python int, both sides) and
        ///     <c>case tuple((int(before), int(after)))</c> (a tuple of exactly two Python ints).
        /// </summary>
        /// <param name="value">The dict value.</param>
        /// <param name="before">The width before the axis.</param>
        /// <param name="after">The width after the axis.</param>
        /// <param name="isBool">Every matched item is a Python bool (the row's array dtype would be bool).</param>
        /// <param name="huge">A matched item lies outside int64 (the row's array dtype would not be int64).</param>
        /// <returns>False when neither pattern matches (NumPy's <c>typing.assert_never</c>).</returns>
        private static bool TryPadWidthItem(object value, out long before, out long after, out bool isBool, out bool huge)
        {
            if (TryPythonInt(value, out before, out isBool, out huge))
            {
                after = before;
                return true;
            }
            if (value is System.Runtime.CompilerServices.ITuple t && t.Length == 2
                && TryPythonInt(t[0], out before, out bool b0, out bool h0)
                && TryPythonInt(t[1], out after, out bool b1, out bool h1))
            {
                isBool = b0 && b1;
                huge = h0 || h1;
                return true;
            }
            before = after = 0;
            isBool = huge = false;
            return false;
        }

        /// <summary>
        ///     Python's <c>isinstance(x, int)</c> under the house map: a C# integer primitive, a <see cref="System.Numerics.BigInteger"/>
        ///     or a <see cref="bool"/> (an int subclass in Python).
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="value">Its value when it fits int64 (0 / 1 for a bool).</param>
        /// <param name="isBool">The value is a bool.</param>
        /// <param name="huge">The value lies outside int64 (<paramref name="value"/> is then meaningless).</param>
        /// <returns>False for anything that is not a Python int.</returns>
        private static bool TryPythonInt(object x, out long value, out bool isBool, out bool huge)
        {
            isBool = huge = false;
            switch (x)
            {
                case bool b:
                    value = b ? 1 : 0;
                    isBool = true;
                    return true;
                case ulong ul when ul > long.MaxValue:
                    value = 0;
                    huge = true;
                    return true;
                case System.Numerics.BigInteger bi:
                    if (bi < long.MinValue || bi > long.MaxValue)
                    {
                        value = 0;
                        huge = true;
                        return true;
                    }
                    value = (long)bi;
                    return true;
                default:
                    return TryToInt64(x, out value);
            }
        }

        /// <summary>
        ///     NumPy <c>_as_pairs</c> port for integer pair specs (pad_width / stat_length).
        ///     Broadcasts <paramref name="x"/> to a <c>long[ndim, 2]</c> table.
        ///     Accepts: <c>null</c>, scalar int, <c>int[]</c> / <c>long[]</c>,
        ///     <c>int[,]</c> / <c>long[,]</c>, <c>(int, int)</c>, or
        ///     <c>IDictionary&lt;int, object&gt;</c> (stat_length only: a pad_width dict is converted first by
        ///     <see cref="DictPadWidth"/>).
        /// </summary>
        /// <param name="x">The spec.</param>
        /// <param name="ndim">The number of pairs (the array's rank, at least 1).</param>
        /// <param name="asIndex">The pairs are widths / lengths: a negative one is NumPy's ValueError.</param>
        /// <returns>The <c>(ndim, 2)</c> table.</returns>
        /// <exception cref="ValueError">A negative value with <paramref name="asIndex"/> ("index can't contain negative
        ///     values").</exception>
        /// <exception cref="ArgumentException">A spec that does not broadcast to <c>(ndim, 2)</c>, or an unsupported type.</exception>
        internal static long[,] _AsPairs(object x, int ndim, bool asIndex)
        {
            var result = new long[ndim, 2];
            if (x is null)
            {
                // sentinel value — used by stat_length="full axis" path which
                // tests for negative on the consumer side
                for (int i = 0; i < ndim; i++) { result[i, 0] = -1; result[i, 1] = -1; }
                return result;
            }

            // Dict path: per-axis (negative keys allowed) overlaid on (0,0) default. Reached by a stat_length dict only (a
            // NumSharp extension); a pad_width dict arrives here as DictPadWidth's table.
            if (x is IDictionary dict)
            {
                foreach (DictionaryEntry kv in dict)
                {
                    int axis = ToInt32Key(kv.Key);
                    if (axis < 0) axis += ndim;
                    if (axis < 0 || axis >= ndim)
                        throw new ArgumentException($"pad_width dict axis {kv.Key} out of bounds for ndim={ndim}");
                    var pair = NormalizeSinglePair(kv.Value, asIndex);
                    result[axis, 0] = pair.before;
                    result[axis, 1] = pair.after;
                }
                return result;
            }

            // Tuple path
            if (x is ValueTuple<int, int> vti)
            {
                long b = vti.Item1, a = vti.Item2;
                if (asIndex && (b < 0 || a < 0))
                    throw new ValueError("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = b; result[i, 1] = a; }
                return result;
            }
            if (x is ValueTuple<long, long> vtl)
            {
                long b = vtl.Item1, a = vtl.Item2;
                if (asIndex && (b < 0 || a < 0))
                    throw new ValueError("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = b; result[i, 1] = a; }
                return result;
            }

            // 2-D rectangular: int[,] / long[,]
            if (x is int[,] i2)
            {
                int rows = i2.GetLength(0), cols = i2.GetLength(1);
                if (cols != 2)
                    throw new ArgumentException("pad_width 2D array must have shape (N, 2)");
                if (rows == 1)
                {
                    if (asIndex && (i2[0, 0] < 0 || i2[0, 1] < 0))
                        throw new ValueError("index can't contain negative values");
                    for (int i = 0; i < ndim; i++) { result[i, 0] = i2[0, 0]; result[i, 1] = i2[0, 1]; }
                    return result;
                }
                if (rows != ndim)
                    throw new ArgumentException($"pad_width shape ({rows},{cols}) is not broadcastable to ({ndim},2)");
                for (int i = 0; i < rows; i++)
                {
                    long b = i2[i, 0], a = i2[i, 1];
                    if (asIndex && (b < 0 || a < 0))
                        throw new ValueError("index can't contain negative values");
                    result[i, 0] = b; result[i, 1] = a;
                }
                return result;
            }
            if (x is long[,] l2)
            {
                int rows = l2.GetLength(0), cols = l2.GetLength(1);
                if (cols != 2)
                    throw new ArgumentException("pad_width 2D array must have shape (N, 2)");
                if (rows == 1)
                {
                    if (asIndex && (l2[0, 0] < 0 || l2[0, 1] < 0))
                        throw new ValueError("index can't contain negative values");
                    for (int i = 0; i < ndim; i++) { result[i, 0] = l2[0, 0]; result[i, 1] = l2[0, 1]; }
                    return result;
                }
                if (rows != ndim)
                    throw new ArgumentException($"pad_width shape ({rows},{cols}) is not broadcastable to ({ndim},2)");
                for (int i = 0; i < rows; i++)
                {
                    long b = l2[i, 0], a = l2[i, 1];
                    if (asIndex && (b < 0 || a < 0))
                        throw new ValueError("index can't contain negative values");
                    result[i, 0] = b; result[i, 1] = a;
                }
                return result;
            }

            // 1-D array: int[] / long[]. Sized 1 ⇒ scalar broadcast, sized 2 ⇒ pair broadcast.
            if (x is int[] i1)
            {
                return From1DLong(LongFromInt(i1), ndim, asIndex);
            }
            if (x is long[] l1)
            {
                return From1DLong(l1, ndim, asIndex);
            }

            // Scalar integer (and unsigned variants — accept what NumPy accepts).
            if (TryToInt64(x, out long scalar))
            {
                if (asIndex && scalar < 0)
                    throw new ValueError("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = scalar; result[i, 1] = scalar; }
                return result;
            }

            throw new ArgumentException($"`pad_width` must be of integral type (got {x.GetType().Name})");
        }

        private static long[,] From1DLong(long[] arr, int ndim, bool asIndex)
        {
            var result = new long[ndim, 2];
            if (arr.Length == 1)
            {
                if (asIndex && arr[0] < 0)
                    throw new ValueError("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = arr[0]; result[i, 1] = arr[0]; }
                return result;
            }
            if (arr.Length == 2)
            {
                if (asIndex && (arr[0] < 0 || arr[1] < 0))
                    throw new ValueError("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = arr[0]; result[i, 1] = arr[1]; }
                return result;
            }
            if (arr.Length == ndim)
            {
                for (int i = 0; i < ndim; i++)
                {
                    if (asIndex && arr[i] < 0)
                        throw new ValueError("index can't contain negative values");
                    result[i, 0] = arr[i]; result[i, 1] = arr[i];
                }
                return result;
            }
            throw new ArgumentException($"pad_width length {arr.Length} is not broadcastable to (ndim={ndim}, 2)");
        }

        /// <summary>
        ///     One value of the <c>stat_length</c> dict (a NumSharp extension: NumPy refuses a dict there) as a
        ///     (before, after) pair.
        /// </summary>
        /// <param name="value">An integer (both sides), a two-item integer tuple, or a two-element <c>int[]</c> / <c>long[]</c>.</param>
        /// <param name="asIndex">The pair is a length: a negative side is NumPy's ValueError.</param>
        /// <returns>The pair.</returns>
        /// <exception cref="ValueError">A negative side with <paramref name="asIndex"/>.</exception>
        /// <exception cref="ArgumentException">A null or unsupported value.</exception>
        private static (long before, long after) NormalizeSinglePair(object value, bool asIndex)
        {
            if (value is null) throw new ArgumentException("pad_width dict value cannot be null");
            if (TryToInt64(value, out long scalar))
            {
                if (asIndex && scalar < 0)
                    throw new ValueError("index can't contain negative values");
                return (scalar, scalar);
            }
            (long before, long after) pair;
            if (value is ValueTuple<int, int> vti) pair = (vti.Item1, vti.Item2);
            else if (value is ValueTuple<long, long> vtl) pair = (vtl.Item1, vtl.Item2);
            else if (value is int[] arr && arr.Length == 2) pair = (arr[0], arr[1]);
            else if (value is long[] arrL && arrL.Length == 2) pair = (arrL[0], arrL[1]);
            else throw new ArgumentException($"pad_width dict value must be int or (before, after) pair");
            // A pair is checked like a scalar: a negative side used to pass and reached the slicing as a garbage shape.
            if (asIndex && (pair.before < 0 || pair.after < 0))
                throw new ValueError("index can't contain negative values");
            return pair;
        }

        /// <summary>
        ///     Same as <see cref="_AsPairs"/> but for value pairs (constant_values, end_values)
        ///     where the per-axis value is a scalar of any dtype (not necessarily integer).
        ///     Returns <c>object[ndim, 2]</c> with each cell holding the boxed scalar.
        /// </summary>
        /// <param name="x">The argument (null, a scalar, a pair, a dictionary, a C# array or an <c>object[,]</c>).</param>
        /// <param name="ndim">The number of axes.</param>
        /// <returns>The (ndim, 2) table.</returns>
        /// <exception cref="ValueError">The values do not broadcast to (ndim, 2) (NumPy's <c>np.broadcast_to</c> text).</exception>
        /// <exception cref="ArgumentException">A dictionary axis out of bounds, or an <c>object[,]</c> that is not (N, 2).</exception>
        internal static object[,] _AsPairsValues(object x, int ndim)
        {
            var result = new object[ndim, 2];
            if (x is null)
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = null; result[i, 1] = null; }
                return result;
            }

            if (x is IDictionary dict)
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = 0; result[i, 1] = 0; }
                foreach (DictionaryEntry kv in dict)
                {
                    int axis = ToInt32Key(kv.Key);
                    if (axis < 0) axis += ndim;
                    if (axis < 0 || axis >= ndim)
                        throw new ArgumentException($"values dict axis {kv.Key} out of bounds");
                    var (b, a) = ExtractValuePair(kv.Value);
                    result[axis, 0] = b; result[axis, 1] = a;
                }
                return result;
            }

            // Tuple (before, after) — broadcast across axes
            if (x is ITuple2 t2box)
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = t2box.Before; result[i, 1] = t2box.After; }
                return result;
            }

            // object[,] shape (N,2) or (1,2)
            if (x is object[,] o2)
            {
                int rows = o2.GetLength(0), cols = o2.GetLength(1);
                if (cols != 2)
                    throw new ArgumentException("constant_values 2D array must have shape (N, 2)");
                if (rows == 1)
                {
                    for (int i = 0; i < ndim; i++) { result[i, 0] = o2[0, 0]; result[i, 1] = o2[0, 1]; }
                    return result;
                }
                // NumPy's _as_pairs ends in np.broadcast_to(x, (ndim, 2)), whose refusal names both shapes.
                if (rows != ndim)
                    throw new ValueError(
                        $"operands could not be broadcast together with remapped shapes [original->remapped]: ({rows},{cols})  and requested shape ({ndim},2)");
                for (int i = 0; i < rows; i++) { result[i, 0] = o2[i, 0]; result[i, 1] = o2[i, 1]; }
                return result;
            }

            // 1-D arrays
            if (x is object[] o1) return FromValues1D(o1, ndim);
            if (x is int[] iv) return FromValues1D(BoxIntArray(iv), ndim);
            if (x is long[] lv) return FromValues1D(BoxLongArray(lv), ndim);
            if (x is double[] dv) return FromValues1D(BoxDoubleArray(dv), ndim);

            // Recognised tuple of two scalar primitives — extract.
            if (TryExtractPair(x, out object pb, out object pa))
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = pb; result[i, 1] = pa; }
                return result;
            }

            // Bare scalar
            for (int i = 0; i < ndim; i++) { result[i, 0] = x; result[i, 1] = x; }
            return result;
        }

        /// <summary>
        ///     <c>_as_pairs</c> for a 1-D list of values: one value for both sides of every axis, two values for (before,
        ///     after) of every axis.
        /// </summary>
        /// <param name="arr">The values.</param>
        /// <param name="ndim">The number of axes.</param>
        /// <returns>The (ndim, 2) table of boxed values.</returns>
        /// <exception cref="ValueError">Any other length: NumPy's <c>np.broadcast_to(x, (ndim, 2))</c> refusal.</exception>
        private static object[,] FromValues1D(object[] arr, int ndim)
        {
            var result = new object[ndim, 2];
            if (arr.Length == 1)
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = arr[0]; result[i, 1] = arr[0]; }
                return result;
            }
            if (arr.Length == 2)
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = arr[0]; result[i, 1] = arr[1]; }
                return result;
            }
            // NumPy's _as_pairs ends in np.broadcast_to(x, (ndim, 2)): a 1-D array of any other length cannot broadcast.
            throw new ValueError(
                $"operands could not be broadcast together with remapped shapes [original->remapped]: ({arr.Length},)  and requested shape ({ndim},2)");
        }

        private static (object before, object after) ExtractValuePair(object v)
        {
            if (v is ValueTuple<int, int> vti) return (vti.Item1, vti.Item2);
            if (v is ValueTuple<long, long> vtl) return (vtl.Item1, vtl.Item2);
            if (v is ValueTuple<double, double> vtd) return (vtd.Item1, vtd.Item2);
            if (v is object[] arr && arr.Length == 2) return (arr[0], arr[1]);
            if (v is int[] iv && iv.Length == 2) return (iv[0], iv[1]);
            if (v is long[] lv && lv.Length == 2) return (lv[0], lv[1]);
            if (v is double[] dv && dv.Length == 2) return (dv[0], dv[1]);
            return (v, v);
        }

        private static bool TryExtractPair(object x, out object before, out object after)
        {
            if (x is ValueTuple<int, int> vti) { before = vti.Item1; after = vti.Item2; return true; }
            if (x is ValueTuple<long, long> vtl) { before = vtl.Item1; after = vtl.Item2; return true; }
            if (x is ValueTuple<double, double> vtd) { before = vtd.Item1; after = vtd.Item2; return true; }
            before = after = null;
            return false;
        }

        // Sentinel marker used only inside _AsPairsValues for dict-with-tuple inputs.
        private interface ITuple2
        {
            object Before { get; }
            object After { get; }
        }

        // ---------------------------- scalar utilities ----------------------------

        private static bool TryToInt64(object x, out long result)
        {
            switch (x)
            {
                case sbyte sb: result = sb; return true;
                case byte b: result = b; return true;
                case short s: result = s; return true;
                case ushort us: result = us; return true;
                case int i: result = i; return true;
                case uint ui: result = ui; return true;
                case long l: result = l; return true;
                case ulong ul: result = (long)ul; return true;
                default: result = 0; return false;
            }
        }

        private static int ToInt32Key(object key)
        {
            if (TryToInt64(key, out long v))
            {
                if (v > int.MaxValue || v < int.MinValue)
                    throw new ArgumentException($"axis key out of int range: {v}");
                return (int)v;
            }
            throw new ArgumentException($"pad_width dict key must be int (got {key?.GetType().Name})");
        }

        private static long[] LongFromInt(int[] src)
        {
            var result = new long[src.Length];
            for (int i = 0; i < src.Length; i++) result[i] = src[i];
            return result;
        }

        private static object[] BoxIntArray(int[] src)
        {
            var r = new object[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i];
            return r;
        }

        private static object[] BoxLongArray(long[] src)
        {
            var r = new object[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i];
            return r;
        }

        private static object[] BoxDoubleArray(double[] src)
        {
            var r = new object[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i];
            return r;
        }

        private static object BoxedZero(NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return false;
                case NPTypeCode.Byte: return (byte)0;
                case NPTypeCode.SByte: return (sbyte)0;
                case NPTypeCode.Int16: return (short)0;
                case NPTypeCode.UInt16: return (ushort)0;
                case NPTypeCode.Int32: return 0;
                case NPTypeCode.UInt32: return 0u;
                case NPTypeCode.Int64: return 0L;
                case NPTypeCode.UInt64: return 0ul;
                case NPTypeCode.Char: return '\0';
                case NPTypeCode.Half: return (Half)0;
                case NPTypeCode.Single: return 0f;
                case NPTypeCode.Double: return 0d;
                case NPTypeCode.Decimal: return 0m;
                case NPTypeCode.Complex: return new System.Numerics.Complex(0, 0);
                default: return 0;
            }
        }

        /// <summary>
        ///     Converts one pad constant into the padded array's dtype the way NumPy stores it: <c>_as_pairs</c> runs
        ///     <c>np.array(constant_values)</c> and indexes the result, so every constant reaches
        ///     <c>padded[...] = value</c> as a NumPy SCALAR of its discovered dtype, which <c>PyArray_Pack</c> int()-converts
        ///     with the range check into a SIGNED integer array and casts into every other
        ///     (<see cref="ReductionInitial.PackNumPyScalar"/>).
        /// </summary>
        /// <param name="value">One constant: a C# number (a NumPy scalar of its own C# type), a one-element
        ///     <see cref="NDArray"/> (its element), or <c>null</c> (NumPy's default 0).</param>
        /// <param name="tc">The padded array's dtype.</param>
        /// <returns>The constant boxed as <paramref name="tc"/>'s exact CLR type; an array of several elements or a value
        ///     that is not a number returns through the house conversion (<see cref="Converts.ChangeType(object, NPTypeCode)"/>)
        ///     or as is (an array), for the broadcast-assignment route.</returns>
        /// <exception cref="OverflowException">A signed integer array and a constant whose int() does not fit it:
        ///     <c>np.pad(int8, 1, constant_values=1000)</c> raises "Python integer 1000 out of bounds for int8", as in NumPy
        ///     (an unsigned array casts instead: <c>np.pad(uint8, 1, constant_values=1000)</c> pads with 232); a float
        ///     infinity into a signed integer.</exception>
        /// <exception cref="ValueError">A float NaN constant into a signed integer array ("cannot convert float NaN to
        ///     integer"); a text constant into a numeric array whose text does not parse (the cast of NumPy's
        ///     <c>np.str_</c> scalar: "invalid literal for int() with base 10: np.str_('z')").</exception>
        private static unsafe object CastBoxToDType(object value, NPTypeCode tc)
        {
            if (value is null) return BoxedZero(tc);
            // A one-element array constant is the NumPy scalar np.array(...) indexes out of it; a larger one keeps the
            // broadcast-assignment route.
            if (value is NDArray nd)
            {
                if (nd.size != 1 || nd.typecode == NPTypeCode.String)
                    return value;
                value = nd.GetAtIndex(0);
            }

            // A text constant reaches NumPy's setitem as the NumPy scalar np.array(...) indexes out of _as_pairs (np.str_ /
            // np.bytes_), which the numeric dtype CASTS: its refusal quotes np.str_('z'), not a Python str (probed:
            // np.pad(np.zeros(3, int), 1, constant_values='z') is "invalid literal for int() with base 10: np.str_('z')").
            if (value is string or char or byte[] && tc != NPTypeCode.String)
            {
                using var text = Backends.Strings.NDStringArrays.FromScalar(value, null);
                using var cast = text.astype(DType.From(tc), true, 'K', "unsafe");
                return cast.GetAtIndex(0);
            }
            if (!NDArray.IsStoreScalar(value) || tc == NPTypeCode.String)
                return NumSharp.Utilities.Converts.ChangeType(value, tc);

            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.PackNumPyScalar(value, tc, new Span<byte>(slot, ReductionInitial.SlotBytes));
            return ReductionInitial.Box(new ReadOnlySpan<byte>(slot, ReductionInitial.SlotBytes), tc);
        }

        // ---------------------------- assignment glue ----------------------------

        /// <summary>
        ///     Assign <paramref name="value"/> (scalar or NDArray) into the slice
        ///     <paramref name="slices"/> of <paramref name="padded"/>. Scalars are
        ///     wrapped in a 0-D NDArray and broadcast via <see cref="NDIter.Copy"/>'s
        ///     broadcast fast path (convert-once + memcpy per outer row).
        ///
        ///     Routes through <see cref="NDIter.Copy"/> directly rather than the
        ///     <c>padded[slices] = value</c> indexer — the indexer's contiguous
        ///     fast path skips broadcast stretching when both shapes are contig
        ///     but sizes differ, which truncates broadcast writes (e.g. a
        ///     <c>(1, N)</c> edge view into a <c>(pad_width, N)</c> band only
        ///     fills the first row). <see cref="NDIter.Copy"/> always honours
        ///     broadcasting.
        /// </summary>
        private static void AssignSliceValue(NDArray padded, Slice[] slices, object value)
        {
            var view = padded[slices];
            try
            {
                if (value is NDArray nd)
                {
                    NDIter.Copy(view, nd);
                    return;
                }
                using var scalarNd = NDArray.Scalar(value, padded.GetTypeCode);
                NDIter.Copy(view, scalarNd);
            }
            finally { view.Dispose(); }
        }

        // ---------------------------- validation ----------------------------

        private static void ValidateModeKwargs(string mode, object constantValues, object endValues,
            object statLength, string reflectType)
        {
            // Reject unsupported kwargs the way NumPy does. We just check that the
            // value supplied to an irrelevant kwarg is null/default.
            switch (mode)
            {
                case "constant":
                    if (endValues != null) throw new ArgumentException("unsupported keyword arguments for mode 'constant': end_values");
                    if (statLength != null) throw new ArgumentException("unsupported keyword arguments for mode 'constant': stat_length");
                    break;
                case "edge":
                case "wrap":
                case "empty":
                    if (constantValues != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': constant_values");
                    if (endValues != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': end_values");
                    if (statLength != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': stat_length");
                    break;
                case "linear_ramp":
                    if (constantValues != null) throw new ArgumentException("unsupported keyword arguments for mode 'linear_ramp': constant_values");
                    if (statLength != null) throw new ArgumentException("unsupported keyword arguments for mode 'linear_ramp': stat_length");
                    break;
                case "maximum":
                case "minimum":
                case "mean":
                case "median":
                    if (constantValues != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': constant_values");
                    if (endValues != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': end_values");
                    break;
                case "reflect":
                case "symmetric":
                    if (constantValues != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': constant_values");
                    if (endValues != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': end_values");
                    if (statLength != null) throw new ArgumentException($"unsupported keyword arguments for mode '{mode}': stat_length");
                    if (reflectType != "even" && reflectType != "odd")
                        throw new ArgumentException($"reflect_type must be 'even' or 'odd' (got '{reflectType}')");
                    break;
                default:
                    throw new ArgumentException($"mode '{mode}' is not supported");
            }
        }

        // ---------------------------- predicates ----------------------------

        private static bool AllPairsZero(long[,] padPairs)
        {
            for (int i = 0; i < padPairs.GetLength(0); i++)
                if (padPairs[i, 0] != 0 || padPairs[i, 1] != 0)
                    return false;
            return true;
        }

        /// <summary>
        ///     Whether a pad whose widths are all zero is a plain copy in NumPy, i.e. whether no part of NumPy's mode body —
        ///     which <c>numpy.pad</c> runs for every axis whatever its widths — can raise for these arguments.
        /// </summary>
        /// <param name="mode">The (validated) mode.</param>
        /// <param name="constantValues">The <c>constant_values</c> argument, or null.</param>
        /// <returns>
        ///     True for the movement modes (<c>edge</c>, <c>wrap</c>, <c>reflect</c>, <c>symmetric</c>, <c>empty</c>), whose
        ///     zero-width bodies only read; and for <c>constant</c> with NumPy's default 0 (an explicit value is converted to
        ///     the array's dtype for every axis — <c>np.pad(np.zeros(3, int), 0, constant_values='z')</c> raises — and an
        ///     array of values goes through <c>_as_pairs</c>' broadcast, which refuses a bad shape). False for <c>linear_ramp</c> (its
        ///     <c>np.linspace</c> runs per axis and refuses text on either side) and for the stat modes (their reductions run
        ///     per axis: a text input refuses, <c>stat_length=0</c> refuses <c>maximum</c> / <c>minimum</c>), whose bodies
        ///     then skip the zero-width work they can prove cannot raise.
        /// </returns>
        /// <remarks>Probed against 2.4.2: <c>np.pad(np.array(['ab']), 0, 'linear_ramp')</c> raises NumPy's
        ///     DTypePromotionError, <c>np.pad(np.array(['ab']), 0, 'maximum')</c> the maximum no-loop TypeError,
        ///     <c>np.pad(np.zeros(3), 0, 'maximum', stat_length=0)</c> <c>stat_length of 0 yields no value for padding</c>,
        ///     <c>np.pad(np.zeros(3), 0, constant_values=[1, 2, 3])</c> the broadcast ValueError, <c>np.pad(np.zeros(3, int),
        ///     0, constant_values='z')</c> the int() ValueError.</remarks>
        private static bool ZeroWidthIsCopy(string mode, object constantValues)
            => mode switch
            {
                "edge" or "wrap" or "reflect" or "symmetric" or "empty" => true,
                "constant" => constantValues is null,
                _ => false,
            };

    }
}
