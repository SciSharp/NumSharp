using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // ============================== np.pad (lib/_arraypad_impl.py) ==============================
    // NumPy's np.pad statements as members of the array object (P3's batch 7): the np.pad facades keep their
    // numpy.matrix regions and call Pad / PadCallable. The helpers keep their NumPy names (_AsPairs, _PadSimple,
    // _SliceAtAxis, _ViewRoi, _SetPadArea, _GetEdges, _GetStats, _SetReflectBoth, _SetWrapBoth): the
    // corner-propagation logic that turns per-axis padding into N-D padding tracks the upstream structure. The
    // statements call the members the np.* functions dispatch to (the plan's phase P6), and the file sits under
    // Backends/, beside the array object's class (plan C3).
    public partial class NDArray
    {
        // ---------------------------- top-level dispatcher ----------------------------

        // Scoped here (not per-overload): every value-mode pad overload funnels through this
        // dispatcher (Pad), so one ambient scope reclaims the whole mode-body call tree (the
        // `2 * edge` temps in _SetReflectBoth, BuildLinearRamp's reshape, …) for all of them.
        // PadCallable is deliberately NOT scoped: its tree disposes everything it creates,
        // and an ambient scope there would reach into the user's np.PadFunc callback and dispose
        // arrays the callback creates for its own use.
        /// <summary>
        ///     NumPy's <c>numpy.pad</c> (<c>lib/_arraypad_impl.py</c>) for a named mode, after the facade's numpy.matrix
        ///     region: the dict branch of <paramref name="pad_width"/>, <c>_as_pairs</c>, the mode's keyword check, then the mode
        ///     body over <c>_pad_simple</c>'s array (constant, edge, empty, wrap, reflect, symmetric, the four statistics,
        ///     linear_ramp).
        /// </summary>
        /// <param name="array">The array to pad (not null).</param>
        /// <param name="pad_width">The widths in any C# form the facades take: an <see cref="int"/>, an <c>int[]</c>, an
        ///     <c>int[,]</c>, a <c>(before, after)</c> tuple or an <c>IDictionary&lt;int, object&gt;</c>.</param>
        /// <param name="mode">The mode's name (not null).</param>
        /// <param name="constant_values">The fill of <c>'constant'</c>, or null for NumPy's default 0.</param>
        /// <param name="end_values">The ends of <c>'linear_ramp'</c>, or null for 0.</param>
        /// <param name="stat_length">The window of the statistic modes, or null for the whole axis.</param>
        /// <param name="reflect_type"><c>'odd'</c> for the odd reflection of <c>'reflect'</c> / <c>'symmetric'</c>; any other
        ///     value is the even one, as in NumPy, whose loop tests only <c>method == "odd"</c>.</param>
        /// <returns>A new padded array of <paramref name="array"/>'s dtype.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="TypeError">A null <paramref name="pad_width"/> (NumPy's <c>np.asarray(None)</c> is an object
        ///     array: "`pad_width` must be of integral type."), or a dict of widths that is no integer table.</exception>
        /// <exception cref="ValueError">A negative width ("index can't contain negative values"), widths that do not
        ///     broadcast to <c>(ndim, 2)</c>, an empty width array; then a mode NumPy does not know, null included ("mode
        ///     'None' is not supported"); a keyword the mode does not take ("unsupported keyword arguments for mode
        ///     'edge': {'constant_values'}"); an empty axis padded by a mode other than constant / empty.</exception>
        /// <remarks>
        ///     <para>
        ///     Scoped: every value-mode overload funnels through it, so one scope reclaims the whole mode-body call tree's
        ///     temporaries.
        ///     </para>
        ///     <para>
        ///     The checks run in NumPy's order: the dict branch, <c>np.asarray(pad_width)</c>'s integral test and
        ///     <c>_as_pairs</c> over the array's TRUE rank (0 for a 0-d array: a pad_width that does not broadcast to
        ///     <c>(0, 2)</c> is refused there too), then the mode and its keywords. A C# keyword left at its default — null,
        ///     or <c>"even"</c> for <paramref name="reflect_type"/> — is a keyword NumPy did not receive.
        ///     </para>
        /// </remarks>
        [NDScoped]
        internal static NDArray Pad(NDArray array, object pad_width, string mode,
            object constant_values, object end_values, object stat_length, string reflect_type)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));

            // A dict pad_width runs NumPy's dict branch first, against the array's TRUE rank (a 0-d array's list has no
            // slot, so every key is an IndexError there); the (ndim, 2) table it builds then goes through _as_pairs
            // like any other pad_width.
            if (pad_width is IDictionary<int, object> widths)
                pad_width = DictPadWidth(widths, array.ndim);
            long[,] padPairs = PadWidthPairs(pad_width, array.ndim);

            // NumPy looks the mode up only now, after the widths (np.pad(a, -1, None) is the negative-width error).
            ValidateModeKwargs(mode, constant_values, end_values, stat_length, reflect_type);

            // No-op fast path: every pad value is zero. NumPy itself has no such shortcut — its mode bodies run for every
            // axis, widths of 0 included — so it is taken only where no mode body can raise (ZeroWidthIsCopy):
            // linear_ramp's linspace over text, the stat modes' reductions over text or with stat_length=0, and a
            // constant_values array that does not broadcast all raise in NumPy. The result is _pad_simple's array, the
            // one every zero-width mode body returns unchanged: an F-ordered input (F-contiguous, not C) stays F-ordered.
            if (AllPairsZero(padPairs) && ZeroWidthIsCopy(mode, constant_values))
                return _PadSimple(array, padPairs, fillValue: null).padded;

            // Empty-axis guard: only 'constant' and 'empty' may extend an empty axis. Past it NumPy returns _pad_simple's
            // array as it is ("_pad_simple already returned the correct result"): no other mode body runs over an empty
            // array, so none of their checks (a text linear_ramp's linspace, a text stat) can raise there.
            if (array.size == 0 && mode != "constant" && mode != "empty")
            {
                for (int ax = 0; ax < array.ndim; ax++)
                {
                    if (array.Shape.dimensions[ax] == 0 && (padPairs[ax, 0] != 0 || padPairs[ax, 1] != 0))
                        throw new ValueError(
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
                    // Unreachable: ValidateModeKwargs refused every name outside NumPy's table above.
                    throw new ValueError($"mode '{mode}' is not supported");
            }
        }

        /// <summary>
        ///     NumPy's <c>numpy.pad</c> with a callable mode, after the facade's numpy.matrix region: the dict branch, <c>_as_pairs</c>,
        ///     the zero-filled <c>_pad_simple</c> array, then <paramref name="func"/> called on every 1-D vector along every
        ///     axis (NumPy's moveaxis + ndindex walk), writing the pad areas in place.
        /// </summary>
        /// <param name="array">The array to pad (not null).</param>
        /// <param name="pad_width">The widths in any C# form the facades take.</param>
        /// <param name="func">The callable; null is NumPy's <c>mode=None</c>, refused after the widths are read.</param>
        /// <param name="kwargs">Handed to <paramref name="func"/> unchanged.</param>
        /// <returns>The padded array: F-ordered for an F-ordered input (F-contiguous and not C), as
        ///     <c>_pad_simple</c> allocates it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
        /// <exception cref="TypeError">A null <paramref name="pad_width"/>, or a dict of widths that is no integer
        ///     table.</exception>
        /// <exception cref="ValueError">A negative width or widths that do not broadcast to <c>(ndim, 2)</c>; then a null
        ///     <paramref name="func"/> ("mode 'None' is not supported").</exception>
        /// <remarks>
        ///     <para>
        ///     Deliberately not scoped: a scope would reach into <paramref name="func"/> and dispose the arrays it creates
        ///     for its own use; the walk disposes what it creates itself.
        ///     </para>
        ///     <para>
        ///     NumPy calls the function for EVERY 1-D vector along every axis, widths of 0 included, so there is no
        ///     zero-width shortcut: a callable that writes outside the pad area, or counts its calls, sees each vector. A
        ///     0-d input has no axis: its result is the value itself (<c>_pad_simple</c> copies it over the zero fill).
        ///     </para>
        /// </remarks>
        internal static NDArray PadCallable(NDArray array, object pad_width, np.PadFunc func, object kwargs)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));
            // The dict branch first, as in Pad (NumPy's pad runs it before it looks at the mode).
            if (pad_width is IDictionary<int, object> widths)
                pad_width = DictPadWidth(widths, array.ndim);
            long[,] padPairs = PadWidthPairs(pad_width, array.ndim);
            // NumPy's callable(mode) is false for None, which then fails the mode table's lookup ("mode 'None' is not
            // supported") - after the widths, so a bad width is reported first.
            if (func is null) throw new ValueError("mode 'None' is not supported");

            // NumPy: zero-fill the padded buffer, then visit each 1-D vector along
            // each axis (via ndindex over view.shape[:-1]) and let user code mutate.
            var (padded, _) = _PadSimple(array, padPairs, fillValue: BoxedZero(array.GetTypeCode));
            if (array.ndim == 0) return padded;

            for (int axis = 0; axis < padded.ndim; axis++)
            {
                // Move pad axis to the end so we can iterate the outer (ndim-1)
                // dims and pass each 1-D vector by reference to the user function.
                var view = padded.TensorEngine.MoveAxis(padded, new[] {axis}, new[] {padded.ndim - 1});
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
                    // A Python scalar of tolist() and the NumPy scalar of the same value print alike (str(1) and
                    // str(np.int32(1)) are both '1'; a float16 element reached tolist() as the Python float of its value).
                    object v = UnwrapPyListItem(valuePairs[axis, side]);
                    // Text stays text; a number becomes its 0-d array, which the area assignment writes as its str()
                    // (NumPy's _as_pairs makes every constant a NumPy scalar through np.array).
                    encoded[axis, side] = v is NDArray given ? given
                        : v is string or byte[] or char ? Backends.UnmanagedStorage.FromAnyScalar(v, null)
                        : v is null ? Backends.UnmanagedStorage.FromAnyScalar(null, array.dtype)
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
                throw new ValueError("stat_length of 0 yields no value for padding");

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
        /// <remarks>
        ///     <para>
        ///     NumPy's statement runs on the chunk VIEW itself, whatever its layout: a reduction's order follows the view's
        ///     memory order (an F-ordered padded array reduces its axis-0 chunks along their contiguous runs), and the mean
        ///     of an integer chunk is <c>_mean</c>'s float64 sum and division. Both used to be rewritten - a contiguous copy
        ///     of a 3-D chunk first, and the mean as <c>sum(chunk) / n</c> in the CHUNK's dtype, which wrapped an unsigned
        ///     sum before the division (<c>np.pad(uint64 [3, 2**64 - 1, 2**64 - 4, 5], 1, 'mean')</c> padded with 1 where
        ///     NumPy pads with 2**63).
        ///     </para>
        ///     <para>
        ///     One shortcut stays, because it gives NumPy's values: a 1-D chunk reduces flat (one run either way), and the 0-d
        ///     statistic broadcasts into the pad area as the (1,) one does.
        ///     </para>
        /// </remarks>
        private static NDArray StatReduce(NDArray chunk, int axis, string mode)
        {
            // Text takes NumPy's statement exactly (np.amax / np.amin / np.mean / np.median with the axis and
            // keepdims=True): a flat reduction changes the answer for text — np.mean of StringDType text reaches the
            // divide's no-loop refusal, a flat one the str / int refusal (probed against 2.4.2).
            if (chunk.typecode == NPTypeCode.String)
                return mode switch
                {
                    "maximum" => chunk.TensorEngine.ReduceAMax(chunk, axis, true, null, null, null, null),
                    "minimum" => chunk.TensorEngine.ReduceAMin(chunk, axis, true, null, null, null, null),
                    "mean" => chunk.TensorEngine.Mean(chunk, axis, null, true),
                    "median" => chunk.TensorEngine.Median(chunk, new[] {axis}, null, false, true),
                    _ => throw new NotSupportedException($"stat mode '{mode}'"),
                };

            // 1-D: the flat reduction of the one run (the same order and value as the axis form); a 0-d statistic, which
            // broadcasts identically.
            if (chunk.ndim == 1 && axis == 0)
            {
                switch (mode)
                {
                    case "maximum": return chunk.TensorEngine.ReduceAMax(chunk, null, false, null, null, null, null);
                    case "minimum": return chunk.TensorEngine.ReduceAMin(chunk, null, false, null, null, null, null);
                    case "mean":    return chunk.TensorEngine.Mean(chunk);
                    case "median":  return chunk.TensorEngine.Median(chunk, null, null, false, false);
                }
            }

            // stat_func(chunk, axis=axis, keepdims=True) on the view itself.
            switch (mode)
            {
                case "maximum": return chunk.TensorEngine.ReduceAMax(chunk, axis, true, null, null, null, null);
                case "minimum": return chunk.TensorEngine.ReduceAMin(chunk, axis, true, null, null, null, null);
                case "mean": return chunk.TensorEngine.Mean(chunk, axis, null, true);
                case "median": return chunk.TensorEngine.Median(chunk, new[] {axis}, null, false, true);
                default: throw new NotSupportedException($"stat mode '{mode}'");
            }
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
            var rounded = stat.TensorEngine.Round(stat, 0, null, null);   // np.around(stat)'s statement
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
            DType computeType = LinearRampComputeType(array, endValues);
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
                        // Each side is one np.linspace over the whole edge (NumPy's any_step_zero is decided per call).
                        if (left > 0)
                        {
                            using var ramp = BuildLinearRamp(leftEdge, axis, left, endPairs[axis, 0], computeType, padded.GetTypeCode, reverse: false);
                            var padSlices = _SliceAtAxis(new Slice(0, left), axis, padded.ndim);
                            AssignSliceValue(roi, padSlices, ramp);
                        }
                        if (right > 0)
                        {
                            using var ramp = BuildLinearRamp(rightEdge, axis, right, endPairs[axis, 1], computeType, padded.GetTypeCode, reverse: true);
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
                case string:
                    return (DTypeRegistry.Str.singleton, true);   // np.dtype("U")
                // A C# char is NumPy's np.uint16 scalar (the house boundary map), never a one-character str: over a
                // numeric array linspace computes with its code point, over a text array the promotion refuses it.
                case char:
                    return (np.uint16, true);
                case byte[]:
                    return (DTypeRegistry.Bytes.singleton, true);   // np.dtype("S")
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
        /// <param name="values">The C# values (the boundary map's Python scalars; a <c>char</c> is the np.uint16 scalar and
        ///     counts as an integer; an <see cref="NDArray"/> element counts as its dtype's kind).</param>
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
                    case string: anyStr = true; break;
                    // A C# char is np.uint16 (the house boundary map): an integer that is never negative. It cannot
                    // share the integer case below, whose Convert.ToDouble has no char conversion.
                    case char: anyInt = true; break;
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
            if (anyStr) return DTypeRegistry.Str.singleton;      // np.dtype("U")
            if (anyBytes) return DTypeRegistry.Bytes.singleton;  // np.dtype("S")
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
        ///     The dtype NumPy's <c>np.linspace</c> computes a ramp in: <c>_array_converter(start, stop).result_type(
        ///     ensure_inexact=True)</c>, the start's class or descriptor, the edge's descriptor and the abstract Python float.
        /// </summary>
        /// <param name="array">The input (its descriptor is every edge's: the edges are slices of the padded array).</param>
        /// <param name="endValues">The <c>end_values</c> argument as given (null for NumPy's default 0).</param>
        /// <returns>The computation dtype: float64 for an integer or bool input with NumPy scalar starts (the default 0 is an
        ///     <c>np.int64</c>, a float or a pair <c>np.float64</c>s), the input's own float dtype with Python-scalar starts
        ///     (<c>tolist()</c>'s weak values: a float16 input then computes in float16), complex128 where either side is
        ///     complex.</returns>
        /// <remarks>Every start comes from ONE <c>np.array(end_values)</c>, so one dtype serves every axis and side. Two
        ///     NumSharp dtypes without a NumPy class: <c>char</c> promotes as its uint16 proxy (an integer), <c>decimal</c>
        ///     computes in decimal (exact, no promotion).</remarks>
        private static DType LinearRampComputeType(NDArray array, object endValues)
        {
            if (array.typecode == NPTypeCode.Decimal)
                return np.@decimal;
            var (startDescr, strong) = LinearRampEndValuesArray(endValues);
            object start = strong ? startDescr : WeakStartClass(startDescr);
            DType edge = array.typecode == NPTypeCode.Char ? np.uint16 : array.dtype;
            return DTypePromotion.ResultType(start, edge, DTypeRegistry.PyFloat);
        }

        /// <summary>
        ///     NumPy's <c>np.linspace(start=end_value, stop=edge.squeeze(axis), num=width, endpoint=False,
        ///     dtype=padded.dtype, axis=axis)</c> inside <c>_get_linear_ramps</c>, statement for statement, the right ramp
        ///     reversed along the axis.
        /// </summary>
        /// <param name="edge">The 1-thick edge slice (the axis kept at length 1: it broadcasts as NumPy's squeezed stop).</param>
        /// <param name="axis">The padded axis.</param>
        /// <param name="width">The pad width on this side (at least 1; NumPy's zero-width linspace is empty).</param>
        /// <param name="endValue">The start: a boxed NumPy scalar, or a <see cref="PyListItem"/> Python scalar.</param>
        /// <param name="computeType">The linspace's computation dtype (<see cref="LinearRampComputeType"/>).</param>
        /// <param name="dtype">The padded array's dtype (linspace's <c>dtype=</c>).</param>
        /// <param name="reverse">Reverse along <paramref name="axis"/> (the right ramp).</param>
        /// <returns>The ramp of <paramref name="dtype"/>: <paramref name="width"/> long along <paramref name="axis"/>, the
        ///     edge's extent elsewhere (a reversed view for the right side).</returns>
        /// <exception cref="NotSupportedException">An integer input with a complex start: NumPy's <c>floor</c> of the complex
        ///     ramp has no loop (NumPy raises its TypeError).</exception>
        /// <remarks>
        ///     <para>
        ///     The arithmetic is linspace's, in <paramref name="computeType"/>: <c>delta = stop - start</c>, <c>step = delta /
        ///     num</c>, <c>y = arange(num) * step + start</c> - except that when ANY element of <c>step</c> is zero (a stop equal
        ///     to the start, or a difference that underflows), NumPy computes the whole call as <c>arange(num) / num * delta +
        ///     start</c> (its subnormal guard, gh-5437). The two orders round differently, so the test runs over the whole edge
        ///     like NumPy's. An integer <paramref name="dtype"/> floors the ramp before the cast (<c>linspace(0, -3, 2,
        ///     endpoint=False)</c> is <c>[0, -2]</c>); a bool one only casts (a nonzero ramp value is True).
        ///     </para>
        ///     <para>
        ///     The previous form computed <c>start + (k / num) * delta</c>, which rounds otherwise: <c>np.pad([0.7, 1.3, 2.9],
        ///     3, 'linear_ramp')</c> differed in the last bit, and a bool input failed on the bool subtract.
        ///     </para>
        /// </remarks>
        [NDScopedCovered] // only caller: PadLinearRamp ← [NDScoped] Pad (every intermediate is disposed here as well)
        private static NDArray BuildLinearRamp(NDArray edge, int axis, long width, object endValue,
            DType computeType, NPTypeCode dtype, bool reverse)
        {
            NPTypeCode work = computeType.typecode;
            // linspace's operands in its loop dtype: np.subtract(stop, start, dtype=dt) casts both, and the Python int num of
            // `delta / div` / `y /= div` is a weak operand of the same loop.
            using var start = NDArray.Scalar(NumSharp.Utilities.Converts.ChangeType(UnwrapPyListItem(endValue), work), work);
            using var div = NDArray.Scalar(NumSharp.Utilities.Converts.ChangeType(width, work), work);
            using var delta = edge.TensorEngine.Subtract(edge, start, computeType, null, null);
            using var step = delta / div;

            // y = arange(0, num, dtype=dt), laid along the axis (NumPy reshapes it to (num, 1, ...) and moves axis 0 to `axis`
            // at the end: the same elementwise products).
            int ndim = edge.ndim;
            var yShape = new long[ndim];
            for (int i = 0; i < ndim; i++) yShape[i] = i == axis ? width : 1L;
            using var counts = UnmanagedStorage.ArangeObj(0L, width, 1L, null);   // np.arange(width)
            using var countsTyped = counts.astype(computeType);
            using var y = countsTyped.reshape(Shape.Adopt(yShape));

            // any_step_zero over the whole call: the subnormal guard's order for every element, or the plain one for all.
            bool anyStepZero;
            using (var zero = step == 0)
                anyStepZero = zero.TensorEngine.Any(zero);
            NDArray scaled;
            if (anyStepZero)
            {
                using var fraction = y / div;
                scaled = fraction * delta;
            }
            else
            {
                scaled = y * step;
            }

            NDArray ramp;
            using (scaled)
                ramp = scaled + start;

            // linspace(dtype=padded.dtype): an integer dtype floors, then every dtype casts (bool: nonzero is True).
            if (ramp.GetTypeCode != dtype)
            {
                NDArray toCast = ramp;
                if (IsIntegerDtype(dtype) || dtype == NPTypeCode.Char)
                {
                    toCast = ramp.TensorEngine.Floor(ramp, null, null, null);
                    ramp.Dispose();
                }
                var cast = toCast.astype(NumSharp.NPTypeCodeExtensions.AsType(dtype));
                if (!ReferenceEquals(toCast, cast)) toCast.Dispose();
                ramp = cast;
            }

            if (!reverse)
                return ramp;
            // right_ramp[_slice_at_axis(slice(None, None, -1), axis)]: a reversed view, which keeps the ramp's buffer alive.
            var reversed = ramp[_SliceAtAxis(new Slice(null, null, -1), axis, ndim)];
            ramp.Dispose();
            return reversed;
        }

        // ---------------------------- helpers (1:1 with NumPy) ----------------------------

        /// <summary>
        ///     Allocate <c>np.empty</c> with shape <c>arr.shape + 2*pad</c>, optionally
        ///     fill with <paramref name="fillValue"/>, then copy <paramref name="array"/>
        ///     into the center.  Returns <c>(padded, originalAreaSlice)</c> where
        ///     <c>originalAreaSlice</c> identifies the unpadded region for later
        ///     <see cref="_ViewRoi"/> calls.
        /// </summary>
        /// <param name="array">The input.</param>
        /// <param name="padPairs">The (before, after) widths per axis (no row for a 0-d input).</param>
        /// <param name="fillValue">The pad area's value (null leaves it uninitialized, NumPy's <c>fill_value=None</c>).</param>
        /// <returns>The padded array (F-ordered when the input is F-contiguous and not C-contiguous, else C-ordered) and the
        ///     slices of its original area.</returns>
        /// <remarks>A 0-d input is its own padded array: NumPy fills the 0-d result and then copies the value over the
        ///     fill (<c>padded[()] = array</c>), so the value always survives - <c>np.pad(np.array(5), 2, func)</c> is 5,
        ///     never the callable's zero fill.</remarks>
        private static (NDArray padded, Slice[] originalSlice) _PadSimple(
            NDArray array, long[,] padPairs, object fillValue)
        {
            int ndim = array.ndim;
            // NumPy: 0-D input → newShape is the empty tuple; the fill is overwritten by the copy of the value.
            if (ndim == 0)
                return (array.copy(), System.Array.Empty<Slice>());

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
            var padded = new NDArray(array.dtype, Shape.Adopt(newDims, order), false);

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
        ///     Writes <c>values.left</c> into the left pad band and
        ///     <c>values.right</c> into the right pad band of
        ///     <paramref name="padded"/> along <paramref name="axis"/>. Values may
        ///     be scalars (broadcast-fill) or NDArrays (broadcast across non-axis dims).
        /// </summary>
        /// <param name="padded">The array (or region-of-interest view) whose pad bands are written.</param>
        /// <param name="axis">The padded axis.</param>
        /// <param name="width">The (before, after) widths: the bands are the first <c>width.left</c> and the last
        ///     <c>width.right</c> positions along <paramref name="axis"/> (an empty band is not written at all).</param>
        /// <param name="values">The (before, after) values: scalars (cast into the dtype by <see cref="AssignSliceValue"/>)
        ///     or arrays that broadcast into the band.</param>
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
        ///     NumPy's three statements after the dict branch of <c>np.pad</c>: <c>np.asarray(pad_width)</c>, its integral
        ///     test, and <c>_as_pairs(pad_width, array.ndim, as_index=True)</c>.
        /// </summary>
        /// <param name="padWidth">The widths in a form the facades take (an <see cref="int"/>, an <c>int[]</c>, an
        ///     <c>int[,]</c>, a <c>(before, after)</c> tuple, or the <c>long[,]</c> table <see cref="DictPadWidth"/> built),
        ///     or null.</param>
        /// <param name="ndim">The padded array's TRUE rank (0 for a 0-d array).</param>
        /// <returns>The <c>(ndim, 2)</c> width table.</returns>
        /// <exception cref="TypeError">A null <paramref name="padWidth"/>: NumPy's <c>np.asarray(None)</c> is an object
        ///     array, "`pad_width` must be of integral type.".</exception>
        /// <exception cref="ValueError">See <see cref="_AsPairs"/>.</exception>
        /// <remarks>Every C# form the facades take is an integer array (a C# integer is a Python int, a typed C# array an
        ///     ndarray of its own dtype), so only null fails the integral test here; the dict branch raises its own
        ///     TypeError for the tables that are not integer arrays.</remarks>
        private static long[,] PadWidthPairs(object padWidth, int ndim)
        {
            if (padWidth is null)
                throw new TypeError("`pad_width` must be of integral type.");
            return _AsPairs(padWidth, ndim, asIndex: true);
        }

        /// <summary>
        ///     NumPy's <c>_as_pairs(x, ndim, as_index=True)</c> for the integer pair arguments (<c>pad_width</c> after its
        ///     integral test, <c>stat_length</c>), statement for statement: <c>np.array(x)</c> rounded to <c>intp</c>; for
        ///     an array of fewer than 3 dimensions, ONE value pads both sides of every axis and TWO values (unless shaped
        ///     (2, 1)) are the (before, after) pair of every axis; otherwise the negative test over every value and
        ///     <c>np.broadcast_to(x, (ndim, 2))</c>.
        /// </summary>
        /// <param name="x">The argument: null (<c>stat_length=None</c>), a C# integer or real scalar, a two-item tuple, an
        ///     integer or real C# array of rank 1 or 2, an integer or real <see cref="NDArray"/>, or a dictionary (a
        ///     <c>stat_length</c> extension of NumSharp: NumPy refuses a dict there).</param>
        /// <param name="ndim">The number of pairs: the padded array's TRUE rank (0 for a 0-d array).</param>
        /// <param name="asIndex">The pairs are widths or lengths: a negative one is NumPy's ValueError. Every caller passes
        ///     true.</param>
        /// <returns>The <c>(ndim, 2)</c> table; for a null <paramref name="x"/> every entry is -1, the "whole axis"
        ///     sentinel of a <c>stat_length</c> of None.</returns>
        /// <exception cref="ValueError">A negative value with <paramref name="asIndex"/> ("index can't contain negative
        ///     values"; a NaN or out-of-range real rounds to <c>intp</c>'s minimum, which is negative too); an empty array
        ///     ("zero-size array to reduction operation minimum which has no identity", its <c>x.min()</c>); values that do
        ///     not broadcast to <c>(ndim, 2)</c> (<c>np.broadcast_to</c>'s "operands could not be broadcast together with
        ///     remapped shapes [original-&gt;remapped]: (3,)  and requested shape (2,2)", or "input operand has more
        ///     dimensions than allowed by the axis remapping" for 3 or more dimensions).</exception>
        /// <exception cref="ArgumentException">A value of a kind NumSharp does not read as an integer array, or a dictionary
        ///     axis out of range.</exception>
        /// <remarks>
        ///     <para>
        ///     The order is NumPy's: the one-value and pair shortcuts test their own values only and return, so a 3-D
        ///     array of one element skips them (NumPy's <c>x.ndim &lt; 3</c>); the negative test runs over every value
        ///     BEFORE the broadcast, so <c>[[1], [-2]]</c> is the negative-value error even where it would not broadcast,
        ///     and an empty array is <c>min()</c>'s error. A (2, 1) table is one width per axis for both sides
        ///     (<c>[[1], [2]]</c> pads axis 0 by 1 and axis 1 by 2), never a pair.
        ///     </para>
        ///     <para>
        ///     <c>np.round(x).astype(np.intp)</c> is the identity for integers; a real <c>stat_length</c> rounds half to even
        ///     (2.5 is 2, 3.5 is 4), and a NaN, an infinity or a value past int64 becomes int64's minimum, as x86's
        ///     conversion makes it in NumPy - which the negative test then refuses.
        ///     </para>
        /// </remarks>
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

            // np.array(x) as intp: the values in C order and the array's shape.
            if (!TryIndexArray(x, out long[] values, out long[] shape))
                throw new ArgumentException($"`pad_width` must be of integral type (got {x.GetType().Name})");

            if (shape.Length < 3)
            {
                // x was supplied as a single value: both sides of every axis.
                if (values.Length == 1)
                {
                    if (asIndex && values[0] < 0)
                        throw new ValueError("index can't contain negative values");
                    for (int i = 0; i < ndim; i++) { result[i, 0] = values[0]; result[i, 1] = values[0]; }
                    return result;
                }
                // x was supplied with a value for each side - except the (2, 1) table, one value per axis for both sides.
                if (values.Length == 2 && !(shape.Length == 2 && shape[0] == 2 && shape[1] == 1))
                {
                    if (asIndex && (values[0] < 0 || values[1] < 0))
                        throw new ValueError("index can't contain negative values");
                    for (int i = 0; i < ndim; i++) { result[i, 0] = values[0]; result[i, 1] = values[1]; }
                    return result;
                }
            }

            // x.min() < 0: over every value, before the broadcast (an empty array has no minimum).
            if (asIndex)
            {
                if (values.Length == 0)
                    throw new ValueError("zero-size array to reduction operation minimum which has no identity");
                foreach (long v in values)
                    if (v < 0)
                        throw new ValueError("index can't contain negative values");
            }

            // np.broadcast_to(x, (ndim, 2)).tolist()
            var (rowStride, colStride) = BroadcastToPairs(shape, ndim);
            for (int i = 0; i < ndim; i++)
            {
                result[i, 0] = values[i * rowStride];
                result[i, 1] = values[i * rowStride + colStride];
            }
            return result;
        }

        /// <summary>
        ///     <c>np.broadcast_to(x, (ndim, 2))</c> for an array of shape <paramref name="shape"/>: the C-order element
        ///     strides of the broadcast table's rows and columns, or NumPy's refusal.
        /// </summary>
        /// <param name="shape">The array's shape.</param>
        /// <param name="ndim">The number of rows (the padded array's true rank).</param>
        /// <returns>The flat-index step per row and per column (0 where the array's axis has length 1 or is missing).</returns>
        /// <exception cref="ValueError">The shape has more than 2 dimensions ("input operand has more dimensions than
        ///     allowed by the axis remapping"), or an axis is neither 1 nor the requested length (the nditer's text,
        ///     "operands could not be broadcast together with remapped shapes [original-&gt;remapped]: (3,)  and requested
        ///     shape (2,2)", whose two spaces are NumPy's: the operand's shape ends in a space before " and requested
        ///     shape ").</exception>
        private static (long rowStride, long colStride) BroadcastToPairs(long[] shape, int ndim)
        {
            if (shape.Length > 2)
                throw new ValueError("input operand has more dimensions than allowed by the axis remapping");
            // The shape right-aligned against (ndim, 2): a missing leading axis broadcasts like a length-1 one.
            long rows = shape.Length == 2 ? shape[0] : 1;
            long cols = shape.Length >= 1 ? shape[shape.Length - 1] : 1;
            if (cols != 1 && cols != 2 || rows != 1 && rows != ndim)
                throw new ValueError(
                    "operands could not be broadcast together with remapped shapes [original->remapped]: "
                    + Shape.ConvertShapeToString(shape) + "  and requested shape (" + ndim + ",2)");
            return (rows == 1 ? 0 : cols, cols == 1 ? 0 : 1);
        }

        /// <summary>
        ///     NumPy's <c>np.round(np.array(x)).astype(np.intp)</c> for the C# forms of an integer pair argument: the values
        ///     in C order and the array's shape.
        /// </summary>
        /// <param name="x">The argument (not null, not a dictionary).</param>
        /// <param name="values">The values as int64.</param>
        /// <param name="shape">The shape of NumPy's array: empty for a scalar.</param>
        /// <returns>False for a value NumSharp does not read as an integer or real array (text, complex, an object
        ///     sequence).</returns>
        /// <remarks>A C# integer or a <see cref="bool"/> is a Python int (a bool is 0 or 1), a real scalar a Python float
        ///     (rounded half to even); a two-item tuple is a Python tuple; a typed C# array of rank 1 or 2 is an ndarray of
        ///     its own dtype; an <see cref="NDArray"/> is read through its logical C order whatever its layout.</remarks>
        private static bool TryIndexArray(object x, out long[] values, out long[] shape)
        {
            switch (x)
            {
                case NDArray nd when nd.typecode != NPTypeCode.String && nd.typecode != NPTypeCode.Complex:
                {
                    shape = nd.shape;
                    values = new long[nd.size];
                    for (long i = 0; i < values.Length; i++)
                        if (!TryIndexValue(nd.GetAtIndex(i), out values[i]))
                            return false;
                    return true;
                }
                case System.Runtime.CompilerServices.ITuple t when t.Length == 2:
                {
                    shape = new long[] { 2 };
                    values = new long[2];
                    return TryIndexValue(t[0], out values[0]) && TryIndexValue(t[1], out values[1]);
                }
                case Array arr when arr.Rank <= 2 && (arr.Rank == 1 || arr.GetType().GetElementType() != typeof(object)):
                {
                    shape = arr.Rank == 1 ? new long[] { arr.Length } : new long[] { arr.GetLength(0), arr.GetLength(1) };
                    values = new long[arr.Length];
                    int k = 0;
                    // A rectangular C# array enumerates in row-major order, NumPy's C order.
                    foreach (var item in arr)
                        if (!TryIndexValue(item, out values[k++]))
                            return false;
                    return true;
                }
                default:
                    shape = System.Array.Empty<long>();
                    values = new long[1];
                    return TryIndexValue(x, out values[0]);
            }
        }

        /// <summary>
        ///     One value of <c>np.round(x).astype(np.intp)</c>: an integer as it is (a <c>ulong</c> past int64 wraps, as
        ///     NumPy's cast of a uint64 does), a bool as 0 / 1, a real rounded half to even (NaN, infinities and values past
        ///     int64 become int64's minimum, x86's conversion result in NumPy).
        /// </summary>
        /// <param name="v">The boxed value.</param>
        /// <param name="result">The int64.</param>
        /// <returns>False for anything else (text, complex, null, a nested sequence).</returns>
        private static bool TryIndexValue(object v, out long result)
        {
            if (TryToInt64(v, out result))
                return true;
            double d;
            switch (v)
            {
                case bool b: result = b ? 1 : 0; return true;
                case char c: result = c; return true;
                case Half h: d = (double)h; break;
                case float f: d = f; break;
                case double dd: d = dd; break;
                case decimal m: d = (double)m; break;
                default: result = 0; return false;
            }
            double r = Math.Round(d, MidpointRounding.ToEven);
            // .NET's conversion saturates; NumPy's astype(intp) on x86 (cvttsd2si) gives int64's minimum for a NaN, an
            // infinity or a value outside int64.
            result = double.IsNaN(r) || r >= 9223372036854775808.0 || r < -9223372036854775808.0 ? long.MinValue : (long)r;
            return true;
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
        ///     NumPy's <c>_as_pairs(x, ndim)</c> for the value pair arguments (<c>constant_values</c>, <c>end_values</c>),
        ///     statement for statement: <c>np.array(x)</c>; for an array of fewer than 3 dimensions, ONE value is both sides
        ///     of every axis and TWO values (unless shaped (2, 1)) the (before, after) pair of every axis - handed out as
        ///     NumPy SCALARS (<c>x[0]</c>, <c>x[1]</c>); any other array goes through <c>np.broadcast_to(x, (ndim,
        ///     2)).tolist()</c>, whose values are PYTHON scalars (<see cref="PyListItem"/>).
        /// </summary>
        /// <param name="x">The argument: null (not given), a C# scalar, a string or <c>byte[]</c>, a tuple, a C# array or
        ///     collection, an <see cref="NDArray"/>, or a dictionary (a NumSharp form: NumPy refuses a dict here).</param>
        /// <param name="ndim">The number of axes: the padded array's TRUE rank (0 for a 0-d array, whose values are still
        ///     checked against (0, 2)).</param>
        /// <returns>The (ndim, 2) table: null cells for a null <paramref name="x"/>; a boxed C# value for a NumPy scalar
        ///     (a C# number of its own type, a string for <c>np.str_</c>, a <c>byte[]</c> for <c>np.bytes_</c>); a
        ///     <see cref="PyListItem"/> for a Python scalar of <c>tolist()</c>.</returns>
        /// <exception cref="ValueError">The values do not broadcast to (ndim, 2): NumPy's <c>np.broadcast_to</c> texts
        ///     (<see cref="BroadcastToPairs"/>).</exception>
        /// <exception cref="ArgumentException">A dictionary axis out of bounds.</exception>
        /// <exception cref="NotSupportedException">A value NumSharp has no array form for (np.asanyarray's refusal).</exception>
        /// <remarks>
        ///     <para>
        ///     The kind of each value decides how NumPy's <c>padded[...] = value</c> stores it (<see cref="CastBoxToDType"/>):
        ///     a NumPy scalar is int()-converted with the range check into a SIGNED integer array and CAST into every other
        ///     (<c>np.int32(-1)</c> into uint8 is 255), while a Python scalar is range-checked into EVERY integer array
        ///     (<c>-1</c> into uint8 is "Python integer -1 out of bounds for uint8") and a Python str's error quotes the str
        ///     (<c>'x'</c>, not <c>np.str_('x')</c>). So <c>constant_values=[[1, -1]]</c> pads a uint8 array with 1 and 255
        ///     where <c>[[1, -1], [2, -2]]</c> raises: probed against 2.4.2.
        ///     </para>
        ///     <para>
        ///     A (2, 1) table is one value per axis for both sides (<c>[[10], [30]]</c> pads axis 0 with 10 and axis 1 with
        ///     30), and a 3-D array of one element skips the shortcuts and fails the broadcast, as in NumPy.
        ///     </para>
        /// </remarks>
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

            // A scalar: np.array(x) is 0-d, its one value the NumPy scalar x[0] on both sides of every axis (CastBoxToDType
            // packs a boxed C# number as the NumPy scalar of its own type, a string as np.str_, a byte[] as np.bytes_).
            if (IsStoreScalar(x) || x is string || x is byte[])
            {
                for (int i = 0; i < ndim; i++) { result[i, 0] = x; result[i, 1] = x; }
                return result;
            }

            // np.array(x): an NDArray as it is; a C# array, tuple or collection through NumPy's coercion of a sequence (a
            // tuple of C# ints is int64, an object[] promotes its items in NumPy's order).
            NDArray arr = x as NDArray;
            bool owned = arr is null;
            if (owned)
                arr = AnyArrayFromObject(x, null, 'K');
            try
            {
                if (arr.ndim < 3)
                {
                    // x was supplied as a single value: the NumPy scalar x[0] on both sides of every axis.
                    if (arr.size == 1)
                    {
                        object v = arr.GetAtIndex(0);
                        for (int i = 0; i < ndim; i++) { result[i, 0] = v; result[i, 1] = v; }
                        return result;
                    }
                    // A value for each side - except the (2, 1) table, one value per axis for both sides.
                    if (arr.size == 2 && !(arr.ndim == 2 && arr.Shape.dimensions[0] == 2 && arr.Shape.dimensions[1] == 1))
                    {
                        object before = arr.GetAtIndex(0), after = arr.GetAtIndex(1);
                        for (int i = 0; i < ndim; i++) { result[i, 0] = before; result[i, 1] = after; }
                        return result;
                    }
                }

                // np.broadcast_to(x, (ndim, 2)).tolist(): every value a Python scalar.
                var (rowStride, colStride) = BroadcastToPairs(arr.shape, ndim);
                for (int i = 0; i < ndim; i++)
                {
                    result[i, 0] = new PyListItem(arr.GetAtIndex(i * rowStride));
                    result[i, 1] = new PyListItem(arr.GetAtIndex(i * rowStride + colStride));
                }
                return result;
            }
            finally
            {
                if (owned)
                    arr.Dispose();
            }
        }

        /// <summary>
        ///     A value of NumPy's <c>ndarray.tolist()</c> inside <see cref="_AsPairsValues"/>: the Python scalar of an array
        ///     element - a Python int for every integer dtype, a Python float for every real float dtype (float16 included),
        ///     a Python complex, a Python bool, a Python str or bytes - which NumPy's store packs by its Python-scalar rules.
        /// </summary>
        /// <remarks>
        ///     A marker type, because the boxed C# values of the two kinds coincide (a boxed <see cref="int"/> is both the
        ///     element of an int32 array and that element's Python int): <see cref="CastBoxToDType"/> packs a marked value
        ///     with <see cref="ReductionInitial.TryPack"/>, an unmarked one with <see cref="ReductionInitial.PackNumPyScalar"/>.
        /// </remarks>
        private sealed class PyListItem
        {
            /// <summary>
            ///     Wraps one array element as the Python scalar <c>tolist()</c> makes of it.
            /// </summary>
            /// <param name="element">The boxed element (<see cref="NDArray.GetAtIndex(long)"/>).</param>
            /// <remarks>A <see cref="Half"/> becomes the <see cref="double"/> of its value (a Python float: the house map reads
            ///     a boxed Half as the NumPy scalar np.float16) and a <c>char</c> its code point (NumSharp's uint16 proxy, an
            ///     integer). Every other boxed type already reads as the Python scalar of the house map.</remarks>
            public PyListItem(object element)
                => Value = element switch
                {
                    Half h => (double)h,
                    char c => (long)c,
                    _ => element,
                };

            /// <summary>The Python scalar, in the house boundary map's C# form.</summary>
            public object Value { get; }

            /// <summary>The value's own text, for debugging.</summary>
            /// <returns><see cref="Value"/>'s text.</returns>
            public override string ToString() => Value?.ToString() ?? "None";
        }

        /// <summary>
        ///     A constant or end value as the plain C# value the computation reads: a <see cref="PyListItem"/> unwrapped to its
        ///     Python scalar, anything else as it is.
        /// </summary>
        /// <param name="value">A cell of <see cref="_AsPairsValues"/>.</param>
        /// <returns>The value.</returns>
        private static object UnwrapPyListItem(object value) => value is PyListItem py ? py.Value : value;

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
        ///     <see cref="NDArray"/> (its element), a <see cref="PyListItem"/> (a Python scalar of <c>tolist()</c>), or
        ///     <c>null</c> (NumPy's default 0).</param>
        /// <param name="tc">The padded array's dtype.</param>
        /// <returns>The constant boxed as <paramref name="tc"/>'s exact CLR type; an array of several elements or a value
        ///     that is not a number returns through the house conversion (<see cref="NumSharp.Utilities.Converts.ChangeType(object, NPTypeCode)"/>)
        ///     or as is (an array), for the broadcast-assignment route.</returns>
        /// <exception cref="OverflowException">A signed integer array and a constant whose int() does not fit it:
        ///     <c>np.pad(int8, 1, constant_values=1000)</c> raises "Python integer 1000 out of bounds for int8", as in NumPy
        ///     (an unsigned array casts instead: <c>np.pad(uint8, 1, constant_values=1000)</c> pads with 232); a float
        ///     infinity into a signed integer; a Python int of <c>tolist()</c> out of ANY integer array's range
        ///     ("Python integer -1 out of bounds for uint8").</exception>
        /// <exception cref="ValueError">A float NaN constant into a signed integer array ("cannot convert float NaN to
        ///     integer"); a text constant into a numeric array whose text does not parse (the cast of NumPy's
        ///     <c>np.str_</c> scalar: "invalid literal for int() with base 10: np.str_('z')"; a Python str of
        ///     <c>tolist()</c> quotes the str itself, <c>'z'</c>).</exception>
        /// <exception cref="TypeError">A Python complex of <c>tolist()</c> into a real or integer array (float()'s /
        ///     int()'s text).</exception>
        private static unsafe object CastBoxToDType(object value, NPTypeCode tc)
        {
            if (value is null) return BoxedZero(tc);
            // A Python scalar of _as_pairs' tolist() packs by PyArray_Pack's Python-scalar rules: an int is range-checked
            // into every integer dtype (a NumPy scalar only into a signed one), a str parses quoting the str itself. A text
            // array converts its constants on its own route, so it receives the plain value.
            if (value is PyListItem py)
            {
                if (tc == NPTypeCode.String)
                    return py.Value;
                byte* pySlot = stackalloc byte[ReductionInitial.SlotBytes];
                var pySpan = new Span<byte>(pySlot, ReductionInitial.SlotBytes);
                ReductionInitial.TryPack(py.Value, tc, pySpan);
                return ReductionInitial.Box(pySpan, tc);
            }
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
            // A C# char is not text here: the house boundary map makes it NumPy's np.uint16 scalar, which the pack below
            // int()-converts into a signed integer array (np.pad(int8, 1, constant_values=(char)300) raises "Python
            // integer 300 out of bounds for int8") and casts into every other dtype, as the weak_scalar tier pins.
            if (value is string or byte[] && tc != NPTypeCode.String)
            {
                using var text = Backends.UnmanagedStorage.FromAnyScalar(value, null);
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
        ///     wrapped in a 0-D NDArray and broadcast via <see cref="NDIter.Copy(NDArray, NDArray)"/>'s
        ///     broadcast fast path (convert-once + memcpy per outer row).
        ///
        ///     Routes through <see cref="NDIter.Copy(NDArray, NDArray)"/> directly rather than the
        ///     <c>padded[slices] = value</c> indexer — the indexer's contiguous
        ///     fast path skips broadcast stretching when both shapes are contig
        ///     but sizes differ, which truncates broadcast writes (e.g. a
        ///     <c>(1, N)</c> edge view into a <c>(pad_width, N)</c> band only
        ///     fills the first row). <see cref="NDIter.Copy(NDArray, NDArray)"/> always honours
        ///     broadcasting.
        /// </summary>
        /// <param name="padded">The array written into.</param>
        /// <param name="slices">The band: one slice per axis of <paramref name="padded"/>.</param>
        /// <param name="value">A boxed scalar already converted to <paramref name="padded"/>'s dtype, or an array that
        ///     broadcasts into the band (cast by the copy).</param>
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

        /// <summary>
        ///     NumPy's mode lookup and keyword check, statement for statement: <c>set(kwargs) - set(allowed_kwargs[mode])</c>,
        ///     whose missing key is "mode '...' is not supported" and whose non-empty difference names the keywords the mode
        ///     does not take.
        /// </summary>
        /// <param name="mode">The mode's name; null is NumPy's <c>None</c>.</param>
        /// <param name="constantValues">The <c>constant_values</c> argument (null: not given).</param>
        /// <param name="endValues">The <c>end_values</c> argument (null: not given).</param>
        /// <param name="statLength">The <c>stat_length</c> argument (null: not given).</param>
        /// <param name="reflectType">The <c>reflect_type</c> argument (null or the facades' default <c>"even"</c>: not
        ///     given).</param>
        /// <exception cref="ValueError">"mode 'None' is not supported" / "mode 'Constant' is not supported" for a name outside
        ///     NumPy's table; "unsupported keyword arguments for mode 'edge': {'constant_values', 'stat_length'}" for keywords
        ///     the mode does not take.</exception>
        /// <remarks>
        ///     <para>
        ///     C# cannot tell a keyword left at its default from one given with that value, so the facades' defaults stand for
        ///     "not given": a null <c>constant_values</c> / <c>end_values</c> / <c>stat_length</c> (NumPy's defaults 0, 0 and
        ///     None follow from it in the mode bodies) and a <c>reflect_type</c> of null or <c>"even"</c> (NumPy's
        ///     <c>kwargs.get("reflect_type", "even")</c>). Any other <c>reflect_type</c> is given, and accepted by
        ///     <c>'reflect'</c> / <c>'symmetric'</c> whatever it says: NumPy has no check of its value, its loop tests
        ///     <c>method == "odd"</c> and reads everything else as even.
        ///     </para>
        ///     <para>
        ///     The keyword set is Python's: <c>{'end_values', 'stat_length'}</c>. With two or more names NumPy's order follows
        ///     the strings' hashes, which Python randomizes per process, so no order is NumPy's; NumSharp lists them in the
        ///     signature's order (constant_values, end_values, stat_length, reflect_type).
        ///     </para>
        /// </remarks>
        private static void ValidateModeKwargs(string mode, object constantValues, object endValues,
            object statLength, string reflectType)
        {
            // allowed_kwargs[mode]: a KeyError for any other name, None included (NumPy's f-string prints it as 'None').
            string[] allowed = mode switch
            {
                "empty" or "edge" or "wrap" => System.Array.Empty<string>(),
                "constant" => new[] { "constant_values" },
                "linear_ramp" => new[] { "end_values" },
                "maximum" or "mean" or "median" or "minimum" => new[] { "stat_length" },
                "reflect" or "symmetric" => new[] { "reflect_type" },
                _ => null,
            };
            if (allowed is null)
                throw new ValueError($"mode '{mode ?? "None"}' is not supported");

            // set(kwargs) - set(allowed): the keywords the caller gave that the mode does not take.
            var unsupported = new List<string>(4);
            void Check(bool given, string name)
            {
                if (given && System.Array.IndexOf(allowed, name) < 0)
                    unsupported.Add("'" + name + "'");
            }
            Check(constantValues != null, "constant_values");
            Check(endValues != null, "end_values");
            Check(statLength != null, "stat_length");
            Check(reflectType != null && reflectType != "even", "reflect_type");
            if (unsupported.Count > 0)
                throw new ValueError($"unsupported keyword arguments for mode '{mode}': {{{string.Join(", ", unsupported)}}}");
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
