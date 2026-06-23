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
        //   * padded[band] = src routes through NpyIter.Copy → StridedCastKernel
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
            => PadImpl(array, (object)pad_width, mode, constant_values, end_values, stat_length, reflect_type);

        public static NDArray pad(NDArray array, int[] pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
            => PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);

        public static NDArray pad(NDArray array, int[,] pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
            => PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);

        public static NDArray pad(NDArray array, (int before, int after) pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
            => PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);

        public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, string mode = "constant",
            object constant_values = null, object end_values = null,
            object stat_length = null, string reflect_type = "even")
            => PadImpl(array, pad_width, mode, constant_values, end_values, stat_length, reflect_type);

        // Callable mode (overloads matching each pad_width shape).
        public static NDArray pad(NDArray array, int pad_width, PadFunc mode, object kwargs = null)
            => PadCallableImpl(array, (object)pad_width, mode, kwargs);

        public static NDArray pad(NDArray array, int[] pad_width, PadFunc mode, object kwargs = null)
            => PadCallableImpl(array, pad_width, mode, kwargs);

        public static NDArray pad(NDArray array, int[,] pad_width, PadFunc mode, object kwargs = null)
            => PadCallableImpl(array, pad_width, mode, kwargs);

        public static NDArray pad(NDArray array, (int before, int after) pad_width, PadFunc mode, object kwargs = null)
            => PadCallableImpl(array, pad_width, mode, kwargs);

        public static NDArray pad(NDArray array, IDictionary<int, object> pad_width, PadFunc mode, object kwargs = null)
            => PadCallableImpl(array, pad_width, mode, kwargs);

        // ---------------------------- top-level dispatcher ----------------------------

        private static NDArray PadImpl(NDArray array, object pad_width, string mode,
            object constant_values, object end_values, object stat_length, string reflect_type)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (mode is null) throw new ArgumentNullException(nameof(mode));

            int ndim = Math.Max(array.ndim, 1); // 0-D treated as 1-D for pad_width normalization
            long[,] padPairs = _AsPairs(pad_width, ndim, asIndex: true);

            ValidateModeKwargs(mode, constant_values, end_values, stat_length, reflect_type);

            // No-op fast path: every pad value is zero. Still allocates a new array per NumPy.
            if (AllPairsZero(padPairs))
                return array.copy();

            // Empty-axis guard: only 'constant' and 'empty' may extend an empty axis.
            if (array.size == 0 && mode != "constant" && mode != "empty")
            {
                for (int ax = 0; ax < array.ndim; ax++)
                {
                    if (array.shape[ax] == 0 && (padPairs[ax, 0] != 0 || padPairs[ax, 1] != 0))
                        throw new ArgumentException(
                            $"can't extend empty axis {ax} using modes other than 'constant' or 'empty'");
                }
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
                long axisLen = view.shape[outerNdim];
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
                for (int i = 0; i < outerNdim; i++) totalOuter *= view.shape[i];
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
                        if (coords[d] < view.shape[d]) break;
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
                long originalPeriod = roi.shape[axis] - right - left;
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
                if (array.shape[axis] == 1)
                {
                    var (leftEdge, rightEdge) = _GetEdges(padded, axis, (left, right));
                    try { _SetPadArea(padded, axis, (left, right), (leftEdge, rightEdge)); }
                    finally { leftEdge.Dispose(); rightEdge.Dispose(); }
                    continue;
                }

                var roi = _ViewRoi(padded, originalSlice, axis);
                long originalPeriod = array.shape[axis];
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
        private static (long left, long right) _SetReflectBoth(
            NDArray padded, int axis, (long left, long right) width, string method, long originalPeriod, bool includeEdge)
        {
            long leftPad = width.left, rightPad = width.right;
            long oldLength = padded.shape[axis] - rightPad - leftPad;
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

                    long padStart = padded.shape[axis] - rightPad;
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
            long period = padded.shape[axis] - rightPad - leftPad;
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
            if (ndim == 0) return array.copy();
            // -1 sentinel == "use full valid axis length"
            long[,] lengthPairs = _AsPairs(statLength, ndim, asIndex: true);
            var (padded, originalSlice) = _PadSimple(array, padPairs, fillValue: null);

            for (int axis = 0; axis < ndim; axis++)
            {
                long left = padPairs[axis, 0], right = padPairs[axis, 1];
                if (left == 0 && right == 0) continue;
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
            long rightIndex = padded.shape[axis] - width.right;
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

        private static NDArray StatReduce(NDArray chunk, int axis, string mode)
        {
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
                        long n = src.shape[axis];
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

        private static NDArray PadLinearRamp(NDArray array, long[,] padPairs, object endValues)
        {
            int ndim = array.ndim;
            if (ndim == 0) return array.copy();
            object[,] endPairs = _AsPairsValues(endValues ?? 0, ndim);
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
                            long axisLen = roi.shape[axis];
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
            var coefReshaped = coef.reshape(new Shape(coefShape));

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
                long size = array.shape[i];
                long left = padPairs[i, 0];
                long right = padPairs[i, 1];
                newDims[i] = left + size + right;
                originalSlice[i] = new Slice(left, left + size);
            }

            // NumPy: order = 'F' if array.flags.fnc else 'C'  (F-contig and NOT also C).
            char order = (array.Shape.IsFContiguous && !array.Shape.IsContiguous) ? 'F' : 'C';
            var padded = np.empty(new Shape(newDims, order), array.dtype);

            if (fillValue != null)
            {
                // Whole-buffer fill via the underlying IArraySlice.Fill — uses
                // InitBlockUnaligned for byte-sized dtypes, 8x-unrolled scalar
                // otherwise. Routes around the per-axis broadcast loop.
                padded.Storage.InternalArray.Fill(CastBoxToDType(fillValue, array.GetTypeCode));
            }

            // Copy original values into the center.  padded[originalSlice] = array
            // routes through NpyIter.Copy → IL StridedCastKernel (per-row memcpy).
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
                long axisLen = padded.shape[axis];
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
            long axisLen = padded.shape[axis];
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
        ///     NumPy <c>_as_pairs</c> port for integer pair specs (pad_width / stat_length).
        ///     Broadcasts <paramref name="x"/> to a <c>long[ndim, 2]</c> table.
        ///     Accepts: <c>null</c>, scalar int, <c>int[]</c> / <c>long[]</c>,
        ///     <c>int[,]</c> / <c>long[,]</c>, <c>(int, int)</c>, or
        ///     <c>IDictionary&lt;int, object&gt;</c>.
        /// </summary>
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

            // Dict path: per-axis (negative keys allowed) overlaid on (0,0) default
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
                    throw new ArgumentException("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = b; result[i, 1] = a; }
                return result;
            }
            if (x is ValueTuple<long, long> vtl)
            {
                long b = vtl.Item1, a = vtl.Item2;
                if (asIndex && (b < 0 || a < 0))
                    throw new ArgumentException("index can't contain negative values");
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
                        throw new ArgumentException("index can't contain negative values");
                    for (int i = 0; i < ndim; i++) { result[i, 0] = i2[0, 0]; result[i, 1] = i2[0, 1]; }
                    return result;
                }
                if (rows != ndim)
                    throw new ArgumentException($"pad_width shape ({rows},{cols}) is not broadcastable to ({ndim},2)");
                for (int i = 0; i < rows; i++)
                {
                    long b = i2[i, 0], a = i2[i, 1];
                    if (asIndex && (b < 0 || a < 0))
                        throw new ArgumentException("index can't contain negative values");
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
                        throw new ArgumentException("index can't contain negative values");
                    for (int i = 0; i < ndim; i++) { result[i, 0] = l2[0, 0]; result[i, 1] = l2[0, 1]; }
                    return result;
                }
                if (rows != ndim)
                    throw new ArgumentException($"pad_width shape ({rows},{cols}) is not broadcastable to ({ndim},2)");
                for (int i = 0; i < rows; i++)
                {
                    long b = l2[i, 0], a = l2[i, 1];
                    if (asIndex && (b < 0 || a < 0))
                        throw new ArgumentException("index can't contain negative values");
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
                    throw new ArgumentException("index can't contain negative values");
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
                    throw new ArgumentException("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = arr[0]; result[i, 1] = arr[0]; }
                return result;
            }
            if (arr.Length == 2)
            {
                if (asIndex && (arr[0] < 0 || arr[1] < 0))
                    throw new ArgumentException("index can't contain negative values");
                for (int i = 0; i < ndim; i++) { result[i, 0] = arr[0]; result[i, 1] = arr[1]; }
                return result;
            }
            if (arr.Length == ndim)
            {
                for (int i = 0; i < ndim; i++)
                {
                    if (asIndex && arr[i] < 0)
                        throw new ArgumentException("index can't contain negative values");
                    result[i, 0] = arr[i]; result[i, 1] = arr[i];
                }
                return result;
            }
            throw new ArgumentException($"pad_width length {arr.Length} is not broadcastable to (ndim={ndim}, 2)");
        }

        private static (long before, long after) NormalizeSinglePair(object value, bool asIndex)
        {
            if (value is null) throw new ArgumentException("pad_width dict value cannot be null");
            if (TryToInt64(value, out long scalar))
            {
                if (asIndex && scalar < 0)
                    throw new ArgumentException("index can't contain negative values");
                return (scalar, scalar);
            }
            if (value is ValueTuple<int, int> vti) return (vti.Item1, vti.Item2);
            if (value is ValueTuple<long, long> vtl) return (vtl.Item1, vtl.Item2);
            if (value is int[] arr && arr.Length == 2) return (arr[0], arr[1]);
            if (value is long[] arrL && arrL.Length == 2) return (arrL[0], arrL[1]);
            throw new ArgumentException($"pad_width dict value must be int or (before, after) pair");
        }

        /// <summary>
        ///     Same as <see cref="_AsPairs"/> but for value pairs (constant_values, end_values)
        ///     where the per-axis value is a scalar of any dtype (not necessarily integer).
        ///     Returns <c>object[ndim, 2]</c> with each cell holding the boxed scalar.
        /// </summary>
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
                if (rows != ndim)
                    throw new ArgumentException($"constant_values shape ({rows},{cols}) not broadcastable to ({ndim},2)");
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
            throw new ArgumentException($"values length {arr.Length} not broadcastable to (ndim={ndim}, 2)");
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

        private static object CastBoxToDType(object value, NPTypeCode tc)
        {
            if (value is null) return BoxedZero(tc);
            return NumSharp.Utilities.Converts.ChangeType(value, tc);
        }

        // ---------------------------- assignment glue ----------------------------

        /// <summary>
        ///     Assign <paramref name="value"/> (scalar or NDArray) into the slice
        ///     <paramref name="slices"/> of <paramref name="padded"/>. Scalars are
        ///     wrapped in a 0-D NDArray and broadcast via <see cref="NpyIter.Copy"/>'s
        ///     broadcast fast path (convert-once + memcpy per outer row).
        ///
        ///     Routes through <see cref="NpyIter.Copy"/> directly rather than the
        ///     <c>padded[slices] = value</c> indexer — the indexer's contiguous
        ///     fast path skips broadcast stretching when both shapes are contig
        ///     but sizes differ, which truncates broadcast writes (e.g. a
        ///     <c>(1, N)</c> edge view into a <c>(pad_width, N)</c> band only
        ///     fills the first row). <see cref="NpyIter.Copy"/> always honours
        ///     broadcasting.
        /// </summary>
        private static void AssignSliceValue(NDArray padded, Slice[] slices, object value)
        {
            var view = padded[slices];
            try
            {
                if (value is NDArray nd)
                {
                    NpyIter.Copy(view, nd);
                    return;
                }
                using var scalarNd = NDArray.Scalar(value, padded.GetTypeCode);
                NpyIter.Copy(view, scalarNd);
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

    }
}
