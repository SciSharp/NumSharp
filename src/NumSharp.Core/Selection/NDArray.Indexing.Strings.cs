using System;
using NumSharp.Backends;
using NumSharp.Backends.Strings;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        // =============================================================================================================
        // Advanced (fancy / boolean) indexing of fixed-width string arrays.
        //
        // A string element has no element kernel; its bytes are copied by the numeric machinery instead. So an advanced
        // get or set runs the SAME index on the array's UNIT VIEW (uint8 for S, uint32 for U, one extra trailing axis of
        // characters — UnmanagedStorage.StringUnitView) with one more ':' for that axis, and the unit result is re-tagged
        // as the string array it holds. The appended ':' can only land on the character axis or on a string axis (where
        // it equals the implicit full slice), so a valid index means the same thing on both arrays; an index that is too
        // long for the string array is too long for the unit view too. When the unit route raises, the index is replayed
        // on a numeric PROXY of the string array's own shape, so the error names the string shapes (NumSharp's numeric
        // texts for the same shapes) instead of the unit view's.
        // =============================================================================================================

        /// <summary>
        ///     Whether a normalized index tuple holds an ADVANCED component — an index array or mask, a raw integer array, a
        ///     nested tuple, or a scalar <see cref="bool"/> — rather than only slices, slice strings and integer scalars
        ///     (which the view path serves for strings unchanged).
        /// </summary>
        /// <param name="normalized">The index tuple after <see cref="NormalizeIndexInputs"/>.</param>
        /// <returns>True when the tuple needs the advanced (copying) path.</returns>
        private static bool HasAdvancedIndex(object[] normalized)
        {
            foreach (var item in normalized)
            {
                switch (item)
                {
                    case Slice _:
                    case string _:
                    case int _:
                    case long _:
                    case uint _:
                    case short _:
                    case ushort _:
                    case byte _:
                    case sbyte _:
                        continue;
                    default:
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        ///     The index tuple for the unit view. Without an <see cref="Slice.Ellipsis"/> it is the string array's own index:
        ///     the character axis is a trailing axis the index does not reach, so it is implicitly whole — and an index that
        ///     is ONE integer array then reaches the fancy-index kernel route (whole-element slabs) instead of the general
        ///     index path an explicit trailing <c>:</c> forces (measured 28x slower: <c>units[idx, :]</c> 10.2 ms vs
        ///     <c>units[idx]</c> 0.37 ms at 100K <c>&lt;U8</c>). An ellipsis would stretch over the character axis, so that
        ///     tuple gets <see cref="Slice.All"/> appended to pin it.
        /// </summary>
        /// <param name="normalized">The string array's (normalized) index tuple.</param>
        /// <returns>The same tuple, or a new one item longer when it holds an ellipsis.</returns>
        private static object[] WithCharacterAxis(object[] normalized)
        {
            bool ellipsis = false;
            foreach (var item in normalized)
                ellipsis |= item is Slice s && s.IsEllipsis;
            if (!ellipsis)
                return normalized;
            var unit = new object[normalized.Length + 1];
            System.Array.Copy(normalized, unit, normalized.Length);
            unit[normalized.Length] = Slice.All;
            return unit;
        }

        /// <summary>
        ///     A boolean-mask gather on a string array's unit view. When the mask covers the leading axes EXACTLY
        ///     (<c>mask.shape == units.shape[:mask.ndim]</c>, at least one axis) and the unit view is C-contiguous, those axes
        ///     are one run of rows: the gather is <c>flatnonzero(mask)</c> plus one integer-index gather of whole rows, which
        ///     takes the fancy-index kernel route — every other case runs <paramref name="fallback"/>, the route it replaces.
        /// </summary>
        /// <param name="units">The unit view (<c>shape + (chars,)</c>).</param>
        /// <param name="mask">The boolean mask over the string array's leading axes.</param>
        /// <param name="fallback">The general mask gather, for a mask or layout the fast path does not cover.</param>
        /// <returns>The gathered unit rows: <c>(count,) + units.shape[mask.ndim:]</c>, a fresh C-contiguous array.</returns>
        /// <remarks>
        ///     NumPy gathers the selected elements in C order of the mask, which is the order <c>flatnonzero</c> lists them;
        ///     <c>(count,) + trailing</c> is NumPy's result shape for a prefix mask.
        /// </remarks>
        private static NDArray MaskedUnitRows(NDArray units, NDArray mask, Func<NDArray, NDArray> fallback)
        {
            int k = mask.ndim;
            if (k == 0 || k >= units.ndim || !units.Shape.IsContiguous)
                return fallback(units);
            long rows = 1;
            for (int d = 0; d < k; d++)
            {
                if (mask.Shape.dimensions[d] != units.Shape.dimensions[d])
                    return fallback(units);
                rows *= units.Shape.dimensions[d];
            }

            var dims = new long[1 + units.ndim - k];
            dims[0] = rows;
            for (int d = k; d < units.ndim; d++)
                dims[1 + d - k] = units.Shape.dimensions[d];
            using var flat = units.reshape(Shape.Adopt(dims)); // a view: the unit view is C-contiguous
            using var picks = np.flatnonzero(mask);
            return flat[picks];
        }

        /// <summary>
        ///     The unit view of this string array as an <see cref="NDArray"/> the caller disposes (it shares this array's
        ///     memory and engine).
        /// </summary>
        /// <returns>A <c>uint8</c>/<c>uint32</c> array of shape <c>shape + (chars,)</c>.</returns>
        private NDArray UnitArray() => new NDArray(Storage.StringUnitView()) { TensorEngine = TensorEngine };

        /// <summary>
        ///     An advanced GET on this string array through its unit view: <paramref name="onUnits"/> selects on the unit
        ///     array, the result is turned back into a string array (a view stays a view of this array, a fresh gather is
        ///     re-tagged without a copy).
        /// </summary>
        /// <param name="onUnits">The selection, applied to the unit array.</param>
        /// <param name="onProxy">The same selection applied to a numeric array of THIS array's shape; replayed only when
        ///     the unit route raises, so the error describes this array's shapes.</param>
        /// <returns>The selected string array.</returns>
        /// <exception cref="Exception">Whatever the selection raises for this array's shape (IndexError,
        ///     IndexOutOfRangeException, ValueError, …) — the numeric texts for the same index.</exception>
        private NDArray StringAdvancedGet(Func<NDArray, NDArray> onUnits, Func<NDArray, NDArray> onProxy)
        {
            using var units = UnitArray();
            NDArray picked;
            try
            {
                picked = onUnits(units);
            }
            catch (Exception e) when (IsReplayable(e))
            {
                using (var proxy = np.broadcast_to(NDArray.Scalar((byte)0), new Shape((long[])Shape.dimensions.Clone())))
                    onProxy(proxy)?.Dispose();
                throw; // the proxy accepted the index: the unit route's own error stands
            }
            return FromUnitResult(picked, units);
        }

        /// <summary>
        ///     NumPy's <c>prepare_index</c> rank check for a normalized index tuple: every entry consumes one axis — a boolean
        ///     array as many as its rank, an ellipsis or newaxis none — and more than this array has is an IndexError. The
        ///     string routes run it before they touch the unit view, whose trailing character axis would otherwise accept one
        ///     index too many (and before a value is converted, which NumPy does only after the index passed).
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <exception cref="IndexError"><c>too many indices for array: array is {n}-dimensional, but {k} were indexed</c>.</exception>
        private void ThrowIfTooManyIndices(object[] normalized)
        {
            int consumed = 0;
            foreach (var entry in normalized)
            {
                switch (entry)
                {
                    case Slice s when s.IsEllipsis || s.IsNewAxis:
                        break;
                    case NDArray nd when nd.typecode == NPTypeCode.Boolean:
                        consumed += nd.ndim;
                        break;
                    default:
                        consumed++;
                        break;
                }
            }
            if (consumed > ndim)
                throw new IndexError($"too many indices for array: array is {ndim}-dimensional, but {consumed} were indexed");
        }

        /// <summary>
        ///     An advanced SET on this string array through its unit view. The value is first converted to THIS array's
        ///     dtype the way NumPy's setitem converts it (<c>FORCECAST</c>: truncated / NUL-padded, <c>S ↔ U</c> ASCII-checked;
        ///     numbers are refused — formatting them is the computational phase), then its unit view is assigned to the
        ///     selection of this array's unit view.
        /// </summary>
        /// <param name="value">The value (a string array, any shape broadcastable to the selection).</param>
        /// <param name="onUnits">The assignment, applied to (unit array, unit value).</param>
        /// <param name="onProxy">The same assignment applied to numeric arrays of this array's and the value's shapes;
        ///     replayed only when the unit route raises.</param>
        /// <exception cref="ValueError">This array is read-only, or the value does not broadcast to the selection.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> value is not ASCII and this array is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> value is not ASCII and this array is <c>U</c>.</exception>
        /// <exception cref="StringComputationNotSupportedException">The value is numeric.</exception>
        private void StringAdvancedSet(NDArray value, Action<NDArray, NDArray> onUnits, Action<NDArray, NDArray> onProxy)
        {
            ThrowIfNotWriteable();
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            if (!value.Storage.IsString)
                throw new StringComputationNotSupportedException(
                    $"Cannot assign {value.dtype.ToString(true)} values to a {dtype.ToString(true)} array: converting numbers to text " +
                    "is not part of NumSharp's fixed-width string storage phase yet.");

            // setitem converts the value to the destination's dtype first (NumPy: PyArray_FromAny(value, descr, FORCECAST)).
            bool same = value.dtype.Meta == dtype.Meta && value.dtype.itemsize == dtype.itemsize;
            var converted = same ? value : value.astype(dtype, true, 'K', "unsafe");
            try
            {
                using var units = UnitArray();
                using var valueUnits = new NDArray(converted.Storage.StringUnitView()) { TensorEngine = TensorEngine };
                try
                {
                    onUnits(units, valueUnits);
                }
                catch (Exception e) when (IsReplayable(e))
                {
                    using (var proxy = np.zeros(new Shape((long[])Shape.dimensions.Clone()), NPTypeCode.Byte))
                    using (var valueProxy = np.zeros(new Shape((long[])converted.Shape.dimensions.Clone()), NPTypeCode.Byte))
                        onProxy(proxy, valueProxy);
                    throw;
                }
            }
            finally
            {
                if (!ReferenceEquals(converted, value))
                    converted.Dispose();
            }
        }

        /// <summary>
        ///     Whether an exception from the unit route is an INDEXING error worth replaying on the proxy (its text may name
        ///     the unit shapes) — every error but the conversions', which the unit route never raises.
        /// </summary>
        /// <param name="e">The exception.</param>
        /// <returns>True to replay the index on a numeric proxy of the string array's shape.</returns>
        private static bool IsReplayable(Exception e)
            => !(e is UnicodeError || e is SystemError || e is NotSupportedException || e is OutOfMemoryException);

        /// <summary>
        ///     Turns a unit-route result back into a string array: a VIEW of the unit view becomes the matching view of this
        ///     array (strides and offset divided by the characters per element); a fresh gather is re-tagged in place
        ///     (<see cref="UnmanagedStorage.StringsOverUnits"/>), keeping NumPy's ownership (a fancy result owns its data).
        /// </summary>
        /// <param name="picked">The unit result (disposed here).</param>
        /// <param name="units">The unit view it was selected from.</param>
        /// <returns>The string result.</returns>
        private NDArray FromUnitResult(NDArray picked, NDArray units)
        {
            try
            {
                var unitShape = picked.Shape;
                int nd = unitShape.NDim - 1;
                long chars = Storage.StringSlice.Chars;

                if (ReferenceEquals(picked.Storage.InternalArray, units.Storage.InternalArray))
                {
                    // A view of this array's memory: the same element grid, in element units.
                    var dims = new long[nd];
                    var strides = new long[nd];
                    for (int d = 0; d < nd; d++)
                    {
                        dims[d] = unitShape.dimensions[d];
                        strides[d] = unitShape.strides[d] / chars;
                    }
                    var shape = new Shape(dims, strides, unitShape.offset / chars, unitShape.bufferSize / chars);
                    return new NDArray(Storage.Alias(shape)) { TensorEngine = TensorEngine };
                }

                NDArray dense = picked;
                if (!picked.Shape.IsContiguous)
                    dense = picked.copy();
                try
                {
                    return new NDArray(UnmanagedStorage.StringsOverUnits(dense.Storage, dtype)) { TensorEngine = TensorEngine };
                }
                finally
                {
                    if (!ReferenceEquals(dense, picked))
                        dense.Dispose();
                }
            }
            finally
            {
                picked.Dispose(); // the string result holds its own counted reference on the buffer
            }
        }

        /// <summary>
        ///     <c>this[indices]</c> GET for a string array whose (normalized) index has an advanced component.
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <returns>The selected string array.</returns>
        private NDArray StringFetchIndices(object[] normalized)
        {
            // A lone boolean mask over leading axes gathers whole rows (MaskedUnitRows); any other index runs on the unit
            // view as given.
            if (normalized.Length == 1 && normalized[0] is NDArray m && m.typecode == NPTypeCode.Boolean)
                return StringAdvancedGet(u => MaskedUnitRows(u, m, units => units.FetchIndices(WithCharacterAxis(normalized))),
                                         p => p.FetchIndices((object[])normalized.Clone()));
            return StringAdvancedGet(u => u.FetchIndices(WithCharacterAxis(normalized)), p => p.FetchIndices((object[])normalized.Clone()));
        }

        /// <summary>
        ///     <c>this[indices] = value</c> for a string array whose (normalized) index has an advanced component.
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <param name="value">The value.</param>
        private void StringSetIndices(object[] normalized, NDArray value)
            => StringAdvancedSet(value, (u, v) => u.SetIndices(WithCharacterAxis(normalized), v), (p, v) => p.SetIndices((object[])normalized.Clone(), v));

        /// <summary>
        ///     <c>this[index arrays]</c> GET for a string array. More index arrays than axes would reach the character axis of
        ///     the unit view, so that error is raised by the proxy (this array's shape) without touching the unit route.
        /// </summary>
        /// <param name="selection">The index arrays, one per leading axis.</param>
        /// <returns>The gathered string array.</returns>
        private NDArray StringFetchIndices(NDArray[] selection)
        {
            if (selection.Length > ndim)
            {
                using (var proxy = np.broadcast_to(NDArray.Scalar((byte)0), new Shape((long[])Shape.dimensions.Clone())))
                    FetchIndices(proxy, selection, null, true)?.Dispose();
                throw new IndexError($"too many indices for array: array is {ndim}-dimensional, but {selection.Length} were indexed");
            }
            return StringAdvancedGet(u => FetchIndices(u, selection, null, true), p => FetchIndices(p, selection, null, true));
        }

        /// <summary>
        ///     <c>this[index arrays] = value</c> for a string array (see <see cref="StringFetchIndices(NDArray[])"/>).
        /// </summary>
        /// <param name="selection">The index arrays, one per leading axis.</param>
        /// <param name="value">The value.</param>
        private void StringSetIndices(NDArray[] selection, NDArray value)
        {
            if (selection.Length > ndim)
            {
                using (var proxy = np.zeros(new Shape((long[])Shape.dimensions.Clone()), NPTypeCode.Byte))
                    SetIndices(proxy, selection, NDArray.Scalar((byte)0));
                throw new IndexError($"too many indices for array: array is {ndim}-dimensional, but {selection.Length} were indexed");
            }
            StringAdvancedSet(value, (u, v) => SetIndices(u, selection, v), (p, v) => SetIndices(p, selection, v));
        }
    }
}
