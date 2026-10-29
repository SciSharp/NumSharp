using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
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
        ///     NumPy's <c>prepare_index</c> walk and rank check for a normalized index tuple, in NumPy's order: each entry is
        ///     classified as it is reached, a refused entry raising at once (an index array of a non-index dtype, a scalar that
        ///     is not an integer); after the walk, every entry has consumed its axes — a boolean array as many as its rank, an
        ///     ellipsis, a new axis or a scalar bool none, anything else one — and more than this array has is an IndexError.
        ///     The string routes run it before they touch the unit view, whose trailing character axis would otherwise accept
        ///     one index too many (and before a value is converted, which NumPy does only after the index passed).
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <exception cref="IndexError"><c>arrays used as indices must be of integer (or boolean) type</c> for an index array
        ///     of another dtype, <see cref="InvalidIndexMessage"/> for a scalar that is no integer (a float, a float16, a
        ///     decimal, a complex number), both raised at the entry; then <c>too many indices for array: array is
        ///     {n}-dimensional, but {k} were indexed</c>.</exception>
        /// <remarks>Until 2026-10-29 the rank was checked first, so a 0-d string array indexed by a float array or a double
        ///     reported "too many indices" where NumPy names the refused entry.</remarks>
        private void ThrowIfTooManyIndices(object[] normalized)
        {
            int consumed = 0;
            foreach (var entry in normalized)
            {
                switch (entry)
                {
                    case Slice s when s.IsEllipsis || s.IsNewAxis:
                        break;
                    // A scalar bool is NumPy's HAS_0D_BOOL: a new axis of length 1 / 0 that consumes no axis of this array
                    // (s[True] of a 0-d string array is (1,), not "too many indices").
                    case bool:
                        break;
                    case NDArray nd:
                        // prepare_index refuses an index array of a non-index dtype where it meets it, before any count.
                        ThrowIfNotIndexArrayDtype(nd);
                        consumed += nd.typecode == NPTypeCode.Boolean ? nd.ndim : 1;
                        break;
                    case Slice:
                    case string:
                    // System.Array spelled out: inside NDArray the bare name is the Array property, a constant pattern.
                    case System.Array:
                    case null:
                        consumed++;
                        break;
                    default:
                        // A scalar entry: an integer consumes an axis (on a 0-d array as NumPy's 0-d integer array, which
                        // the count below reports); any other scalar is refused where it stands.
                        if (!TryAsIntegerScalar(entry, out _))
                            throw new IndexError(InvalidIndexMessage);
                        consumed++;
                        break;
                }
            }
            if (consumed > ndim)
                throw new IndexError($"too many indices for array: array is {ndim}-dimensional, but {consumed} were indexed");
        }

        /// <summary>
        ///     An advanced SET on this string array through its unit view. The value is converted to THIS array's dtype the
        ///     way NumPy's setitem converts it (truncated / NUL-padded, <c>S ↔ U</c> ASCII-checked, numbers formatted as their
        ///     <c>str()</c>), then its unit view is assigned to the selection of this array's unit view.
        /// </summary>
        /// <param name="value">The value (a string array, any shape broadcastable to the selection; or numbers; or null —
        ///     Python's <c>None</c>, stored as the text <c>'None'</c>, truncated to the width).</param>
        /// <param name="onUnits">The assignment, applied to (unit array, unit value).</param>
        /// <param name="onProxy">The same assignment applied to numeric arrays of this array's and the value's shapes;
        ///     replayed when the unit route raises, and run FIRST for a value converted while the selection is written.</param>
        /// <exception cref="ValueError">This array is read-only, or the value does not broadcast to the selection.</exception>
        /// <exception cref="UnicodeEncodeError">A <c>U</c> value is not ASCII and this array is <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> value is not ASCII and this array is <c>U</c>.</exception>
        /// <remarks>
        ///     <b>When the value is converted (NumPy's <c>array_assign_subscript</c>).</b> A SCALAR text value — a C# literal,
        ///     NumPy's Python <c>str</c> — is converted before the index is resolved (<c>PyArray_FromAny(value, descr,
        ///     FORCECAST)</c>), so its conversion error wins even over an out-of-range index. An ARRAY value is converted element
        ///     by element as the selection is written (the map iterator's cast, the boolean assignment's transfer function) —
        ///     after the index and broadcast checks, and not at all when the selection is empty (<c>a[[]] = np.array(['é'])</c>
        ///     into an <c>S</c> array succeeds). A number formats without fail (its <c>str()</c>, truncated to the width), so a
        ///     numeric value is converted only when the selection holds an element, after the same checks.
        /// </remarks>
        private void StringAdvancedSet(NDArray value, Action<NDArray, NDArray> onUnits, Action<NDArray, NDArray> onProxy)
        {
            ThrowIfNotWriteable();
            if (value is null)
            {
                // Python's None: STRING_setitem / UNICODE_setitem store any object by its str(), so None is the text 'None'
                // (truncated to the width) at every selected element. It is a scalar, converted before the index is
                // resolved like any text literal, and its conversion cannot fail.
                using var none = Backends.UnmanagedStorage.FromAnyScalar(null, ((StringDTypeMeta)dtype.Meta).singleton);
                StringAdvancedSet(none, onUnits, onProxy);
                return;
            }

            bool text = value.Storage.IsString;
            bool same = text && value.dtype.Meta == dtype.Meta && value.dtype.itemsize == dtype.itemsize;
            if (!same && !(text && value.ndim == 0))
            {
                // Converted while the selection is written: NumPy's index and broadcast checks come first (the proxy raises
                // them for this array's and the value's shapes), and an empty selection converts nothing.
                if (!ProxySelectionWrites(value, onProxy))
                    return;
            }

            // The value in this array's dtype (a scalar's conversion runs before the index is resolved, as in NumPy).
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
        ///     Runs an advanced assignment on numeric PROXIES — a zeroed <c>uint8</c> array of this array's shape and a
        ///     <c>uint8</c> array of ones of the value's shape — and reports whether it wrote any element. The assignment
        ///     raises NumPy's index and broadcast errors exactly as the real one would (the same index, the same shapes),
        ///     before any value is converted.
        /// </summary>
        /// <param name="value">The value whose shape the proxy value takes.</param>
        /// <param name="onProxy">The assignment, applied to (proxy, value proxy).</param>
        /// <returns>True when the selection holds at least one element.</returns>
        /// <exception cref="Exception">The index / broadcast error the assignment raises (IndexError, ValueError, …).</exception>
        /// <remarks>An extra pass over a byte copy of this array's shape, taken only by a value converted while the
        ///     selection is written (another string class or width, or numbers) — never by the same-dtype route.</remarks>
        private bool ProxySelectionWrites(NDArray value, Action<NDArray, NDArray> onProxy)
        {
            using var proxy = np.zeros(new Shape((long[])Shape.dimensions.Clone()), NPTypeCode.Byte);
            using var valueProxy = np.ones(new Shape((long[])value.Shape.dimensions.Clone()), NPTypeCode.Byte);
            onProxy(proxy, valueProxy);
            return np.any(proxy);
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
            // Elements that hold references (the descriptor answers, as NumPy's PyDataType_REFCHK: StringDType) take the
            // cells + duplication route, never the unit route (NDArray.Indexing.VString.cs).
            if (dtype.Meta.HasReferences)
                return VStringFetchIndices(normalized);
            if (normalized.Length == 1 && normalized[0] is NDArray only)
            {
                // A lone boolean mask over leading axes: the direct mask gather, else whole unit-view rows
                // (MaskedUnitRows); a lone integer index array: the direct row gather. Any other index runs on the unit
                // view as given.
                if (only.typecode == NPTypeCode.Boolean)
                    return TryStringMaskGather(only, out var masked)
                        ? masked
                        : StringAdvancedGet(u => MaskedUnitRows(u, only, units => units.FetchIndices(WithCharacterAxis(normalized))),
                                            p => p.FetchIndices((object[])normalized.Clone()));
                if (TryStringGatherRows(only, out var gathered))
                    return gathered;
            }
            return StringAdvancedGet(u => u.FetchIndices(WithCharacterAxis(normalized)), p => p.FetchIndices((object[])normalized.Clone()));
        }

        /// <summary>
        ///     The direct boolean-mask gather of a fixed-width string array: the selected elements (or rows of the trailing
        ///     axes) are counted, allocated and copied by one kernel (<see cref="DirectILKernelGenerator.GetMaskGatherKernel"/>),
        ///     with no unit view and no index array. NumPy's result for a mask over leading axes: <c>(count,) +
        ///     shape[mask.ndim:]</c>, a fresh C-contiguous array that owns its data, in C order of the mask.
        /// </summary>
        /// <param name="mask">A boolean array; served when it is C-contiguous, has at least one axis, and its shape is a
        ///     prefix of this array's shape.</param>
        /// <param name="result">The selection, when served.</param>
        /// <returns>True when served; false for a layout or mask this route does not cover (a non-contiguous source or
        ///     mask, a 0-d mask, a shape mismatch, an empty trailing block, no runtime IL), which the caller's unit route
        ///     takes with its own checks and errors.</returns>
        /// <remarks>Every element of the result is written (one slab per non-zero mask byte, counted from the same bytes),
        ///     so a <c>U</c> result skips its zero fill (<see cref="ForOverwrite"/>).</remarks>
        [NDAnchored] // this array's and the mask's raw addresses go to the kernel: both stay rooted until it returns
        private unsafe bool TryStringMaskGather(NDArray mask, out NDArray result)
        {
            result = null;
            int lead = mask.ndim;
            if (lead == 0 || lead > ndim || mask.typecode != NPTypeCode.Boolean || !Storage.IsFixedString
                || !Shape.IsContiguous || !mask.Shape.IsContiguous || !DirectILKernelGenerator.Enabled)
                return false;
            var dims = Shape.dimensions;
            for (int d = 0; d < lead; d++)
                if (mask.Shape.dimensions[d] != dims[d])
                    return false;
            var slice = Storage.StringSlice;
            long slabBytes = slice.ItemSize;
            for (int d = lead; d < dims.Length; d++)
                slabBytes *= dims[d];
            if (slabBytes == 0)
                return false;
            var kernel = DirectILKernelGenerator.GetMaskGatherKernel(DirectILKernelGenerator.FixedCopyKindFor(slabBytes));
            if (kernel is null)
                return false;

            byte* maskPtr = (byte*)mask.Storage.Address + mask.Shape.offset;
            long count = DirectILKernelGenerator.CountTrueSimdHelper((bool*)maskPtr, mask.size);
            var resultDims = new long[1 + dims.Length - lead];
            resultDims[0] = count;
            for (int d = lead; d < dims.Length; d++)
                resultDims[1 + d - lead] = dims[d];
            var picked = ForOverwrite(dtype, Shape.Adopt(resultDims));
            picked.TensorEngine = TensorEngine;
            kernel(slice.Address + Shape.offset * (long)slice.ItemSize, maskPtr, mask.size, slabBytes, (byte*)picked.Storage.Address);
            result = picked;
            return true;
        }

        /// <summary>
        ///     The direct integer-array gather of a fixed-width string array, <c>a[idx]</c> over the first axis: the selected
        ///     elements (or rows of the trailing axes) are copied by the fancy-index kernel route
        ///     (<see cref="FancyIndexKernelRoute.Gather"/>) straight from the index buffer, with no unit view. NumPy's result:
        ///     <c>idx.shape + shape[1:]</c>, a fresh C-contiguous array that owns its data.
        /// </summary>
        /// <param name="index">The index array; served when it is an int32 / int64 C-contiguous array with at least one
        ///     axis.</param>
        /// <param name="result">The selection, when served.</param>
        /// <returns>True when served; false for an index or layout this route does not cover (another index dtype, a 0-d
        ///     or non-contiguous index, a non-contiguous source, an empty trailing block, no runtime IL), which the caller's
        ///     unit route takes with its own checks and errors.</returns>
        /// <exception cref="IndexError">An index outside <c>[-size, size)</c> of the first axis: NumPy's
        ///     <c>index 5 is out of bounds for axis 0 with size 3</c> (the partial result is released first).</exception>
        [NDAnchored] // this array's and the index array's raw addresses go to the kernel: both stay rooted until it returns
        private unsafe bool TryStringGatherRows(NDArray index, out NDArray result)
        {
            result = null;
            if (ndim == 0 || index.ndim == 0 || !Storage.IsFixedString || !Shape.IsContiguous || !DirectILKernelGenerator.Enabled
                || !FancyIndexKernelRoute.TryGetIndexPointer(index, out void* indexPtr, out bool index32))
                return false;
            var dims = Shape.dimensions;
            var slice = Storage.StringSlice;
            long slabBytes = slice.ItemSize;
            for (int d = 1; d < dims.Length; d++)
                slabBytes *= dims[d];
            if (slabBytes == 0)
                return false;

            var indexDims = index.Shape.dimensions;
            var resultDims = new long[indexDims.Length + dims.Length - 1];
            System.Array.Copy(indexDims, resultDims, indexDims.Length);
            System.Array.Copy(dims, 1, resultDims, indexDims.Length, dims.Length - 1);
            // Every element is written unless an index is out of bounds, and then the result is released, never returned: a
            // U result skips its zero fill.
            var picked = ForOverwrite(dtype, Shape.Adopt(resultDims));
            long fail = FancyIndexKernelRoute.Gather(slice.Address + Shape.offset * (long)slice.ItemSize, indexPtr, index32,
                                                     index.size, dims[0], slabBytes, (byte*)picked.Storage.Address);
            if (fail == -1)
            {
                picked.TensorEngine = TensorEngine;
                result = picked;
                return true;
            }

            picked.Dispose();
            if (fail >= 0)
                throw new IndexError($"index {FancyIndexKernelRoute.ReadIndex(indexPtr, index32, fail)} is out of bounds for axis 0 with size {dims[0]}");
            return false;   // -2: the IL kernels are unavailable on this host
        }

        /// <summary>
        ///     <c>this[indices] = value</c> for a string array whose (normalized) index has an advanced component.
        /// </summary>
        /// <param name="normalized">The normalized index tuple.</param>
        /// <param name="value">The value.</param>
        private void StringSetIndices(object[] normalized, NDArray value)
        {
            // Elements that hold references (the descriptor answers, as NumPy's PyDataType_REFCHK: StringDType) take the
            // marks + duplication route (NDArray.Indexing.VString.cs).
            if (dtype.Meta.HasReferences)
            {
                VStringSetIndices(normalized, value);
                return;
            }
            StringAdvancedSet(value, (u, v) => u.SetIndices(WithCharacterAxis(normalized), v), (p, v) => p.SetIndices((object[])normalized.Clone(), v));
        }

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
                    Backends.UnmanagedStorage.MapIterGet(proxy, selection, null, true)?.Dispose();
                throw new IndexError($"too many indices for array: array is {ndim}-dimensional, but {selection.Length} were indexed");
            }
            if (selection.Length == 1 && TryStringGatherRows(selection[0], out var gathered))
                return gathered;
            return StringAdvancedGet(u => Backends.UnmanagedStorage.MapIterGet(u, selection, null, true), p => Backends.UnmanagedStorage.MapIterGet(p, selection, null, true));
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
                    Backends.UnmanagedStorage.MapIterSet(proxy, selection, NDArray.Scalar((byte)0));
                throw new IndexError($"too many indices for array: array is {ndim}-dimensional, but {selection.Length} were indexed");
            }
            StringAdvancedSet(value, (u, v) => Backends.UnmanagedStorage.MapIterSet(u, selection, v), (p, v) => Backends.UnmanagedStorage.MapIterSet(p, selection, v));
        }
    }
}
