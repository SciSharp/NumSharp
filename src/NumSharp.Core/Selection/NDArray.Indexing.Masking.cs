using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     <c>a[mask]</c> (GET) for a lone boolean mask: the selected elements, or the selected sub-arrays of a mask over
        ///     this array's leading axes, in C order — the body of the old <c>this[NDArray&lt;bool&gt;]</c> indexer, which the
        ///     indexer (<see cref="this[object[]]"/>) dispatches to.
        /// </summary>
        /// <param name="mask">The boolean mask: of this array's leading dimensions (<c>mask.shape ==
        ///     a.shape[:mask.ndim]</c>, a zero-length mask axis matching any length), or 0-d.</param>
        /// <returns>A new array of shape <c>(count_true,) + a.shape[mask.ndim:]</c> (a 0-d mask gives <c>(1,)</c> or
        ///     <c>(0,)</c> + <c>a.shape</c>).</returns>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html</remarks>
        /// <exception cref="IndexError">The mask has more dimensions than this array (<c>too many indices for array: array
        ///     is 1-dimensional, but 2 were indexed</c>), or a mask axis of non-zero length differs from the array axis it
        ///     indexes (<c>boolean index did not match indexed array along axis 0; size of axis is 4 but size of
        ///     corresponding boolean axis is 2</c>) — NumPy's <c>prepare_index</c> texts (<see cref="BooleanIndexError"/>).
        ///     </exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        private protected NDArray GetMasked(NDArray<bool> mask)
        {
            // A 0-d mask is NumPy's HAS_0D_BOOL (HAS_BOOL on a 0-d array): MapIter's copy keeps this array's memory order,
            // which the C-order gather below did not (an F-ordered or broadcast source gave a C-ordered result).
            if (mask.ndim == 0)
                return ZeroDBoolSelection((bool)mask);

            // NumPy boolean indexing (numpy/_core/src/multiarray/mapping.c):
            // the mask matches a leading PREFIX of arr's shape — mask.ndim <= arr.ndim
            // and mask.shape == arr.shape[:mask.ndim]. For each True (in C-order) the
            // trailing sub-tensor arr.shape[mask.ndim:] is gathered. Result:
            // (count_true,) + arr.shape[mask.ndim:]. This single prefix-match case covers
            // them all through the unified NDIter gather (DefaultEngine.BooleanMask):
            //   • full element mask (mask.ndim == arr.ndim)         -> 1-D result;
            //   • axis-0 row mask / partial mask (mask.ndim < ndim) -> (count,)+trailing;
            //   • 0-D mask (arr[True]/arr[False], mask.ndim == 0)    -> (1,)/(0,)+arr.shape.
            // A 1-D length-1 mask is a NORMAL mask (not a 0-D scalar), so it is NOT
            // special-cased — NumSharp represents true 0-D arrays as ndim 0.
            if (mask.ndim <= this.ndim && IsPartialShapeMatch(mask))
            {
                // A string element has no element kernel: gather on the unit view (the mask covers the same leading
                // axes there; the trailing character axis rides along).
                // A StringDType element is a cell tied to its allocator: the mask runs on the cell view and the picked
                // cells are duplicated (NDVStringMove.Gather).
                if (Storage.IsVString)
                    return Backends.Strings.NDVStringMove.Gather(this, cells => cells[mask]);
                if (typecode == NPTypeCode.String)
                    return TryStringMaskGather(mask, out var masked)
                        ? masked
                        : StringAdvancedGet(u => MaskedUnitRows(u, mask, units => units[mask]), p => p[mask]);
                return KeepLaneSharingClass(this.TensorEngine.BooleanMask(this, mask));
            }

            // The mask does not match a leading prefix of the array shape: NumPy's prepare_index error.
            throw BooleanIndexError(mask);
        }

        /// <summary>
        ///     <c>a[True]</c> / <c>a[False]</c> (GET) for a lone Python bool or 0-d bool array: NumPy's <c>HAS_0D_BOOL</c> — an
        ///     ADVANCED index that adds one leading axis of length 1 (True) or 0 (False) and consumes none — and, on a 0-d
        ///     array, <c>HAS_BOOL</c>, whose boolean subscript gives the same <c>(1,)</c> / <c>(0,)</c> answer.
        /// </summary>
        /// <param name="selected">The bool.</param>
        /// <returns>A new array of shape <c>(1,) + a.shape</c> holding a copy of this array, laid out in this array's memory
        ///     order (MapIter's subspace keeps the source's strides: an F-ordered source gives an F-ordered result, a broadcast
        ///     one keeps its stride-0 axis innermost); or a new empty array of shape <c>(0,) + a.shape</c>. Never a view.</returns>
        /// <remarks>
        ///     The KEEPORDER copy of the new-axis view reproduces MapIter's layout. Two routes used to answer instead: the mask
        ///     route (a boolean gather in C order, for a 0-d bool ARRAY) and a fixed-width string array's unit route (for a C#
        ///     bool too), both C-ordered whatever the source's order. The new-axis view is built from the storage
        ///     (<see cref="NumSharp.Backends.UnmanagedStorage.GetView(Slice[])"/>), not through <c>np.expand_dims</c>: a public function would see
        ///     an <see cref="NDMatrix"/> object indexed through an <see cref="NDArray"/> reference as a matrix operand
        ///     (<see cref="NDMatrixBoundary"/>). The result keeps this array's lane-sharing C-type class
        ///     (<see cref="KeepLaneSharingClass"/>).
        /// </remarks>
        private protected NDArray ZeroDBoolSelection(bool selected)
        {
            // The new-axis view is this call's own: only the copy (or the empty array) leaves.
            using var expanded = new NDArray(Storage.GetView(new[] { Slice.NewAxis }));
            if (selected)
                return KeepLaneSharingClass(expanded.copy('K'));
            var emptyDims = (long[])expanded.Shape.dimensions.Clone();
            emptyDims[0] = 0;
            return KeepLaneSharingClass(np.empty(Shape.Adopt(emptyDims), dtype));
        }

        /// <summary>
        ///     A new 0-d boolean mask holding <paramref name="selected"/>: the 0-d bool array NumPy's <c>prepare_index</c> makes
        ///     of a Python bool (<c>PyArray_FROM_O(True)</c>), typed for <see cref="SetMasked"/>.
        /// </summary>
        /// <param name="selected">The bool.</param>
        /// <returns>The mask. The caller owns it and disposes it.</returns>
        /// <remarks>The typed wrapper is an alias (<see cref="MakeGeneric{T}"/>) with its own counted reference, so the untyped
        ///     0-d array is released at once; <see cref="AsGeneric{T}"/> could hand back the very object the <c>using</c>
        ///     releases.</remarks>
        private static NDArray<bool> ZeroDBoolMask(bool selected)
        {
            using var scalar = NDArray.Scalar(selected);
            return scalar.MakeGeneric<bool>();
        }

        /// <summary>
        ///     The index tuple with every C# <see cref="bool"/> entry replaced by the 0-d bool array NumPy's
        ///     <c>prepare_index</c> makes of a Python bool (<c>PyArray_FROM_O(True)</c>), so that a bool beside other entries
        ///     takes the routes written for an <c>np.array(True)</c> entry: <c>HAS_0D_BOOL</c>, an advanced index adding one
        ///     axis of length 1 or 0 that broadcasts with the other advanced entries and joins their block.
        /// </summary>
        /// <param name="entries">The normalized index tuple; not modified.</param>
        /// <returns><paramref name="entries"/> itself when it holds no C# bool; else a new tuple whose bool entries are new
        ///     0-d bool arrays.</returns>
        /// <remarks>
        ///     <para>
        ///     The C# bool was read by routes of its own, which treated it as a basic index: a newaxis for True, an early
        ///     return for False. So <c>a[True, True]</c> and <c>a[True, False]</c> raised a mask IndexError, <c>a[False, 0]</c>
        ///     and <c>a[1:3, False]</c> a NullReferenceException, <c>a[True, [0, 2]]</c> returned the wrong elements,
        ///     <c>a[True, 0]</c> a view where NumPy copies, and a False store skipped NumPy's value check.
        ///     </para>
        ///     <para>
        ///     Lifetime: the arrays made here are left to the caller's <see cref="NDScopedAttribute"/> scope (the
        ///     <c>FetchIndices</c> / <c>SetIndices</c> dispatches), which reclaims them after the index is resolved; the copy
        ///     keeps them out of the caller's own key array, which outlives the scope.
        ///     </para>
        /// </remarks>
        [NDScopedCovered] // callers: the [NDScoped] FetchIndices(object[]) and SetIndices(object[], values) dispatches
        private static object[] BoolEntriesAsArrays(object[] entries)
        {
            object[] converted = null;
            for (int i = 0; i < entries.Length; i++)
            {
                if (entries[i] is bool b)
                {
                    converted ??= (object[])entries.Clone();
                    converted[i] = NDArray.Scalar(b);
                }
            }
            return converted ?? entries;
        }

        /// <summary>
        ///     <c>a[mask] = value</c> (SET) for a lone boolean mask — the body of the old <c>this[NDArray&lt;bool&gt;]</c>
        ///     setter, which the indexer (<see cref="this[object[]]"/>) dispatches to.
        /// </summary>
        /// <param name="mask">The boolean mask (the getter's prefix-match contract).</param>
        /// <param name="value">The value: broadcast to the selection <c>(count_true,) + a.shape[mask.ndim:]</c>; a C#
        ///     literal is packed as NumPy's Python scalar (range-checked) after the mask's shape is validated.</param>
        /// <exception cref="ValueError">The array is read-only (checked first), or a 1-D value of the wrong length through a
        ///     mask of this array's own shape (NumPy's <c>cannot assign 3 input values to the 6 output values where the mask
        ///     is true</c>), or a value that does not broadcast.</exception>
        /// <exception cref="TypeError">A value of two or more dimensions through a mask of this array's own shape.</exception>
        /// <exception cref="IndexError">The mask does not fit this array (<see cref="BooleanIndexError"/>).</exception>
        /// <exception cref="OverflowException">A C# literal outside this array's integer dtype.</exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        private void SetMasked(NDArray<bool> mask, NDArray value)
        {
            NumSharpException.ThrowIfNotWriteable(Shape);
            // NumPy's single boolean subscript of this array's EXACT shape (mapping.c HAS_BOOL ->
            // array_assign_boolean_subscript) takes a 0-d or 1-D value, and a 1-D value of more than one element
            // supplies exactly one value per True — checked before anything is written, with NumPy's texts.
            ThrowIfBooleanSubscriptAssignMismatch(mask, value);

            // Prefix-match (full / axis-0 / partial / 0-D), symmetric with the getter.
            // The unified NDIter scatter streams value into the selected slots; value
            // broadcasts to the selection shape (count,)+arr.shape[mask.ndim:] by NumPy
            // rules (raises on an incompatible value). A 1-D length-1 mask is a normal
            // mask; a 0-D mask (arr[True]=v / arr[False]=v) assigns all / nothing.
            if (mask.ndim <= this.ndim && IsPartialShapeMatch(mask))
            {
                if (Storage.IsVString)
                {
                    // The mask runs on marks; the marked elements receive the strings (NDArray.Indexing.VString.cs).
                    VStringAdvancedSet(value, (marks, index) => marks[mask] = index);
                    return;
                }
                if (typecode == NPTypeCode.String)
                {
                    StringAdvancedSet(value, (u, v) => u[mask] = v, (p, v) => p[mask] = v);
                    return;
                }

                // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar: the HAS_BOOL assignment converts it
                // with PyArray_FromAny into this dtype (PyArray_Pack's range checks: a[mask] = 1000 on int8 raises)
                // after the mask's shape was validated above and before anything is written.
                var packed = PackForStore(value, typecode, pyArrayPackRoute: false);
                try
                {
                    this.TensorEngine.BooleanMaskSet(this, mask, packed);
                }
                finally
                {
                    if (!ReferenceEquals(packed, value))
                        packed.Dispose();
                }

                return;
            }

            // NumPy's assignment runs the same prepare_index as the getter (after the writeability check above).
            throw BooleanIndexError(mask);
        }

        /// <summary>
        ///     The IndexError NumPy's <c>prepare_index</c> (<c>mapping.c</c>) raises for a boolean index array that does
        ///     not fit this array: a mask converts to one integer index per mask axis (its nonzero), so more mask axes than
        ///     array axes is the too-many-indices error, checked first; otherwise the FIRST mask axis whose non-zero length
        ///     differs from the array axis it acts on.
        /// </summary>
        /// <param name="mask">The boolean index array, known not to fit.</param>
        /// <returns>The error to throw (NumPy's texts, verbatim).</returns>
        /// <remarks>
        ///     A mask axis of length 0 matches any array axis: NumPy records the mask axis length and skips the comparison
        ///     when it is 0 (<c>indices[i].value &gt; 0</c>), so <c>a[np.zeros(0, bool)]</c> selects nothing from an axis
        ///     of any length. The axis NumPy names is the array axis — for a single mask, the mask axis itself. This
        ///     replaced an <see cref="IndexOutOfRangeException"/> whose text always named axis 0 and the mask's first
        ///     length, even for a mask with too many dimensions ("size of axis is 2 but size of boolean index is 2").
        /// </remarks>
        internal IndexError BooleanIndexError(NDArray mask)
        {
            if (mask.ndim > this.ndim)
                return new IndexError($"too many indices for array: array is {this.ndim}-dimensional, but {mask.ndim} were indexed");

            var dims = this.Shape.dimensions;
            var maskDims = mask.Shape.dimensions;
            for (int i = 0; i < mask.ndim; i++)
            {
                if (maskDims[i] != 0 && maskDims[i] != dims[i])
                    return new IndexError(
                        $"boolean index did not match indexed array along axis {i}; size of axis is {dims[i]} but size of " +
                        $"corresponding boolean axis is {maskDims[i]}");
            }

            // Unreachable for a mask the caller found not to fit; kept so a future caller cannot throw null.
            return new IndexError("boolean index did not match indexed array");
        }

        /// <summary>
        ///     NumPy's checks for <c>a[mask] = value</c> when <paramref name="mask"/> has exactly this array's shape (a
        ///     partial mask goes through the fancy-index route and its broadcast rule instead): a value of more than one
        ///     dimension is refused, and a 1-D value whose length is not 1 must hold one value per True element.
        /// </summary>
        /// <param name="mask">The boolean mask.</param>
        /// <param name="value">The value being assigned (null is left to the caller's own check).</param>
        /// <exception cref="TypeError"><c>NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input
        ///     has {n} dimensions</c>.</exception>
        /// <exception cref="ValueError"><c>NumPy boolean array indexing assignment cannot assign {k} input values to the {t}
        ///     output values where the mask is true</c>.</exception>
        internal void ThrowIfBooleanSubscriptAssignMismatch(NDArray mask, NDArray value)
        {
            if (value is null || mask.ndim != ndim)
                return;
            var maskDims = mask.Shape.dimensions;
            var dims = Shape.dimensions;
            for (int i = 0; i < dims.Length; i++)
                if (maskDims[i] != dims[i])
                    return;
            if (value.ndim > 1)
                throw new TypeError(
                    $"NumPy boolean array indexing assignment requires a 0 or 1-dimensional input, input has {value.ndim} dimensions");
            if (value.ndim == 1 && value.Shape.dimensions[0] != 1)
            {
                long trues = np.count_nonzero(mask);
                long given = value.Shape.dimensions[0];
                if (trues != given)
                    throw new ValueError(
                        $"NumPy boolean array indexing assignment cannot assign {given} input values to the {trues} output values where the mask is true");
            }
        }

        /// <summary>
        /// Checks if mask.shape matches arr.shape[:mask.ndim] (partial shape match). A mask axis
        /// of length 0 is EXEMPT from the match — a zero-length boolean axis selects nothing and is
        /// valid against an axis of any size (NumPy: A[np.zeros(0,bool)] -> (0,)+trailing on a
        /// size-3 axis, A[np.zeros((3,0),bool)] -> (0,)); only a NON-zero mask axis must equal the
        /// array axis (A[np.zeros((0,2),bool)] still raises on the mismatched size-2 axis).
        /// </summary>
        private bool IsPartialShapeMatch(NDArray mask)
        {
            for (int i = 0; i < mask.ndim; i++)
            {
                if (mask.Shape.dimensions[i] != this.Shape.dimensions[i] && mask.Shape.dimensions[i] != 0)
                    return false;
            }
            return true;
        }
    }
}
