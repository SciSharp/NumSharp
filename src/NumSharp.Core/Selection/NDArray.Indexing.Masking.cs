using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Used to perform selection based on a boolean mask.
        /// </summary>
        /// <param name="mask">The boolean mask: of this array's leading dimensions (<c>mask.shape ==
        ///     a.shape[:mask.ndim]</c>, a zero-length mask axis matching any length), or 0-d.</param>
        /// <remarks>https://numpy.org/doc/stable/user/basics.indexing.html</remarks>
        /// <exception cref="IndexError">The mask has more dimensions than this array (<c>too many indices for array: array
        ///     is 1-dimensional, but 2 were indexed</c>), or a mask axis of non-zero length differs from the array axis it
        ///     indexes (<c>boolean index did not match indexed array along axis 0; size of axis is 4 but size of
        ///     corresponding boolean axis is 2</c>) — NumPy's <c>prepare_index</c> texts (<see cref="BooleanIndexError"/>).
        ///     </exception>
        [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
        public NDArray this[NDArray<bool> mask]
        {
            get
            {
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
                    if (typecode == NPTypeCode.String)
                        return StringAdvancedGet(u => MaskedUnitRows(u, mask, units => units[mask]), p => p[mask]);
                    return this.TensorEngine.BooleanMask(this, mask);
                }

                // The mask does not match a leading prefix of the array shape: NumPy's prepare_index error.
                throw BooleanIndexError(mask);
            }
            set
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
