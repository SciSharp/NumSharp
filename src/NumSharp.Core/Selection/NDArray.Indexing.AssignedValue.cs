using System;

namespace NumSharp
{
    public partial class NDArray
    {
        // ───────────────────────────────────────────────────────────────────────────────────────
        //  The value of an advanced-index assignment — NumPy 2.4.2, mapping.c, PyArray_MapIterNew's
        //  `extra_op` block:
        //
        //      if (PyArray_NDIM(extra_op) > mit->nd) {
        //          /* Usual assignments allows removal of leading one dimensions. */
        //          permute.len = mit->nd;
        //          permute.ptr = &PyArray_DIMS(extra_op)[PyArray_NDIM(extra_op) - mit->nd];
        //          tmp_arr = PyArray_Newshape(extra_op, &permute, NPY_CORDER);
        //          if (tmp_arr == NULL) goto broadcast_error;
        //      }
        //      ... the iterator then broadcasts extra_op against mit->dimensions ...
        //    broadcast_error:
        //      "shape mismatch: value array of shape %S could not be broadcast to indexing result of shape %S"
        //
        //  Every advanced route of the setter funnels its value through PrepareAssignedValue, so the rule and
        //  the text are written once. (A boolean mask of the array's own shape is NOT this rule: NumPy runs
        //  array_assign_boolean_subscript there, see DefaultEngine.BooleanMaskSet.)
        // ───────────────────────────────────────────────────────────────────────────────────────

        /// <summary>
        ///     Prepares the value of an advanced-index assignment (<c>a[idx] = values</c>) the way NumPy's
        ///     <c>MapIter</c> does: a value with MORE axes than the indexing result loses its leading axes (they
        ///     must hold one element in total), and the remaining shape must broadcast to the result. Call it
        ///     before any element is written, so an incompatible value leaves the destination untouched.
        /// </summary>
        /// <param name="values">The assigned value. Not modified and not disposed.</param>
        /// <param name="resultShape">The shape the same index gives as a GETTER (<c>a[idx].shape</c>): the
        ///     index arrays' broadcast shape and the lengths of the sliced axes, in NumPy's order. Read only.</param>
        /// <returns>
        ///     <paramref name="values"/> itself when it has no axis to drop; otherwise a reshaped VIEW of it with
        ///     exactly <c>resultShape.Length</c> axes (a copy when the value is not contiguous). The caller still
        ///     broadcasts it: this method only proves that the broadcast is legal.
        /// </returns>
        /// <exception cref="ValueError">The value cannot be broadcast to <paramref name="resultShape"/>. The text
        ///     is NumPy's and names the ORIGINAL shape of <paramref name="values"/>:
        ///     <c>shape mismatch: value array of shape (1,4) could not be broadcast to indexing result of shape (2,)</c>.</exception>
        /// <remarks>
        ///     The leading axes are removed with a reshape, as in NumPy, so the test is on the element count and
        ///     not on each axis: a <c>(1, 1, 4)</c> value fits a <c>(2, 4)</c> result, a <c>(2, 1, 4)</c> value
        ///     does not, and a <c>(5, 0)</c> value fits a <c>(0,)</c> result (both hold no element). A matrix
        ///     value is always two-dimensional, which makes this rule the one that lets
        ///     <c>m[rows, cols] = other_row</c> work.
        ///     <para>
        ///     Lifetime: the reshaped view is a new <see cref="NDArray"/>. Every caller runs inside an
        ///     <see cref="NDScopedAttribute"/> boundary (the <c>SetIndices</c> dispatches, their <c>Try*</c>
        ///     helpers, <c>DefaultEngine.BooleanMaskSet</c>), which reclaims it.
        ///     </para>
        /// </remarks>
        [NDScopedCovered]
        internal static NDArray PrepareAssignedValue(NDArray values, long[] resultShape)
        {
            long[] original = values.Shape.dimensions;
            long[] dims = original;
            int nd = resultShape.Length;
            int extra = dims.Length - nd;

            if (extra > 0)
            {
                // PyArray_Newshape(value, its trailing nd dims): legal exactly when the trailing axes already
                // hold every element. NumPy compares sizes, so an empty value passes with any leading axes.
                long trailing = 1;
                for (int i = extra; i < dims.Length; i++)
                    trailing *= dims[i];
                if (trailing != values.size)
                    throw AssignedValueMismatch(original, resultShape);

                var kept = new long[nd];
                System.Array.Copy(dims, extra, kept, 0, nd);
                values = values.reshape(Shape.Adopt(kept));
                dims = kept;
            }

            // The iterator's broadcast, right-aligned: an axis fits when it has length 1 or the result's length.
            for (int i = dims.Length - 1, j = nd - 1; i >= 0; i--, j--)
            {
                if (dims[i] != 1 && dims[i] != resultShape[j])
                    throw AssignedValueMismatch(original, resultShape);
            }

            return values;
        }

        /// <summary>Builds NumPy's <c>broadcast_error</c> of an advanced-index assignment.</summary>
        /// <param name="valueShape">The value's shape as the caller passed it (before any axis was dropped).</param>
        /// <param name="resultShape">The indexing result's shape.</param>
        /// <returns>The exception to throw; this method does not throw it.</returns>
        private static ValueError AssignedValueMismatch(long[] valueShape, long[] resultShape)
            => new ValueError($"shape mismatch: value array of shape {PyShapeTuple(valueShape)} " +
                              $"could not be broadcast to indexing result of shape {PyShapeTuple(resultShape)}");

        /// <summary>
        ///     Formats a shape as NumPy's <c>convert_shape_to_string</c> does in error texts: no space after
        ///     the commas, and a one-axis shape closes with a comma (<c>(3,)</c>, <c>(2,4)</c>, <c>()</c>).
        /// </summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The shape text.</returns>
        private static string PyShapeTuple(long[] dims)
            => dims.Length == 1 ? $"({dims[0]},)" : "(" + string.Join(",", dims) + ")";
    }
}
