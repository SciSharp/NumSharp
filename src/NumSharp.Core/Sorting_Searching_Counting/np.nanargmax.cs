using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the index of the maximum value in the flattened array, ignoring NaNs.
        ///     For an all-NaN array a <see cref="ValueError"/> is raised. Warning: the result
        ///     cannot be trusted if the array contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>Index of the maximum value in the flattened array (NumPy intp = int64).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html</remarks>
        public static long nanargmax(NDArray a)
        {
            var prepared = NanArgReplace(a, double.NegativeInfinity, axis: null);
            return (long)prepared.TensorEngine.ArgMax(prepared);
        }

        /// <summary>
        ///     Return the indices of the maximum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and -Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html</remarks>
        public static NDArray nanargmax(NDArray a, int? axis, bool keepdims = false)
        {
            ValidateNanArgAxis(a, axis);
            var prepared = NanArgReplace(a, double.NegativeInfinity, axis);
            return prepared.TensorEngine.ReduceArgMax(prepared, axis, keepdims);
        }

        /// <summary>
        ///     Return the index of the minimum value in the flattened array, ignoring NaNs.
        ///     For an all-NaN array a <see cref="ValueError"/> is raised. Warning: the result
        ///     cannot be trusted if the array contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <returns>Index of the minimum value in the flattened array (NumPy intp = int64).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html</remarks>
        public static long nanargmin(NDArray a)
        {
            var prepared = NanArgReplace(a, double.PositiveInfinity, axis: null);
            return (long)prepared.TensorEngine.ArgMin(prepared);
        }

        /// <summary>
        ///     Return the indices of the minimum values in the specified axis, ignoring NaNs.
        ///     For all-NaN slices a <see cref="ValueError"/> is raised. Warning: the results
        ///     cannot be trusted if a slice contains only NaNs and Infs (NumPy caveat).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">Axis along which to operate. If <c>null</c>, the flattened input is used.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one
        ///     (with <paramref name="axis"/> <c>null</c> the result has shape <c>(1,) * a.ndim</c>, like NumPy).</param>
        /// <returns>Array of int64 indices (or a 0-d scalar for the flattened form).</returns>
        /// <exception cref="ValueError">All-NaN slice encountered.</exception>
        /// <exception cref="AxisError">Axis out of bounds (reports the original axis, like NumPy).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html</remarks>
        public static NDArray nanargmin(NDArray a, int? axis, bool keepdims = false)
        {
            ValidateNanArgAxis(a, axis);
            var prepared = NanArgReplace(a, double.PositiveInfinity, axis);
            return prepared.TensorEngine.ReduceArgMin(prepared, axis, keepdims);
        }

        /// <summary>
        ///     Validate the reduction axis against NumPy's rule, up front and for every dtype path:
        ///     allowed iff <c>-max(ndim,1) &lt;= axis &lt; max(ndim,1)</c> — a 0-d array accepts axis 0 and -1
        ///     (NumPy's legacy reduction quirk, probed against 2.4.2). Reports the ORIGINAL axis.
        ///     Doing it here (rather than trusting the delegates) keeps the integer passthrough path from
        ///     inheriting the engine's silent negative-axis wrap-around.
        /// </summary>
        private static void ValidateNanArgAxis(NDArray a, int? axis)
        {
            if (axis is not int ax)
                return;
            int bound = Math.Max(a.ndim, 1);
            if (ax < -bound || ax >= bound)
                throw new AxisError(ax, a.ndim);
        }

        /// <summary>
        ///     Port of NumPy's <c>_replace_nan</c> + the all-NaN slice guard from
        ///     <c>numpy/lib/_nanfunctions_impl.py</c> (nanargmin/nanargmax): for inexact dtypes
        ///     (Half/Single/Double/Complex) replace every NaN with <paramref name="fill"/> in a copy and
        ///     raise <see cref="ValueError"/> when any slice along <paramref name="axis"/> is entirely NaN.
        ///     Every other dtype (NumPy's <c>mask=None</c> branch — integral/bool, and NumSharp's
        ///     Char/Decimal which carry no NaN) passes through untouched.
        /// </summary>
        private static NDArray NanArgReplace(NDArray a, double fill, int? axis)
        {
            switch (a.typecode)
            {
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                    break;
                default:
                    return a;
            }

            var mask = np.isnan(a);
            // NumPy checks `mask is not None and mask.size` — an empty array skips the all-NaN
            // guard and lets argmax/argmin raise their own empty-sequence error. A NaN-free array
            // needs no replacement copy (argmax only reads, so returning `a` is observably identical).
            if (mask.size == 0 || !np.any(mask))
                return a;

            bool anyAllNanSlice = axis is int ax
                ? np.any(np.all(mask, ax))
                : np.all(mask);
            if (anyAllNanSlice)
                throw new ValueError("All-NaN slice encountered");

            // np.copyto(a_copy, val, where=mask): the double scalar casts into the operand's own
            // dtype (Half/float/double; Complex becomes (fill, 0) — probed identical to NumPy).
            var replaced = a.copy();
            np.copyto(replaced, NDArray.Scalar(fill), where: mask);
            return replaced;
        }
    }
}
