using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test whether any array element evaluates to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>True if any element evaluates to True (non-zero).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static bool any(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.Any(a);
        }

        /// <summary>
        ///     Test whether any array element along a given axis evaluates to True.
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="axis">Axis along which a logical OR reduction is performed.</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static NDArray<bool> any(NDArray nd, int axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            if (nd.TensorEngine is DefaultEngine defaultEngine)
                return defaultEngine.Any(nd, axis, keepdims);

            var result = nd.TensorEngine.Any(nd, axis);
            if (keepdims && nd.ndim > 0)
            {
                axis = DefaultEngine.NormalizeAxis(axis, nd.ndim);
                var dims = (long[])nd.shape.Clone();
                dims[axis] = 1;
                result.Storage.Reshape(new Shape(dims));
            }

            return result;
        }

        /// <summary>
        ///     Test whether any array element along the given axes evaluates to True.
        ///     Multiple axes can be specified by passing an array of ints.
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="axis">Tuple of axes along which a logical OR reduction is performed.
        ///     An empty array returns the input cast to bool (no reduction).</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static NDArray<bool> any(NDArray nd, int[] axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            if (nd.TensorEngine is DefaultEngine defaultEngine)
                return defaultEngine.Any(nd, axis, keepdims);

            if (axis.Length == 0)
                return DefaultEngine.CastToBoolPreservingShape(nd);

            var normalized = axis.Select(a => DefaultEngine.NormalizeAxis(a, nd.ndim)).ToArray();
            Array.Sort(normalized);

            NDArray<bool> result = null;
            NDArray current = nd;
            for (int i = normalized.Length - 1; i >= 0; i--)
            {
                result = any(current, normalized[i], keepdims: true);
                current = result;
            }

            if (!keepdims)
                result = SqueezeAxes(result, normalized);

            return result;
        }

        /// <summary>
        ///     Test whether any array element evaluates to True, optionally keeping reduced dimensions.
        ///     Reduces over all axes (axis = None semantics).
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="keepdims">If True, the result has all dimensions as size 1 (broadcast-compatible
        ///     with the input). Otherwise the result is a 0-d array.</param>
        /// <returns>A new boolean ndarray.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static NDArray<bool> any(NDArray nd, bool keepdims)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            bool scalar = any(nd);
            if (!keepdims || nd.ndim == 0)
                return np.array(scalar).MakeGeneric<bool>();

            var dims = new long[nd.ndim];
            for (int i = 0; i < dims.Length; i++) dims[i] = 1;
            var result = np.array(scalar).MakeGeneric<bool>();
            result.Storage.Reshape(new Shape(dims));
            return result;
        }

        /// <summary>
        ///     Test whether any array element along the given axis evaluates to True, with optional
        ///     out= destination and where= mask. Matches NumPy 2.x:
        ///     <c>any(a, axis=None, out=None, keepdims=False, *, where=True)</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which to reduce. Pass <c>null</c> for axis=None (all axes).</param>
        /// <param name="out">Destination array. Its dtype is preserved. Pass <c>null</c> to allocate fresh.</param>
        /// <param name="keepdims">If True, the reduced axes are left as size-one dimensions.</param>
        /// <param name="where">Boolean (or numeric-treated-as-bool) mask, broadcastable against
        ///     <paramref name="a"/>. Elements where <c>where=False</c> are excluded from the reduction
        ///     and contribute the identity value (False for <c>any</c>). Pass <c>null</c> for no mask.</param>
        /// <returns>The reduced array, or <paramref name="out"/> when supplied.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static NDArray any(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            NDArray<bool> reduced = ReduceAnyWithWhere(a, axis, keepdims, @where);
            return @out is null ? reduced : WriteToOut(reduced, @out);
        }

        /// <summary>
        ///     Tuple-axis variant of the full <c>np.any</c> overload (with <paramref name="out"/> and <paramref name="where"/>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axes along which to reduce.</param>
        /// <param name="out">Destination array. Its dtype is preserved.</param>
        /// <param name="keepdims">If True, reduced axes are left as size-one dimensions.</param>
        /// <param name="where">Boolean mask, broadcastable against <paramref name="a"/>.</param>
        /// <returns>The reduced array, or <paramref name="out"/> when supplied.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.any.html</remarks>
        public static NDArray any(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            NDArray effective = @where is null ? a : ApplyWhereForAny(a, @where);
            NDArray<bool> reduced = any(effective, axis, keepdims);
            return @out is null ? reduced : WriteToOut(reduced, @out);
        }

        private static NDArray<bool> ReduceAnyWithWhere(NDArray a, int? axis, bool keepdims, NDArray @where)
        {
            NDArray effective = @where is null ? a : ApplyWhereForAny(a, @where);

            if (!axis.HasValue)
                return any(effective, keepdims);

            return any(effective, axis.Value, keepdims);
        }
    }
}
