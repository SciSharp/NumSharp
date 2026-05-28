using System;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Test whether all array elements evaluate to True.
        /// </summary>
        /// <param name="a">Input array or object that can be converted to an array.</param>
        /// <returns>True if all elements evaluate to True (non-zero).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html</remarks>
        public static bool all(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            return a.TensorEngine.All(a);
        }

        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.
        /// </summary>
        /// <param name="nd">Input array or object that can be converted to an array.</param>
        /// <param name="axis">Axis along which a logical AND reduction is performed.</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html</remarks>
        public static NDArray<bool> all(NDArray nd, int axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            if (nd.TensorEngine is DefaultEngine defaultEngine)
                return defaultEngine.All(nd, axis, keepdims);

            var result = nd.TensorEngine.All(nd, axis);
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
        ///     Test whether all array elements along the given axes evaluate to True.
        ///     Multiple axes can be specified by passing an array of ints.
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="axis">Tuple of axes along which a logical AND reduction is performed.
        ///     An empty array returns the input cast to bool (no reduction).</param>
        /// <param name="keepdims">If True, the reduced axes are left in the result as dimensions with size one.</param>
        /// <returns>A new boolean ndarray is returned.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html</remarks>
        public static NDArray<bool> all(NDArray nd, int[] axis, bool keepdims = false)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            if (nd.TensorEngine is DefaultEngine defaultEngine)
                return defaultEngine.All(nd, axis, keepdims);

            // Fallback for non-default engines: chain single-axis reductions.
            if (axis.Length == 0)
                return DefaultEngine.CastToBoolPreservingShape(nd);

            var normalized = axis.Select(a => DefaultEngine.NormalizeAxis(a, nd.ndim)).ToArray();
            Array.Sort(normalized);

            NDArray<bool> result = null;
            NDArray current = nd;
            for (int i = normalized.Length - 1; i >= 0; i--)
            {
                result = all(current, normalized[i], keepdims: true);
                current = result;
            }

            if (!keepdims)
                result = SqueezeAxes(result, normalized);

            return result;
        }

        /// <summary>
        ///     Test whether all array elements evaluate to True, optionally keeping reduced dimensions.
        ///     Reduces over all axes (axis = None semantics).
        /// </summary>
        /// <param name="nd">Input array.</param>
        /// <param name="keepdims">If True, the result is broadcast-compatible with the input
        ///     (every dimension becomes size 1). Otherwise the result is a 0-d array.</param>
        /// <returns>A new boolean ndarray.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html</remarks>
        public static NDArray<bool> all(NDArray nd, bool keepdims)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            bool scalar = all(nd);
            if (!keepdims || nd.ndim == 0)
                return np.array(scalar).MakeGeneric<bool>();

            var dims = new long[nd.ndim];
            for (int i = 0; i < dims.Length; i++) dims[i] = 1;
            var result = np.array(scalar).MakeGeneric<bool>();
            result.Storage.Reshape(new Shape(dims));
            return result;
        }

        /// <summary>
        ///     Test whether all array elements along the given axis evaluate to True, with optional
        ///     out= destination and where= mask. Matches NumPy 2.x:
        ///     <c>all(a, axis=None, out=None, keepdims=False, *, where=True)</c>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axis along which to reduce. Pass <c>null</c> for axis=None (all axes).</param>
        /// <param name="out">Destination array. Its dtype is preserved (e.g. an int <paramref name="out"/>
        ///     stores 0/1 instead of bool). Pass <c>null</c> to allocate a fresh boolean array.</param>
        /// <param name="keepdims">If True, the reduced axes are left as dimensions with size one.</param>
        /// <param name="where">Boolean (or numeric-treated-as-bool) mask, broadcastable against
        ///     <paramref name="a"/>. Elements where <c>where=False</c> are excluded from the reduction
        ///     and contribute the identity value (True for <c>all</c>). Pass <c>null</c> for no mask.</param>
        /// <returns>The reduced array, or <paramref name="out"/> when supplied.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html</remarks>
        public static NDArray all(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            NDArray<bool> reduced = ReduceAllWithWhere(a, axis, keepdims, @where);
            return @out is null ? reduced : WriteToOut(reduced, @out);
        }

        /// <summary>
        ///     Tuple-axis variant of the full <c>np.all</c> overload (with <paramref name="out"/> and <paramref name="where"/>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axis">Axes along which to reduce.</param>
        /// <param name="out">Destination array. Its dtype is preserved.</param>
        /// <param name="keepdims">If True, reduced axes are left as size-one dimensions.</param>
        /// <param name="where">Boolean mask, broadcastable against <paramref name="a"/>.</param>
        /// <returns>The reduced array, or <paramref name="out"/> when supplied.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.all.html</remarks>
        public static NDArray all(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray @where = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            NDArray effective = @where is null ? a : ApplyWhereForAll(a, @where);
            NDArray<bool> reduced = all(effective, axis, keepdims);
            return @out is null ? reduced : WriteToOut(reduced, @out);
        }

        // === shared helpers (also used by np.any) ===

        private static NDArray<bool> ReduceAllWithWhere(NDArray a, int? axis, bool keepdims, NDArray @where)
        {
            NDArray effective = @where is null ? a : ApplyWhereForAll(a, @where);

            if (!axis.HasValue)
                return all(effective, keepdims);

            return all(effective, axis.Value, keepdims);
        }

        // Builds the effective bool array under a where= mask for `all`:
        //   identity is True, so elements with mask=False contribute True.
        //   effective[i] = !mask[i] | bool(a[i])
        private static NDArray ApplyWhereForAll(NDArray a, NDArray @where)
        {
            NDArray<bool> maskBool = ToBool(@where);
            NDArray<bool> aBool = ToBool(a);
            // (!maskBool) | aBool — broadcasting is handled by the operators.
            return !maskBool | aBool;
        }

        // For np.any: effective[i] = mask[i] & bool(a[i]) (identity is False).
        internal static NDArray ApplyWhereForAny(NDArray a, NDArray @where)
        {
            NDArray<bool> maskBool = ToBool(@where);
            NDArray<bool> aBool = ToBool(a);
            return maskBool & aBool;
        }

        // Convert any-dtype NDArray to NDArray<bool> using NumPy truthiness (non-zero = True).
        internal static NDArray<bool> ToBool(NDArray nd)
        {
            if (nd.GetTypeCode == NPTypeCode.Boolean)
                return nd.MakeGeneric<bool>();
            return nd != 0;
        }

        // Write a boolean result into the user-supplied `out` array. `out`'s dtype is preserved
        // (NumPy lets out=float receive 0.0/1.0). Shape must match the reduction's natural output.
        internal static NDArray WriteToOut(NDArray<bool> reduced, NDArray @out)
        {
            if (@out is null)
                throw new ArgumentNullException(nameof(@out));
            if (!@out.Shape.Equals(reduced.Shape))
                throw new ArgumentException(
                    $"output shape mismatch: expected {reduced.Shape}, got {@out.Shape}",
                    nameof(@out));

            np.copyto(@out, reduced, casting: "unsafe");
            return @out;
        }

        internal static NDArray<bool> SqueezeAxes(NDArray<bool> arr, int[] reducedAxes)
        {
            var axesSet = new System.Collections.Generic.HashSet<int>(reducedAxes);
            var newDims = new System.Collections.Generic.List<long>(arr.ndim - reducedAxes.Length);
            long[] shape = arr.shape;
            for (int d = 0; d < shape.Length; d++)
            {
                if (!axesSet.Contains(d))
                    newDims.Add(shape[d]);
            }

            Shape newShape = newDims.Count == 0 ? Shape.Scalar : new Shape(newDims.ToArray());
            arr.Storage.Reshape(newShape);
            return arr;
        }
    }
}
