using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(NDArray nd, int axis)
        {
            return check_and_adjust_axis(nd.ndim, axis);
        }

        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(int ndims, int axis)
        {
            int adjusted = axis >= 0 ? axis : ndims + axis;
            if (adjusted < 0 || adjusted >= ndims)
                throw new AxisOutOfRangeException(ndims, axis);
            return adjusted;
        }

        /// <summary>
        ///     Normalizes an axis argument into a tuple of non-negative integer axes.
        ///     This handles shorthands such as ``1`` and converts them to ``(1,)``,
        ///     as well as performing the handling of negative indices covered by
        ///     `normalize_axis_index`.
        ///     By default, this forbids axes from being specified multiple times.
        ///         Used internally by multi-axis-checking logic.
        /// </summary>
        /// <param name="axis">The un-normalized index or indices of the axis.</param>
        /// <param name="ndim">The number of dimensions of the array that `axis` should be normalized against.</param>
        /// <param name="argname">A prefix to put before the error message, typically the name of the argument.</param>
        /// <param name="allow_duplicate">If False, the default, disallow an axis from being specified twice.</param>
        /// <returns>The normalized axis index, such that `0 <= normalized_axis < ndim`</returns>
        public static int[] normalize_axis_tuple(int[] axis, object argname = null, bool allow_duplicate = false)
        {
            for (int i = 0; i < axis.Length; i++)
            {
                axis[i] = check_and_adjust_axis(axis.Length, axis[i]);
            }

            return allow_duplicate ? (int[])axis.Clone() : axis.Distinct().ToArray();
        }
        /// <summary>
        ///     Normalizes an axis argument into a tuple of non-negative integer axes.
        ///     This handles shorthands such as ``1`` and converts them to ``(1,)``,
        ///     as well as performing the handling of negative indices covered by
        ///     `normalize_axis_index`.
        ///     By default, this forbids axes from being specified multiple times.
        ///         Used internally by multi-axis-checking logic.
        /// </summary>
        /// <param name="axis">The un-normalized index or indices of the axis.</param>
        /// <param name="ndim">The number of dimensions of the array that `axis` should be normalized against.</param>
        /// <param name="argname">A prefix to put before the error message, typically the name of the argument.</param>
        /// <param name="allow_duplicate">If False, the default, disallow an axis from being specified twice.</param>
        /// <returns>The normalized axis index, such that `0 <= normalized_axis < ndim`</returns>
        public static int[] normalize_axis_tuple(int axis, object argname = null, bool allow_duplicate = false)
        {
            return normalize_axis_tuple(new int[] {axis}, argname, allow_duplicate);
        }

        public override NDArray MoveAxis(in NDArray nd, int[] source, int[] destinition)
        {
            source = normalize_axis_tuple(source);
            destinition = normalize_axis_tuple(destinition);
            if (source.Length != destinition.Length)
                throw new Exception("`source` and `destination` arguments must have the same number of elements'");

            var order = Enumerable.Range(0, nd.ndim).Except(source).ToList();

            foreach (var (dest, src) in destinition.Zip(source, (i, i1) => (dest: i, src: i1)).OrderBy(tuple => tuple.dest).ThenBy(tuple => tuple.src))
                order.Insert(dest, src);

            return Transpose(nd, order.ToArray());
        }

        public override NDArray SwapAxes(in NDArray nd, int axis1, int axis2)
        {
            var ndims = nd.ndim;
            var dims = new int[ndims];
            for (int i = 0; i < ndims; i++)
                dims[i] = i;

            axis1 = check_and_adjust_axis(nd, axis1);
            axis2 = check_and_adjust_axis(nd, axis2);

            dims[axis1] = axis2;
            dims[axis2] = axis1;

            return Transpose(nd, dims);
        }

        public override NDArray RollAxis(in NDArray nd, int axis, int start = 0)
        {
            axis = check_and_adjust_axis(nd, axis);
            int n = nd.ndim;
            if (start < 0)
                start += n;

            if (start >= n + 1 || start < 0)
                throw new Exception($"start arg requires start <= n + 1 but start={start} and n={n}");

            if (axis < start)
                start -= 1;

            if (axis == start)
                return nd;  // NumPy returns the array itself for identity case

            var premutes = new List<int>(n);
            for (int i = 0; i < n; i++) 
                premutes.Add(i);

            premutes.RemoveAt(axis);
            premutes.Insert(start, axis);
            return Transpose(nd, premutes.ToArray());
        }

        public override NDArray Transpose(in NDArray nd, int[] premute = null)
        {
            int i, n;
            var permutation = new int[nd.ndim];
            var reverse_permutation = new int[nd.ndim];
            if (premute == null || premute.Length == 0)
            {
                n = nd.ndim;
                for (i = 0; i < n; i++)
                {
                    permutation[i] = n - 1 - i;
                }
            }
            else
            {
                n = premute.Length;
                int[] axes = premute;
                if (n != nd.ndim)
                    throw new Exception("axes don't match array");

                for (i = 0; i < n; i++)
                    reverse_permutation[i] = -1;

                for (i = 0; i < n; i++)
                {
                    int axis = check_and_adjust_axis(nd, axes[i]);
                    if (reverse_permutation[axis] != -1)
                        throw new Exception("repeated axis in transpose");

                    reverse_permutation[axis] = i;
                    permutation[i] = axis;
                }
            }

            // Handle empty arrays: just create a new array with permuted dimensions, no data copy needed
            if (nd.Shape.size == 0)
            {
                var emptyDims = new int[n];
                for (i = 0; i < n; i++)
                    emptyDims[i] = nd.Shape.dimensions[permutation[i]];
                return new NDArray(nd.dtype, emptyDims);
            }

            // NumPy-aligned: Transpose returns a VIEW by permuting strides.
            // For contiguous arrays, this is a simple stride permutation.
            // For non-contiguous arrays (sliced, already transposed), we need to
            // permute the CURRENT strides, which already encode the view's layout.
            //
            // No data copy is needed - transpose is always O(1).
            // The transposed shape shares memory with the original.
            var shape = nd.Shape;
            var srcDims = shape.dimensions;
            var srcStrides = shape.strides;

            // Permute dimensions and strides
            var permutedDims = new int[n];
            var permutedStrides = new int[n];
            for (i = 0; i < n; i++)
            {
                permutedDims[i] = srcDims[permutation[i]];
                permutedStrides[i] = srcStrides[permutation[i]];
            }

            // Create the transposed shape via constructor (immutable)
            // IsContiguous is computed from strides and will be false for transposed arrays
            int bufSize = shape.bufferSize > 0 ? shape.bufferSize : shape.size;
            var newShape = new Shape(permutedDims, permutedStrides, shape.offset, bufSize);

            // Return an alias (view) with the permuted shape
            return new NDArray(nd.Storage.Alias(newShape));
        }
    }
}
