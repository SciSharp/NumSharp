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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int check_and_adjust_axis(NDArray nd, int axis)
        {
            return check_and_adjust_axis(nd.ndim, axis);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int check_and_adjust_axis(int ndims, int axis)
        {
            if (axis >= 0)
                return axis;
            return ndims + axis;
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
                return nd.Clone();

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

            UnmanagedStorage src;
            if (nd.Shape.IsContiguous)
                src = nd.Storage.Alias(nd.Shape.Clone());
            else
                src = nd.Storage.Clone();

            for (i = 0; i < n; i++)
            {
                src.Shape.dimensions[i] = nd.Shape.dimensions[permutation[i]];
                src.Shape.strides[i] = nd.Shape.strides[permutation[i]];
            }

            src.ShapeReference.SetStridesModified(true);

            //Linear copy of all the sliced items.

            var dst = new UnmanagedStorage(ArraySlice.Allocate(src.TypeCode, src.Shape.size, false), new Shape((int[])src.Shape.dimensions.Clone()));
            MultiIterator.Assign(dst, src);

            return new NDArray(dst);
        }
    }
}
