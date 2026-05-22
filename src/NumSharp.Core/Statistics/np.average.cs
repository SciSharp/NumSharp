using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the weighted average along the specified axis.
        ///     Equivalent to <c>sum(a * weights) / sum(weights)</c>. When <paramref name="weights"/>
        ///     is null this reduces to <see cref="mean(NDArray)"/> over the same axes.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static NDArray average(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return AverageCore(a, axisArr, weights, keepdims, returned: false).avg;
        }

        /// <summary>
        ///     Compute the weighted average along a tuple of axes. Equivalent to
        ///     <c>np.average(a, axis, weights, keepdims)</c> in NumPy with a tuple axis.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static NDArray average(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return AverageCore(a, axis, weights, keepdims, returned: false).avg;
        }

        /// <summary>
        ///     Compute the weighted average and return a tuple <c>(avg, sum_of_weights)</c>.
        ///     Equivalent to <c>numpy.average(..., returned=True)</c>. When <paramref name="weights"/>
        ///     is null, <c>sum_of_weights</c> is the number of elements per output cell
        ///     (broadcast to the average's shape).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int? axis = null, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return AverageCore(a, axisArr, weights, keepdims, returned: true);
        }

        /// <summary>
        ///     Tuple-axis overload of <see cref="average_returned(NDArray, int?, NDArray, bool)"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.average.html</remarks>
        public static (NDArray avg, NDArray sumOfWeights) average_returned(NDArray a, int[] axis, NDArray weights = null, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return AverageCore(a, axis, weights, keepdims, returned: true);
        }

        private static (NDArray avg, NDArray sumOfWeights) AverageCore(
            NDArray a, int[] axis, NDArray weights, bool keepdims, bool returned)
        {
            int ndim = a.ndim;
            int[] normalizedAxis = NormalizeAxisTuple(axis, ndim);

            if (weights is null)
            {
                NDArray avg = MeanWithAxes(a, normalizedAxis, keepdims);

                if (!returned) return (avg, null);

                double count = avg.size == 0 ? 0.0 : (double)a.size / avg.size;
                NDArray scl = NDArray.Scalar(count).astype(avg.typecode);
                if (!scl.shape.SequenceEqual(avg.shape))
                    scl = np.broadcast_to(scl, avg).copy();
                return (avg, scl);
            }

            NDArray wgt = WeightsAreValid(weights, a, normalizedAxis);

            NPTypeCode resultDtype = ComputeResultDtype(a.typecode, wgt.typecode);

            NDArray wgtCast = wgt.typecode == resultDtype ? wgt : wgt.astype(resultDtype);
            NDArray aCast = a.typecode == resultDtype ? a : a.astype(resultDtype);

            NDArray scl_ = SumWithAxes(wgtCast, normalizedAxis, resultDtype, keepdims);

            if (HasZero(scl_))
                throw new DivideByZeroException("Weights sum to zero, can't be normalized");

            NDArray prod = aCast * wgtCast;
            NDArray num = SumWithAxes(prod, normalizedAxis, resultDtype, keepdims);
            NDArray avg_ = num / scl_;

            if (!returned) return (avg_, null);

            if (!scl_.shape.SequenceEqual(avg_.shape))
                scl_ = np.broadcast_to(scl_, avg_).copy();
            return (avg_, scl_);
        }

        private static int[] NormalizeAxisTuple(int[] axis, int ndim)
        {
            if (axis is null) return null;

            int[] normalized = new int[axis.Length];
            for (int i = 0; i < axis.Length; i++)
            {
                int ax = axis[i];
                if (ax < 0) ax += ndim;
                if (ax < 0 || ax >= ndim)
                    throw new ArgumentOutOfRangeException(nameof(axis),
                        $"axis {axis[i]} is out of bounds for array of dimension {ndim}");
                normalized[i] = ax;
            }

            // Duplicate detection
            int[] sorted = (int[])normalized.Clone();
            Array.Sort(sorted);
            for (int i = 1; i < sorted.Length; i++)
                if (sorted[i] == sorted[i - 1])
                    throw new ArgumentException("duplicate value in 'axis'");

            return normalized;
        }

        private static NDArray MeanWithAxes(NDArray a, int[] axes, bool keepdims)
        {
            if (axes is null) return np.mean(a, keepdims);
            if (axes.Length == 1) return np.mean(a, axes[0], keepdims);

            int[] sortedDesc = (int[])axes.Clone();
            Array.Sort(sortedDesc);
            Array.Reverse(sortedDesc);

            NDArray cur = a;
            foreach (int ax in sortedDesc)
                cur = np.mean(cur, ax, keepdims: true);

            if (!keepdims)
            {
                int ndim = a.ndim;
                var kept = new List<long>(ndim - axes.Length);
                for (int i = 0; i < ndim; i++)
                {
                    if (Array.IndexOf(axes, i) < 0)
                        kept.Add(a.shape[i]);
                }
                cur = cur.reshape(kept.ToArray());
            }

            return cur;
        }

        private static NDArray SumWithAxes(NDArray a, int[] axes, NPTypeCode dtype, bool keepdims)
        {
            if (axes is null)
                return np.sum(a, axis: null, keepdims: keepdims, typeCode: dtype);

            if (axes.Length == 0)
            {
                NDArray asd = a.typecode == dtype ? a : a.astype(dtype);
                return asd;
            }

            if (axes.Length == 1)
                return np.sum(a, axis: axes[0], keepdims: keepdims, typeCode: dtype);

            int[] sortedDesc = (int[])axes.Clone();
            Array.Sort(sortedDesc);
            Array.Reverse(sortedDesc);

            NDArray cur = a.typecode == dtype ? a : a.astype(dtype);
            foreach (int ax in sortedDesc)
                cur = np.sum(cur, axis: ax, keepdims: true, typeCode: dtype);

            if (!keepdims)
            {
                int ndim = a.ndim;
                var kept = new List<long>(ndim - axes.Length);
                for (int i = 0; i < ndim; i++)
                {
                    if (Array.IndexOf(axes, i) < 0)
                        kept.Add(a.shape[i]);
                }
                cur = cur.reshape(kept.ToArray());
            }

            return cur;
        }

        // Mirrors numpy's lib/_function_base_impl._weights_are_valid.
        private static NDArray WeightsAreValid(NDArray weights, NDArray a, int[] axis)
        {
            NDArray wgt = weights;
            if (a.shape.SequenceEqual(wgt.shape))
                return wgt;

            if (axis is null)
                throw new ArgumentException(
                    "Axis must be specified when shapes of a and weights differ.");

            long[] expected = new long[axis.Length];
            for (int i = 0; i < axis.Length; i++)
                expected[i] = a.shape[axis[i]];

            if (!wgt.shape.SequenceEqual(expected))
                throw new ArgumentException(
                    "Shape of weights must be consistent with shape of a along specified axis.");

            // wgt = wgt.transpose(np.argsort(axis))
            if (axis.Length > 1)
            {
                int[] perm = ArgSort(axis);
                bool identity = true;
                for (int i = 0; i < perm.Length; i++)
                    if (perm[i] != i) { identity = false; break; }
                if (!identity)
                    wgt = wgt.transpose(perm);
            }

            int ndim = a.ndim;
            long[] newShape = new long[ndim];
            for (int i = 0; i < ndim; i++)
                newShape[i] = Array.IndexOf(axis, i) >= 0 ? a.shape[i] : 1L;
            wgt = wgt.reshape(newShape);
            return wgt;
        }

        private static int[] ArgSort(int[] arr)
        {
            int[] idx = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++) idx[i] = i;
            Array.Sort(idx, (x, y) => arr[x].CompareTo(arr[y]));
            return idx;
        }

        private static NPTypeCode ComputeResultDtype(NPTypeCode aType, NPTypeCode wType)
        {
            NPTypeCode common = _FindCommonArrayType(aType, wType);
            if (IsIntegralOrBool(aType))
                return _FindCommonArrayType(common, NPTypeCode.Double);
            return common;
        }

        private static bool IsIntegralOrBool(NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return true;
                default:
                    return false;
            }
        }

        private static bool HasZero(NDArray scl)
        {
            long n = scl.size;
            for (long i = 0; i < n; i++)
            {
                object v = scl.GetAtIndex(i);
                if (v is null) continue;
                double d = Convert.ToDouble(v);
                if (d == 0.0) return true;
            }
            return false;
        }
    }
}
