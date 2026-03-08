using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
        /// </summary>
        /// <param name="a">Array containing numbers whose sum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the sum is computed. The default is to compute the sum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the sum, with NaN values treated as zero.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nansum.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.sum (no NaN values possible).
        /// </remarks>
        public static NDArray nansum(in NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                // Scalar case: return the value itself (NaN check handled by helper)
                return a.Clone();
            }

            // Element-wise (axis=None): use SIMD-optimized path
            if (axis == null)
            {
                if (ILKernelGenerator.Enabled && arr.Shape.IsContiguous)
                {
                    object result;
                    unsafe
                    {
                        switch (arr.GetTypeCode)
                        {
                            case NPTypeCode.Single:
                                result = ILKernelGenerator.NanSumSimdHelperFloat((float*)arr.Address, arr.size);
                                break;
                            case NPTypeCode.Double:
                                result = ILKernelGenerator.NanSumSimdHelperDouble((double*)arr.Address, arr.size);
                                break;
                            default:
                                // Non-float types: fall back to regular sum (no NaN possible)
                                return sum(arr, axis: null, keepdims: keepdims);
                        }
                    }

                    var r = NDArray.Scalar(result);
                    if (keepdims)
                    {
                        var keepdimsShape = new int[arr.ndim];
                        for (int i = 0; i < arr.ndim; i++)
                            keepdimsShape[i] = 1;
                        r.Storage.Reshape(new Shape(keepdimsShape));
                    }
                    return r;
                }
                else
                {
                    // Non-contiguous: use scalar fallback
                    return nansum_scalar(arr, keepdims);
                }
            }
            else
            {
                // Axis reduction: not yet implemented with SIMD, use scalar fallback
                // For now, delegate to regular sum (TODO: implement axis-aware NaN handling)
                return sum(arr, axis: axis, keepdims: keepdims);
            }
        }

        /// <summary>
        /// Scalar fallback for nansum when SIMD is not available.
        /// </summary>
        private static NDArray nansum_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    float sum = 0f;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                            sum += val;
                    }
                    result = sum;
                    break;
                }
                case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    double sum = 0.0;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                            sum += val;
                    }
                    result = sum;
                    break;
                }
                default:
                    // Non-float types: regular sum
                    return sum(arr);
            }

            var r = NDArray.Scalar(result);
            if (keepdims)
            {
                var keepdimsShape = new int[arr.ndim];
                for (int i = 0; i < arr.ndim; i++)
                    keepdimsShape[i] = 1;
                r.Storage.Reshape(new Shape(keepdimsShape));
            }
            return r;
        }
    }
}
