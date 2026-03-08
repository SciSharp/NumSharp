using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return minimum of an array or minimum along an axis, ignoring any NaNs.
        /// </summary>
        /// <param name="a">Array containing numbers whose minimum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the minimum is computed. The default is to compute the minimum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the minimum. If all values are NaN, returns NaN.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.amin (no NaN values possible).
        /// </remarks>
        public static NDArray nanmin(in NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
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
                                result = ILKernelGenerator.NanMinSimdHelperFloat((float*)arr.Address, arr.size);
                                break;
                            case NPTypeCode.Double:
                                result = ILKernelGenerator.NanMinSimdHelperDouble((double*)arr.Address, arr.size);
                                break;
                            default:
                                // Non-float types: fall back to regular amin (no NaN possible)
                                return amin(arr);
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
                    return nanmin_scalar(arr, keepdims);
                }
            }
            else
            {
                // Axis reduction: not yet implemented with SIMD
                return amin(arr, axis.Value);
            }
        }

        /// <summary>
        /// Scalar fallback for nanmin when SIMD is not available.
        /// </summary>
        private static NDArray nanmin_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    float minVal = float.PositiveInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                        {
                            if (val < minVal)
                                minVal = val;
                            foundNonNaN = true;
                        }
                    }
                    result = foundNonNaN ? minVal : float.NaN;
                    break;
                }
                case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    double minVal = double.PositiveInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                        {
                            if (val < minVal)
                                minVal = val;
                            foundNonNaN = true;
                        }
                    }
                    result = foundNonNaN ? minVal : double.NaN;
                    break;
                }
                default:
                    return amin(arr);
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
