using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return maximum of an array or maximum along an axis, ignoring any NaNs.
        /// </summary>
        /// <param name="a">Array containing numbers whose maximum is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the maximum is computed. The default is to compute the maximum of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the maximum. If all values are NaN, returns NaN.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.amax (no NaN values possible).
        /// </remarks>
        public static NDArray nanmax(in NDArray a, int? axis = null, bool keepdims = false)
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
                                result = ILKernelGenerator.NanMaxSimdHelperFloat((float*)arr.Address, arr.size);
                                break;
                            case NPTypeCode.Double:
                                result = ILKernelGenerator.NanMaxSimdHelperDouble((double*)arr.Address, arr.size);
                                break;
                            default:
                                // Non-float types: fall back to regular amax (no NaN possible)
                                return amax(arr);
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
                    return nanmax_scalar(arr, keepdims);
                }
            }
            else
            {
                // Axis reduction: not yet implemented with SIMD
                return amax(arr, axis.Value);
            }
        }

        /// <summary>
        /// Scalar fallback for nanmax when SIMD is not available.
        /// </summary>
        private static NDArray nanmax_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    float maxVal = float.NegativeInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                        {
                            if (val > maxVal)
                                maxVal = val;
                            foundNonNaN = true;
                        }
                    }
                    result = foundNonNaN ? maxVal : float.NaN;
                    break;
                }
                case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    double maxVal = double.NegativeInfinity;
                    bool foundNonNaN = false;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                        {
                            if (val > maxVal)
                                maxVal = val;
                            foundNonNaN = true;
                        }
                    }
                    result = foundNonNaN ? maxVal : double.NaN;
                    break;
                }
                default:
                    return amax(arr);
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
