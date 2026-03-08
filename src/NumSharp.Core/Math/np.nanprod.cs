using System;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.
        /// </summary>
        /// <param name="a">Array containing numbers whose product is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the product is computed. The default is to compute the product of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the product, with NaN values treated as one.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html
        /// Only applicable to float and double arrays.
        /// For integer arrays, this is equivalent to np.prod (no NaN values possible).
        /// </remarks>
        public static NDArray nanprod(in NDArray a, int? axis = null, bool keepdims = false)
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
                                result = ILKernelGenerator.NanProdSimdHelperFloat((float*)arr.Address, arr.size);
                                break;
                            case NPTypeCode.Double:
                                result = ILKernelGenerator.NanProdSimdHelperDouble((double*)arr.Address, arr.size);
                                break;
                            default:
                                // Non-float types: fall back to regular prod (no NaN possible)
                                return prod(arr);
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
                    return nanprod_scalar(arr, keepdims);
                }
            }
            else
            {
                // Axis reduction: not yet implemented with SIMD
                return prod(arr);
            }
        }

        /// <summary>
        /// Scalar fallback for nanprod when SIMD is not available.
        /// </summary>
        private static NDArray nanprod_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    float prod = 1f;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                            prod *= val;
                    }
                    result = prod;
                    break;
                }
                case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    double prod = 1.0;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                            prod *= val;
                    }
                    result = prod;
                    break;
                }
                default:
                    return prod(arr);
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
