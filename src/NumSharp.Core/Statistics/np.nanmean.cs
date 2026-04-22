using System;
using System.Numerics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute the arithmetic mean along the specified axis, ignoring NaNs.
        /// Returns the average of the array elements. The average is taken over the flattened array by default,
        /// otherwise over the specified axis. float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="a">Array containing numbers whose mean is desired. If a is not an array, a conversion is attempted.</param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <returns>A new array containing the mean values, with NaN values ignored. If all values along an axis are NaN, returns NaN for that slice.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html</remarks>
        public static NDArray nanmean(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                return a.Clone();
            }

            // Element-wise (axis=None): compute mean ignoring NaN
            if (axis == null)
            {
                return nanmean_scalar(arr, keepdims);
            }
            else
            {
                // Axis reduction: compute mean along axis ignoring NaN
                return nanmean_axis(arr, axis.Value, keepdims);
            }
        }

        /// <summary>
        /// Scalar fallback for nanmean - computes mean of all elements ignoring NaN.
        /// </summary>
        private static NDArray nanmean_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    var iter = arr.AsIterator<float>();
                    double sum = 0.0;
                    long count = 0;
                    while (iter.HasNext())
                    {
                        float val = iter.MoveNext();
                        if (!float.IsNaN(val))
                        {
                            sum += val;
                            count++;
                        }
                    }
                    result = count > 0 ? (float)(sum / count) : float.NaN;
                    break;
                }
                case NPTypeCode.Double:
                {
                    var iter = arr.AsIterator<double>();
                    double sum = 0.0;
                    long count = 0;
                    while (iter.HasNext())
                    {
                        double val = iter.MoveNext();
                        if (!double.IsNaN(val))
                        {
                            sum += val;
                            count++;
                        }
                    }
                    result = count > 0 ? sum / count : double.NaN;
                    break;
                }
                case NPTypeCode.Half:
                {
                    // Half nanmean returns Half (NumPy parity: np.nanmean(float16) -> float16).
                    // Accumulate in double for precision, convert result to Half.
                    var iter = arr.AsIterator<Half>();
                    double sum = 0.0;
                    long count = 0;
                    while (iter.HasNext())
                    {
                        Half val = iter.MoveNext();
                        if (!Half.IsNaN(val))
                        {
                            sum += (double)val;
                            count++;
                        }
                    }
                    result = count > 0 ? (Half)(sum / count) : Half.NaN;
                    break;
                }
                case NPTypeCode.Complex:
                {
                    // Complex nanmean returns Complex. "NaN" = either real or imag is NaN.
                    var iter = arr.AsIterator<Complex>();
                    double sumR = 0.0, sumI = 0.0;
                    long count = 0;
                    while (iter.HasNext())
                    {
                        Complex val = iter.MoveNext();
                        if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
                        {
                            sumR += val.Real;
                            sumI += val.Imaginary;
                            count++;
                        }
                    }
                    result = count > 0
                        ? new Complex(sumR / count, sumI / count)
                        : new Complex(double.NaN, double.NaN);
                    break;
                }
                default:
                    // Non-float types: regular mean (no NaN possible)
                    return mean(arr);
            }

            var r = NDArray.Scalar(result);
            if (keepdims)
            {
                var keepdimsShape = new long[arr.ndim];
                for (int i = 0; i < arr.ndim; i++)
                    keepdimsShape[i] = 1;
                r.Storage.Reshape(new Shape(keepdimsShape));
            }
            return r;
        }

        /// <summary>
        /// Axis-aware nanmean - computes mean along specified axis ignoring NaN.
        /// </summary>
        private static NDArray nanmean_axis(NDArray arr, int axis, bool keepdims)
        {
            // Handle negative axis
            if (axis < 0)
                axis = arr.ndim + axis;

            if (axis < 0 || axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {arr.ndim}");

            // Half: axis-aware NaN-skipping mean, returns Half.
            if (arr.GetTypeCode == NPTypeCode.Half)
                return nanmean_axis_half(arr, axis, keepdims);

            // Complex: axis-aware NaN-skipping mean, returns Complex.
            if (arr.GetTypeCode == NPTypeCode.Complex)
                return nanmean_axis_complex(arr, axis, keepdims);

            // Non-float types: regular mean
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double)
            {
                return mean(arr, axis, keepdims);
            }

            // Build output shape
            var inputShape = arr.shape;
            var outputShapeList = new System.Collections.Generic.List<long>();
            for (int i = 0; i < inputShape.Length; i++)
            {
                if (i != axis)
                    outputShapeList.Add(inputShape[i]);
            }
            if (outputShapeList.Count == 0)
                outputShapeList.Add(1);

            var outputShape = outputShapeList.ToArray();
            long axisLen = inputShape[axis];

            // Create output array using unmanaged allocation (supports >2GB)
            NDArray result;
            if (arr.GetTypeCode == NPTypeCode.Single)
            {
                result = new NDArray(NPTypeCode.Single, new Shape(outputShape));
                long outputSize = result.size;

                // Iterate over output positions
                for (long outIdx = 0; outIdx < outputSize; outIdx++)
                {
                    double sum = 0.0;
                    long count = 0;

                    // Convert flat index to coordinates in output shape
                    var outCoords = new long[outputShape.Length];
                    long temp = outIdx;
                    for (int i = outputShape.Length - 1; i >= 0; i--)
                    {
                        outCoords[i] = temp % outputShape[i];
                        temp /= outputShape[i];
                    }

                    // Iterate along the axis
                    for (long k = 0; k < axisLen; k++)
                    {
                        // Build input coordinates
                        var inCoords = new long[inputShape.Length];
                        int outCoordIdx = 0;
                        for (int i = 0; i < inputShape.Length; i++)
                        {
                            if (i == axis)
                                inCoords[i] = k;
                            else
                                inCoords[i] = outCoords[outCoordIdx++];
                        }

                        float val = arr.GetSingle(inCoords);
                        if (!float.IsNaN(val))
                        {
                            sum += val;
                            count++;
                        }
                    }

                    result.SetSingle(count > 0 ? (float)(sum / count) : float.NaN, outCoords);
                }
            }
            else // Double
            {
                result = new NDArray(NPTypeCode.Double, new Shape(outputShape));
                long outputSize = result.size;

                for (long outIdx = 0; outIdx < outputSize; outIdx++)
                {
                    double sum = 0.0;
                    long count = 0;

                    var outCoords = new long[outputShape.Length];
                    long temp = outIdx;
                    for (int i = outputShape.Length - 1; i >= 0; i--)
                    {
                        outCoords[i] = temp % outputShape[i];
                        temp /= outputShape[i];
                    }

                    for (long k = 0; k < axisLen; k++)
                    {
                        var inCoords = new long[inputShape.Length];
                        int outCoordIdx = 0;
                        for (int i = 0; i < inputShape.Length; i++)
                        {
                            if (i == axis)
                                inCoords[i] = k;
                            else
                                inCoords[i] = outCoords[outCoordIdx++];
                        }

                        double val = arr.GetDouble(inCoords);
                        if (!double.IsNaN(val))
                        {
                            sum += val;
                            count++;
                        }
                    }

                    result.SetDouble(count > 0 ? sum / count : double.NaN, outCoords);
                }
            }

            // Handle keepdims
            if (keepdims)
            {
                var keepdimsShapeDims = new long[arr.ndim];
                int srcIdx = 0;
                for (int i = 0; i < arr.ndim; i++)
                {
                    if (i == axis)
                        keepdimsShapeDims[i] = 1;
                    else
                        keepdimsShapeDims[i] = outputShape[srcIdx++];
                }
                result = result.reshape(keepdimsShapeDims);
            }

            return result;
        }

        private static NDArray nanmean_axis_half(NDArray arr, int axis, bool keepdims)
        {
            var inputShape = arr.shape;
            var outputShapeList = new System.Collections.Generic.List<long>();
            for (int i = 0; i < inputShape.Length; i++)
                if (i != axis) outputShapeList.Add(inputShape[i]);
            if (outputShapeList.Count == 0) outputShapeList.Add(1);
            var outputShape = outputShapeList.ToArray();
            long axisLen = inputShape[axis];

            var result = new NDArray(NPTypeCode.Half, new Shape(outputShape));
            long outputSize = result.size;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                var outCoords = new long[outputShape.Length];
                long temp = outIdx;
                for (int i = outputShape.Length - 1; i >= 0; i--)
                {
                    outCoords[i] = temp % outputShape[i];
                    temp /= outputShape[i];
                }

                double sum = 0.0;
                long count = 0;
                for (long k = 0; k < axisLen; k++)
                {
                    var inCoords = new long[inputShape.Length];
                    int outCoordIdx = 0;
                    for (int i = 0; i < inputShape.Length; i++)
                        inCoords[i] = (i == axis) ? k : outCoords[outCoordIdx++];

                    Half val = arr.GetHalf(inCoords);
                    if (!Half.IsNaN(val))
                    {
                        sum += (double)val;
                        count++;
                    }
                }

                result.SetHalf(count > 0 ? (Half)(sum / count) : Half.NaN, outCoords);
            }

            return ApplyKeepdims(result, arr.ndim, axis, outputShape, keepdims);
        }

        private static NDArray nanmean_axis_complex(NDArray arr, int axis, bool keepdims)
        {
            var inputShape = arr.shape;
            var outputShapeList = new System.Collections.Generic.List<long>();
            for (int i = 0; i < inputShape.Length; i++)
                if (i != axis) outputShapeList.Add(inputShape[i]);
            if (outputShapeList.Count == 0) outputShapeList.Add(1);
            var outputShape = outputShapeList.ToArray();
            long axisLen = inputShape[axis];

            var result = new NDArray(NPTypeCode.Complex, new Shape(outputShape));
            long outputSize = result.size;

            for (long outIdx = 0; outIdx < outputSize; outIdx++)
            {
                var outCoords = new long[outputShape.Length];
                long temp = outIdx;
                for (int i = outputShape.Length - 1; i >= 0; i--)
                {
                    outCoords[i] = temp % outputShape[i];
                    temp /= outputShape[i];
                }

                double sumR = 0.0, sumI = 0.0;
                long count = 0;
                for (long k = 0; k < axisLen; k++)
                {
                    var inCoords = new long[inputShape.Length];
                    int outCoordIdx = 0;
                    for (int i = 0; i < inputShape.Length; i++)
                        inCoords[i] = (i == axis) ? k : outCoords[outCoordIdx++];

                    Complex val = arr.GetComplex(inCoords);
                    if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
                    {
                        sumR += val.Real;
                        sumI += val.Imaginary;
                        count++;
                    }
                }

                result.SetComplex(
                    count > 0 ? new Complex(sumR / count, sumI / count) : new Complex(double.NaN, double.NaN),
                    outCoords);
            }

            return ApplyKeepdims(result, arr.ndim, axis, outputShape, keepdims);
        }

        private static NDArray ApplyKeepdims(NDArray result, int ndim, int axis, long[] outputShape, bool keepdims)
        {
            if (!keepdims) return result;
            var keepdimsShapeDims = new long[ndim];
            int srcIdx = 0;
            for (int i = 0; i < ndim; i++)
                keepdimsShapeDims[i] = (i == axis) ? 1 : outputShape[srcIdx++];
            return result.reshape(keepdimsShapeDims);
        }
    }
}
