using System;
using System.Numerics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute the variance along the specified axis, while ignoring NaNs.
        /// Returns the variance of the array elements, a measure of the spread of a distribution.
        /// The variance is computed for the flattened array by default, otherwise over the specified axis.
        /// </summary>
        /// <param name="a">Array containing numbers whose variance is desired.</param>
        /// <param name="axis">Axis or axes along which the variance is computed. The default is to compute the variance of the flattened array.</param>
        /// <param name="keepdims">If this is set to True, the axes which are reduced are left in the result as dimensions with size one.</param>
        /// <param name="ddof">Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of non-NaN elements. By default ddof is zero.</param>
        /// <returns>A new array containing the variance. If all values along an axis are NaN, returns NaN for that slice.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html</remarks>
        public static NDArray nanvar(NDArray a, int? axis = null, bool keepdims = false, int ddof = 0)
        {
            var arr = a;
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return arr;

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
            {
                // Single element: variance is 0 (or NaN if element is NaN)
                if (arr.GetTypeCode == NPTypeCode.Single)
                {
                    float val = arr.GetSingle();
                    return NDArray.Scalar(float.IsNaN(val) ? float.NaN : 0f);
                }
                else if (arr.GetTypeCode == NPTypeCode.Double)
                {
                    double val = arr.GetDouble();
                    return NDArray.Scalar(double.IsNaN(val) ? double.NaN : 0.0);
                }
                else if (arr.GetTypeCode == NPTypeCode.Half)
                {
                    Half val = arr.GetHalf();
                    return NDArray.Scalar(Half.IsNaN(val) ? Half.NaN : (Half)0);
                }
                else if (arr.GetTypeCode == NPTypeCode.Complex)
                {
                    // NumPy: nanvar of complex returns float64.
                    Complex val = arr.GetComplex();
                    bool isNaN = double.IsNaN(val.Real) || double.IsNaN(val.Imaginary);
                    return NDArray.Scalar(isNaN ? double.NaN : 0.0);
                }
                return NDArray.Scalar(0.0);
            }

            // Element-wise (axis=None): compute variance ignoring NaN
            if (axis == null)
            {
                return nanvar_scalar(arr, keepdims, ddof);
            }
            else
            {
                // Axis reduction: compute variance along axis ignoring NaN
                return nanvar_axis(arr, axis.Value, keepdims, ddof);
            }
        }

        /// <summary>
        /// Scalar fallback for nanvar - computes variance of all elements ignoring NaN.
        /// </summary>
        private static NDArray nanvar_scalar(NDArray arr, bool keepdims, int ddof)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    // Two-pass algorithm: first compute mean, then variance
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

                    if (count <= ddof)
                    {
                        result = float.NaN;
                    }
                    else
                    {
                        double mean = sum / count;
                        iter.Reset();
                        double sumSq = 0.0;
                        while (iter.HasNext())
                        {
                            float val = iter.MoveNext();
                            if (!float.IsNaN(val))
                            {
                                double diff = val - mean;
                                sumSq += diff * diff;
                            }
                        }
                        result = (float)(sumSq / (count - ddof));
                    }
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

                    if (count <= ddof)
                    {
                        result = double.NaN;
                    }
                    else
                    {
                        double mean = sum / count;
                        iter.Reset();
                        double sumSq = 0.0;
                        while (iter.HasNext())
                        {
                            double val = iter.MoveNext();
                            if (!double.IsNaN(val))
                            {
                                double diff = val - mean;
                                sumSq += diff * diff;
                            }
                        }
                        result = sumSq / (count - ddof);
                    }
                    break;
                }
                case NPTypeCode.Half:
                {
                    // Half nanvar returns Half (NumPy parity).
                    // Two-pass: compute mean, then mean(|x - mean|²).
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

                    if (count <= ddof)
                    {
                        result = Half.NaN;
                    }
                    else
                    {
                        double mean = sum / count;
                        iter.Reset();
                        double sumSq = 0.0;
                        while (iter.HasNext())
                        {
                            Half val = iter.MoveNext();
                            if (!Half.IsNaN(val))
                            {
                                double diff = (double)val - mean;
                                sumSq += diff * diff;
                            }
                        }
                        result = (Half)(sumSq / (count - ddof));
                    }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    // Complex nanvar returns float64 (NumPy parity).
                    // Variance = mean(|z - mean(z)|²). NaN-containing = Re or Im is NaN.
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

                    if (count <= ddof)
                    {
                        result = double.NaN;
                    }
                    else
                    {
                        double meanR = sumR / count;
                        double meanI = sumI / count;
                        iter.Reset();
                        double sumSq = 0.0;
                        while (iter.HasNext())
                        {
                            Complex val = iter.MoveNext();
                            if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
                            {
                                double dR = val.Real - meanR;
                                double dI = val.Imaginary - meanI;
                                sumSq += dR * dR + dI * dI;
                            }
                        }
                        result = sumSq / (count - ddof);
                    }
                    break;
                }
                default:
                    // Non-float types: regular var (no NaN possible)
                    return var(arr, ddof: ddof);
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
        /// Axis-aware nanvar - computes variance along specified axis ignoring NaN.
        /// </summary>
        private static NDArray nanvar_axis(NDArray arr, int axis, bool keepdims, int ddof)
        {
            // Handle negative axis
            if (axis < 0)
                axis = arr.ndim + axis;

            if (axis < 0 || axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis), $"axis {axis} is out of bounds for array of dimension {arr.ndim}");

            if (arr.GetTypeCode == NPTypeCode.Half)
                return nanvar_axis_half(arr, axis, keepdims, ddof);

            if (arr.GetTypeCode == NPTypeCode.Complex)
                return nanvar_axis_complex(arr, axis, keepdims, ddof);

            // Non-float types: regular var
            if (arr.GetTypeCode != NPTypeCode.Single && arr.GetTypeCode != NPTypeCode.Double)
            {
                return var(arr, axis, ddof: ddof);
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

                for (long outIdx = 0; outIdx < outputSize; outIdx++)
                {
                    // Convert flat index to coordinates in output shape
                    var outCoords = new long[outputShape.Length];
                    long temp = outIdx;
                    for (int i = outputShape.Length - 1; i >= 0; i--)
                    {
                        outCoords[i] = temp % outputShape[i];
                        temp /= outputShape[i];
                    }

                    // First pass: compute mean
                    double sum = 0.0;
                    long count = 0;
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

                        float val = arr.GetSingle(inCoords);
                        if (!float.IsNaN(val))
                        {
                            sum += val;
                            count++;
                        }
                    }

                    if (count <= ddof)
                    {
                        result.SetSingle(float.NaN, outCoords);
                    }
                    else
                    {
                        double mean = sum / count;

                        // Second pass: compute variance
                        double sumSq = 0.0;
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

                            float val = arr.GetSingle(inCoords);
                            if (!float.IsNaN(val))
                            {
                                double diff = val - mean;
                                sumSq += diff * diff;
                            }
                        }

                        result.SetSingle((float)(sumSq / (count - ddof)), outCoords);
                    }
                }
            }
            else // Double
            {
                result = new NDArray(NPTypeCode.Double, new Shape(outputShape));
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

                    // First pass: compute mean
                    double sum = 0.0;
                    long count = 0;
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

                    if (count <= ddof)
                    {
                        result.SetDouble(double.NaN, outCoords);
                    }
                    else
                    {
                        double mean = sum / count;

                        // Second pass: compute variance
                        double sumSq = 0.0;
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
                                double diff = val - mean;
                                sumSq += diff * diff;
                            }
                        }

                        result.SetDouble(sumSq / (count - ddof), outCoords);
                    }
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

        private static NDArray nanvar_axis_half(NDArray arr, int axis, bool keepdims, int ddof)
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

                // Pass 1: mean
                double sum = 0.0;
                long count = 0;
                for (long k = 0; k < axisLen; k++)
                {
                    var inCoords = new long[inputShape.Length];
                    int outCoordIdx = 0;
                    for (int i = 0; i < inputShape.Length; i++)
                        inCoords[i] = (i == axis) ? k : outCoords[outCoordIdx++];
                    Half val = arr.GetHalf(inCoords);
                    if (!Half.IsNaN(val)) { sum += (double)val; count++; }
                }

                if (count <= ddof)
                {
                    result.SetHalf(Half.NaN, outCoords);
                    continue;
                }

                double mean = sum / count;
                double sumSq = 0.0;
                for (long k = 0; k < axisLen; k++)
                {
                    var inCoords = new long[inputShape.Length];
                    int outCoordIdx = 0;
                    for (int i = 0; i < inputShape.Length; i++)
                        inCoords[i] = (i == axis) ? k : outCoords[outCoordIdx++];
                    Half val = arr.GetHalf(inCoords);
                    if (!Half.IsNaN(val))
                    {
                        double diff = (double)val - mean;
                        sumSq += diff * diff;
                    }
                }
                result.SetHalf((Half)(sumSq / (count - ddof)), outCoords);
            }

            return ApplyKeepdims(result, arr.ndim, axis, outputShape, keepdims);
        }

        private static NDArray nanvar_axis_complex(NDArray arr, int axis, bool keepdims, int ddof)
        {
            var inputShape = arr.shape;
            var outputShapeList = new System.Collections.Generic.List<long>();
            for (int i = 0; i < inputShape.Length; i++)
                if (i != axis) outputShapeList.Add(inputShape[i]);
            if (outputShapeList.Count == 0) outputShapeList.Add(1);
            var outputShape = outputShapeList.ToArray();
            long axisLen = inputShape[axis];

            // NumPy: nanvar of complex returns float64.
            var result = new NDArray(NPTypeCode.Double, new Shape(outputShape));
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

                if (count <= ddof)
                {
                    result.SetDouble(double.NaN, outCoords);
                    continue;
                }

                double meanR = sumR / count;
                double meanI = sumI / count;
                double sumSq = 0.0;
                for (long k = 0; k < axisLen; k++)
                {
                    var inCoords = new long[inputShape.Length];
                    int outCoordIdx = 0;
                    for (int i = 0; i < inputShape.Length; i++)
                        inCoords[i] = (i == axis) ? k : outCoords[outCoordIdx++];
                    Complex val = arr.GetComplex(inCoords);
                    if (!double.IsNaN(val.Real) && !double.IsNaN(val.Imaginary))
                    {
                        double dR = val.Real - meanR;
                        double dI = val.Imaginary - meanI;
                        sumSq += dR * dR + dI * dI;
                    }
                }
                result.SetDouble(sumSq / (count - ddof), outCoords);
            }

            return ApplyKeepdims(result, arr.ndim, axis, outputShape, keepdims);
        }
    }
}
