using System;

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
            long outputSize = 1;
            foreach (var dim in outputShape)
                outputSize *= dim;

            // .NET arrays are int-indexed
            if (outputSize > int.MaxValue)
                throw new InvalidOperationException("Output size exceeds maximum supported array size");

            long axisLen = inputShape[axis];

            // Create output array
            NDArray result;
            if (arr.GetTypeCode == NPTypeCode.Single)
            {
                var outputData = new float[(int)outputSize];

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
                        outputData[outIdx] = float.NaN;
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

                        outputData[outIdx] = (float)(sumSq / (count - ddof));
                    }
                }

                result = new NDArray(outputData).reshape(outputShape);
            }
            else // Double
            {
                var outputData = new double[(int)outputSize];

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
                        outputData[outIdx] = double.NaN;
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

                        outputData[outIdx] = sumSq / (count - ddof);
                    }
                }

                result = new NDArray(outputData).reshape(outputShape);
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
    }
}
