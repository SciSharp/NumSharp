using System;

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
            long outputSize = 1;
            foreach (var dim in outputShape)
                outputSize *= dim;

            // .NET arrays are int-indexed
            if (outputSize > int.MaxValue)
                throw new InvalidOperationException($"Output size {outputSize} exceeds int.MaxValue ({int.MaxValue}). C#/.NET managed arrays are limited to int32 indexing.");

            long axisLen = inputShape[axis];

            // Create output array
            NDArray result;
            if (arr.GetTypeCode == NPTypeCode.Single)
            {
                var outputData = new float[(int)outputSize];

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

                    outputData[outIdx] = count > 0 ? (float)(sum / count) : float.NaN;
                }

                result = new NDArray(outputData).reshape(outputShape);
            }
            else // Double
            {
                var outputData = new double[(int)outputSize];

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

                    outputData[outIdx] = count > 0 ? sum / count : double.NaN;
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
