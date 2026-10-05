using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

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
        [NDScoped]
        public static NDArray nanmean(NDArray a, int? axis = null, bool keepdims = false)
        {
            var arr = a;
            var shape = arr.Shape;

            if (shape.IsEmpty)
                return arr;

            // A one-element input. NumPy's nanmean is np.sum(replaced) / count-of-non-NaN, which for one element is
            // (0 + x) / 1 — np.mean of it — or 0 / 0 = NaN when x is NaN (np.mean of a NaN is NaN too; NaN bits are not
            // part of the contract, and a complex NaN part poisons both parts either way). So np.mean's one-element
            // route answers: 0-d (keepdims: (1,)*ndim), float64 for an integer input, the +0.0 identity for a -0.0.
            // The old shortcut returned the input's clone — shape (1,), the integer dtype, the -0.0 sign.
            if (shape.size == 1)
            {
                int? ax = axis;
                if (ax != null && arr.ndim == 0)
                {
                    // nanmean reduces through np.sum, so a 0-d input takes axis 0 / -1 as every axis (ufunc.reduce's
                    // rule) where np.mean itself would raise.
                    if (ax.Value != 0 && ax.Value != -1)
                        throw new AxisError(ax.Value, 0);
                    ax = null;
                }

                return ax.HasValue ? mean(arr, ax.Value, keepdims) : mean(arr, keepdims);
            }

            // Element-wise (axis=None): compute mean ignoring NaN
            if (axis == null)
            {
                return nanmean_scalar(arr, keepdims);
            }
            else
            {
                // Axis reduction: compute mean along axis ignoring NaN. The result is allocated in
                // KEEPORDER (F for an F-contig input) and filled by coordinate, so it matches NumPy's
                // layout with no post-hoc copy (issue #610).
                return nanmean_axis(arr, axis.Value, keepdims);
            }
        }

        /// <summary>
        /// Scalar fallback for nanmean - computes mean of all elements ignoring NaN.
        /// </summary>
        private static unsafe NDArray nanmean_scalar(NDArray arr, bool keepdims)
        {
            object result;

            switch (arr.GetTypeCode)
            {
                case NPTypeCode.Single:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMeanFloatKernel, NanMeanAccumulator>(default, default);
                    result = accum.Count > 0 ? (float)(accum.Sum / accum.Count) : float.NaN;
                    break;
                }
                case NPTypeCode.Double:
                {
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMeanDoubleKernel, NanMeanAccumulator>(default, default);
                    result = accum.Count > 0 ? accum.Sum / accum.Count : double.NaN;
                    break;
                }
                case NPTypeCode.Half:
                {
                    // Half nanmean returns Half (NumPy parity: np.nanmean(float16) -> float16).
                    // Accumulate in double for precision, convert result to Half.
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMeanHalfKernel, NanMeanAccumulator>(default, default);
                    result = accum.Count > 0 ? (Half)(accum.Sum / accum.Count) : Half.NaN;
                    break;
                }
                case NPTypeCode.Complex:
                {
                    // Complex nanmean returns Complex. "NaN" = either real or imag is NaN.
                    using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
                    var accum = iter.ExecuteReducing<NanMeanComplexKernel, NanMeanComplexAccumulator>(default, default);
                    result = accum.Count > 0
                        ? accum.Sum / accum.Count
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
                r.Storage.Reshape(Shape.Adopt(keepdimsShape));
            }
            return r.MarkReductionScalar();
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
            var inputShape = arr.Shape.dimensions;
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
                result = Backends.DefaultEngine.AllocateReductionResult(NPTypeCode.Single, outputShape, arr.Shape);
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
                result = Backends.DefaultEngine.AllocateReductionResult(NPTypeCode.Double, outputShape, arr.Shape);
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

            // Handle keepdims (ExpandDimension re-inserts the reduced axis preserving the KEEPORDER layout; a reshape
            // to a fresh C-shape would reset it to C) and the 0-d result of a 1-D input.
            return ApplyKeepdims(result, arr.ndim, axis, outputShape, keepdims);
        }

        private static NDArray nanmean_axis_half(NDArray arr, int axis, bool keepdims)
        {
            var inputShape = arr.Shape.dimensions;
            var outputShapeList = new System.Collections.Generic.List<long>();
            for (int i = 0; i < inputShape.Length; i++)
                if (i != axis) outputShapeList.Add(inputShape[i]);
            if (outputShapeList.Count == 0) outputShapeList.Add(1);
            var outputShape = outputShapeList.ToArray();
            long axisLen = inputShape[axis];

            var result = Backends.DefaultEngine.AllocateReductionResult(NPTypeCode.Half, outputShape, arr.Shape);
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
            var inputShape = arr.Shape.dimensions;
            var outputShapeList = new System.Collections.Generic.List<long>();
            for (int i = 0; i < inputShape.Length; i++)
                if (i != axis) outputShapeList.Add(inputShape[i]);
            if (outputShapeList.Count == 0) outputShapeList.Add(1);
            var outputShape = outputShapeList.ToArray();
            long axisLen = inputShape[axis];

            var result = Backends.DefaultEngine.AllocateReductionResult(NPTypeCode.Complex, outputShape, arr.Shape);
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

        /// <summary>
        ///     Gives an axis nanmean result NumPy's final shape: a 1-D input reduced over its only axis is 0-d (a
        ///     read-only scalar; <c>(1,)</c> under keepdims), and otherwise keepdims re-inserts the reduced axis.
        /// </summary>
        /// <param name="result">The computed means, over the loops' output shape — a placeholder <c>(1,)</c> for a 1-D input.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <param name="axis">The normalized reduced axis.</param>
        /// <param name="outputShape">The output shape the loops used (unused; kept for the call sites' symmetry).</param>
        /// <param name="keepdims">Whether the reduced axis stays with length 1.</param>
        /// <returns><paramref name="result"/>, relabelled in place (one element or an axis of length 1: nothing moves).</returns>
        private static NDArray ApplyKeepdims(NDArray result, int ndim, int axis, long[] outputShape, bool keepdims)
        {
            if (ndim == 1)
            {
                // The loops ran over a placeholder (1,) output (they need a coordinate to write); NumPy's result for a
                // 1-D input is 0-d — which also kept keepdims from turning it into (1,1).
                result.Storage.Reshape(keepdims ? new Shape(1L) : Shape.Scalar);
                return result.MarkReductionScalar();
            }

            if (!keepdims) return result;
            // ExpandDimension re-inserts the reduced axis preserving the KEEPORDER layout (issue #610).
            result.Storage.ExpandDimension(axis);
            return result;
        }
    }
}
