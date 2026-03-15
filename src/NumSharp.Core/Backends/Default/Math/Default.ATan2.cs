using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise arc tangent of y/x choosing the quadrant correctly.
        /// NumPy: arctan2(y, x) returns the angle in radians between the positive x-axis
        /// and the point (x, y), with correct quadrant determination.
        /// </summary>
        /// <param name="y">y-coordinates</param>
        /// <param name="x">x-coordinates. If y.shape != x.shape, they must be broadcastable.</param>
        /// <param name="dtype">Output dtype (overrides type promotion)</param>
        /// <returns>Array of angles in radians, range [-pi, pi]</returns>
        public override NDArray ATan2(NDArray y, NDArray x, Type dtype)
            => ATan2(y, x, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise arc tangent of y/x choosing the quadrant correctly.
        /// NumPy: arctan2(y, x) returns the angle in radians between the positive x-axis
        /// and the point (x, y), with correct quadrant determination.
        /// </summary>
        /// <param name="y">y-coordinates</param>
        /// <param name="x">x-coordinates. If y.shape != x.shape, they must be broadcastable.</param>
        /// <param name="typeCode">Output dtype (overrides type promotion). If null, uses NumPy rules.</param>
        /// <returns>Array of angles in radians, range [-pi, pi]</returns>
        public override NDArray ATan2(NDArray y, NDArray x, NPTypeCode? typeCode = null)
        {
            // Handle empty array
            if (y.size == 0)
                return y.Clone();

            if (x.size == 0)
                return x.Clone();

            // Execute using IL kernel infrastructure
            // ATan2 is a binary operation: atan2(y, x)
            return ExecuteATan2Op(y, x, typeCode);
        }

        /// <summary>
        /// Execute ATan2 using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// </summary>
        private unsafe NDArray ExecuteATan2Op(NDArray y, NDArray x, NPTypeCode? typeCode)
        {
            var yType = y.GetTypeCode;
            var xType = x.GetTypeCode;

            // Determine result type using NumPy arctan2 rules:
            // - float32 inputs -> float32 output
            // - float64 or integer inputs -> float64 output
            NPTypeCode resultType;
            if (typeCode.HasValue)
            {
                resultType = typeCode.Value;
            }
            else
            {
                // NumPy arctan2 type promotion:
                // float32 + float32 -> float32
                // anything else -> float64
                if (yType == NPTypeCode.Single && xType == NPTypeCode.Single)
                {
                    resultType = NPTypeCode.Single;
                }
                else if (yType == NPTypeCode.Decimal || xType == NPTypeCode.Decimal)
                {
                    resultType = NPTypeCode.Decimal;
                }
                else
                {
                    resultType = NPTypeCode.Double;
                }
            }

            // Handle scalar x scalar case
            if (y.Shape.IsScalar && x.Shape.IsScalar)
            {
                return ExecuteATan2ScalarScalar(y, x, yType, xType, resultType);
            }

            // Broadcast shapes
            var (yShape, xShape) = Broadcast(y.Shape, x.Shape);
            var resultShape = yShape.Clean();

            // Allocate result
            var result = new NDArray(resultType, resultShape, false);

            // Classify execution path using strides
            ExecutionPath path;
            fixed (long* yStrides = yShape.strides)
            fixed (long* xStrides = xShape.strides)
            fixed (long* shape = resultShape.dimensions)
            {
                path = ClassifyATan2Path(yStrides, xStrides, shape, resultShape.NDim);
            }

            // Get kernel key
            var key = new MixedTypeKernelKey(yType, xType, resultType, BinaryOp.ATan2, path);

            // Get or generate kernel
            var kernel = ILKernelGenerator.GetMixedTypeKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteATan2Kernel(kernel, y, x, result, yShape, xShape);
            }
            else
            {
                throw new NotSupportedException(
                    $"IL kernel not available for arctan2({yType}, {xType}) -> {resultType}. " +
                    "Please report this as a bug.");
            }

            return result;
        }

        /// <summary>
        /// Execute scalar x scalar ATan2 operation.
        /// </summary>
        private static NDArray ExecuteATan2ScalarScalar(
            NDArray y, NDArray x,
            NPTypeCode yType, NPTypeCode xType, NPTypeCode resultType)
        {
            // Get values as double for Math.Atan2
            double yVal = ConvertToDouble(y, yType);
            double xVal = ConvertToDouble(x, xType);

            double result = Math.Atan2(yVal, xVal);

            // Convert to result type
            return resultType switch
            {
                NPTypeCode.Single => NDArray.Scalar((float)result),
                NPTypeCode.Double => NDArray.Scalar(result),
                NPTypeCode.Decimal => NDArray.Scalar(Utilities.DecimalMath.ATan2(
                    ConvertToDecimal(y, yType), ConvertToDecimal(x, xType))),
                _ => NDArray.Scalar(result)
            };
        }

        /// <summary>
        /// Convert NDArray scalar to double.
        /// </summary>
        private static double ConvertToDouble(NDArray arr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => arr.GetBoolean(Array.Empty<long>()) ? 1.0 : 0.0,
                NPTypeCode.Byte => arr.GetByte(Array.Empty<long>()),
                NPTypeCode.Int16 => arr.GetInt16(Array.Empty<long>()),
                NPTypeCode.UInt16 => arr.GetUInt16(Array.Empty<long>()),
                NPTypeCode.Int32 => arr.GetInt32(Array.Empty<long>()),
                NPTypeCode.UInt32 => arr.GetUInt32(Array.Empty<long>()),
                NPTypeCode.Int64 => arr.GetInt64(Array.Empty<long>()),
                NPTypeCode.UInt64 => arr.GetUInt64(Array.Empty<long>()),
                NPTypeCode.Char => arr.GetChar(Array.Empty<long>()),
                NPTypeCode.Single => arr.GetSingle(Array.Empty<long>()),
                NPTypeCode.Double => arr.GetDouble(Array.Empty<long>()),
                NPTypeCode.Decimal => (double)arr.GetDecimal(Array.Empty<long>()),
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Convert NDArray scalar to decimal.
        /// </summary>
        private static decimal ConvertToDecimal(NDArray arr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => arr.GetBoolean(Array.Empty<long>()) ? 1m : 0m,
                NPTypeCode.Byte => arr.GetByte(Array.Empty<long>()),
                NPTypeCode.Int16 => arr.GetInt16(Array.Empty<long>()),
                NPTypeCode.UInt16 => arr.GetUInt16(Array.Empty<long>()),
                NPTypeCode.Int32 => arr.GetInt32(Array.Empty<long>()),
                NPTypeCode.UInt32 => arr.GetUInt32(Array.Empty<long>()),
                NPTypeCode.Int64 => arr.GetInt64(Array.Empty<long>()),
                NPTypeCode.UInt64 => arr.GetUInt64(Array.Empty<long>()),
                NPTypeCode.Char => arr.GetChar(Array.Empty<long>()),
                NPTypeCode.Single => (decimal)arr.GetSingle(Array.Empty<long>()),
                NPTypeCode.Double => (decimal)arr.GetDouble(Array.Empty<long>()),
                NPTypeCode.Decimal => arr.GetDecimal(Array.Empty<long>()),
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Classify execution path for ATan2 based on strides.
        /// </summary>
        private static unsafe ExecutionPath ClassifyATan2Path(
            long* yStrides, long* xStrides, long* shape, int ndim)
        {
            if (ndim == 0)
                return ExecutionPath.SimdFull;

            bool yContiguous = StrideDetector.IsContiguous(yStrides, shape, ndim);
            bool xContiguous = StrideDetector.IsContiguous(xStrides, shape, ndim);

            if (yContiguous && xContiguous)
                return ExecutionPath.SimdFull;

            // SimdScalarRight/Left require the non-scalar operand to be contiguous
            bool xScalar = StrideDetector.IsScalar(xStrides, ndim);
            if (xScalar && yContiguous)
                return ExecutionPath.SimdScalarRight;

            bool yScalar = StrideDetector.IsScalar(yStrides, ndim);
            if (yScalar && xContiguous)
                return ExecutionPath.SimdScalarLeft;

            // Check for inner-contiguous (chunk-able)
            long yInner = yStrides[ndim - 1];
            long xInner = xStrides[ndim - 1];
            if ((yInner == 1 || yInner == 0) && (xInner == 1 || xInner == 0))
                return ExecutionPath.SimdChunk;

            return ExecutionPath.General;
        }

        /// <summary>
        /// Execute the IL-generated ATan2 kernel.
        /// </summary>
        private static unsafe void ExecuteATan2Kernel(
            MixedTypeKernel kernel,
            NDArray y, NDArray x, NDArray result,
            Shape yShape, Shape xShape)
        {
            // Get element sizes for offset calculation
            int yElemSize = y.dtypesize;
            int xElemSize = x.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            byte* yAddr = (byte*)y.Address + yShape.offset * yElemSize;
            byte* xAddr = (byte*)x.Address + xShape.offset * xElemSize;

            fixed (long* yStrides = yShape.strides)
            fixed (long* xStrides = xShape.strides)
            fixed (long* shape = result.shape)
            {
                kernel(
                    (void*)yAddr,
                    (void*)xAddr,
                    (void*)result.Address,
                    yStrides,
                    xStrides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }
    }
}
