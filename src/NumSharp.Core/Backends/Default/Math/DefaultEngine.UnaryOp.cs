using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Unary operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute a unary operation using IL-generated kernels.
        /// Handles type promotion, strided arrays, and kernel dispatch.
        /// </summary>
        /// <param name="nd">Input array</param>
        /// <param name="op">Operation to perform</param>
        /// <param name="typeCode">Optional output type (null = same as input or float for trig/sqrt)</param>
        /// <returns>Result array with specified or promoted type</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray ExecuteUnaryOp(in NDArray nd, UnaryOp op, NPTypeCode? typeCode = null)
        {
            if (nd.size == 0)
                return nd.Clone();

            var inputType = nd.GetTypeCode;

            // Determine output type:
            // - If explicit type provided, use it directly
            // - For trig/math functions (Sin, Cos, Exp, Log, Sqrt), use ResolveUnaryReturnType (promotes to float)
            // - For arithmetic functions (Negate, Abs), preserve input type
            NPTypeCode outputType;
            if (typeCode.HasValue)
            {
                outputType = typeCode.Value;
            }
            else if (op == UnaryOp.Negate || op == UnaryOp.Abs)
            {
                // Arithmetic operations preserve type
                outputType = inputType;
            }
            else
            {
                // Math functions promote to computing type (typically float/double)
                outputType = ResolveUnaryReturnType(nd, (NPTypeCode?)null);
            }

            // Handle scalar case
            if (nd.Shape.IsScalar)
            {
                return ExecuteScalarUnary(nd, op, outputType);
            }

            // Determine if array is contiguous
            bool isContiguous = nd.Shape.IsContiguous;

            // Allocate result (always contiguous)
            var result = new NDArray(outputType, nd.Shape.Clean(), false);

            // Get kernel key
            var key = new UnaryKernelKey(inputType, outputType, op, isContiguous);

            // Get or generate kernel
            var kernel = ILKernelGenerator.TryGetUnaryKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteUnaryKernel(kernel, nd, result);
            }
            else
            {
                // Fallback - should not happen for implemented operations
                throw new NotSupportedException(
                    $"IL kernel not available for {op}({inputType}) -> {outputType}. " +
                    "Please report this as a bug.");
            }

            return result;
        }

        /// <summary>
        /// Execute scalar unary operation using IL-generated delegate.
        /// </summary>
        private NDArray ExecuteScalarUnary(in NDArray nd, UnaryOp op, NPTypeCode outputType)
        {
            var inputType = nd.GetTypeCode;
            var key = new UnaryScalarKernelKey(inputType, outputType, op);
            var func = ILKernelGenerator.GetUnaryScalarDelegate(key);

            // Dispatch based on input type to avoid boxing
            return inputType switch
            {
                NPTypeCode.Boolean => InvokeUnaryScalar(func, nd.GetBoolean(), outputType),
                NPTypeCode.Byte => InvokeUnaryScalar(func, nd.GetByte(), outputType),
                NPTypeCode.Int16 => InvokeUnaryScalar(func, nd.GetInt16(), outputType),
                NPTypeCode.UInt16 => InvokeUnaryScalar(func, nd.GetUInt16(), outputType),
                NPTypeCode.Int32 => InvokeUnaryScalar(func, nd.GetInt32(), outputType),
                NPTypeCode.UInt32 => InvokeUnaryScalar(func, nd.GetUInt32(), outputType),
                NPTypeCode.Int64 => InvokeUnaryScalar(func, nd.GetInt64(), outputType),
                NPTypeCode.UInt64 => InvokeUnaryScalar(func, nd.GetUInt64(), outputType),
                NPTypeCode.Char => InvokeUnaryScalar(func, nd.GetChar(), outputType),
                NPTypeCode.Single => InvokeUnaryScalar(func, nd.GetSingle(), outputType),
                NPTypeCode.Double => InvokeUnaryScalar(func, nd.GetDouble(), outputType),
                NPTypeCode.Decimal => InvokeUnaryScalar(func, nd.GetDecimal(), outputType),
                _ => throw new NotSupportedException($"Input type {inputType} not supported")
            };
        }

        /// <summary>
        /// Invoke a unary scalar delegate and create the result NDArray.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeUnaryScalar<TInput>(Delegate func, TInput input, NPTypeCode outputType)
        {
            // Dispatch based on output type to avoid boxing on result
            return outputType switch
            {
                NPTypeCode.Boolean => NDArray.Scalar(((Func<TInput, bool>)func)(input)),
                NPTypeCode.Byte => NDArray.Scalar(((Func<TInput, byte>)func)(input)),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TInput, short>)func)(input)),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TInput, ushort>)func)(input)),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TInput, int>)func)(input)),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TInput, uint>)func)(input)),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TInput, long>)func)(input)),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TInput, ulong>)func)(input)),
                NPTypeCode.Char => NDArray.Scalar(((Func<TInput, char>)func)(input)),
                NPTypeCode.Single => NDArray.Scalar(((Func<TInput, float>)func)(input)),
                NPTypeCode.Double => NDArray.Scalar(((Func<TInput, double>)func)(input)),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TInput, decimal>)func)(input)),
                _ => throw new NotSupportedException($"Output type {outputType} not supported")
            };
        }

        /// <summary>
        /// Execute the IL-generated unary kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExecuteUnaryKernel(
            UnaryKernel kernel,
            in NDArray input, NDArray result)
        {
            int inputElemSize = input.dtypesize;
            var inputShape = input.Shape;

            // Calculate base address accounting for shape offset (for sliced views)
            byte* inputAddr = (byte*)input.Address + inputShape.offset * inputElemSize;

            fixed (int* strides = inputShape.strides)
            fixed (int* shape = result.shape)
            {
                kernel(
                    (void*)inputAddr,
                    (void*)result.Address,
                    strides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }
    }
}
