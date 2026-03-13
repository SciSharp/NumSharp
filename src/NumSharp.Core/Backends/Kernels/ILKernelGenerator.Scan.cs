using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Scan.cs - Scan (prefix sum) kernel generation
// =============================================================================
//
// ARCHITECTURE OVERVIEW
// ---------------------
// Scan operations (cumsum, cumprod) are fundamentally different from reductions:
// - Reduction: N inputs -> 1 output (scalar)
// - Scan: N inputs -> N outputs (each output depends on all previous inputs)
//
// This makes SIMD optimization challenging for scans because each element
// depends on the previous element (sequential dependency). However, we can
// still optimize:
// 1. Contiguous same-type: Direct pointer access without iterator overhead
// 2. Type conversion: Efficient widening during accumulation
// 3. Memory access: Sequential writes are cache-friendly
//
// For contiguous arrays where input and output types match, we use a simple
// scalar loop with direct pointer access. For non-contiguous arrays, we fall
// back to coordinate-based iteration.
//
// =============================================================================
// PARTIAL CLASS FILE OWNERSHIP
// =============================================================================
//
// ILKernelGenerator.Scan.cs (THIS FILE)
//   OWNERSHIP: Scan (cumulative) operations
//   RESPONSIBILITY:
//     - CumSum: cumulative sum (running total)
//     - CumProd: cumulative product (future)
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for np.cumsum, np.cumprod
//   KEY MEMBERS:
//     - CumulativeKernel delegate (defined in ReductionKernel.cs)
//     - CumulativeKernelKey record struct (defined in ReductionKernel.cs)
//     - _scanCache - caches by CumulativeKernelKey
//     - GetCumulativeKernel(), TryGetCumulativeKernel()
//     - CumSumHelper<TIn, TOut>() - optimized same-type cumsum
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
    {
        #region Scan Kernel Generation

        /// <summary>
        /// Cache for cumulative (scan) kernels.
        /// Key: CumulativeKernelKey (InputType, OutputType, Op, IsContiguous)
        /// </summary>
        private static readonly ConcurrentDictionary<CumulativeKernelKey, Delegate> _scanCache = new();

        /// <summary>
        /// Number of scan kernels in cache.
        /// </summary>
        public static int ScanCachedCount => _scanCache.Count;

        /// <summary>
        /// Clear the scan kernel caches.
        /// </summary>
        public static void ClearScan()
        {
            _scanCache.Clear();
        }

        /// <summary>
        /// Get or generate a cumulative (scan) kernel.
        /// Returns a delegate that computes running accumulation over all elements.
        /// </summary>
        public static CumulativeKernel GetCumulativeKernel(CumulativeKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            var kernel = _scanCache.GetOrAdd(key, GenerateCumulativeKernel);
            return (CumulativeKernel)kernel;
        }

        /// <summary>
        /// Try to get or generate a cumulative kernel.
        /// </summary>
        public static CumulativeKernel? TryGetCumulativeKernel(CumulativeKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                var kernel = _scanCache.GetOrAdd(key, GenerateCumulativeKernel);
                return (CumulativeKernel)kernel;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Generate a cumulative (scan) kernel.
        /// </summary>
        private static Delegate GenerateCumulativeKernel(CumulativeKernelKey key)
        {
            // CumulativeKernel signature:
            // void(void* input, void* output, int* strides, int* shape, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"Scan_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*),  // input
                    typeof(void*),  // output
                    typeof(int*),   // strides
                    typeof(int*),   // shape
                    typeof(int),    // ndim
                    typeof(int)     // totalSize
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int inputSize = GetTypeSize(key.InputType);
            int outputSize = GetTypeSize(key.OutputType);

            if (key.IsContiguous && key.IsSameType)
            {
                // Fast path: contiguous array with same input/output type
                // Use helper method for cleaner implementation
                EmitScanHelperCall(il, key);
            }
            else if (key.IsContiguous)
            {
                // Contiguous but needs type conversion
                EmitScanContiguousWithConversion(il, key, inputSize, outputSize);
            }
            else
            {
                // Non-contiguous: use coordinate-based iteration
                EmitScanStridedLoop(il, key, inputSize, outputSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<CumulativeKernel>();
        }

        /// <summary>
        /// Emit a call to the appropriate scan helper method.
        /// Used for contiguous same-type arrays.
        /// </summary>
        private static void EmitScanHelperCall(ILGenerator il, CumulativeKernelKey key)
        {
            // Find the appropriate helper method based on operation
            string helperName = key.Op switch
            {
                ReductionOp.CumSum => nameof(CumSumHelperSameType),
                // ReductionOp.CumProd => nameof(CumProdHelperSameType), // Future
                _ => throw new NotSupportedException($"Scan operation {key.Op} not supported")
            };

            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                helperName,
                BindingFlags.NonPublic | BindingFlags.Static);

            var genericHelper = helperMethod!.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: CumSumHelperSameType<T>(input, output, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.EmitCall(OpCodes.Call, genericHelper, null);
        }

        /// <summary>
        /// SIMD-optimized cumulative sum helper for same-type contiguous arrays.
        /// While scan is inherently sequential, we optimize memory access patterns.
        /// </summary>
        internal static unsafe void CumSumHelperSameType<T>(void* input, void* output, int totalSize)
            where T : unmanaged, IAdditionOperators<T, T, T>
        {
            if (totalSize == 0)
                return;

            T* src = (T*)input;
            T* dst = (T*)output;

            // Scan is inherently sequential - each output depends on previous sum
            // We use direct pointer access for optimal performance
            T sum = default;
            for (int i = 0; i < totalSize; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Emit contiguous scan loop with type conversion.
        /// </summary>
        private static void EmitScanContiguousWithConversion(ILGenerator il, CumulativeKernelKey key, int inputSize, int outputSize)
        {
            // Args: void* input (0), void* output (1), int* strides (2), int* shape (3), int ndim (4), int totalSize (5)

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locAccum = il.DeclareLocal(GetClrType(key.OutputType)); // accumulator

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Initialize accumulator to 0 (identity for addition)
            EmitLoadZero(il, key.OutputType);
            il.Emit(OpCodes.Stloc, locAccum);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Load input[i], convert to output type, add to accumulator
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Add to accumulator
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitScanCombine(il, key.Op, key.OutputType);
            il.Emit(OpCodes.Stloc, locAccum);

            // Store accumulator to output[i]
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit a strided scan loop for non-contiguous arrays.
        /// </summary>
        private static void EmitScanStridedLoop(ILGenerator il, CumulativeKernelKey key, int inputSize, int outputSize)
        {
            // Args: void* input (0), void* output (1), int* strides (2), int* shape (3), int ndim (4), int totalSize (5)

            var locI = il.DeclareLocal(typeof(int)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locOffset = il.DeclareLocal(typeof(int)); // input offset
            var locCoord = il.DeclareLocal(typeof(int)); // current coordinate
            var locIdx = il.DeclareLocal(typeof(int)); // temp for coordinate calculation
            var locAccum = il.DeclareLocal(GetClrType(key.OutputType)); // accumulator

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblDimLoop = il.DefineLabel();
            var lblDimLoopEnd = il.DefineLabel();

            // Initialize accumulator
            EmitLoadZero(il, key.OutputType);
            il.Emit(OpCodes.Stloc, locAccum);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // Main loop
            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate offset from linear index
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locOffset);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            // d = ndim - 1
            il.Emit(OpCodes.Ldarg_S, (byte)4); // ndim
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.MarkLabel(lblDimLoop);

            // if (d < 0) goto DimLoopEnd
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Blt, lblDimLoopEnd);

            // coord = idx % shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_3); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locCoord);

            // idx /= shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_3); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locIdx);

            // offset += coord * strides[d]
            il.Emit(OpCodes.Ldloc, locOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_2); // strides
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOffset);

            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.Emit(OpCodes.Br, lblDimLoop);
            il.MarkLabel(lblDimLoopEnd);

            // Load input[offset]
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Add to accumulator
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitScanCombine(il, key.Op, key.OutputType);
            il.Emit(OpCodes.Stloc, locAccum);

            // Store accumulator to output[i] (output is always contiguous)
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit scan combine operation.
        /// Stack has [value, accumulator], result is combined value.
        /// </summary>
        private static void EmitScanCombine(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            // Special handling for decimal
            if (type == NPTypeCode.Decimal)
            {
                var method = op switch
                {
                    ReductionOp.CumSum => typeof(decimal).GetMethod("op_Addition",
                        BindingFlags.Public | BindingFlags.Static,
                        null, new[] { typeof(decimal), typeof(decimal) }, null),
                    // ReductionOp.CumProd => typeof(decimal).GetMethod("op_Multiply", ...),
                    _ => throw new NotSupportedException($"Scan operation {op} not supported for decimal")
                };
                il.EmitCall(OpCodes.Call, method!, null);
                return;
            }

            switch (op)
            {
                case ReductionOp.CumSum:
                    il.Emit(OpCodes.Add);
                    break;
                // case ReductionOp.CumProd:
                //     il.Emit(OpCodes.Mul);
                //     break;
                default:
                    throw new NotSupportedException($"Scan operation {op} not supported");
            }
        }

        // Note: EmitLoadZero is defined in ILKernelGenerator.Reduction.cs

        #endregion

        #region Public SIMD Helpers for Direct Calls

        /// <summary>
        /// SIMD-optimized cumulative sum for contiguous arrays with type conversion.
        /// Called directly by DefaultEngine for the fast path.
        /// </summary>
        /// <typeparam name="TIn">Input element type</typeparam>
        /// <typeparam name="TOut">Output element type</typeparam>
        /// <param name="input">Pointer to input data</param>
        /// <param name="output">Pointer to output data</param>
        /// <param name="totalSize">Number of elements</param>
        public static unsafe void CumSumHelper<TIn, TOut>(void* input, void* output, int totalSize)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            if (totalSize == 0)
                return;

            // Dispatch based on types for optimal performance
            // Most common paths first
            if (typeof(TIn) == typeof(TOut))
            {
                // Same type - use generic math
                CumSumSameTypeDispatch<TIn>(input, output, totalSize);
            }
            else
            {
                // Type conversion required - use scalar loop with conversion
                CumSumWithConversion<TIn, TOut>(input, output, totalSize);
            }
        }

        /// <summary>
        /// Dispatch same-type cumsum to appropriate implementation.
        /// </summary>
        private static unsafe void CumSumSameTypeDispatch<T>(void* input, void* output, int totalSize)
            where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                CumSumFloat((float*)input, (float*)output, totalSize);
            }
            else if (typeof(T) == typeof(double))
            {
                CumSumDouble((double*)input, (double*)output, totalSize);
            }
            else if (typeof(T) == typeof(int))
            {
                CumSumInt32((int*)input, (int*)output, totalSize);
            }
            else if (typeof(T) == typeof(long))
            {
                CumSumInt64((long*)input, (long*)output, totalSize);
            }
            else if (typeof(T) == typeof(byte))
            {
                CumSumByte((byte*)input, (byte*)output, totalSize);
            }
            else if (typeof(T) == typeof(short))
            {
                CumSumInt16((short*)input, (short*)output, totalSize);
            }
            else if (typeof(T) == typeof(uint))
            {
                CumSumUInt32((uint*)input, (uint*)output, totalSize);
            }
            else if (typeof(T) == typeof(ulong))
            {
                CumSumUInt64((ulong*)input, (ulong*)output, totalSize);
            }
            else if (typeof(T) == typeof(ushort))
            {
                CumSumUInt16((ushort*)input, (ushort*)output, totalSize);
            }
            else if (typeof(T) == typeof(decimal))
            {
                CumSumDecimal((decimal*)input, (decimal*)output, totalSize);
            }
            else
            {
                // Fallback for char and other types
                CumSumGeneric<T>((T*)input, (T*)output, totalSize);
            }
        }

        /// <summary>
        /// Generic cumsum using IAdditionOperators constraint.
        /// </summary>
        private static unsafe void CumSumGeneric<T>(T* src, T* dst, int size)
            where T : unmanaged
        {
            // Use dynamic for types that don't have IAdditionOperators
            dynamic sum = default(T)!;
            for (int i = 0; i < size; i++)
            {
                sum += (dynamic)src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for float arrays.
        /// </summary>
        private static unsafe void CumSumFloat(float* src, float* dst, int size)
        {
            float sum = 0f;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for double arrays.
        /// </summary>
        private static unsafe void CumSumDouble(double* src, double* dst, int size)
        {
            double sum = 0.0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for int arrays.
        /// </summary>
        private static unsafe void CumSumInt32(int* src, int* dst, int size)
        {
            int sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for long arrays.
        /// </summary>
        private static unsafe void CumSumInt64(long* src, long* dst, int size)
        {
            long sum = 0L;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for byte arrays.
        /// </summary>
        private static unsafe void CumSumByte(byte* src, byte* dst, int size)
        {
            byte sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for short arrays.
        /// </summary>
        private static unsafe void CumSumInt16(short* src, short* dst, int size)
        {
            short sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for uint arrays.
        /// </summary>
        private static unsafe void CumSumUInt32(uint* src, uint* dst, int size)
        {
            uint sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for ulong arrays.
        /// </summary>
        private static unsafe void CumSumUInt64(ulong* src, ulong* dst, int size)
        {
            ulong sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for ushort arrays.
        /// </summary>
        private static unsafe void CumSumUInt16(ushort* src, ushort* dst, int size)
        {
            ushort sum = 0;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Optimized cumsum for decimal arrays.
        /// </summary>
        private static unsafe void CumSumDecimal(decimal* src, decimal* dst, int size)
        {
            decimal sum = 0m;
            for (int i = 0; i < size; i++)
            {
                sum += src[i];
                dst[i] = sum;
            }
        }

        /// <summary>
        /// Cumsum with type conversion from TIn to TOut.
        /// </summary>
        private static unsafe void CumSumWithConversion<TIn, TOut>(void* input, void* output, int totalSize)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            // Use the most common widening scenarios with specialized code
            // TIn -> TOut type promotion scenarios

            // int32 -> int64 (most common for NumPy default behavior)
            if (typeof(TIn) == typeof(int) && typeof(TOut) == typeof(long))
            {
                int* src = (int*)input;
                long* dst = (long*)output;
                long sum = 0L;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += src[i];
                    dst[i] = sum;
                }
                return;
            }

            // float -> double
            if (typeof(TIn) == typeof(float) && typeof(TOut) == typeof(double))
            {
                float* src = (float*)input;
                double* dst = (double*)output;
                double sum = 0.0;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += src[i];
                    dst[i] = sum;
                }
                return;
            }

            // byte -> int64
            if (typeof(TIn) == typeof(byte) && typeof(TOut) == typeof(long))
            {
                byte* src = (byte*)input;
                long* dst = (long*)output;
                long sum = 0L;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += src[i];
                    dst[i] = sum;
                }
                return;
            }

            // short -> int64
            if (typeof(TIn) == typeof(short) && typeof(TOut) == typeof(long))
            {
                short* src = (short*)input;
                long* dst = (long*)output;
                long sum = 0L;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += src[i];
                    dst[i] = sum;
                }
                return;
            }

            // uint32 -> uint64
            if (typeof(TIn) == typeof(uint) && typeof(TOut) == typeof(ulong))
            {
                uint* src = (uint*)input;
                ulong* dst = (ulong*)output;
                ulong sum = 0UL;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += src[i];
                    dst[i] = sum;
                }
                return;
            }

            // General fallback using Convert
            CumSumWithConversionGeneral<TIn, TOut>(input, output, totalSize);
        }

        /// <summary>
        /// General cumsum with type conversion using Convert class.
        /// </summary>
        private static unsafe void CumSumWithConversionGeneral<TIn, TOut>(void* input, void* output, int totalSize)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            TIn* src = (TIn*)input;
            TOut* dst = (TOut*)output;

            // Use double as intermediate for most conversions
            if (typeof(TOut) == typeof(double))
            {
                double sum = 0.0;
                double* dstDouble = (double*)dst;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += Convert.ToDouble(src[i]);
                    dstDouble[i] = sum;
                }
            }
            else if (typeof(TOut) == typeof(long))
            {
                long sum = 0L;
                long* dstLong = (long*)dst;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += Convert.ToInt64(src[i]);
                    dstLong[i] = sum;
                }
            }
            else if (typeof(TOut) == typeof(decimal))
            {
                decimal sum = 0m;
                decimal* dstDecimal = (decimal*)dst;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += Convert.ToDecimal(src[i]);
                    dstDecimal[i] = sum;
                }
            }
            else
            {
                // Ultimate fallback using dynamic
                dynamic sum = default(TOut)!;
                for (int i = 0; i < totalSize; i++)
                {
                    sum += (dynamic)Convert.ChangeType(src[i], typeof(TOut))!;
                    dst[i] = sum;
                }
            }
        }

        #endregion
    }
}
