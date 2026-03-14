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

            // Initialize accumulator to identity value (0 for sum, 1 for prod)
            EmitScanIdentity(il, key.Op, key.OutputType);
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

            // Initialize accumulator to identity value (0 for sum, 1 for prod)
            EmitScanIdentity(il, key.Op, key.OutputType);
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

        /// <summary>
        /// Emit identity value for scan operations.
        /// CumSum uses 0 (additive identity), CumProd uses 1 (multiplicative identity).
        /// </summary>
        private static void EmitScanIdentity(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            switch (op)
            {
                case ReductionOp.CumSum:
                    EmitLoadZero(il, type);
                    break;
                case ReductionOp.CumProd:
                    EmitLoadOne(il, type);
                    break;
                default:
                    throw new NotSupportedException($"Scan operation {op} has no identity");
            }
        }

        // Note: EmitLoadZero and EmitLoadOne are defined in ILKernelGenerator.Reduction.cs

        #endregion

        #region Axis Scan Kernel Generation

        /// <summary>
        /// Cache for cumulative axis (scan along axis) kernels.
        /// Key: CumulativeAxisKernelKey (InputType, OutputType, Op, InnerAxisContiguous)
        /// </summary>
        private static readonly ConcurrentDictionary<CumulativeAxisKernelKey, Delegate> _axisScanCache = new();

        /// <summary>
        /// Number of axis scan kernels in cache.
        /// </summary>
        public static int AxisScanCachedCount => _axisScanCache.Count;

        /// <summary>
        /// Get or generate a cumulative axis (scan along axis) kernel.
        /// Returns a delegate that computes running accumulation along a specific axis.
        /// </summary>
        public static CumulativeAxisKernel GetCumulativeAxisKernel(CumulativeAxisKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            var kernel = _axisScanCache.GetOrAdd(key, GenerateCumulativeAxisKernel);
            return (CumulativeAxisKernel)kernel;
        }

        /// <summary>
        /// Try to get or generate a cumulative axis kernel.
        /// </summary>
        public static CumulativeAxisKernel? TryGetCumulativeAxisKernel(CumulativeAxisKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                var kernel = _axisScanCache.GetOrAdd(key, GenerateCumulativeAxisKernel);
                return (CumulativeAxisKernel)kernel;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Generate a cumulative axis (scan along axis) kernel.
        /// </summary>
        private static Delegate GenerateCumulativeAxisKernel(CumulativeAxisKernelKey key)
        {
            // CumulativeAxisKernel signature:
            // void(void* input, void* output, int* inputStrides, int* shape, int axis, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"AxisScan_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*),  // input
                    typeof(void*),  // output
                    typeof(int*),   // inputStrides
                    typeof(int*),   // shape
                    typeof(int),    // axis
                    typeof(int),    // ndim
                    typeof(int)     // totalSize
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // For axis cumsum, we call a helper method that handles the iteration
            // The IL kernel just dispatches to the right helper based on types
            EmitAxisScanHelperCall(il, key);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<CumulativeAxisKernel>();
        }

        /// <summary>
        /// Emit a call to the appropriate axis scan helper method.
        /// </summary>
        private static void EmitAxisScanHelperCall(ILGenerator il, CumulativeAxisKernelKey key)
        {
            // Find the appropriate helper method based on operation
            string helperName = key.Op switch
            {
                ReductionOp.CumSum => nameof(AxisCumSumHelper),
                // ReductionOp.CumProd => nameof(AxisCumProdHelper), // Future
                _ => throw new NotSupportedException($"Axis scan operation {key.Op} not supported")
            };

            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                helperName,
                BindingFlags.NonPublic | BindingFlags.Static);

            var genericHelper = helperMethod!.MakeGenericMethod(
                GetClrType(key.InputType),
                GetClrType(key.OutputType));

            // Call helper: AxisCumSumHelper<TIn, TOut>(input, output, inputStrides, shape, axis, ndim, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldarg_2); // inputStrides
            il.Emit(OpCodes.Ldarg_3); // shape
            il.Emit(OpCodes.Ldarg_S, (byte)4); // axis
            il.Emit(OpCodes.Ldarg_S, (byte)5); // ndim
            il.Emit(OpCodes.Ldarg_S, (byte)6); // totalSize
            il.EmitCall(OpCodes.Call, genericHelper, null);
        }

        /// <summary>
        /// Axis cumulative sum helper. Computes cumsum along a specific axis.
        /// Uses optimized iteration pattern based on axis position.
        /// </summary>
        internal static unsafe void AxisCumSumHelper<TIn, TOut>(
            void* input, void* output, int* inputStrides, int* shape,
            int axis, int ndim, int totalSize)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            if (totalSize == 0)
                return;

            TIn* src = (TIn*)input;
            TOut* dst = (TOut*)output;

            int axisSize = shape[axis];
            int axisStride = inputStrides[axis];

            // Calculate outer size (product of dimensions before axis)
            // and inner size (product of dimensions after axis)
            int outerSize = 1;
            int innerSize = 1;
            for (int d = 0; d < axis; d++)
                outerSize *= shape[d];
            for (int d = axis + 1; d < ndim; d++)
                innerSize *= shape[d];

            // Calculate output strides (output is always contiguous)
            int outputAxisStride = innerSize;
            int outputOuterStride = axisSize * innerSize;

            // Dispatch to specialized helper based on types
            if (typeof(TIn) == typeof(TOut))
            {
                // When TIn == TOut, we can cast safely and use same-type optimized path
                AxisCumSumSameType<TIn>((TIn*)src, (TIn*)(void*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
            else
            {
                AxisCumSumWithConversion<TIn, TOut>(src, dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
        }

        /// <summary>
        /// Axis cumulative product helper. Computes cumprod along a specific axis.
        /// Uses optimized iteration pattern based on axis position.
        /// </summary>
        internal static unsafe void AxisCumProdHelper<TIn, TOut>(
            void* input, void* output, int* inputStrides, int* shape,
            int axis, int ndim, int totalSize)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            if (totalSize == 0)
                return;

            TIn* src = (TIn*)input;
            TOut* dst = (TOut*)output;

            int axisSize = shape[axis];
            int axisStride = inputStrides[axis];

            // Calculate outer size (product of dimensions before axis)
            // and inner size (product of dimensions after axis)
            int outerSize = 1;
            int innerSize = 1;
            for (int d = 0; d < axis; d++)
                outerSize *= shape[d];
            for (int d = axis + 1; d < ndim; d++)
                innerSize *= shape[d];

            // Calculate output strides (output is always contiguous)
            int outputAxisStride = innerSize;
            int outputOuterStride = axisSize * innerSize;

            // Dispatch to specialized helper based on types
            if (typeof(TIn) == typeof(TOut))
            {
                // When TIn == TOut, we can cast safely and use same-type optimized path
                AxisCumProdSameType<TIn>((TIn*)src, (TIn*)(void*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
            else
            {
                AxisCumProdWithConversion<TIn, TOut>(src, dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
        }

        /// <summary>
        /// Same-type axis cumprod implementation.
        /// </summary>
        private static unsafe void AxisCumProdSameType<T>(
            T* src, T* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where T : unmanaged
        {
            // General case: iterate using coordinate-based access with multiplication
            AxisCumProdGeneral(src, dst, inputStrides, shape, axis, ndim,
                axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
        }

        /// <summary>
        /// General axis cumprod using coordinate-based iteration.
        /// </summary>
        private static unsafe void AxisCumProdGeneral<T>(
            T* src, T* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where T : unmanaged
        {
            // Type-specific dispatch for common types
            if (typeof(T) == typeof(double))
            {
                AxisCumProdGeneralDouble((double*)src, (double*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
            else if (typeof(T) == typeof(float))
            {
                AxisCumProdGeneralFloat((float*)src, (float*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
            else if (typeof(T) == typeof(long))
            {
                AxisCumProdGeneralInt64((long*)src, (long*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
            else if (typeof(T) == typeof(int))
            {
                AxisCumProdGeneralInt32((int*)src, (int*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
            else
            {
                // Generic fallback
                AxisCumProdGeneralGeneric(src, dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
            }
        }

        /// <summary>
        /// General axis cumprod for double type.
        /// </summary>
        private static unsafe void AxisCumProdGeneralDouble(
            double* src, double* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    // Calculate input base offset for this (outer, inner) combination
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    // Calculate output base offset
                    int outputOffset = outer * outputOuterStride + inner;

                    // Cumprod along axis
                    double product = 1.0;
                    for (int i = 0; i < axisSize; i++)
                    {
                        product *= src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = product;
                    }
                }
            }
        }

        /// <summary>
        /// General axis cumprod for float type.
        /// </summary>
        private static unsafe void AxisCumProdGeneralFloat(
            float* src, float* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    float product = 1f;
                    for (int i = 0; i < axisSize; i++)
                    {
                        product *= src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = product;
                    }
                }
            }
        }

        /// <summary>
        /// General axis cumprod for long type.
        /// </summary>
        private static unsafe void AxisCumProdGeneralInt64(
            long* src, long* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    long product = 1L;
                    for (int i = 0; i < axisSize; i++)
                    {
                        product *= src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = product;
                    }
                }
            }
        }

        /// <summary>
        /// General axis cumprod for int type.
        /// </summary>
        private static unsafe void AxisCumProdGeneralInt32(
            int* src, int* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    int product = 1;
                    for (int i = 0; i < axisSize; i++)
                    {
                        product *= src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = product;
                    }
                }
            }
        }

        /// <summary>
        /// Generic fallback for axis cumprod.
        /// </summary>
        private static unsafe void AxisCumProdGeneralGeneric<T>(
            T* src, T* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where T : unmanaged
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    // Initialize product to 1 using dynamic
                    dynamic product = (dynamic)Convert.ChangeType(1, typeof(T))!;
                    for (int i = 0; i < axisSize; i++)
                    {
                        product *= (dynamic)src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = product;
                    }
                }
            }
        }

        /// <summary>
        /// Axis cumprod with type conversion (e.g., int32 input -> int64 output).
        /// </summary>
        private static unsafe void AxisCumProdWithConversion<TIn, TOut>(
            TIn* src, TOut* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            // Common case: int32 -> int64
            if (typeof(TIn) == typeof(int) && typeof(TOut) == typeof(long))
            {
                AxisCumProdInt32ToInt64((int*)src, (long*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
                return;
            }

            // General fallback using Convert
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    // Use appropriate accumulator type
                    if (typeof(TOut) == typeof(long))
                    {
                        long product = 1L;
                        long* dstTyped = (long*)dst;
                        for (int i = 0; i < axisSize; i++)
                        {
                            product *= Convert.ToInt64(src[inputOffset + i * axisStride]);
                            dstTyped[outputOffset + i * outputAxisStride] = product;
                        }
                    }
                    else if (typeof(TOut) == typeof(double))
                    {
                        double product = 1.0;
                        double* dstTyped = (double*)dst;
                        for (int i = 0; i < axisSize; i++)
                        {
                            product *= Convert.ToDouble(src[inputOffset + i * axisStride]);
                            dstTyped[outputOffset + i * outputAxisStride] = product;
                        }
                    }
                    else
                    {
                        // Ultimate fallback using dynamic
                        dynamic product = (dynamic)Convert.ChangeType(1, typeof(TOut))!;
                        for (int i = 0; i < axisSize; i++)
                        {
                            product *= (dynamic)Convert.ChangeType(src[inputOffset + i * axisStride], typeof(TOut))!;
                            dst[outputOffset + i * outputAxisStride] = product;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Specialized int32 -> int64 axis cumprod.
        /// </summary>
        private static unsafe void AxisCumProdInt32ToInt64(
            int* src, long* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    long product = 1L;
                    for (int i = 0; i < axisSize; i++)
                    {
                        product *= src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = product;
                    }
                }
            }
        }

        /// <summary>
        /// Same-type axis cumsum implementation.
        /// </summary>
        private static unsafe void AxisCumSumSameType<T>(
            T* src, T* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where T : unmanaged
        {
            // Special case: innermost axis (axis = ndim - 1)
            // This is the most common case and can be optimized
            if (axis == ndim - 1 && axisStride == 1)
            {
                AxisCumSumInnerContiguous(src, dst, inputStrides, shape, ndim,
                    axisSize, outerSize, outputOuterStride);
                return;
            }

            // General case: iterate using coordinate-based access
            AxisCumSumGeneral(src, dst, inputStrides, shape, axis, ndim,
                axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
        }

        /// <summary>
        /// Optimized cumsum for innermost contiguous axis.
        /// Each "row" is a contiguous block that we can scan directly.
        /// </summary>
        private static unsafe void AxisCumSumInnerContiguous<T>(
            T* src, T* dst, int* inputStrides, int* shape, int ndim,
            int axisSize, int outerSize, int outputOuterStride)
            where T : unmanaged
        {
            // For innermost axis with stride=1, we can process each row directly
            // Calculate input row stride (total stride for incrementing outer dimensions)
            int inputRowStride = inputStrides[ndim - 2 >= 0 ? ndim - 2 : 0];
            if (ndim == 1) inputRowStride = axisSize;

            // Dispatch to type-specific implementation for best performance
            if (typeof(T) == typeof(double))
            {
                AxisCumSumInnerContiguousDouble((double*)src, (double*)dst, inputRowStride, axisSize, outerSize, outputOuterStride);
            }
            else if (typeof(T) == typeof(float))
            {
                AxisCumSumInnerContiguousFloat((float*)src, (float*)dst, inputRowStride, axisSize, outerSize, outputOuterStride);
            }
            else if (typeof(T) == typeof(long))
            {
                AxisCumSumInnerContiguousInt64((long*)src, (long*)dst, inputRowStride, axisSize, outerSize, outputOuterStride);
            }
            else if (typeof(T) == typeof(int))
            {
                AxisCumSumInnerContiguousInt32((int*)src, (int*)dst, inputRowStride, axisSize, outerSize, outputOuterStride);
            }
            else
            {
                // Generic fallback using dynamic
                for (int outer = 0; outer < outerSize; outer++)
                {
                    T* srcRow = src + outer * inputRowStride;
                    T* dstRow = dst + outer * outputOuterStride;

                    dynamic sum = default(T)!;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += (dynamic)srcRow[i];
                        dstRow[i] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// Type-specific inner contiguous cumsum for double.
        /// </summary>
        private static unsafe void AxisCumSumInnerContiguousDouble(
            double* src, double* dst, int inputRowStride, int axisSize, int outerSize, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                double* srcRow = src + outer * inputRowStride;
                double* dstRow = dst + outer * outputOuterStride;

                double sum = 0.0;
                for (int i = 0; i < axisSize; i++)
                {
                    sum += srcRow[i];
                    dstRow[i] = sum;
                }
            }
        }

        /// <summary>
        /// Type-specific inner contiguous cumsum for float.
        /// </summary>
        private static unsafe void AxisCumSumInnerContiguousFloat(
            float* src, float* dst, int inputRowStride, int axisSize, int outerSize, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                float* srcRow = src + outer * inputRowStride;
                float* dstRow = dst + outer * outputOuterStride;

                float sum = 0f;
                for (int i = 0; i < axisSize; i++)
                {
                    sum += srcRow[i];
                    dstRow[i] = sum;
                }
            }
        }

        /// <summary>
        /// Type-specific inner contiguous cumsum for long.
        /// </summary>
        private static unsafe void AxisCumSumInnerContiguousInt64(
            long* src, long* dst, int inputRowStride, int axisSize, int outerSize, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                long* srcRow = src + outer * inputRowStride;
                long* dstRow = dst + outer * outputOuterStride;

                long sum = 0L;
                for (int i = 0; i < axisSize; i++)
                {
                    sum += srcRow[i];
                    dstRow[i] = sum;
                }
            }
        }

        /// <summary>
        /// Type-specific inner contiguous cumsum for int.
        /// </summary>
        private static unsafe void AxisCumSumInnerContiguousInt32(
            int* src, int* dst, int inputRowStride, int axisSize, int outerSize, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                int* srcRow = src + outer * inputRowStride;
                int* dstRow = dst + outer * outputOuterStride;

                int sum = 0;
                for (int i = 0; i < axisSize; i++)
                {
                    sum += srcRow[i];
                    dstRow[i] = sum;
                }
            }
        }

        /// <summary>
        /// General axis cumsum using coordinate-based iteration.
        /// Handles non-contiguous axes and complex stride patterns.
        /// </summary>
        private static unsafe void AxisCumSumGeneral<T>(
            T* src, T* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where T : unmanaged
        {
            // For each combination of outer and inner indices, compute cumsum along axis
            // Output is always contiguous, input may be strided

            // Precompute inner and outer strides for input
            int* outerStrides = stackalloc int[ndim];
            int* innerStrides = stackalloc int[ndim];

            int outerStride = 1;
            int innerStride = 1;

            // Outer dimensions: 0 to axis-1
            for (int d = axis - 1; d >= 0; d--)
            {
                outerStrides[d] = outerStride;
                outerStride *= shape[d];
            }

            // Inner dimensions: axis+1 to ndim-1
            for (int d = ndim - 1; d > axis; d--)
            {
                innerStrides[d] = innerStride;
                innerStride *= shape[d];
            }

            // Type-specific dispatch for common types
            if (typeof(T) == typeof(double))
            {
                AxisCumSumGeneralDouble((double*)src, (double*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride,
                    outerStrides, innerStrides);
            }
            else if (typeof(T) == typeof(float))
            {
                AxisCumSumGeneralFloat((float*)src, (float*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride,
                    outerStrides, innerStrides);
            }
            else if (typeof(T) == typeof(long))
            {
                AxisCumSumGeneralInt64((long*)src, (long*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride,
                    outerStrides, innerStrides);
            }
            else if (typeof(T) == typeof(int))
            {
                AxisCumSumGeneralInt32((int*)src, (int*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride,
                    outerStrides, innerStrides);
            }
            else
            {
                // Generic fallback
                AxisCumSumGeneralGeneric(src, dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride,
                    outerStrides, innerStrides);
            }
        }

        /// <summary>
        /// General axis cumsum for double type.
        /// </summary>
        private static unsafe void AxisCumSumGeneralDouble(
            double* src, double* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride, int* outerStrides, int* innerStrides)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    // Calculate input base offset for this (outer, inner) combination
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    // Calculate output base offset
                    int outputOffset = outer * outputOuterStride + inner;

                    // Cumsum along axis
                    double sum = 0.0;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// General axis cumsum for float type.
        /// </summary>
        private static unsafe void AxisCumSumGeneralFloat(
            float* src, float* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride, int* outerStrides, int* innerStrides)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    float sum = 0f;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// General axis cumsum for long type.
        /// </summary>
        private static unsafe void AxisCumSumGeneralInt64(
            long* src, long* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride, int* outerStrides, int* innerStrides)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    long sum = 0L;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// General axis cumsum for int type.
        /// </summary>
        private static unsafe void AxisCumSumGeneralInt32(
            int* src, int* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride, int* outerStrides, int* innerStrides)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    int sum = 0;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// Generic fallback for axis cumsum.
        /// </summary>
        private static unsafe void AxisCumSumGeneralGeneric<T>(
            T* src, T* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride, int* outerStrides, int* innerStrides)
            where T : unmanaged
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    dynamic sum = default(T)!;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += (dynamic)src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// Axis cumsum with type conversion (e.g., int32 input -> int64 output).
        /// </summary>
        private static unsafe void AxisCumSumWithConversion<TIn, TOut>(
            TIn* src, TOut* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
            where TIn : unmanaged
            where TOut : unmanaged
        {
            // Common case: int32 -> int64
            if (typeof(TIn) == typeof(int) && typeof(TOut) == typeof(long))
            {
                AxisCumSumInt32ToInt64((int*)src, (long*)dst, inputStrides, shape, axis, ndim,
                    axisSize, axisStride, outerSize, innerSize, outputAxisStride, outputOuterStride);
                return;
            }

            // General fallback using Convert
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    // Use appropriate accumulator type
                    if (typeof(TOut) == typeof(long))
                    {
                        long sum = 0L;
                        long* dstTyped = (long*)dst;
                        for (int i = 0; i < axisSize; i++)
                        {
                            sum += Convert.ToInt64(src[inputOffset + i * axisStride]);
                            dstTyped[outputOffset + i * outputAxisStride] = sum;
                        }
                    }
                    else if (typeof(TOut) == typeof(double))
                    {
                        double sum = 0.0;
                        double* dstTyped = (double*)dst;
                        for (int i = 0; i < axisSize; i++)
                        {
                            sum += Convert.ToDouble(src[inputOffset + i * axisStride]);
                            dstTyped[outputOffset + i * outputAxisStride] = sum;
                        }
                    }
                    else
                    {
                        // Ultimate fallback using dynamic
                        dynamic sum = default(TOut)!;
                        for (int i = 0; i < axisSize; i++)
                        {
                            sum += (dynamic)Convert.ChangeType(src[inputOffset + i * axisStride], typeof(TOut))!;
                            dst[outputOffset + i * outputAxisStride] = sum;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Specialized int32 -> int64 axis cumsum.
        /// </summary>
        private static unsafe void AxisCumSumInt32ToInt64(
            int* src, long* dst, int* inputStrides, int* shape, int axis, int ndim,
            int axisSize, int axisStride, int outerSize, int innerSize,
            int outputAxisStride, int outputOuterStride)
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int inputOffset = 0;
                    int outerIdx = outer;
                    for (int d = axis - 1; d >= 0; d--)
                    {
                        int coord = outerIdx % shape[d];
                        outerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int innerIdx = inner;
                    for (int d = ndim - 1; d > axis; d--)
                    {
                        int coord = innerIdx % shape[d];
                        innerIdx /= shape[d];
                        inputOffset += coord * inputStrides[d];
                    }

                    int outputOffset = outer * outputOuterStride + inner;

                    long sum = 0L;
                    for (int i = 0; i < axisSize; i++)
                    {
                        sum += src[inputOffset + i * axisStride];
                        dst[outputOffset + i * outputAxisStride] = sum;
                    }
                }
            }
        }

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
