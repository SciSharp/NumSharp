using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Reduction Kernel Generation

        /// <summary>
        /// Cache for element-wise reduction kernels.
        /// Key: ElementReductionKernelKey
        /// </summary>
        private static readonly ConcurrentDictionary<ElementReductionKernelKey, Delegate> _elementReductionCache = new();

        /// <summary>
        /// Number of element reduction kernels in cache.
        /// </summary>
        public static int ElementReductionCachedCount => _elementReductionCache.Count;

        /// <summary>
        /// Clear the reduction kernel caches.
        /// </summary>
        public static void ClearReduction()
        {
            _elementReductionCache.Clear();
        }

        /// <summary>
        /// Get or generate a typed element-wise reduction kernel.
        /// Returns a delegate that reduces all elements to a single value of type TResult.
        /// </summary>
        public static TypedElementReductionKernel<TResult> GetTypedElementReductionKernel<TResult>(ElementReductionKernelKey key)
            where TResult : unmanaged
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            var kernel = _elementReductionCache.GetOrAdd(key, GenerateTypedElementReductionKernel<TResult>);
            return (TypedElementReductionKernel<TResult>)kernel;
        }

        /// <summary>
        /// Try to get or generate an element reduction kernel.
        /// </summary>
        public static TypedElementReductionKernel<TResult>? TryGetTypedElementReductionKernel<TResult>(ElementReductionKernelKey key)
            where TResult : unmanaged
        {
            if (!Enabled)
                return null;

            try
            {
                var kernel = _elementReductionCache.GetOrAdd(key, GenerateTypedElementReductionKernel<TResult>);
                return (TypedElementReductionKernel<TResult>)kernel;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Generate a typed element-wise reduction kernel.
        /// </summary>
        private static Delegate GenerateTypedElementReductionKernel<TResult>(ElementReductionKernelKey key)
            where TResult : unmanaged
        {
            // TypedElementReductionKernel<TResult> signature:
            // TResult(void* input, int* strides, int* shape, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"ElemReduce_{key}",
                returnType: typeof(TResult),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(int*), typeof(int*), typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int inputSize = GetTypeSize(key.InputType);
            int accumSize = GetTypeSize(key.AccumulatorType);

            if (key.IsContiguous)
            {
                // Check if we can use SIMD
                bool canSimd = CanUseReductionSimd(key);
                if (canSimd)
                {
                    EmitReductionSimdLoop(il, key, inputSize);
                }
                else
                {
                    EmitReductionScalarLoop(il, key, inputSize);
                }
            }
            else
            {
                EmitReductionStridedLoop(il, key, inputSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<TypedElementReductionKernel<TResult>>();
        }

        /// <summary>
        /// Check if SIMD can be used for this reduction operation.
        /// </summary>
        private static bool CanUseReductionSimd(ElementReductionKernelKey key)
        {
            // Must be contiguous
            if (!key.IsContiguous)
                return false;

            // SIMD for numeric types (not bool, char, decimal)
            if (!CanUseSimd(key.InputType))
                return false;

            // Only certain operations have SIMD support
            // Sum: Vector.Sum() or manual horizontal add
            // Max/Min: Reduce vector then scalar reduce remainder
            // Prod: Manual horizontal multiply
            // ArgMax/ArgMin: Need to track indices, more complex
            return key.Op == ReductionOp.Sum || key.Op == ReductionOp.Max || key.Op == ReductionOp.Min ||
                   key.Op == ReductionOp.Prod;
        }

        /// <summary>
        /// Emit a SIMD reduction loop for contiguous arrays.
        /// Uses Vector256 for horizontal reductions.
        /// </summary>
        private static void EmitReductionSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            int vectorCount = GetVectorCount(key.InputType);

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int)); // totalSize - vectorCount
            var locAccum = il.DeclareLocal(GetClrType(key.AccumulatorType)); // scalar accumulator

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // Initialize accumulator with identity value
            EmitLoadIdentity(il, key.Op, key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locAccum);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // === SIMD LOOP ===
            il.MarkLabel(lblSimdLoop);

            // if (i > vectorEnd) goto SimdLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

            // Load vector from input[i]
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);

            // Perform horizontal reduction on vector and combine with accumulator
            EmitVectorHorizontalReduction(il, key.Op, key.InputType);

            // Combine with accumulator
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitReductionCombine(il, key.Op, key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locAccum);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === TAIL LOOP ===
            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // Load input[i], convert to accumulator type
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.AccumulatorType);

            // Combine with accumulator
            il.Emit(OpCodes.Ldloc, locAccum);
            EmitReductionCombine(il, key.Op, key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locAccum);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            // Return accumulator
            il.Emit(OpCodes.Ldloc, locAccum);
        }

        /// <summary>
        /// Emit a scalar reduction loop for contiguous arrays (no SIMD).
        /// </summary>
        private static void EmitReductionScalarLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Args: void* input (0), int* strides (1), int* shape (2), int ndim (3), int totalSize (4)

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locAccum = il.DeclareLocal(GetClrType(key.AccumulatorType)); // accumulator
            var locIdx = il.DeclareLocal(typeof(int)); // index for ArgMax/ArgMin

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Initialize accumulator with identity value
            EmitLoadIdentity(il, key.Op, key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locAccum);

            // For ArgMax/ArgMin, initialize index to 0
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, locIdx);
            }

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Load input[i], convert to accumulator type
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.AccumulatorType);

            // Combine with accumulator (and track index for ArgMax/ArgMin)
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                EmitArgReductionStep(il, key.Op, key.AccumulatorType, locAccum, locIdx, locI);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locAccum);
                EmitReductionCombine(il, key.Op, key.AccumulatorType);
                il.Emit(OpCodes.Stloc, locAccum);
            }

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            // Return accumulator or index
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                il.Emit(OpCodes.Ldloc, locIdx);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locAccum);
            }
        }

        /// <summary>
        /// Emit a strided reduction loop for non-contiguous arrays.
        /// </summary>
        private static void EmitReductionStridedLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Args: void* input (0), int* strides (1), int* shape (2), int ndim (3), int totalSize (4)

            var locI = il.DeclareLocal(typeof(int)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locOffset = il.DeclareLocal(typeof(int)); // input offset
            var locCoord = il.DeclareLocal(typeof(int)); // current coordinate
            var locIdx = il.DeclareLocal(typeof(int)); // temp for coordinate calculation
            var locAccum = il.DeclareLocal(GetClrType(key.AccumulatorType)); // accumulator
            var locArgIdx = il.DeclareLocal(typeof(int)); // index for ArgMax/ArgMin

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblDimLoop = il.DefineLabel();
            var lblDimLoopEnd = il.DefineLabel();

            // Initialize accumulator
            EmitLoadIdentity(il, key.Op, key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locAccum);

            // For ArgMax/ArgMin, initialize index to 0
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, locArgIdx);
            }

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // Main loop
            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate offset from linear index
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locOffset);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            // d = ndim - 1
            il.Emit(OpCodes.Ldarg_3); // ndim
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
            il.Emit(OpCodes.Ldarg_2); // shape
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
            il.Emit(OpCodes.Ldarg_2); // shape
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
            il.Emit(OpCodes.Ldarg_1); // strides
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
            EmitConvertTo(il, key.InputType, key.AccumulatorType);

            // Combine with accumulator
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                EmitArgReductionStep(il, key.Op, key.AccumulatorType, locAccum, locArgIdx, locI);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locAccum);
                EmitReductionCombine(il, key.Op, key.AccumulatorType);
                il.Emit(OpCodes.Stloc, locAccum);
            }

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            // Return accumulator or index
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                il.Emit(OpCodes.Ldloc, locArgIdx);
            }
            else
            {
                il.Emit(OpCodes.Ldloc, locAccum);
            }
        }

        #region Reduction IL Helpers

        /// <summary>
        /// Load the identity value for a reduction operation.
        /// </summary>
        private static void EmitLoadIdentity(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            switch (op)
            {
                case ReductionOp.Sum:
                case ReductionOp.Mean:
                case ReductionOp.CumSum:
                    // Identity is 0
                    EmitLoadZero(il, type);
                    break;

                case ReductionOp.Prod:
                    // Identity is 1
                    EmitLoadOne(il, type);
                    break;

                case ReductionOp.Max:
                    // Identity is minimum value (so first element becomes max)
                    EmitLoadMinValue(il, type);
                    break;

                case ReductionOp.Min:
                    // Identity is maximum value (so first element becomes min)
                    EmitLoadMaxValue(il, type);
                    break;

                case ReductionOp.ArgMax:
                case ReductionOp.ArgMin:
                    // For ArgMax/ArgMin, accumulator holds current best value
                    // Initialize with first element value (handled separately)
                    if (op == ReductionOp.ArgMax)
                        EmitLoadMinValue(il, type);
                    else
                        EmitLoadMaxValue(il, type);
                    break;

                default:
                    throw new NotSupportedException($"Identity for {op} not supported");
            }
        }

        /// <summary>
        /// Load zero for a type.
        /// </summary>
        private static void EmitLoadZero(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Ldc_I4_0);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, 0f);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, 0d);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldsfld, typeof(decimal).GetField("Zero")!);
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported");
            }
        }

        /// <summary>
        /// Load one for a type.
        /// </summary>
        private static void EmitLoadOne(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Ldc_I4_1);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldc_I8, 1L);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, 1f);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, 1d);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldsfld, typeof(decimal).GetField("One")!);
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported");
            }
        }

        /// <summary>
        /// Load minimum value for a type.
        /// </summary>
        private static void EmitLoadMinValue(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Ldc_I4, (int)byte.MinValue);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Ldc_I4, (int)short.MinValue);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Ldc_I4, (int)ushort.MinValue);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Ldc_I4, int.MinValue);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Ldc_I4, unchecked((int)uint.MinValue));
                    break;
                case NPTypeCode.Int64:
                    il.Emit(OpCodes.Ldc_I8, long.MinValue);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldc_I8, unchecked((long)ulong.MinValue));
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, float.NegativeInfinity);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, double.NegativeInfinity);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldsfld, typeof(decimal).GetField("MinValue")!);
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported");
            }
        }

        /// <summary>
        /// Load maximum value for a type.
        /// </summary>
        private static void EmitLoadMaxValue(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Ldc_I4, (int)byte.MaxValue);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Ldc_I4, (int)short.MaxValue);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Ldc_I4, (int)ushort.MaxValue);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Ldc_I4, int.MaxValue);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Ldc_I4, unchecked((int)uint.MaxValue));
                    break;
                case NPTypeCode.Int64:
                    il.Emit(OpCodes.Ldc_I8, long.MaxValue);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldc_I8, unchecked((long)ulong.MaxValue));
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, float.PositiveInfinity);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, double.PositiveInfinity);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldsfld, typeof(decimal).GetField("MaxValue")!);
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported");
            }
        }

        /// <summary>
        /// Emit horizontal reduction of a Vector (adapts to V128/V256/V512).
        /// Stack has Vector, result is scalar reduction.
        /// </summary>
        private static void EmitVectorHorizontalReduction(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            switch (op)
            {
                case ReductionOp.Sum:
                    // Use Vector.Sum<T>()
                    var sumMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                        .Where(m => m.Name == "Sum" && m.IsGenericMethod && m.GetParameters().Length == 1)
                        .Select(m => m.MakeGenericMethod(clrType))
                        .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType);

                    if (sumMethod != null)
                    {
                        il.EmitCall(OpCodes.Call, sumMethod, null);
                    }
                    else
                    {
                        // Fallback: manual horizontal add using GetElement
                        EmitManualHorizontalSum(il, type);
                    }
                    break;

                case ReductionOp.Max:
                case ReductionOp.Min:
                    // No built-in horizontal max/min, need to reduce manually
                    EmitManualHorizontalMinMax(il, op, type);
                    break;

                case ReductionOp.Prod:
                    // Manual horizontal multiply
                    EmitManualHorizontalProd(il, type);
                    break;

                default:
                    throw new NotSupportedException($"SIMD horizontal reduction for {op} not supported");
            }
        }

        /// <summary>
        /// Emit manual horizontal sum using tree reduction (O(log N) instead of O(N)).
        /// Uses GetLower/GetUpper + Add to reduce vector width by half each step.
        /// </summary>
        private static void EmitManualHorizontalSum(ILGenerator il, NPTypeCode type)
        {
            var clrType = GetClrType(type);

            // Tree reduction: reduce vector width by half each iteration
            // Vector512 -> Vector256 -> Vector128 -> scalar
            EmitTreeReduction(il, type, ReductionOp.Sum);
        }

        /// <summary>
        /// Emit manual horizontal min/max using tree reduction (O(log N) instead of O(N)).
        /// Uses GetLower/GetUpper + Min/Max to reduce vector width by half each step.
        /// </summary>
        private static void EmitManualHorizontalMinMax(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            // Tree reduction: reduce vector width by half each iteration
            EmitTreeReduction(il, type, op);
        }

        /// <summary>
        /// Emit manual horizontal product using tree reduction (O(log N) instead of O(N)).
        /// Uses GetLower/GetUpper + Multiply to reduce vector width by half each step.
        /// </summary>
        private static void EmitManualHorizontalProd(ILGenerator il, NPTypeCode type)
        {
            // Tree reduction: reduce vector width by half each iteration
            EmitTreeReduction(il, type, ReductionOp.Prod);
        }

        /// <summary>
        /// Get the Math.Max or Math.Min method for a type.
        /// </summary>
        private static MethodInfo? GetMathMinMaxMethod(ReductionOp op, Type clrType)
        {
            string name = op == ReductionOp.Max ? "Max" : "Min";
            return typeof(Math).GetMethod(name, new[] { clrType, clrType });
        }

        /// <summary>
        /// Emit tree reduction for horizontal operations (Sum, Min, Max, Prod).
        /// Uses GetLower/GetUpper to halve vector width each step: O(log N) vs O(N).
        /// Stack has vector on entry, scalar result on exit.
        /// </summary>
        private static void EmitTreeReduction(ILGenerator il, NPTypeCode type, ReductionOp op)
        {
            var clrType = GetClrType(type);
            int currentBits = VectorBits;

            // Step 1: Reduce from current width down to 128-bit using GetLower/GetUpper + op
            while (currentBits > 128)
            {
                int nextBits = currentBits / 2;
                var currentContainer = currentBits switch
                {
                    512 => typeof(Vector512),
                    256 => typeof(Vector256),
                    _ => throw new InvalidOperationException()
                };
                var nextContainer = nextBits switch
                {
                    256 => typeof(Vector256),
                    128 => typeof(Vector128),
                    _ => throw new InvalidOperationException()
                };
                var currentVecType = currentBits switch
                {
                    512 => typeof(Vector512<>).MakeGenericType(clrType),
                    256 => typeof(Vector256<>).MakeGenericType(clrType),
                    _ => throw new InvalidOperationException()
                };
                var nextVecType = nextBits switch
                {
                    256 => typeof(Vector256<>).MakeGenericType(clrType),
                    128 => typeof(Vector128<>).MakeGenericType(clrType),
                    _ => throw new InvalidOperationException()
                };

                // Store current vector
                var locVec = il.DeclareLocal(currentVecType);
                il.Emit(OpCodes.Stloc, locVec);

                // GetLower (returns half-width vector)
                var getLowerMethod = currentContainer.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == "GetLower" && m.IsGenericMethod)
                    .Select(m => m.MakeGenericMethod(clrType))
                    .First();
                il.Emit(OpCodes.Ldloc, locVec);
                il.EmitCall(OpCodes.Call, getLowerMethod, null);

                // GetUpper
                var getUpperMethod = currentContainer.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == "GetUpper" && m.IsGenericMethod)
                    .Select(m => m.MakeGenericMethod(clrType))
                    .First();
                il.Emit(OpCodes.Ldloc, locVec);
                il.EmitCall(OpCodes.Call, getUpperMethod, null);

                // Apply reduction operation on the two half-vectors
                EmitVectorReductionOp(il, op, nextContainer, nextVecType, clrType);

                currentBits = nextBits;
            }

            // Step 2: Now we have Vector128. Reduce to scalar.
            // Vector128 has 2-16 elements depending on type. Use GetElement for final few.
            var vec128Type = typeof(Vector128<>).MakeGenericType(clrType);
            int elemCount = 16 / GetTypeSize(type); // Vector128 is 16 bytes

            var locFinal = il.DeclareLocal(vec128Type);
            il.Emit(OpCodes.Stloc, locFinal);

            // Get first element as accumulator
            var getElementMethod = typeof(Vector128).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "GetElement" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(clrType))
                .First();

            il.Emit(OpCodes.Ldloc, locFinal);
            il.Emit(OpCodes.Ldc_I4_0);
            il.EmitCall(OpCodes.Call, getElementMethod, null);

            // Reduce remaining elements (only 1-3 more for most types)
            for (int i = 1; i < elemCount; i++)
            {
                il.Emit(OpCodes.Ldloc, locFinal);
                il.Emit(OpCodes.Ldc_I4, i);
                il.EmitCall(OpCodes.Call, getElementMethod, null);
                EmitScalarReductionOp(il, op, type);
            }
        }

        /// <summary>
        /// Emit vector reduction operation (Add, Min, Max, Multiply).
        /// Stack has [vec1, vec2], result is combined vector.
        /// </summary>
        private static void EmitVectorReductionOp(ILGenerator il, ReductionOp op,
            Type containerType, Type vectorType, Type clrType)
        {
            string methodName = op switch
            {
                ReductionOp.Sum => "Add",
                ReductionOp.Min => "Min",
                ReductionOp.Max => "Max",
                ReductionOp.Prod => "Multiply",
                _ => throw new NotSupportedException($"Reduction {op} not supported for tree reduction")
            };

            var method = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == methodName && m.IsGenericMethod && m.GetParameters().Length == 2)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType);

            if (method == null)
                throw new InvalidOperationException($"Could not find {containerType.Name}.{methodName}<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit scalar reduction operation.
        /// Stack has [accum, value], result is combined scalar.
        /// </summary>
        private static void EmitScalarReductionOp(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            switch (op)
            {
                case ReductionOp.Sum:
                    il.Emit(OpCodes.Add);
                    break;
                case ReductionOp.Prod:
                    il.Emit(OpCodes.Mul);
                    break;
                case ReductionOp.Min:
                case ReductionOp.Max:
                    var mathMethod = GetMathMinMaxMethod(op, GetClrType(type));
                    if (mathMethod != null)
                    {
                        il.EmitCall(OpCodes.Call, mathMethod, null);
                    }
                    else
                    {
                        EmitScalarMinMax(il, op, type);
                    }
                    break;
                default:
                    throw new NotSupportedException($"Scalar reduction {op} not supported");
            }
        }

        /// <summary>
        /// Emit scalar min/max comparison.
        /// Stack has [value1, value2], result is min or max.
        /// </summary>
        private static void EmitScalarMinMax(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            // Use comparison: (a > b) ? a : b for Max, (a < b) ? a : b for Min
            var locA = il.DeclareLocal(GetClrType(type));
            var locB = il.DeclareLocal(GetClrType(type));
            var lblFalse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.Emit(OpCodes.Stloc, locB);
            il.Emit(OpCodes.Stloc, locA);

            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Ldloc, locB);

            if (op == ReductionOp.Max)
            {
                if (IsUnsigned(type))
                    il.Emit(OpCodes.Bgt_Un, lblFalse);
                else
                    il.Emit(OpCodes.Bgt, lblFalse);

                // a <= b, return b
                il.Emit(OpCodes.Ldloc, locB);
                il.Emit(OpCodes.Br, lblEnd);

                il.MarkLabel(lblFalse);
                // a > b, return a
                il.Emit(OpCodes.Ldloc, locA);
            }
            else
            {
                if (IsUnsigned(type))
                    il.Emit(OpCodes.Blt_Un, lblFalse);
                else
                    il.Emit(OpCodes.Blt, lblFalse);

                // a >= b, return b
                il.Emit(OpCodes.Ldloc, locB);
                il.Emit(OpCodes.Br, lblEnd);

                il.MarkLabel(lblFalse);
                // a < b, return a
                il.Emit(OpCodes.Ldloc, locA);
            }

            il.MarkLabel(lblEnd);
        }

        /// <summary>
        /// Emit reduction combine operation.
        /// Stack has [newValue, accumulator], result is combined value.
        /// </summary>
        private static void EmitReductionCombine(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            switch (op)
            {
                case ReductionOp.Sum:
                case ReductionOp.Mean:
                case ReductionOp.CumSum:
                    // Add
                    if (type == NPTypeCode.Decimal)
                    {
                        il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Addition", new[] { typeof(decimal), typeof(decimal) })!, null);
                    }
                    else
                    {
                        il.Emit(OpCodes.Add);
                    }
                    break;

                case ReductionOp.Prod:
                    // Multiply
                    if (type == NPTypeCode.Decimal)
                    {
                        il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Multiply", new[] { typeof(decimal), typeof(decimal) })!, null);
                    }
                    else
                    {
                        il.Emit(OpCodes.Mul);
                    }
                    break;

                case ReductionOp.Max:
                    {
                        var clrType = GetClrType(type);
                        var mathMethod = GetMathMinMaxMethod(op, clrType);
                        if (mathMethod != null)
                        {
                            il.EmitCall(OpCodes.Call, mathMethod, null);
                        }
                        else
                        {
                            EmitScalarMinMax(il, op, type);
                        }
                    }
                    break;

                case ReductionOp.Min:
                    {
                        var clrType = GetClrType(type);
                        var mathMethod = GetMathMinMaxMethod(op, clrType);
                        if (mathMethod != null)
                        {
                            il.EmitCall(OpCodes.Call, mathMethod, null);
                        }
                        else
                        {
                            EmitScalarMinMax(il, op, type);
                        }
                    }
                    break;

                default:
                    throw new NotSupportedException($"Reduction combine for {op} not supported");
            }
        }

        /// <summary>
        /// Emit ArgMax/ArgMin step - compare new value with accumulator, update index if better.
        /// Stack has [newValue]. Updates locAccum and locIdx.
        /// </summary>
        private static void EmitArgReductionStep(ILGenerator il, ReductionOp op, NPTypeCode type,
            LocalBuilder locAccum, LocalBuilder locIdx, LocalBuilder locI)
        {
            // newValue is on stack, compare with locAccum
            var lblSkip = il.DefineLabel();

            il.Emit(OpCodes.Dup); // [newValue, newValue]
            il.Emit(OpCodes.Ldloc, locAccum); // [newValue, newValue, accum]

            // Compare: newValue > accum (for ArgMax) or newValue < accum (for ArgMin)
            if (op == ReductionOp.ArgMax)
            {
                if (IsUnsigned(type))
                    il.Emit(OpCodes.Ble_Un, lblSkip);
                else
                    il.Emit(OpCodes.Ble, lblSkip);
            }
            else // ArgMin
            {
                if (IsUnsigned(type))
                    il.Emit(OpCodes.Bge_Un, lblSkip);
                else
                    il.Emit(OpCodes.Bge, lblSkip);
            }

            // Update: newValue is better
            // Stack has [newValue]
            il.Emit(OpCodes.Stloc, locAccum); // accum = newValue
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx); // idx = i
            var lblEnd = il.DefineLabel();
            il.Emit(OpCodes.Br, lblEnd);

            il.MarkLabel(lblSkip);
            // Not better, pop newValue
            il.Emit(OpCodes.Pop);

            il.MarkLabel(lblEnd);
        }

        #endregion

        #endregion
    }
}
