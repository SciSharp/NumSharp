using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator - IL-based SIMD kernel generation using DynamicMethod
// =============================================================================
//
// ARCHITECTURE OVERVIEW
// ---------------------
// This partial class generates high-performance kernels at runtime using IL emission.
// The JIT compiler can then optimize these kernels with full SIMD support (V128/V256/V512).
// Kernels are cached by operation key to avoid repeated IL generation.
//
// FLOW: Caller (DefaultEngine, np.*, NDArray ops)
//         -> Requests kernel via Get*Kernel() or *Helper() methods
//         -> ILKernelGenerator checks cache, generates IL if needed
//         -> Returns delegate that caller invokes with array pointers
//
// =============================================================================
// PARTIAL CLASS FILES
// =============================================================================
//
// ILKernelGenerator.cs
//   OWNERSHIP: Core infrastructure - foundation for all other partial files
//   RESPONSIBILITY:
//     - Global state: Enabled flag, VectorBits/VectorBytes (detected at startup)
//     - Type mapping: NPTypeCode <-> CLR Type <-> Vector type conversions
//     - Shared IL emission primitives used by all other partials
//   DEPENDENCIES: None (other partials depend on this)
//
// ILKernelGenerator.Binary.cs
//   OWNERSHIP: Same-type binary operations on contiguous arrays (fast path)
//   RESPONSIBILITY:
//     - Optimized kernels when both operands have identical type and layout
//     - SIMD loop + scalar tail for Add, Sub, Mul, Div
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for same-type contiguous operations
//
// ILKernelGenerator.MixedType.cs
//   OWNERSHIP: Mixed-type binary operations with type promotion
//   RESPONSIBILITY:
//     - Handles all binary ops where operand types may differ
//     - Generates path-specific kernels based on stride patterns
//     - Owns ClearAll() which clears ALL caches across all partials
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for general binary operations
//
// ILKernelGenerator.Unary.cs
//   OWNERSHIP: Unary element-wise operations
//   RESPONSIBILITY:
//     - Math functions: Negate, Abs, Sqrt, Sin, Cos, Exp, Log, Sign, Floor, Ceil, etc.
//     - Scalar delegate generation for single-value operations (Func<TIn,TOut>)
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for unary ops; scalar delegates used in broadcasting
//
// ILKernelGenerator.Comparison.cs
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: ==, !=, <, >, <=, >=
//     - SIMD comparison with efficient mask-to-bool extraction
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by NDArray comparison operators
//
// ILKernelGenerator.Reduction.cs (THIS FILE)
//   OWNERSHIP: Reduction operations and specialized SIMD helpers
//   RESPONSIBILITY:
//     - IL-generated reduction kernels: Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any
//     - HORIZONTAL SIMD: Tree reduction pattern (O(log N) vs O(N)):
//       * Vector512 -> Vector256 -> Vector128 -> scalar using GetLower/GetUpper
//     - SIMD HELPER METHODS (called directly, not via IL kernels):
//       * AllSimdHelper<T>() - returns false on first zero (early-exit)
//       * AnySimdHelper<T>() - returns true on first non-zero (early-exit)
//       * ArgMaxSimdHelper<T>() - two-pass: find max value with SIMD, then find index
//       * ArgMinSimdHelper<T>() - two-pass: find min value with SIMD, then find index
//       * NonZeroSimdHelper<T>() - collects indices where elements != 0
//       * ConvertFlatIndicesToCoordinates() - flat indices -> per-dimension arrays
//       * CountTrueSimdHelper() - counts true values in bool array
//       * CopyMaskedElementsHelper<T>() - copies elements where mask is true
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW:
//     - Reduction kernels: Called by DefaultEngine for np.sum, np.max, etc.
//     - Helper methods: Called DIRECTLY by np.all, np.any, np.nonzero, boolean indexing
//   KEY MEMBERS:
//     - TypedElementReductionKernel<T> delegate - returns TResult from array
//     - _elementReductionCache - caches by ElementReductionKernelKey
//     - GetTypedElementReductionKernel<T>(), ClearReduction()
//     - AllSimdHelper<T>(), AnySimdHelper<T>() - direct SIMD for boolean reductions
//     - ArgMaxSimdHelper<T>(), ArgMinSimdHelper<T>() - SIMD with index tracking
//     - NonZeroSimdHelper<T>(), ConvertFlatIndicesToCoordinates() - for np.nonzero
//     - CountTrueSimdHelper(), CopyMaskedElementsHelper<T>() - boolean masking
//     - EmitTreeReduction() - O(log N) horizontal reduction pattern
//     - EmitLoadIdentity(), EmitLoadZero/One/MinValue/MaxValue() - reduction identities
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public sealed partial class ILKernelGenerator
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

            // For Sum/Prod, SIMD vectors work on same type - can't widen int32 to int64 in SIMD
            // When accumulator type differs from input type, use scalar path to prevent overflow
            if ((key.Op == ReductionOp.Sum || key.Op == ReductionOp.Prod) &&
                key.InputType != key.AccumulatorType)
            {
                return false;
            }

            // Only certain operations have SIMD support
            // Sum: Vector.Sum() or manual horizontal add
            // Max/Min: Reduce vector then scalar reduce remainder
            // Prod: Manual horizontal multiply
            // All/Any: SIMD comparison with early-exit
            // ArgMax/ArgMin: SIMD with index tracking
            return key.Op == ReductionOp.Sum || key.Op == ReductionOp.Max || key.Op == ReductionOp.Min ||
                   key.Op == ReductionOp.Prod || key.Op == ReductionOp.All || key.Op == ReductionOp.Any ||
                   key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin;
        }

        /// <summary>
        /// Emit a SIMD reduction loop for contiguous arrays.
        /// Uses vector accumulator for O(N + log(vectorWidth)) instead of O(N * log(vectorWidth)).
        /// </summary>
        private static void EmitReductionSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // All/Any use special early-exit logic
            if (key.Op == ReductionOp.All || key.Op == ReductionOp.Any)
            {
                EmitAllAnySimdLoop(il, key, inputSize);
                return;
            }

            // ArgMax/ArgMin use special index-tracking logic
            if (key.Op == ReductionOp.ArgMax || key.Op == ReductionOp.ArgMin)
            {
                EmitArgMaxMinSimdLoop(il, key, inputSize);
                return;
            }

            int vectorCount = GetVectorCount(key.InputType);
            var clrType = GetClrType(key.InputType);
            var vectorType = GetVectorType(clrType);

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int)); // totalSize - vectorCount
            var locVecAccum = il.DeclareLocal(vectorType); // VECTOR accumulator (optimized)
            var locScalarAccum = il.DeclareLocal(GetClrType(key.AccumulatorType)); // scalar for tail

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // Initialize VECTOR accumulator with identity value broadcast
            EmitVectorIdentity(il, key.Op, key.InputType);
            il.Emit(OpCodes.Stloc, locVecAccum);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // === SIMD LOOP (vector * vector accumulation) ===
            il.MarkLabel(lblSimdLoop);

            // if (i > vectorEnd) goto SimdLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

            // Load vector accumulator
            il.Emit(OpCodes.Ldloc, locVecAccum);

            // Load vector from input[i]
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);

            // vecAccum = vecAccum OP inputVec (vector-vector operation)
            EmitVectorBinaryReductionOp(il, key.Op, key.InputType);
            il.Emit(OpCodes.Stloc, locVecAccum);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === HORIZONTAL REDUCTION (once at end) ===
            // Reduce vector accumulator to scalar
            il.Emit(OpCodes.Ldloc, locVecAccum);
            EmitVectorHorizontalReduction(il, key.Op, key.InputType);
            il.Emit(OpCodes.Stloc, locScalarAccum);

            // === TAIL LOOP (scalar) ===
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

            // Combine with scalar accumulator
            il.Emit(OpCodes.Ldloc, locScalarAccum);
            EmitReductionCombine(il, key.Op, key.AccumulatorType);
            il.Emit(OpCodes.Stloc, locScalarAccum);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            // Return scalar accumulator
            il.Emit(OpCodes.Ldloc, locScalarAccum);
        }

        /// <summary>
        /// Emit vector identity value (broadcast identity to all lanes).
        /// </summary>
        private static void EmitVectorIdentity(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            // Load scalar identity
            EmitLoadIdentity(il, op, type);
            // Broadcast to vector
            EmitVectorCreate(il, type);
        }

        /// <summary>
        /// Emit vector-vector binary reduction operation.
        /// Stack has [vec1, vec2], result is combined vector.
        /// </summary>
        private static void EmitVectorBinaryReductionOp(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            string methodName = op switch
            {
                ReductionOp.Sum => "Add",
                ReductionOp.Prod => "Multiply",
                ReductionOp.Max => "Max",
                ReductionOp.Min => "Min",
                ReductionOp.Mean => "Add", // Mean uses Sum internally
                _ => throw new NotSupportedException($"Vector binary op for {op} not supported")
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

        /// <summary>
        /// Emit All/Any SIMD loop with early-exit.
        /// All: returns false immediately when any zero is found
        /// Any: returns true immediately when any non-zero is found
        /// </summary>
        private static void EmitAllAnySimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // For All/Any, we use a helper method approach because:
            // 1. Early-exit logic is complex to emit in IL
            // 2. The helper method can be JIT-optimized effectively
            // 3. This matches the pattern used elsewhere in the codebase

            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                key.Op == ReductionOp.All ? nameof(AllSimdHelper) : nameof(AnySimdHelper),
                BindingFlags.NonPublic | BindingFlags.Static);

            var genericHelper = helperMethod!.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: AllSimdHelper<T>(input, totalSize) or AnySimdHelper<T>(input, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.EmitCall(OpCodes.Call, genericHelper, null);

            // Result (bool) is already on stack, but we need to convert to TResult (also bool for All/Any)
            // Stack has: bool result - convert to byte (0 or 1) for bool return
        }

        /// <summary>
        /// SIMD helper for All reduction with early-exit.
        /// Returns true if ALL elements are non-zero.
        /// </summary>
        internal static unsafe bool AllSimdHelper<T>(void* input, int totalSize) where T : unmanaged
        {
            if (totalSize == 0)
                return true; // NumPy: all([]) == True (vacuous truth)

            T* src = (T*)input;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;
                var zero = Vector256<T>.Zero;
                int i = 0;

                // SIMD loop with early exit
                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, zero);

                    // If ANY element equals zero, return false
                    if (Vector256.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }

                // Scalar tail
                for (; i < totalSize; i++)
                {
                    if (System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return false;
                }

                return true;
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;
                var zero = Vector128<T>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, zero);

                    if (Vector128.ExtractMostSignificantBits(mask) != 0)
                        return false;
                }

                for (; i < totalSize; i++)
                {
                    if (System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return false;
                }

                return true;
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < totalSize; i++)
                {
                    if (System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return false;
                }
                return true;
            }
        }

        /// <summary>
        /// SIMD helper for Any reduction with early-exit.
        /// Returns true if ANY element is non-zero.
        /// </summary>
        internal static unsafe bool AnySimdHelper<T>(void* input, int totalSize) where T : unmanaged
        {
            if (totalSize == 0)
                return false; // NumPy: any([]) == False

            T* src = (T*)input;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;
                var zero = Vector256<T>.Zero;
                uint allZeroMask = (1u << vectorCount) - 1;
                int i = 0;

                // SIMD loop with early exit
                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);

                    // If NOT all elements are zero, we found a non-zero
                    if (bits != allZeroMask)
                        return true;
                }

                // Scalar tail
                for (; i < totalSize; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return true;
                }

                return false;
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;
                var zero = Vector128<T>.Zero;
                uint allZeroMask = (1u << vectorCount) - 1;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);

                    if (bits != allZeroMask)
                        return true;
                }

                for (; i < totalSize; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return true;
                }

                return false;
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < totalSize; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        return true;
                }
                return false;
            }
        }

        /// <summary>
        /// Emit ArgMax/ArgMin SIMD loop.
        /// Uses helper methods for clean implementation with SIMD index tracking.
        /// </summary>
        private static void EmitArgMaxMinSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            var helperMethod = typeof(ILKernelGenerator).GetMethod(
                key.Op == ReductionOp.ArgMax ? nameof(ArgMaxSimdHelper) : nameof(ArgMinSimdHelper),
                BindingFlags.NonPublic | BindingFlags.Static);

            var genericHelper = helperMethod!.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: ArgMaxSimdHelper<T>(input, totalSize) or ArgMinSimdHelper<T>(input, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.EmitCall(OpCodes.Call, genericHelper, null);

            // Result (int) is already on stack
        }

        /// <summary>
        /// SIMD helper for ArgMax reduction.
        /// Returns the index of the maximum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        internal static unsafe int ArgMaxSimdHelper<T>(void* input, int totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            int bestIndex = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count * 2)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                // First pass: find the maximum value using SIMD
                var maxVec = Vector256.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    maxVec = Vector256.Max(maxVec, vec);
                }

                // Horizontal reduce the max vector to find the scalar max
                T maxValue = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = maxVec.GetElement(j);
                    if (elem.CompareTo(maxValue) > 0)
                        maxValue = elem;
                }

                // Process scalar tail for max value
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) > 0)
                        maxValue = src[i];
                }

                // Second pass: find the first index with the max value
                // Use SIMD to quickly scan for the max value
                var targetVec = Vector256.Create(maxValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, targetVec);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        // Found it! Return index of first match
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                // Check scalar tail
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) == 0)
                        return i;
                }

                return 0; // Should never reach here
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count * 2)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                var maxVec = Vector128.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    maxVec = Vector128.Max(maxVec, vec);
                }

                T maxValue = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = maxVec.GetElement(j);
                    if (elem.CompareTo(maxValue) > 0)
                        maxValue = elem;
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) > 0)
                        maxValue = src[i];
                }

                var targetVec = Vector128.Create(maxValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, targetVec);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) == 0)
                        return i;
                }

                return 0;
            }
            else
            {
                // Scalar fallback
                for (int i = 1; i < totalSize; i++)
                {
                    if (src[i].CompareTo(bestValue) > 0)
                    {
                        bestValue = src[i];
                        bestIndex = i;
                    }
                }
                return bestIndex;
            }
        }

        /// <summary>
        /// SIMD helper for ArgMin reduction.
        /// Returns the index of the minimum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        internal static unsafe int ArgMinSimdHelper<T>(void* input, int totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            int bestIndex = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= Vector256<T>.Count * 2)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                // First pass: find the minimum value using SIMD
                var minVec = Vector256.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    minVec = Vector256.Min(minVec, vec);
                }

                // Horizontal reduce the min vector to find the scalar min
                T minValue = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = minVec.GetElement(j);
                    if (elem.CompareTo(minValue) < 0)
                        minValue = elem;
                }

                // Process scalar tail for min value
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) < 0)
                        minValue = src[i];
                }

                // Second pass: find the first index with the min value
                var targetVec = Vector256.Create(minValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, targetVec);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) == 0)
                        return i;
                }

                return 0;
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= Vector128<T>.Count * 2)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = totalSize - vectorCount;

                var minVec = Vector128.Load(src);
                int i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    minVec = Vector128.Min(minVec, vec);
                }

                T minValue = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = minVec.GetElement(j);
                    if (elem.CompareTo(minValue) < 0)
                        minValue = elem;
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) < 0)
                        minValue = src[i];
                }

                var targetVec = Vector128.Create(minValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, targetVec);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) == 0)
                        return i;
                }

                return 0;
            }
            else
            {
                // Scalar fallback
                for (int i = 1; i < totalSize; i++)
                {
                    if (src[i].CompareTo(bestValue) < 0)
                    {
                        bestValue = src[i];
                        bestIndex = i;
                    }
                }
                return bestIndex;
            }
        }

        #region NonZero SIMD Helpers

        /// <summary>
        /// SIMD helper for NonZero operation.
        /// Finds all indices where elements are non-zero.
        /// </summary>
        /// <param name="src">Source array pointer</param>
        /// <param name="size">Number of elements</param>
        /// <param name="indices">Output list to populate with non-zero indices</param>
        internal static unsafe void NonZeroSimdHelper<T>(T* src, int size, System.Collections.Generic.List<int> indices)
            where T : unmanaged
        {
            if (size == 0)
                return;

            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && size >= Vector256<T>.Count)
            {
                int vectorCount = Vector256<T>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector256<T>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);

                    // Invert: we want non-zero elements
                    uint nonZeroBits = ~bits & ((1u << vectorCount) - 1);

                    // Extract indices where bits are set
                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        indices.Add(i + bitPos);
                        nonZeroBits &= nonZeroBits - 1; // Clear lowest bit
                    }
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        indices.Add(i);
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && size >= Vector128<T>.Count)
            {
                int vectorCount = Vector128<T>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector128<T>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);

                    uint nonZeroBits = ~bits & ((1u << vectorCount) - 1);

                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        indices.Add(i + bitPos);
                        nonZeroBits &= nonZeroBits - 1;
                    }
                }

                for (; i < size; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        indices.Add(i);
                }
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < size; i++)
                {
                    if (!System.Collections.Generic.EqualityComparer<T>.Default.Equals(src[i], default))
                        indices.Add(i);
                }
            }
        }

        /// <summary>
        /// Convert flat indices to per-dimension coordinate arrays.
        /// </summary>
        /// <param name="flatIndices">List of flat (linear) indices</param>
        /// <param name="shape">Shape of the array</param>
        /// <returns>Array of NDArray&lt;int&gt;, one per dimension</returns>
        internal static unsafe NumSharp.Generic.NDArray<int>[] ConvertFlatIndicesToCoordinates(
            System.Collections.Generic.List<int> flatIndices, int[] shape)
        {
            int ndim = shape.Length;
            int len = flatIndices.Count;

            // Create result arrays
            var result = new NumSharp.Generic.NDArray<int>[ndim];
            for (int d = 0; d < ndim; d++)
                result[d] = new NumSharp.Generic.NDArray<int>(len);

            // Get addresses for direct writing
            var addresses = new int*[ndim];
            for (int d = 0; d < ndim; d++)
                addresses[d] = (int*)result[d].Address;

            // Pre-compute strides for index conversion
            var strides = new int[ndim];
            strides[ndim - 1] = 1;
            for (int d = ndim - 2; d >= 0; d--)
                strides[d] = strides[d + 1] * shape[d + 1];

            // Convert each flat index to coordinates
            for (int i = 0; i < len; i++)
            {
                int flatIdx = flatIndices[i];
                for (int d = 0; d < ndim; d++)
                {
                    addresses[d][i] = flatIdx / strides[d];
                    flatIdx %= strides[d];
                }
            }

            return result;
        }

        #endregion

        #region Boolean Masking SIMD Helpers

        /// <summary>
        /// SIMD helper to count true values in a boolean array.
        /// </summary>
        internal static unsafe int CountTrueSimdHelper(bool* mask, int size)
        {
            if (size == 0)
                return 0;

            int count = 0;

            if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
            {
                int vectorCount = Vector256<byte>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector256<byte>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load((byte*)(mask + i));
                    var cmp = Vector256.Equals(vec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(cmp);

                    // Count non-zero (true) values: invert mask, popcount
                    uint nonZeroBits = ~bits;
                    count += System.Numerics.BitOperations.PopCount(nonZeroBits);
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (mask[i])
                        count++;
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<byte>.IsSupported && size >= Vector128<byte>.Count)
            {
                int vectorCount = Vector128<byte>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector128<byte>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load((byte*)(mask + i));
                    var cmp = Vector128.Equals(vec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(cmp);

                    uint nonZeroBits = ~bits & 0xFFFFu;
                    count += System.Numerics.BitOperations.PopCount(nonZeroBits);
                }

                for (; i < size; i++)
                {
                    if (mask[i])
                        count++;
                }
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < size; i++)
                {
                    if (mask[i])
                        count++;
                }
            }

            return count;
        }

        /// <summary>
        /// SIMD helper to copy elements where mask is true.
        /// Copies from src to dst where mask[i] is true.
        /// </summary>
        /// <returns>Number of elements copied</returns>
        internal static unsafe int CopyMaskedElementsHelper<T>(T* src, bool* mask, T* dst, int size)
            where T : unmanaged
        {
            int dstIdx = 0;

            // For masking, we can't easily vectorize the gather/scatter
            // But we can vectorize the mask scanning to find true indices faster
            if (Vector256.IsHardwareAccelerated && Vector256<byte>.IsSupported && size >= Vector256<byte>.Count)
            {
                int vectorCount = Vector256<byte>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector256<byte>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var maskVec = Vector256.Load((byte*)(mask + i));
                    var cmp = Vector256.Equals(maskVec, zero);
                    uint bits = Vector256.ExtractMostSignificantBits(cmp);
                    uint nonZeroBits = ~bits;

                    // Copy elements where mask is true
                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        dst[dstIdx++] = src[i + bitPos];
                        nonZeroBits &= nonZeroBits - 1;
                    }
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<byte>.IsSupported && size >= Vector128<byte>.Count)
            {
                int vectorCount = Vector128<byte>.Count;
                int vectorEnd = size - vectorCount;
                var zero = Vector128<byte>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var maskVec = Vector128.Load((byte*)(mask + i));
                    var cmp = Vector128.Equals(maskVec, zero);
                    uint bits = Vector128.ExtractMostSignificantBits(cmp);
                    uint nonZeroBits = ~bits & 0xFFFFu;

                    while (nonZeroBits != 0)
                    {
                        int bitPos = System.Numerics.BitOperations.TrailingZeroCount(nonZeroBits);
                        dst[dstIdx++] = src[i + bitPos];
                        nonZeroBits &= nonZeroBits - 1;
                    }
                }

                for (; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }
            else
            {
                // Scalar fallback
                for (int i = 0; i < size; i++)
                {
                    if (mask[i])
                        dst[dstIdx++] = src[i];
                }
            }

            return dstIdx;
        }

        /// <summary>
        /// SIMD helper for computing variance of a contiguous array.
        /// Uses two-pass algorithm: compute mean, then sum of squared differences.
        /// </summary>
        /// <typeparam name="T">Element type (float or double)</typeparam>
        /// <param name="src">Pointer to contiguous data</param>
        /// <param name="size">Number of elements</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population variance, 1 for sample variance)</param>
        /// <returns>The variance as double</returns>
        internal static unsafe double VarSimdHelper<T>(T* src, int size, int ddof = 0)
            where T : unmanaged
        {
            if (size == 0)
                return double.NaN;

            if (size <= ddof)
                return double.NaN; // Division by zero or negative

            // Pass 1: Compute mean
            double sum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)src;

                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    }

                    // Horizontal sum
                    sum = Vector256.Sum(sumVec);

                    // Scalar tail
                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
                {
                    int vectorCount = Vector128<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector128<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector128.Add(sumVec, Vector128.Load(p + i));
                    }

                    sum = Vector128.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }

                double mean = sum / size;

                // Pass 2: Sum of squared differences
                double sqDiffSum = 0;

                if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(mean);
                    var sqDiffVec = Vector256<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector256.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
                {
                    int vectorCount = Vector128<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector128.Create(mean);
                    var sqDiffVec = Vector128<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(p + i);
                        var diff = Vector128.Subtract(vec, meanVec);
                        sqDiffVec = Vector128.Add(sqDiffVec, Vector128.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector128.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                return sqDiffSum / (size - ddof);
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)src;

                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    }

                    sum = Vector256.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
                {
                    int vectorCount = Vector128<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector128<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector128.Add(sumVec, Vector128.Load(p + i));
                    }

                    sum = Vector128.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }

                double mean = sum / size;

                // Pass 2: Sum of squared differences (compute in double for precision)
                double sqDiffSum = 0;

                if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create((float)mean);
                    var sqDiffVec = Vector256<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector256.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
                {
                    int vectorCount = Vector128<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector128.Create((float)mean);
                    var sqDiffVec = Vector128<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(p + i);
                        var diff = Vector128.Subtract(vec, meanVec);
                        sqDiffVec = Vector128.Add(sqDiffVec, Vector128.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector128.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                return sqDiffSum / (size - ddof);
            }
            else
            {
                // For integer types, convert to double and compute
                double doubleSum = 0;
                for (int i = 0; i < size; i++)
                {
                    doubleSum += Convert.ToDouble(src[i]);
                }
                double mean = doubleSum / size;

                double sqDiffSum = 0;
                for (int i = 0; i < size; i++)
                {
                    double diff = Convert.ToDouble(src[i]) - mean;
                    sqDiffSum += diff * diff;
                }

                return sqDiffSum / (size - ddof);
            }
        }

        /// <summary>
        /// SIMD helper for computing standard deviation of a contiguous array.
        /// Returns sqrt(variance).
        /// </summary>
        /// <typeparam name="T">Element type (float or double)</typeparam>
        /// <param name="src">Pointer to contiguous data</param>
        /// <param name="size">Number of elements</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population std, 1 for sample std)</param>
        /// <returns>The standard deviation as double</returns>
        internal static unsafe double StdSimdHelper<T>(T* src, int size, int ddof = 0)
            where T : unmanaged
        {
            double variance = VarSimdHelper(src, size, ddof);
            return Math.Sqrt(variance);
        }

        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        /// <param name="src">Pointer to contiguous float data</param>
        /// <param name="size">Number of elements</param>
        /// <returns>Sum of non-NaN elements</returns>
        internal static unsafe float NanSumSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return 0f;

            float sum = 0f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector256<float>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Create mask where NaN becomes 0, valid values stay
                    // NaN comparison: x != x is true for NaN
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN, false for NaN
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector128<float>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous double array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        internal static unsafe double NanSumSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return 0.0;

            double sum = 0.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector256<double>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var sumVec = Vector128<double>.Zero;
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous float array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe float NanProdSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return 1f;

            float prod = 1f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1f);
                var oneVec = Vector256.Create(1f);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with 1: if NaN, use 1; otherwise use original
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                // Horizontal product (no built-in, do manually)
                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1f);
                var oneVec = Vector128.Create(1f);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous double array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe double NanProdSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return 1.0;

            double prod = 1.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1.0);
                var oneVec = Vector256.Create(1.0);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1.0);
                var oneVec = Vector128.Create(1.0);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware minimum of a contiguous float array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe float NanMinSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return float.NaN;

            float minVal = float.PositiveInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector256.Create(float.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with +Inf so they don't affect minimum
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.PositiveInfinity));
                    minVec = Vector256.Min(minVec, cleaned);
                }

                // Horizontal min
                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !float.IsPositiveInfinity(minVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector128.Create(float.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.PositiveInfinity));
                    minVec = Vector128.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !float.IsPositiveInfinity(minVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        if (src[i] < minVal)
                            minVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? minVal : float.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware minimum of a contiguous double array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe double NanMinSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return double.NaN;

            double minVal = double.PositiveInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector256.Create(double.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.PositiveInfinity));
                    minVec = Vector256.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !double.IsPositiveInfinity(minVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var minVec = Vector128.Create(double.PositiveInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.PositiveInfinity));
                    minVec = Vector128.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !double.IsPositiveInfinity(minVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        if (src[i] < minVal)
                            minVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? minVal : double.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware maximum of a contiguous float array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe float NanMaxSimdHelperFloat(float* src, int size)
        {
            if (size == 0)
                return float.NaN;

            float maxVal = float.NegativeInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector256.Create(float.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with -Inf so they don't affect maximum
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.NegativeInfinity));
                    maxVec = Vector256.Max(maxVec, cleaned);
                }

                // Horizontal max
                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !float.IsNegativeInfinity(maxVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector128.Create(float.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.NegativeInfinity));
                    maxVec = Vector128.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !float.IsNegativeInfinity(maxVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        if (src[i] > maxVal)
                            maxVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? maxVal : float.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware maximum of a contiguous double array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe double NanMaxSimdHelperDouble(double* src, int size)
        {
            if (size == 0)
                return double.NaN;

            double maxVal = double.NegativeInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector256.Create(double.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.NegativeInfinity));
                    maxVec = Vector256.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !double.IsNegativeInfinity(maxVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                int vectorEnd = size - vectorCount;
                var maxVec = Vector128.Create(double.NegativeInfinity);
                int i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.NegativeInfinity));
                    maxVec = Vector128.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !double.IsNegativeInfinity(maxVal);
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        if (src[i] > maxVal)
                            maxVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? maxVal : double.NaN;
        }

        #endregion

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

                case ReductionOp.All:
                    // Identity for AND is true (vacuous truth)
                    il.Emit(OpCodes.Ldc_I4_1);
                    break;

                case ReductionOp.Any:
                    // Identity for OR is false
                    il.Emit(OpCodes.Ldc_I4_0);
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

        #region Axis Reduction SIMD Helpers

        /// <summary>
        /// Cache for axis reduction kernels (delegates that call SIMD helpers).
        /// </summary>
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<AxisReductionKernelKey, AxisReductionKernel> _axisReductionCache = new();

        /// <summary>
        /// Number of axis reduction kernels in cache.
        /// </summary>
        public static int AxisReductionCachedCount => _axisReductionCache.Count;

        /// <summary>
        /// Clear axis reduction cache.
        /// </summary>
        public static void ClearAxisReduction()
        {
            _axisReductionCache.Clear();
        }

        /// <summary>
        /// Try to get an axis reduction kernel.
        /// Supports all reduction operations and all types including type promotion.
        /// Uses SIMD for capable types, scalar loop for others.
        /// </summary>
        public static AxisReductionKernel? TryGetAxisReductionKernel(AxisReductionKernelKey key)
        {
            if (!Enabled)
                return null;

            // Support Sum, Prod, Min, Max, Mean operations
            // ArgMax/ArgMin require special index tracking - handled separately
            if (key.Op != ReductionOp.Sum && key.Op != ReductionOp.Prod &&
                key.Op != ReductionOp.Min && key.Op != ReductionOp.Max &&
                key.Op != ReductionOp.Mean)
            {
                return null;
            }

            // All types supported - SIMD for capable types, scalar for others
            return _axisReductionCache.GetOrAdd(key, CreateAxisReductionKernel);
        }

        /// <summary>
        /// Create an axis reduction kernel that dispatches to the appropriate helper.
        /// Handles all types including type promotion.
        /// </summary>
        private static AxisReductionKernel CreateAxisReductionKernel(AxisReductionKernelKey key)
        {
            // For type promotion cases or non-SIMD types, use the general dispatcher
            if (key.InputType != key.AccumulatorType || !CanUseSimd(key.InputType))
            {
                return CreateAxisReductionKernelGeneral(key);
            }

            // Same-type SIMD path - dispatch based on input type
            return key.InputType switch
            {
                NPTypeCode.Byte => CreateAxisReductionKernelTyped<byte>(key),
                NPTypeCode.Int16 => CreateAxisReductionKernelTyped<short>(key),
                NPTypeCode.UInt16 => CreateAxisReductionKernelTyped<ushort>(key),
                NPTypeCode.Int32 => CreateAxisReductionKernelTyped<int>(key),
                NPTypeCode.UInt32 => CreateAxisReductionKernelTyped<uint>(key),
                NPTypeCode.Int64 => CreateAxisReductionKernelTyped<long>(key),
                NPTypeCode.UInt64 => CreateAxisReductionKernelTyped<ulong>(key),
                NPTypeCode.Single => CreateAxisReductionKernelTyped<float>(key),
                NPTypeCode.Double => CreateAxisReductionKernelTyped<double>(key),
                _ => CreateAxisReductionKernelGeneral(key) // Fallback for Boolean, Char, Decimal
            };
        }

        /// <summary>
        /// Create a general axis reduction kernel for type promotion or non-SIMD types.
        /// Uses scalar loop with type conversion.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelGeneral(AxisReductionKernelKey key)
        {
            // Dispatch based on input and accumulator type combination
            return (key.InputType, key.AccumulatorType) switch
            {
                // Same-type scalar paths (for non-SIMD types like Decimal)
                (NPTypeCode.Decimal, NPTypeCode.Decimal) => CreateAxisReductionKernelScalar<decimal, decimal>(key),
                (NPTypeCode.Boolean, NPTypeCode.Boolean) => CreateAxisReductionKernelScalar<bool, bool>(key),
                (NPTypeCode.Char, NPTypeCode.Char) => CreateAxisReductionKernelScalar<char, char>(key),

                // Common type promotion paths (input -> wider accumulator)
                // byte -> int32/int64/double
                (NPTypeCode.Byte, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<byte, int>(key),
                (NPTypeCode.Byte, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<byte, long>(key),
                (NPTypeCode.Byte, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<byte, uint>(key),
                (NPTypeCode.Byte, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<byte, ulong>(key),
                (NPTypeCode.Byte, NPTypeCode.Double) => CreateAxisReductionKernelScalar<byte, double>(key),

                // int16 -> int32/int64/double
                (NPTypeCode.Int16, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<short, int>(key),
                (NPTypeCode.Int16, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<short, long>(key),
                (NPTypeCode.Int16, NPTypeCode.Double) => CreateAxisReductionKernelScalar<short, double>(key),

                // uint16 -> int32/uint32/int64/uint64/double
                (NPTypeCode.UInt16, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<ushort, int>(key),
                (NPTypeCode.UInt16, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<ushort, uint>(key),
                (NPTypeCode.UInt16, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<ushort, long>(key),
                (NPTypeCode.UInt16, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<ushort, ulong>(key),
                (NPTypeCode.UInt16, NPTypeCode.Double) => CreateAxisReductionKernelScalar<ushort, double>(key),

                // int32 -> int64/double
                (NPTypeCode.Int32, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<int, long>(key),
                (NPTypeCode.Int32, NPTypeCode.Double) => CreateAxisReductionKernelScalar<int, double>(key),

                // uint32 -> int64/uint64/double
                (NPTypeCode.UInt32, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<uint, long>(key),
                (NPTypeCode.UInt32, NPTypeCode.UInt64) => CreateAxisReductionKernelScalar<uint, ulong>(key),
                (NPTypeCode.UInt32, NPTypeCode.Double) => CreateAxisReductionKernelScalar<uint, double>(key),

                // int64 -> double
                (NPTypeCode.Int64, NPTypeCode.Double) => CreateAxisReductionKernelScalar<long, double>(key),

                // uint64 -> double
                (NPTypeCode.UInt64, NPTypeCode.Double) => CreateAxisReductionKernelScalar<ulong, double>(key),

                // float -> double
                (NPTypeCode.Single, NPTypeCode.Double) => CreateAxisReductionKernelScalar<float, double>(key),

                // char -> int32/int64
                (NPTypeCode.Char, NPTypeCode.Int32) => CreateAxisReductionKernelScalar<char, int>(key),
                (NPTypeCode.Char, NPTypeCode.Int64) => CreateAxisReductionKernelScalar<char, long>(key),
                (NPTypeCode.Char, NPTypeCode.UInt32) => CreateAxisReductionKernelScalar<char, uint>(key),

                // decimal -> double (for mean)
                (NPTypeCode.Decimal, NPTypeCode.Double) => CreateAxisReductionKernelScalar<decimal, double>(key),

                // Default fallback - use double accumulator
                _ => CreateAxisReductionKernelWithConversion(key)
            };
        }

        /// <summary>
        /// Create a fallback kernel using runtime type conversion.
        /// Used for rare type combinations not explicitly handled.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelWithConversion(AxisReductionKernelKey key)
        {
            // For rare combinations, use a runtime conversion approach via double
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisReductionWithConversionHelper(
                    input, output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.InputType, key.AccumulatorType, key.Op);
            };
        }

        /// <summary>
        /// Helper for axis reduction with runtime type conversion.
        /// </summary>
        private static unsafe void AxisReductionWithConversionHelper(
            void* input, void* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            NPTypeCode inputType, NPTypeCode accumType, ReductionOp op)
        {
            int axisStride = inputStrides[axis];
            int inputElemSize = inputType.SizeOf();
            int outputElemSize = accumType.SizeOf();

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            byte* inputBytes = (byte*)input;
            byte* outputBytes = (byte*)output;

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Reduce along axis using double as intermediate
                double accum = op switch
                {
                    ReductionOp.Sum or ReductionOp.Mean => 0.0,
                    ReductionOp.Prod => 1.0,
                    ReductionOp.Min => double.PositiveInfinity,
                    ReductionOp.Max => double.NegativeInfinity,
                    _ => 0.0
                };

                for (int i = 0; i < axisSize; i++)
                {
                    int inputOffset = inputBaseOffset + i * axisStride;
                    double val = ReadAsDouble(inputBytes + inputOffset * inputElemSize, inputType);

                    accum = op switch
                    {
                        ReductionOp.Sum or ReductionOp.Mean => accum + val,
                        ReductionOp.Prod => accum * val,
                        ReductionOp.Min => Math.Min(accum, val),
                        ReductionOp.Max => Math.Max(accum, val),
                        _ => accum
                    };
                }

                // For Mean, divide by count
                if (op == ReductionOp.Mean)
                    accum /= axisSize;

                // Write result
                WriteFromDouble(outputBytes + outputOffset * outputElemSize, accum, accumType);
            }
        }

        /// <summary>
        /// Read a value as double from typed memory.
        /// </summary>
        private static unsafe double ReadAsDouble(byte* ptr, NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Byte => *(byte*)ptr,
                NPTypeCode.Int16 => *(short*)ptr,
                NPTypeCode.UInt16 => *(ushort*)ptr,
                NPTypeCode.Int32 => *(int*)ptr,
                NPTypeCode.UInt32 => *(uint*)ptr,
                NPTypeCode.Int64 => *(long*)ptr,
                NPTypeCode.UInt64 => *(ulong*)ptr,
                NPTypeCode.Single => *(float*)ptr,
                NPTypeCode.Double => *(double*)ptr,
                NPTypeCode.Decimal => (double)*(decimal*)ptr,
                NPTypeCode.Char => *(char*)ptr,
                NPTypeCode.Boolean => *(bool*)ptr ? 1.0 : 0.0,
                _ => 0.0
            };
        }

        /// <summary>
        /// Write a double value to typed memory.
        /// </summary>
        private static unsafe void WriteFromDouble(byte* ptr, double value, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Byte: *(byte*)ptr = (byte)value; break;
                case NPTypeCode.Int16: *(short*)ptr = (short)value; break;
                case NPTypeCode.UInt16: *(ushort*)ptr = (ushort)value; break;
                case NPTypeCode.Int32: *(int*)ptr = (int)value; break;
                case NPTypeCode.UInt32: *(uint*)ptr = (uint)value; break;
                case NPTypeCode.Int64: *(long*)ptr = (long)value; break;
                case NPTypeCode.UInt64: *(ulong*)ptr = (ulong)value; break;
                case NPTypeCode.Single: *(float*)ptr = (float)value; break;
                case NPTypeCode.Double: *(double*)ptr = value; break;
                case NPTypeCode.Decimal: *(decimal*)ptr = (decimal)value; break;
                case NPTypeCode.Char: *(char*)ptr = (char)(int)value; break;
                case NPTypeCode.Boolean: *(bool*)ptr = value != 0; break;
            }
        }

        /// <summary>
        /// Create a typed scalar axis reduction kernel with type promotion.
        /// Uses scalar loop - no SIMD, but handles type conversion at compile time.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelScalar<TInput, TAccum>(AxisReductionKernelKey key)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisReductionScalarHelper<TInput, TAccum>(
                    (TInput*)input, (TAccum*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// Scalar axis reduction helper with type promotion.
        /// </summary>
        internal static unsafe void AxisReductionScalarHelper<TInput, TAccum>(
            TInput* input, TAccum* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            int axisStride = inputStrides[axis];

            // Compute output dimension strides for coordinate calculation
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute offsets
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    int inputDim = d >= axis ? d + 1 : d;
                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];
                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Reduce along axis with type conversion
                TAccum accum = GetIdentityValueTyped<TAccum>(op);
                TInput* axisStart = input + inputBaseOffset;

                for (int i = 0; i < axisSize; i++)
                {
                    TInput val = axisStart[i * axisStride];
                    accum = CombineScalarsPromoted<TInput, TAccum>(accum, val, op);
                }

                // For Mean, divide by count
                if (op == ReductionOp.Mean)
                    accum = DivideByCount<TAccum>(accum, axisSize);

                output[outputOffset] = accum;
            }
        }

        /// <summary>
        /// Combine accumulator with input value, promoting input to accumulator type.
        /// </summary>
        private static TAccum CombineScalarsPromoted<TInput, TAccum>(TAccum accum, TInput val, ReductionOp op)
            where TInput : unmanaged
            where TAccum : unmanaged
        {
            // Convert input to double for arithmetic, then to accumulator type
            double dAccum = ConvertToDouble(accum);
            double dVal = ConvertToDouble(val);

            double result = op switch
            {
                ReductionOp.Sum or ReductionOp.Mean => dAccum + dVal,
                ReductionOp.Prod => dAccum * dVal,
                ReductionOp.Min => Math.Min(dAccum, dVal),
                ReductionOp.Max => Math.Max(dAccum, dVal),
                _ => dAccum
            };

            return ConvertFromDouble<TAccum>(result);
        }

        /// <summary>
        /// Divide accumulator by count (for Mean).
        /// </summary>
        private static TAccum DivideByCount<TAccum>(TAccum accum, int count) where TAccum : unmanaged
        {
            double result = ConvertToDouble(accum) / count;
            return ConvertFromDouble<TAccum>(result);
        }

        /// <summary>
        /// Convert any numeric type to double.
        /// </summary>
        private static double ConvertToDouble<T>(T value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (byte)(object)value;
            if (typeof(T) == typeof(short)) return (short)(object)value;
            if (typeof(T) == typeof(ushort)) return (ushort)(object)value;
            if (typeof(T) == typeof(int)) return (int)(object)value;
            if (typeof(T) == typeof(uint)) return (uint)(object)value;
            if (typeof(T) == typeof(long)) return (long)(object)value;
            if (typeof(T) == typeof(ulong)) return (ulong)(object)value;
            if (typeof(T) == typeof(float)) return (float)(object)value;
            if (typeof(T) == typeof(double)) return (double)(object)value;
            if (typeof(T) == typeof(decimal)) return (double)(decimal)(object)value;
            if (typeof(T) == typeof(char)) return (char)(object)value;
            if (typeof(T) == typeof(bool)) return (bool)(object)value ? 1.0 : 0.0;
            return 0.0;
        }

        /// <summary>
        /// Convert double to target type.
        /// </summary>
        private static T ConvertFromDouble<T>(double value) where T : unmanaged
        {
            if (typeof(T) == typeof(byte)) return (T)(object)(byte)value;
            if (typeof(T) == typeof(short)) return (T)(object)(short)value;
            if (typeof(T) == typeof(ushort)) return (T)(object)(ushort)value;
            if (typeof(T) == typeof(int)) return (T)(object)(int)value;
            if (typeof(T) == typeof(uint)) return (T)(object)(uint)value;
            if (typeof(T) == typeof(long)) return (T)(object)(long)value;
            if (typeof(T) == typeof(ulong)) return (T)(object)(ulong)value;
            if (typeof(T) == typeof(float)) return (T)(object)(float)value;
            if (typeof(T) == typeof(double)) return (T)(object)value;
            if (typeof(T) == typeof(decimal)) return (T)(object)(decimal)value;
            if (typeof(T) == typeof(char)) return (T)(object)(char)(int)value;
            if (typeof(T) == typeof(bool)) return (T)(object)(value != 0);
            return default;
        }

        /// <summary>
        /// Get typed identity value for reduction operation.
        /// </summary>
        private static T GetIdentityValueTyped<T>(ReductionOp op) where T : unmanaged
        {
            double identity = op switch
            {
                ReductionOp.Sum or ReductionOp.Mean => 0.0,
                ReductionOp.Prod => 1.0,
                ReductionOp.Min => double.PositiveInfinity,
                ReductionOp.Max => double.NegativeInfinity,
                _ => 0.0
            };
            return ConvertFromDouble<T>(identity);
        }

        /// <summary>
        /// Create a typed axis reduction kernel.
        /// </summary>
        private static unsafe AxisReductionKernel CreateAxisReductionKernelTyped<T>(AxisReductionKernelKey key)
            where T : unmanaged
        {
            return (void* input, void* output, int* inputStrides, int* inputShape,
                    int* outputStrides, int axis, int axisSize, int ndim, int outputSize) =>
            {
                AxisReductionSimdHelper<T>(
                    (T*)input, (T*)output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    key.Op);
            };
        }

        /// <summary>
        /// SIMD helper for axis reduction operations.
        /// Reduces along a specific axis, writing results to output array.
        /// </summary>
        /// <typeparam name="T">Element type</typeparam>
        /// <param name="input">Input data pointer</param>
        /// <param name="output">Output data pointer</param>
        /// <param name="inputStrides">Input strides (element units)</param>
        /// <param name="inputShape">Input shape</param>
        /// <param name="outputStrides">Output strides (element units)</param>
        /// <param name="axis">Axis to reduce along</param>
        /// <param name="axisSize">Size of the axis being reduced</param>
        /// <param name="ndim">Number of input dimensions</param>
        /// <param name="outputSize">Total number of output elements</param>
        /// <param name="op">Reduction operation</param>
        internal static unsafe void AxisReductionSimdHelper<T>(
            T* input, T* output,
            int* inputStrides, int* inputShape, int* outputStrides,
            int axis, int axisSize, int ndim, int outputSize,
            ReductionOp op)
            where T : unmanaged
        {
            int axisStride = inputStrides[axis];

            // Check if the reduction axis is contiguous (stride == 1)
            bool axisContiguous = axisStride == 1;

            // Compute output shape strides for coordinate calculation
            // Output has ndim-1 dimensions (axis removed)
            int outputNdim = ndim - 1;
            Span<int> outputDimStrides = stackalloc int[outputNdim > 0 ? outputNdim : 1];
            if (outputNdim > 0)
            {
                outputDimStrides[outputNdim - 1] = 1;
                for (int d = outputNdim - 2; d >= 0; d--)
                {
                    // Map output dimension d to input dimension (d if d < axis, d+1 if d >= axis)
                    int inputDim = d >= axis ? d + 1 : d;
                    int nextInputDim = (d + 1) >= axis ? d + 2 : d + 1;
                    outputDimStrides[d] = outputDimStrides[d + 1] * inputShape[nextInputDim];
                }
            }

            // Iterate over all output elements
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                // Convert linear output index to coordinates and compute input base offset
                int remaining = outIdx;
                int inputBaseOffset = 0;
                int outputOffset = 0;

                for (int d = 0; d < outputNdim; d++)
                {
                    // Map output dimension d to input dimension
                    int inputDim = d >= axis ? d + 1 : d;

                    int coord = remaining / outputDimStrides[d];
                    remaining = remaining % outputDimStrides[d];

                    inputBaseOffset += coord * inputStrides[inputDim];
                    outputOffset += coord * outputStrides[d];
                }

                // Now reduce along the axis
                T* axisStart = input + inputBaseOffset;

                // For Mean, use Sum operation then divide
                ReductionOp actualOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;

                T result;
                if (axisContiguous)
                {
                    // Fast path: axis is contiguous, use SIMD
                    result = ReduceContiguousAxis(axisStart, axisSize, actualOp);
                }
                else
                {
                    // Strided path: axis is not contiguous
                    result = ReduceStridedAxis(axisStart, axisSize, axisStride, actualOp);
                }

                // For Mean, divide by count
                if (op == ReductionOp.Mean)
                    result = DivideByCountTyped(result, axisSize);

                output[outputOffset] = result;
            }
        }

        /// <summary>
        /// Divide a typed value by count (for Mean operation in SIMD path).
        /// </summary>
        private static T DivideByCountTyped<T>(T value, int count) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float result = (float)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(double))
            {
                double result = (double)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(int))
            {
                // Integer division
                int result = (int)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(long))
            {
                long result = (long)(object)value / count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(byte))
            {
                byte result = (byte)((byte)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(short))
            {
                short result = (short)((short)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort result = (ushort)((ushort)(object)value / count);
                return (T)(object)result;
            }
            if (typeof(T) == typeof(uint))
            {
                uint result = (uint)(object)value / (uint)count;
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong result = (ulong)(object)value / (ulong)count;
                return (T)(object)result;
            }
            // Fallback via double
            double dval = ConvertToDouble(value);
            return ConvertFromDouble<T>(dval / count);
        }

        /// <summary>
        /// Reduce a contiguous axis using SIMD.
        /// </summary>
        private static unsafe T ReduceContiguousAxis<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            if (size == 0)
            {
                return GetIdentityValue<T>(op);
            }

            if (size == 1)
            {
                return data[0];
            }

            // Use SIMD for Sum, Prod, Min, Max
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && size >= Vector256<T>.Count)
            {
                return ReduceContiguousAxisSimd256(data, size, op);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && size >= Vector128<T>.Count)
            {
                return ReduceContiguousAxisSimd128(data, size, op);
            }
            else
            {
                return ReduceContiguousAxisScalar(data, size, op);
            }
        }

        /// <summary>
        /// Reduce contiguous axis using Vector256 SIMD.
        /// </summary>
        private static unsafe T ReduceContiguousAxisSimd256<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            int vectorCount = Vector256<T>.Count;
            int vectorEnd = size - vectorCount;

            // Initialize accumulator vector
            var accumVec = CreateIdentityVector256<T>(op);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec = CombineVectors256(accumVec, vec, op);
            }

            // Horizontal reduce the vector
            T result = HorizontalReduce256(accumVec, op);

            // Process scalar tail
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce contiguous axis using Vector128 SIMD.
        /// </summary>
        private static unsafe T ReduceContiguousAxisSimd128<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            int vectorCount = Vector128<T>.Count;
            int vectorEnd = size - vectorCount;

            // Initialize accumulator vector
            var accumVec = CreateIdentityVector128<T>(op);

            int i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector128.Load(data + i);
                accumVec = CombineVectors128(accumVec, vec, op);
            }

            // Horizontal reduce the vector
            T result = HorizontalReduce128(accumVec, op);

            // Process scalar tail
            for (; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce contiguous axis using scalar loop.
        /// </summary>
        private static unsafe T ReduceContiguousAxisScalar<T>(T* data, int size, ReductionOp op)
            where T : unmanaged
        {
            T result = GetIdentityValue<T>(op);

            for (int i = 0; i < size; i++)
            {
                result = CombineScalars(result, data[i], op);
            }

            return result;
        }

        /// <summary>
        /// Reduce a strided axis (non-contiguous).
        /// </summary>
        private static unsafe T ReduceStridedAxis<T>(T* data, int size, int stride, ReductionOp op)
            where T : unmanaged
        {
            T result = GetIdentityValue<T>(op);

            for (int i = 0; i < size; i++)
            {
                result = CombineScalars(result, data[i * stride], op);
            }

            return result;
        }

        /// <summary>
        /// Get the identity value for a reduction operation.
        /// </summary>
        private static T GetIdentityValue<T>(ReductionOp op) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float val = op switch
                {
                    ReductionOp.Sum => 0f,
                    ReductionOp.Prod => 1f,
                    ReductionOp.Min => float.PositiveInfinity,
                    ReductionOp.Max => float.NegativeInfinity,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(double))
            {
                double val = op switch
                {
                    ReductionOp.Sum => 0.0,
                    ReductionOp.Prod => 1.0,
                    ReductionOp.Min => double.PositiveInfinity,
                    ReductionOp.Max => double.NegativeInfinity,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(int))
            {
                int val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => int.MaxValue,
                    ReductionOp.Max => int.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(long))
            {
                long val = op switch
                {
                    ReductionOp.Sum => 0L,
                    ReductionOp.Prod => 1L,
                    ReductionOp.Min => long.MaxValue,
                    ReductionOp.Max => long.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(byte))
            {
                byte val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => byte.MaxValue,
                    ReductionOp.Max => byte.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(short))
            {
                short val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => short.MaxValue,
                    ReductionOp.Max => short.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort val = op switch
                {
                    ReductionOp.Sum => 0,
                    ReductionOp.Prod => 1,
                    ReductionOp.Min => ushort.MaxValue,
                    ReductionOp.Max => ushort.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(uint))
            {
                uint val = op switch
                {
                    ReductionOp.Sum => 0u,
                    ReductionOp.Prod => 1u,
                    ReductionOp.Min => uint.MaxValue,
                    ReductionOp.Max => uint.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong val = op switch
                {
                    ReductionOp.Sum => 0UL,
                    ReductionOp.Prod => 1UL,
                    ReductionOp.Min => ulong.MaxValue,
                    ReductionOp.Max => ulong.MinValue,
                    _ => throw new NotSupportedException()
                };
                return (T)(object)val;
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported for axis reduction");
        }

        /// <summary>
        /// Create identity Vector256 for reduction operation.
        /// </summary>
        private static Vector256<T> CreateIdentityVector256<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector256.Create(identity);
        }

        /// <summary>
        /// Create identity Vector128 for reduction operation.
        /// </summary>
        private static Vector128<T> CreateIdentityVector128<T>(ReductionOp op) where T : unmanaged
        {
            T identity = GetIdentityValue<T>(op);
            return Vector128.Create(identity);
        }

        /// <summary>
        /// Combine two Vector256 values using reduction operation.
        /// </summary>
        private static Vector256<T> CombineVectors256<T>(Vector256<T> a, Vector256<T> b, ReductionOp op)
            where T : unmanaged
        {
            return op switch
            {
                ReductionOp.Sum => Vector256.Add(a, b),
                ReductionOp.Prod => Vector256.Multiply(a, b),
                ReductionOp.Min => Vector256.Min(a, b),
                ReductionOp.Max => Vector256.Max(a, b),
                _ => throw new NotSupportedException()
            };
        }

        /// <summary>
        /// Combine two Vector128 values using reduction operation.
        /// </summary>
        private static Vector128<T> CombineVectors128<T>(Vector128<T> a, Vector128<T> b, ReductionOp op)
            where T : unmanaged
        {
            return op switch
            {
                ReductionOp.Sum => Vector128.Add(a, b),
                ReductionOp.Prod => Vector128.Multiply(a, b),
                ReductionOp.Min => Vector128.Min(a, b),
                ReductionOp.Max => Vector128.Max(a, b),
                _ => throw new NotSupportedException()
            };
        }

        /// <summary>
        /// Horizontal reduce Vector256 to scalar.
        /// </summary>
        private static T HorizontalReduce256<T>(Vector256<T> vec, ReductionOp op) where T : unmanaged
        {
            // First reduce to Vector128
            var lower = vec.GetLower();
            var upper = vec.GetUpper();
            var combined = CombineVectors128(lower, upper, op);

            return HorizontalReduce128(combined, op);
        }

        /// <summary>
        /// Horizontal reduce Vector128 to scalar.
        /// </summary>
        private static T HorizontalReduce128<T>(Vector128<T> vec, ReductionOp op) where T : unmanaged
        {
            int count = Vector128<T>.Count;
            T result = vec.GetElement(0);

            for (int i = 1; i < count; i++)
            {
                result = CombineScalars(result, vec.GetElement(i), op);
            }

            return result;
        }

        /// <summary>
        /// Combine two scalar values using reduction operation.
        /// </summary>
        private static T CombineScalars<T>(T a, T b, ReductionOp op) where T : unmanaged
        {
            if (typeof(T) == typeof(float))
            {
                float fa = (float)(object)a;
                float fb = (float)(object)b;
                float result = op switch
                {
                    ReductionOp.Sum => fa + fb,
                    ReductionOp.Prod => fa * fb,
                    ReductionOp.Min => Math.Min(fa, fb),
                    ReductionOp.Max => Math.Max(fa, fb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(double))
            {
                double da = (double)(object)a;
                double db = (double)(object)b;
                double result = op switch
                {
                    ReductionOp.Sum => da + db,
                    ReductionOp.Prod => da * db,
                    ReductionOp.Min => Math.Min(da, db),
                    ReductionOp.Max => Math.Max(da, db),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(int))
            {
                int ia = (int)(object)a;
                int ib = (int)(object)b;
                int result = op switch
                {
                    ReductionOp.Sum => ia + ib,
                    ReductionOp.Prod => ia * ib,
                    ReductionOp.Min => Math.Min(ia, ib),
                    ReductionOp.Max => Math.Max(ia, ib),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(long))
            {
                long la = (long)(object)a;
                long lb = (long)(object)b;
                long result = op switch
                {
                    ReductionOp.Sum => la + lb,
                    ReductionOp.Prod => la * lb,
                    ReductionOp.Min => Math.Min(la, lb),
                    ReductionOp.Max => Math.Max(la, lb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(byte))
            {
                int ba = (byte)(object)a;
                int bb = (byte)(object)b;
                byte result = op switch
                {
                    ReductionOp.Sum => (byte)(ba + bb),
                    ReductionOp.Prod => (byte)(ba * bb),
                    ReductionOp.Min => (byte)Math.Min(ba, bb),
                    ReductionOp.Max => (byte)Math.Max(ba, bb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(short))
            {
                int sa = (short)(object)a;
                int sb = (short)(object)b;
                short result = op switch
                {
                    ReductionOp.Sum => (short)(sa + sb),
                    ReductionOp.Prod => (short)(sa * sb),
                    ReductionOp.Min => (short)Math.Min(sa, sb),
                    ReductionOp.Max => (short)Math.Max(sa, sb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ushort))
            {
                int usa = (ushort)(object)a;
                int usb = (ushort)(object)b;
                ushort result = op switch
                {
                    ReductionOp.Sum => (ushort)(usa + usb),
                    ReductionOp.Prod => (ushort)(usa * usb),
                    ReductionOp.Min => (ushort)Math.Min(usa, usb),
                    ReductionOp.Max => (ushort)Math.Max(usa, usb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(uint))
            {
                uint ua = (uint)(object)a;
                uint ub = (uint)(object)b;
                uint result = op switch
                {
                    ReductionOp.Sum => ua + ub,
                    ReductionOp.Prod => ua * ub,
                    ReductionOp.Min => Math.Min(ua, ub),
                    ReductionOp.Max => Math.Max(ua, ub),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong ula = (ulong)(object)a;
                ulong ulb = (ulong)(object)b;
                ulong result = op switch
                {
                    ReductionOp.Sum => ula + ulb,
                    ReductionOp.Prod => ula * ulb,
                    ReductionOp.Min => Math.Min(ula, ulb),
                    ReductionOp.Max => Math.Max(ula, ulb),
                    _ => throw new NotSupportedException()
                };
                return (T)(object)result;
            }

            throw new NotSupportedException($"Type {typeof(T)} not supported");
        }

        #endregion

        #region IKernelProvider SIMD Helper Interface Implementation

        /// <inheritdoc />
        unsafe bool IKernelProvider.All<T>(T* data, int size)
        {
            return AllSimdHelper<T>(data, size);
        }

        /// <inheritdoc />
        unsafe bool IKernelProvider.Any<T>(T* data, int size)
        {
            return AnySimdHelper<T>(data, size);
        }

        /// <inheritdoc />
        unsafe void IKernelProvider.FindNonZero<T>(T* data, int size, System.Collections.Generic.List<int> indices)
        {
            NonZeroSimdHelper(data, size, indices);
        }

        /// <inheritdoc />
        NumSharp.Generic.NDArray<int>[] IKernelProvider.ConvertFlatToCoordinates(System.Collections.Generic.List<int> flatIndices, int[] shape)
        {
            return ConvertFlatIndicesToCoordinates(flatIndices, shape);
        }

        /// <inheritdoc />
        unsafe int IKernelProvider.CountTrue(bool* data, int size)
        {
            return CountTrueSimdHelper(data, size);
        }

        /// <inheritdoc />
        unsafe int IKernelProvider.CopyMasked<T>(T* src, bool* mask, T* dest, int size)
        {
            return CopyMaskedElementsHelper(src, mask, dest, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.Variance<T>(T* data, int size, int ddof)
        {
            return VarSimdHelper(data, size, ddof);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.StandardDeviation<T>(T* data, int size, int ddof)
        {
            return StdSimdHelper(data, size, ddof);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanSumFloat(float* data, int size)
        {
            return NanSumSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanSumDouble(double* data, int size)
        {
            return NanSumSimdHelperDouble(data, size);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanProdFloat(float* data, int size)
        {
            return NanProdSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanProdDouble(double* data, int size)
        {
            return NanProdSimdHelperDouble(data, size);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanMinFloat(float* data, int size)
        {
            return NanMinSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanMinDouble(double* data, int size)
        {
            return NanMinSimdHelperDouble(data, size);
        }

        /// <inheritdoc />
        unsafe float IKernelProvider.NanMaxFloat(float* data, int size)
        {
            return NanMaxSimdHelperFloat(data, size);
        }

        /// <inheritdoc />
        unsafe double IKernelProvider.NanMaxDouble(double* data, int size)
        {
            return NanMaxSimdHelperDouble(data, size);
        }

        #endregion
    }
}
