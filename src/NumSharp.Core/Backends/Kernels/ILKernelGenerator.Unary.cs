using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Unary.cs - Unary Kernel Infrastructure
// =============================================================================
//
// RESPONSIBILITY:
//   - Unary kernel cache and API (GetUnaryKernel, TryGetUnaryKernel)
//   - SIMD loop emission with 4x unrolling
//   - Scalar and strided fallback loops
//   - Capability detection (CanUseUnarySimd, IsPredicateOp)
//
// RELATED FILES:
//   - ILKernelGenerator.Unary.Math.cs - Math function emission
//   - ILKernelGenerator.Unary.Predicate.cs - IsNaN/IsFinite/IsInf
//   - ILKernelGenerator.Unary.Decimal.cs - Decimal operations
//   - ILKernelGenerator.Unary.Vector.cs - SIMD vector operations
//   - ILKernelGenerator.Scalar.cs - Scalar kernel delegates
//
// =============================================================================

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
// ILKernelGenerator.Unary.cs (THIS FILE)
//   OWNERSHIP: Unary element-wise operations and scalar delegates
//   RESPONSIBILITY:
//     - Array kernels for unary math: Negate, Abs, Sqrt, Sin, Cos, Exp, Log,
//       Sign, Floor, Ceil, Round, Tan, Sinh, Cosh, Tanh, ASin, ACos, ATan,
//       Exp2, Expm1, Log2, Log10, Log1p
//     - SIMD support for Negate, Abs, Sqrt, Floor, Ceil on float/double
//     - Scalar delegates (Func<TIn, TOut>) for single-value operations
//     - Binary scalar delegates (Func<TLhs, TRhs, TResult>) for mixed-type scalars
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW:
//     - Array kernels: Called by DefaultEngine for np.sqrt, np.sin, etc.
//     - Scalar delegates: Used internally for broadcasting and element access
//   KEY MEMBERS:
//     - UnaryKernel delegate, _unaryCache - array operations
//     - _unaryScalarCache - Func<TIn, TOut> for scalar unary ops
//     - _binaryScalarCache - Func<TLhs, TRhs, TResult> for scalar binary ops
//     - GetUnaryKernel(), TryGetUnaryKernel(), ClearUnary()
//     - GetUnaryScalarDelegate(), GetBinaryScalarDelegate()
//     - EmitUnaryScalarOperation() - central dispatcher for all unary ops
//     - EmitMathCall(), EmitSignCall() - Math/MathF function emission
//     - EmitUnarySimdLoop(), EmitUnaryScalarLoop(), EmitUnaryStridedLoop()
//
// ILKernelGenerator.Comparison.cs
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: ==, !=, <, >, <=, >=
//     - SIMD comparison with efficient mask-to-bool extraction
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by NDArray comparison operators
//
// ILKernelGenerator.Reduction.cs
//   OWNERSHIP: Reduction operations and specialized SIMD helpers
//   RESPONSIBILITY:
//     - Reductions: Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any
//     - SIMD helpers called directly by np.all/any/nonzero/masking
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Kernels called by DefaultEngine; helpers called directly by np.*
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Unary Kernel Generation

        /// <summary>
        /// Cache for unary kernels.
        /// Key: UnaryKernelKey (InputType, OutputType, Op, IsContiguous)
        /// </summary>
        private static readonly ConcurrentDictionary<UnaryKernelKey, UnaryKernel> _unaryCache = new();

        /// <summary>
        /// Number of unary kernels in cache.
        /// </summary>
        public static int UnaryCachedCount => _unaryCache.Count;

        /// <summary>
        /// Get or generate a unary kernel for the specified key.
        /// </summary>
        public static UnaryKernel GetUnaryKernel(UnaryKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _unaryCache.GetOrAdd(key, GenerateUnaryKernel);
        }

        /// <summary>
        /// Try to get or generate a unary kernel. Returns null if generation fails.
        /// </summary>
        public static UnaryKernel? TryGetUnaryKernel(UnaryKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                return _unaryCache.GetOrAdd(key, GenerateUnaryKernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetUnaryKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Generate a unary kernel for the specified key.
        /// </summary>
        private static UnaryKernel GenerateUnaryKernel(UnaryKernelKey key)
        {
            // UnaryKernel signature:
            // void(void* input, void* output, int* strides, int* shape, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"Unary_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*),
                    typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int inputSize = GetTypeSize(key.InputType);
            int outputSize = GetTypeSize(key.OutputType);

            if (key.IsContiguous)
            {
                // Check if we can use SIMD for this operation
                bool canSimd = CanUseUnarySimd(key);
                if (canSimd)
                {
                    EmitUnarySimdLoop(il, key, inputSize, outputSize);
                }
                else
                {
                    EmitUnaryScalarLoop(il, key, inputSize, outputSize);
                }
            }
            else
            {
                EmitUnaryStridedLoop(il, key, inputSize, outputSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<UnaryKernel>();
        }

        /// <summary>
        /// Check if this is a predicate operation (returns bool based on input type).
        /// These operations should NOT convert input to output type before the operation.
        /// </summary>
        private static bool IsPredicateOp(UnaryOp op)
        {
            return op == UnaryOp.IsFinite || op == UnaryOp.IsNan || op == UnaryOp.IsInf;
        }

        /// <summary>
        /// Check if SIMD can be used for this unary operation.
        /// </summary>
        private static bool CanUseUnarySimd(UnaryKernelKey key)
        {
            // SIMD only for same-type operations
            if (!key.IsSameType)
                return false;

            // BitwiseNot works for integer types only (and bool)
            if (key.Op == UnaryOp.BitwiseNot)
            {
                return key.InputType != NPTypeCode.Single &&
                       key.InputType != NPTypeCode.Double &&
                       key.InputType != NPTypeCode.Decimal;
            }

            // LogicalNot is boolean-only, no SIMD (uses scalar comparison)
            if (key.Op == UnaryOp.LogicalNot)
                return false;

            // Float/double operations with SIMD support
            if (key.InputType != NPTypeCode.Single && key.InputType != NPTypeCode.Double)
                return false;

            // Operations with SIMD support for float/double
            return key.Op == UnaryOp.Negate || key.Op == UnaryOp.Abs || key.Op == UnaryOp.Sqrt ||
                   key.Op == UnaryOp.Floor || key.Op == UnaryOp.Ceil || key.Op == UnaryOp.Round ||
                   key.Op == UnaryOp.Truncate || key.Op == UnaryOp.Reciprocal || key.Op == UnaryOp.Square ||
                   key.Op == UnaryOp.Deg2Rad || key.Op == UnaryOp.Rad2Deg;
        }

        /// <summary>
        /// Emit SIMD loop for contiguous unary operations with 4x unrolling.
        /// </summary>
        private static void EmitUnarySimdLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            int vectorCount = GetVectorCount(key.InputType);
            int unrollFactor = 4;
            int unrollStep = vectorCount * unrollFactor;

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locUnrollEnd = il.DeclareLocal(typeof(int)); // totalSize - unrollStep
            var locVectorEnd = il.DeclareLocal(typeof(int)); // totalSize - vectorCount

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollLoopEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // unrollEnd = totalSize - unrollStep
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Ldc_I4, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // === 4x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnrollLoop);

            // if (i > unrollEnd) goto UnrollLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollLoopEnd);

            // Process 4 vectors
            for (int n = 0; n < unrollFactor; n++)
            {
                int offset = n * vectorCount;

                // Load input vector at (i + offset) * inputSize
                il.Emit(OpCodes.Ldarg_0); // input
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, inputSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.InputType);

                // Vector operation
                EmitUnaryVectorOperation(il, key.Op, key.InputType);

                // Store result vector at (i + offset) * outputSize
                il.Emit(OpCodes.Ldarg_1); // output
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, outputSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorStore(il, key.OutputType);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblUnrollLoop);
            il.MarkLabel(lblUnrollLoopEnd);

            // === REMAINDER SIMD LOOP (0-3 vectors) ===
            il.MarkLabel(lblRemainderLoop);

            // if (i > vectorEnd) goto RemainderLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemainderLoopEnd);

            // Load input vector
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.InputType);

            // Vector operation
            EmitUnaryVectorOperation(il, key.Op, key.InputType);

            // Store result vector
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.OutputType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblRemainderLoop);
            il.MarkLabel(lblRemainderLoopEnd);

            // === SCALAR TAIL LOOP ===
            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // output[i] = op(input[i])
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);

            // For predicate operations (IsFinite, IsNan, IsInf), operate on INPUT type
            if (IsPredicateOp(key.Op))
            {
                EmitUnaryScalarOperation(il, key.Op, key.InputType);
            }
            else
            {
                EmitConvertTo(il, key.InputType, key.OutputType);
                EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            }

            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);
        }

        /// <summary>
        /// Emit scalar loop for contiguous unary operations (no SIMD).
        /// </summary>
        private static void EmitUnaryScalarLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            // Args: void* input (0), void* output (1),
            //       int* strides (2), int* shape (3),
            //       int ndim (4), int totalSize (5)

            var locI = il.DeclareLocal(typeof(int)); // loop counter

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // output[i] = op(input[i])
            // Load output address
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load input[i]
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);

            // For predicate operations (IsFinite, IsNan, IsInf), operate on INPUT type
            // and the operation itself produces bool. For other ops, convert first.
            if (IsPredicateOp(key.Op))
            {
                // Perform operation on input type - produces bool
                EmitUnaryScalarOperation(il, key.Op, key.InputType);
            }
            else
            {
                // Convert to output type, then perform operation
                EmitConvertTo(il, key.InputType, key.OutputType);
                EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            }

            // Store result
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
        /// Emit strided loop for non-contiguous unary operations.
        /// Uses coordinate-based iteration.
        /// </summary>
        private static void EmitUnaryStridedLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            // Args: void* input (0), void* output (1),
            //       int* strides (2), int* shape (3),
            //       int ndim (4), int totalSize (5)

            var locI = il.DeclareLocal(typeof(int)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locInputOffset = il.DeclareLocal(typeof(int)); // input offset
            var locCoord = il.DeclareLocal(typeof(int)); // current coordinate
            var locIdx = il.DeclareLocal(typeof(int)); // temp for coordinate calculation

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblDimLoop = il.DefineLabel();
            var lblDimLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // Main loop
            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate inputOffset from linear index
            // inputOffset = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locInputOffset);

            // idx = i (for coordinate calculation)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            // For each dimension (right to left): coord = idx % shape[d], idx /= shape[d]
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
            il.Emit(OpCodes.Ldc_I4_4); // sizeof(int)
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

            // inputOffset += coord * strides[d]
            il.Emit(OpCodes.Ldloc, locInputOffset);
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
            il.Emit(OpCodes.Stloc, locInputOffset);

            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.Emit(OpCodes.Br, lblDimLoop);
            il.MarkLabel(lblDimLoopEnd);

            // Now compute: output[i] = op(input[inputOffset])
            // Load output address (contiguous output)
            il.Emit(OpCodes.Ldarg_1); // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, outputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load input[inputOffset]
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locInputOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);

            // For predicate operations (IsFinite, IsNan, IsInf), operate on INPUT type
            if (IsPredicateOp(key.Op))
            {
                EmitUnaryScalarOperation(il, key.Op, key.InputType);
            }
            else
            {
                EmitConvertTo(il, key.InputType, key.OutputType);
                EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            }

            // Store
            EmitStoreIndirect(il, key.OutputType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        #endregion
    }
}
