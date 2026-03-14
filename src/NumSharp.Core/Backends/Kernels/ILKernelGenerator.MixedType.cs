using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

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
// ILKernelGenerator.MixedType.cs (THIS FILE)
//   OWNERSHIP: Mixed-type binary operations with type promotion (general case)
//   RESPONSIBILITY:
//     - Handles ALL binary ops regardless of operand types or memory layout
//     - Selects optimal execution path based on stride analysis:
//       * SimdFull: both operands contiguous, same type -> full SIMD
//       * SimdScalarRight/Left: one operand is scalar -> broadcast SIMD
//       * SimdChunk: inner dimension contiguous -> chunked SIMD
//       * General: arbitrary strides -> coordinate-based iteration
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine as the general binary operation handler
//   KEY MEMBERS:
//     - MixedTypeKernel delegate - full signature with strides/shape/ndim
//     - _mixedTypeCache - caches by MixedTypeKernelKey (types, op, path)
//     - GetMixedTypeKernel(), TryGetMixedTypeKernel() - main entry points
//     - GenerateSimdFullKernel(), GenerateSimdScalarRight/LeftKernel()
//     - GenerateSimdChunkKernel(), GenerateGeneralKernel()
//     - EmitScalarFullLoop(), EmitSimdFullLoop(), EmitGeneralLoop(), etc.
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
    /// <summary>
    /// Mixed-type binary operations and IL loop emission.
    /// </summary>
    public static partial class ILKernelGenerator
    {
        #region Mixed-Type Kernel Generation

        /// <summary>
        /// Cache for mixed-type kernels.
        /// Key: MixedTypeKernelKey (LhsType, RhsType, ResultType, Op, Path)
        /// </summary>
        private static readonly ConcurrentDictionary<MixedTypeKernelKey, MixedTypeKernel> _mixedTypeCache = new();

        /// <summary>
        /// Number of mixed-type kernels in cache.
        /// </summary>
        public static int MixedTypeCachedCount => _mixedTypeCache.Count;

        /// <summary>
        /// Get or generate a mixed-type kernel for the specified key.
        /// </summary>
        public static MixedTypeKernel GetMixedTypeKernel(MixedTypeKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _mixedTypeCache.GetOrAdd(key, GenerateMixedTypeKernel);
        }

        /// <summary>
        /// Try to get or generate a mixed-type kernel. Returns null if generation fails.
        /// </summary>
        public static MixedTypeKernel? TryGetMixedTypeKernel(MixedTypeKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                return _mixedTypeCache.GetOrAdd(key, GenerateMixedTypeKernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetMixedTypeKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Generate a mixed-type kernel for the specified key.
        /// </summary>
        private static MixedTypeKernel GenerateMixedTypeKernel(MixedTypeKernelKey key)
        {
            return key.Path switch
            {
                ExecutionPath.SimdFull => GenerateSimdFullKernel(key),
                ExecutionPath.SimdScalarRight => GenerateSimdScalarRightKernel(key),
                ExecutionPath.SimdScalarLeft => GenerateSimdScalarLeftKernel(key),
                ExecutionPath.SimdChunk => GenerateSimdChunkKernel(key),
                ExecutionPath.General => GenerateGeneralKernel(key),
                _ => throw new NotSupportedException($"Path {key.Path} not supported")
            };
        }

        #endregion

        #region Path-Specific Kernel Generation

        /// <summary>
        /// Generate a SimdFull kernel for contiguous arrays (both operands contiguous).
        /// Uses Vector256 SIMD for supported types and operations, scalar loop otherwise.
        /// </summary>
        private static MixedTypeKernel GenerateSimdFullKernel(MixedTypeKernelKey key)
        {
            // MixedTypeKernel signature:
            // void(void* lhs, void* rhs, void* result, long* lhsStrides, long* rhsStrides, int* shape, int ndim, long totalSize)
            var dm = new DynamicMethod(
                name: $"MixedType_SimdFull_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            int resultSize = GetTypeSize(key.ResultType);

            // Can only use SIMD for same-type, supported types, and supported operations
            // Mod doesn't have SIMD support (no Vector256 modulo operator)
            bool canSimd = CanUseSimd(key.ResultType) && key.IsSameType && CanUseSimdForOp(key.Op);

            if (canSimd)
            {
                EmitSimdFullLoop(il, key, lhsSize, rhsSize, resultSize);
            }
            else
            {
                EmitScalarFullLoop(il, key, lhsSize, rhsSize, resultSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<MixedTypeKernel>();
        }

        /// <summary>
        /// Check if operation has SIMD support via Vector256.
        /// </summary>
        private static bool CanUseSimdForOp(BinaryOp op)
        {
            // Add, Subtract, Multiply, Divide have Vector256 operators
            // BitwiseAnd, BitwiseOr, BitwiseXor use Vector256.BitwiseAnd/Or/Xor
            // Mod requires scalar implementation
            return op == BinaryOp.Add || op == BinaryOp.Subtract ||
                   op == BinaryOp.Multiply || op == BinaryOp.Divide ||
                   op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor;
        }

        /// <summary>
        /// Generate a SimdScalarRight kernel (right operand is scalar).
        /// Uses SIMD when LHS type equals result type (no per-element conversion needed).
        /// </summary>
        private static MixedTypeKernel GenerateSimdScalarRightKernel(MixedTypeKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"MixedType_SimdScalarRight_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            int resultSize = GetTypeSize(key.ResultType);

            // Use SIMD when: LHS type == Result type (no per-element conversion needed),
            // result type supports SIMD, and operation has SIMD support
            bool canUseSimd = key.LhsType == key.ResultType &&
                              CanUseSimd(key.ResultType) &&
                              CanUseSimdForOp(key.Op);

            if (canUseSimd)
            {
                EmitSimdScalarRightLoop(il, key, resultSize);
            }
            else
            {
                EmitScalarRightLoop(il, key, lhsSize, rhsSize, resultSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<MixedTypeKernel>();
        }

        /// <summary>
        /// Generate a SimdScalarLeft kernel (left operand is scalar).
        /// Uses SIMD when RHS type equals result type (no per-element conversion needed).
        /// </summary>
        private static MixedTypeKernel GenerateSimdScalarLeftKernel(MixedTypeKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"MixedType_SimdScalarLeft_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            int resultSize = GetTypeSize(key.ResultType);

            // Use SIMD when: RHS type == Result type (no per-element conversion needed),
            // result type supports SIMD, and operation has SIMD support
            bool canUseSimd = key.RhsType == key.ResultType &&
                              CanUseSimd(key.ResultType) &&
                              CanUseSimdForOp(key.Op);

            if (canUseSimd)
            {
                EmitSimdScalarLeftLoop(il, key, resultSize);
            }
            else
            {
                EmitScalarLeftLoop(il, key, lhsSize, rhsSize, resultSize);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<MixedTypeKernel>();
        }

        /// <summary>
        /// Generate a SimdChunk kernel (inner dimension contiguous/broadcast).
        /// </summary>
        private static MixedTypeKernel GenerateSimdChunkKernel(MixedTypeKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"MixedType_SimdChunk_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            int resultSize = GetTypeSize(key.ResultType);

            EmitChunkLoop(il, key, lhsSize, rhsSize, resultSize);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<MixedTypeKernel>();
        }

        /// <summary>
        /// Generate a General kernel (arbitrary strides, coordinate-based iteration).
        /// </summary>
        private static MixedTypeKernel GenerateGeneralKernel(MixedTypeKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"MixedType_General_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            int resultSize = GetTypeSize(key.ResultType);

            EmitGeneralLoop(il, key, lhsSize, rhsSize, resultSize);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<MixedTypeKernel>();
        }

        #endregion

        #region IL Loop Emission

        /// <summary>
        /// Emit a scalar loop for contiguous arrays (no SIMD).
        /// </summary>
        private static void EmitScalarFullLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            // Args: void* lhs (0), void* rhs (1), void* result (2),
            //       long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //       int ndim (6), long totalSize (7)

            var locI = il.DeclareLocal(typeof(long)); // loop counter

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = op(lhs[i], rhs[i])
            // Load result address
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load lhs[i] and convert to result type
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load rhs[i] and convert to result type
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // Perform operation
            EmitScalarOperation(il, key.Op, key.ResultType);

            // Store result
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit a SIMD loop for contiguous same-type arrays with 4x unrolling.
        /// </summary>
        private static void EmitSimdFullLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            // For same-type operations, use Vector256
            int vectorCount = GetVectorCount(key.ResultType);
            int unrollStep = vectorCount * 4;

            var locI = il.DeclareLocal(typeof(long)); // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(long)); // totalSize - vectorCount (for remainder loop)
            var locUnrollEnd = il.DeclareLocal(typeof(long)); // totalSize - vectorCount*4 (for 4x unrolled loop)

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollLoopEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // vectorEnd = totalSize - vectorCount (for remainder loop)
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // unrollEnd = totalSize - vectorCount*4 (for 4x unrolled loop)
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I4, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // === 4x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnrollLoop);

            // if (i > unrollEnd) goto UnrollLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollLoopEnd);

            // Process 4 vectors per iteration
            for (int u = 0; u < 4; u++)
            {
                int offset = vectorCount * u;

                // Load lhs vector at (i + offset)
                il.Emit(OpCodes.Ldarg_0); // lhs
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, lhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.LhsType);

                // Load rhs vector at (i + offset)
                il.Emit(OpCodes.Ldarg_1); // rhs
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, rhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.RhsType);

                // Vector operation
                EmitVectorOperation(il, key.Op, key.ResultType);

                // Store result vector at (i + offset)
                il.Emit(OpCodes.Ldarg_2); // result
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, resultSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorStore(il, key.ResultType);
            }

            // i += vectorCount * 4
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

            // Load lhs vector
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.LhsType);

            // Load rhs vector
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.RhsType);

            // Vector operation
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store result vector
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblRemainderLoop);
            il.MarkLabel(lblRemainderLoopEnd);

            // === TAIL LOOP (scalar) ===
            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // result[i] = op(lhs[i], rhs[i])
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);

            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);
        }

        /// <summary>
        /// Emit loop for scalar right operand (broadcast scalar to array).
        /// </summary>
        private static void EmitScalarRightLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            var locI = il.DeclareLocal(typeof(long)); // loop counter
            var locRhsVal = il.DeclareLocal(GetClrType(key.ResultType)); // scalar value

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Load rhs[0] and convert to result type, store in local
            il.Emit(OpCodes.Ldarg_1); // rhs
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);
            il.Emit(OpCodes.Stloc, locRhsVal);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = op(lhs[i], rhsVal)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load lhs[i] and convert
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load cached rhs scalar
            il.Emit(OpCodes.Ldloc, locRhsVal);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit loop for scalar left operand (broadcast scalar to array).
        /// </summary>
        private static void EmitScalarLeftLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            var locI = il.DeclareLocal(typeof(long)); // loop counter
            var locLhsVal = il.DeclareLocal(GetClrType(key.ResultType)); // scalar value

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Load lhs[0] and convert to result type, store in local
            il.Emit(OpCodes.Ldarg_0); // lhs
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);
            il.Emit(OpCodes.Stloc, locLhsVal);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = op(lhsVal, rhs[i])
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load cached lhs scalar
            il.Emit(OpCodes.Ldloc, locLhsVal);

            // Load rhs[i] and convert
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit SIMD loop for scalar right operand (broadcast scalar to vector).
        /// Requires: LHS type == Result type (no per-element conversion needed).
        /// </summary>
        private static void EmitSimdScalarRightLoop(ILGenerator il, MixedTypeKernelKey key, int elemSize)
        {
            // Args: void* lhs (0), void* rhs (1), void* result (2),
            //       long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //       int ndim (6), long totalSize (7)

            int vectorCount = GetVectorCount(key.ResultType);
            var clrType = GetClrType(key.ResultType);
            var vectorType = GetVectorType(clrType);

            var locI = il.DeclareLocal(typeof(long));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(long));   // totalSize - vectorCount
            var locScalarVec = il.DeclareLocal(vectorType);    // broadcasted scalar vector

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // === Load scalar, convert to result type, broadcast to vector ===
            // Load rhs[0] (the scalar)
            il.Emit(OpCodes.Ldarg_1); // rhs
            EmitLoadIndirect(il, key.RhsType);
            // Convert to result type if needed
            if (key.RhsType != key.ResultType)
            {
                EmitConvertTo(il, key.RhsType, key.ResultType);
            }
            // Broadcast to Vector256: Vector256.Create(scalar)
            EmitVectorCreate(il, key.ResultType);
            il.Emit(OpCodes.Stloc, locScalarVec);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
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

            // Load lhs vector: Vector256.Load(lhs + i * elemSize)
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.LhsType);

            // Load scalar vector
            il.Emit(OpCodes.Ldloc, locScalarVec);

            // Vector operation: lhsVec op scalarVec
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store result vector: Vector256.Store(resultVec, result + i * elemSize)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === TAIL LOOP (scalar remainder) ===
            // Load scalar value once for tail loop
            var locScalarVal = il.DeclareLocal(clrType);
            il.Emit(OpCodes.Ldarg_1); // rhs
            EmitLoadIndirect(il, key.RhsType);
            if (key.RhsType != key.ResultType)
            {
                EmitConvertTo(il, key.RhsType, key.ResultType);
            }
            il.Emit(OpCodes.Stloc, locScalarVal);

            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // result[i] = lhs[i] op scalarVal
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);

            il.Emit(OpCodes.Ldloc, locScalarVal);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);
        }

        /// <summary>
        /// Emit SIMD loop for scalar left operand (broadcast scalar to vector).
        /// Requires: RHS type == Result type (no per-element conversion needed).
        /// </summary>
        private static void EmitSimdScalarLeftLoop(ILGenerator il, MixedTypeKernelKey key, int elemSize)
        {
            // Args: void* lhs (0), void* rhs (1), void* result (2),
            //       long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //       int ndim (6), long totalSize (7)

            int vectorCount = GetVectorCount(key.ResultType);
            var clrType = GetClrType(key.ResultType);
            var vectorType = GetVectorType(clrType);

            var locI = il.DeclareLocal(typeof(long));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(long));   // totalSize - vectorCount
            var locScalarVec = il.DeclareLocal(vectorType);    // broadcasted scalar vector

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // === Load scalar, convert to result type, broadcast to vector ===
            // Load lhs[0] (the scalar)
            il.Emit(OpCodes.Ldarg_0); // lhs
            EmitLoadIndirect(il, key.LhsType);
            // Convert to result type if needed
            if (key.LhsType != key.ResultType)
            {
                EmitConvertTo(il, key.LhsType, key.ResultType);
            }
            // Broadcast to Vector256: Vector256.Create(scalar)
            EmitVectorCreate(il, key.ResultType);
            il.Emit(OpCodes.Stloc, locScalarVec);

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
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

            // Load scalar vector
            il.Emit(OpCodes.Ldloc, locScalarVec);

            // Load rhs vector: Vector256.Load(rhs + i * elemSize)
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.RhsType);

            // Vector operation: scalarVec op rhsVec
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store result vector: Vector256.Store(resultVec, result + i * elemSize)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === TAIL LOOP (scalar remainder) ===
            // Load scalar value once for tail loop
            var locScalarVal = il.DeclareLocal(clrType);
            il.Emit(OpCodes.Ldarg_0); // lhs
            EmitLoadIndirect(il, key.LhsType);
            if (key.LhsType != key.ResultType)
            {
                EmitConvertTo(il, key.LhsType, key.ResultType);
            }
            il.Emit(OpCodes.Stloc, locScalarVal);

            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // result[i] = scalarVal op rhs[i]
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldloc, locScalarVal);

            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);
        }

        /// <summary>
        /// Emit chunked loop for inner-contiguous arrays.
        /// This is more complex - processes the inner dimension as a chunk.
        /// </summary>
        private static void EmitChunkLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            // For simplicity in initial implementation, use general loop
            // TODO: Implement proper chunked SIMD processing
            EmitGeneralLoop(il, key, lhsSize, rhsSize, resultSize);
        }

        /// <summary>
        /// Emit general coordinate-based iteration loop.
        /// Handles arbitrary strides.
        /// </summary>
        private static void EmitGeneralLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            // Args: void* lhs (0), void* rhs (1), void* result (2),
            //       long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //       int ndim (6), long totalSize (7)

            var locI = il.DeclareLocal(typeof(long)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locLhsOffset = il.DeclareLocal(typeof(long)); // lhs offset
            var locRhsOffset = il.DeclareLocal(typeof(long)); // rhs offset
            var locCoord = il.DeclareLocal(typeof(long)); // current coordinate (long for int64 shapes)
            var locIdx = il.DeclareLocal(typeof(long)); // temp for coordinate calculation (long for int64 shapes)

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
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate lhsOffset and rhsOffset from linear index
            // lhsOffset = 0, rhsOffset = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Stloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Stloc, locRhsOffset);

            // idx = i (for coordinate calculation)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            // For each dimension (right to left): coord = idx % shape[d], idx /= shape[d]
            // d = ndim - 1
            il.Emit(OpCodes.Ldarg_S, (byte)6); // ndim
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
            il.Emit(OpCodes.Ldarg_S, (byte)5); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8); // sizeof(long)
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locCoord);

            // idx /= shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locIdx);

            // lhsOffset += coord * lhsStrides[d]
            il.Emit(OpCodes.Ldloc, locLhsOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_3); // lhsStrides
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locLhsOffset);

            // rhsOffset += coord * rhsStrides[d]
            il.Emit(OpCodes.Ldloc, locRhsOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_S, (byte)4); // rhsStrides
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locRhsOffset);

            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.Emit(OpCodes.Br, lblDimLoop);
            il.MarkLabel(lblDimLoopEnd);

            // Now compute: result[i] = op(lhs[lhsOffset], rhs[rhsOffset])
            // Load result address (contiguous output)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load lhs[lhsOffset]
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locLhsOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load rhs[rhsOffset]
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locRhsOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // Operation
            EmitScalarOperation(il, key.Op, key.ResultType);

            // Store
            EmitStoreIndirect(il, key.ResultType);

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
