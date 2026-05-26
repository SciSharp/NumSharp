using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator - IL-based SIMD kernel generation using DynamicMethod
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
//         -> DirectILKernelGenerator checks cache, generates IL if needed
//         -> Returns delegate that caller invokes with array pointers
//
// =============================================================================
// PARTIAL CLASS FILES
// =============================================================================
//
// DirectILKernelGenerator.cs
//   OWNERSHIP: Core infrastructure - foundation for all other partial files
//   RESPONSIBILITY:
//     - Global state: Enabled flag, VectorBits/VectorBytes (detected at startup)
//     - Type mapping: NPTypeCode <-> CLR Type <-> Vector type conversions
//     - Shared IL emission primitives used by all other partials
//   DEPENDENCIES: None (other partials depend on this)
//
// DirectILKernelGenerator.Binary.cs
//   OWNERSHIP: Same-type binary operations on contiguous arrays (fast path)
//   RESPONSIBILITY:
//     - Optimized kernels when both operands have identical type and layout
//     - SIMD loop + scalar tail for Add, Sub, Mul, Div
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for same-type contiguous operations
//
// DirectILKernelGenerator.MixedType.cs (THIS FILE)
//   OWNERSHIP: Mixed-type binary operations with type promotion (general case)
//   RESPONSIBILITY:
//     - Handles ALL binary ops regardless of operand types or memory layout
//     - Selects optimal execution path based on stride analysis:
//       * SimdFull: both operands contiguous, same type -> full SIMD
//       * SimdScalarRight/Left: one operand is scalar -> broadcast SIMD
//       * SimdChunk: inner dimension contiguous -> chunked SIMD
//       * General: arbitrary strides -> coordinate-based iteration
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by DefaultEngine as the general binary operation handler
//   KEY MEMBERS:
//     - MixedTypeKernel delegate - full signature with strides/shape/ndim
//     - _mixedTypeCache - caches by MixedTypeKernelKey (types, op, path)
//     - GetMixedTypeKernel(), TryGetMixedTypeKernel() - main entry points
//     - GenerateSimdFullKernel(), GenerateSimdScalarRight/LeftKernel()
//     - GenerateSimdChunkKernel(), GenerateGeneralKernel()
//     - EmitScalarFullLoop(), EmitSimdFullLoop(), EmitGeneralLoop(), etc.
//
// DirectILKernelGenerator.Unary.cs
//   OWNERSHIP: Unary element-wise operations
//   RESPONSIBILITY:
//     - Math functions: Negate, Abs, Sqrt, Sin, Cos, Exp, Log, Sign, Floor, Ceil, etc.
//     - Scalar delegate generation for single-value operations (Func<TIn,TOut>)
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for unary ops; scalar delegates used in broadcasting
//
// DirectILKernelGenerator.Comparison.cs
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: ==, !=, <, >, <=, >=
//     - SIMD comparison with efficient mask-to-bool extraction
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by NDArray comparison operators
//
// DirectILKernelGenerator.Reduction.cs
//   OWNERSHIP: Reduction operations and specialized SIMD helpers
//   RESPONSIBILITY:
//     - Reductions: Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any
//     - SIMD helpers called directly by np.all/any/nonzero/masking
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Kernels called by DefaultEngine; helpers called directly by np.*
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Mixed-type binary operations and IL loop emission.
    /// </summary>
    public static partial class DirectILKernelGenerator
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
        [Obsolete("Unused. Callers use GetMixedTypeKernel directly. Marked obsolete pending removal.", error: true)]
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
            // void(void* lhs, void* rhs, void* result, long* lhsStrides, long* rhsStrides, long* shape, int ndim, long totalSize)
            var dm = new DynamicMethod(
                name: $"MixedType_SimdFull_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(DirectILKernelGenerator),
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
        internal static bool CanUseSimdForOp(BinaryOp op)
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
                owner: typeof(DirectILKernelGenerator),
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
                owner: typeof(DirectILKernelGenerator),
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
                owner: typeof(DirectILKernelGenerator),
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
                owner: typeof(DirectILKernelGenerator),
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
            il.Emit(OpCodes.Ldc_I8, 0L);
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
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load lhs[i] and convert to result type
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load rhs[i] and convert to result type
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // Perform operation
            EmitScalarOperation(il, key.Op, key.ResultType);

            // Store result
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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
            long vectorCount = GetVectorCount(key.ResultType);
            long unrollStep = vectorCount * 4;

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
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // unrollEnd = totalSize - vectorCount*4 (for 4x unrolled loop)
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
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
                long offset = vectorCount * u;

                // Load lhs vector at (i + offset)
                il.Emit(OpCodes.Ldarg_0); // lhs
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.LhsType);

                // Load rhs vector at (i + offset)
                il.Emit(OpCodes.Ldarg_1); // rhs
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.RhsType);

                // Vector operation
                EmitVectorOperation(il, key.Op, key.ResultType);

                // Store result vector at (i + offset)
                il.Emit(OpCodes.Ldarg_2); // result
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldc_I8, (long)resultSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorStore(il, key.ResultType);
            }

            // i += vectorCount * 4
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
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
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.LhsType);

            // Load rhs vector
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.RhsType);

            // Vector operation
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store result vector
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
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
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);

            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = op(lhs[i], rhsVal)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load lhs[i] and convert
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load cached rhs scalar
            il.Emit(OpCodes.Ldloc, locRhsVal);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = op(lhsVal, rhs[i])
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load cached lhs scalar
            il.Emit(OpCodes.Ldloc, locLhsVal);

            // Load rhs[i] and convert
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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

            long vectorCount = GetVectorCount(key.ResultType);
            var clrType = GetClrType(key.ResultType);
            var vectorType = VectorMethodCache.V(VectorBits, clrType);

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
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
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
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.LhsType);

            // Load scalar vector
            il.Emit(OpCodes.Ldloc, locScalarVec);

            // Vector operation: lhsVec op scalarVec
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store result vector: Vector256.Store(resultVec, result + i * elemSize)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
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
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);

            il.Emit(OpCodes.Ldloc, locScalarVal);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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

            long vectorCount = GetVectorCount(key.ResultType);
            var clrType = GetClrType(key.ResultType);
            var vectorType = VectorMethodCache.V(VectorBits, clrType);

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
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
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
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, key.RhsType);

            // Vector operation: scalarVec op rhsVec
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store result vector: Vector256.Store(resultVec, result + i * elemSize)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
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
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldloc, locScalarVal);

            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);
        }

        /// <summary>
        /// Emit chunked loop for inner-contiguous arrays.
        /// This is more complex - processes the inner dimension as a chunk.
        /// </summary>
        /// <summary>
        ///     L3-a: Real SimdChunk kernel. Outer loop walks the (ndim-1) outer dims,
        ///     computing per-row offsets via mod/div ONCE per row (not per element).
        ///     Inner loop iterates the innermost dim with a tight scalar load + op +
        ///     store sequence — addresses computed by simple multiply-add (no mod/div).
        ///
        ///     This is the key win over the pre-L3-a stub that called EmitGeneralLoop:
        ///     General does mod+div for EVERY element (e.g. 4 expensive ops/element on
        ///     2-D), whereas chunk amortizes them across innerSize elements per outer.
        ///     For typical 2-D broadcast and strided patterns this drops the per-call
        ///     time from ~10000us to ~1500us (roughly 6× faster).
        ///
        ///     Inner-stride dispatch is implicit: <c>lhsInnerStride * i</c> evaluates
        ///     to 0 when the operand is broadcast on the inner dim (stride=0), so a
        ///     single emitted loop handles {contig, contig}, {bcast, contig},
        ///     {contig, bcast}, and {bcast, bcast} (the last is dead because that's
        ///     <see cref="ExecutionPath.SimdScalarLeft"/>/<see cref="ExecutionPath.SimdScalarRight"/>).
        /// </summary>
        private static void EmitChunkLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            // Args (8 total): void* lhs (0), void* rhs (1), void* result (2),
            //                 long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //                 int ndim (6), long totalSize (7)

            var locInnerSize  = il.DeclareLocal(typeof(long));  // shape[ndim-1]
            var locOuterTotal = il.DeclareLocal(typeof(long));  // totalSize / innerSize
            var locLhsInner   = il.DeclareLocal(typeof(long));  // lhsStrides[ndim-1]
            var locRhsInner   = il.DeclareLocal(typeof(long));  // rhsStrides[ndim-1]
            var locOuterNdim  = il.DeclareLocal(typeof(int));   // ndim - 1
            var locO          = il.DeclareLocal(typeof(long));  // outer index
            var locRemaining  = il.DeclareLocal(typeof(long));  // remaining for decomposition
            var locLhsRowOff  = il.DeclareLocal(typeof(long));  // per-row lhs offset (elements)
            var locRhsRowOff  = il.DeclareLocal(typeof(long));  // per-row rhs offset (elements)
            var locD          = il.DeclareLocal(typeof(int));   // outer-dim counter
            var locCoord      = il.DeclareLocal(typeof(long));  // current dim coordinate
            var locI          = il.DeclareLocal(typeof(long));  // inner index
            var locResBase    = il.DeclareLocal(typeof(long));  // o * innerSize (result element offset)

            var lblOuterLoop    = il.DefineLabel();
            var lblOuterEnd     = il.DefineLabel();
            var lblOuterDimLoop = il.DefineLabel();
            var lblOuterDimEnd  = il.DefineLabel();
            var lblInnerLoop    = il.DefineLabel();
            var lblInnerEnd     = il.DefineLabel();

            // ─── innerSize = shape[ndim-1] ───────────────────────────────────────
            il.Emit(OpCodes.Ldarg_S, (byte)5); // shape
            il.Emit(OpCodes.Ldarg_S, (byte)6); // ndim
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);          // sizeof(long)
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerSize);

            // ─── outerTotal = totalSize / innerSize ──────────────────────────────
            il.Emit(OpCodes.Ldarg_S, (byte)7);  // totalSize
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locOuterTotal);

            // ─── lhsInner = lhsStrides[ndim-1] ───────────────────────────────────
            il.Emit(OpCodes.Ldarg_3);           // lhsStrides
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locLhsInner);

            // ─── rhsInner = rhsStrides[ndim-1] ───────────────────────────────────
            il.Emit(OpCodes.Ldarg_S, (byte)4);  // rhsStrides
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locRhsInner);

            // ─── outerNdim = ndim - 1 ────────────────────────────────────────────
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOuterNdim);

            // ─── o = 0 ───────────────────────────────────────────────────────────
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locO);

            // ════════════════════════ OUTER LOOP ═════════════════════════════════
            il.MarkLabel(lblOuterLoop);

            // if (o >= outerTotal) goto outerEnd
            il.Emit(OpCodes.Ldloc, locO);
            il.Emit(OpCodes.Ldloc, locOuterTotal);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // ─── Decompose o into outer coords, accumulate row offsets ───────────
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locLhsRowOff);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locRhsRowOff);

            il.Emit(OpCodes.Ldloc, locO);
            il.Emit(OpCodes.Stloc, locRemaining);

            // d = outerNdim - 1 (walk right to left)
            il.Emit(OpCodes.Ldloc, locOuterNdim);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.MarkLabel(lblOuterDimLoop);

            // if (d < 0) goto outerDimEnd
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Blt, lblOuterDimEnd);

            // coord = remaining % shape[d]
            il.Emit(OpCodes.Ldloc, locRemaining);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locCoord);

            // remaining = remaining / shape[d]
            il.Emit(OpCodes.Ldloc, locRemaining);
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locRemaining);

            // lhsRowOff += coord * lhsStrides[d]
            il.Emit(OpCodes.Ldloc, locLhsRowOff);
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
            il.Emit(OpCodes.Stloc, locLhsRowOff);

            // rhsRowOff += coord * rhsStrides[d]
            il.Emit(OpCodes.Ldloc, locRhsRowOff);
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
            il.Emit(OpCodes.Stloc, locRhsRowOff);

            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            il.Emit(OpCodes.Br, lblOuterDimLoop);
            il.MarkLabel(lblOuterDimEnd);

            // resBase = o * innerSize  (result is always C-contig; output position is linear)
            il.Emit(OpCodes.Ldloc, locO);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locResBase);

            // i = 0 (used by both scalar inner and SIMD-then-scalar-tail; must be
            // initialized BEFORE any branch that could jump to lblInnerLoop or
            // lblSimdInner).
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ─── L3-c: SIMD inner branch ─────────────────────────────────────────
            // At emit time we know if (lhsType==rhsType==resultType && SIMD-capable
            // && op has SIMD support). When yes, also check at runtime that both
            // inner strides are 1 (contig+contig) — that covers the broadcast
            // (1,N)+(M,N) case and the bog-standard 2-D add. Falls to scalar inner
            // for stride>1 (general strided), stride==0 (M,1 broadcast), and any
            // emit-time mismatch.
            bool canSimdInner =
                key.LhsType == key.RhsType && key.LhsType == key.ResultType &&
                CanUseSimd(key.ResultType) && CanUseSimdForOp(key.Op);

            var lblSimdInner    = il.DefineLabel(); // contig+contig
            var lblSimdScalarL  = il.DefineLabel(); // lhs broadcast (inner=0), rhs contig
            var lblSimdScalarR  = il.DefineLabel(); // lhs contig, rhs broadcast (inner=0)
            var lblSimdInnerEnd = il.DefineLabel();

            if (canSimdInner)
            {
                // ── 4-way runtime dispatch on (lhsInner, rhsInner) ─────────────
                // We branch into one of {simdCC, simdScalarL, simdScalarR, scalarInner}
                // based on the (1, 1), (0, 1), (1, 0), or other combination.
                // (0, 0) is theoretically possible but ClassifyPath sends that to
                // SimdScalarLeft/Right earlier — so we just fall to scalar inner.
                var lblCheckScalarL = il.DefineLabel();
                var lblCheckScalarR = il.DefineLabel();

                // if (lhsInner == 1) jump to check rhsInner against {1, 0}
                il.Emit(OpCodes.Ldloc, locLhsInner);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Bne_Un, lblCheckScalarL); // lhsInner != 1 → check SL
                il.Emit(OpCodes.Ldloc, locRhsInner);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Beq, lblSimdInner);        // (1,1) → CC
                il.Emit(OpCodes.Ldloc, locRhsInner);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblSimdScalarR);      // (1,0) → SR
                il.Emit(OpCodes.Br, lblInnerLoop);         // (1, other) → scalar

                // lhsInner != 1: check if it's 0
                il.MarkLabel(lblCheckScalarL);
                il.Emit(OpCodes.Ldloc, locLhsInner);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, lblInnerLoop);     // lhsInner != 0 either → scalar
                il.Emit(OpCodes.Ldloc, locRhsInner);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Beq, lblSimdScalarL);      // (0,1) → SL
                il.Emit(OpCodes.Br, lblInnerLoop);         // (0, other) → scalar
            }

            // ════════════════════════ SCALAR INNER LOOP ══════════════════════════
            il.MarkLabel(lblInnerLoop);

            // if (i >= innerSize) goto innerEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // result_addr = result + (resBase + i) * resultSize
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locResBase);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // lhs_value = *(lhs + (lhsRowOff + i*lhsInner) * lhsSize) converted to result type
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locLhsRowOff);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locLhsInner);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // rhs_value = *(rhs + (rhsRowOff + i*rhsInner) * rhsSize) converted to result type
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locRhsRowOff);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locRhsInner);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // op + store
            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblInnerLoop);
            il.MarkLabel(lblInnerEnd);

            // ════════════════════════ SIMD INNER LOOP (L3-c) ═════════════════════
            // 1-vector-at-a-time SIMD load+op+store (NO 4× unroll yet — keeps the
            // emitted kernel small while still giving big wins). Tail handled by
            // jumping back to the scalar inner loop with `i` already advanced.
            if (canSimdInner)
            {
                il.Emit(OpCodes.Br, lblSimdInnerEnd);  // skip if not taken
                il.MarkLabel(lblSimdInner);

                long vectorCount = GetVectorCount(key.ResultType);
                var locVecEnd = il.DeclareLocal(typeof(long));

                // vecEnd = innerSize - vectorCount
                il.Emit(OpCodes.Ldloc, locInnerSize);
                il.Emit(OpCodes.Ldc_I8, vectorCount);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locVecEnd);

                // i is already 0 (initialized before the SIMD-vs-scalar branch)

                var lblSimdLoop = il.DefineLabel();
                var lblSimdLoopEnd = il.DefineLabel();
                var lblSimdTail = il.DefineLabel();
                var lblSimdTailEnd = il.DefineLabel();

                il.MarkLabel(lblSimdLoop);
                // if (i > vecEnd) goto simdLoopEnd
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locVecEnd);
                il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

                // v_lhs = Vector.Load(lhs + (lhsRowOff + i) * lhsSize)
                il.Emit(OpCodes.Ldarg_0); // lhs
                il.Emit(OpCodes.Ldloc, locLhsRowOff);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.LhsType);

                // v_rhs = Vector.Load(rhs + (rhsRowOff + i) * rhsSize)
                il.Emit(OpCodes.Ldarg_1); // rhs
                il.Emit(OpCodes.Ldloc, locRhsRowOff);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.RhsType);

                // v_result = op(v_lhs, v_rhs)
                EmitVectorOperation(il, key.Op, key.ResultType);

                // Store: Vector.Store(v_result, result + (resBase + i) * resultSize)
                il.Emit(OpCodes.Ldarg_2); // result
                il.Emit(OpCodes.Ldloc, locResBase);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)resultSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorStore(il, key.ResultType);

                // i += vectorCount
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, vectorCount);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblSimdLoop);
                il.MarkLabel(lblSimdLoopEnd);

                // ─── Scalar tail (handle remaining elements i..innerSize) ────────
                il.MarkLabel(lblSimdTail);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locInnerSize);
                il.Emit(OpCodes.Bge, lblSimdTailEnd);

                // result_addr
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locResBase);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)resultSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);

                // lhs_val
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locLhsRowOff);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, key.LhsType);
                EmitConvertTo(il, key.LhsType, key.ResultType);

                // rhs_val
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locRhsRowOff);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect(il, key.RhsType);
                EmitConvertTo(il, key.RhsType, key.ResultType);

                // op + store
                EmitScalarOperation(il, key.Op, key.ResultType);
                EmitStoreIndirect(il, key.ResultType);

                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblSimdTail);
                il.MarkLabel(lblSimdTailEnd);

                // ════════════════════════ SIMD INNER (SCALAR LHS) ════════════════
                // (M,1)+(M,N) pattern: lhs has inner stride 0 — every inner element
                // reads the SAME scalar at lhsRowOff. Hoist Vector.Create(*lhsAddr)
                // outside the loop; inside, just SIMD load rhs, op, store.
                il.Emit(OpCodes.Br, lblSimdInnerEnd);
                il.MarkLabel(lblSimdScalarL);
                EmitChunkSimdScalarBlock(il, key, lhsSize, rhsSize, resultSize,
                    locLhsRowOff, locRhsRowOff, locResBase, locInnerSize, locVecEnd, locI,
                    scalarIsLhs: true, vectorCount);

                // ════════════════════════ SIMD INNER (SCALAR RHS) ════════════════
                // (M,N)+(M,1) pattern — symmetric to ScalarLhs.
                il.Emit(OpCodes.Br, lblSimdInnerEnd);
                il.MarkLabel(lblSimdScalarR);
                EmitChunkSimdScalarBlock(il, key, lhsSize, rhsSize, resultSize,
                    locLhsRowOff, locRhsRowOff, locResBase, locInnerSize, locVecEnd, locI,
                    scalarIsLhs: false, vectorCount);

                il.MarkLabel(lblSimdInnerEnd);
            }

            // o++
            il.Emit(OpCodes.Ldloc, locO);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locO);

            il.Emit(OpCodes.Br, lblOuterLoop);
            il.MarkLabel(lblOuterEnd);
        }

        /// <summary>
        ///     L3-d: SIMD inner loop where ONE operand is scalar-broadcast on the
        ///     inner dim (stride == 0). Hoists <c>Vector.Create(*scalarPtr)</c>
        ///     outside the loop so per-iter work is just one SIMD load + op + store
        ///     against the other (contig) operand. Symmetric for scalarIsLhs=true/false.
        /// </summary>
        private static void EmitChunkSimdScalarBlock(
            ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize,
            System.Reflection.Emit.LocalBuilder locLhsRowOff,
            System.Reflection.Emit.LocalBuilder locRhsRowOff,
            System.Reflection.Emit.LocalBuilder locResBase,
            System.Reflection.Emit.LocalBuilder locInnerSize,
            System.Reflection.Emit.LocalBuilder locVecEnd,
            System.Reflection.Emit.LocalBuilder locI,
            bool scalarIsLhs, long vectorCount)
        {
            var clrType = GetClrType(key.ResultType);
            var vecType = VectorMethodCache.V(VectorBits, clrType);
            var locVScalar = il.DeclareLocal(vecType);

            // ── vecEnd = innerSize - vectorCount (shared with CC block; re-set here) ──
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // ── Pre-compute the broadcast vector from the scalar value at row start ──
            // For scalarIsLhs: load lhs[lhsRowOff], broadcast to V<T>
            // For scalarIsLhs=false: load rhs[rhsRowOff], broadcast
            int scalarArg = scalarIsLhs ? 0 : 1;
            int scalarSize = scalarIsLhs ? lhsSize : rhsSize;
            var locScalarRowOff = scalarIsLhs ? locLhsRowOff : locRhsRowOff;
            NPTypeCode scalarType = scalarIsLhs ? key.LhsType : key.RhsType;

            il.Emit(scalarArg == 0 ? OpCodes.Ldarg_0 : OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locScalarRowOff);
            il.Emit(OpCodes.Ldc_I8, (long)scalarSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, scalarType);
            // Convert scalar to result type if necessary (still scalar at this point).
            EmitConvertTo(il, scalarType, key.ResultType);
            EmitVectorCreate(il, key.ResultType);
            il.Emit(OpCodes.Stloc, locVScalar);

            // ── Loop variables ───────────────────────────────────────────────────
            // i is already 0 (initialized in EmitChunkLoop before the dispatch).
            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            il.MarkLabel(lblSimdLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

            // ── SIMD body: order arguments to match (LHS op RHS) regardless of mode ──
            if (scalarIsLhs)
            {
                // v_lhs = broadcast scalar; v_rhs = load from rhs
                il.Emit(OpCodes.Ldloc, locVScalar);
                // v_rhs = Vector.Load(rhs + (rhsRowOff + i) * rhsSize)
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locRhsRowOff);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.RhsType);
            }
            else
            {
                // v_lhs = Vector.Load(lhs + (lhsRowOff + i) * lhsSize); v_rhs = broadcast scalar
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locLhsRowOff);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, key.LhsType);
                il.Emit(OpCodes.Ldloc, locVScalar);
            }

            // v_result = op(v_lhs, v_rhs)
            EmitVectorOperation(il, key.Op, key.ResultType);

            // Store v_result to result + (resBase + i) * resultSize
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locResBase);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.ResultType);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // ── Scalar tail (i..innerSize) ───────────────────────────────────────
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Bge, lblTailEnd);

            // result_addr = result + (resBase + i) * resultSize
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locResBase);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // lhs_val: either scalar (lhs[lhsRowOff] reused — inner stride 0) or
            // strided load at (lhsRowOff + i*lhsInner). For ScalarLhs we know
            // lhsInner==0, so just load lhs[lhsRowOff]. For ScalarRhs lhsInner==1.
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locLhsRowOff);
            if (!scalarIsLhs)
            {
                // lhsInner == 1, so add i
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // rhs_val
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locRhsRowOff);
            if (scalarIsLhs)
            {
                // rhsInner == 1, so add i
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            EmitScalarOperation(il, key.Op, key.ResultType);
            EmitStoreIndirect(il, key.ResultType);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
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
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // Main loop
            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate lhsOffset and rhsOffset from linear index
            // lhsOffset = 0, rhsOffset = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I8, 0L);
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
            il.Emit(OpCodes.Ldc_I8, (long)resultSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load lhs[lhsOffset]
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load rhs[rhsOffset]
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locRhsOffset);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // Operation
            EmitScalarOperation(il, key.Op, key.ResultType);

            // Store
            EmitStoreIndirect(il, key.ResultType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        #endregion
    }
}
