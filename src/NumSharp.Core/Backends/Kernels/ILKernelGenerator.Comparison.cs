using System;
using System.Collections.Concurrent;
using System.Linq;
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
// ILKernelGenerator.Comparison.cs (THIS FILE)
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: Equal (==), NotEqual (!=), Less (<),
//       Greater (>), LessEqual (<=), GreaterEqual (>=)
//     - Type promotion: operands promoted to common type before comparison
//     - SIMD comparison using Vector.Equals/LessThan/etc. with mask extraction
//     - Efficient mask-to-bool conversion using ExtractMostSignificantBits
//     - Path-specific kernels: SimdFull, ScalarRight, ScalarLeft, General
//     - Scalar comparison delegates for single-value operations
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by NDArray comparison operators (==, !=, <, >, <=, >=)
//   KEY MEMBERS:
//     - ComparisonKernel delegate - writes bool* result array
//     - _comparisonCache - caches by ComparisonKernelKey (types, op, path)
//     - _comparisonScalarCache - Func<TLhs, TRhs, bool> for scalar comparisons
//     - GetComparisonKernel(), TryGetComparisonKernel()
//     - GetComparisonScalarDelegate() - for element-by-element comparison
//     - EmitVectorComparison() - SIMD comparison producing mask vector
//     - EmitMaskToBoolExtraction() - efficient mask bits -> bool array
//     - EmitComparisonOperation() - scalar comparison IL emission
//     - EmitComparisonSimdLoop(), EmitComparisonScalarLoop(), EmitComparisonGeneralLoop()
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
        #region Comparison Kernel Generation

        /// <summary>
        /// Cache for comparison kernels.
        /// Key: ComparisonKernelKey (LhsType, RhsType, Op, Path)
        /// </summary>
        private static readonly ConcurrentDictionary<ComparisonKernelKey, ComparisonKernel> _comparisonCache = new();

        /// <summary>
        /// Number of comparison kernels in cache.
        /// </summary>
        public static int ComparisonCachedCount => _comparisonCache.Count;

        /// <summary>
        /// Get or generate a comparison kernel for the specified key.
        /// </summary>
        public static ComparisonKernel GetComparisonKernel(ComparisonKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _comparisonCache.GetOrAdd(key, GenerateComparisonKernel);
        }

        /// <summary>
        /// Try to get or generate a comparison kernel. Returns null if generation fails.
        /// </summary>
        public static ComparisonKernel? TryGetComparisonKernel(ComparisonKernelKey key)
        {
            if (!Enabled)
                return null;

            try
            {
                return _comparisonCache.GetOrAdd(key, GenerateComparisonKernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetComparisonKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Check if SIMD can be used for this comparison operation.
        /// </summary>
        private static bool CanUseComparisonSimd(ComparisonKernelKey key)
        {
            if (VectorBits == 0) return false;
            if (!CanUseSimd(key.ComparisonType)) return false;

            // SIMD comparison only works when both operands have the same type.
            // Mixed-type comparisons require scalar conversion before comparison,
            // which cannot be done efficiently with SIMD vector loads.
            if (key.LhsType != key.RhsType) return false;

            // Only for contiguous (SimdFull) path initially
            return key.Path == ExecutionPath.SimdFull;
        }

        /// <summary>
        /// Generate a comparison kernel for the specified key.
        /// </summary>
        private static ComparisonKernel GenerateComparisonKernel(ComparisonKernelKey key)
        {
            return key.Path switch
            {
                ExecutionPath.SimdFull => GenerateComparisonSimdFullKernel(key),
                ExecutionPath.SimdScalarRight => GenerateComparisonScalarRightKernel(key),
                ExecutionPath.SimdScalarLeft => GenerateComparisonScalarLeftKernel(key),
                ExecutionPath.SimdChunk => GenerateComparisonGeneralKernel(key), // Fall through to general
                ExecutionPath.General => GenerateComparisonGeneralKernel(key),
                _ => throw new NotSupportedException($"Path {key.Path} not supported")
            };
        }

        /// <summary>
        /// Generate a comparison kernel for contiguous arrays.
        /// </summary>
        private static ComparisonKernel GenerateComparisonSimdFullKernel(ComparisonKernelKey key)
        {
            // ComparisonKernel signature:
            // void(void* lhs, void* rhs, bool* result, int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"Comparison_SimdFull_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(bool*),
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            var comparisonType = key.ComparisonType;

            if (CanUseComparisonSimd(key))
            {
                EmitComparisonSimdLoop(il, key, lhsSize, rhsSize, comparisonType);
            }
            else
            {
                EmitComparisonScalarLoop(il, key, lhsSize, rhsSize, comparisonType);
            }

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ComparisonKernel>();
        }

        /// <summary>
        /// Generate a comparison kernel for scalar right operand.
        /// </summary>
        private static ComparisonKernel GenerateComparisonScalarRightKernel(ComparisonKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"Comparison_ScalarRight_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(bool*),
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            var comparisonType = key.ComparisonType;

            EmitComparisonScalarRightLoop(il, key, lhsSize, rhsSize, comparisonType);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ComparisonKernel>();
        }

        /// <summary>
        /// Generate a comparison kernel for scalar left operand.
        /// </summary>
        private static ComparisonKernel GenerateComparisonScalarLeftKernel(ComparisonKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"Comparison_ScalarLeft_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(bool*),
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            var comparisonType = key.ComparisonType;

            EmitComparisonScalarLeftLoop(il, key, lhsSize, rhsSize, comparisonType);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ComparisonKernel>();
        }

        /// <summary>
        /// Generate a general comparison kernel for arbitrary strides.
        /// </summary>
        private static ComparisonKernel GenerateComparisonGeneralKernel(ComparisonKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"Comparison_General_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(bool*),
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            int lhsSize = GetTypeSize(key.LhsType);
            int rhsSize = GetTypeSize(key.RhsType);
            var comparisonType = key.ComparisonType;

            EmitComparisonGeneralLoop(il, key, lhsSize, rhsSize, comparisonType);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<ComparisonKernel>();
        }

        #region Comparison Loop Emission

        /// <summary>
        /// Emit a SIMD loop for contiguous comparison with 4x unrolling (adapts to V128/V256/V512).
        /// </summary>
        private static void EmitComparisonSimdLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            int vectorCount = GetVectorCount(comparisonType);
            int unrollFactor = 4;
            int unrollStep = vectorCount * unrollFactor;
            var clrType = GetClrType(comparisonType);
            var vectorType = GetVectorType(clrType);

            // Args: void* lhs (0), void* rhs (1), bool* result (2),
            //       int* lhsStrides (3), int* rhsStrides (4), int* shape (5),
            //       int ndim (6), int totalSize (7)

            var locI = il.DeclareLocal(typeof(int));
            var locUnrollEnd = il.DeclareLocal(typeof(int));
            var locVectorEnd = il.DeclareLocal(typeof(int));

            // Declare mask locals for 4x unrolling
            var locMask0 = il.DeclareLocal(vectorType);
            var locMask1 = il.DeclareLocal(vectorType);
            var locMask2 = il.DeclareLocal(vectorType);
            var locMask3 = il.DeclareLocal(vectorType);
            var maskLocals = new[] { locMask0, locMask1, locMask2, locMask3 };

            // unrollEnd = totalSize - unrollStep + 1 (last valid 4x start position)
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I4, unrollStep - 1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = totalSize - vectorCount + 1 (last valid SIMD start position)
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I4, vectorCount - 1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // === 4x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnrollLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            // Load 4 lhs vectors, 4 rhs vectors, compare, store masks
            for (int n = 0; n < unrollFactor; n++)
            {
                int offset = n * vectorCount;

                // Load lhs vector at (i + offset) * lhsSize
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
                EmitVectorLoad(il, comparisonType);

                // Load rhs vector at (i + offset) * rhsSize
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
                EmitVectorLoad(il, comparisonType);

                // Compare: produces mask vector
                EmitVectorComparison(il, key.Op, comparisonType);
                il.Emit(OpCodes.Stloc, maskLocals[n]);
            }

            // Extract all 4 masks to booleans
            for (int n = 0; n < unrollFactor; n++)
            {
                int offset = n * vectorCount;

                // Create a temporary local to hold (i + offset) for extraction
                var locIOffset = il.DeclareLocal(typeof(int));
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Stloc, locIOffset);

                EmitMaskToBoolExtraction(il, comparisonType, vectorCount, locIOffset, maskLocals[n]);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnrollLoop);

            // === REMAINDER SIMD LOOP (0-3 vectors) ===
            il.MarkLabel(lblUnrollEnd);
            il.MarkLabel(lblRemainderLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemainderEnd);

            // Load lhs vector: lhs + i * elemSize
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, comparisonType);

            // Load rhs vector: rhs + i * elemSize
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, comparisonType);

            // Compare: produces mask vector
            EmitVectorComparison(il, key.Op, comparisonType);
            il.Emit(OpCodes.Stloc, locMask0);

            // Extract mask to booleans
            EmitMaskToBoolExtraction(il, comparisonType, vectorCount, locI, locMask0);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRemainderLoop);

            // === SCALAR TAIL LOOP ===
            il.MarkLabel(lblRemainderEnd);
            il.MarkLabel(lblTailLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblTailEnd);

            // result[i] = (lhs[i] op rhs[i])
            il.Emit(OpCodes.Ldarg_2); // result (bool*)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);

            // Load lhs[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            if (key.LhsType != comparisonType)
                EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs[i]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            if (key.RhsType != comparisonType)
                EmitConvertTo(il, key.RhsType, comparisonType);

            // Compare
            EmitComparisonOperation(il, key.Op, comparisonType);

            // Store result
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTailLoop);

            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        /// Emit a SIMD comparison operation (adapts to V128/V256/V512).
        /// Stack has [lhs_vector, rhs_vector], result is comparison mask vector.
        /// </summary>
        private static void EmitVectorComparison(ILGenerator il, ComparisonOp op, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            string methodName = op switch
            {
                ComparisonOp.Equal => "Equals",
                ComparisonOp.NotEqual => "Equals",  // Invert later
                ComparisonOp.Less => "LessThan",
                ComparisonOp.LessEqual => "LessThanOrEqual",
                ComparisonOp.Greater => "GreaterThan",
                ComparisonOp.GreaterEqual => "GreaterThanOrEqual",
                _ => throw new NotSupportedException($"Comparison op {op} not supported for SIMD")
            };

            var cmpMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == methodName && m.IsGenericMethod && m.GetParameters().Length == 2)
                .Select(m => m.MakeGenericMethod(clrType))
                .First(m => m.GetParameters()[0].ParameterType == vectorType);

            il.EmitCall(OpCodes.Call, cmpMethod, null);

            // For NotEqual, invert the result using OnesComplement
            if (op == ComparisonOp.NotEqual)
            {
                var onesCompMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == "OnesComplement" && m.IsGenericMethod && m.GetParameters().Length == 1)
                    .Select(m => m.MakeGenericMethod(clrType))
                    .First(m => m.GetParameters()[0].ParameterType == vectorType);

                il.EmitCall(OpCodes.Call, onesCompMethod, null);
            }
        }

        /// <summary>
        /// Emit extraction of comparison mask vector to individual booleans.
        /// Uses ExtractMostSignificantBits for O(1) extraction instead of O(N) GetElement calls.
        /// </summary>
        private static void EmitMaskToBoolExtraction(ILGenerator il, NPTypeCode type,
            int vectorCount, LocalBuilder locI, LocalBuilder locMask)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            // ExtractMostSignificantBits gives us a uint where each bit is the MSB of each lane
            // For comparison masks (all 1s = true, all 0s = false), MSB=1 means true
            var extractMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "ExtractMostSignificantBits" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(clrType))
                .First();

            // bits = ExtractMostSignificantBits(mask)
            il.Emit(OpCodes.Ldloc, locMask);
            il.EmitCall(OpCodes.Call, extractMethod, null);
            var locBits = il.DeclareLocal(typeof(uint));
            il.Emit(OpCodes.Stloc, locBits);

            // For each lane j, store (bits >> j) & 1 as bool
            for (int j = 0; j < vectorCount; j++)
            {
                // result + i + j
                il.Emit(OpCodes.Ldarg_2); // result (bool*)
                il.Emit(OpCodes.Ldloc, locI);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, j);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Add);

                // (bits >> j) & 1
                il.Emit(OpCodes.Ldloc, locBits);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, j);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.And);

                // Store as bool (1 byte)
                il.Emit(OpCodes.Stind_I1);
            }
        }

        /// <summary>
        /// Emit a scalar loop for contiguous comparison.
        /// </summary>
        private static void EmitComparisonScalarLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            // Args: void* lhs (0), void* rhs (1), bool* result (2),
            //       int* lhsStrides (3), int* rhsStrides (4), int* shape (5),
            //       int ndim (6), int totalSize (7)

            var locI = il.DeclareLocal(typeof(int)); // loop counter

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

            // result[i] = (lhs[i] op rhs[i])
            // Load result address
            il.Emit(OpCodes.Ldarg_2); // result (bool*)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add); // bool is 1 byte, so just add i

            // Load lhs[i] and convert to comparison type
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs[i] and convert to comparison type
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, comparisonType);

            // Perform comparison
            EmitComparisonOperation(il, key.Op, comparisonType);

            // Store bool result
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit loop for scalar right operand comparison.
        /// </summary>
        private static void EmitComparisonScalarRightLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locRhsVal = il.DeclareLocal(GetClrType(comparisonType)); // scalar value

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Load rhs[0] and convert to comparison type, store in local
            il.Emit(OpCodes.Ldarg_1); // rhs
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, comparisonType);
            il.Emit(OpCodes.Stloc, locRhsVal);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = (lhs[i] op rhsVal)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);

            // Load lhs[i] and convert
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load cached rhs scalar
            il.Emit(OpCodes.Ldloc, locRhsVal);

            EmitComparisonOperation(il, key.Op, comparisonType);
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit loop for scalar left operand comparison.
        /// </summary>
        private static void EmitComparisonScalarLeftLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locLhsVal = il.DeclareLocal(GetClrType(comparisonType)); // scalar value

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Load lhs[0] and convert to comparison type, store in local
            il.Emit(OpCodes.Ldarg_0); // lhs
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);
            il.Emit(OpCodes.Stloc, locLhsVal);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = (lhsVal op rhs[i])
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
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
            EmitConvertTo(il, key.RhsType, comparisonType);

            EmitComparisonOperation(il, key.Op, comparisonType);
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit general coordinate-based iteration loop for comparison.
        /// </summary>
        private static void EmitComparisonGeneralLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            // Args: void* lhs (0), void* rhs (1), bool* result (2),
            //       int* lhsStrides (3), int* rhsStrides (4), int* shape (5),
            //       int ndim (6), int totalSize (7)

            var locI = il.DeclareLocal(typeof(int)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locLhsOffset = il.DeclareLocal(typeof(int)); // lhs offset
            var locRhsOffset = il.DeclareLocal(typeof(int)); // rhs offset
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
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Calculate lhsOffset and rhsOffset from linear index
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locRhsOffset);

            // idx = i (for coordinate calculation)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

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
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locCoord);

            // idx /= shape[d]
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg_S, (byte)5); // shape
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locIdx);

            // lhsOffset += coord * lhsStrides[d]
            il.Emit(OpCodes.Ldloc, locLhsOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_3); // lhsStrides
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locLhsOffset);

            // rhsOffset += coord * rhsStrides[d]
            il.Emit(OpCodes.Ldloc, locRhsOffset);
            il.Emit(OpCodes.Ldloc, locCoord);
            il.Emit(OpCodes.Ldarg_S, (byte)4); // rhsStrides
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
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

            // result[i] = (lhs[lhsOffset] op rhs[rhsOffset])
            // Load result address (contiguous output)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);

            // Load lhs[lhsOffset]
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locLhsOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs[rhsOffset]
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locRhsOffset);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, comparisonType);

            // Comparison
            EmitComparisonOperation(il, key.Op, comparisonType);

            // Store bool
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        #endregion

        #region Comparison Operation Emission

        /// <summary>
        /// Emit comparison operation. Stack has two values of comparisonType, result is bool (0 or 1).
        /// </summary>
        internal static void EmitComparisonOperation(ILGenerator il, ComparisonOp op, NPTypeCode comparisonType)
        {
            // Special handling for decimal comparisons
            if (comparisonType == NPTypeCode.Decimal)
            {
                EmitDecimalComparison(il, op);
                return;
            }

            bool isUnsigned = IsUnsigned(comparisonType);
            bool isFloat = comparisonType == NPTypeCode.Single || comparisonType == NPTypeCode.Double;

            switch (op)
            {
                case ComparisonOp.Equal:
                    il.Emit(OpCodes.Ceq);
                    break;

                case ComparisonOp.NotEqual:
                    il.Emit(OpCodes.Ceq);
                    // Negate: result = !result (xor with 1)
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                    break;

                case ComparisonOp.Less:
                    if (isUnsigned)
                        il.Emit(OpCodes.Clt_Un);
                    else
                        il.Emit(OpCodes.Clt);
                    break;

                case ComparisonOp.LessEqual:
                    // a <= b is !(a > b)
                    if (isUnsigned)
                        il.Emit(OpCodes.Cgt_Un);
                    else
                        il.Emit(OpCodes.Cgt);
                    // Negate
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                    break;

                case ComparisonOp.Greater:
                    if (isUnsigned)
                        il.Emit(OpCodes.Cgt_Un);
                    else
                        il.Emit(OpCodes.Cgt);
                    break;

                case ComparisonOp.GreaterEqual:
                    // a >= b is !(a < b)
                    if (isUnsigned)
                        il.Emit(OpCodes.Clt_Un);
                    else
                        il.Emit(OpCodes.Clt);
                    // Negate
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                    break;

                default:
                    throw new NotSupportedException($"Comparison operation {op} not supported");
            }
        }

        /// <summary>
        /// Emit decimal comparison using operator methods.
        /// </summary>
        private static void EmitDecimalComparison(ILGenerator il, ComparisonOp op)
        {
            // decimal has comparison operators that return bool
            string methodName = op switch
            {
                ComparisonOp.Equal => "op_Equality",
                ComparisonOp.NotEqual => "op_Inequality",
                ComparisonOp.Less => "op_LessThan",
                ComparisonOp.LessEqual => "op_LessThanOrEqual",
                ComparisonOp.Greater => "op_GreaterThan",
                ComparisonOp.GreaterEqual => "op_GreaterThanOrEqual",
                _ => throw new NotSupportedException($"Comparison {op} not supported for decimal")
            };

            var method = typeof(decimal).GetMethod(
                methodName,
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { typeof(decimal), typeof(decimal) },
                null
            );

            il.EmitCall(OpCodes.Call, method!, null);
        }

        #endregion

        #region Comparison Scalar Kernel Generation

        // ComparisonScalarKernelKey is now defined in ScalarKernel.cs

        /// <summary>
        /// Cache for comparison scalar kernels.
        /// </summary>
        private static readonly ConcurrentDictionary<ComparisonScalarKernelKey, Delegate> _comparisonScalarCache = new();

        /// <summary>
        /// Number of comparison scalar kernels in cache.
        /// </summary>
        public static int ComparisonScalarCachedCount => _comparisonScalarCache.Count;

        /// <summary>
        /// Get or generate a comparison scalar delegate.
        /// Returns a Func&lt;TLhs, TRhs, bool&gt; delegate.
        /// </summary>
        public static Delegate GetComparisonScalarDelegate(ComparisonScalarKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _comparisonScalarCache.GetOrAdd(key, GenerateComparisonScalarDelegate);
        }

        /// <summary>
        /// Generate an IL-based comparison scalar delegate.
        /// </summary>
        private static Delegate GenerateComparisonScalarDelegate(ComparisonScalarKernelKey key)
        {
            var lhsClr = GetClrType(key.LhsType);
            var rhsClr = GetClrType(key.RhsType);
            var comparisonType = key.ComparisonType;

            // Create DynamicMethod: bool Method(TLhs lhs, TRhs rhs)
            var dm = new DynamicMethod(
                name: $"ScalarComparison_{key}",
                returnType: typeof(bool),
                parameterTypes: new[] { lhsClr, rhsClr },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Load lhs, convert to comparison type
            il.Emit(OpCodes.Ldarg_0);
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs, convert to comparison type
            il.Emit(OpCodes.Ldarg_1);
            EmitConvertTo(il, key.RhsType, comparisonType);

            // Perform comparison
            EmitComparisonOperation(il, key.Op, comparisonType);

            // Return
            il.Emit(OpCodes.Ret);

            // Create typed Func<TLhs, TRhs, bool>
            var funcType = typeof(Func<,,>).MakeGenericType(lhsClr, rhsClr, typeof(bool));
            return dm.CreateDelegate(funcType);
        }

        #endregion

        #endregion
    }
}
