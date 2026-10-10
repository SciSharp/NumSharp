using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

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
// DirectILKernelGenerator.MixedType.cs
//   OWNERSHIP: Mixed-type binary operations with type promotion
//   RESPONSIBILITY:
//     - Handles all binary ops where operand types may differ
//     - Generates path-specific kernels based on stride patterns
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for general binary operations
//
// DirectILKernelGenerator.Unary.cs
//   OWNERSHIP: Unary element-wise operations
//   RESPONSIBILITY:
//     - Math functions: Negate, Abs, Sqrt, Sin, Cos, Exp, Log, Sign, Floor, Ceil, etc.
//     - Scalar delegate generation for single-value operations (Func<TIn,TOut>)
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for unary ops; scalar delegates used in broadcasting
//
// DirectILKernelGenerator.Comparison.cs (THIS FILE)
//   OWNERSHIP: Comparison operations returning boolean arrays
//   RESPONSIBILITY:
//     - Element-wise comparisons: Equal (==), NotEqual (!=), Less (<),
//       Greater (>), LessEqual (<=), GreaterEqual (>=)
//     - Type promotion: operands promoted to common type before comparison
//     - SIMD comparison using Vector.Equals/LessThan/etc. with mask extraction
//     - Efficient mask-to-bool conversion using ExtractMostSignificantBits
//     - Path-specific kernels: SimdFull, ScalarRight, ScalarLeft, General
//     - Scalar comparison delegates for single-value operations
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
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
    public static partial class DirectILKernelGenerator
    {
        #region Comparison Kernel Generation

        /// <summary>
        /// Cache for comparison kernels.
        /// Key: ComparisonKernelKey (LhsType, RhsType, Op, Path)
        /// </summary>
        internal static readonly ConcurrentDictionary<ComparisonKernelKey, ComparisonKernel> _comparisonCache = new();

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
        /// Check if SIMD can be used for this comparison operation.
        /// </summary>
        private static bool CanUseComparisonSimd(ComparisonKernelKey key)
        {
            if (VectorBits == 0) return false;
            // Boolean rides the byte lanes: EmitVectorComparison's Boolean branch computes the
            // normalized truth-value comparison directly as 0/1 result bytes (NumPy's b8
            // comparison loops), stored whole by EmitMaskToBoolExtraction's Boolean branch.
            if (!CanUseSimd(key.ComparisonType) && key.ComparisonType != NPTypeCode.Boolean)
                return false;

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
            // f16 comparisons are pure bit logic (sign-magnitude key order + NaN/±0 masks), so
            // (Half,Half) keys ride static bit-level AVX2 kernels with zero float conversions —
            // see Comparison.Half.cs. SimdChunk/General keep the emitted scalar IL below.
            if (key.LhsType == NPTypeCode.Half && key.RhsType == NPTypeCode.Half)
            {
                var halfCmp = TryGetHalfComparisonKernel(key.Op, key.Path);
                if (halfCmp != null)
                    return halfCmp;
            }

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
            // void(void* lhs, void* rhs, bool* result, long* lhsStrides, long* rhsStrides, long* shape, int ndim, long totalSize)
            var dm = new DynamicMethod(
                name: $"Comparison_SimdFull_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(bool*),
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(DirectILKernelGenerator),
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
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(DirectILKernelGenerator),
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
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(DirectILKernelGenerator),
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
                    typeof(long*), typeof(long*), typeof(long*),
                    typeof(int), typeof(long)
                },
                owner: typeof(DirectILKernelGenerator),
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
            long vectorCount = GetVectorCount(comparisonType);
            int unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;
            // Mask LOCALS take the lane type (Boolean -> byte; Vector256<bool> is not a type).
            var vectorType = GetVectorType(GetSimdLaneType(comparisonType));

            // Args: void* lhs (0), void* rhs (1), bool* result (2),
            //       long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //       int ndim (6), long totalSize (7)

            var locI = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            // Declare mask locals for 4x unrolling
            var locMask0 = il.DeclareLocal(vectorType);
            var locMask1 = il.DeclareLocal(vectorType);
            var locMask2 = il.DeclareLocal(vectorType);
            var locMask3 = il.DeclareLocal(vectorType);
            var maskLocals = new[] { locMask0, locMask1, locMask2, locMask3 };

            // unrollEnd = totalSize - unrollStep: the LAST start of four whole vectors (the loop runs while i <= it, so
            // i + unrollStep <= totalSize). It was totalSize - unrollStep + 1, which ran one body too many whenever
            // totalSize % unrollStep == unrollStep - 1 (n = 31 of float32): one element read past both inputs and one
            // bool written past the result — over the next slot of an out= view's parent buffer.
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = totalSize - vectorCount: the last start of one whole vector (same off-by-one fixed).
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
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
                long offset = n * vectorCount;

                // Load lhs vector at (i + offset) * lhsSize
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
                EmitVectorLoad(il, comparisonType);

                // Load rhs vector at (i + offset) * rhsSize
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
                EmitVectorLoad(il, comparisonType);

                // Compare: produces mask vector
                EmitVectorComparison(il, key.Op, comparisonType);
                il.Emit(OpCodes.Stloc, maskLocals[n]);
            }

            // Store all 4 masks as bools: packed together on AVX2 hosts (NumPy's npyv_pack_b8_*), else one by one.
            EmitMasksToBools(il, comparisonType, (int)vectorCount, locI, maskLocals);

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
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
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, comparisonType);

            // Load rhs vector: rhs + i * elemSize
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad(il, comparisonType);

            // Compare: produces mask vector
            EmitVectorComparison(il, key.Op, comparisonType);
            il.Emit(OpCodes.Stloc, locMask0);

            // Extract mask to booleans
            EmitMaskToBoolExtraction(il, comparisonType, (int)vectorCount, locI, locMask0);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
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
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            if (key.LhsType != comparisonType)
                EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs[i]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
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
            il.Emit(OpCodes.Ldc_I8, 1L);
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
            // Boolean compares by TRUTH VALUE, not raw byte (NumPy loops_comparison b8 loops:
            // both operands are cmpeq-normalized first, so frombuffer 0x01 == 0x80). Produces
            // 0/1 RESULT bytes directly — no 0xFF mask, no extraction step needed.
            if (type == NPTypeCode.Boolean)
            {
                EmitBoolVectorComparison(il, op);
                return;
            }

            var clrType = GetClrType(type);

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

            // Equals has the bool-returning EqualsAll overload too — VectorMethodCache.Equals
            // discriminates by return type. Other compares are unambiguous at 2 params.
            var cmpMethod = methodName == "Equals"
                ? VectorMethodCache.Equals(VectorBits, clrType)
                : VectorMethodCache.Generic(VectorBits, methodName, clrType, paramCount: 2);

            il.EmitCall(OpCodes.Call, cmpMethod, null);

            if (op == ComparisonOp.NotEqual)
                il.EmitCall(OpCodes.Call, VectorMethodCache.OnesComplement(VectorBits, clrType), null);
        }

        /// <summary>
        /// Boolean comparison over byte lanes with NumPy's normalized semantics
        /// (loops_comparison.dispatch.c.src b8 loops: VOP = xnor/xor/andc/orc over the ==0 masks).
        /// Stack: [a(V&lt;byte&gt;), b(V&lt;byte&gt;)] -> [result(V&lt;byte&gt;)], every lane 0 or 1.
        /// With na = min(a,1), nb = min(b,1) (byte_to_true — one pminub):
        ///   eq: (na ^ nb) ^ 1     ne: na ^ nb
        ///   lt: (na ^ 1) &amp; nb     le: (na &amp; (nb ^ 1)) ^ 1
        ///   gt: na &amp; (nb ^ 1)     ge: ((na ^ 1) &amp; nb) ^ 1
        /// </summary>
        private static void EmitBoolVectorComparison(ILGenerator il, ComparisonOp op)
        {
            var laneType = typeof(byte);
            var vecType = VectorMethodCache.V(VectorBits, laneType);
            var xor = VectorMethodCache.BinaryX86(VectorBits, "Xor", laneType)
                      ?? VectorMethodCache.Generic(VectorBits, "Xor", laneType, paramCount: 2);
            var and = VectorMethodCache.BinaryX86(VectorBits, "BitwiseAnd", laneType)
                      ?? VectorMethodCache.Generic(VectorBits, "BitwiseAnd", laneType, paramCount: 2);

            void PushOnes()
            {
                il.Emit(OpCodes.Ldc_I4_1);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(VectorBits, laneType), null);
            }

            // Normalize both operands: [a, b] -> locNa, locNb (0/1 bytes).
            var locNb = il.DeclareLocal(vecType);
            var locNa = il.DeclareLocal(vecType);
            PushOnes();                                          // [a, b, 1]
            EmitRawVectorMinOrMax(il, "Min", laneType);          // [a, nb]
            il.Emit(OpCodes.Stloc, locNb);                       // [a]
            PushOnes();                                          // [a, 1]
            EmitRawVectorMinOrMax(il, "Min", laneType);          // [na]
            il.Emit(OpCodes.Stloc, locNa);                       // []

            void LoadNa() => il.Emit(OpCodes.Ldloc, locNa);
            void LoadNb() => il.Emit(OpCodes.Ldloc, locNb);
            void NotBit() { PushOnes(); il.EmitCall(OpCodes.Call, xor, null); }   // v ^ 1 (0/1 lanes)

            switch (op)
            {
                case ComparisonOp.Equal:
                    LoadNa(); LoadNb(); il.EmitCall(OpCodes.Call, xor, null); NotBit();
                    return;
                case ComparisonOp.NotEqual:
                    LoadNa(); LoadNb(); il.EmitCall(OpCodes.Call, xor, null);
                    return;
                case ComparisonOp.Less:
                    LoadNa(); NotBit(); LoadNb(); il.EmitCall(OpCodes.Call, and, null);
                    return;
                case ComparisonOp.LessEqual:
                    LoadNa(); LoadNb(); NotBit(); il.EmitCall(OpCodes.Call, and, null); NotBit();
                    return;
                case ComparisonOp.Greater:
                    LoadNa(); LoadNb(); NotBit(); il.EmitCall(OpCodes.Call, and, null);
                    return;
                case ComparisonOp.GreaterEqual:
                    LoadNa(); NotBit(); LoadNb(); il.EmitCall(OpCodes.Call, and, null); NotBit();
                    return;
                default:
                    throw new NotSupportedException($"Comparison op {op} not supported for Boolean SIMD");
            }
        }

        // BMI2 PDEP method handle — resolved once at static init so the
        // EmitMaskToBoolExtraction codegen can decide between the fast
        // single-PDEP-then-pointer-store path and the scalar shift-mask-
        // store-per-lane fallback without paying GetMethod cost per emit.
        // The X64 variant is required because we want a 64-bit deposit
        // (8 bytes packed at once into a single ulong).
        private static readonly MethodInfo? s_bmi2X64Pdep =
            System.Runtime.Intrinsics.X86.Bmi2.X64.IsSupported
                ? typeof(System.Runtime.Intrinsics.X86.Bmi2.X64).GetMethod(
                    nameof(System.Runtime.Intrinsics.X86.Bmi2.X64.ParallelBitDeposit),
                    BindingFlags.Public | BindingFlags.Static,
                    new[] { typeof(ulong), typeof(ulong) })
                : null;

        /// <summary>
        ///     Emits the store of ONE comparison mask vector (every lane all-ones or all-zeros) as <paramref name="vectorCount"/>
        ///     result bytes, 1 or 0, at <c>result + i</c>.
        /// </summary>
        /// <param name="il">The kernel's IL generator; argument 2 is the <c>bool*</c> result.</param>
        /// <param name="type">The comparison dtype the mask's lanes came from (its width decides the method).</param>
        /// <param name="vectorCount">The lanes in one vector of <paramref name="type"/>: 2 to 64.</param>
        /// <param name="locI">The local holding the first result index this mask covers.</param>
        /// <param name="locMask">The local holding the mask vector.</param>
        /// <remarks>
        ///     <list type="bullet">
        ///     <item><b>Bool lanes</b>: <see cref="EmitBoolVectorComparison"/> already produced the 0/1 bytes; one store.</item>
        ///     <item><b>1-byte lanes</b>: a mask lane is <c>0xFF</c> or <c>0x00</c>, so <c>mask &amp; 1</c> IS the result
        ///     bytes: one AND and one whole-vector store, at every vector width. The per-lane path this replaced cost about
        ///     four instructions an element, so int8 / uint8 comparisons ran ten times slower than their compare.</item>
        ///     <item><b>Wider lanes with BMI2</b>: the lane sign bits (<c>ExtractMostSignificantBits</c>) become bytes
        ///     through one PDEP per 8 lanes, each stored as one 8-byte write (2 or 4 lanes store exactly 2 or 4 bytes, so
        ///     nothing past the vector's lanes is written). The bits are read as a <c>ulong</c>, because a Vector512's
        ///     extract returns 64 bits: a 32-bit local would drop lanes 32-63.</item>
        ///     <item><b>No BMI2</b> (ARM, old x86): one shift-and-store per lane.</item>
        ///     </list>
        ///     The 4×-unrolled loops store four masks together through <see cref="EmitMasksToBools"/>, which packs them on
        ///     AVX2 hosts; this single-mask form serves their one-vector remainder loops and the other widths.
        /// </remarks>
        private static void EmitMaskToBoolExtraction(ILGenerator il, NPTypeCode type,
            int vectorCount, LocalBuilder locI, LocalBuilder locMask)
        {
            // Boolean: EmitBoolVectorComparison already produced 0/1 RESULT bytes — the "mask"
            // IS the answer. One whole-vector store to result + i replaces the extraction.
            if (type == NPTypeCode.Boolean)
            {
                il.Emit(OpCodes.Ldloc, locMask);   // [vec]
                il.Emit(OpCodes.Ldarg_2);          // [vec, result(bool*)]
                il.Emit(OpCodes.Ldloc, locI);      // [vec, result, i]
                il.Emit(OpCodes.Add);              // [vec, ptr]
                EmitVectorStore(il, NPTypeCode.Boolean);
                return;
            }

            var clrType = GetClrType(type);

            // One-byte lanes: 0xFF & 1 = 1, 0x00 & 1 = 0, so the AND of the mask IS the result vector.
            if (GetTypeSize(type) == 1)
            {
                var and = VectorMethodCache.BinaryX86(VectorBits, "BitwiseAnd", clrType)
                          ?? VectorMethodCache.Generic(VectorBits, "BitwiseAnd", clrType, paramCount: 2);
                il.Emit(OpCodes.Ldloc, locMask);
                il.Emit(OpCodes.Ldc_I4_1);
                EmitVectorCreate(il, type);
                il.EmitCall(OpCodes.Call, and, null);   // [bytes]
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Add);                   // [bytes, ptr]
                EmitVectorStore(il, type);
                return;
            }

            // The lane sign bits, widened to 64 bits whatever the extract returns (uint for V128/V256, ulong for V512).
            il.Emit(OpCodes.Ldloc, locMask);
            il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(VectorBits, clrType), null);
            il.Emit(OpCodes.Conv_U8);
            var locBits = il.DeclareLocal(typeof(ulong));
            il.Emit(OpCodes.Stloc, locBits);

            if (s_bmi2X64Pdep != null)
            {
                // PDEP takes the LOW 8 source bits for the 8 set bits of 0x0101…01, so chunk c only needs the shift.
                for (int chunk = 0; chunk * 8 < vectorCount; chunk++)
                {
                    int lanes = Math.Min(8, vectorCount - chunk * 8);
                    il.Emit(OpCodes.Ldarg_2);
                    il.Emit(OpCodes.Ldloc, locI);
                    if (chunk > 0)
                    {
                        il.Emit(OpCodes.Ldc_I8, (long)(chunk * 8));
                        il.Emit(OpCodes.Add);
                    }
                    il.Emit(OpCodes.Add);                                   // [ptr]
                    il.Emit(OpCodes.Ldloc, locBits);
                    if (chunk > 0)
                    {
                        il.Emit(OpCodes.Ldc_I4, chunk * 8);
                        il.Emit(OpCodes.Shr_Un);
                    }
                    il.Emit(OpCodes.Ldc_I8, unchecked((long)0x0101010101010101UL));
                    il.EmitCall(OpCodes.Call, s_bmi2X64Pdep, null);        // [ptr, packed]
                    // Store exactly the lanes' bytes: a 2- or 4-lane vector must not write past its lanes.
                    switch (lanes)
                    {
                        case 8:
                            il.Emit(OpCodes.Stind_I8);
                            break;
                        case 4:
                            il.Emit(OpCodes.Conv_U4);
                            il.Emit(OpCodes.Stind_I4);
                            break;
                        case 2:
                            il.Emit(OpCodes.Conv_U2);
                            il.Emit(OpCodes.Stind_I2);
                            break;
                        default:
                            il.Emit(OpCodes.Conv_U1);
                            il.Emit(OpCodes.Stind_I1);
                            break;
                    }
                }
                return;
            }

            // No BMI2: (bits >> j) & 1 per lane.
            for (int j = 0; j < vectorCount; j++)
            {
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locI);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, (long)j);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, locBits);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, j);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Conv_U4);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stind_I1);
            }
        }

        /// <summary>
        ///     Emits the stores of the 4×-unrolled body's four comparison masks as consecutive result bytes, starting at
        ///     <c>result + i</c> (<c>4 * vectorCount</c> bytes).
        /// </summary>
        /// <param name="il">The kernel's IL generator; argument 2 is the <c>bool*</c> result.</param>
        /// <param name="type">The comparison dtype the masks' lanes came from.</param>
        /// <param name="vectorCount">The lanes in one vector of <paramref name="type"/>.</param>
        /// <param name="locI">The local holding the first result index of the four masks.</param>
        /// <param name="masks">The four mask locals, in element order.</param>
        /// <remarks>
        ///     On an AVX2 host with 256-bit vectors the masks of 2-, 4- and 8-byte lanes are PACKED into result bytes the way
        ///     NumPy's comparison loops do it (<c>npyv_pack_b8_b16 / _b32 / _b64</c> in <c>loops_comparison.dispatch.c.src</c>):
        ///     saturating packs narrow the all-ones / all-zeros lanes to bytes, a permute undoes AVX2's per-128-bit-half
        ///     packing order, an AND with 1 makes the bytes 0/1, and one vector store writes them. Measured over 100K
        ///     elements against the per-vector PDEP store: float32 7.05 → 5.15 µs, float64 14.7 → 9.95 µs, int16 6.8 →
        ///     3.8 µs. Every other case stores each mask through <see cref="EmitMaskToBoolExtraction"/>.
        /// </remarks>
        private static void EmitMasksToBools(ILGenerator il, NPTypeCode type, int vectorCount, LocalBuilder locI,
            LocalBuilder[] masks)
        {
            int laneBytes = type == NPTypeCode.Boolean ? 1 : GetTypeSize(type);
            if (ComparisonPackIntrinsics.Available && masks.Length == 4 && laneBytes >= 2)
            {
                EmitPackedMasksToBools(il, type, laneBytes, vectorCount, locI, masks);
                return;
            }

            for (int n = 0; n < masks.Length; n++)
            {
                var locIOffset = il.DeclareLocal(typeof(long));
                il.Emit(OpCodes.Ldloc, locI);
                if (n > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, (long)n * vectorCount);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Stloc, locIOffset);
                EmitMaskToBoolExtraction(il, type, vectorCount, locIOffset, masks[n]);
            }
        }

        /// <summary>
        ///     The AVX2 packed store of four 256-bit masks (see <see cref="EmitMasksToBools"/>).
        /// </summary>
        /// <param name="il">The kernel's IL generator; argument 2 is the <c>bool*</c> result.</param>
        /// <param name="type">The comparison dtype (2-, 4- or 8-byte lanes).</param>
        /// <param name="laneBytes">The lane width in bytes: 2, 4 or 8.</param>
        /// <param name="vectorCount">The lanes per vector: 16, 8 or 4.</param>
        /// <param name="locI">The local holding the first result index.</param>
        /// <param name="masks">The four mask locals, in element order.</param>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="laneBytes"/> is not 2, 4 or 8 — a caller bug.</exception>
        /// <remarks>
        ///     AVX2's packs work inside each 128-bit half, so each width ends with its own reorder:
        ///     <list type="bullet">
        ///     <item><b>2-byte lanes</b> (two pairs, 64 bools): <c>packsswb(m0, m1)</c> holds <c>[m0 lo, m1 lo | m0 hi,
        ///     m1 hi]</c> as quadwords; <c>vpermq 0xD8</c> reorders them to <c>[m0 lo, m0 hi, m1 lo, m1 hi]</c>.</item>
        ///     <item><b>4-byte lanes</b> (32 bools): two <c>packssdw</c> and one <c>packsswb</c> leave the dwords as
        ///     <c>[m0 lo, m1 lo, m2 lo, m3 lo | m0 hi, m1 hi, m2 hi, m3 hi]</c>; <c>vpermd (0,4,1,5,2,6,3,7)</c> restores
        ///     element order (NumPy's <c>npyv_pack_b8_b32</c>).</item>
        ///     <item><b>8-byte lanes</b> (16 bools): each 64-bit lane is two equal dwords, so the same three packs make
        ///     every lane appear twice; <c>vpshufb</c> keeps the even bytes of each half (<c>[m0q0, m0q1, m1q0, m1q1, …]</c>
        ///     low, <c>[m0q2, m0q3, m1q2, m1q3, …]</c> high) and <c>punpcklwd</c> of the halves interleaves their 2-byte
        ///     pairs back into element order; one 16-byte store.</item>
        ///     </list>
        ///     A saturating pack keeps a lane's -1 / 0 (the mask is all-ones or all-zeros whatever its dtype: a float mask
        ///     reinterpreted as int32 is -1).
        /// </remarks>
        private static void EmitPackedMasksToBools(ILGenerator il, NPTypeCode type, int laneBytes, int vectorCount,
            LocalBuilder locI, LocalBuilder[] masks)
        {
            var lane = GetClrType(type);
            var P = ComparisonPackIntrinsics.Instance;

            void LoadAs(LocalBuilder mask, Type to)
            {
                il.Emit(OpCodes.Ldloc, mask);
                if (lane != to)
                    il.EmitCall(OpCodes.Call, P.As256(lane, to), null);
            }

            // result + i + offset as a byte pointer.
            void PushResultPtr(long offset)
            {
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Add);
            }

            switch (laneBytes)
            {
                case 2:
                {
                    for (int pair = 0; pair < 2; pair++)
                    {
                        var locBytes = il.DeclareLocal(typeof(Vector256<sbyte>));
                        LoadAs(masks[2 * pair], typeof(short));
                        LoadAs(masks[2 * pair + 1], typeof(short));
                        il.EmitCall(OpCodes.Call, P.PackInt16, null);                          // V256<sbyte>
                        il.EmitCall(OpCodes.Call, P.As256(typeof(sbyte), typeof(long)), null);
                        il.Emit(OpCodes.Ldc_I4, 0xD8);
                        il.EmitCall(OpCodes.Call, P.Permute4x64, null);
                        il.EmitCall(OpCodes.Call, P.As256(typeof(long), typeof(sbyte)), null);
                        P.EmitOnes256(il);
                        il.EmitCall(OpCodes.Call, P.And256, null);
                        il.Emit(OpCodes.Stloc, locBytes);
                        PushResultPtr(pair * 2L * vectorCount);
                        il.Emit(OpCodes.Ldloc, locBytes);
                        il.EmitCall(OpCodes.Call, P.Store256, null);
                    }
                    return;
                }
                case 4:
                {
                    var locBytes = il.DeclareLocal(typeof(Vector256<sbyte>));
                    LoadAs(masks[0], typeof(int));
                    LoadAs(masks[1], typeof(int));
                    il.EmitCall(OpCodes.Call, P.PackInt32, null);                              // p01: V256<short>
                    LoadAs(masks[2], typeof(int));
                    LoadAs(masks[3], typeof(int));
                    il.EmitCall(OpCodes.Call, P.PackInt32, null);                              // p23
                    il.EmitCall(OpCodes.Call, P.PackInt16, null);                              // V256<sbyte>
                    il.EmitCall(OpCodes.Call, P.As256(typeof(sbyte), typeof(int)), null);
                    P.EmitDwordOrder(il);
                    il.EmitCall(OpCodes.Call, P.PermuteVar8x32, null);
                    il.EmitCall(OpCodes.Call, P.As256(typeof(int), typeof(sbyte)), null);
                    P.EmitOnes256(il);
                    il.EmitCall(OpCodes.Call, P.And256, null);
                    il.Emit(OpCodes.Stloc, locBytes);
                    PushResultPtr(0);
                    il.Emit(OpCodes.Ldloc, locBytes);
                    il.EmitCall(OpCodes.Call, P.Store256, null);
                    return;
                }
                case 8:
                {
                    var locPicked = il.DeclareLocal(typeof(Vector256<sbyte>));
                    var locBytes = il.DeclareLocal(typeof(Vector128<sbyte>));
                    LoadAs(masks[0], typeof(int));
                    LoadAs(masks[1], typeof(int));
                    il.EmitCall(OpCodes.Call, P.PackInt32, null);
                    LoadAs(masks[2], typeof(int));
                    LoadAs(masks[3], typeof(int));
                    il.EmitCall(OpCodes.Call, P.PackInt32, null);
                    il.EmitCall(OpCodes.Call, P.PackInt16, null);                              // every lane twice
                    P.EmitEvenBytesControl(il);
                    il.EmitCall(OpCodes.Call, P.ShuffleBytes, null);
                    il.Emit(OpCodes.Stloc, locPicked);
                    il.Emit(OpCodes.Ldloc, locPicked);
                    il.EmitCall(OpCodes.Call, P.GetLower, null);
                    il.EmitCall(OpCodes.Call, P.As128(typeof(sbyte), typeof(short)), null);
                    il.Emit(OpCodes.Ldloc, locPicked);
                    il.EmitCall(OpCodes.Call, P.GetUpper, null);
                    il.EmitCall(OpCodes.Call, P.As128(typeof(sbyte), typeof(short)), null);
                    il.EmitCall(OpCodes.Call, P.UnpackLowInt16, null);
                    il.EmitCall(OpCodes.Call, P.As128(typeof(short), typeof(sbyte)), null);
                    P.EmitOnes128(il);
                    il.EmitCall(OpCodes.Call, P.And128, null);
                    il.Emit(OpCodes.Stloc, locBytes);
                    PushResultPtr(0);
                    il.Emit(OpCodes.Ldloc, locBytes);
                    il.EmitCall(OpCodes.Call, P.Store128, null);
                    return;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(laneBytes), laneBytes, "EmitPackedMasksToBools: 2-, 4- or 8-byte lanes only — caller bug.");
            }
        }

        /// <summary>
        ///     The AVX2 intrinsics <see cref="EmitPackedMasksToBools"/> calls, resolved once. <see cref="Available"/> is
        ///     false on any host whose kernels are not 256-bit AVX2 (128-bit, AVX-512 and ARM hosts store each mask
        ///     through <see cref="EmitMaskToBoolExtraction"/>), and the instance is then never built.
        /// </summary>
        private sealed class ComparisonPackIntrinsics
        {
            /// <summary>True when the kernels emit 256-bit vectors on a host with AVX2.</summary>
            internal static readonly bool Available = VectorBits == 256 && System.Runtime.Intrinsics.X86.Avx2.IsSupported;

            /// <summary>The resolved handles (null when <see cref="Available"/> is false: nothing may read them then).</summary>
            internal static readonly ComparisonPackIntrinsics Instance = Available ? new ComparisonPackIntrinsics() : null;

            internal readonly MethodInfo PackInt32, PackInt16, PermuteVar8x32, Permute4x64, ShuffleBytes, And256, Store256,
                GetLower, GetUpper, UnpackLowInt16, And128, Store128, Create256Int32x8, Create256SByteX32, Create256SByte,
                Create128SByte;
            private readonly MethodInfo _as256, _as128;

            private ComparisonPackIntrinsics()
            {
                var avx2 = typeof(System.Runtime.Intrinsics.X86.Avx2);
                var avx = typeof(System.Runtime.Intrinsics.X86.Avx);
                var sse2 = typeof(System.Runtime.Intrinsics.X86.Sse2);
                PackInt32 = Get(avx2, "PackSignedSaturate", typeof(Vector256<int>), typeof(Vector256<int>));
                PackInt16 = Get(avx2, "PackSignedSaturate", typeof(Vector256<short>), typeof(Vector256<short>));
                PermuteVar8x32 = Get(avx2, "PermuteVar8x32", typeof(Vector256<int>), typeof(Vector256<int>));
                Permute4x64 = Get(avx2, "Permute4x64", typeof(Vector256<long>), typeof(byte));
                ShuffleBytes = Get(avx2, "Shuffle", typeof(Vector256<sbyte>), typeof(Vector256<sbyte>));
                And256 = Get(avx2, "And", typeof(Vector256<sbyte>), typeof(Vector256<sbyte>));
                Store256 = Get(avx, "Store", typeof(sbyte*), typeof(Vector256<sbyte>));
                UnpackLowInt16 = Get(sse2, "UnpackLow", typeof(Vector128<short>), typeof(Vector128<short>));
                And128 = Get(sse2, "And", typeof(Vector128<sbyte>), typeof(Vector128<sbyte>));
                Store128 = Get(sse2, "Store", typeof(sbyte*), typeof(Vector128<sbyte>));
                GetLower = GenericOne(typeof(Vector256), "GetLower", typeof(sbyte));
                GetUpper = GenericOne(typeof(Vector256), "GetUpper", typeof(sbyte));
                Create256Int32x8 = Get(typeof(Vector256), "Create", Enumerable.Repeat(typeof(int), 8).ToArray());
                Create256SByteX32 = Get(typeof(Vector256), "Create", Enumerable.Repeat(typeof(sbyte), 32).ToArray());
                Create256SByte = Get(typeof(Vector256), "Create", typeof(sbyte));
                Create128SByte = Get(typeof(Vector128), "Create", typeof(sbyte));
                _as256 = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Single(m => m.Name == "As" && m.GetGenericArguments().Length == 2);
                _as128 = typeof(Vector128).GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Single(m => m.Name == "As" && m.GetGenericArguments().Length == 2);
            }

            /// <summary>A public static method by exact parameter types; a missing one fails fast at first use.</summary>
            /// <exception cref="MissingMethodException">The runtime has no such overload.</exception>
            private static MethodInfo Get(Type owner, string name, params Type[] parameters)
                => owner.GetMethod(name, BindingFlags.Public | BindingFlags.Static, null, parameters, null)
                   ?? throw new MissingMethodException(owner.FullName, name);

            /// <summary>A one-type-parameter generic static method closed over <paramref name="t"/>.</summary>
            /// <exception cref="InvalidOperationException">The owner has no such single generic method.</exception>
            private static MethodInfo GenericOne(Type owner, string name, Type t)
                => owner.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Single(m => m.Name == name && m.IsGenericMethodDefinition && m.GetGenericArguments().Length == 1
                                 && m.GetParameters().Length == 1)
                    .MakeGenericMethod(t);

            /// <summary><c>Vector256.As&lt;from, to&gt;</c>: a free reinterpretation of the 256 bits.</summary>
            internal MethodInfo As256(Type from, Type to) => _as256.MakeGenericMethod(from, to);

            /// <summary><c>Vector128.As&lt;from, to&gt;</c>.</summary>
            internal MethodInfo As128(Type from, Type to) => _as128.MakeGenericMethod(from, to);

            /// <summary>Pushes <c>Vector256.Create((sbyte)1)</c>, the AND that turns -1 / 0 bytes into 1 / 0.</summary>
            internal void EmitOnes256(ILGenerator il)
            {
                il.Emit(OpCodes.Ldc_I4_1);
                il.EmitCall(OpCodes.Call, Create256SByte, null);
            }

            /// <summary>Pushes <c>Vector128.Create((sbyte)1)</c>.</summary>
            internal void EmitOnes128(ILGenerator il)
            {
                il.Emit(OpCodes.Ldc_I4_1);
                il.EmitCall(OpCodes.Call, Create128SByte, null);
            }

            /// <summary>Pushes the dword order (0, 4, 1, 5, 2, 6, 3, 7) that undoes the per-half packing.</summary>
            internal void EmitDwordOrder(ILGenerator il)
            {
                foreach (int k in new[] { 0, 4, 1, 5, 2, 6, 3, 7 })
                    il.Emit(OpCodes.Ldc_I4, k);
                il.EmitCall(OpCodes.Call, Create256Int32x8, null);
            }

            /// <summary>Pushes the <c>vpshufb</c> control that moves each half's even bytes to its low 8 bytes (the high
            ///     8 bytes, control -1, become zero and are never stored).</summary>
            internal void EmitEvenBytesControl(ILGenerator il)
            {
                for (int half = 0; half < 2; half++)
                {
                    for (int k = 0; k < 8; k++)
                        il.Emit(OpCodes.Ldc_I4, 2 * k);
                    for (int k = 0; k < 8; k++)
                        il.Emit(OpCodes.Ldc_I4_M1);
                }
                il.EmitCall(OpCodes.Call, Create256SByteX32, null);
            }
        }

        /// <summary>
        /// Emit a scalar loop for contiguous comparison.
        /// </summary>
        private static void EmitComparisonScalarLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            // Args: void* lhs (0), void* rhs (1), bool* result (2),
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

            // result[i] = (lhs[i] op rhs[i])
            // Load result address
            il.Emit(OpCodes.Ldarg_2); // result (bool*)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add); // bool is 1 byte, so just add i

            // Load lhs[i] and convert to comparison type
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs[i] and convert to comparison type
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, comparisonType);

            // Perform comparison
            EmitComparisonOperation(il, key.Op, comparisonType);

            // Store bool result
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        ///     Tells whether a scalar-broadcast comparison kernel can compare its array operand a whole vector at a time.
        /// </summary>
        /// <param name="arrayType">The dtype of the array operand (the one that is not one element).</param>
        /// <param name="comparisonType">The dtype both operands are compared in (<see cref="ComparisonKernelKey.ComparisonType"/>).</param>
        /// <returns>
        ///     <see langword="true"/> when SIMD hardware is present, the array is already in the comparison dtype and that
        ///     dtype has vector compares (every integer width, float32, float64, and bool through its byte lanes).
        /// </returns>
        /// <remarks>
        ///     The array must already be in the comparison dtype because the vector body loads its elements as they are;
        ///     a widening comparison (an int8 array against an int32 scalar) would need a vector conversion per element and
        ///     stays on the scalar loop. The dispatcher re-packs a Python-int scalar that fits an integer array's dtype into
        ///     that dtype (<c>DefaultEngine.AdoptWeakComparisonScalar</c>), so the common <c>uint8 &gt; 128</c> form reaches
        ///     this path. The scalar's own dtype does not matter: it is converted once, exactly as the scalar loop converts it.
        /// </remarks>
        private static bool CanUseComparisonBroadcastSimd(NPTypeCode arrayType, NPTypeCode comparisonType)
        {
            if (VectorBits == 0)
                return false;
            if (arrayType != comparisonType)
                return false;
            return CanUseSimd(comparisonType) || comparisonType == NPTypeCode.Boolean;
        }

        /// <summary>
        ///     Emits the vector part of a scalar-broadcast comparison: a 4×-unrolled loop and a one-vector loop over the
        ///     array operand, each lane compared with the scalar broadcast into a vector once.
        /// </summary>
        /// <param name="il">The kernel's IL generator. The kernel's arguments follow <see cref="ComparisonKernel"/>.</param>
        /// <param name="key">The kernel key; only its comparison operator is read.</param>
        /// <param name="comparisonType">The dtype both operands are compared in, which is also the array operand's dtype.</param>
        /// <param name="scalarOnRight">
        ///     <see langword="true"/> for <c>array op scalar</c>, <see langword="false"/> for <c>scalar op array</c>. The
        ///     operand order is kept: <c>&lt;</c> is not symmetric, and NaN lanes must see the same compare as the scalar loop.
        /// </param>
        /// <param name="arrayArg">The kernel argument index that holds the array's base pointer: 0 (lhs) or 1 (rhs).</param>
        /// <param name="arraySize">The array element size in bytes.</param>
        /// <param name="locScalar">The local that already holds the scalar, converted to <paramref name="comparisonType"/>.</param>
        /// <param name="locI">
        ///     The element counter, 0 on entry. On exit it holds the first element the vector loops did not process; the
        ///     caller's scalar loop continues from there.
        /// </param>
        /// <remarks>
        ///     The vector compare is the one the same-dtype <c>SimdFull</c> kernel uses (<see cref="EmitVectorComparison"/>):
        ///     ordered float compares, so a NaN lane is False except under <c>!=</c>, and bool lanes compared by truth value.
        ///     The four masks of the unrolled body become result bytes through <see cref="EmitMasksToBools"/> (packed on
        ///     AVX2 hosts), the remainder's single mask through <see cref="EmitMaskToBoolExtraction"/>. Each vector is a
        ///     full load from a contiguous run, so the loops stop while a whole vector still fits and never read past the
        ///     array.
        /// </remarks>
        private static void EmitComparisonBroadcastSimdLoop(ILGenerator il, ComparisonKernelKey key, NPTypeCode comparisonType,
            bool scalarOnRight, int arrayArg, int arraySize, LocalBuilder locScalar, LocalBuilder locI)
        {
            long vectorCount = GetVectorCount(comparisonType);
            const int unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;
            // Bool rides the byte lanes (GetSimdLaneType), as the SimdFull kernel's masks do.
            var vectorType = GetVectorType(GetSimdLaneType(comparisonType));
            var arrayLoad = arrayArg == 0 ? OpCodes.Ldarg_0 : OpCodes.Ldarg_1;

            // The scalar is broadcast ONCE, before the loops: the body then reads a register, not memory.
            var locBroadcast = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Ldloc, locScalar);
            EmitVectorCreate(il, comparisonType);
            il.Emit(OpCodes.Stloc, locBroadcast);

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));
            // unrollEnd = totalSize - unrollStep (the last start of four whole vectors: the loop runs while i <= it, so
            // i + unrollStep <= totalSize); vectorEnd likewise for one vector. Never past the last element.
            il.Emit(OpCodes.Ldarg_S, (byte)7);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);
            il.Emit(OpCodes.Ldarg_S, (byte)7);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            var maskLocals = new LocalBuilder[unrollFactor];
            for (int n = 0; n < unrollFactor; n++)
                maskLocals[n] = il.DeclareLocal(vectorType);

            // Pushes array[i + offset] as a vector, then compares it with the broadcast in the kernel's operand order.
            void EmitCompareAt(long offset)
            {
                if (!scalarOnRight)
                    il.Emit(OpCodes.Ldloc, locBroadcast);
                il.Emit(arrayLoad);
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I8, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldc_I8, (long)arraySize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad(il, comparisonType);
                if (scalarOnRight)
                    il.Emit(OpCodes.Ldloc, locBroadcast);
                EmitVectorComparison(il, key.Op, comparisonType);
            }

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderEnd = il.DefineLabel();

            // === 4x UNROLLED: four independent compares, then four mask stores ===
            il.MarkLabel(lblUnrollLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);
            for (int n = 0; n < unrollFactor; n++)
            {
                EmitCompareAt(n * vectorCount);
                il.Emit(OpCodes.Stloc, maskLocals[n]);
            }
            EmitMasksToBools(il, comparisonType, (int)vectorCount, locI, maskLocals);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnrollLoop);

            // === ONE VECTOR AT A TIME for the last 0-3 whole vectors ===
            il.MarkLabel(lblUnrollEnd);
            il.MarkLabel(lblRemainderLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemainderEnd);
            EmitCompareAt(0);
            il.Emit(OpCodes.Stloc, maskLocals[0]);
            EmitMaskToBoolExtraction(il, comparisonType, (int)vectorCount, locI, maskLocals[0]);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRemainderLoop);

            il.MarkLabel(lblRemainderEnd);
        }

        /// <summary>
        /// Emit loop for scalar right operand comparison.
        /// </summary>
        /// <remarks>
        ///     An array already in the comparison dtype runs the vector loops first
        ///     (<see cref="EmitComparisonBroadcastSimdLoop"/>); the scalar loop below then finishes the last partial vector,
        ///     or the whole array when the array must be converted per element.
        /// </remarks>
        private static void EmitComparisonScalarRightLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            var locI = il.DeclareLocal(typeof(long)); // loop counter
            var locRhsVal = il.DeclareLocal(GetClrType(comparisonType)); // scalar value

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Load rhs[0] and convert to comparison type, store in local
            il.Emit(OpCodes.Ldarg_1); // rhs
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, comparisonType);
            il.Emit(OpCodes.Stloc, locRhsVal);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (CanUseComparisonBroadcastSimd(key.LhsType, comparisonType))
                EmitComparisonBroadcastSimdLoop(il, key, comparisonType, scalarOnRight: true, arrayArg: 0,
                    arraySize: lhsSize, locScalar: locRhsVal, locI: locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = (lhs[i] op rhsVal)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
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
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load cached rhs scalar
            il.Emit(OpCodes.Ldloc, locRhsVal);

            EmitComparisonOperation(il, key.Op, comparisonType);
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit loop for scalar left operand comparison.
        /// </summary>
        /// <remarks>
        ///     The mirror of <see cref="EmitComparisonScalarRightLoop"/>: the vector loops run first when the array (the
        ///     right operand) is already in the comparison dtype, keeping <c>scalar op array</c> operand order.
        /// </remarks>
        private static void EmitComparisonScalarLeftLoop(ILGenerator il, ComparisonKernelKey key,
            int lhsSize, int rhsSize, NPTypeCode comparisonType)
        {
            var locI = il.DeclareLocal(typeof(long)); // loop counter
            var locLhsVal = il.DeclareLocal(GetClrType(comparisonType)); // scalar value

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // Load lhs[0] and convert to comparison type, store in local
            il.Emit(OpCodes.Ldarg_0); // lhs
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);
            il.Emit(OpCodes.Stloc, locLhsVal);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (CanUseComparisonBroadcastSimd(key.RhsType, comparisonType))
                EmitComparisonBroadcastSimdLoop(il, key, comparisonType, scalarOnRight: false, arrayArg: 1,
                    arraySize: rhsSize, locScalar: locLhsVal, locI: locI);

            il.MarkLabel(lblLoop);

            // if (i >= totalSize) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)7); // totalSize
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // result[i] = (lhsVal op rhs[i])
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
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
            EmitConvertTo(il, key.RhsType, comparisonType);

            EmitComparisonOperation(il, key.Op, comparisonType);
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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
            //       long* lhsStrides (3), long* rhsStrides (4), long* shape (5),
            //       int ndim (6), long totalSize (7)

            var locI = il.DeclareLocal(typeof(long)); // linear index
            var locD = il.DeclareLocal(typeof(int)); // dimension counter
            var locLhsOffset = il.DeclareLocal(typeof(long)); // lhs offset
            var locRhsOffset = il.DeclareLocal(typeof(long)); // rhs offset
            var locCoord = il.DeclareLocal(typeof(long)); // current coordinate
            var locIdx = il.DeclareLocal(typeof(long)); // temp for coordinate calculation

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
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I8, 0L);
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

            // result[i] = (lhs[lhsOffset] op rhs[rhsOffset])
            // Load result address (contiguous output)
            il.Emit(OpCodes.Ldarg_2); // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Add);

            // Load lhs[lhsOffset]
            il.Emit(OpCodes.Ldarg_0); // lhs
            il.Emit(OpCodes.Ldloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I8, (long)lhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.LhsType);
            EmitConvertTo(il, key.LhsType, comparisonType);

            // Load rhs[rhsOffset]
            il.Emit(OpCodes.Ldarg_1); // rhs
            il.Emit(OpCodes.Ldloc, locRhsOffset);
            il.Emit(OpCodes.Ldc_I8, (long)rhsSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.RhsType);
            EmitConvertTo(il, key.RhsType, comparisonType);

            // Comparison
            EmitComparisonOperation(il, key.Op, comparisonType);

            // Store bool
            il.Emit(OpCodes.Stind_I1);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
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
            // Boolean compares by TRUTH VALUE (NumPy loops_comparison b8 scalar tails:
            // `a = *src1 != 0; b = *src2 != 0; *dst = a OP b`) — a raw byte Ceq would call
            // frombuffer 0x01 == 0x80 False where NumPy says True. Normalize both to 0/1
            // first; the SIMD body (EmitBoolVectorComparison) uses the same semantics, so
            // body and tail agree on every byte value. Stack in: [a(u1-as-i4), b(u1-as-i4)].
            if (comparisonType == NPTypeCode.Boolean)
            {
                var locB = il.DeclareLocal(typeof(int));
                il.Emit(OpCodes.Stloc, locB);       // [a]
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Cgt_Un);            // [na]
                il.Emit(OpCodes.Ldloc, locB);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Cgt_Un);            // [na, nb]
                switch (op)
                {
                    case ComparisonOp.Equal:
                        il.Emit(OpCodes.Ceq);
                        break;
                    case ComparisonOp.NotEqual:
                        il.Emit(OpCodes.Ceq);
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                        break;
                    case ComparisonOp.Less:
                        il.Emit(OpCodes.Clt);
                        break;
                    case ComparisonOp.LessEqual:
                        il.Emit(OpCodes.Cgt);
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                        break;
                    case ComparisonOp.Greater:
                        il.Emit(OpCodes.Cgt);
                        break;
                    case ComparisonOp.GreaterEqual:
                        il.Emit(OpCodes.Clt);
                        il.Emit(OpCodes.Ldc_I4_0);
                        il.Emit(OpCodes.Ceq);
                        break;
                    default:
                        throw new NotSupportedException($"Comparison operation {op} not supported");
                }
                return;
            }

            // Special handling for decimal comparisons
            if (comparisonType == NPTypeCode.Decimal)
            {
                EmitDecimalComparison(il, op);
                return;
            }

            // Special handling for Half comparisons (uses operator methods)
            if (comparisonType == NPTypeCode.Half)
            {
                EmitHalfComparison(il, op);
                return;
            }

            // Special handling for Complex comparisons (only == and != supported)
            if (comparisonType == NPTypeCode.Complex)
            {
                EmitComplexComparison(il, op);
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
                    // a <= b is !(a > b). For floats the ">" must be the UNORDERED compare
                    // (Cgt_Un): with NaN it yields true, so the negation is false — matching
                    // IEEE/NumPy where every ordered comparison with NaN is false. Signed Cgt
                    // would yield false for NaN, negating to true (the bug). Cgt_Un also serves
                    // unsigned integer comparison (its original use here).
                    if (isUnsigned || isFloat)
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
                    // a >= b is !(a < b). For floats the "<" must be the UNORDERED compare
                    // (Clt_Un) so a NaN operand yields true and the negation is false (IEEE/NumPy).
                    // Clt_Un also serves unsigned integer comparison.
                    if (isUnsigned || isFloat)
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

            il.EmitCall(OpCodes.Call, ScalarMethodCache.BinaryOp(typeof(decimal), methodName), null);
        }

        /// <summary>
        /// Emit Half comparison using operator methods.
        /// </summary>
        private static void EmitHalfComparison(ILGenerator il, ComparisonOp op)
        {
            string methodName = op switch
            {
                ComparisonOp.Equal => "op_Equality",
                ComparisonOp.NotEqual => "op_Inequality",
                ComparisonOp.Less => "op_LessThan",
                ComparisonOp.LessEqual => "op_LessThanOrEqual",
                ComparisonOp.Greater => "op_GreaterThan",
                ComparisonOp.GreaterEqual => "op_GreaterThanOrEqual",
                _ => throw new NotSupportedException($"Comparison {op} not supported for Half")
            };
            il.EmitCall(OpCodes.Call, ScalarMethodCache.BinaryOp(typeof(Half), methodName), null);
        }

        /// <summary>
        /// Emit Complex comparison using operator methods or lexicographic ordering.
        /// NumPy 2.x supports ordered comparisons (&lt;, &gt;, &lt;=, &gt;=) using lexicographic ordering:
        /// first by real part, then by imaginary part.
        /// </summary>
        private static void EmitComplexComparison(ILGenerator il, ComparisonOp op)
        {
            // For == and !=, use the built-in operators
            if (op == ComparisonOp.Equal || op == ComparisonOp.NotEqual)
            {
                string methodName = op == ComparisonOp.Equal ? "op_Equality" : "op_Inequality";
                il.EmitCall(OpCodes.Call,
                    ScalarMethodCache.BinaryOp(typeof(System.Numerics.Complex), methodName), null);
                return;
            }

            // For ordered comparisons, use lexicographic ordering (NumPy 2.x behavior):
            //   a vs b  =  first by Real, then by Imaginary.
            // Stack: [lhs: Complex a, rhs: Complex b]
            EmitComplexLexCompare(il, op);
        }

        /// <summary>
        /// Emit IL for Complex lexicographic ordered comparison. Pseudo-C#:
        /// <code>
        ///   if (strict(a.Real, b.Real)) return true;    // Real strictly on the "true" side
        ///   if (strict(b.Real, a.Real)) return false;   // Real strictly on the "false" side
        ///   return strict(a.Imag, b.Imag) || (inclusive &amp;&amp; a.Imag == b.Imag);
        /// </code>
        /// where <c>strict</c> is &lt; for Less/LessEqual and &gt; for Greater/GreaterEqual,
        /// and <c>inclusive</c> accepts equality for LessEqual / GreaterEqual.
        ///
        /// Stack contract: expects [Complex a, Complex b] (a pushed first, b on top),
        /// leaves [bool] on top.
        /// </summary>
        private static void EmitComplexLexCompare(ILGenerator il, ComparisonOp op)
        {
            // Map the 4 ops to the 3 opcode choices that fully parameterize the emit.
            // realBranchTrue  — branch to "return true" when real parts are strictly on the "true" side
            //                   (Less/LessEqual: aR < bR → true; Greater/GreaterEqual: aR > bR → true)
            // realBranchFalse — branch to "return false" when reals are strictly on the reverse side
            //                   (Less/LessEqual: aR > bR → false; Greater/GreaterEqual: aR < bR → false)
            // imagStrictCmp   — Clt or Cgt for the final imaginary compare; inclusive adds |Ceq.
            OpCode realBranchTrue, realBranchFalse, imagStrictCmp;
            bool inclusive;
            switch (op)
            {
                case ComparisonOp.Less:
                    realBranchTrue = OpCodes.Blt; realBranchFalse = OpCodes.Bgt;
                    imagStrictCmp = OpCodes.Clt; inclusive = false; break;
                case ComparisonOp.LessEqual:
                    realBranchTrue = OpCodes.Blt; realBranchFalse = OpCodes.Bgt;
                    imagStrictCmp = OpCodes.Clt; inclusive = true; break;
                case ComparisonOp.Greater:
                    realBranchTrue = OpCodes.Bgt; realBranchFalse = OpCodes.Blt;
                    imagStrictCmp = OpCodes.Cgt; inclusive = false; break;
                case ComparisonOp.GreaterEqual:
                    realBranchTrue = OpCodes.Bgt; realBranchFalse = OpCodes.Blt;
                    imagStrictCmp = OpCodes.Cgt; inclusive = true; break;
                default:
                    throw new NotSupportedException($"Comparison {op} not supported for Complex");
            }

            var locA = il.DeclareLocal(typeof(System.Numerics.Complex));
            var locB = il.DeclareLocal(typeof(System.Numerics.Complex));
            var locAR = il.DeclareLocal(typeof(double));
            var locBR = il.DeclareLocal(typeof(double));
            var lblTrue = il.DefineLabel();
            var lblFalse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // Pop b then a (stack LIFO)
            il.Emit(OpCodes.Stloc, locB);
            il.Emit(OpCodes.Stloc, locA);

            // Cache the Real components — they're referenced twice in the real-branch chain.
            il.Emit(OpCodes.Ldloca, locA); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetReal, null); il.Emit(OpCodes.Stloc, locAR);
            il.Emit(OpCodes.Ldloca, locB); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetReal, null); il.Emit(OpCodes.Stloc, locBR);

            // B25: NaN short-circuit. NumPy returns False for any ordered comparison when
            // *any* component of either operand is NaN. Without this guard, aR=NaN would
            // fall through Blt/Bgt (both false for NaN) into the imag compare which could
            // return true. Check all four components up front; bail to lblFalse on NaN.
            il.Emit(OpCodes.Ldloc, locAR);
            il.EmitCall(OpCodes.Call, CachedMethods.DoubleIsNaN, null);
            il.Emit(OpCodes.Brtrue, lblFalse);
            il.Emit(OpCodes.Ldloc, locBR);
            il.EmitCall(OpCodes.Call, CachedMethods.DoubleIsNaN, null);
            il.Emit(OpCodes.Brtrue, lblFalse);
            il.Emit(OpCodes.Ldloca, locA); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
            il.EmitCall(OpCodes.Call, CachedMethods.DoubleIsNaN, null);
            il.Emit(OpCodes.Brtrue, lblFalse);
            il.Emit(OpCodes.Ldloca, locB); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
            il.EmitCall(OpCodes.Call, CachedMethods.DoubleIsNaN, null);
            il.Emit(OpCodes.Brtrue, lblFalse);

            // if (strict(aR, bR)) goto lblTrue;
            il.Emit(OpCodes.Ldloc, locAR); il.Emit(OpCodes.Ldloc, locBR);
            il.Emit(realBranchTrue, lblTrue);
            // if (reverseStrict(aR, bR)) goto lblFalse;
            il.Emit(OpCodes.Ldloc, locAR); il.Emit(OpCodes.Ldloc, locBR);
            il.Emit(realBranchFalse, lblFalse);

            // Reals tied — compare imaginaries: strict(aI, bI) [| ceq(aI, bI) if inclusive]
            il.Emit(OpCodes.Ldloca, locA); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
            il.Emit(OpCodes.Ldloca, locB); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
            il.Emit(imagStrictCmp);
            if (inclusive)
            {
                il.Emit(OpCodes.Ldloca, locA); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
                il.Emit(OpCodes.Ldloca, locB); il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
                il.Emit(OpCodes.Ceq);
                il.Emit(OpCodes.Or);
            }
            il.Emit(OpCodes.Br, lblEnd);

            il.MarkLabel(lblTrue);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Br, lblEnd);

            il.MarkLabel(lblFalse);
            il.Emit(OpCodes.Ldc_I4_0);

            il.MarkLabel(lblEnd);
        }

        #endregion

        #region Comparison Scalar Kernel Generation

        // ComparisonScalarKernelKey is now defined in ScalarKernel.cs

        /// <summary>
        /// Cache for comparison scalar kernels.
        /// </summary>
        internal static readonly ConcurrentDictionary<ComparisonScalarKernelKey, Delegate> _comparisonScalarCache = new();

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
                owner: typeof(DirectILKernelGenerator),
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
