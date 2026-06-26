using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
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
// DirectILKernelGenerator.Binary.cs (THIS FILE)
//   OWNERSHIP: Same-type binary operations on contiguous arrays (fast path)
//   RESPONSIBILITY:
//     - Optimized kernels when both operands have identical type and layout
//     - SIMD loop + scalar tail for Add, Sub, Mul, Div
//   DEPENDENCIES: Uses core emit helpers from DirectILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for same-type contiguous operations
//   KEY MEMBERS:
//     - ContiguousKernel<T> delegate - simplified signature for contiguous arrays
//     - _contiguousKernelCache - caches generated kernels by (BinaryOp, Type)
//     - GetContiguousKernel<T>() - main entry point for contiguous kernels
//     - TryGenerateContiguousKernelIL<T>() - IL generation with SIMD loop
//     - Generic helpers duplicated for type safety: IsSimdSupported<T>(), etc.
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
    /// Delegate for contiguous (SimdFull) binary operations.
    /// Simplified signature - no strides needed since both arrays are contiguous.
    /// </summary>
    /// <typeparam name="T">Element type (must be unmanaged).</typeparam>
    /// <param name="lhs">Pointer to left operand data.</param>
    /// <param name="rhs">Pointer to right operand data.</param>
    /// <param name="result">Pointer to output data.</param>
    /// <param name="count">Number of elements to process.</param>
    public unsafe delegate void ContiguousKernel<T>(T* lhs, T* rhs, T* result, long count) where T : unmanaged;

    /// <summary>
    /// Binary operations (same-type) - contiguous kernels and generic helpers.
    /// </summary>
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Cache of IL-generated contiguous kernels.
        /// Key: (Operation, Type)
        /// </summary>
        internal static readonly ConcurrentDictionary<(BinaryOp, Type), Delegate> _contiguousKernelCache = new();

        #region Public API

        #endregion

        #region Contiguous Kernel Generation

        // NOTE: ContiguousKernel<T> delegate is defined in KernelSignatures.cs

        /// <summary>
        /// Try to generate an IL-based contiguous kernel for the given operation and type.
        /// </summary>
        private static ContiguousKernel<T>? TryGenerateContiguousKernel<T>(BinaryOp op) where T : unmanaged
        {
            // Only support types with Vector256 support for SIMD ops,
            // but Power/FloorDivide can work with scalar loop on any type
            bool isSimdOp = op == BinaryOp.Add || op == BinaryOp.Subtract ||
                           op == BinaryOp.Multiply || op == BinaryOp.Divide ||
                           op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor;

            bool isScalarOnlyOp = op == BinaryOp.Power || op == BinaryOp.FloorDivide;

            if (isSimdOp && !IsSimdSupported<T>())
                return null;

            // Only support basic arithmetic, bitwise operations, and scalar-only ops
            if (!isSimdOp && !isScalarOnlyOp)
                return null;

            // Bitwise operations only supported on integer types
            if ((op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor) &&
                !IsIntegerType<T>())
                return null;

            try
            {
                return GenerateContiguousKernelIL<T>(op);
            }
            catch (Exception ex)
            {
                // IL generation failed - fall back to C#
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGenerateContiguousKernel<{typeof(T).Name}>({op}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Generate IL for a contiguous SIMD kernel.
        /// For scalar-only ops (Power, FloorDivide), generates a pure scalar loop.
        /// </summary>
        private static unsafe ContiguousKernel<T> GenerateContiguousKernelIL<T>(BinaryOp op) where T : unmanaged
        {
            var dm = new DynamicMethod(
                name: $"IL_Contiguous_{op}_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(T*), typeof(T*), typeof(T*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Check if this is a scalar-only operation (no SIMD support)
            bool isScalarOnly = op == BinaryOp.Power || op == BinaryOp.FloorDivide;

            // Declare locals
            var locI = il.DeclareLocal(typeof(long));           // loop counter

            int elementSize = Unsafe.SizeOf<T>();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (!isScalarOnly)
            {
                // SIMD-capable operations: generate 4x unrolled SIMD loop + remainder + tail loop
                var locVectorEnd = il.DeclareLocal(typeof(long));   // count - vectorCount (for remainder loop)
                var locUnrollEnd = il.DeclareLocal(typeof(long));   // count - vectorCount*4 (for 4x unrolled loop)

                // Define labels
                var lblUnrollLoop = il.DefineLabel();
                var lblUnrollLoopEnd = il.DefineLabel();
                var lblRemainderLoop = il.DefineLabel();
                var lblRemainderLoopEnd = il.DefineLabel();
                var lblTailLoop = il.DefineLabel();
                var lblTailLoopEnd = il.DefineLabel();

                int vectorCount = GetVectorCount<T>();
                int unrollStep = vectorCount * 4;

                // vectorEnd = count - vectorCount (for remainder loop)
                il.Emit(OpCodes.Ldarg_3);                      // count (now long)
                il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locVectorEnd);

                // unrollEnd = count - vectorCount*4 (for 4x unrolled loop)
                il.Emit(OpCodes.Ldarg_3);                      // count (now long)
                il.Emit(OpCodes.Ldc_I8, (long)unrollStep);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locUnrollEnd);

                // ========== 4x UNROLLED SIMD LOOP ==========
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
                    il.Emit(OpCodes.Ldarg_0);                      // lhs
                    il.Emit(OpCodes.Ldloc, locI);
                    if (offset > 0)
                    {
                        il.Emit(OpCodes.Ldc_I8, (long)offset);
                        il.Emit(OpCodes.Add);
                    }
                    il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    EmitVectorLoad<T>(il);

                    // Load rhs vector at (i + offset)
                    il.Emit(OpCodes.Ldarg_1);                      // rhs
                    il.Emit(OpCodes.Ldloc, locI);
                    if (offset > 0)
                    {
                        il.Emit(OpCodes.Ldc_I8, (long)offset);
                        il.Emit(OpCodes.Add);
                    }
                    il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    EmitVectorLoad<T>(il);

                    // Perform vector operation
                    EmitVectorOperation<T>(il, op);

                    // Store result at (i + offset)
                    il.Emit(OpCodes.Ldarg_2);                      // result
                    il.Emit(OpCodes.Ldloc, locI);
                    if (offset > 0)
                    {
                        il.Emit(OpCodes.Ldc_I8, (long)offset);
                        il.Emit(OpCodes.Add);
                    }
                    il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    EmitVectorStore<T>(il);
                }

                // i += vectorCount * 4
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)unrollStep);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblUnrollLoop);
                il.MarkLabel(lblUnrollLoopEnd);

                // ========== REMAINDER SIMD LOOP (0-3 vectors) ==========
                il.MarkLabel(lblRemainderLoop);

                // if (i > vectorEnd) goto RemainderLoopEnd
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locVectorEnd);
                il.Emit(OpCodes.Bgt, lblRemainderLoopEnd);

                // Load lhs vector: Vector256.Load(lhs + i)
                il.Emit(OpCodes.Ldarg_0);                      // lhs
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad<T>(il);

                // Load rhs vector: Vector256.Load(rhs + i)
                il.Emit(OpCodes.Ldarg_1);                      // rhs
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorLoad<T>(il);

                // Perform vector operation
                EmitVectorOperation<T>(il, op);

                // Store result: Vector256.Store(result, result + i)
                il.Emit(OpCodes.Ldarg_2);                      // result
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitVectorStore<T>(il);

                // i += vectorCount
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblRemainderLoop);
                il.MarkLabel(lblRemainderLoopEnd);

                // ========== TAIL LOOP (scalar) ==========
                il.MarkLabel(lblTailLoop);

                // if (i >= count) goto TailLoopEnd
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_3);                      // count (now long)
                il.Emit(OpCodes.Bge, lblTailLoopEnd);

                // result[i] = lhs[i] op rhs[i]
                EmitScalarLoopBody<T>(il, op, locI, elementSize);

                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblTailLoop);
                il.MarkLabel(lblTailLoopEnd);
            }
            else
            {
                // Scalar-only operations: generate pure scalar loop (no SIMD)
                var lblLoop = il.DefineLabel();
                var lblLoopEnd = il.DefineLabel();

                il.MarkLabel(lblLoop);

                // if (i >= count) goto LoopEnd
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_3);                      // count (now long)
                il.Emit(OpCodes.Bge, lblLoopEnd);

                // result[i] = lhs[i] op rhs[i]
                EmitScalarLoopBody<T>(il, op, locI, elementSize);

                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblLoop);
                il.MarkLabel(lblLoopEnd);
            }

            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<ContiguousKernel<T>>();
        }

        /// <summary>
        /// Emit the body of a scalar loop iteration: result[i] = lhs[i] op rhs[i]
        /// </summary>
        private static void EmitScalarLoopBody<T>(ILGenerator il, BinaryOp op, LocalBuilder locI, int elementSize) where T : unmanaged
        {
            // Address: result + i * elementSize
            il.Emit(OpCodes.Ldarg_2);                      // result
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load lhs[i]
            il.Emit(OpCodes.Ldarg_0);                      // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect<T>(il);

            // Load rhs[i]
            il.Emit(OpCodes.Ldarg_1);                      // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect<T>(il);

            // Perform scalar operation
            EmitScalarOperation<T>(il, op);

            // Store to result[i]
            EmitStoreIndirect<T>(il);
        }

        #endregion

        #region IL Emission Helpers (Generic)

        private static bool IsSimdSupported<T>() where T : unmanaged
        {
            return typeof(T) == typeof(int) ||
                   typeof(T) == typeof(long) ||
                   typeof(T) == typeof(float) ||
                   typeof(T) == typeof(double) ||
                   typeof(T) == typeof(byte) ||
                   typeof(T) == typeof(sbyte) ||
                   typeof(T) == typeof(short) ||
                   typeof(T) == typeof(uint) ||
                   typeof(T) == typeof(ulong) ||
                   typeof(T) == typeof(ushort);
        }

        private static bool IsIntegerType<T>() where T : unmanaged
        {
            return typeof(T) == typeof(int) ||
                   typeof(T) == typeof(long) ||
                   typeof(T) == typeof(byte) ||
                   typeof(T) == typeof(short) ||
                   typeof(T) == typeof(uint) ||
                   typeof(T) == typeof(ulong) ||
                   typeof(T) == typeof(ushort) ||
                   typeof(T) == typeof(sbyte);
        }

        private static int GetVectorCount<T>() where T : unmanaged
        {
            return VectorBits switch
            {
                512 => Vector512<T>.Count,
                256 => Vector256<T>.Count,
                128 => Vector128<T>.Count,
                _ => 1
            };
        }

        private static void EmitVectorLoad<T>(ILGenerator il) where T : unmanaged
            => il.EmitCall(OpCodes.Call, VectorMethodCache.Load(VectorBits, typeof(T)), null);

        private static void EmitVectorStore<T>(ILGenerator il) where T : unmanaged
            => il.EmitCall(OpCodes.Call, VectorMethodCache.Store(VectorBits, typeof(T)), null);

        private static void EmitVectorOperation<T>(ILGenerator il, BinaryOp op) where T : unmanaged
        {
            // Bitwise operations use static methods on Vector256/Vector128 container.
            if (op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor)
            {
                string methodName = op switch
                {
                    BinaryOp.BitwiseAnd => "BitwiseAnd",
                    BinaryOp.BitwiseOr => "BitwiseOr",
                    BinaryOp.BitwiseXor => "Xor",
                    _ => throw new NotSupportedException()
                };
                il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(VectorBits, methodName, typeof(T), paramCount: 2), null);
                return;
            }

            // Arithmetic operations use operator overloads on Vector256<T>/Vector128<T>.
            string operatorName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException($"Operation {op} not supported for SIMD")
            };
            il.EmitCall(OpCodes.Call, VectorMethodCache.Operator(VectorBits, typeof(T), operatorName), null);
        }

        private static void EmitScalarOperation<T>(ILGenerator il, BinaryOp op) where T : unmanaged
        {
            // Handle Power operation - requires Math.Pow call
            if (op == BinaryOp.Power)
            {
                EmitPowerOperation<T>(il);
                return;
            }

            // Handle FloorDivide operation
            if (op == BinaryOp.FloorDivide)
            {
                EmitFloorDivideOperation<T>(il);
                return;
            }

            // For scalar operations, use IL opcodes
            // Stack has two T values
            var opcode = op switch
            {
                BinaryOp.Add => OpCodes.Add,
                BinaryOp.Subtract => OpCodes.Sub,
                BinaryOp.Multiply => OpCodes.Mul,
                BinaryOp.Divide => GetDivOpcode<T>(),
                BinaryOp.BitwiseAnd => OpCodes.And,
                BinaryOp.BitwiseOr => OpCodes.Or,
                BinaryOp.BitwiseXor => OpCodes.Xor,
                _ => throw new NotSupportedException($"Operation {op} not supported")
            };

            il.Emit(opcode);
        }

        /// <summary>
        /// Emit Power operation for generic type T (same-type contiguous kernel path).
        /// Stack: [base, exponent] -> [result]
        ///
        /// - Integer T: routes to <see cref="Utilities.NpyIntegerPower"/> for dtype-native wrapping.
        /// - float: routes to <c>MathF.Pow</c> (single-precision parity with NumPy <c>powf</c>).
        /// - double: routes to <c>Math.Pow</c>.
        /// </summary>
        private static void EmitPowerOperation<T>(ILGenerator il) where T : unmanaged
        {
            // Integer types: call the matching NpyIntegerPower helper.
            // Stack already holds two T values; the helper signature is (T, T) -> T.
            MethodInfo? intPow = GetIntegerPowMethod<T>();
            if (intPow != null)
            {
                il.EmitCall(OpCodes.Call, intPow, null);
                return;
            }

            // float: MathF.Pow (no f64 round-trip)
            if (typeof(T) == typeof(float))
            {
                il.EmitCall(OpCodes.Call, CachedMethods.MathFPow, null);
                return;
            }

            // double: Math.Pow directly (operands already double)
            if (typeof(T) == typeof(double))
            {
                il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
                return;
            }

            // Fallback for unhandled T (e.g. Half, Complex, Decimal route through their own emit paths).
            // Convert both to double, call Math.Pow, convert back.
            var locExp = il.DeclareLocal(typeof(T));
            il.Emit(OpCodes.Stloc, locExp);
            EmitConvertToDouble<T>(il);
            il.Emit(OpCodes.Ldloc, locExp);
            EmitConvertToDouble<T>(il);
            il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
            EmitConvertFromDouble<T>(il);
        }

        /// <summary>
        /// Lookup the matching <see cref="Utilities.NpyIntegerPower"/> helper for T,
        /// or null if T is not an integer type supported by the helper.
        /// </summary>
        private static MethodInfo? GetIntegerPowMethod<T>() where T : unmanaged
        {
            var t = typeof(T);
            if (t == typeof(sbyte)) return CachedMethods.IntPowSByte;
            if (t == typeof(byte)) return CachedMethods.IntPowByte;
            if (t == typeof(short)) return CachedMethods.IntPowInt16;
            if (t == typeof(ushort)) return CachedMethods.IntPowUInt16;
            if (t == typeof(char)) return CachedMethods.IntPowChar;
            if (t == typeof(int)) return CachedMethods.IntPowInt32;
            if (t == typeof(uint)) return CachedMethods.IntPowUInt32;
            if (t == typeof(long)) return CachedMethods.IntPowInt64;
            if (t == typeof(ulong)) return CachedMethods.IntPowUInt64;
            return null;
        }

        /// <summary>
        /// Emit FloorDivide for generic type T via the <see cref="Utilities.NpyDivision"/> helpers,
        /// matching NumPy's <c>floor_div_@TYPE@</c> (integer ÷0 -> 0, signed floor toward -inf,
        /// MIN/-1 wrap) and <c>npy_floor_divide</c> (CPython divmod port for floats: ÷0 -> ±inf/nan,
        /// snap-to-nearest). Stack: [dividend, divisor] -> [result]
        /// </summary>
        private static void EmitFloorDivideOperation<T>(ILGenerator il) where T : unmanaged
        {
            var m = GetFloorDivideMethod(typeof(T));
            if (m != null)
            {
                il.EmitCall(OpCodes.Call, m, null);
                return;
            }
            // Non-numeric T should not reach this path; fall back to plain division.
            il.Emit(OpCodes.Div);
        }

        /// <summary>
        /// Return the <see cref="Utilities.NpyDivision"/> floor-division helper for the CLR type
        /// <paramref name="t"/>, or null if it routes elsewhere.
        /// </summary>
        private static MethodInfo? GetFloorDivideMethod(Type t)
        {
            if (t == typeof(sbyte)) return CachedMethods.FloorDivSByte;
            if (t == typeof(byte)) return CachedMethods.FloorDivByte;
            if (t == typeof(short)) return CachedMethods.FloorDivInt16;
            if (t == typeof(ushort)) return CachedMethods.FloorDivUInt16;
            if (t == typeof(char)) return CachedMethods.FloorDivChar;
            if (t == typeof(int)) return CachedMethods.FloorDivInt32;
            if (t == typeof(uint)) return CachedMethods.FloorDivUInt32;
            if (t == typeof(long)) return CachedMethods.FloorDivInt64;
            if (t == typeof(ulong)) return CachedMethods.FloorDivUInt64;
            if (t == typeof(float)) return CachedMethods.FloorDivSingle;
            if (t == typeof(double)) return CachedMethods.FloorDivDouble;
            return null;
        }

        /// <summary>
        /// Emit floor(div) with inf replaced by NaN, matching NumPy's floor_divide rule.
        /// Stack on entry: [div as double]. Stack on exit: [floor(div), or NaN if div was ±inf].
        /// floor(NaN) passes through; floor(finite) = floor(div).
        /// </summary>
        internal static void EmitFloorWithInfToNaN(ILGenerator il)
        {
            var floorMethod = ScalarMethodCache.MathFn1(typeof(double), "Floor");
            var isInfMethod = ScalarMethodCache.Predicate(typeof(double), nameof(double.IsInfinity));

            il.EmitCall(OpCodes.Call, floorMethod, null);
            var locR = il.DeclareLocal(typeof(double));
            il.Emit(OpCodes.Stloc, locR);
            il.Emit(OpCodes.Ldloc, locR);
            il.EmitCall(OpCodes.Call, isInfMethod, null);
            var lblFinite = il.DefineLabel();
            var lblDone = il.DefineLabel();
            il.Emit(OpCodes.Brfalse, lblFinite);
            il.Emit(OpCodes.Ldc_R8, double.NaN);
            il.Emit(OpCodes.Br, lblDone);
            il.MarkLabel(lblFinite);
            il.Emit(OpCodes.Ldloc, locR);
            il.MarkLabel(lblDone);
        }

        /// <summary>
        /// Emit conversion from T to double.
        /// </summary>
        private static void EmitConvertToDouble<T>(ILGenerator il) where T : unmanaged
        {
            if (typeof(T) == typeof(double))
                return; // Already double

            // For unsigned types, use Conv_R_Un first
            if (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) ||
                typeof(T) == typeof(uint) || typeof(T) == typeof(ulong))
            {
                il.Emit(OpCodes.Conv_R_Un);
            }

            il.Emit(OpCodes.Conv_R8);
        }

        /// <summary>
        /// Emit conversion from double to T.
        /// </summary>
        private static void EmitConvertFromDouble<T>(ILGenerator il) where T : unmanaged
        {
            if (typeof(T) == typeof(double))
                return; // Already double

            if (typeof(T) == typeof(float))
                il.Emit(OpCodes.Conv_R4);
            else if (typeof(T) == typeof(byte))
                il.Emit(OpCodes.Conv_U1);
            else if (typeof(T) == typeof(sbyte))
                il.Emit(OpCodes.Conv_I1);
            else if (typeof(T) == typeof(short))
                il.Emit(OpCodes.Conv_I2);
            else if (typeof(T) == typeof(ushort))
                il.Emit(OpCodes.Conv_U2);
            else if (typeof(T) == typeof(int))
                il.Emit(OpCodes.Conv_I4);
            else if (typeof(T) == typeof(uint))
                il.Emit(OpCodes.Conv_U4);
            else if (typeof(T) == typeof(long))
                il.Emit(OpCodes.Conv_I8);
            else if (typeof(T) == typeof(ulong))
                il.Emit(OpCodes.Conv_U8);
        }

        private static OpCode GetDivOpcode<T>() where T : unmanaged
        {
            // Use Div_Un for unsigned types
            if (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) ||
                typeof(T) == typeof(uint) || typeof(T) == typeof(ulong))
            {
                return OpCodes.Div_Un;
            }
            return OpCodes.Div;
        }

        private static void EmitLoadIndirect<T>(ILGenerator il) where T : unmanaged
        {
            // Emit the appropriate ldind instruction based on type
            if (typeof(T) == typeof(byte)) il.Emit(OpCodes.Ldind_U1);
            else if (typeof(T) == typeof(sbyte)) il.Emit(OpCodes.Ldind_I1);
            else if (typeof(T) == typeof(short)) il.Emit(OpCodes.Ldind_I2);
            else if (typeof(T) == typeof(ushort)) il.Emit(OpCodes.Ldind_U2);
            else if (typeof(T) == typeof(int)) il.Emit(OpCodes.Ldind_I4);
            else if (typeof(T) == typeof(uint)) il.Emit(OpCodes.Ldind_U4);
            else if (typeof(T) == typeof(long)) il.Emit(OpCodes.Ldind_I8);
            else if (typeof(T) == typeof(ulong)) il.Emit(OpCodes.Ldind_I8);  // Same as long
            else if (typeof(T) == typeof(float)) il.Emit(OpCodes.Ldind_R4);
            else if (typeof(T) == typeof(double)) il.Emit(OpCodes.Ldind_R8);
            else throw new NotSupportedException($"Type {typeof(T)} not supported for ldind");
        }

        private static void EmitStoreIndirect<T>(ILGenerator il) where T : unmanaged
        {
            // Emit the appropriate stind instruction based on type
            if (typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte)) il.Emit(OpCodes.Stind_I1);
            else if (typeof(T) == typeof(short) || typeof(T) == typeof(ushort)) il.Emit(OpCodes.Stind_I2);
            else if (typeof(T) == typeof(int) || typeof(T) == typeof(uint)) il.Emit(OpCodes.Stind_I4);
            else if (typeof(T) == typeof(long) || typeof(T) == typeof(ulong)) il.Emit(OpCodes.Stind_I8);
            else if (typeof(T) == typeof(float)) il.Emit(OpCodes.Stind_R4);
            else if (typeof(T) == typeof(double)) il.Emit(OpCodes.Stind_R8);
            else throw new NotSupportedException($"Type {typeof(T)} not supported for stind");
        }

        #endregion
    }
}
