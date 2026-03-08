using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
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
// ILKernelGenerator.Binary.cs (THIS FILE)
//   OWNERSHIP: Same-type binary operations on contiguous arrays (fast path)
//   RESPONSIBILITY:
//     - Optimized kernels when both operands have identical type and layout
//     - SIMD loop + scalar tail for Add, Sub, Mul, Div
//     - Dispatching kernel that routes to IL or C# based on stride detection
//   DEPENDENCIES: Uses core emit helpers from ILKernelGenerator.cs
//   FLOW: Called by DefaultEngine for same-type contiguous operations
//   KEY MEMBERS:
//     - ContiguousKernel<T> delegate - simplified signature for contiguous arrays
//     - _contiguousKernelCache - caches generated kernels by (BinaryOp, Type)
//     - GetContiguousKernel<T>() - main entry point for contiguous kernels
//     - GenerateUnifiedKernel<T>() - creates dispatcher with IL fast path
//     - TryGenerateContiguousKernelIL<T>() - IL generation with SIMD loop
//     - Generic helpers duplicated for type safety: IsSimdSupported<T>(), etc.
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
    /// Binary operations (same-type) - contiguous kernels and generic helpers.
    /// </summary>
    public sealed partial class ILKernelGenerator
    {
        /// <summary>
        /// Cache of IL-generated contiguous kernels.
        /// Key: (Operation, Type)
        /// </summary>
        private static readonly ConcurrentDictionary<(BinaryOp, Type), Delegate> _contiguousKernelCache = new();

        /// <summary>
        /// Number of IL-generated kernels in cache.
        /// </summary>
        public static int CachedCount => _contiguousKernelCache.Count;

        /// <summary>
        /// Clear the contiguous kernel cache.
        /// </summary>
        public static void Clear() => _contiguousKernelCache.Clear();

        #region Public API

        /// <summary>
        /// Get or generate an IL-based kernel for contiguous (SimdFull) operations.
        /// Returns null if IL generation is not supported for this type/operation.
        /// </summary>
        public static ContiguousKernel<T>? GetContiguousKernel<T>(BinaryOp op) where T : unmanaged
        {
            if (!Enabled)
                return null;

            var key = (op, typeof(T));

            // Check cache first
            if (_contiguousKernelCache.TryGetValue(key, out var cached))
                return (ContiguousKernel<T>)cached;

            // Generate new kernel
            var kernel = TryGenerateContiguousKernel<T>(op);
            if (kernel == null)
                return null;

            // Try to add to cache; if another thread added first, use theirs
            if (_contiguousKernelCache.TryAdd(key, kernel))
                return kernel;

            // Another thread beat us - return the cached version
            return (ContiguousKernel<T>)_contiguousKernelCache[key];
        }

        /// <summary>
        /// Generate a full unified kernel that handles all execution paths.
        /// Uses IL-generated code for hot paths, falls back to C# for others.
        /// </summary>
        public static BinaryKernel<T>? GenerateUnifiedKernel<T>(BinaryOp op) where T : unmanaged
        {
            if (!Enabled)
                return null;

            // Get the IL-generated contiguous kernel
            var contiguousKernel = GetContiguousKernel<T>(op);
            if (contiguousKernel == null)
                return null;

            // Create a wrapper that dispatches based on strides
            return CreateDispatchingKernel<T>(op, contiguousKernel);
        }

        #endregion

        #region Contiguous Kernel Generation

        // NOTE: ContiguousKernel<T> delegate is now defined in KernelSignatures.cs
        // to support IKernelProvider interface.

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
            catch
            {
                // IL generation failed - fall back to C#
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
                parameterTypes: new[] { typeof(T*), typeof(T*), typeof(T*), typeof(int) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Check if this is a scalar-only operation (no SIMD support)
            bool isScalarOnly = op == BinaryOp.Power || op == BinaryOp.FloorDivide;

            // Declare locals
            var locI = il.DeclareLocal(typeof(int));           // loop counter

            int elementSize = Unsafe.SizeOf<T>();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            if (!isScalarOnly)
            {
                // SIMD-capable operations: generate SIMD loop + tail loop
                var locVectorEnd = il.DeclareLocal(typeof(int));   // totalSize - vectorCount

                // Define labels
                var lblSimdLoop = il.DefineLabel();
                var lblSimdLoopEnd = il.DefineLabel();
                var lblTailLoop = il.DefineLabel();
                var lblTailLoopEnd = il.DefineLabel();

                int vectorCount = GetVectorCount<T>();

                // vectorEnd = count - vectorCount
                il.Emit(OpCodes.Ldarg_3);                      // count
                il.Emit(OpCodes.Ldc_I4, vectorCount);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locVectorEnd);

                // ========== SIMD LOOP ==========
                il.MarkLabel(lblSimdLoop);

                // if (i > vectorEnd) goto SimdLoopEnd
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locVectorEnd);
                il.Emit(OpCodes.Bgt, lblSimdLoopEnd);

                // Load lhs vector: Vector256.Load(lhs + i)
                il.Emit(OpCodes.Ldarg_0);                      // lhs
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorLoad<T>(il);

                // Load rhs vector: Vector256.Load(rhs + i)
                il.Emit(OpCodes.Ldarg_1);                      // rhs
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorLoad<T>(il);

                // Perform vector operation
                EmitVectorOperation<T>(il, op);

                // Store result: Vector256.Store(result, result + i)
                il.Emit(OpCodes.Ldarg_2);                      // result
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorStore<T>(il);

                // i += vectorCount
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I4, vectorCount);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);

                il.Emit(OpCodes.Br, lblSimdLoop);
                il.MarkLabel(lblSimdLoopEnd);

                // ========== TAIL LOOP ==========
                il.MarkLabel(lblTailLoop);

                // if (i >= count) goto TailLoopEnd
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_3);                      // count
                il.Emit(OpCodes.Bge, lblTailLoopEnd);

                // result[i] = lhs[i] op rhs[i]
                EmitScalarLoopBody<T>(il, op, locI, elementSize);

                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I4_1);
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
                il.Emit(OpCodes.Ldarg_3);                      // count
                il.Emit(OpCodes.Bge, lblLoopEnd);

                // result[i] = lhs[i] op rhs[i]
                EmitScalarLoopBody<T>(il, op, locI, elementSize);

                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I4_1);
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
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load lhs[i]
            il.Emit(OpCodes.Ldarg_0);                      // lhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect<T>(il);

            // Load rhs[i]
            il.Emit(OpCodes.Ldarg_1);                      // rhs
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect<T>(il);

            // Perform scalar operation
            EmitScalarOperation<T>(il, op);

            // Store to result[i]
            EmitStoreIndirect<T>(il);
        }

        #endregion

        #region Unified Kernel with Dispatch

        /// <summary>
        /// Create a unified kernel that dispatches to IL-generated code for contiguous arrays.
        /// </summary>
        private static unsafe BinaryKernel<T> CreateDispatchingKernel<T>(BinaryOp op, ContiguousKernel<T> contiguousKernel)
            where T : unmanaged
        {
            // Get the C# fallback kernel method
            var csharpKernel = GetCSharpKernel<T>(op);

            return (T* lhs, T* rhs, T* result, int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize) =>
            {
                var path = StrideDetector.Classify<T>(lhsStrides, rhsStrides, shape, ndim);

                if (path == ExecutionPath.SimdFull)
                {
                    // Use IL-generated kernel for contiguous arrays
                    contiguousKernel(lhs, rhs, result, totalSize);
                }
                else
                {
                    // Fall back to C# implementation for other paths
                    csharpKernel(lhs, rhs, result, lhsStrides, rhsStrides, shape, ndim, totalSize);
                }
            };
        }

        /// <summary>
        /// Get the C# reference implementation for fallback.
        /// </summary>
        private static unsafe BinaryKernel<T> GetCSharpKernel<T>(BinaryOp op) where T : unmanaged
        {
            // Return the appropriate C# kernel based on type and operation
            if (op == BinaryOp.Add)
            {
                if (typeof(T) == typeof(int))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<int>)SimdKernels.Add_Int32;
                if (typeof(T) == typeof(long))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<long>)SimdKernels.Add_Int64;
                if (typeof(T) == typeof(float))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<float>)SimdKernels.Add_Single;
                if (typeof(T) == typeof(double))
                    return (BinaryKernel<T>)(Delegate)(BinaryKernel<double>)SimdKernels.Add_Double;
            }

            throw new NotSupportedException($"C# kernel not available for {op} on {typeof(T).Name}");
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
        {
            var containerType = GetVectorContainerType();

            var loadMethod = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Load" && m.IsGenericMethod &&
                            m.GetParameters().Length == 1 &&
                            m.GetParameters()[0].ParameterType.IsPointer)
                .MakeGenericMethod(typeof(T));

            il.EmitCall(OpCodes.Call, loadMethod, null);
        }

        private static void EmitVectorStore<T>(ILGenerator il) where T : unmanaged
        {
            var containerType = GetVectorContainerType();
            var vectorType = GetVectorType(typeof(T));

            var storeMethod = containerType
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Store" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2 &&
                            m.GetParameters()[0].ParameterType.IsGenericType)
                .MakeGenericMethod(typeof(T));

            il.EmitCall(OpCodes.Call, storeMethod, null);
        }

        private static void EmitVectorOperation<T>(ILGenerator il, BinaryOp op) where T : unmanaged
        {
            var vectorType = GetVectorType(typeof(T));
            var containerType = GetVectorContainerType();

            // Bitwise operations use static methods on Vector256/Vector128 container
            if (op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor)
            {
                string methodName = op switch
                {
                    BinaryOp.BitwiseAnd => "BitwiseAnd",
                    BinaryOp.BitwiseOr => "BitwiseOr",
                    BinaryOp.BitwiseXor => "Xor",
                    _ => throw new NotSupportedException()
                };

                var opMethod = containerType
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == methodName && m.IsGenericMethod &&
                                m.GetParameters().Length == 2)
                    .MakeGenericMethod(typeof(T));

                il.EmitCall(OpCodes.Call, opMethod, null);
                return;
            }

            // Arithmetic operations use operator overloads on Vector256<T>/Vector128<T>
            string operatorName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException($"Operation {op} not supported for SIMD")
            };

            var operatorMethod = vectorType.GetMethod(operatorName,
                BindingFlags.Public | BindingFlags.Static,
                null, new[] { vectorType, vectorType }, null);

            if (operatorMethod == null)
                throw new InvalidOperationException($"Could not find {operatorName} for {vectorType.Name}");

            il.EmitCall(OpCodes.Call, operatorMethod, null);
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
        /// Emit Power operation using Math.Pow for generic type T.
        /// Stack: [base, exponent] -> [result]
        /// </summary>
        private static void EmitPowerOperation<T>(ILGenerator il) where T : unmanaged
        {
            // Math.Pow(double, double) -> double
            // We need to convert both operands to double, call Math.Pow, then convert back

            // Store exponent temporarily
            var locExp = il.DeclareLocal(typeof(T));
            il.Emit(OpCodes.Stloc, locExp);

            // Convert base to double
            EmitConvertToDouble<T>(il);

            // Load and convert exponent to double
            il.Emit(OpCodes.Ldloc, locExp);
            EmitConvertToDouble<T>(il);

            // Call Math.Pow(double, double)
            var powMethod = typeof(Math).GetMethod(nameof(Math.Pow), new[] { typeof(double), typeof(double) });
            il.EmitCall(OpCodes.Call, powMethod!, null);

            // Convert result back to T
            EmitConvertFromDouble<T>(il);
        }

        /// <summary>
        /// Emit FloorDivide operation for generic type T.
        /// NumPy floor_divide always floors toward negative infinity.
        /// For floats: divide then Math.Floor.
        /// For unsigned integers: regular division (same as floor for positive).
        /// For signed integers: correct floor division toward negative infinity.
        /// Stack: [dividend, divisor] -> [result]
        /// </summary>
        private static void EmitFloorDivideOperation<T>(ILGenerator il) where T : unmanaged
        {
            // For floating-point types, divide then floor
            if (typeof(T) == typeof(float))
            {
                il.Emit(OpCodes.Div);
                il.Emit(OpCodes.Conv_R8);
                var floorMethod = typeof(Math).GetMethod(nameof(Math.Floor), new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, floorMethod!, null);
                il.Emit(OpCodes.Conv_R4);
            }
            else if (typeof(T) == typeof(double))
            {
                il.Emit(OpCodes.Div);
                var floorMethod = typeof(Math).GetMethod(nameof(Math.Floor), new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, floorMethod!, null);
            }
            else if (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) ||
                     typeof(T) == typeof(uint) || typeof(T) == typeof(ulong))
            {
                // Unsigned integers: floor = regular division
                il.Emit(OpCodes.Div_Un);
            }
            else
            {
                // Signed integers: need true floor division (toward negative infinity)
                // NumPy: floor_divide(-7, 3) = -3, not -2
                // C# division truncates toward zero, so we need adjustment
                // Approach: convert to double, divide, floor, convert back
                // Stack on entry: [dividend, divisor]

                // Store divisor first (it's on top)
                var locDivisor = il.DeclareLocal(typeof(T));
                il.Emit(OpCodes.Stloc, locDivisor);

                // Convert dividend to double
                EmitConvertToDouble<T>(il);
                var locDividendDouble = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locDividendDouble);

                // Convert divisor to double
                il.Emit(OpCodes.Ldloc, locDivisor);
                EmitConvertToDouble<T>(il);

                // Load dividend and divisor as doubles
                var locDivisorDouble = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locDivisorDouble);
                il.Emit(OpCodes.Ldloc, locDividendDouble);
                il.Emit(OpCodes.Ldloc, locDivisorDouble);

                // Divide and floor
                il.Emit(OpCodes.Div);
                var floorMethod = typeof(Math).GetMethod(nameof(Math.Floor), new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, floorMethod!, null);

                // Convert back to T
                EmitConvertFromDouble<T>(il);
            }
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
