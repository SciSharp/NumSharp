using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Binary operations (same-type) - contiguous kernels and generic helpers.
    /// </summary>
    public static partial class ILKernelGenerator
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

        /// <summary>
        /// Delegate for contiguous (SimdFull) operations.
        /// Simplified signature - no strides needed since both arrays are contiguous.
        /// </summary>
        public unsafe delegate void ContiguousKernel<T>(T* lhs, T* rhs, T* result, int count) where T : unmanaged;

        /// <summary>
        /// Try to generate an IL-based contiguous kernel for the given operation and type.
        /// </summary>
        private static ContiguousKernel<T>? TryGenerateContiguousKernel<T>(BinaryOp op) where T : unmanaged
        {
            // Only support types with Vector256 support
            if (!IsSimdSupported<T>())
                return null;

            // Only support basic arithmetic operations
            if (op != BinaryOp.Add && op != BinaryOp.Subtract &&
                op != BinaryOp.Multiply && op != BinaryOp.Divide)
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

            // Declare locals
            var locI = il.DeclareLocal(typeof(int));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int));   // totalSize - vectorCount

            // Define labels
            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            int vectorCount = GetVectorCount<T>();
            int elementSize = Unsafe.SizeOf<T>();

            // vectorEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_3);                      // count
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

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

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<ContiguousKernel<T>>();
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

            string methodName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException($"Operation {op} not supported for SIMD")
            };

            var opMethod = vectorType.GetMethod(methodName,
                BindingFlags.Public | BindingFlags.Static,
                null, new[] { vectorType, vectorType }, null);

            if (opMethod == null)
                throw new InvalidOperationException($"Could not find {methodName} for {vectorType.Name}");

            il.EmitCall(OpCodes.Call, opMethod, null);
        }

        private static void EmitScalarOperation<T>(ILGenerator il, BinaryOp op) where T : unmanaged
        {
            // For scalar operations, use IL opcodes
            // Stack has two T values
            var opcode = op switch
            {
                BinaryOp.Add => OpCodes.Add,
                BinaryOp.Subtract => OpCodes.Sub,
                BinaryOp.Multiply => OpCodes.Mul,
                BinaryOp.Divide => GetDivOpcode<T>(),
                _ => throw new NotSupportedException($"Operation {op} not supported")
            };

            il.Emit(opcode);
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
