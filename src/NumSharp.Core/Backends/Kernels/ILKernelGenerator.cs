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
    /// Generates IL-based SIMD kernels using DynamicMethod.
    /// These kernels provide ~10-15% speedup over the C# reference implementations
    /// by allowing the JIT to inline Vector256 operations more aggressively.
    ///
    /// Currently implements the SimdFull execution path (both operands contiguous).
    /// Falls back to C# implementations for other paths.
    /// </summary>
    public static class ILKernelGenerator
    {
        /// <summary>
        /// Cache of IL-generated contiguous kernels.
        /// Key: (Operation, Type)
        /// </summary>
        private static readonly ConcurrentDictionary<(BinaryOp, Type), Delegate> _contiguousKernelCache = new();

        /// <summary>
        /// Whether IL generation is enabled. Can be disabled for debugging.
        /// </summary>
        public static bool Enabled { get; set; } = true;

        /// <summary>
        /// Number of IL-generated kernels in cache.
        /// </summary>
        public static int CachedCount => _contiguousKernelCache.Count;

        /// <summary>
        /// Clear the IL kernel cache.
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

        #region IL Emission Helpers

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
            return Vector256<T>.Count;
        }

        private static void EmitVectorLoad<T>(ILGenerator il) where T : unmanaged
        {
            // Call Vector256.Load<T>(T*)
            var loadMethod = typeof(Vector256).GetMethod(
                nameof(Vector256.Load),
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { typeof(T).MakePointerType() },
                null
            );

            if (loadMethod == null)
            {
                // Try the generic version
                loadMethod = typeof(Vector256)
                    .GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .First(m => m.Name == "Load" && m.IsGenericMethod && m.GetParameters().Length == 1)
                    .MakeGenericMethod(typeof(T));
            }

            il.EmitCall(OpCodes.Call, loadMethod!, null);
        }

        private static void EmitVectorStore<T>(ILGenerator il) where T : unmanaged
        {
            // Stack has: [Vector256<T>, T*]
            // Need to call Vector256.Store<T>(Vector256<T> source, T* destination)
            // But Store takes (this Vector256<T>, T*) so we need the extension method

            var storeMethod = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Store" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(typeof(T)))
                .FirstOrDefault(m =>
                {
                    var p = m.GetParameters();
                    return p.Length == 2 &&
                           p[0].ParameterType == typeof(Vector256<T>) &&
                           p[1].ParameterType == typeof(T).MakePointerType();
                });

            if (storeMethod == null)
                throw new InvalidOperationException($"Could not find Vector256.Store<{typeof(T).Name}> method");

            il.EmitCall(OpCodes.Call, storeMethod, null);
        }

        private static void EmitVectorOperation<T>(ILGenerator il, BinaryOp op) where T : unmanaged
        {
            // Stack has two Vector256<T> values, need to emit the operation
            string methodName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException($"Operation {op} not supported for SIMD")
            };

            // Look for the operator on Vector256<T>
            var vectorType = typeof(Vector256<T>);
            var opMethod = vectorType.GetMethod(
                methodName,
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { vectorType, vectorType },
                null
            );

            if (opMethod == null)
                throw new InvalidOperationException($"Could not find {methodName} for Vector256<{typeof(T).Name}>");

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
        /// Clear both kernel caches.
        /// </summary>
        public static void ClearAll()
        {
            _contiguousKernelCache.Clear();
            _mixedTypeCache.Clear();
        }

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
            catch
            {
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
            // void(void* lhs, void* rhs, void* result, int* lhsStrides, int* rhsStrides, int* shape, int ndim, int totalSize)
            var dm = new DynamicMethod(
                name: $"MixedType_SimdFull_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*), typeof(void*), typeof(void*),
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
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
            // Only Add, Subtract, Multiply, Divide have Vector256 operators
            // Mod requires scalar implementation
            return op == BinaryOp.Add || op == BinaryOp.Subtract ||
                   op == BinaryOp.Multiply || op == BinaryOp.Divide;
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
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
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
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
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
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
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
                    typeof(int*), typeof(int*), typeof(int*),
                    typeof(int), typeof(int)
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
        /// Emit a SIMD loop for contiguous same-type arrays.
        /// </summary>
        private static void EmitSimdFullLoop(ILGenerator il, MixedTypeKernelKey key,
            int lhsSize, int rhsSize, int resultSize)
        {
            // For same-type operations, use Vector256
            int vectorCount = GetVectorCount(key.ResultType);

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int)); // totalSize - vectorCount

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

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

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === TAIL LOOP ===
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
            var locI = il.DeclareLocal(typeof(int)); // loop counter
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
            var locI = il.DeclareLocal(typeof(int)); // loop counter
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
            //       int* lhsStrides (3), int* rhsStrides (4), int* shape (5),
            //       int ndim (6), int totalSize (7)

            int vectorCount = GetVectorCount(key.ResultType);
            var clrType = GetClrType(key.ResultType);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);

            var locI = il.DeclareLocal(typeof(int));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int));   // totalSize - vectorCount
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
            //       int* lhsStrides (3), int* rhsStrides (4), int* shape (5),
            //       int ndim (6), int totalSize (7)

            int vectorCount = GetVectorCount(key.ResultType);
            var clrType = GetClrType(key.ResultType);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);

            var locI = il.DeclareLocal(typeof(int));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int));   // totalSize - vectorCount
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
            // lhsOffset = 0, rhsOffset = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locLhsOffset);
            il.Emit(OpCodes.Ldc_I4_0);
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
            il.Emit(OpCodes.Ldc_I4_4); // sizeof(int)
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

        #region NPTypeCode-Based IL Helpers

        /// <summary>
        /// Get size in bytes for NPTypeCode.
        /// </summary>
        internal static int GetTypeSize(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => 1,
                NPTypeCode.Byte => 1,
                NPTypeCode.Int16 => 2,
                NPTypeCode.UInt16 => 2,
                NPTypeCode.Int32 => 4,
                NPTypeCode.UInt32 => 4,
                NPTypeCode.Int64 => 8,
                NPTypeCode.UInt64 => 8,
                NPTypeCode.Char => 2,
                NPTypeCode.Single => 4,
                NPTypeCode.Double => 8,
                NPTypeCode.Decimal => 16,
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Get CLR Type for NPTypeCode.
        /// </summary>
        internal static Type GetClrType(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Boolean => typeof(bool),
                NPTypeCode.Byte => typeof(byte),
                NPTypeCode.Int16 => typeof(short),
                NPTypeCode.UInt16 => typeof(ushort),
                NPTypeCode.Int32 => typeof(int),
                NPTypeCode.UInt32 => typeof(uint),
                NPTypeCode.Int64 => typeof(long),
                NPTypeCode.UInt64 => typeof(ulong),
                NPTypeCode.Char => typeof(char),
                NPTypeCode.Single => typeof(float),
                NPTypeCode.Double => typeof(double),
                NPTypeCode.Decimal => typeof(decimal),
                _ => throw new NotSupportedException($"Type {type} not supported")
            };
        }

        /// <summary>
        /// Check if type supports SIMD Vector256 operations.
        /// </summary>
        internal static bool CanUseSimd(NPTypeCode type)
        {
            return type switch
            {
                NPTypeCode.Byte => true,
                NPTypeCode.Int16 => true,
                NPTypeCode.UInt16 => true,
                NPTypeCode.Int32 => true,
                NPTypeCode.UInt32 => true,
                NPTypeCode.Int64 => true,
                NPTypeCode.UInt64 => true,
                NPTypeCode.Single => true,
                NPTypeCode.Double => true,
                _ => false // bool, char, decimal don't have Vector256 support
            };
        }

        /// <summary>
        /// Get Vector256 element count for type.
        /// </summary>
        internal static int GetVectorCount(NPTypeCode type)
        {
            return 32 / GetTypeSize(type); // Vector256 is 32 bytes
        }

        /// <summary>
        /// Emit load indirect for NPTypeCode.
        /// </summary>
        internal static void EmitLoadIndirect(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Ldind_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Ldind_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Ldind_U2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Ldind_I4);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Ldind_U4);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldind_I8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldind_R4);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldind_R8);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Ldobj, typeof(decimal));
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported for ldind");
            }
        }

        /// <summary>
        /// Emit store indirect for NPTypeCode.
        /// </summary>
        internal static void EmitStoreIndirect(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Stind_I1);
                    break;
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Stind_I2);
                    break;
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Stind_I4);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Stind_I8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Stind_R4);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Stind_R8);
                    break;
                case NPTypeCode.Decimal:
                    il.Emit(OpCodes.Stobj, typeof(decimal));
                    break;
                default:
                    throw new NotSupportedException($"Type {type} not supported for stind");
            }
        }

        /// <summary>
        /// Emit type conversion from source to target type.
        /// </summary>
        internal static void EmitConvertTo(ILGenerator il, NPTypeCode from, NPTypeCode to)
        {
            if (from == to)
                return; // No conversion needed

            // Special case: decimal conversions require method calls
            if (from == NPTypeCode.Decimal || to == NPTypeCode.Decimal)
            {
                EmitDecimalConversion(il, from, to);
                return;
            }

            // For numeric types, use conv.* opcodes
            switch (to)
            {
                case NPTypeCode.Boolean:
                    // Convert to bool: != 0
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    break;
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Conv_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Conv_U2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Conv_I4);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    break;
                case NPTypeCode.Int64:
                    if (IsUnsigned(from))
                        il.Emit(OpCodes.Conv_U8);
                    else
                        il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    if (IsUnsigned(from))
                        il.Emit(OpCodes.Conv_R_Un);
                    il.Emit(OpCodes.Conv_R4);
                    break;
                case NPTypeCode.Double:
                    if (IsUnsigned(from))
                        il.Emit(OpCodes.Conv_R_Un);
                    il.Emit(OpCodes.Conv_R8);
                    break;
                default:
                    throw new NotSupportedException($"Conversion to {to} not supported");
            }
        }

        /// <summary>
        /// Emit decimal-specific conversions.
        /// </summary>
        private static void EmitDecimalConversion(ILGenerator il, NPTypeCode from, NPTypeCode to)
        {
            if (to == NPTypeCode.Decimal)
            {
                // Convert to decimal - need to handle bool/char first
                if (from == NPTypeCode.Boolean)
                {
                    // bool -> int -> decimal
                    il.Emit(OpCodes.Conv_I4);
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!, null);
                    return;
                }
                if (from == NPTypeCode.Char)
                {
                    // char -> int -> decimal
                    il.Emit(OpCodes.Conv_I4);
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!, null);
                    return;
                }

                var method = from switch
                {
                    NPTypeCode.Byte => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(byte) }),
                    NPTypeCode.Int16 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(short) }),
                    NPTypeCode.UInt16 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(ushort) }),
                    NPTypeCode.Int32 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) }),
                    NPTypeCode.UInt32 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(uint) }),
                    NPTypeCode.Int64 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(long) }),
                    NPTypeCode.UInt64 => typeof(decimal).GetMethod("op_Implicit", new[] { typeof(ulong) }),
                    NPTypeCode.Single => typeof(decimal).GetMethod("op_Explicit", new[] { typeof(float) }),
                    NPTypeCode.Double => typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) }),
                    _ => throw new NotSupportedException($"Cannot convert {from} to decimal")
                };
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // Convert from decimal - need to handle bool/char
                if (to == NPTypeCode.Boolean)
                {
                    // decimal -> int -> bool (compare with 0)
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) })!, null);
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    return;
                }
                if (to == NPTypeCode.Char)
                {
                    // decimal -> int -> char
                    il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) })!, null);
                    il.Emit(OpCodes.Conv_U2);
                    return;
                }

                var method = to switch
                {
                    NPTypeCode.Byte => typeof(decimal).GetMethod("ToByte", new[] { typeof(decimal) }),
                    NPTypeCode.Int16 => typeof(decimal).GetMethod("ToInt16", new[] { typeof(decimal) }),
                    NPTypeCode.UInt16 => typeof(decimal).GetMethod("ToUInt16", new[] { typeof(decimal) }),
                    NPTypeCode.Int32 => typeof(decimal).GetMethod("ToInt32", new[] { typeof(decimal) }),
                    NPTypeCode.UInt32 => typeof(decimal).GetMethod("ToUInt32", new[] { typeof(decimal) }),
                    NPTypeCode.Int64 => typeof(decimal).GetMethod("ToInt64", new[] { typeof(decimal) }),
                    NPTypeCode.UInt64 => typeof(decimal).GetMethod("ToUInt64", new[] { typeof(decimal) }),
                    NPTypeCode.Single => typeof(decimal).GetMethod("ToSingle", new[] { typeof(decimal) }),
                    NPTypeCode.Double => typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) }),
                    _ => throw new NotSupportedException($"Cannot convert decimal to {to}")
                };
                il.EmitCall(OpCodes.Call, method!, null);
            }
        }

        /// <summary>
        /// Check if type is unsigned.
        /// </summary>
        private static bool IsUnsigned(NPTypeCode type)
        {
            return type == NPTypeCode.Byte || type == NPTypeCode.UInt16 ||
                   type == NPTypeCode.UInt32 || type == NPTypeCode.UInt64 ||
                   type == NPTypeCode.Char;
        }

        /// <summary>
        /// Emit scalar operation for NPTypeCode.
        /// </summary>
        internal static void EmitScalarOperation(ILGenerator il, BinaryOp op, NPTypeCode resultType)
        {
            // Special handling for decimal (uses operator methods)
            if (resultType == NPTypeCode.Decimal)
            {
                EmitDecimalOperation(il, op);
                return;
            }

            // Special handling for boolean
            if (resultType == NPTypeCode.Boolean)
            {
                // For bool, only meaningful ops are probably logical, but we'll support arithmetic
                // Treat as byte arithmetic
            }

            var opcode = op switch
            {
                BinaryOp.Add => OpCodes.Add,
                BinaryOp.Subtract => OpCodes.Sub,
                BinaryOp.Multiply => OpCodes.Mul,
                BinaryOp.Divide => IsUnsigned(resultType) ? OpCodes.Div_Un : OpCodes.Div,
                BinaryOp.Mod => IsUnsigned(resultType) ? OpCodes.Rem_Un : OpCodes.Rem,
                BinaryOp.BitwiseAnd => OpCodes.And,
                BinaryOp.BitwiseOr => OpCodes.Or,
                BinaryOp.BitwiseXor => OpCodes.Xor,
                _ => throw new NotSupportedException($"Operation {op} not supported")
            };

            il.Emit(opcode);
        }

        /// <summary>
        /// Emit decimal-specific operation using operator methods.
        /// </summary>
        private static void EmitDecimalOperation(ILGenerator il, BinaryOp op)
        {
            // Bitwise operations not supported for decimal
            if (op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr || op == BinaryOp.BitwiseXor)
                throw new NotSupportedException($"Bitwise operation {op} not supported for decimal type");

            var methodName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                BinaryOp.Mod => "op_Modulus",
                _ => throw new NotSupportedException($"Operation {op} not supported for decimal")
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

        /// <summary>
        /// Emit Vector256.Load for NPTypeCode.
        /// </summary>
        internal static void EmitVectorLoad(ILGenerator il, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var loadMethod = typeof(Vector256)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Load" && m.IsGenericMethod && m.GetParameters().Length == 1)
                .MakeGenericMethod(clrType);

            il.EmitCall(OpCodes.Call, loadMethod, null);
        }

        /// <summary>
        /// Emit Vector256.Create for NPTypeCode (broadcasts scalar to all vector elements).
        /// Stack must have scalar value on top; result is Vector256 on stack.
        /// </summary>
        internal static void EmitVectorCreate(ILGenerator il, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            // Vector256.Create<T>(T value) - generic method that takes single scalar
            var createMethod = typeof(Vector256)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Create" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(clrType))
                .First(m =>
                {
                    var p = m.GetParameters();
                    return p.Length == 1 && p[0].ParameterType == clrType;
                });

            il.EmitCall(OpCodes.Call, createMethod, null);
        }

        /// <summary>
        /// Emit Vector256.Store for NPTypeCode.
        /// </summary>
        internal static void EmitVectorStore(ILGenerator il, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);
            var ptrType = clrType.MakePointerType();

            var storeMethod = typeof(Vector256)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Store" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m =>
                {
                    var p = m.GetParameters();
                    return p.Length == 2 &&
                           p[0].ParameterType == vectorType &&
                           p[1].ParameterType == ptrType;
                });

            if (storeMethod == null)
                throw new InvalidOperationException($"Could not find Vector256.Store for {type}");

            il.EmitCall(OpCodes.Call, storeMethod, null);
        }

        /// <summary>
        /// Emit Vector256 operation for NPTypeCode.
        /// </summary>
        internal static void EmitVectorOperation(ILGenerator il, BinaryOp op, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);

            var methodName = op switch
            {
                BinaryOp.Add => "op_Addition",
                BinaryOp.Subtract => "op_Subtraction",
                BinaryOp.Multiply => "op_Multiply",
                BinaryOp.Divide => "op_Division",
                _ => throw new NotSupportedException($"SIMD operation {op} not supported")
            };

            var opMethod = vectorType.GetMethod(
                methodName,
                BindingFlags.Public | BindingFlags.Static,
                null,
                new[] { vectorType, vectorType },
                null
            );

            if (opMethod == null)
                throw new InvalidOperationException($"Could not find {methodName} for Vector256<{type}>");

            il.EmitCall(OpCodes.Call, opMethod, null);
        }

        #endregion

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
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Clear the unary kernel cache.
        /// </summary>
        public static void ClearUnary() => _unaryCache.Clear();

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
        /// Check if SIMD can be used for this unary operation.
        /// </summary>
        private static bool CanUseUnarySimd(UnaryKernelKey key)
        {
            // SIMD only for same-type operations on float/double
            if (!key.IsSameType)
                return false;

            // Only float and double have good SIMD support for unary ops
            if (key.InputType != NPTypeCode.Single && key.InputType != NPTypeCode.Double)
                return false;

            // Only certain operations have SIMD support
            return key.Op == UnaryOp.Negate || key.Op == UnaryOp.Abs || key.Op == UnaryOp.Sqrt;
        }

        /// <summary>
        /// Emit SIMD loop for contiguous unary operations.
        /// </summary>
        private static void EmitUnarySimdLoop(ILGenerator il, UnaryKernelKey key,
            int inputSize, int outputSize)
        {
            int vectorCount = GetVectorCount(key.InputType);

            var locI = il.DeclareLocal(typeof(int)); // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int)); // totalSize - vectorCount

            var lblSimdLoop = il.DefineLabel();
            var lblSimdLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // vectorEnd = totalSize - vectorCount
            il.Emit(OpCodes.Ldarg_S, (byte)5); // totalSize
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

            il.Emit(OpCodes.Br, lblSimdLoop);
            il.MarkLabel(lblSimdLoopEnd);

            // === TAIL LOOP ===
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
            EmitConvertTo(il, key.InputType, key.OutputType);

            EmitUnaryScalarOperation(il, key.Op, key.OutputType);
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

            // Load input[i] and convert to output type
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, inputSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.InputType);
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Perform operation
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);

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
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Operation
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);

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

        /// <summary>
        /// Emit unary scalar operation.
        /// </summary>
        internal static void EmitUnaryScalarOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            // Special handling for decimal
            if (type == NPTypeCode.Decimal)
            {
                EmitUnaryDecimalOperation(il, op);
                return;
            }

            switch (op)
            {
                case UnaryOp.Negate:
                    // For unsigned types, use two's complement: ~val + 1
                    // For signed types, use neg opcode
                    if (IsUnsigned(type))
                    {
                        // ~val + 1 = two's complement negation
                        il.Emit(OpCodes.Not);
                        il.Emit(OpCodes.Ldc_I4_1);
                        // Need to widen to correct type before add
                        if (type == NPTypeCode.UInt64)
                        {
                            il.Emit(OpCodes.Conv_U8);
                        }
                        il.Emit(OpCodes.Add);
                    }
                    else
                    {
                        il.Emit(OpCodes.Neg);
                    }
                    break;

                case UnaryOp.Abs:
                    EmitMathCall(il, "Abs", type);
                    break;

                case UnaryOp.Sqrt:
                    EmitMathCall(il, "Sqrt", type);
                    break;

                case UnaryOp.Exp:
                    EmitMathCall(il, "Exp", type);
                    break;

                case UnaryOp.Log:
                    EmitMathCall(il, "Log", type);
                    break;

                case UnaryOp.Sin:
                    EmitMathCall(il, "Sin", type);
                    break;

                case UnaryOp.Cos:
                    EmitMathCall(il, "Cos", type);
                    break;

                case UnaryOp.Tan:
                    EmitMathCall(il, "Tan", type);
                    break;

                case UnaryOp.Sinh:
                    EmitMathCall(il, "Sinh", type);
                    break;

                case UnaryOp.Cosh:
                    EmitMathCall(il, "Cosh", type);
                    break;

                case UnaryOp.Tanh:
                    EmitMathCall(il, "Tanh", type);
                    break;

                case UnaryOp.ASin:
                    EmitMathCall(il, "Asin", type);
                    break;

                case UnaryOp.ACos:
                    EmitMathCall(il, "Acos", type);
                    break;

                case UnaryOp.ATan:
                    EmitMathCall(il, "Atan", type);
                    break;

                case UnaryOp.Exp2:
                    // Use Math.Pow(2, x) since Math.Exp2 may not be available
                    EmitExp2Call(il, type);
                    break;

                case UnaryOp.Expm1:
                    // exp(x) - 1: call Exp then subtract 1
                    EmitMathCall(il, "Exp", type);
                    EmitSubtractOne(il, type);
                    break;

                case UnaryOp.Log2:
                    EmitMathCall(il, "Log2", type);
                    break;

                case UnaryOp.Log10:
                    EmitMathCall(il, "Log10", type);
                    break;

                case UnaryOp.Log1p:
                    // log(1 + x): add 1 then call Log
                    EmitAddOne(il, type);
                    EmitMathCall(il, "Log", type);
                    break;

                case UnaryOp.Sign:
                    EmitSignCall(il, type);
                    break;

                case UnaryOp.Ceil:
                    EmitMathCall(il, "Ceiling", type);
                    break;

                case UnaryOp.Floor:
                    EmitMathCall(il, "Floor", type);
                    break;

                case UnaryOp.Round:
                    EmitMathCall(il, "Round", type);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported");
            }
        }

        /// <summary>
        /// Emit call to Math.X method with appropriate overload.
        /// </summary>
        private static void EmitMathCall(ILGenerator il, string methodName, NPTypeCode type)
        {
            MethodInfo? method;

            if (type == NPTypeCode.Single)
            {
                // Use MathF for float
                method = typeof(MathF).GetMethod(methodName, new[] { typeof(float) });
            }
            else if (type == NPTypeCode.Double)
            {
                // Use Math for double
                method = typeof(Math).GetMethod(methodName, new[] { typeof(double) });
            }
            else
            {
                // For integer types, convert to double, call Math, convert back
                // Stack has: value (as output type)
                // Need to: conv to double, call Math.X, conv back

                // Convert to double first
                EmitConvertToDouble(il, type);

                // Call Math.X(double)
                method = typeof(Math).GetMethod(methodName, new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);

                // Convert back to target type
                EmitConvertFromDouble(il, type);
                return;
            }

            il.EmitCall(OpCodes.Call, method!, null);
        }

        /// <summary>
        /// Convert stack value to double.
        /// </summary>
        private static void EmitConvertToDouble(ILGenerator il, NPTypeCode from)
        {
            if (from == NPTypeCode.Double)
                return;

            if (IsUnsigned(from))
                il.Emit(OpCodes.Conv_R_Un);
            il.Emit(OpCodes.Conv_R8);
        }

        /// <summary>
        /// Convert double on stack to target type.
        /// </summary>
        private static void EmitConvertFromDouble(ILGenerator il, NPTypeCode to)
        {
            if (to == NPTypeCode.Double)
                return;

            switch (to)
            {
                case NPTypeCode.Boolean:
                    il.Emit(OpCodes.Ldc_R8, 0.0);
                    il.Emit(OpCodes.Cgt_Un);
                    break;
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Conv_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Conv_U2);
                    break;
                case NPTypeCode.Int32:
                    il.Emit(OpCodes.Conv_I4);
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    break;
                case NPTypeCode.Int64:
                    il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Conv_R4);
                    break;
                default:
                    throw new NotSupportedException($"Conversion to {to} not supported");
            }
        }

        /// <summary>
        /// Emit 2^x calculation using Math.Pow(2, x).
        /// </summary>
        private static void EmitExp2Call(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // For float: convert to double, call Pow, convert back
                il.Emit(OpCodes.Conv_R8);
                il.Emit(OpCodes.Ldc_R8, 2.0);
                // Stack: [exponent, base] - but Pow expects (base, exponent)
                // Need to swap them
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                // Now push base then exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!, null);
                il.Emit(OpCodes.Conv_R4);
            }
            else if (type == NPTypeCode.Double)
            {
                // For double: just call Pow
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!, null);
            }
            else
            {
                // For integer types: convert to double, call Pow, convert back
                EmitConvertToDouble(il, type);
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!, null);
                EmitConvertFromDouble(il, type);
            }
        }

        /// <summary>
        /// Emit subtraction of 1 from the value on stack.
        /// Used for expm1 = exp(x) - 1.
        /// </summary>
        private static void EmitSubtractOne(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, 1.0f);
                    il.Emit(OpCodes.Sub);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    break;
                default:
                    // For integer types, value is already double from math call
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    break;
            }
        }

        /// <summary>
        /// Emit addition of 1 to the value on stack.
        /// Used for log1p = log(1 + x).
        /// </summary>
        private static void EmitAddOne(ILGenerator il, NPTypeCode type)
        {
            // Convert to appropriate float type first, then add 1
            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, 1.0f);
                il.Emit(OpCodes.Add);
            }
            else if (type == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Add);
            }
            else
            {
                // For integer types, convert to double first, then add 1
                // The conversion to double will happen in EmitMathCall
                EmitConvertToDouble(il, type);
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Add);
            }
        }

        /// <summary>
        /// Emit Math.Sign call with proper type conversion.
        /// Math.Sign returns int, so we need to convert back to target type.
        /// NumPy: sign(NaN) returns NaN, but .NET Math.Sign throws ArithmeticException.
        /// We check for NaN first and return it directly.
        /// </summary>
        private static void EmitSignCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // NumPy: sign(NaN) = NaN. .NET MathF.Sign(NaN) throws.
                // Check for NaN first: if (float.IsNaN(x)) return x; else return MathF.Sign(x);
                var lblNotNaN = il.DefineLabel();
                var lblEnd = il.DefineLabel();

                il.Emit(OpCodes.Dup);  // duplicate for NaN check
                il.EmitCall(OpCodes.Call, typeof(float).GetMethod("IsNaN", new[] { typeof(float) })!, null);
                il.Emit(OpCodes.Brfalse, lblNotNaN);

                // Is NaN - value is already on stack, jump to end
                il.Emit(OpCodes.Br, lblEnd);

                il.MarkLabel(lblNotNaN);
                // Not NaN - call MathF.Sign
                var method = typeof(MathF).GetMethod("Sign", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
                il.Emit(OpCodes.Conv_R4);

                il.MarkLabel(lblEnd);
            }
            else if (type == NPTypeCode.Double)
            {
                // NumPy: sign(NaN) = NaN. .NET Math.Sign(NaN) throws.
                // Check for NaN first: if (double.IsNaN(x)) return x; else return Math.Sign(x);
                var lblNotNaN = il.DefineLabel();
                var lblEnd = il.DefineLabel();

                il.Emit(OpCodes.Dup);  // duplicate for NaN check
                il.EmitCall(OpCodes.Call, typeof(double).GetMethod("IsNaN", new[] { typeof(double) })!, null);
                il.Emit(OpCodes.Brfalse, lblNotNaN);

                // Is NaN - value is already on stack, jump to end
                il.Emit(OpCodes.Br, lblEnd);

                il.MarkLabel(lblNotNaN);
                // Not NaN - call Math.Sign
                var method = typeof(Math).GetMethod("Sign", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
                il.Emit(OpCodes.Conv_R8);

                il.MarkLabel(lblEnd);
            }
            else if (type == NPTypeCode.Decimal)
            {
                // Decimal has its own Sign method that returns int
                var method = typeof(Math).GetMethod("Sign", new[] { typeof(decimal) });
                il.EmitCall(OpCodes.Call, method!, null);
                // Convert int to decimal
                il.EmitCall(OpCodes.Call, typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!, null);
            }
            else
            {
                // For integer types: convert to double, call Math.Sign, convert back
                EmitConvertToDouble(il, type);
                var method = typeof(Math).GetMethod("Sign", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
                // Convert int result back to target type
                EmitConvertFromInt(il, type);
            }
        }

        /// <summary>
        /// Convert int on stack to target type.
        /// </summary>
        private static void EmitConvertFromInt(ILGenerator il, NPTypeCode to)
        {
            switch (to)
            {
                case NPTypeCode.Boolean:
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    break;
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Conv_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Conv_U2);
                    break;
                case NPTypeCode.Int32:
                    // Already int, no conversion needed
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    break;
                case NPTypeCode.Int64:
                    il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Conv_R4);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Conv_R8);
                    break;
                default:
                    throw new NotSupportedException($"Conversion from int to {to} not supported");
            }
        }

        /// <summary>
        /// Emit unary operation for decimal type.
        /// </summary>
        private static void EmitUnaryDecimalOperation(ILGenerator il, UnaryOp op)
        {
            switch (op)
            {
                case UnaryOp.Negate:
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_UnaryNegation", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Abs:
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Abs", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Sign:
                    // Math.Sign(decimal) returns int, convert back to decimal
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Sign", new[] { typeof(decimal) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Implicit", new[] { typeof(int) })!,
                        null);
                    break;

                case UnaryOp.Ceil:
                    // Math.Ceiling has decimal overload
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Ceiling", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Floor:
                    // Math.Floor has decimal overload
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Floor", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Round:
                    // Math.Round has decimal overload
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Round", new[] { typeof(decimal) })!,
                        null);
                    break;

                case UnaryOp.Sqrt:
                case UnaryOp.Exp:
                case UnaryOp.Log:
                case UnaryOp.Sin:
                case UnaryOp.Cos:
                case UnaryOp.Tan:
                case UnaryOp.Sinh:
                case UnaryOp.Cosh:
                case UnaryOp.Tanh:
                case UnaryOp.ASin:
                case UnaryOp.ACos:
                case UnaryOp.ATan:
                case UnaryOp.Log2:
                case UnaryOp.Log10:
                    // Convert to double, perform operation, convert back
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);

                    string mathMethod = op switch
                    {
                        UnaryOp.Sqrt => "Sqrt",
                        UnaryOp.Exp => "Exp",
                        UnaryOp.Log => "Log",
                        UnaryOp.Sin => "Sin",
                        UnaryOp.Cos => "Cos",
                        UnaryOp.Tan => "Tan",
                        UnaryOp.Sinh => "Sinh",
                        UnaryOp.Cosh => "Cosh",
                        UnaryOp.Tanh => "Tanh",
                        UnaryOp.ASin => "Asin",
                        UnaryOp.ACos => "Acos",
                        UnaryOp.ATan => "Atan",
                        UnaryOp.Log2 => "Log2",
                        UnaryOp.Log10 => "Log10",
                        _ => throw new NotSupportedException()
                    };

                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod(mathMethod, new[] { typeof(double) })!,
                        null);

                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                case UnaryOp.Exp2:
                    // 2^x for decimal: convert to double, use Math.Pow, convert back
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);
                    // Stack: [exponent (double)] - need to call Pow(2, exponent)
                    var locExpDec = il.DeclareLocal(typeof(double));
                    il.Emit(OpCodes.Stloc, locExpDec);
                    il.Emit(OpCodes.Ldc_R8, 2.0);
                    il.Emit(OpCodes.Ldloc, locExpDec);
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Pow", new[] { typeof(double), typeof(double) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                case UnaryOp.Expm1:
                    // exp(x) - 1 for decimal
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Exp", new[] { typeof(double) })!,
                        null);
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                case UnaryOp.Log1p:
                    // log(1 + x) for decimal
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("ToDouble", new[] { typeof(decimal) })!,
                        null);
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Add);
                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod("Log", new[] { typeof(double) })!,
                        null);
                    il.EmitCall(OpCodes.Call,
                        typeof(decimal).GetMethod("op_Explicit", new[] { typeof(double) })!,
                        null);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported for decimal");
            }
        }

        /// <summary>
        /// Emit Vector256 unary operation.
        /// </summary>
        private static void EmitUnaryVectorOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);

            string methodName = op switch
            {
                UnaryOp.Negate => "op_UnaryNegation",
                UnaryOp.Abs => "Abs",
                UnaryOp.Sqrt => "Sqrt",
                _ => throw new NotSupportedException($"SIMD operation {op} not supported")
            };

            MethodInfo? method;

            if (op == UnaryOp.Negate)
            {
                // Negation is an operator on Vector256<T>
                method = vectorType.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static,
                    null, new[] { vectorType }, null);
            }
            else
            {
                // Abs and Sqrt are static methods on Vector256
                method = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == methodName && m.IsGenericMethod && m.GetParameters().Length == 1)
                    .Select(m => m.MakeGenericMethod(clrType))
                    .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType);
            }

            if (method == null)
                throw new InvalidOperationException($"Could not find {methodName} for Vector256<{type}>");

            il.EmitCall(OpCodes.Call, method, null);
        }

        #endregion

        #region Scalar Kernel Generation

        /// <summary>
        /// Cache for unary scalar kernels.
        /// Key: UnaryScalarKernelKey (InputType, OutputType, Op)
        /// Value: Delegate (Func&lt;TInput, TOutput&gt;)
        /// </summary>
        private static readonly ConcurrentDictionary<UnaryScalarKernelKey, Delegate> _unaryScalarCache = new();

        /// <summary>
        /// Cache for binary scalar kernels.
        /// Key: BinaryScalarKernelKey (LhsType, RhsType, ResultType, Op)
        /// Value: Delegate (Func&lt;TLhs, TRhs, TResult&gt;)
        /// </summary>
        private static readonly ConcurrentDictionary<BinaryScalarKernelKey, Delegate> _binaryScalarCache = new();

        /// <summary>
        /// Number of unary scalar kernels in cache.
        /// </summary>
        public static int UnaryScalarCachedCount => _unaryScalarCache.Count;

        /// <summary>
        /// Number of binary scalar kernels in cache.
        /// </summary>
        public static int BinaryScalarCachedCount => _binaryScalarCache.Count;

        /// <summary>
        /// Clear the scalar kernel caches.
        /// </summary>
        public static void ClearScalar()
        {
            _unaryScalarCache.Clear();
            _binaryScalarCache.Clear();
        }

        /// <summary>
        /// Get or generate an IL-based unary scalar delegate.
        /// Returns a Func&lt;TInput, TOutput&gt; delegate.
        /// </summary>
        public static Delegate GetUnaryScalarDelegate(UnaryScalarKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _unaryScalarCache.GetOrAdd(key, GenerateUnaryScalarDelegate);
        }

        /// <summary>
        /// Get or generate an IL-based binary scalar delegate.
        /// Returns a Func&lt;TLhs, TRhs, TResult&gt; delegate.
        /// </summary>
        public static Delegate GetBinaryScalarDelegate(BinaryScalarKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _binaryScalarCache.GetOrAdd(key, GenerateBinaryScalarDelegate);
        }

        /// <summary>
        /// Generate an IL-based unary scalar delegate.
        /// Creates a Func&lt;TInput, TOutput&gt; that performs the operation.
        /// </summary>
        private static Delegate GenerateUnaryScalarDelegate(UnaryScalarKernelKey key)
        {
            var inputClr = GetClrType(key.InputType);
            var outputClr = GetClrType(key.OutputType);

            // Create DynamicMethod: TOutput Method(TInput input)
            var dm = new DynamicMethod(
                name: $"ScalarUnary_{key}",
                returnType: outputClr,
                parameterTypes: new[] { inputClr },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Load input argument
            il.Emit(OpCodes.Ldarg_0);

            // Convert to output type if different
            EmitConvertTo(il, key.InputType, key.OutputType);

            // Perform the unary operation (result is on stack)
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);

            // Return
            il.Emit(OpCodes.Ret);

            // Create typed Func<TInput, TOutput>
            var funcType = typeof(Func<,>).MakeGenericType(inputClr, outputClr);
            return dm.CreateDelegate(funcType);
        }

        /// <summary>
        /// Generate an IL-based binary scalar delegate.
        /// Creates a Func&lt;TLhs, TRhs, TResult&gt; that performs the operation.
        /// </summary>
        private static Delegate GenerateBinaryScalarDelegate(BinaryScalarKernelKey key)
        {
            var lhsClr = GetClrType(key.LhsType);
            var rhsClr = GetClrType(key.RhsType);
            var resultClr = GetClrType(key.ResultType);

            // Create DynamicMethod: TResult Method(TLhs lhs, TRhs rhs)
            var dm = new DynamicMethod(
                name: $"ScalarBinary_{key}",
                returnType: resultClr,
                parameterTypes: new[] { lhsClr, rhsClr },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Load lhs, convert to result type
            il.Emit(OpCodes.Ldarg_0);
            EmitConvertTo(il, key.LhsType, key.ResultType);

            // Load rhs, convert to result type
            il.Emit(OpCodes.Ldarg_1);
            EmitConvertTo(il, key.RhsType, key.ResultType);

            // Perform binary operation
            EmitScalarOperation(il, key.Op, key.ResultType);

            // Return
            il.Emit(OpCodes.Ret);

            // Create typed Func<TLhs, TRhs, TResult>
            var funcType = typeof(Func<,,>).MakeGenericType(lhsClr, rhsClr, resultClr);
            return dm.CreateDelegate(funcType);
        }

        #endregion

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
        /// Clear the comparison kernel cache.
        /// </summary>
        public static void ClearComparison() => _comparisonCache.Clear();

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
            catch
            {
                return null;
            }
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

            EmitComparisonScalarLoop(il, key, lhsSize, rhsSize, comparisonType);

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

        /// <summary>
        /// Cache key for comparison scalar operation kernels.
        /// </summary>
        public readonly record struct ComparisonScalarKernelKey(
            NPTypeCode LhsType,
            NPTypeCode RhsType,
            ComparisonOp Op
        )
        {
            public NPTypeCode ComparisonType => np._FindCommonScalarType(LhsType, RhsType);
            public override string ToString() => $"ScalarCmp_{Op}_{LhsType}_{RhsType}";
        }

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
            // Sum: Vector256<T>.Sum() or manual horizontal add
            // Max/Min: Reduce vector then scalar reduce remainder
            // Prod: No SIMD (no horizontal multiply)
            // ArgMax/ArgMin: Need to track indices, more complex
            return key.Op == ReductionOp.Sum || key.Op == ReductionOp.Max || key.Op == ReductionOp.Min;
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
        /// Emit horizontal reduction of a Vector256.
        /// Stack has Vector256, result is scalar reduction.
        /// </summary>
        private static void EmitVectorHorizontalReduction(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);

            switch (op)
            {
                case ReductionOp.Sum:
                    // Use Vector256.Sum<T>()
                    var sumMethod = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
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

                default:
                    throw new NotSupportedException($"SIMD horizontal reduction for {op} not supported");
            }
        }

        /// <summary>
        /// Emit manual horizontal sum using GetElement.
        /// </summary>
        private static void EmitManualHorizontalSum(ILGenerator il, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);
            int count = GetVectorCount(type);

            // Store vector in local
            var locVec = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locVec);

            // Load first element
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldc_I4_0);
            var getElementMethod = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "GetElement" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(clrType))
                .First();
            il.EmitCall(OpCodes.Call, getElementMethod, null);

            // Add remaining elements
            for (int i = 1; i < count; i++)
            {
                il.Emit(OpCodes.Ldloc, locVec);
                il.Emit(OpCodes.Ldc_I4, i);
                il.EmitCall(OpCodes.Call, getElementMethod, null);
                il.Emit(OpCodes.Add);
            }
        }

        /// <summary>
        /// Emit manual horizontal min/max using GetElement.
        /// </summary>
        private static void EmitManualHorizontalMinMax(ILGenerator il, ReductionOp op, NPTypeCode type)
        {
            var clrType = GetClrType(type);
            var vectorType = typeof(Vector256<>).MakeGenericType(clrType);
            int count = GetVectorCount(type);

            // Store vector in local
            var locVec = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locVec);

            // Load first element as initial accumulator
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldc_I4_0);
            var getElementMethod = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "GetElement" && m.IsGenericMethod)
                .Select(m => m.MakeGenericMethod(clrType))
                .First();
            il.EmitCall(OpCodes.Call, getElementMethod, null);

            // Compare with remaining elements using Math.Max/Math.Min
            var mathMethod = GetMathMinMaxMethod(op, clrType);

            for (int i = 1; i < count; i++)
            {
                il.Emit(OpCodes.Ldloc, locVec);
                il.Emit(OpCodes.Ldc_I4, i);
                il.EmitCall(OpCodes.Call, getElementMethod, null);

                if (mathMethod != null)
                {
                    il.EmitCall(OpCodes.Call, mathMethod, null);
                }
                else
                {
                    // Fallback for types without Math.Max/Min (use comparison)
                    EmitScalarMinMax(il, op, type);
                }
            }
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
