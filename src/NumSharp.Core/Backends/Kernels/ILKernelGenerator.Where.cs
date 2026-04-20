using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// ILKernelGenerator.Where - IL-generated np.where(condition, x, y) kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - Generate optimized kernels for conditional selection
//   - result[i] = cond[i] ? x[i] : y[i]
//
// ARCHITECTURE:
//   Uses IL emission to generate type-specific kernels at runtime.
//   The challenge is bool mask expansion: condition is bool[] (1 byte per element),
//   but x/y can be any dtype (1-8 bytes per element).
//
//   | Element Size | V256 Elements | Bools to Load |
//   |--------------|---------------|---------------|
//   | 1 byte       | 32            | 32            |
//   | 2 bytes      | 16            | 16            |
//   | 4 bytes      | 8             | 8             |
//   | 8 bytes      | 4             | 4             |
//
// KERNEL TYPES:
//   - WhereKernel<T>: Main kernel delegate (cond*, x*, y*, result*, count)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Delegate for where operation kernels.
    /// </summary>
    public unsafe delegate void WhereKernel<T>(bool* cond, T* x, T* y, T* result, long count) where T : unmanaged;

    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// Cache of IL-generated where kernels.
        /// Key: Type
        /// </summary>
        private static readonly ConcurrentDictionary<Type, Delegate> _whereKernelCache = new();

        #region Public API

        /// <summary>
        /// Get or generate an IL-based where kernel for the specified type.
        /// Returns null if IL generation is disabled or fails.
        /// </summary>
        public static WhereKernel<T>? GetWhereKernel<T>() where T : unmanaged
        {
            if (!Enabled)
                return null;

            var type = typeof(T);

            if (_whereKernelCache.TryGetValue(type, out var cached))
                return (WhereKernel<T>)cached;

            var kernel = TryGenerateWhereKernel<T>();
            if (kernel == null)
                return null;

            if (_whereKernelCache.TryAdd(type, kernel))
                return kernel;

            return (WhereKernel<T>)_whereKernelCache[type];
        }

        /// <summary>
        /// Execute where operation using IL-generated kernel or fallback to static helper.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void WhereExecute<T>(bool* cond, T* x, T* y, T* result, long count) where T : unmanaged
        {
            if (count == 0)
                return;

            var kernel = GetWhereKernel<T>();
            if (kernel != null)
            {
                kernel(cond, x, y, result, count);
            }
            else
            {
                // Fallback to scalar loop
                WhereScalar(cond, x, y, result, count);
            }
        }

        #endregion

        #region Kernel Generation

        private static WhereKernel<T>? TryGenerateWhereKernel<T>() where T : unmanaged
        {
            try
            {
                return GenerateWhereKernelIL<T>();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGenerateWhereKernel<{typeof(T).Name}>: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static unsafe WhereKernel<T> GenerateWhereKernelIL<T>() where T : unmanaged
        {
            int elementSize = Unsafe.SizeOf<T>();

            // Determine if we can use SIMD
            bool canSimd = elementSize <= 8 && IsSimdSupported<T>();

            var dm = new DynamicMethod(
                name: $"IL_Where_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(T*), typeof(T*), typeof(T*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Locals
            var locI = il.DeclareLocal(typeof(long));  // loop counter

            // Labels
            var lblScalarLoop = il.DefineLabel();
            var lblScalarLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (canSimd && VectorBits >= 128)
            {
                // Generate SIMD path
                EmitWhereSIMDLoop<T>(il, locI);
            }

            // Scalar loop for remainder
            il.MarkLabel(lblScalarLoop);

            // if (i >= count) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg, 4);  // count
            il.Emit(OpCodes.Bge, lblScalarLoopEnd);

            // result[i] = cond[i] ? x[i] : y[i]
            EmitWhereScalarElement<T>(il, locI);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblScalarLoop);

            il.MarkLabel(lblScalarLoopEnd);
            il.Emit(OpCodes.Ret);

            return (WhereKernel<T>)dm.CreateDelegate(typeof(WhereKernel<T>));
        }

        private static void EmitWhereSIMDLoop<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            long vectorCount = VectorBits >= 256 ? (32 / elementSize) : (16 / elementSize);
            long unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;
            bool useV256 = VectorBits >= 256;

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollLoopEnd = il.DefineLabel();
            var lblVectorLoop = il.DefineLabel();
            var lblVectorLoopEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep (for 4x unrolled loop)
            il.Emit(OpCodes.Ldarg, 4);  // count
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = count - vectorCount (for remainder loop)
            il.Emit(OpCodes.Ldarg, 4);  // count
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // ========== 4x UNROLLED SIMD LOOP ==========
            il.MarkLabel(lblUnrollLoop);

            // if (i > unrollEnd) goto UnrollLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollLoopEnd);

            // Process 4 vectors per iteration
            for (long u = 0; u < unrollFactor; u++)
            {
                long offset = vectorCount * u;
                if (useV256)
                    EmitWhereV256BodyWithOffset<T>(il, locI, elementSize, offset);
                else
                    EmitWhereV128BodyWithOffset<T>(il, locI, elementSize, offset);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblUnrollLoop);

            il.MarkLabel(lblUnrollLoopEnd);

            // ========== REMAINDER SIMD LOOP (1 vector at a time) ==========
            il.MarkLabel(lblVectorLoop);

            // if (i > vectorEnd) goto VectorLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblVectorLoopEnd);

            // Process 1 vector
            if (useV256)
                EmitWhereV256BodyWithOffset<T>(il, locI, elementSize, 0L);
            else
                EmitWhereV128BodyWithOffset<T>(il, locI, elementSize, 0L);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblVectorLoop);

            il.MarkLabel(lblVectorLoopEnd);
        }

        private static void EmitWhereV256BodyWithOffset<T>(ILGenerator il, LocalBuilder locI, long elementSize, long offset) where T : unmanaged
        {
            // Get Vector256 methods via reflection - need to find generic method definitions first
            var loadMethod = Array.Find(typeof(Vector256).GetMethods(),
                m => m.Name == "Load" && m.IsGenericMethodDefinition && m.GetParameters().Length == 1)!
                .MakeGenericMethod(typeof(T));
            var storeMethod = Array.Find(typeof(Vector256).GetMethods(),
                m => m.Name == "Store" && m.IsGenericMethodDefinition && m.GetParameters().Length == 2)!
                .MakeGenericMethod(typeof(T));
            var selectMethod = Array.Find(typeof(Vector256).GetMethods(),
                m => m.Name == "ConditionalSelect" && m.IsGenericMethodDefinition)!
                .MakeGenericMethod(typeof(T));

            // Load address: cond + (i + offset)
            il.Emit(OpCodes.Ldarg_0);  // cond
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Inline mask creation - emit AVX2 instructions directly instead of calling helper
            EmitInlineMaskCreationV256(il, (int)elementSize);

            // Load x vector: x + (i + offset) * elementSize
            il.Emit(OpCodes.Ldarg_1);  // x
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // Load y vector: y + (i + offset) * elementSize
            il.Emit(OpCodes.Ldarg_2);  // y
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // Stack: mask, xVec, yVec
            // ConditionalSelect(mask, x, y)
            il.Emit(OpCodes.Call, selectMethod);

            // Store result: result + (i + offset) * elementSize
            il.Emit(OpCodes.Ldarg_3);  // result
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, storeMethod);
        }

        private static void EmitWhereV128BodyWithOffset<T>(ILGenerator il, LocalBuilder locI, long elementSize, long offset) where T : unmanaged
        {
            // Get Vector128 methods via reflection - need to find generic method definitions first
            var loadMethod = Array.Find(typeof(Vector128).GetMethods(),
                m => m.Name == "Load" && m.IsGenericMethodDefinition && m.GetParameters().Length == 1)!
                .MakeGenericMethod(typeof(T));
            var storeMethod = Array.Find(typeof(Vector128).GetMethods(),
                m => m.Name == "Store" && m.IsGenericMethodDefinition && m.GetParameters().Length == 2)!
                .MakeGenericMethod(typeof(T));
            var selectMethod = Array.Find(typeof(Vector128).GetMethods(),
                m => m.Name == "ConditionalSelect" && m.IsGenericMethodDefinition)!
                .MakeGenericMethod(typeof(T));

            // Load address: cond + (i + offset)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Inline mask creation - emit SSE4.1 instructions directly
            EmitInlineMaskCreationV128(il, (int)elementSize);

            // Load x vector
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // Load y vector
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // ConditionalSelect
            il.Emit(OpCodes.Call, selectMethod);

            // Store
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, storeMethod);
        }

        private static void EmitWhereScalarElement<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            var typeCode = InfoOf<T>.NPTypeCode;

            // result[i] = cond[i] ? x[i] : y[i]
            var lblFalse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // Load result address: result + i * elementSize
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load cond[i]: cond + i (bool is 1 byte)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);  // Load bool as byte

            // if (!cond[i]) goto lblFalse
            il.Emit(OpCodes.Brfalse, lblFalse);

            // True branch: load x[i]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, typeCode);
            il.Emit(OpCodes.Br, lblEnd);

            // False branch: load y[i]
            il.MarkLabel(lblFalse);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, typeCode);

            il.MarkLabel(lblEnd);
            // Stack: result_ptr, value
            EmitStoreIndirect(il, typeCode);
        }

        #endregion

        #region Inline Mask IL Emission

        // Cache reflection lookups for inline emission
        private static readonly MethodInfo _v128LoadByte = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "Load" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(byte));
        private static readonly MethodInfo _v256LoadByte = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "Load" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(byte));

        private static readonly MethodInfo _v128CreateScalarUInt = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "CreateScalar" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(uint));
        private static readonly MethodInfo _v128CreateScalarULong = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "CreateScalar" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ulong));
        private static readonly MethodInfo _v128CreateScalarUShort = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "CreateScalar" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ushort));

        // AsByte is an extension method on Vector128 static class, not instance method
        private static readonly MethodInfo _v128UIntAsByte = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "AsByte" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(uint));
        private static readonly MethodInfo _v128ULongAsByte = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "AsByte" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ulong));
        private static readonly MethodInfo _v128UShortAsByte = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "AsByte" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ushort));

        private static readonly MethodInfo _avx2ConvertToV256Int64 = typeof(Avx2).GetMethod("ConvertToVector256Int64", new[] { typeof(Vector128<byte>) })!;
        private static readonly MethodInfo _avx2ConvertToV256Int32 = typeof(Avx2).GetMethod("ConvertToVector256Int32", new[] { typeof(Vector128<byte>) })!;
        private static readonly MethodInfo _avx2ConvertToV256Int16 = typeof(Avx2).GetMethod("ConvertToVector256Int16", new[] { typeof(Vector128<byte>) })!;

        private static readonly MethodInfo _sse41ConvertToV128Int64 = typeof(Sse41).GetMethod("ConvertToVector128Int64", new[] { typeof(Vector128<byte>) })!;
        private static readonly MethodInfo _sse41ConvertToV128Int32 = typeof(Sse41).GetMethod("ConvertToVector128Int32", new[] { typeof(Vector128<byte>) })!;
        private static readonly MethodInfo _sse41ConvertToV128Int16 = typeof(Sse41).GetMethod("ConvertToVector128Int16", new[] { typeof(Vector128<byte>) })!;

        // As* methods are extension methods on Vector256/Vector128 static classes
        private static readonly MethodInfo _v256LongAsULong = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "AsUInt64" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(long));
        private static readonly MethodInfo _v256IntAsUInt = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "AsUInt32" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(int));
        private static readonly MethodInfo _v256ShortAsUShort = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "AsUInt16" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(short));

        private static readonly MethodInfo _v128LongAsULong = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "AsUInt64" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(long));
        private static readonly MethodInfo _v128IntAsUInt = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "AsUInt32" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(int));
        private static readonly MethodInfo _v128ShortAsUShort = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "AsUInt16" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(short));

        private static readonly MethodInfo _v256GreaterThanULong = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ulong));
        private static readonly MethodInfo _v256GreaterThanUInt = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(uint));
        private static readonly MethodInfo _v256GreaterThanUShort = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ushort));
        private static readonly MethodInfo _v256GreaterThanByte = Array.Find(typeof(Vector256).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(byte));

        private static readonly MethodInfo _v128GreaterThanULong = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ulong));
        private static readonly MethodInfo _v128GreaterThanUInt = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(uint));
        private static readonly MethodInfo _v128GreaterThanUShort = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(ushort));
        private static readonly MethodInfo _v128GreaterThanByte = Array.Find(typeof(Vector128).GetMethods(),
            m => m.Name == "GreaterThan" && m.IsGenericMethodDefinition)!.MakeGenericMethod(typeof(byte));

        private static readonly MethodInfo _v256GetZeroULong = typeof(Vector256<ulong>).GetProperty("Zero")!.GetMethod!;
        private static readonly MethodInfo _v256GetZeroUInt = typeof(Vector256<uint>).GetProperty("Zero")!.GetMethod!;
        private static readonly MethodInfo _v256GetZeroUShort = typeof(Vector256<ushort>).GetProperty("Zero")!.GetMethod!;
        private static readonly MethodInfo _v256GetZeroByte = typeof(Vector256<byte>).GetProperty("Zero")!.GetMethod!;

        private static readonly MethodInfo _v128GetZeroULong = typeof(Vector128<ulong>).GetProperty("Zero")!.GetMethod!;
        private static readonly MethodInfo _v128GetZeroUInt = typeof(Vector128<uint>).GetProperty("Zero")!.GetMethod!;
        private static readonly MethodInfo _v128GetZeroUShort = typeof(Vector128<ushort>).GetProperty("Zero")!.GetMethod!;
        private static readonly MethodInfo _v128GetZeroByte = typeof(Vector128<byte>).GetProperty("Zero")!.GetMethod!;

        /// <summary>
        /// Emit inline V256 mask creation. Stack: byte* -> Vector256{T} (as mask)
        /// </summary>
        private static void EmitInlineMaskCreationV256(ILGenerator il, int elementSize)
        {
            // Stack has: byte* pointing to condition bools

            switch (elementSize)
            {
                case 8: // double/long: load 4 bytes, expand to 4 qwords
                    // *(uint*)ptr
                    il.Emit(OpCodes.Ldind_U4);
                    // Vector128.CreateScalar<uint>(value)
                    il.Emit(OpCodes.Call, _v128CreateScalarUInt);
                    // .AsByte()
                    il.Emit(OpCodes.Call, _v128UIntAsByte);
                    // Avx2.ConvertToVector256Int64(bytes)
                    il.Emit(OpCodes.Call, _avx2ConvertToV256Int64);
                    // .AsUInt64()
                    il.Emit(OpCodes.Call, _v256LongAsULong);
                    // Vector256<ulong>.Zero
                    il.Emit(OpCodes.Call, _v256GetZeroULong);
                    // Vector256.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, _v256GreaterThanULong);
                    break;

                case 4: // float/int: load 8 bytes, expand to 8 dwords
                    // *(ulong*)ptr
                    il.Emit(OpCodes.Ldind_I8);
                    // Vector128.CreateScalar<ulong>(value)
                    il.Emit(OpCodes.Call, _v128CreateScalarULong);
                    // .AsByte()
                    il.Emit(OpCodes.Call, _v128ULongAsByte);
                    // Avx2.ConvertToVector256Int32(bytes)
                    il.Emit(OpCodes.Call, _avx2ConvertToV256Int32);
                    // .AsUInt32()
                    il.Emit(OpCodes.Call, _v256IntAsUInt);
                    // Vector256<uint>.Zero
                    il.Emit(OpCodes.Call, _v256GetZeroUInt);
                    // Vector256.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, _v256GreaterThanUInt);
                    break;

                case 2: // short/char: load 16 bytes, expand to 16 words
                    // Vector128.Load<byte>(ptr)
                    il.Emit(OpCodes.Call, _v128LoadByte);
                    // Avx2.ConvertToVector256Int16(bytes)
                    il.Emit(OpCodes.Call, _avx2ConvertToV256Int16);
                    // .AsUInt16()
                    il.Emit(OpCodes.Call, _v256ShortAsUShort);
                    // Vector256<ushort>.Zero
                    il.Emit(OpCodes.Call, _v256GetZeroUShort);
                    // Vector256.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, _v256GreaterThanUShort);
                    break;

                case 1: // byte/bool: load 32 bytes, compare directly
                    // Vector256.Load<byte>(ptr)
                    il.Emit(OpCodes.Call, _v256LoadByte);
                    // Vector256<byte>.Zero
                    il.Emit(OpCodes.Call, _v256GetZeroByte);
                    // Vector256.GreaterThan(vec, zero)
                    il.Emit(OpCodes.Call, _v256GreaterThanByte);
                    break;

                default:
                    throw new NotSupportedException($"Element size {elementSize} not supported");
            }
        }

        /// <summary>
        /// Emit inline V128 mask creation. Stack: byte* -> Vector128{T} (as mask)
        /// </summary>
        private static void EmitInlineMaskCreationV128(ILGenerator il, int elementSize)
        {
            switch (elementSize)
            {
                case 8: // double/long: load 2 bytes, expand to 2 qwords
                    // *(ushort*)ptr
                    il.Emit(OpCodes.Ldind_U2);
                    // Vector128.CreateScalar<ushort>(value)
                    il.Emit(OpCodes.Call, _v128CreateScalarUShort);
                    // .AsByte()
                    il.Emit(OpCodes.Call, _v128UShortAsByte);
                    // Sse41.ConvertToVector128Int64(bytes)
                    il.Emit(OpCodes.Call, _sse41ConvertToV128Int64);
                    // .AsUInt64()
                    il.Emit(OpCodes.Call, _v128LongAsULong);
                    // Vector128<ulong>.Zero
                    il.Emit(OpCodes.Call, _v128GetZeroULong);
                    // Vector128.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, _v128GreaterThanULong);
                    break;

                case 4: // float/int: load 4 bytes, expand to 4 dwords
                    // *(uint*)ptr
                    il.Emit(OpCodes.Ldind_U4);
                    // Vector128.CreateScalar<uint>(value)
                    il.Emit(OpCodes.Call, _v128CreateScalarUInt);
                    // .AsByte()
                    il.Emit(OpCodes.Call, _v128UIntAsByte);
                    // Sse41.ConvertToVector128Int32(bytes)
                    il.Emit(OpCodes.Call, _sse41ConvertToV128Int32);
                    // .AsUInt32()
                    il.Emit(OpCodes.Call, _v128IntAsUInt);
                    // Vector128<uint>.Zero
                    il.Emit(OpCodes.Call, _v128GetZeroUInt);
                    // Vector128.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, _v128GreaterThanUInt);
                    break;

                case 2: // short/char: load 8 bytes, expand to 8 words
                    // *(ulong*)ptr
                    il.Emit(OpCodes.Ldind_I8);
                    // Vector128.CreateScalar<ulong>(value)
                    il.Emit(OpCodes.Call, _v128CreateScalarULong);
                    // .AsByte()
                    il.Emit(OpCodes.Call, _v128ULongAsByte);
                    // Sse41.ConvertToVector128Int16(bytes)
                    il.Emit(OpCodes.Call, _sse41ConvertToV128Int16);
                    // .AsUInt16()
                    il.Emit(OpCodes.Call, _v128ShortAsUShort);
                    // Vector128<ushort>.Zero
                    il.Emit(OpCodes.Call, _v128GetZeroUShort);
                    // Vector128.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, _v128GreaterThanUShort);
                    break;

                case 1: // byte/bool: load 16 bytes, compare directly
                    // Vector128.Load<byte>(ptr)
                    il.Emit(OpCodes.Call, _v128LoadByte);
                    // Vector128<byte>.Zero
                    il.Emit(OpCodes.Call, _v128GetZeroByte);
                    // Vector128.GreaterThan(vec, zero)
                    il.Emit(OpCodes.Call, _v128GreaterThanByte);
                    break;

                default:
                    throw new NotSupportedException($"Element size {elementSize} not supported");
            }
        }

        #endregion

        #region Scalar Fallback

        /// <summary>
        /// Scalar fallback for where operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void WhereScalar<T>(bool* cond, T* x, T* y, T* result, long count) where T : unmanaged
        {
            for (long i = 0; i < count; i++)
            {
                result[i] = cond[i] ? x[i] : y[i];
            }
        }

        #endregion
    }
}
