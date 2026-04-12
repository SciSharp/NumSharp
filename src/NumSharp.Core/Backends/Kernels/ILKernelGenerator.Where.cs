using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

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
            // Get the appropriate mask creation method based on element size
            var maskMethod = GetMaskCreationMethod256((int)elementSize);
            var loadMethod = typeof(Vector256).GetMethod("Load", new[] { typeof(T*) })!.MakeGenericMethod(typeof(T));
            var storeMethod = typeof(Vector256).GetMethod("Store", new[] { typeof(Vector256<>).MakeGenericType(typeof(T)), typeof(T*) })!;
            var selectMethod = typeof(Vector256).GetMethod("ConditionalSelect", new[] {
                typeof(Vector256<>).MakeGenericType(typeof(T)),
                typeof(Vector256<>).MakeGenericType(typeof(T)),
                typeof(Vector256<>).MakeGenericType(typeof(T))
            })!;

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

            // Call mask creation: returns Vector256<T> on stack
            il.Emit(OpCodes.Call, maskMethod);

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
            var maskMethod = GetMaskCreationMethod128((int)elementSize);
            var loadMethod = typeof(Vector128).GetMethod("Load", new[] { typeof(T*) })!.MakeGenericMethod(typeof(T));
            var storeMethod = typeof(Vector128).GetMethod("Store", new[] { typeof(Vector128<>).MakeGenericType(typeof(T)), typeof(T*) })!;
            var selectMethod = typeof(Vector128).GetMethod("ConditionalSelect", new[] {
                typeof(Vector128<>).MakeGenericType(typeof(T)),
                typeof(Vector128<>).MakeGenericType(typeof(T)),
                typeof(Vector128<>).MakeGenericType(typeof(T))
            })!;

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
            il.Emit(OpCodes.Call, maskMethod);

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
            var typeCode = GetNPTypeCode<T>();

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

        private static NPTypeCode GetNPTypeCode<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(bool)) return NPTypeCode.Boolean;
            if (typeof(T) == typeof(byte)) return NPTypeCode.Byte;
            if (typeof(T) == typeof(short)) return NPTypeCode.Int16;
            if (typeof(T) == typeof(ushort)) return NPTypeCode.UInt16;
            if (typeof(T) == typeof(int)) return NPTypeCode.Int32;
            if (typeof(T) == typeof(uint)) return NPTypeCode.UInt32;
            if (typeof(T) == typeof(long)) return NPTypeCode.Int64;
            if (typeof(T) == typeof(ulong)) return NPTypeCode.UInt64;
            if (typeof(T) == typeof(char)) return NPTypeCode.Char;
            if (typeof(T) == typeof(float)) return NPTypeCode.Single;
            if (typeof(T) == typeof(double)) return NPTypeCode.Double;
            if (typeof(T) == typeof(decimal)) return NPTypeCode.Decimal;
            return NPTypeCode.Empty;
        }

        #endregion

        #region Mask Creation Methods

        private static MethodInfo GetMaskCreationMethod256(int elementSize)
        {
            return elementSize switch
            {
                1 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV256_1Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                2 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV256_2Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                4 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV256_4Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                8 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV256_8Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                _ => throw new NotSupportedException($"Element size {elementSize} not supported for SIMD where")
            };
        }

        private static MethodInfo GetMaskCreationMethod128(int elementSize)
        {
            return elementSize switch
            {
                1 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV128_1Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                2 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV128_2Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                4 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV128_4Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                8 => typeof(ILKernelGenerator).GetMethod(nameof(CreateMaskV128_8Byte), BindingFlags.NonPublic | BindingFlags.Static)!,
                _ => throw new NotSupportedException($"Element size {elementSize} not supported for SIMD where")
            };
        }

        /// <summary>
        /// Create V256 mask from 32 bools for 1-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<byte> CreateMaskV256_1Byte(byte* bools)
        {
            var vec = Vector256.Load(bools);
            var zero = Vector256<byte>.Zero;
            var isZero = Vector256.Equals(vec, zero);
            return Vector256.OnesComplement(isZero);
        }

        /// <summary>
        /// Create V256 mask from 16 bools for 2-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<ushort> CreateMaskV256_2Byte(byte* bools)
        {
            return Vector256.Create(
                bools[0] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[1] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[2] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[3] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[4] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[5] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[6] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[7] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[8] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[9] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[10] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[11] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[12] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[13] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[14] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[15] != 0 ? (ushort)0xFFFF : (ushort)0
            );
        }

        /// <summary>
        /// Create V256 mask from 8 bools for 4-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<uint> CreateMaskV256_4Byte(byte* bools)
        {
            return Vector256.Create(
                bools[0] != 0 ? 0xFFFFFFFFu : 0u,
                bools[1] != 0 ? 0xFFFFFFFFu : 0u,
                bools[2] != 0 ? 0xFFFFFFFFu : 0u,
                bools[3] != 0 ? 0xFFFFFFFFu : 0u,
                bools[4] != 0 ? 0xFFFFFFFFu : 0u,
                bools[5] != 0 ? 0xFFFFFFFFu : 0u,
                bools[6] != 0 ? 0xFFFFFFFFu : 0u,
                bools[7] != 0 ? 0xFFFFFFFFu : 0u
            );
        }

        /// <summary>
        /// Create V256 mask from 4 bools for 8-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<ulong> CreateMaskV256_8Byte(byte* bools)
        {
            return Vector256.Create(
                bools[0] != 0 ? 0xFFFFFFFFFFFFFFFFul : 0ul,
                bools[1] != 0 ? 0xFFFFFFFFFFFFFFFFul : 0ul,
                bools[2] != 0 ? 0xFFFFFFFFFFFFFFFFul : 0ul,
                bools[3] != 0 ? 0xFFFFFFFFFFFFFFFFul : 0ul
            );
        }

        /// <summary>
        /// Create V128 mask from 16 bools for 1-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<byte> CreateMaskV128_1Byte(byte* bools)
        {
            var vec = Vector128.Load(bools);
            var zero = Vector128<byte>.Zero;
            var isZero = Vector128.Equals(vec, zero);
            return Vector128.OnesComplement(isZero);
        }

        /// <summary>
        /// Create V128 mask from 8 bools for 2-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<ushort> CreateMaskV128_2Byte(byte* bools)
        {
            return Vector128.Create(
                bools[0] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[1] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[2] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[3] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[4] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[5] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[6] != 0 ? (ushort)0xFFFF : (ushort)0,
                bools[7] != 0 ? (ushort)0xFFFF : (ushort)0
            );
        }

        /// <summary>
        /// Create V128 mask from 4 bools for 4-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<uint> CreateMaskV128_4Byte(byte* bools)
        {
            return Vector128.Create(
                bools[0] != 0 ? 0xFFFFFFFFu : 0u,
                bools[1] != 0 ? 0xFFFFFFFFu : 0u,
                bools[2] != 0 ? 0xFFFFFFFFu : 0u,
                bools[3] != 0 ? 0xFFFFFFFFu : 0u
            );
        }

        /// <summary>
        /// Create V128 mask from 2 bools for 8-byte elements.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<ulong> CreateMaskV128_8Byte(byte* bools)
        {
            return Vector128.Create(
                bools[0] != 0 ? 0xFFFFFFFFFFFFFFFFul : 0ul,
                bools[1] != 0 ? 0xFFFFFFFFFFFFFFFFul : 0ul
            );
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
