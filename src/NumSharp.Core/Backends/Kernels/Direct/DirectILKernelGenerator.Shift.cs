using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Shift.cs - Shift operations (LeftShift, RightShift)
// =============================================================================
//
// OWNERSHIP: Bit shift operations for integer types
// RESPONSIBILITY:
//   - LeftShift (<<) and RightShift (>>) operations
//   - SIMD path for array << scalar (uniform shift amount)
//   - Scalar loop for array << array (element-wise different shifts)
//   - Arithmetic shift for signed types, logical shift for unsigned
//
// SUPPORTED TYPES: Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64
// NOT SUPPORTED: Boolean, Char, Single, Double, Decimal (non-integer)
//
// NUMPY COMPATIBILITY - Shift Overflow Handling:
//   C# masks shift amount to bit width: x << 32 becomes x << (32 & 31) = x << 0 = x
//   NumPy behavior for shift >= bit width:
//     - Left shift: always returns 0
//     - Right shift (unsigned): always returns 0
//     - Right shift (signed positive): returns 0
//     - Right shift (signed negative): returns -1 (all ones, sign extension)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Shift operations - LeftShift and RightShift with SIMD optimization.
    /// </summary>
    public static partial class DirectILKernelGenerator
    {
        #region Shift Kernel Delegates

        /// <summary>
        /// Delegate for shift operation with scalar shift amount.
        /// This is the SIMD-optimized path for uniform shifts.
        /// </summary>
        /// <typeparam name="T">Element type (must be integer)</typeparam>
        /// <param name="input">Pointer to input data</param>
        /// <param name="output">Pointer to output data</param>
        /// <param name="shiftAmount">Number of bits to shift</param>
        /// <param name="count">Number of elements to process</param>
        public unsafe delegate void ShiftScalarKernel<T>(T* input, T* output, int shiftAmount, long count) where T : unmanaged;

        /// <summary>
        /// Delegate for shift operation with per-element shift amounts.
        /// This is the scalar loop path for element-wise shifts.
        /// </summary>
        /// <typeparam name="T">Element type (must be integer)</typeparam>
        /// <param name="input">Pointer to input data</param>
        /// <param name="shifts">Pointer to shift amounts (as int32)</param>
        /// <param name="output">Pointer to output data</param>
        /// <param name="count">Number of elements to process</param>
        public unsafe delegate void ShiftArrayKernel<T>(T* input, int* shifts, T* output, long count) where T : unmanaged;

        #endregion

        #region Shift Kernel Cache

        private static readonly ConcurrentDictionary<(BinaryOp, Type, bool), Delegate> _shiftKernelCache = new();

        #endregion

        #region Public API

        /// <summary>
        /// Get or generate a SIMD-optimized shift kernel for uniform shift amount.
        /// </summary>
        /// <typeparam name="T">Integer element type</typeparam>
        /// <param name="isLeftShift">True for left shift, false for right shift</param>
        /// <returns>Kernel delegate or null if not supported</returns>
        public static ShiftScalarKernel<T>? GetShiftScalarKernel<T>(bool isLeftShift) where T : unmanaged
        {
            if (!Enabled || !IsShiftSupported<T>())
                return null;

            var op = isLeftShift ? BinaryOp.LeftShift : BinaryOp.RightShift;
            var key = (op, typeof(T), true); // true = scalar shift

            if (_shiftKernelCache.TryGetValue(key, out var cached))
                return (ShiftScalarKernel<T>)cached;

            try
            {
                var kernel = GenerateShiftScalarKernel<T>(isLeftShift);
                if (_shiftKernelCache.TryAdd(key, kernel))
                    return kernel;
                return (ShiftScalarKernel<T>)_shiftKernelCache[key];
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetShiftScalarKernel<{typeof(T).Name}>({(isLeftShift ? "Left" : "Right")}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Get or generate a shift kernel for element-wise shift amounts.
        /// </summary>
        /// <typeparam name="T">Integer element type</typeparam>
        /// <param name="isLeftShift">True for left shift, false for right shift</param>
        /// <returns>Kernel delegate or null if not supported</returns>
        public static ShiftArrayKernel<T>? GetShiftArrayKernel<T>(bool isLeftShift) where T : unmanaged
        {
            if (!Enabled || !IsShiftSupported<T>())
                return null;

            var op = isLeftShift ? BinaryOp.LeftShift : BinaryOp.RightShift;
            var key = (op, typeof(T), false); // false = array shift

            if (_shiftKernelCache.TryGetValue(key, out var cached))
                return (ShiftArrayKernel<T>)cached;

            try
            {
                var kernel = GenerateShiftArrayKernel<T>(isLeftShift);
                if (_shiftKernelCache.TryAdd(key, kernel))
                    return kernel;
                return (ShiftArrayKernel<T>)_shiftKernelCache[key];
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetShiftArrayKernel<{typeof(T).Name}>({(isLeftShift ? "Left" : "Right")}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        #endregion

        #region Kernel Generation - Scalar Shift (SIMD)

        /// <summary>
        /// Generate IL kernel for shift with scalar shift amount.
        /// Uses SIMD for the main loop and scalar for the tail.
        /// </summary>
        private static unsafe ShiftScalarKernel<T> GenerateShiftScalarKernel<T>(bool isLeftShift) where T : unmanaged
        {
            var dm = new DynamicMethod(
                name: $"IL_Shift{(isLeftShift ? "Left" : "Right")}_Scalar_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(T*), typeof(T*), typeof(int), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Locals
            var locI = il.DeclareLocal(typeof(long));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(long));   // count - vectorCount
            var locUnrollEnd = il.DeclareLocal(typeof(long));   // count - vectorCount*4

            int elementSize = Unsafe.SizeOf<T>();
            int bitWidth = GetBitWidth<T>();
            int vectorCount = GetShiftVectorCount<T>();
            int unrollStep = vectorCount * 4;

            // Labels
            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollLoopEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();
            var lblOverflowHandled = il.DefineLabel();
            var lblNormalShift = il.DefineLabel();

            // ========== OVERFLOW CHECK ==========
            // NumPy: shift >= bitWidth has special handling
            // Left shift: always 0
            // Right shift unsigned: always 0
            // Right shift signed: 0 for positive, -1 for negative (handled per-element in non-overflow path)
            // For scalar shift, we can check once and fill entire output with appropriate value

            // if ((uint)shiftAmount < bitWidth) goto NormalShift.
            // Unsigned compare so a negative count maps to a huge value -> overflow fill,
            // matching NumPy (left_shift(x, -1) == 0, right_shift(x, -1) == sign fill).
            il.Emit(OpCodes.Ldarg_2);                      // shiftAmount
            il.Emit(OpCodes.Ldc_I4, bitWidth);
            il.Emit(OpCodes.Blt_Un, lblNormalShift);

            // Overflow case: shift >= bitWidth
            if (isLeftShift || IsUnsignedType<T>())
            {
                // Left shift or unsigned right shift: fill with zeros
                // Use Unsafe.InitBlockUnaligned(output, 0, count * elementSize)
                il.Emit(OpCodes.Ldarg_1);                  // output
                il.Emit(OpCodes.Ldc_I4_0);                 // value = 0
                il.Emit(OpCodes.Ldarg_3);                  // count
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_U4);                  // size as uint (InitBlockUnaligned limit)
                il.EmitCall(OpCodes.Call, CachedMethods.UnsafeInitBlockUnaligned, null);
                il.Emit(OpCodes.Ret);
            }
            else
            {
                // Signed right shift: need to check sign of each element
                // For negative values, fill with -1; for positive/zero, fill with 0
                // We'll use a simple loop for this edge case
                EmitSignedRightShiftOverflow<T>(il, elementSize);
                il.Emit(OpCodes.Ret);
            }

            il.MarkLabel(lblNormalShift);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // vectorEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_3);                      // count (now long)
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // unrollEnd = count - vectorCount*4
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

                // Load input vector at (i + offset)
                il.Emit(OpCodes.Ldarg_0);                      // input
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

                // Load shift amount
                il.Emit(OpCodes.Ldarg_2);                      // shiftAmount

                // Perform vector shift
                EmitVectorShift<T>(il, isLeftShift);

                // Store result at (i + offset)
                il.Emit(OpCodes.Ldarg_1);                      // output
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

            // Load input vector
            il.Emit(OpCodes.Ldarg_0);                      // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorLoad<T>(il);

            // Load shift amount
            il.Emit(OpCodes.Ldarg_2);                      // shiftAmount

            // Perform vector shift
            EmitVectorShift<T>(il, isLeftShift);

            // Store result
            il.Emit(OpCodes.Ldarg_1);                      // output
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

            // output[i] = input[i] << shiftAmount (or >>)
            EmitScalarShiftBody<T>(il, isLeftShift, locI, elementSize, useArrayShift: false);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<ShiftScalarKernel<T>>();
        }

        #endregion

        #region Kernel Generation - Array Shift (Scalar Loop)

        /// <summary>
        /// Generate IL kernel for shift with per-element shift amounts.
        /// Uses pure scalar loop since SIMD doesn't support variable shifts per element.
        /// </summary>
        private static unsafe ShiftArrayKernel<T> GenerateShiftArrayKernel<T>(bool isLeftShift) where T : unmanaged
        {
            var dm = new DynamicMethod(
                name: $"IL_Shift{(isLeftShift ? "Left" : "Right")}_Array_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(T*), typeof(int*), typeof(T*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Locals
            var locI = il.DeclareLocal(typeof(long));           // loop counter

            int elementSize = Unsafe.SizeOf<T>();

            // Labels
            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ========== SCALAR LOOP ==========
            il.MarkLabel(lblLoop);

            // if (i >= count) goto LoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_3);                      // count (now long)
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // output[i] = input[i] << shifts[i]
            EmitScalarShiftBody<T>(il, isLeftShift, locI, elementSize, useArrayShift: true);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<ShiftArrayKernel<T>>();
        }

        #endregion

        #region IL Emission Helpers

        /// <summary>
        /// Check if type is supported for shift operations.
        /// </summary>
        private static bool IsShiftSupported<T>() where T : unmanaged
        {
            return typeof(T) == typeof(byte) ||
                   typeof(T) == typeof(sbyte) ||
                   typeof(T) == typeof(short) ||
                   typeof(T) == typeof(ushort) ||
                   typeof(T) == typeof(int) ||
                   typeof(T) == typeof(uint) ||
                   typeof(T) == typeof(long) ||
                   typeof(T) == typeof(ulong);
        }

        /// <summary>
        /// NPTypeCode form of <see cref="IsShiftSupported{T}"/> — the integer dtypes that have
        /// a Vector{N}.Shift* overload (so the SIMD scalar-shift kernel can be generated). Char
        /// is excluded (no Vector{N}&lt;char&gt;); it rides the scalar NpyIter path instead.
        /// </summary>
        internal static bool IsShiftSimdSupported(NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Check if type is unsigned for right shift selection.
        /// </summary>
        private static bool IsUnsignedType<T>() where T : unmanaged
        {
            return typeof(T) == typeof(byte) ||
                   typeof(T) == typeof(ushort) ||
                   typeof(T) == typeof(uint) ||
                   typeof(T) == typeof(ulong);
        }

        /// <summary>
        /// Get bit width for a type (for shift overflow checking).
        /// </summary>
        private static int GetBitWidth<T>() where T : unmanaged
        {
            return Unsafe.SizeOf<T>() * 8;
        }

        /// <summary>
        /// Emit code for signed right shift overflow case (shift >= bitWidth).
        /// For negative values, result is -1 (all ones); for positive/zero, result is 0.
        /// Stack: empty, Params: input (arg0), output (arg1), shiftAmount (arg2), count (arg3)
        /// </summary>
        private static void EmitSignedRightShiftOverflow<T>(ILGenerator il, int elementSize) where T : unmanaged
        {
            // Simple loop: for (i = 0; i < count; i++)
            //   output[i] = input[i] < 0 ? -1 : 0
            var locI = il.DeclareLocal(typeof(long));
            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            var lblNegative = il.DefineLabel();
            var lblStoreResult = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblLoop);

            // if (i >= count) goto LoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_3);                      // count
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // Load output address for store
            il.Emit(OpCodes.Ldarg_1);                      // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load input[i] and check if negative
            il.Emit(OpCodes.Ldarg_0);                      // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect<T>(il);

            // Compare with 0
            il.Emit(OpCodes.Ldc_I4_0);
            if (typeof(T) == typeof(long))
            {
                il.Emit(OpCodes.Conv_I8);
            }
            else if (typeof(T) == typeof(short))
            {
                il.Emit(OpCodes.Conv_I2);
            }
            // int and sbyte work with I4

            il.Emit(OpCodes.Blt, lblNegative);

            // Positive/zero case: store 0
            il.Emit(OpCodes.Ldc_I4_0);
            if (typeof(T) == typeof(long))
            {
                il.Emit(OpCodes.Conv_I8);
            }
            else if (typeof(T) == typeof(short))
            {
                il.Emit(OpCodes.Conv_I2);
            }
            il.Emit(OpCodes.Br, lblStoreResult);

            // Negative case: store -1
            il.MarkLabel(lblNegative);
            il.Emit(OpCodes.Ldc_I4_M1);
            if (typeof(T) == typeof(long))
            {
                il.Emit(OpCodes.Conv_I8);
            }
            else if (typeof(T) == typeof(short))
            {
                il.Emit(OpCodes.Conv_I2);
            }

            il.MarkLabel(lblStoreResult);
            EmitStoreIndirect<T>(il);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Get vector count for shift operations based on current SIMD width.
        /// </summary>
        private static int GetShiftVectorCount<T>() where T : unmanaged
        {
            return VectorBits switch
            {
                512 => Vector512<T>.Count,
                256 => Vector256<T>.Count,
                128 => Vector128<T>.Count,
                _ => 1
            };
        }

        /// <summary>
        /// Emit vector shift operation.
        /// Stack: [vector, shiftAmount] -> [shifted vector]
        /// </summary>
        private static void EmitVectorShift<T>(ILGenerator il, bool isLeftShift) where T : unmanaged
        {
            string methodName = isLeftShift
                ? "ShiftLeft"
                : (IsUnsignedType<T>() ? "ShiftRightLogical" : "ShiftRightArithmetic");

            il.EmitCall(OpCodes.Call,
                VectorMethodCache.ShiftByScalar(VectorBits, typeof(T), methodName), null);
        }

        /// <summary>
        /// Emit scalar shift operation body.
        /// For scalar shift (useArrayShift=false): output[i] = input[i] << shiftAmount (arg2)
        /// For array shift (useArrayShift=true): output[i] = input[i] << shifts[i] (arg1)
        /// </summary>
        private static void EmitScalarShiftBody<T>(ILGenerator il, bool isLeftShift, LocalBuilder locI, int elementSize, bool useArrayShift) where T : unmanaged
        {
            int bitWidth = GetBitWidth<T>();

            if (useArrayShift)
            {
                // For array shifts, each shift amount may be different, so we need per-element overflow check
                var lblNoOverflow = il.DefineLabel();
                var lblStoreZero = il.DefineLabel();
                var lblStoreNegOne = il.DefineLabel();
                var lblStore = il.DefineLabel();
                var lblDone = il.DefineLabel();

                // Load shift amount first to check for overflow
                il.Emit(OpCodes.Ldarg_1);                  // shifts (arg1 for array shift)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_4);                 // sizeof(int)
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I4);

                // Duplicate shift amount for comparison and potential use
                il.Emit(OpCodes.Dup);

                // if ((uint)shiftAmount < bitWidth) goto NoOverflow.
                // Unsigned compare so a negative per-element count overflows like NumPy.
                il.Emit(OpCodes.Ldc_I4, bitWidth);
                il.Emit(OpCodes.Blt_Un, lblNoOverflow);

                // Overflow case: shift >= bitWidth
                il.Emit(OpCodes.Pop);  // remove duplicate shift amount

                // Load output address for overflow store
                il.Emit(OpCodes.Ldarg_2);                      // output (arg2 for array shift)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);

                if (isLeftShift || IsUnsignedType<T>())
                {
                    // Left shift or unsigned right shift: result is 0
                    il.Emit(OpCodes.Ldc_I4_0);
                    if (typeof(T) == typeof(long) || typeof(T) == typeof(ulong))
                    {
                        il.Emit(OpCodes.Conv_I8);
                    }
                    EmitStoreIndirect<T>(il);
                }
                else
                {
                    // Signed right shift: result is 0 for positive, -1 for negative
                    // Load input[i] to check sign
                    il.Emit(OpCodes.Ldarg_0);                      // input
                    il.Emit(OpCodes.Ldloc, locI);
                    il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    EmitLoadIndirect<T>(il);

                    // Compare with 0
                    il.Emit(OpCodes.Ldc_I4_0);
                    if (typeof(T) == typeof(long))
                    {
                        il.Emit(OpCodes.Conv_I8);
                    }
                    il.Emit(OpCodes.Blt, lblStoreNegOne);

                    // Positive/zero: store 0
                    il.Emit(OpCodes.Ldc_I4_0);
                    if (typeof(T) == typeof(long))
                    {
                        il.Emit(OpCodes.Conv_I8);
                    }
                    il.Emit(OpCodes.Br, lblStore);

                    // Negative: store -1
                    il.MarkLabel(lblStoreNegOne);
                    il.Emit(OpCodes.Ldc_I4_M1);
                    if (typeof(T) == typeof(long))
                    {
                        il.Emit(OpCodes.Conv_I8);
                    }

                    il.MarkLabel(lblStore);
                    EmitStoreIndirect<T>(il);
                }
                il.Emit(OpCodes.Br, lblDone);

                // Normal shift case
                il.MarkLabel(lblNoOverflow);

                // Stack has: shiftAmount
                // Store it in a temp local for reuse
                var locShift = il.DeclareLocal(typeof(int));
                il.Emit(OpCodes.Stloc, locShift);

                // Address: output + i * elementSize
                il.Emit(OpCodes.Ldarg_2);                      // output (arg2 for array shift)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);

                // Load input[i]
                il.Emit(OpCodes.Ldarg_0);                      // input
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect<T>(il);

                // Load shift amount from local
                il.Emit(OpCodes.Ldloc, locShift);

                // Perform scalar shift
                EmitScalarShift<T>(il, isLeftShift);

                // Store to output[i]
                EmitStoreIndirect<T>(il);

                il.MarkLabel(lblDone);
            }
            else
            {
                // Scalar shift: overflow already handled at the start of the kernel
                // Address: output + i * elementSize
                il.Emit(OpCodes.Ldarg_1);                      // output (arg1 for scalar shift)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);

                // Load input[i]
                il.Emit(OpCodes.Ldarg_0);                      // input
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                EmitLoadIndirect<T>(il);

                // Load scalar shift amount
                il.Emit(OpCodes.Ldarg_2);                  // shiftAmount

                // Perform scalar shift
                EmitScalarShift<T>(il, isLeftShift);

                // Store to output[i]
                EmitStoreIndirect<T>(il);
            }
        }

        /// <summary>
        /// Emit scalar shift instruction.
        /// Stack: [value, shiftAmount] -> [result]
        /// </summary>
        private static void EmitScalarShift<T>(ILGenerator il, bool isLeftShift) where T : unmanaged
        {
            if (isLeftShift)
            {
                // Left shift is the same for all integer types
                il.Emit(OpCodes.Shl);
            }
            else
            {
                // Right shift: Shr for signed (arithmetic), Shr_Un for unsigned (logical)
                if (IsUnsignedType<T>())
                {
                    il.Emit(OpCodes.Shr_Un);
                }
                else
                {
                    il.Emit(OpCodes.Shr);
                }
            }

            // Apply appropriate truncation for smaller types
            if (typeof(T) == typeof(byte))
            {
                il.Emit(OpCodes.Conv_U1);
            }
            else if (typeof(T) == typeof(sbyte))
            {
                il.Emit(OpCodes.Conv_I1);
            }
            else if (typeof(T) == typeof(short))
            {
                il.Emit(OpCodes.Conv_I2);
            }
            else if (typeof(T) == typeof(ushort))
            {
                il.Emit(OpCodes.Conv_U2);
            }
            // int, uint, long, ulong don't need truncation
        }

        /// <summary>
        /// Emit a NumPy-correct bit shift that consumes <c>[value(t), count(t)]</c> from the
        /// evaluation stack and leaves <c>[shifted(t)]</c>. This is the entry point used by
        /// <see cref="EmitScalarOperation"/> so <see cref="BinaryOp.LeftShift"/> /
        /// <see cref="BinaryOp.RightShift"/> behave as first-class binary ops in every generic
        /// scalar loop — the MixedType General/Chunk kernels, the NpyIter Tier-3B scalar body,
        /// and the scalar×scalar delegate — without a per-dtype switch.
        ///
        /// The generic binary loops convert BOTH operands to <paramref name="t"/> (the promoted
        /// loop dtype) before calling here, so the count arrives at the result width and is
        /// truncated to <c>int</c> only for the IL <c>shl</c>/<c>shr</c> opcode.
        ///
        /// Overflow rule (probed against NumPy 2.4.2): a count that is negative OR &gt;= the
        /// result type's bit width yields 0 for left shift and unsigned right shift, and a sign
        /// fill (-1 when the value is negative, else 0) for signed right shift. The single
        /// unsigned comparison <c>(ulong)count &gt;= bitWidth</c> captures BOTH the negative and
        /// the too-large case in one branch.
        /// </summary>
        internal static void EmitShiftFromStack(ILGenerator il, NPTypeCode t, bool isLeft)
        {
            var clr = GetClrType(t);
            int bitWidth = GetTypeSize(t) * 8;
            bool unsigned = IsUnsigned(t);

            var locValue = il.DeclareLocal(clr);
            var locCount = il.DeclareLocal(clr);
            var lblNormal = il.DefineLabel();
            var lblNeg = il.DefineLabel();
            var lblDone = il.DefineLabel();

            il.Emit(OpCodes.Stloc, locCount);   // pop count
            il.Emit(OpCodes.Stloc, locValue);   // pop value

            // if ((ulong)count < bitWidth) goto normal — negative widens to a huge value
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(unsigned ? OpCodes.Conv_U8 : OpCodes.Conv_I8);
            il.Emit(OpCodes.Ldc_I8, (long)bitWidth);
            il.Emit(OpCodes.Blt_Un, lblNormal);

            // ---- overflow fill ----
            if (isLeft || unsigned)
            {
                EmitShiftIntConst(il, t, 0);
                il.Emit(OpCodes.Br, lblDone);
            }
            else
            {
                // signed right shift: value < 0 ? -1 : 0
                il.Emit(OpCodes.Ldloc, locValue);
                EmitShiftIntConst(il, t, 0);
                il.Emit(OpCodes.Blt, lblNeg);
                EmitShiftIntConst(il, t, 0);
                il.Emit(OpCodes.Br, lblDone);
                il.MarkLabel(lblNeg);
                EmitShiftIntConst(il, t, -1);
                il.Emit(OpCodes.Br, lblDone);
            }

            // ---- normal shift ----
            il.MarkLabel(lblNormal);
            il.Emit(OpCodes.Ldloc, locValue);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Conv_I4);          // shift count is int for shl/shr (already < bitWidth)
            if (isLeft)
                il.Emit(OpCodes.Shl);
            else
                il.Emit(unsigned ? OpCodes.Shr_Un : OpCodes.Shr);
            EmitShiftTruncate(il, t);

            il.MarkLabel(lblDone);
        }

        /// <summary>
        /// Push the integer constant <paramref name="val"/> (only 0 or -1 are used) typed as
        /// <paramref name="t"/> — widened to int64 for the 64-bit dtypes so the stack type at the
        /// merge label matches the shifted value.
        /// </summary>
        private static void EmitShiftIntConst(ILGenerator il, NPTypeCode t, int val)
        {
            il.Emit(val == -1 ? OpCodes.Ldc_I4_M1 : OpCodes.Ldc_I4_0);
            if (t == NPTypeCode.Int64 || t == NPTypeCode.UInt64)
                il.Emit(OpCodes.Conv_I8);
        }

        /// <summary>
        /// Truncate a shift result on the stack back to the sub-word dtype width
        /// (the IL <c>shl</c>/<c>shr</c> produced an int32/int64), matching NumPy's wrapping.
        /// </summary>
        private static void EmitShiftTruncate(ILGenerator il, NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.SByte: il.Emit(OpCodes.Conv_I1); break;
                case NPTypeCode.Byte:  il.Emit(OpCodes.Conv_U1); break;
                case NPTypeCode.Int16: il.Emit(OpCodes.Conv_I2); break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:  il.Emit(OpCodes.Conv_U2); break;
                // Int32/UInt32/Int64/UInt64 need no truncation
            }
        }

        #endregion
    }
}
