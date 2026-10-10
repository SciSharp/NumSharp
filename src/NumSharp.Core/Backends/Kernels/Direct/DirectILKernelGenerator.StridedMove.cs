using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.StridedMove.cs — IL kernel: copy a strided run of
// elements to a strided run, one typed move an element, four elements a pass
// =============================================================================
//
// RESPONSIBILITY
//   dst[i * dstStride] = src[i * srcStride]   for i in [0, count)
//   for one element width (1, 2, 4, 8 or 16 bytes), strides in BYTES and of
//   either sign, a source stride of 0 repeating one element.
//
// WHY IT EXISTS
//   The house strided copies are vector kernels for the two layouts a vector can
//   read without a gather: a step of two elements (de-interleave) and a reversed
//   run. Every other stride of a one- or two-byte dtype — a COLUMN of a C-ordered
//   matrix is a run with a stride of one row — reached a generic loop at 0.71 ns
//   an element, and a strided one-byte STORE reached the diagonal-write kernel at
//   0.41 ns. Both are one load and one store an element with a loop around them;
//   this kernel is that work, unrolled so the loop control is paid once per four
//   elements.
//
// WHO USES IT
//   ComparisonRun (Backends/Kernels/ComparisonRun.cs): staging an input run of an
//   arbitrary stride into a contiguous block, and storing a block of results into
//   a bool output that is not contiguous.
//
// KERNEL (one DynamicMethod per element width, kept for the life of the process)
//
//       d2 = 2 * dstStride ... d4 = 4 * dstStride       (and the same for src)
//       while (count >= 4) {
//           move(dst,      src);
//           move(dst + d1, src + s1);
//           move(dst + d2, src + s2);
//           move(dst + d3, src + s3);
//           dst += d4;  src += s4;  count -= 4;
//       }
//       while (count > 0) { move(dst, src); dst += d1; src += s1; count--; }
//
//   Each element is loaded and stored before the next one is loaded, so the
//   order of the simple loop is kept (an overlapping in-place move behaves the
//   same). A 16-byte element is two 8-byte moves.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     IL-emitted strided copy of fixed-width elements: <c>dst[i * dstByteStride] = src[i * srcByteStride]</c>
    ///     for each <c>i</c> in <c>[0, count)</c>.
    /// </summary>
    /// <param name="dst">First destination element.</param>
    /// <param name="dstByteStride">Signed byte stride between destination elements.</param>
    /// <param name="src">First source element.</param>
    /// <param name="srcByteStride">Signed byte stride between source elements; 0 repeats one element.</param>
    /// <param name="count">Number of elements (0 or less copies nothing).</param>
    public unsafe delegate void StridedMoveKernel(byte* dst, long dstByteStride, byte* src, long srcByteStride, long count);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The kernels by element width (index = width in bytes; only 1, 2, 4, 8 and 16 are ever filled).</summary>
        private static readonly StridedMoveKernel[] s_stridedMoveKernels = new StridedMoveKernel[17];

        /// <summary>
        ///     The strided move kernel for elements of <paramref name="itemSize"/> bytes, or null when the width is not
        ///     one of 1, 2, 4, 8, 16 or runtime IL generation is switched off (the caller then keeps its other copy).
        /// </summary>
        /// <param name="itemSize">Element width in bytes.</param>
        /// <returns>The cached kernel, or null.</returns>
        /// <remarks>Two threads racing on a first call both emit an equivalent kernel; one reference wins the slot.</remarks>
        internal static StridedMoveKernel GetStridedMoveKernel(int itemSize)
        {
            if (!Enabled || CopyKindFor(itemSize) == 0)
                return null;
            var kernel = Volatile.Read(ref s_stridedMoveKernels[itemSize]);
            if (kernel is null)
            {
                kernel = EmitStridedMove(itemSize);
                Volatile.Write(ref s_stridedMoveKernels[itemSize], kernel);
            }
            return kernel;
        }

        /// <summary>Emits the kernel of one element width (see the file header for the loop).</summary>
        /// <param name="itemSize">Element width: 1, 2, 4, 8 or 16 bytes.</param>
        /// <returns>The compiled kernel.</returns>
        private static StridedMoveKernel EmitStridedMove(int itemSize)
        {
            var dm = new DynamicMethod(
                name: $"IL_StridedMove_{itemSize}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 dst
                    typeof(long),   // 1 dstByteStride
                    typeof(byte*),  // 2 src
                    typeof(long),   // 3 srcByteStride
                    typeof(long),   // 4 count
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            var dst = il.DeclareLocal(typeof(byte*));
            var src = il.DeclareLocal(typeof(byte*));
            var count = il.DeclareLocal(typeof(long));
            // Offsets of the second, third and fourth element of a pass, and the advance of a whole pass.
            var d2 = il.DeclareLocal(typeof(long));
            var d3 = il.DeclareLocal(typeof(long));
            var d4 = il.DeclareLocal(typeof(long));
            var s2 = il.DeclareLocal(typeof(long));
            var s3 = il.DeclareLocal(typeof(long));
            var s4 = il.DeclareLocal(typeof(long));

            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, dst);
            il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Stloc, src);
            il.Emit(OpCodes.Ldarg, 4); il.Emit(OpCodes.Stloc, count);

            EmitStridedMoveMultiples(il, OpCodes.Ldarg_1, d2, d3, d4);
            EmitStridedMoveMultiples(il, OpCodes.Ldarg_3, s2, s3, s4);

            var passHead = il.DefineLabel();
            var tailHead = il.DefineLabel();
            var done = il.DefineLabel();

            // ----- four elements a pass -----
            il.MarkLabel(passHead);
            il.Emit(OpCodes.Ldloc, count);
            il.Emit(OpCodes.Ldc_I8, 4L);
            il.Emit(OpCodes.Blt, tailHead);

            EmitStridedMoveElement(il, itemSize, dst, src, null, null);
            EmitStridedMoveElement(il, itemSize, dst, src, il2 => il2.Emit(OpCodes.Ldarg_1), il2 => il2.Emit(OpCodes.Ldarg_3));
            EmitStridedMoveElement(il, itemSize, dst, src, il2 => il2.Emit(OpCodes.Ldloc, d2), il2 => il2.Emit(OpCodes.Ldloc, s2));
            EmitStridedMoveElement(il, itemSize, dst, src, il2 => il2.Emit(OpCodes.Ldloc, d3), il2 => il2.Emit(OpCodes.Ldloc, s3));

            EmitStridedMoveAdvance(il, dst, il2 => il2.Emit(OpCodes.Ldloc, d4));
            EmitStridedMoveAdvance(il, src, il2 => il2.Emit(OpCodes.Ldloc, s4));
            il.Emit(OpCodes.Ldloc, count);
            il.Emit(OpCodes.Ldc_I8, 4L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, count);
            il.Emit(OpCodes.Br, passHead);

            // ----- the last 0 to 3 elements -----
            il.MarkLabel(tailHead);
            il.Emit(OpCodes.Ldloc, count);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, done);

            EmitStridedMoveElement(il, itemSize, dst, src, null, null);
            EmitStridedMoveAdvance(il, dst, il2 => il2.Emit(OpCodes.Ldarg_1));
            EmitStridedMoveAdvance(il, src, il2 => il2.Emit(OpCodes.Ldarg_3));
            il.Emit(OpCodes.Ldloc, count);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, count);
            il.Emit(OpCodes.Br, tailHead);

            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<StridedMoveKernel>();
        }

        /// <summary>Emits <c>x2 = 2 * stride; x3 = 3 * stride; x4 = 4 * stride</c>.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="loadStride">The opcode that pushes the stride argument.</param>
        /// <param name="x2">Receives twice the stride.</param>
        /// <param name="x3">Receives three times the stride.</param>
        /// <param name="x4">Receives four times the stride.</param>
        private static void EmitStridedMoveMultiples(ILGenerator il, OpCode loadStride, LocalBuilder x2, LocalBuilder x3, LocalBuilder x4)
        {
            il.Emit(loadStride); il.Emit(loadStride); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, x2);
            il.Emit(OpCodes.Ldloc, x2); il.Emit(loadStride); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, x3);
            il.Emit(OpCodes.Ldloc, x2); il.Emit(OpCodes.Ldloc, x2); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, x4);
        }

        /// <summary>
        ///     Emits one element move: <c>*(dst + dstOffset) = *(src + srcOffset)</c> as a typed load and store of the
        ///     element width (two 8-byte moves for a 16-byte element).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="itemSize">Element width in bytes.</param>
        /// <param name="dst">The destination cursor.</param>
        /// <param name="src">The source cursor.</param>
        /// <param name="pushDstOffset">Pushes the byte offset from the destination cursor; null for none.</param>
        /// <param name="pushSrcOffset">Pushes the byte offset from the source cursor; null for none.</param>
        private static void EmitStridedMoveElement(ILGenerator il, int itemSize, LocalBuilder dst, LocalBuilder src,
            Action<ILGenerator> pushDstOffset, Action<ILGenerator> pushSrcOffset)
        {
            int parts = itemSize == 16 ? 2 : 1;
            for (int part = 0; part < parts; part++)
            {
                EmitStridedMoveAddress(il, dst, pushDstOffset, part * 8);
                EmitStridedMoveAddress(il, src, pushSrcOffset, part * 8);
                switch (itemSize)
                {
                    case 1: il.Emit(OpCodes.Ldind_U1); il.Emit(OpCodes.Stind_I1); break;
                    case 2: il.Emit(OpCodes.Ldind_U2); il.Emit(OpCodes.Stind_I2); break;
                    case 4: il.Emit(OpCodes.Ldind_U4); il.Emit(OpCodes.Stind_I4); break;
                    default: il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stind_I8); break;   // 8, and each half of 16
                }
            }
        }

        /// <summary>Pushes <c>cursor + offset + extra</c> as a native address.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="cursor">The pointer local.</param>
        /// <param name="pushOffset">Pushes a byte offset (a <c>long</c>); null for none.</param>
        /// <param name="extra">A constant byte offset added on top (the second half of a 16-byte element).</param>
        private static void EmitStridedMoveAddress(ILGenerator il, LocalBuilder cursor, Action<ILGenerator> pushOffset, int extra)
        {
            il.Emit(OpCodes.Ldloc, cursor);
            if (pushOffset is not null)
            {
                pushOffset(il);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            if (extra != 0)
            {
                il.Emit(OpCodes.Ldc_I4, extra);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
        }

        /// <summary>Emits <c>cursor += amount</c>.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="cursor">The pointer local.</param>
        /// <param name="pushAmount">Pushes the byte amount (a <c>long</c>).</param>
        private static void EmitStridedMoveAdvance(ILGenerator il, LocalBuilder cursor, Action<ILGenerator> pushAmount)
        {
            il.Emit(OpCodes.Ldloc, cursor);
            pushAmount(il);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, cursor);
        }
    }
}
