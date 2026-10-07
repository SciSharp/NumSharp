using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.ElementCopy.cs — copy opaque fixed-width elements over a 2-D byte-strided grid
// =============================================================================
//
// RESPONSIBILITY:
//   The element copy of the fixed-width string lane when source and destination share one string class: every
//   destination element receives the first COPY bytes of its source element and, when the destination is wider, ZERO
//   NUL bytes after them (NumPy's string_to_string zero-pad copy / truncation). COPY and ZERO are emit-time constants, so
//   the per-element cpblk / initblk are unrolled by the JIT into a few vector moves.
//
//   The grid is at most 2-D (rows x cols) with independent byte strides per operand — a 1-D strided or reversed view, a
//   transposed matrix, a contiguous array (one row), a broadcast source (stride 0) — which covers the layouts the string
//   copy route used to send through its unit view (one extra character axis), where every element is a short inner row
//   and the iterator pays its per-row cost on each (4-8 ns per element, 2-5x the element move itself).
//
// KERNEL (DynamicMethod-emitted, one per (COPY, ZERO)):
//
//   * ElementCopyKernel(byte* dst, byte* src, long rows, long cols,
//                       long dstRowStride, long dstColStride, long srcRowStride, long srcColStride)
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     An IL-emitted copy of opaque fixed-width elements over a 2-D grid of byte strides: for every
    ///     <c>(r, c)</c>, the destination element at <c>dst + r * dstRowStride + c * dstColStride</c> receives the kernel's
    ///     COPY bytes from <c>src + r * srcRowStride + c * srcColStride</c>, followed by ZERO NUL bytes.
    /// </summary>
    /// <param name="dst">The first destination element.</param>
    /// <param name="src">The first source element.</param>
    /// <param name="rows">The outer extent.</param>
    /// <param name="cols">The inner extent.</param>
    /// <param name="dstRowStride">The destination's outer stride in bytes.</param>
    /// <param name="dstColStride">The destination's inner stride in bytes.</param>
    /// <param name="srcRowStride">The source's outer stride in bytes (0 = broadcast).</param>
    /// <param name="srcColStride">The source's inner stride in bytes (0 = broadcast).</param>
    public unsafe delegate void ElementCopyKernel(byte* dst, byte* src, long rows, long cols,
                                                  long dstRowStride, long dstColStride, long srcRowStride, long srcColStride);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The generated <see cref="ElementCopyKernel"/>s, keyed by (copy bytes, zero bytes).</summary>
        private static readonly ConcurrentDictionary<(int Copy, int Zero), ElementCopyKernel> _elementCopyKernels = new();

        /// <summary>
        ///     The <see cref="ElementCopyKernel"/> that moves <paramref name="copyBytes"/> bytes per element and NUL-fills
        ///     <paramref name="zeroBytes"/> more, generated on first use and cached.
        /// </summary>
        /// <param name="copyBytes">The bytes copied from each source element (at least 1).</param>
        /// <param name="zeroBytes">The NUL bytes written after them (0 when the destination is not wider).</param>
        /// <returns>The kernel, or null when IL generation is disabled or failed (the caller takes its general route).</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="copyBytes"/> is not positive, or
        ///     <paramref name="zeroBytes"/> is negative.</exception>
        public static ElementCopyKernel GetElementCopyKernel(int copyBytes, int zeroBytes)
        {
            if (copyBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(copyBytes), copyBytes, "An element copy moves at least one byte.");
            if (zeroBytes < 0)
                throw new ArgumentOutOfRangeException(nameof(zeroBytes), zeroBytes, "The NUL padding cannot be negative.");
            if (!Enabled)
                return null;
            var key = (copyBytes, zeroBytes);
            if (_elementCopyKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return _elementCopyKernels.GetOrAdd(key, GenerateElementCopyKernelIL(copyBytes, zeroBytes));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetElementCopyKernel({copyBytes}, {zeroBytes}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits the kernel. Pseudocode (<c>COPY</c> and <c>ZERO</c> are constants):
        ///     <code>
        ///     for (long r = 0; r &lt; rows; r++) {
        ///         byte* d = dst + r * dstRowStride;  byte* s = src + r * srcRowStride;
        ///         for (long c = 0; c &lt; cols; c++) {
        ///             cpblk(d, s, COPY);
        ///             if (ZERO != 0) initblk(d + COPY, 0, ZERO);   // emitted only when ZERO != 0
        ///             d += dstColStride;  s += srcColStride;
        ///         }
        ///     }
        ///     </code>
        /// </summary>
        /// <param name="copyBytes">The bytes copied per element.</param>
        /// <param name="zeroBytes">The NUL bytes written after them.</param>
        /// <returns>The compiled kernel.</returns>
        private static unsafe ElementCopyKernel GenerateElementCopyKernelIL(int copyBytes, int zeroBytes)
        {
            var dm = new DynamicMethod(
                name: $"IL_ElementCopy_{copyBytes}_{zeroBytes}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*), typeof(byte*), typeof(long), typeof(long),
                    typeof(long), typeof(long), typeof(long), typeof(long),
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            var r = il.DeclareLocal(typeof(long));
            var c = il.DeclareLocal(typeof(long));
            var d = il.DeclareLocal(typeof(byte*));
            var s = il.DeclareLocal(typeof(byte*));
            var rowLoop = il.DefineLabel();
            var rowsDone = il.DefineLabel();
            var colLoop = il.DefineLabel();
            var colsDone = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, r);

            il.MarkLabel(rowLoop);
            il.Emit(OpCodes.Ldloc, r);
            il.Emit(OpCodes.Ldarg_2);                       // rows
            il.Emit(OpCodes.Bge, rowsDone);

            // d = dst + r * dstRowStride; s = src + r * srcRowStride
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, r);
            il.Emit(OpCodes.Ldarg_S, (byte)4);              // dstRowStride
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, d);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, r);
            il.Emit(OpCodes.Ldarg_S, (byte)6);              // srcRowStride
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, s);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, c);
            il.MarkLabel(colLoop);
            il.Emit(OpCodes.Ldloc, c);
            il.Emit(OpCodes.Ldarg_3);                       // cols
            il.Emit(OpCodes.Bge, colsDone);

            // cpblk(d, s, COPY): a constant size, unrolled by the JIT.
            il.Emit(OpCodes.Ldloc, d);
            il.Emit(OpCodes.Ldloc, s);
            il.Emit(OpCodes.Ldc_I4, copyBytes);
            il.Emit(OpCodes.Unaligned, (byte)1);
            il.Emit(OpCodes.Cpblk);
            if (zeroBytes != 0)
            {
                // initblk(d + COPY, 0, ZERO): the NUL padding of a wider destination.
                il.Emit(OpCodes.Ldloc, d);
                il.Emit(OpCodes.Ldc_I4, copyBytes);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Ldc_I4, zeroBytes);
                il.Emit(OpCodes.Unaligned, (byte)1);
                il.Emit(OpCodes.Initblk);
            }

            // d += dstColStride; s += srcColStride; c++
            il.Emit(OpCodes.Ldloc, d);
            il.Emit(OpCodes.Ldarg_S, (byte)5);              // dstColStride
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, d);
            il.Emit(OpCodes.Ldloc, s);
            il.Emit(OpCodes.Ldarg_S, (byte)7);              // srcColStride
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, s);
            il.Emit(OpCodes.Ldloc, c);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, c);
            il.Emit(OpCodes.Br, colLoop);

            il.MarkLabel(colsDone);
            il.Emit(OpCodes.Ldloc, r);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, r);
            il.Emit(OpCodes.Br, rowLoop);

            il.MarkLabel(rowsDone);
            il.Emit(OpCodes.Ret);
            return (ElementCopyKernel)dm.CreateDelegate(typeof(ElementCopyKernel));
        }
    }
}
