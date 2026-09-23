using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.PutMask.cs — IL kernel for np.putmask
// =============================================================================
//
// RESPONSIBILITY:
//   np.putmask(a, mask, values) writes values into a wherever a boolean mask is
//   True, walking both in C-order. It is the sibling of np.place, and differs in
//   ONE crucial way: the values cursor advances on EVERY element (position i),
//   not once per True. So the value written at position i (when mask[i]) is
//   values[i % nv] — NumPy's npy_fastputmask_impl, where `j` increments in
//   lockstep with `i` and wraps modulo nv. (np.place advances the cursor only on
//   True positions.)
//
//   NumPy special-cases nv == 1 (a scalar broadcast) with a cursor-free loop; we
//   do the same behind a runtime branch so the common putmask(a, cond, scalar)
//   pays no cursor arithmetic on skipped elements.
//
//   The kernel walks the mask byte-by-byte (NumSharp stores bools as 1-byte) and
//   each True triggers an inline typed MOV of elemBytes bytes (cpblk fallback for
//   non-1/2/4/8/16 widths, which do not occur for real dtypes).
//
// KERNEL (DynamicMethod-emitted, singleton per copy-width):
//
//   * PutMaskKernel
//       (byte* dst,            // target buffer (C-contiguous)
//        byte* mask,           // contig bool mask (1 byte each), same size as dst
//        long maskSize,
//        byte* values,         // contig source buffer
//        long valuesCount,     // nv > 0; caller short-circuits the nv == 0 no-op
//        long elemBytes)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted mask-driven scatter for <c>np.putmask</c>. For each <c>i</c> in
    /// <c>[0, maskSize)</c> with <c>mask[i]</c> true, writes
    /// <c>dst[i] = values[i % valuesCount]</c>. The values cursor advances on every
    /// element (NumPy's <c>npy_fastputmask</c>), unlike <see cref="PlaceKernel"/>.
    /// </summary>
    public unsafe delegate void PutMaskKernel(
        byte* dst, byte* mask, long maskSize,
        byte* values, long valuesCount, long elemBytes);

    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<int, PutMaskKernel> _putMaskKernels = new();

        /// <summary>
        /// IL-emitted putmask kernel, generated once per copy-width (<paramref name="copyKind"/> — the
        /// <see cref="CopyKindFor"/> of the dtype itemsize; always 1/2/4/8/16 for real dtypes, so the
        /// masked write is a typed MOV rather than a per-element cpblk).
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static PutMaskKernel GetPutMaskKernel(int copyKind)
        {
            if (!Enabled)
                return null;

            if (_putMaskKernels.TryGetValue(copyKind, out var cached))
                return cached;

            try
            {
                var k = GeneratePutMaskKernelIL(copyKind);
                return _putMaskKernels.GetOrAdd(copyKind, k);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPutMaskKernel({copyKind}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the putmask kernel. Pseudocode:
        /// <code>
        /// void PutMask(byte* dst, byte* mask, long maskSize,
        ///              byte* values, long nv, long elemBytes) {
        ///     if (nv == 1) {                       // scalar broadcast — no cursor
        ///         for (long i = 0; i &lt; maskSize; i++)
        ///             if (mask[i] != 0)
        ///                 copy(dst + i*elemBytes, values, elemBytes);
        ///     } else {
        ///         long j = 0;
        ///         for (long i = 0; i &lt; maskSize; i++) {
        ///             if (mask[i] != 0)
        ///                 copy(dst + i*elemBytes, values + j*elemBytes, elemBytes);
        ///             j++;                          // advance EVERY iteration (position cursor)
        ///             if (j &gt;= nv) j = 0;
        ///         }
        ///     }
        /// }
        /// </code>
        /// </summary>
        private static PutMaskKernel GeneratePutMaskKernelIL(int copyKind)
        {
            var dm = new DynamicMethod(
                name: $"IL_PutMask_c{copyKind}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 dst
                    typeof(byte*),  // 1 mask
                    typeof(long),   // 2 maskSize
                    typeof(byte*),  // 3 values
                    typeof(long),   // 4 valuesCount
                    typeof(long),   // 5 elemBytes
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));
            var locSrcPtr = il.DeclareLocal(typeof(byte*));
            var locDstPtr = il.DeclareLocal(typeof(byte*));

            var lblCursor = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // if (valuesCount != 1) goto cursorPath
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, lblCursor);

            // ------------------------------------------------------------------
            // Scalar broadcast (nv == 1): srcPtr is always `values`, no cursor.
            // ------------------------------------------------------------------
            {
                var lblHead = il.DefineLabel();
                var lblSkip = il.DefineLabel();

                // locSrcPtr = values
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Stloc, locSrcPtr);
                // i = 0
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locI);

                il.MarkLabel(lblHead);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_2);            // maskSize
                il.Emit(OpCodes.Bge, lblEnd);

                // if (mask[i] == 0) goto skip
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Brfalse, lblSkip);

                // dstPtr = dst + i * elemBytes
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locDstPtr);

                EmitElementCopy(il, copyKind, locDstPtr, locSrcPtr, () => il.Emit(OpCodes.Ldarg, 5));

                il.MarkLabel(lblSkip);
                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblHead);
            }

            // ------------------------------------------------------------------
            // General (nv != 1): position cursor j advances on EVERY element.
            // ------------------------------------------------------------------
            il.MarkLabel(lblCursor);
            {
                var lblHead = il.DefineLabel();
                var lblSkip = il.DefineLabel();
                var lblJOk = il.DefineLabel();

                // i = 0; j = 0
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locJ);

                il.MarkLabel(lblHead);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_2);            // maskSize
                il.Emit(OpCodes.Bge, lblEnd);

                // if (mask[i] == 0) goto skip
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Brfalse, lblSkip);

                // srcPtr = values + j * elemBytes
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locSrcPtr);

                // dstPtr = dst + i * elemBytes
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locDstPtr);

                EmitElementCopy(il, copyKind, locDstPtr, locSrcPtr, () => il.Emit(OpCodes.Ldarg, 5));

                il.MarkLabel(lblSkip);

                // j++; if (j >= valuesCount) j = 0;   (advance the position cursor every iteration)
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locJ);
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldarg, 4);           // valuesCount
                il.Emit(OpCodes.Blt, lblJOk);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locJ);
                il.MarkLabel(lblJOk);

                // i++
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblHead);
            }

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);

            return (PutMaskKernel)dm.CreateDelegate(typeof(PutMaskKernel));
        }
    }
}
