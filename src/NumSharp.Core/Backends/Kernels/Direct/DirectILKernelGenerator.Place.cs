using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.Place.cs — IL kernel for np.place
// =============================================================================
//
// RESPONSIBILITY:
//   np.place scatters values into a target array at positions where a boolean
//   mask is True. Values cycle if shorter than the True count. No mode handling
//   (positions are determined by the mask, not by integer indices).
//
//   The kernel walks the mask byte-by-byte (NumSharp stores bools as 1-byte),
//   advancing a separate values cursor that wraps modulo valuesCount. Each
//   True triggers an inline <c>cpblk</c> of <c>elemBytes</c> bytes.
//
// KERNEL (DynamicMethod-emitted, singleton):
//
//   * PlaceKernel
//       (byte* dst,            // target buffer
//        byte* mask,           // contig bool mask (1 byte each)
//        long maskSize,
//        byte* values,         // contig source buffer
//        long valuesCount,     // > 0; caller short-circuits when 0
//        long elemBytes)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted mask-driven scatter for <c>np.place</c>. For each <c>i</c> in
    /// <c>[0, maskSize)</c> with <c>mask[i]</c> true, writes
    /// <c>dst[i] = values[j % valuesCount]</c> and advances <c>j</c>.
    /// </summary>
    public unsafe delegate void PlaceKernel(
        byte* dst, byte* mask, long maskSize,
        byte* values, long valuesCount, long elemBytes);

    public static partial class DirectILKernelGenerator
    {
        private static PlaceKernel _placeKernel;

        /// <summary>
        /// IL-emitted place kernel (singleton — same kernel handles any dtype
        /// via the <c>elemBytes</c> runtime argument). Returns <c>null</c> only
        /// when <see cref="Enabled"/> is false.
        /// </summary>
        public static PlaceKernel GetPlaceKernel()
        {
            if (!Enabled)
                return null;

            var cached = _placeKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GeneratePlaceKernelIL();
                Interlocked.CompareExchange(ref _placeKernel, k, null);
                return _placeKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPlaceKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the place kernel. Pseudocode:
        /// <code>
        /// void Place(byte* dst, byte* mask, long maskSize,
        ///            byte* values, long nv, long elemBytes) {
        ///     long j = 0;
        ///     for (long i = 0; i &lt; maskSize; i++) {
        ///         if (mask[i] != 0) {
        ///             byte* srcPtr = values + (j % nv) * elemBytes;
        ///             byte* dstPtr = dst + i * elemBytes;
        ///             cpblk(dstPtr, srcPtr, elemBytes);
        ///             j++;
        ///         }
        ///     }
        /// }
        /// </code>
        /// </summary>
        private static PlaceKernel GeneratePlaceKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_Place",
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

            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblSkip = il.DefineLabel();

            // i = 0; j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblEnd);

            // if (mask[i] == 0) goto skip
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblSkip);

            // srcPtr = values + (j % valuesCount) * elemBytes
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Rem);
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

            // cpblk(dstPtr, srcPtr, elemBytes)
            il.Emit(OpCodes.Ldloc, locDstPtr);
            il.Emit(OpCodes.Ldloc, locSrcPtr);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Conv_U4);
            il.Emit(OpCodes.Cpblk);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);

            il.MarkLabel(lblSkip);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);

            return (PlaceKernel)dm.CreateDelegate(typeof(PlaceKernel));
        }
    }
}
