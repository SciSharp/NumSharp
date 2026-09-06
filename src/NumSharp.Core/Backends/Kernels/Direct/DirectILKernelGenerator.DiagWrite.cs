using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.DiagWrite.cs — IL kernel for the strided diagonal write
// =============================================================================
//
// RESPONSIBILITY:
//   The construction/fill half of the diagonal family — np.fill_diagonal,
//   np.diag(1-D), np.diagflat — all reduce to the SAME primitive: write `count`
//   elements from a (contiguous, strided, or broadcast) 1-D source into an
//   equally-strided run of a destination buffer. The positions are computed from
//   real strides (offset + i*stride), so this works unchanged through transposed,
//   sliced, F-order and negative-stride destinations exactly as the aliased-view
//   path it replaces did.
//
//   The old path aliased a strided Shape over the buffer and called SetData, which
//   routes a strided destination through NDIter.Copy. That iterator carries real
//   per-strided-write overhead the direct store avoids — measured (JIT, warm target):
//   fill_diagonal at 3162 x 3162 is 37.9 us via SetData vs 10.9 us via this kernel
//   (3.5x, and 0.40x -> 1.46x vs NumPy); diag/diagflat construct at 3162 x 3162
//   3.5 ms -> 2.4 ms. This kernel is that direct store.
//
// KERNEL (DynamicMethod-emitted, one per copy-width, cached forever):
//
//   * DiagWriteKernel
//       (byte* dst,            // destination FIRST target (base + offset already folded in)
//        long  dstByteStride,  // signed byte stride between diagonal targets
//        byte* src,            // source FIRST element (base + offset already folded in)
//        long  srcByteStride,  // signed byte stride between source elements; 0 = broadcast a scalar
//        long  count)          // number of elements to write
//       -> void
//
//   Running-pointer loop (no per-element multiply):
//
//       for (i = 0; i < count; i++) {
//           copy_one(dst, src);       // typed MOV(s) for the baked width
//           dst += dstByteStride;
//           src += srcByteStride;      // 0 for a scalar/broadcast source
//       }
//
//   copyKind (the CopyKindFor of the dtype itemsize) is baked at emit time, so the
//   element copy is a typed load+store (1/2/4/8 byte) or two 8-byte MOVs (16-byte
//   Complex/Decimal) — never a runtime-sized cpblk. Every NumSharp dtype has an
//   itemsize in {1,2,4,8,16}, so copyKind is always one of those five and the
//   fill covers all 15 dtypes with no per-dtype switch.
//
//   The write is always same-dtype (diag/diagflat preserve the source dtype;
//   fill_diagonal casts its values to the destination dtype first), so no cast is
//   ever needed in the loop.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted strided store: writes <c>dst[i*dstStride] = src[i*srcStride]</c>
    /// for each <c>i</c> in <c>[0, count)</c>. A <paramref name="srcByteStride"/> of
    /// 0 broadcasts a single source element (scalar fill). Pointers are pre-offset
    /// by the caller; strides are in bytes and may be negative.
    /// </summary>
    public unsafe delegate void DiagWriteKernel(
        byte* dst, long dstByteStride,
        byte* src, long srcByteStride, long count);

    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<int, DiagWriteKernel> _diagWriteKernels = new();

        /// <summary>
        /// IL-emitted diagonal-write kernel for a given copy-width
        /// (<paramref name="copyKind"/> — the <see cref="CopyKindFor"/> of the dtype
        /// itemsize; always 1/2/4/8/16 for real dtypes, so the store is a typed MOV
        /// rather than a per-element cpblk). Returns <c>null</c> when
        /// <see cref="Enabled"/> is false or emission fails, so callers can fall back
        /// to the SetData path.
        /// </summary>
        public static DiagWriteKernel GetDiagWriteKernel(int copyKind)
        {
            if (!Enabled)
                return null;

            if (_diagWriteKernels.TryGetValue(copyKind, out var cached))
                return cached;

            try
            {
                var k = GenerateDiagWriteKernelIL(copyKind);
                return _diagWriteKernels.GetOrAdd(copyKind, k);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetDiagWriteKernel({copyKind}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static DiagWriteKernel GenerateDiagWriteKernelIL(int copyKind)
        {
            var dm = new DynamicMethod(
                name: $"IL_DiagWrite_c{copyKind}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 dst   (already offset to first target)
                    typeof(long),   // 1 dstByteStride
                    typeof(byte*),  // 2 src   (already offset to first element)
                    typeof(long),   // 3 srcByteStride (0 = broadcast)
                    typeof(long),   // 4 count
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locDst = il.DeclareLocal(typeof(byte*));
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locI = il.DeclareLocal(typeof(long));

            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // locDst = dst; locSrc = src; i = 0
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, locDst);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ----- for (i = 0; i < count; i++) -----
            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg, 4);            // count
            il.Emit(OpCodes.Bge, lblEnd);

            // copy_one(locDst, locSrc). copyKind is always in {1,2,4,8,16} for real
            // dtypes; the pushByteCount closure is a defensive no-op the switch never
            // reaches here.
            EmitElementCopy(il, copyKind, locDst, locSrc, () => il.Emit(OpCodes.Ldc_I4_0));

            // locDst += dstByteStride
            il.Emit(OpCodes.Ldloc, locDst);
            il.Emit(OpCodes.Ldarg_1);             // dstByteStride
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            // locSrc += srcByteStride   (0 for a broadcast/scalar source → pointer parks)
            il.Emit(OpCodes.Ldloc, locSrc);
            il.Emit(OpCodes.Ldarg_3);             // srcByteStride
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);

            return (DiagWriteKernel)dm.CreateDelegate(typeof(DiagWriteKernel));
        }
    }
}
