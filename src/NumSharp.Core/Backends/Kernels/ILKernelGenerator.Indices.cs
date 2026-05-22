using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// ILKernelGenerator.Indices.cs — IL kernel for np.indices
// =============================================================================
//
// RESPONSIBILITY:
//   np.indices((D0, D1, …, Dn-1)) returns an (ndim, D0, D1, …, Dn-1) int64
//   array where result[d, i0, i1, …] = i_d. Each "slab" result[d, …] is
//   structurally a tiled arange(dims[d]) broadcast across the inner-axis
//   strides — we fill it via blockwise SIMD memsets, avoiding the per-element
//   divmod that a naive unravel approach would pay.
//
//   For axis d in C-order:
//     * inner = dimStrides[d]  (= prod(dims[d+1:]))
//     * For each tile in [0, prod / (dims[d] * inner)):
//         For v in [0, dims[d]):
//           Write `inner` consecutive copies of v.
//
// KERNEL (DynamicMethod-emitted, singleton):
//   * IndicesKernel
//       (long* result,       // contig int64 buffer of length ndim * prod
//        long* dimStrides,   // C-order strides (ndim entries)
//        long* dims,
//        long ndim,
//        long prod)
//
//   The inner-fill loop uses Vector{N}<long>.Create(v).Store for chunks of
//   <see cref="ILKernelGenerator.VectorBytes"/> / 8 longs, with a scalar tail
//   for the leftover. For s == 1 (innermost axis) the SIMD chunk never
//   triggers and we run the scalar tail end-to-end.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted slab filler for <c>np.indices</c>. Writes the multi-axis
    /// coordinate of each output position directly via blockwise SIMD memsets;
    /// no per-element divmod or coord advance.
    /// </summary>
    public unsafe delegate void IndicesKernel(
        long* result, long* dimStrides, long* dims, long ndim, long prod);

    public static partial class ILKernelGenerator
    {
        private static IndicesKernel _indicesKernel;

        /// <summary>
        /// IL-emitted indices fill kernel (singleton — same kernel handles any ndim).
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static IndicesKernel GetIndicesKernel()
        {
            if (!Enabled)
                return null;

            var cached = _indicesKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateIndicesKernelIL();
                Interlocked.CompareExchange(ref _indicesKernel, k, null);
                return _indicesKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetIndicesKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the indices kernel. Pseudocode (LANES = <see cref="VectorBits"/> / 64):
        /// <code>
        /// void Fill(long* result, long* dimStrides, long* dims, long ndim, long prod) {
        ///     for (long d = 0; d &lt; ndim; d++) {
        ///         long* slab = result + d * prod;
        ///         long s = dimStrides[d];
        ///         long m = dims[d];
        ///         for (long f = 0; f &lt; prod; f += m * s) {
        ///             for (long v = 0; v &lt; m; v++) {
        ///                 long* writePtr = slab + f + v * s;
        ///                 long k = 0;
        ///                 // SIMD chunk
        ///                 for (; k + LANES &lt;= s; k += LANES)
        ///                     V&lt;long&gt;.Create(v).Store(writePtr + k);
        ///                 // Scalar tail
        ///                 for (; k &lt; s; k++)
        ///                     writePtr[k] = v;
        ///             }
        ///         }
        ///     }
        /// }
        /// </code>
        /// For s == 1 (innermost axis) the SIMD chunk never fires; the kernel falls
        /// through to a tight scalar loop that writes one long per cycle.
        /// </summary>
        private static IndicesKernel GenerateIndicesKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_Indices",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(long*),  // 0 result
                    typeof(long*),  // 1 dimStrides
                    typeof(long*),  // 2 dims
                    typeof(long),   // 3 ndim
                    typeof(long),   // 4 prod
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            int simdBits = VectorBits >= 128 ? VectorBits : 0;
            int lanes = simdBits > 0 ? simdBits / 64 : 0;   // long is 8 bytes = 64 bits

            var locD = il.DeclareLocal(typeof(long));
            var locSlab = il.DeclareLocal(typeof(long*));
            var locS = il.DeclareLocal(typeof(long));
            var locM = il.DeclareLocal(typeof(long));
            var locPeriod = il.DeclareLocal(typeof(long));
            var locF = il.DeclareLocal(typeof(long));
            var locV = il.DeclareLocal(typeof(long));
            var locWritePtr = il.DeclareLocal(typeof(long*));
            var locK = il.DeclareLocal(typeof(long));
            var locSimdEnd = il.DeclareLocal(typeof(long));

            var lblDHead = il.DefineLabel();
            var lblDEnd = il.DefineLabel();
            var lblFHead = il.DefineLabel();
            var lblFEnd = il.DefineLabel();
            var lblVHead = il.DefineLabel();
            var lblVEnd = il.DefineLabel();
            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblTailHead = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // d = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locD);

            // ----- Outer loop over axes -----
            il.MarkLabel(lblDHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, lblDEnd);

            // slab = result + d * prod * 8
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSlab);

            // s = dimStrides[d]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locS);

            // m = dims[d]
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locM);

            // period = m * s (hoisted)
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locPeriod);

            // simdEnd = s - LANES (only valid if simdBits > 0)
            if (simdBits > 0)
            {
                il.Emit(OpCodes.Ldloc, locS);
                il.Emit(OpCodes.Ldc_I8, (long)lanes);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locSimdEnd);
            }

            // f = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locF);

            // ----- Tile loop -----
            il.MarkLabel(lblFHead);
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Bge, lblFEnd);

            // v = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locV);

            // ----- Value loop -----
            il.MarkLabel(lblVHead);
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Bge, lblVEnd);

            // writePtr = slab + (f + v*s) * 8
            il.Emit(OpCodes.Ldloc, locSlab);
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locWritePtr);

            // k = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            // ----- SIMD chunk loop -----
            if (simdBits > 0)
            {
                il.MarkLabel(lblSimdHead);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldloc, locSimdEnd);
                il.Emit(OpCodes.Bgt, lblSimdEnd);

                // V<long>.Create(v).Store(writePtr + k*8)
                il.Emit(OpCodes.Ldloc, locV);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(simdBits, typeof(long)), null);
                il.Emit(OpCodes.Ldloc, locWritePtr);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, 8L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Store(simdBits, typeof(long)), null);

                // k += LANES
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, (long)lanes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, lblSimdHead);

                il.MarkLabel(lblSimdEnd);
            }

            // ----- Scalar tail -----
            il.MarkLabel(lblTailHead);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Bge, lblTailEnd);

            // writePtr[k] = v
            il.Emit(OpCodes.Ldloc, locWritePtr);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Stind_I8);

            // k++
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblTailHead);

            il.MarkLabel(lblTailEnd);

            // v++
            il.Emit(OpCodes.Ldloc, locV);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locV);
            il.Emit(OpCodes.Br, lblVHead);

            il.MarkLabel(lblVEnd);

            // f += period
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldloc, locPeriod);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locF);
            il.Emit(OpCodes.Br, lblFHead);

            il.MarkLabel(lblFEnd);

            // d++
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblDHead);

            il.MarkLabel(lblDEnd);
            il.Emit(OpCodes.Ret);

            return (IndicesKernel)dm.CreateDelegate(typeof(IndicesKernel));
        }
    }
}
