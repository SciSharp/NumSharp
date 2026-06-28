using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.UnravelIndex.cs — IL kernel for np.unravel_index
// =============================================================================
//
// RESPONSIBILITY:
//   np.unravel_index converts an arbitrary array of flat indices into a tuple
//   of per-axis coordinate arrays. The per-element work is `ndim` divmods,
//   which is divmod-bound regardless of layout. The kernel is dtype-agnostic
//   (input is cast to int64 by the caller) — a single DynamicMethod handles
//   any ndim + both C / F order.
//
//   Structural twin of <see cref="NonZeroPerDimKernel"/>, but without the
//   monotonic-index optimization: the input indices are arbitrary, so we
//   cannot use the carry-chain incremental advance — every element pays
//   `ndim` divmods.
//
// KERNEL (DynamicMethod-emitted, singleton):
//
//   * UnravelIndexKernel
//       (long* indices,        // contig int64 buffer (caller casts)
//        long count,
//        long* dims,           // shape, ndim entries
//        long unravelSize,     // product of dims, for OOB validation
//        long** outCols,       // ndim per-axis output buffers
//        long ndim,
//        long idxStart,        // ndim-1 for C-order, 0 for F-order
//        long idxStep)         // -1 for C-order, +1 for F-order
//        -> long: count on success, else the index of the first OOB element.
//
//   Caller reads indices[returned_index] to produce NumPy's diagnostic
//   "index N is out of bounds for array with size M" message.
//
// OPTIMIZATION: SKIP LAST DIVMOD
//   After ndim-1 divmods, `val < dims[final_idx]`, so coords[final_idx] = val
//   directly. Saves 1/ndim of the divmod cost — a 50% reduction for ndim==2,
//   33% for ndim==3, …
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted flat→multi-coord expander for <c>np.unravel_index</c>.
    /// Caller pre-casts indices to int64 and computes <paramref name="unravelSize"/>
    /// = product of <paramref name="dims"/>; the kernel reads <paramref name="indices"/>
    /// linearly, validates each value against <c>[0, unravelSize)</c>, and emits
    /// per-axis coords into <paramref name="outCols"/> using the C-order or F-order
    /// extraction direction selected by <paramref name="idxStart"/> /
    /// <paramref name="idxStep"/>.
    /// </summary>
    /// <returns>
    /// <paramref name="count"/> on success. On failure, the row index of the first
    /// element with <c>val &lt; 0</c> or <c>val &gt;= unravelSize</c>; the caller
    /// reads <c>indices[returned]</c> to produce the error message.
    /// </returns>
    public unsafe delegate long UnravelIndexKernel(
        long* indices, long count, long* dims, long unravelSize,
        long** outCols, long ndim, long idxStart, long idxStep);

    public static partial class DirectILKernelGenerator
    {
        private static UnravelIndexKernel _unravelIndexKernel;

        /// <summary>
        /// IL-emitted unravel kernel (singleton — same kernel handles any ndim and
        /// both C / F order via the runtime <c>idxStart</c> / <c>idxStep</c> args).
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static UnravelIndexKernel GetUnravelIndexKernel()
        {
            if (!Enabled)
                return null;

            var cached = _unravelIndexKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateUnravelIndexKernelIL();
                Interlocked.CompareExchange(ref _unravelIndexKernel, k, null);
                return _unravelIndexKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetUnravelIndexKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the unravel kernel. Pseudocode:
        /// <code>
        /// long Unravel(long* indices, long count, long* dims, long unravelSize,
        ///              long** outCols, long ndim, long idxStart, long idxStep) {
        ///     long ndimMinusOne = ndim - 1;
        ///     for (long i = 0; i &lt; count; i++) {
        ///         long val = indices[i];
        ///         if (val &lt; 0 || val &gt;= unravelSize) return i;
        ///         long idx = idxStart;
        ///         for (long k = 0; k &lt; ndimMinusOne; k++) {
        ///             long m = dims[idx];
        ///             long* col = outCols[idx];
        ///             col[i] = val % m;
        ///             val = val / m;
        ///             idx += idxStep;
        ///         }
        ///         // Last coord: val &lt; dims[idx] already, no divmod needed.
        ///         long* lastCol = outCols[idx];
        ///         lastCol[i] = val;
        ///     }
        ///     return count;
        /// }
        /// </code>
        /// </summary>
        private static UnravelIndexKernel GenerateUnravelIndexKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_UnravelIndex",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(long*),  // 0 indices
                    typeof(long),   // 1 count
                    typeof(long*),  // 2 dims
                    typeof(long),   // 3 unravelSize
                    typeof(long**), // 4 outCols
                    typeof(long),   // 5 ndim
                    typeof(long),   // 6 idxStart
                    typeof(long),   // 7 idxStep
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locVal = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locK = il.DeclareLocal(typeof(long));
            var locM = il.DeclareLocal(typeof(long));
            var locCol = il.DeclareLocal(typeof(long*));
            var locNdimMinusOne = il.DeclareLocal(typeof(long));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();

            // ndimMinusOne = ndim - 1 (hoisted)
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locNdimMinusOne);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ----- Outer loop: for (i = 0; i < count; i++) -----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);   // count
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // val = indices[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locVal);

            // if (val < 0) goto fail
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblFail);

            // if (val >= unravelSize) goto fail
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, lblFail);

            // idx = idxStart
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Stloc, locIdx);

            // k = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            // ----- Inner loop: for (k = 0; k < ndim - 1; k++) -----
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locNdimMinusOne);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // m = dims[idx]
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locM);

            // col = outCols[idx]
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locCol);

            // col[i] = val % m
            il.Emit(OpCodes.Ldloc, locCol);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stind_I8);

            // val = val / m
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locVal);

            // idx += idxStep
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdx);

            // k++
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // ----- Last coord: outCols[idx][i] = val (skip divmod) -----
            // col = outCols[idx]
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locCol);

            // col[i] = val
            il.Emit(OpCodes.Ldloc, locCol);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locVal);
            il.Emit(OpCodes.Stind_I8);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ----- Fail path: return i -----
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ret);

            // ----- Success: return count -----
            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ret);

            return (UnravelIndexKernel)dm.CreateDelegate(typeof(UnravelIndexKernel));
        }
    }
}
