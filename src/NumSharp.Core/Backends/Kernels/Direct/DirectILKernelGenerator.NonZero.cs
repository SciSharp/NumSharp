using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.NonZero.cs — IL-emitted expand kernel for np.nonzero
// =============================================================================
//
// RESPONSIBILITY:
//   np.nonzero shares the per-dtype count + flat-scan IL kernels with np.argwhere
//   (ArgwhereCountKernel, ArgwhereFlatKernel) — the only piece that differs is the
//   "expand flat → coords" stage:
//     * argwhere writes a row-major (count, ndim) matrix
//     * nonzero writes ndim separate (count,) column arrays
//
//   This file emits the column-layout expand kernel:
//
//   * NonZeroPerDimKernel (long* flat, long count, long* dims,
//                          long* dimStrides, long ndim, long** outCols)
//       Dtype-agnostic singleton. Single DM that incrementally advances a
//       stack-allocated coord buffer, propagating the carry chain through
//       the outer dims, and writes coords[d] into outCols[d][i] each step.
//
// CACHE:
//   Singleton field _nonZeroPerDimKernel populated lazily on first use via
//   Interlocked.CompareExchange.
//
// BOOL-ONLY KERNELS (removed):
//   The previous IsAllZeroBoolKernel / NonZeroCountBoolKernel / NonZeroFlatBoolKernel
//   trio existed because the original np.nonzero contig path branched on
//   `typeof(T) == typeof(bool)` and only had a bool-specific IL implementation.
//   The argwhere refactor introduced per-dtype IL kernels (Argwhere*) that cover
//   all 15 dtypes including bool (via byte reinterpretation), so the bool-only
//   kernels are dead code. They were deleted along with the dtype branch.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted coord expand for <c>np.nonzero</c>: converts a flat-index buffer
    /// (monotonic ascending C-order) into <c>ndim</c> separate per-dim coordinate
    /// columns via incremental coord advance — no per-element divmod.
    /// Dtype-agnostic (operates on long*).
    /// <para>
    /// <paramref name="outCols"/> is a pointer to an array of <c>ndim</c>
    /// <c>long*</c> pointers, one per output dimension. <c>outCols[d][i]</c> receives
    /// the coordinate along dim <c>d</c> for the <c>i</c>'th non-zero element.
    /// </para>
    /// </summary>
    public unsafe delegate void NonZeroPerDimKernel(
        long* flat, long count, long* dims, long* dimStrides, long ndim, long** outCols);

    public static partial class DirectILKernelGenerator
    {
        #region Cached delegate (lazy init)

        private static NonZeroPerDimKernel _nonZeroPerDimKernel;

        /// <summary>
        /// IL-emitted per-dim coord expander (singleton — same kernel handles any ndim).
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static NonZeroPerDimKernel GetNonZeroPerDimKernel()
        {
            if (!Enabled)
                return null;

            var cached = _nonZeroPerDimKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateNonZeroPerDimKernelIL();
                Interlocked.CompareExchange(ref _nonZeroPerDimKernel, k, null);
                return _nonZeroPerDimKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetNonZeroPerDimKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        #endregion

        #region Per-dim kernel IL emission

        /// <summary>
        /// Emits the per-dim expand kernel. Pseudocode:
        /// <code>
        /// void Expand(long* flat, long count, long* dims, long* dimStrides, long ndim, long** outCols) {
        ///     long[ndim] coords = stackalloc;
        ///
        ///     // Seed coords from flat[0] via one divmod chain.
        ///     long f = flat[0];
        ///     for (long d = 0; d &lt; ndim; d++) {
        ///         long s = dimStrides[d];
        ///         coords[d] = f / s; f %= s;
        ///     }
        ///     // Write column 0 row 0.
        ///     for (long d = 0; d &lt; ndim; d++) outCols[d][0] = coords[d];
        ///
        ///     long lastFlat = flat[0];
        ///     long innerSize = dims[ndim - 1];
        ///     for (long i = 1; i &lt; count; i++) {
        ///         long fi = flat[i];
        ///         long delta = fi - lastFlat; lastFlat = fi;
        ///         long newInner = coords[ndim - 1] + delta;
        ///         if (newInner &lt; innerSize) {
        ///             coords[ndim - 1] = newInner;
        ///         } else {
        ///             long carry = newInner / innerSize;
        ///             coords[ndim - 1] = newInner % innerSize;
        ///             for (long d = ndim - 2; d &gt;= 0 &amp;&amp; carry &gt; 0; d--) {
        ///                 long sum = coords[d] + carry;
        ///                 if (sum &lt; dims[d]) { coords[d] = sum; carry = 0; }
        ///                 else { coords[d] = sum % dims[d]; carry = sum / dims[d]; }
        ///             }
        ///         }
        ///         for (long d = 0; d &lt; ndim; d++) outCols[d][i] = coords[d];
        ///     }
        /// }
        /// </code>
        ///
        /// The structural twin of <see cref="ArgwhereExpandKernel"/>; the only difference
        /// is the destination layout in the write step — argwhere writes a row-major
        /// (count, ndim) matrix, nonzero writes ndim per-dim columns indexed by the
        /// row index.
        /// </summary>
        private static NonZeroPerDimKernel GenerateNonZeroPerDimKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_NonZeroPerDim",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(long*), typeof(long), typeof(long*), typeof(long*), typeof(long), typeof(long**) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locCoords = il.DeclareLocal(typeof(long*));    // stackalloc'd coord buffer
            var locF = il.DeclareLocal(typeof(long));
            var locS = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));
            var locI = il.DeclareLocal(typeof(long));
            var locInnerSize = il.DeclareLocal(typeof(long));
            var locLastFlat = il.DeclareLocal(typeof(long));
            var locFi = il.DeclareLocal(typeof(long));
            var locDelta = il.DeclareLocal(typeof(long));
            var locNewInner = il.DeclareLocal(typeof(long));
            var locCarry = il.DeclareLocal(typeof(long));
            var locSum = il.DeclareLocal(typeof(long));

            // --- coords = stackalloc long[ndim] ---
            il.Emit(OpCodes.Ldarg, 4);            // ndim
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoords);

            // --- Seed: f = flat[0] ---
            il.Emit(OpCodes.Ldarg_0);             // flat
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locF);
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Stloc, locLastFlat);

            // --- for (d = 0; d < ndim; d++) coords[d] = f / dimStrides[d]; f %= dimStrides[d]; ---
            var lblSeedHead = il.DefineLabel();
            var lblSeedEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locD);

            il.MarkLabel(lblSeedHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Bge, lblSeedEnd);

            // s = dimStrides[d]
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locS);

            // coords[d] = f / s
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stind_I8);

            // f = f % s
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locF);

            // d++
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblSeedHead);

            il.MarkLabel(lblSeedEnd);

            // --- Write row 0 to per-dim columns: for (d = 0; d < ndim; d++) outCols[d][0] = coords[d] ---
            EmitWritePerDimRow(il, /*rowIndexLocal=*/ null, locCoords);

            // --- innerSize = dims[ndim - 1] ---
            il.Emit(OpCodes.Ldarg_2);                        // dims
            il.Emit(OpCodes.Ldarg, 4);                       // ndim
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerSize);

            // --- Outer loop: for (i = 1; i < count; i++) ---
            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // fi = flat[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locFi);

            // delta = fi - lastFlat
            il.Emit(OpCodes.Ldloc, locFi);
            il.Emit(OpCodes.Ldloc, locLastFlat);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locDelta);

            // lastFlat = fi
            il.Emit(OpCodes.Ldloc, locFi);
            il.Emit(OpCodes.Stloc, locLastFlat);

            // newInner = coords[ndim-1] + delta
            var lblInnerNoOverflow = il.DefineLabel();
            var lblAdvanceEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldloc, locDelta);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locNewInner);

            // if (newInner < innerSize) goto inner_no_overflow
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Blt, lblInnerNoOverflow);

            // --- Overflow path: carry chain ---
            // carry = newInner / innerSize; coords[ndim-1] = newInner % innerSize
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locCarry);

            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stind_I8);

            // for (d = ndim - 2; d >= 0 && carry > 0; d--) { ... }
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            var lblCarryHead = il.DefineLabel();
            var lblCarryEnd = il.DefineLabel();

            il.MarkLabel(lblCarryHead);
            // if (d < 0) break
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblCarryEnd);
            // if (carry == 0) break
            il.Emit(OpCodes.Ldloc, locCarry);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblCarryEnd);

            // sum = coords[d] + carry
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldloc, locCarry);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSum);

            // axisSize = dims[d] on the stack for the comparison
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);

            // if (sum <= axisSize) sum < axisSize is the no-overflow predicate; Ble(axisSize, sum) jumps when axisSize <= sum, i.e. sum >= axisSize → overflow.
            var lblCarryOverflow = il.DefineLabel();
            var lblCarryStep = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ble, lblCarryOverflow);

            // sum < axisSize → assign + zero carry
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locCarry);
            il.Emit(OpCodes.Br, lblCarryStep);

            // sum >= axisSize → recompute coords[d] = sum % dims[d] and carry = sum / dims[d].
            // Ble already consumed (axisSize, sum) from the stack.
            il.MarkLabel(lblCarryOverflow);

            // coords[d] = sum % dims[d]
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stind_I8);

            // carry = sum / dims[d]
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locCarry);

            il.MarkLabel(lblCarryStep);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblCarryHead);

            il.MarkLabel(lblCarryEnd);
            il.Emit(OpCodes.Br, lblAdvanceEnd);

            // --- No overflow path: just store newInner into innermost coord ---
            il.MarkLabel(lblInnerNoOverflow);
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Stind_I8);

            il.MarkLabel(lblAdvanceEnd);

            // --- Write row i to per-dim columns: for (d = 0; d < ndim; d++) outCols[d][i] = coords[d] ---
            EmitWritePerDimRow(il, locI, locCoords);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuterHead);

            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ret);

            return (NonZeroPerDimKernel)dm.CreateDelegate(typeof(NonZeroPerDimKernel));
        }

        /// <summary>
        /// Emits an inner IL loop that copies the current <c>coords</c> array into
        /// the per-dim output columns at row index <paramref name="rowIndexLocal"/>.
        /// When <paramref name="rowIndexLocal"/> is <c>null</c> the row index is 0
        /// (used for the seed write before the main loop).
        /// </summary>
        private static void EmitWritePerDimRow(ILGenerator il, LocalBuilder rowIndexLocal, LocalBuilder locCoords)
        {
            var locDw = il.DeclareLocal(typeof(long));
            var locColPtr = il.DeclareLocal(typeof(long*));
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locDw);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldarg, 4); // ndim
            il.Emit(OpCodes.Bge, lblEnd);

            // colPtr = outCols[dw]  i.e.  *(long**)(outCols + dw*sizeof(long*))
            // sizeof(long*) is 8 on x64; the entire codebase targets 64-bit runtimes
            // (UnmanagedStorage uses long for byte sizes etc.). Using 8 here mirrors
            // the rest of the kernel's pointer arithmetic.
            il.Emit(OpCodes.Ldarg, 5);   // outCols (long**)
            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);    // dereference: now have outCols[dw] (long*)
            il.Emit(OpCodes.Stloc, locColPtr);

            // dest = colPtr + rowIndex * 8 (or colPtr for row 0)
            il.Emit(OpCodes.Ldloc, locColPtr);
            if (rowIndexLocal != null)
            {
                il.Emit(OpCodes.Ldloc, rowIndexLocal);
                il.Emit(OpCodes.Ldc_I8, 8L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            // value = coords[dw]
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);

            il.Emit(OpCodes.Stind_I8);

            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDw);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        #endregion
    }
}
