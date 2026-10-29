using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.TakeLeading — the gather behind NumPy's fancy index along a NON-LEADING axis
// =============================================================================
//
// NumPy's a[(slice(None),)*axis + (indices,)] (mapping.c, MapIter) does not return a C-ordered array: it fills a fresh
// buffer whose FIRST axis is the indexed one, the remaining axes following in their order, and returns that buffer
// transposed back into a's axis order (PyArray_MapIterSwapAxes) — a view. np.take's kernel writes the C-ordered result
// instead, so producing NumPy's layout from it needs a second (transposing) pass. This kernel writes NumPy's buffer
// directly, in one pass:
//
//     dst[(j * outer + o) * inner .. +inner] = src[(o * maxItem + indices[j]) * inner .. +inner]
//
// for o in [0, outer), j in [0, indicesCount): src is laid out (outer, maxItem, inner bytes) C-contiguous, dst
// (indicesCount, outer, inner bytes) C-contiguous.
//
// Loop order (measured, 2026-10-05): rows of src in tiles of `tileRows`; inside a tile, for every index j, the tile's
// slabs are copied as ONE contiguous write run of dst. Each tile's source rows (tileRows * maxItem * inner bytes, sized
// by the caller to ~1 MB) stay cache-resident while every index passes over them, so every source line is fetched about
// once, and every write run is contiguous. MapIter's own order (index outer, all rows inner) re-reads source lines from
// further out; per index it walks every row. The typed-copy inner loop is unrolled by 4 (2-3x faster than the plain loop
// on 1-8 byte slabs, where the loop overhead is the cost).
//
// Indices are NOT bounds-checked: the caller guarantees indices[j] in [0, maxItem) (Generator.permutation passes a
// permutation of [0, maxItem)). np.take's kernel is the checked one.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     The fancy-index gather along a non-leading axis, in NumPy's MapIter buffer order: for every index
    ///     <c>j</c> and outer row <c>o</c>, copies <paramref name="innerSize"/> bytes from
    ///     <c>src[(o * maxItem + indices[j]) * innerSize]</c> to <c>dst[(j * outerSize + o) * innerSize]</c>.
    /// </summary>
    /// <param name="src">The source, laid out (outerSize, maxItem, innerSize bytes) C-contiguous.</param>
    /// <param name="indices">int64 indices, each already in <c>[0, maxItem)</c> — not checked here.</param>
    /// <param name="indicesCount">The number of indices (the leading dimension of dst).</param>
    /// <param name="outerSize">The product of the source dims before the indexed axis.</param>
    /// <param name="maxItem">The extent of the indexed axis in the source.</param>
    /// <param name="innerSize">Bytes per slab (element bytes times the dims after the indexed axis).</param>
    /// <param name="tileRows">Source rows processed per tile (at least 1); sizes the cache-resident read window.</param>
    /// <param name="dst">The destination, (indicesCount, outerSize, innerSize bytes) C-contiguous.</param>
    public unsafe delegate void TakeLeadingKernel(
        byte* src, long* indices, long indicesCount, long outerSize,
        long maxItem, long innerSize, long tileRows, byte* dst);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>One compiled gather per copy width (the <see cref="CopyKindFor"/> of the slab size).</summary>
        private static readonly ConcurrentDictionary<int, TakeLeadingKernel> _takeLeadingKernels = new();

        /// <summary>
        ///     The IL gather writing NumPy's fancy-index buffer order (indexed axis first) in one pass, compiled once per
        ///     copy width: <paramref name="copyKind"/> 1/2/4/8/16 copies each slab with typed MOVs in a 4x-unrolled
        ///     loop; 0 copies runtime-sized slabs with <c>cpblk</c> (large slabs, where the loop overhead is moot).
        /// </summary>
        /// <param name="copyKind">The <see cref="CopyKindFor"/> of the slab size in bytes.</param>
        /// <returns>The kernel, or <c>null</c> when IL generation is switched off (<see cref="Enabled"/> false) or the
        ///     emission failed — the caller then falls back to <c>np.take</c> over a moved view, which builds the same
        ///     buffer after a failed emission but needs an IL kernel of its own (with <see cref="Enabled"/> false it
        ///     raises <see cref="NotSupportedException"/>).</returns>
        public static TakeLeadingKernel GetTakeLeadingKernel(int copyKind)
        {
            if (!Enabled)
                return null;
            if (_takeLeadingKernels.TryGetValue(copyKind, out var cached))
                return cached;

            try
            {
                return _takeLeadingKernels.GetOrAdd(copyKind, GenerateTakeLeadingKernelIL(copyKind));
            }
            catch (Exception ex)
            {
                // A failed emission must not take the caller down: it has a correct (slower) fallback.
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTakeLeadingKernel({copyKind}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits the tiled gather. Pseudocode:
        ///     <code>
        ///     void TakeLeading(byte* src, long* indices, long m, long outer, long maxItem, long inner, long tile, byte* dst) {
        ///         long rowStep = maxItem * inner;
        ///         for (long ob = 0; ob &lt; outer; ob += tile) {
        ///             long count = min(tile, outer - ob);
        ///             for (long j = 0; j &lt; m; j++) {
        ///                 byte* s = src + (ob * maxItem + indices[j]) * inner;
        ///                 byte* d = dst + (j * outer + ob) * inner;
        ///                 long k = 0;
        ///                 for (; k + 4 &lt;= count; k += 4) {                       // typed widths only
        ///                     copy(d, s); copy(d + inner, s + rowStep);
        ///                     copy(d + 2*inner, s + 2*rowStep); copy(d + 3*inner, s + 3*rowStep);
        ///                     s += 4 * rowStep; d += 4 * inner;
        ///                 }
        ///                 for (; k &lt; count; k++) { copy(d, s); s += rowStep; d += inner; }
        ///             }
        ///         }
        ///     }
        ///     </code>
        /// </summary>
        /// <param name="copyKind">1/2/4/8/16 for a typed slab copy (then <c>inner == copyKind</c>), 0 for <c>cpblk</c>.</param>
        /// <returns>The compiled kernel.</returns>
        /// <exception cref="InvalidProgramException">The emitted IL is invalid (a bug here; caught by the caller).</exception>
        private static TakeLeadingKernel GenerateTakeLeadingKernelIL(int copyKind)
        {
            var dm = new DynamicMethod(
                name: $"IL_TakeLeading_c{copyKind}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(long*),  // 1 indices
                    typeof(long),   // 2 indicesCount
                    typeof(long),   // 3 outerSize
                    typeof(long),   // 4 maxItem
                    typeof(long),   // 5 innerSize
                    typeof(long),   // 6 tileRows
                    typeof(byte*),  // 7 dst
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locRowStep = il.DeclareLocal(typeof(long));   // maxItem * innerSize: bytes between consecutive src rows
            var locOb = il.DeclareLocal(typeof(long));        // first row of the current tile
            var locCount = il.DeclareLocal(typeof(long));     // rows in the current tile
            var locJ = il.DeclareLocal(typeof(long));
            var locK = il.DeclareLocal(typeof(long));
            var locS = il.DeclareLocal(typeof(byte*));
            var locD = il.DeclareLocal(typeof(byte*));
            var locS2 = il.DeclareLocal(typeof(byte*));       // the unrolled body's per-lane addresses
            var locD2 = il.DeclareLocal(typeof(byte*));

            // For a typed width the slab is exactly copyKind bytes; cpblk reads its runtime size from arg 5.
            Action pushInner = () => il.Emit(OpCodes.Ldarg, 5);

            // rowStep = maxItem * innerSize
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locRowStep);

            // ob = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOb);

            var lblObHead = il.DefineLabel();
            var lblObEnd = il.DefineLabel();
            il.MarkLabel(lblObHead);
            il.Emit(OpCodes.Ldloc, locOb);
            il.Emit(OpCodes.Ldarg_3);                         // outerSize
            il.Emit(OpCodes.Bge, lblObEnd);

            // count = outerSize - ob; if (count > tileRows) count = tileRows;
            var lblCountOk = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locOb);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locCount);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldarg, 6);                        // tileRows
            il.Emit(OpCodes.Ble, lblCountOk);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Stloc, locCount);
            il.MarkLabel(lblCountOk);

            // j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            var lblJHead = il.DefineLabel();
            var lblJEnd = il.DefineLabel();
            il.MarkLabel(lblJHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);                         // indicesCount
            il.Emit(OpCodes.Bge, lblJEnd);

            // s = src + (ob * maxItem + indices[j]) * innerSize
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locOb);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);                        // indices[j]
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locS);

            // d = dst + (j * outerSize + ob) * innerSize
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldloc, locOb);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);

            // k = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            if (copyKind != 0)
            {
                // 4x-unrolled body for the typed widths: four independent slab copies per iteration, so a 1-8 byte
                // slab costs its MOVs rather than the loop's compare/branch/increments.
                var lblU4Head = il.DefineLabel();
                var lblU4End = il.DefineLabel();
                il.MarkLabel(lblU4Head);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, 4L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, locCount);
                il.Emit(OpCodes.Bgt, lblU4End);               // while (k + 4 <= count)

                EmitElementCopy(il, copyKind, locD, locS, pushInner);
                for (int lane = 1; lane < 4; lane++)
                {
                    // d2 = d + lane * copyKind; s2 = s + lane * rowStep
                    il.Emit(OpCodes.Ldloc, locD);
                    il.Emit(OpCodes.Ldc_I4, lane * copyKind);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, locD2);
                    il.Emit(OpCodes.Ldloc, locS);
                    il.Emit(OpCodes.Ldloc, locRowStep);
                    il.Emit(OpCodes.Ldc_I8, (long)lane);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, locS2);
                    EmitElementCopy(il, copyKind, locD2, locS2, pushInner);
                }

                // s += 4 * rowStep; d += 4 * copyKind; k += 4
                il.Emit(OpCodes.Ldloc, locS);
                il.Emit(OpCodes.Ldloc, locRowStep);
                il.Emit(OpCodes.Ldc_I8, 4L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locS);
                il.Emit(OpCodes.Ldloc, locD);
                il.Emit(OpCodes.Ldc_I4, 4 * copyKind);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locD);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, 4L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, lblU4Head);
                il.MarkLabel(lblU4End);
            }

            // Remainder (or the whole run for cpblk): while (k < count) { copy(d, s); s += rowStep; d += innerSize; k++; }
            var lblRHead = il.DefineLabel();
            var lblREnd = il.DefineLabel();
            il.MarkLabel(lblRHead);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Bge, lblREnd);
            EmitElementCopy(il, copyKind, locD, locS, pushInner);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Ldloc, locRowStep);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locS);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblRHead);
            il.MarkLabel(lblREnd);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblJHead);
            il.MarkLabel(lblJEnd);

            // ob += tileRows
            il.Emit(OpCodes.Ldloc, locOb);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOb);
            il.Emit(OpCodes.Br, lblObHead);

            il.MarkLabel(lblObEnd);
            il.Emit(OpCodes.Ret);

            return (TakeLeadingKernel)dm.CreateDelegate(typeof(TakeLeadingKernel));
        }
    }
}
