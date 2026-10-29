using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using WriteOp = NumSharp.Backends.Kernels.ILKernelGenerator.VStringWriteOp;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The managed state of one call of a <c>StringDType</c> writing kernel (<c>add</c>, <c>multiply</c>, <c>maximum</c>,
    ///     <c>minimum</c>): the allocators of the inputs and of the output (locked by the caller for the whole call), the NA
    ///     rule and default string of the descriptor NumPy's loop reads them from, and the facts that pick a fast path. The
    ///     IL kernel resolves it once per call from its auxdata (<see cref="FromAux"/>).
    /// </summary>
    internal sealed unsafe class VStringWriteContext
    {
        /// <summary>The allocator of the first text input (the text operand of a repetition).</summary>
        internal VStringAllocator InA;

        /// <summary>The allocator of the second text input, or null for a repetition.</summary>
        internal VStringAllocator InB;

        /// <summary>The output's allocator.</summary>
        internal VStringAllocator Out;

        /// <summary>The NA rule of the deciding descriptor (NumPy's <c>s1descr</c> / <c>idescr</c> / <c>in1_descr</c>).</summary>
        internal VStringNARule Rule;

        /// <summary>The deciding descriptor's default string as UTF-8 (pinned by the caller for the call), or null when empty.</summary>
        internal byte* DefaultText;

        /// <summary>The byte length of <see cref="DefaultText"/>.</summary>
        internal long DefaultSize;

        /// <summary>
        ///     True when the output's allocator is an input's (NumPy's in-place case, <c>s1descr == odescr</c>): an allocation
        ///     can move that arena, so the inputs' texts are copied out before the output string is packed.
        /// </summary>
        internal bool InPlace;

        /// <summary>
        ///     True when every output cell is still the all-zero cell of a fresh array (nothing to free): a short result can
        ///     be written straight into its cell, and a size pass can reserve the arena exactly.
        /// </summary>
        internal bool FreshOut;

        /// <summary>The ufunc NumPy names in its errors (<c>add</c>, <c>multiply</c>, <c>maximum</c>, <c>minimum</c>).</summary>
        internal string Name;

        /// <summary>The size pass's accumulator: the arena bytes the outputs will take.</summary>
        internal long Reserve;

        /// <summary>
        ///     The arena base of <see cref="InA"/>, read once with the allocators locked (null without an input allocator or an
        ///     arena). Valid for the whole call only when <see cref="FreshOut"/> holds: a fresh output has its own allocator, so
        ///     no allocation of the call moves an input's arena — the fresh fast paths load the input texts through it.
        /// </summary>
        internal byte* ArenaA;

        /// <summary>The arena base of <see cref="InB"/> (see <see cref="ArenaA"/>).</summary>
        internal byte* ArenaB;

        /// <summary>
        ///     The context a writing kernel's auxdata points at: a pointer to the <see cref="GCHandle"/> the caller allocated
        ///     for it.
        /// </summary>
        /// <param name="aux">The kernel's auxdata.</param>
        /// <returns>The context.</returns>
        internal static VStringWriteContext FromAux(void* aux) => (VStringWriteContext)GCHandle.FromIntPtr(*(IntPtr*)aux).Target;
    }

    /// <summary>
    ///     NumPy's <c>StringDType</c> loops that WRITE text: <c>add</c> (<c>add_strided_loop</c>), <c>multiply</c>
    ///     (<c>multiply_loop_core</c>, both operand orders), <c>maximum</c> / <c>minimum</c>
    ///     (<c>minimum_maximum_strided_loop</c>) of <c>stringdtype_ufuncs.cpp</c>, and the <c>isnan</c> loop — run by the
    ///     trivial loop or NDIter over the IL kernels of <see cref="ILKernelGenerator.GetVStringWriteKernel"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Descriptors</b> (NumPy's resolve_descriptors): a <c>U</c> operand of <c>add</c> is cast to a default
    ///     <c>StringDType()</c> first; the output of <c>add</c> / <c>maximum</c> / <c>minimum</c> takes the two NA objects'
    ///     compatible one and the FIRST operand's <c>coerce</c>, the output of <c>multiply</c> the text operand's NA object and
    ///     <c>coerce</c>; an <c>out=</c> keeps its own.
    ///     </para>
    ///     <para>
    ///     <b>NA rules</b> come from ONE descriptor: <c>add</c>'s first operand (so <c>'z' + t</c> reads an NA of <c>t</c> as
    ///     <c>''</c> while <c>t + 'z'</c> follows <c>t</c>'s rule), <c>multiply</c>'s text operand, the first operand of
    ///     <c>maximum</c> / <c>minimum</c> (<c>_compare</c>'s <c>descr_a</c>). A NaN-like NA propagates (<c>add</c>,
    ///     <c>multiply</c>) or is the greatest element (<c>maximum</c> keeps it, <c>minimum</c> drops it); a <c>None</c>-like
    ///     NA raises NumPy's error; a string NA reads as that string.
    ///     </para>
    ///     <para>
    ///     <b>Allocation.</b> A result of at most 15 bytes into a fresh output is built in its cell without the allocator
    ///     (two short cells concatenated, a short cell repeated, a short winner copied); every other element goes through the
    ///     output allocator exactly as NumPy's loop does (<c>load_new_string</c>, or a temporary and <c>NpyString_pack</c> in
    ///     the in-place case). A fresh output's arena is reserved exactly first, from a size pass (strings plan D12).
    ///     </para>
    ///     <para>
    ///     <b>Overlap.</b> An <c>out=</c> that shares memory with an input other than as the same elements gets that input
    ///     copied first (NumPy's COPY_IF_OVERLAP), so the iterator never substitutes an operand behind the kernels' back: the
    ///     kernels read each input's cells through the allocator recorded in the context.
    ///     </para>
    /// </remarks>
    internal static unsafe class NDVStringArithmetic
    {
        /// <summary>The flags-and-size byte of a short cell, with the size nibble cleared.</summary>
        private const uint ShortFlags = VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA;

        /// <summary>
        ///     The masks of a short cell's text by size (see <see cref="NDVStringCells"/>): entries 0–15 the low word's text
        ///     bytes, 16–31 the high word's text bytes (byte 15 cleared).
        /// </summary>
        private static readonly ulong* s_masks = BuildMasks();

        /// <summary>Builds <see cref="s_masks"/>.</summary>
        /// <returns>The 32-word table (never freed: 256 bytes for the process).</returns>
        private static ulong* BuildMasks()
        {
            var t = (ulong*)NativeMemory.Alloc(32, sizeof(ulong));
            for (int s = 0; s < 16; s++)
            {
                t[s] = s >= 8 ? ulong.MaxValue : (1UL << (s * 8)) - 1;
                t[16 + s] = s <= 8 ? 0UL : (1UL << ((s - 8) * 8)) - 1;
            }
            return t;
        }

        // =============================================================================================================
        // add
        // =============================================================================================================

        /// <summary>
        ///     Computes <c>np.add(x1, x2, out=out, where=where)</c> for a pair with a <c>StringDType</c> operand, after
        ///     <see cref="NDStringUfuncs.BinaryGuard"/> accepted the call (a non-empty broadcast).
        /// </summary>
        /// <param name="x1">The first operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="x2">The second operand (<c>T</c> or <c>U</c>).</param>
        /// <param name="out">The <c>out=</c> operand (a <c>StringDType</c> array the guard accepted the cast into, or an
        ///     <c>S</c> / <c>U</c> array that receives the result through NumPy's output cast), or null.</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <param name="output">The loop's output descriptor (the guard's resolution; recomputed here exactly).</param>
        /// <returns><paramref name="out"/> itself, or the fresh <c>StringDType</c> result.</returns>
        /// <exception cref="ValueError">An NA under the <c>None</c>-like rule (<c>Cannot add null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts (<c>Failed to allocate string in add</c>,
        ///     …).</exception>
        internal static NDArray Add(NDArray x1, NDArray x2, NDArray @out, NDArray where, DType output)
        {
            NDArray t1 = null, t2 = null;
            try
            {
                t1 = NDStringUfuncs.IsVString(x1) ? x1 : NDVStringComparison.ToDefaultVString(x1);
                t2 = NDStringUfuncs.IsVString(x2) ? x2 : NDVStringComparison.ToDefaultVString(x2);
                var resultDescr = BinaryOutputDescr(t1.dtype, t2.dtype);
                return Run(WriteOp.Add, WriteOp.AddSize, x1, x2, t1, t2, @out, where, resultDescr, t1.dtype, "add", textIndex: 0);
            }
            finally
            {
                if (t1 is not null && !ReferenceEquals(t1, x1))
                    t1.Dispose();
                if (t2 is not null && !ReferenceEquals(t2, x2))
                    t2.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's <c>binary_resolve_descriptors</c> (add, maximum, minimum): a fresh descriptor with the two NA objects'
        ///     compatible one (the present one, the first when both are) and — NumPy reads <c>descr1-&gt;coerce</c> twice — the
        ///     FIRST descriptor's <c>coerce</c>.
        /// </summary>
        /// <param name="d1">The first (promoted) descriptor.</param>
        /// <param name="d2">The second (promoted) descriptor.</param>
        /// <returns>The output descriptor.</returns>
        /// <exception cref="TypeError">Two different NA objects (the guard reports that first; a direct caller reaches it here).</exception>
        internal static DType BinaryOutputDescr(DType d1, DType d2)
        {
            var p1 = d1.VString;
            var p2 = d2.VString;
            if (!VStringParameters.CompatibleNA(p1, p2, out bool hasNA, out object na))
                throw new TypeError(
                    $"Cannot find a compatible null string value for null strings '{NDPyObject.Repr(p1.na_object)}' and " +
                    $"'{NDPyObject.Repr(p2.na_object)}'");
            return DTypeRegistry.VString.Descr(hasNA, na, p1.coerce && p1.coerce);
        }

        /// <summary>
        ///     <c>out = A + B</c> for one element. Into a fresh output: two short cells whose sum fits a cell are concatenated
        ///     in the cell, and two non-NA texts whose sum needs the arena are appended to it directly
        ///     (<see cref="AddFresh"/>); everything else takes <see cref="AddSlow"/>.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="ValueError">See <see cref="AddSlow"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="AddSlow"/>.</exception>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void AddElement(byte* a, byte* b, byte* o, VStringWriteContext ctx)
        {
            ulong ha = Unsafe.ReadUnaligned<ulong>(a + 8), hb = Unsafe.ReadUnaligned<ulong>(b + 8);
            uint fa = (uint)(ha >> 56), fb = (uint)(hb >> 56);
            if (ctx.FreshOut && (fa & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags && (fb & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags)
            {
                int sa = (int)(fa & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
                int sb = (int)(fb & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
                int n = sa + sb;
                if (n <= VStringAllocator.NPY_SHORT_STRING_MAX_SIZE)
                {
                    if (n != 0)
                        WriteShortConcat(a, sa, ha, b, sb, hb, o, n);
                    // An empty result into a fresh cell is NumPy's NpyString_pack_empty on a never-written cell: it stays
                    // the all-zero cell.
                    return;
                }
            }
            AddNotShort(a, b, o, ctx);
        }

        /// <summary>
        ///     <see cref="AddElement"/> for every element whose sum is not built in a fresh cell: the fresh arena form
        ///     (<see cref="AddFresh"/>), else NumPy's general path (<see cref="AddSlow"/>).
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="ValueError">See <see cref="AddSlow"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="AddSlow"/>.</exception>
        /// <remarks>Out of line so the arena form's locals do not crowd the in-cell form's registers in the kernel.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void AddNotShort(byte* a, byte* b, byte* o, VStringWriteContext ctx)
        {
            if (ctx.FreshOut && AddFresh(a, b, o, ctx))
                return;
            AddSlow(a, b, o, ctx);
        }

        /// <summary>
        ///     The arena form of <see cref="AddElement"/> into a fresh output: both texts loaded through the inputs' arena
        ///     bases (no allocator call: a fresh output never moves an input's arena), the sum appended to the output's arena
        ///     by <see cref="VStringAllocator.AppendFresh"/> and the two texts copied after each other — the bytes NumPy's
        ///     <c>load_new_string</c> + two <c>memcpy</c> leave.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The fresh output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <returns>False when the element is not this form's (an NA, a sum of at most 15 bytes from arena cells, a sum past
        ///     the reserved capacity, a corrupt cell, a scheduled test failure): the caller runs <see cref="AddSlow"/>, which
        ///     gives NumPy's answer for each.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static bool AddFresh(byte* a, byte* b, byte* o, VStringWriteContext ctx)
        {
            // Load's -1 (corrupt) and 1 (NA) both leave this form: the slow path raises / applies the NA rule.
            if ((NDVStringCells.Load(a, ctx.ArenaA, out byte* ba, out long sa) | NDVStringCells.Load(b, ctx.ArenaB, out byte* bb, out long sb)) != 0)
                return false;
            long n = sa + sb;
            if (n <= VStringAllocator.NPY_SHORT_STRING_MAX_SIZE)
                return false;
            byte* dst = ctx.Out.AppendFresh(o, n);
            if (dst == null)
                return false;
            Buffer.MemoryCopy(ba, dst, n, sa);
            Buffer.MemoryCopy(bb, dst + sa, n - sa, sb);
            return true;
        }

        /// <summary>
        ///     Writes the short concatenation of two short texts into a fresh cell: the two masked texts as a 120-bit value
        ///     (B shifted past A), then byte 15 = <c>0x60 | n</c> — the bytes NumPy's <c>newemptysize</c> + <c>memcpy</c> leave
        ///     in an all-zero cell.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="sa">A's size (0–15).</param>
        /// <param name="ha">A's high word (bytes 8–15).</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="sb">B's size.</param>
        /// <param name="hb">B's high word.</param>
        /// <param name="o">The output cell (all zero).</param>
        /// <param name="n">The result size, <c>sa + sb</c>, 1–15.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static void WriteShortConcat(byte* a, int sa, ulong ha, byte* b, int sb, ulong hb, byte* o, int n)
        {
            ulong alo = Unsafe.ReadUnaligned<ulong>(a) & s_masks[sa], ahi = ha & s_masks[16 + sa];
            ulong blo = Unsafe.ReadUnaligned<ulong>(b) & s_masks[sb], bhi = hb & s_masks[16 + sb];
            ulong rlo, rhi;
            if (sa == 0)
            {
                rlo = blo;
                rhi = bhi;
            }
            else if (sa < 8)
            {
                // B starts inside the low word: its low bytes shift up, the ones that cross bit 64 move into the high word.
                int s = sa * 8;
                rlo = alo | (blo << s);
                rhi = ahi | (bhi << s) | (blo >> (64 - s));
            }
            else
            {
                // B starts in the high word; with n <= 15 B has at most 7 bytes, all in its low word.
                int s = (sa - 8) * 8;
                rlo = alo;
                rhi = ahi | (blo << s);
            }
            rhi = (rhi & NDVStringCells.SizeMask) | ((ulong)(ShortFlags | (uint)n) << 56);
            Unsafe.WriteUnaligned(o, rlo);
            Unsafe.WriteUnaligned(o + 8, rhi);
        }

        /// <summary>
        ///     The per-element path of NumPy's <c>add_strided_loop</c>: both texts loaded (through the allocators, so an
        ///     in-place arena that moved is read where it is now), the NA rule, the size, then <c>load_new_string</c> and two
        ///     copies — or, in place, a temporary and <c>NpyString_pack</c>.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="ValueError">An NA under the <c>None</c>-like rule: <c>Cannot add null that is not a nan-like value</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to load string in add</c>, <c>Failed to deallocate string in add</c>,
        ///     <c>Failed to allocate string in add</c>, <c>Failed to pack output string in add</c>.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static void AddSlow(byte* a, byte* b, byte* o, VStringWriteContext ctx)
        {
            int na = ctx.InA.load(a, out byte* ba, out long sa);
            int nb = ctx.InB.load(b, out byte* bb, out long sb);
            if (na < 0 || nb < 0)
                throw new OutOfMemoryException("Failed to load string in add");
            if (na != 0 || nb != 0)
            {
                switch (ctx.Rule)
                {
                    case VStringNARule.NaN:
                        if (ctx.Out.pack_null(o) < 0)
                            throw new OutOfMemoryException("Failed to deallocate string in add");
                        return;
                    case VStringNARule.Null:
                        throw new ValueError("Cannot add null that is not a nan-like value");
                    default:
                        if (na != 0)
                        {
                            ba = ctx.DefaultText;
                            sa = ctx.DefaultSize;
                        }
                        if (nb != 0)
                        {
                            bb = ctx.DefaultText;
                            sb = ctx.DefaultSize;
                        }
                        break;
                }
            }
            long newsize = sa + sb;
            if (newsize < sa)
                throw new OutOfMemoryException("Failed to allocate string in add");
            if (ctx.InPlace)
            {
                // The output allocator is an input's: packing may reallocate the arena under ba / bb, so the texts are
                // copied into a temporary first (NumPy's in-place branch).
                byte* tmp = (byte*)NativeMemory.Alloc((nuint)Math.Max(newsize, 1));
                try
                {
                    Buffer.MemoryCopy(ba, tmp, newsize, sa);
                    Buffer.MemoryCopy(bb, tmp + sa, newsize - sa, sb);
                    if (ctx.Out.pack(o, tmp, newsize) < 0)
                        throw new OutOfMemoryException("Failed to pack output string in add");
                }
                finally
                {
                    NativeMemory.Free(tmp);
                }
                return;
            }
            byte* buf = LoadNewString(o, newsize, ctx.Out, "add");
            Buffer.MemoryCopy(ba, buf, newsize, sa);
            Buffer.MemoryCopy(bb, buf + sa, newsize - sa, sb);
        }

        /// <summary>
        ///     The size pass of <c>add</c>: the arena bytes the element's result takes in a fresh output (nothing for a result
        ///     of at most 15 bytes or an NA); the kernel sums them into <see cref="VStringWriteContext.Reserve"/>.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <returns>The arena bytes, prefix included.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long AddSize(byte* a, byte* b, VStringWriteContext ctx)
        {
            long sa = SizeOrDefault(a, ctx, out bool naA), sb = SizeOrDefault(b, ctx, out bool naB);
            // A NaN-like NA writes an NA cell (no text); a None-like NA raises in the write pass.
            if ((naA | naB) && ctx.Rule is VStringNARule.NaN or VStringNARule.Null)
                return 0;
            return VStringAllocator.ArenaBytesForText(sa + sb);
        }

        /// <summary>
        ///     A cell's text size for a size pass: a short cell's nibble, a vstring's size field; an NA reads as the deciding
        ///     descriptor's default string (its size) and is reported.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="na">Receives whether the cell is an NA.</param>
        /// <returns>The size in bytes.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static long SizeOrDefault(byte* cell, VStringWriteContext ctx, out bool na)
        {
            ulong h = Unsafe.ReadUnaligned<ulong>(cell + 8);
            uint f = (uint)(h >> 56);
            na = (f & VStringAllocator.NPY_STRING_MISSING) != 0;
            // Value selects rather than returns: a size pass over text that mixes short and arena cells at random would
            // otherwise mispredict the kind on every other element.
            long shortSize = f & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK;
            long vsize = (long)(h & NDVStringCells.SizeMask);
            // Masks, not ternaries (the JIT compiled those to branches): all ones for a short cell / an NA cell.
            long shortMask = ((long)((f & VStringAllocator.NPY_STRING_FLAG_MASK) ^ ShortFlags) - 1) >> 63;
            long size = (shortSize & shortMask) | (vsize & ~shortMask);
            long naMask = -(long)((f >> 7) & 1);
            return (ctx.DefaultSize & naMask) | (size & ~naMask);
        }

        /// <summary>
        ///     NumPy's <c>load_new_string</c>: frees the output cell's old text, prepares it for <paramref name="size"/> bytes
        ///     (in the cell for at most 15, else in the arena or on the heap) and returns where the text goes.
        /// </summary>
        /// <param name="o">The output cell.</param>
        /// <param name="size">The new text size.</param>
        /// <param name="allocator">The output's allocator (locked).</param>
        /// <param name="name">The ufunc NumPy names in the error.</param>
        /// <returns>The first byte of the new text (valid until the next allocation in <paramref name="allocator"/>).</returns>
        /// <exception cref="OutOfMemoryException"><c>Failed to deallocate string in {name}</c>, <c>Failed to allocate string in
        ///     {name}</c>, <c>Failed to load string in {name}</c>.</exception>
        private static byte* LoadNewString(byte* o, long size, VStringAllocator allocator, string name)
        {
            if (allocator.free(o) < 0)
                throw new OutOfMemoryException($"Failed to deallocate string in {name}");
            if (allocator.newemptysize(size, o) < 0)
                throw new OutOfMemoryException($"Failed to allocate string in {name}");
            if (allocator.load(o, out byte* buf, out _) < 0)
                throw new OutOfMemoryException($"Failed to load string in {name}");
            return buf;
        }

        // =============================================================================================================
        // multiply
        // =============================================================================================================

        /// <summary>
        ///     Computes <c>np.multiply(x1, x2, out=out, where=where)</c> for a <c>StringDType</c> text and an integer count
        ///     (either order) after <see cref="NDStringUfuncs.BinaryGuard"/> accepted the call. NumPy's promoter reads every
        ///     integer type but <c>uint64</c> as int64 (a negative count is then a huge <c>size_t</c>: NumPy's overflow error
        ///     for any non-empty text); <c>uint64</c> has its own loop.
        /// </summary>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <param name="output">The loop's output descriptor (the text operand's NA object and <c>coerce</c>).</param>
        /// <returns><paramref name="out"/> itself, or the fresh result.</returns>
        /// <exception cref="TypeError">An NA under the <c>None</c>-like rule (<c>Cannot multiply null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OverflowException"><c>Overflow encountered in string multiply</c>: <c>size * count</c> past
        ///     <c>2^63 - 1</c> as unsigned 64-bit arithmetic.</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        /// <remarks>An EMPTY text with a negative count makes NumPy's copy loop run <c>2^64 - k</c> times copying nothing (it
        ///     never returns); NumSharp writes the empty string at once (<c>[Misaligned]</c>).</remarks>
        internal static NDArray Multiply(NDArray x1, NDArray x2, NDArray @out, NDArray where, DType output)
        {
            int textIndex = NDStringUfuncs.IsVString(x1) ? 0 : 1;
            NDArray text = textIndex == 0 ? x1 : x2;
            NDArray counts = textIndex == 0 ? x2 : x1;
            bool unsigned = counts.typecode == NPTypeCode.UInt64;
            // The promoter's int64 (every other integer width, NumSharp's char = uint16 included); uint64 keeps its loop.
            NDArray promoted = unsigned || counts.typecode == NPTypeCode.Int64 ? counts : counts.astype(NPTypeCode.Int64);
            try
            {
                var p = text.dtype.VString;
                var resultDescr = DTypeRegistry.VString.Descr(p.HasNAObject, p.na_object, p.coerce);
                NDArray a1 = textIndex == 0 ? text : promoted, a2 = textIndex == 0 ? promoted : text;
                return Run(unsigned ? WriteOp.MultiplyUnsigned : WriteOp.Multiply,
                           unsigned ? WriteOp.MultiplySizeUnsigned : WriteOp.MultiplySize,
                           x1, x2, a1, a2, @out, where, resultDescr, text.dtype, "multiply", textIndex);
            }
            finally
            {
                if (!ReferenceEquals(promoted, counts))
                    promoted.Dispose();
            }
        }

        /// <summary>
        ///     <c>out = text * count</c> for one element (NumPy's <c>multiply_loop_core</c>): a short text whose repetition fits
        ///     a cell is built in the cell (a fresh output only); everything else takes <see cref="MultiplySlow"/>.
        /// </summary>
        /// <param name="s">The text cell.</param>
        /// <param name="i">The count (int64, or uint64 when <paramref name="unsigned"/>).</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="unsigned">True for the uint64 loop (an emit-time constant).</param>
        /// <exception cref="TypeError">See <see cref="MultiplySlow"/>.</exception>
        /// <exception cref="OverflowException">See <see cref="MultiplySlow"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="MultiplySlow"/>.</exception>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void MultiplyElement(byte* s, byte* i, byte* o, VStringWriteContext ctx, bool unsigned)
        {
            ulong factor = unsigned ? Unsafe.ReadUnaligned<ulong>(i) : (ulong)Unsafe.ReadUnaligned<long>(i);
            ulong h = Unsafe.ReadUnaligned<ulong>(s + 8);
            uint f = (uint)(h >> 56);
            if (ctx.FreshOut && (f & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags)
            {
                ulong size = f & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK;
                // factor <= 15 keeps the product exact; a larger factor fits only an empty text (product 0).
                if (size == 0 || factor <= VStringAllocator.NPY_SHORT_STRING_MAX_SIZE && size * factor <= VStringAllocator.NPY_SHORT_STRING_MAX_SIZE)
                {
                    if (size != 0 && factor != 0)
                        WriteShortRepeat(s, (int)size, (int)factor, o);
                    // Empty result: the all-zero cell stays (NumPy's pack of an empty text into a never-written cell).
                    return;
                }
            }
            MultiplyNotShort(s, factor, o, ctx);
        }

        /// <summary>
        ///     <see cref="MultiplyElement"/> for every element whose repetition is not built in a fresh cell: the fresh arena
        ///     form (<see cref="MultiplyFresh"/>), else NumPy's general path (<see cref="MultiplySlow"/>).
        /// </summary>
        /// <param name="s">The text cell.</param>
        /// <param name="factor">The count as NumPy's <c>size_t</c>.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="TypeError">See <see cref="MultiplySlow"/>.</exception>
        /// <exception cref="OverflowException">See <see cref="MultiplySlow"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="MultiplySlow"/>.</exception>
        /// <remarks>Out of line so the arena form's locals do not crowd the in-cell form's registers in the kernel.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void MultiplyNotShort(byte* s, ulong factor, byte* o, VStringWriteContext ctx)
        {
            if (ctx.FreshOut && MultiplyFresh(s, factor, o, ctx))
                return;
            MultiplySlow(s, factor, o, ctx);
        }

        /// <summary>
        ///     The arena form of <see cref="MultiplyElement"/> into a fresh output: the text loaded through its arena base, the
        ///     product of size and count checked as NumPy's <c>size_t</c> arithmetic, the result appended to the output's arena
        ///     (<see cref="VStringAllocator.AppendFresh"/>) and filled by doubling copies (<see cref="FillRepeat"/>).
        /// </summary>
        /// <param name="s">The text cell.</param>
        /// <param name="factor">The count as NumPy's <c>size_t</c>.</param>
        /// <param name="o">The fresh output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <returns>False for an element this form does not take (an NA, an overflowing product, a result of at most 15
        ///     bytes, a result past the reserved capacity, a corrupt cell, a scheduled test failure): the caller runs
        ///     <see cref="MultiplySlow"/>, which raises or computes NumPy's answer.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static bool MultiplyFresh(byte* s, ulong factor, byte* o, VStringWriteContext ctx)
        {
            if (NDVStringCells.Load(s, ctx.ArenaA, out byte* buf, out long size) != 0)
                return false;
            ulong high = Math.BigMul((ulong)size, factor, out ulong newsize);
            if (high != 0 || newsize > long.MaxValue || newsize <= VStringAllocator.NPY_SHORT_STRING_MAX_SIZE)
                return false;
            byte* dst = ctx.Out.AppendFresh(o, (long)newsize);
            if (dst == null)
                return false;
            // A short text sits in its 16-byte cell, so its repeating pattern can be built in a register (16-byte stores);
            // an arena / heap text is at least 16 bytes and copies by doubling.
            if ((s[15] & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags)
                FillRepeatShort(dst, s, (int)size, (long)newsize);
            else
                FillRepeat(dst, buf, size, (long)newsize);
            return true;
        }

        /// <summary>
        ///     The repeating 16-byte block of a short text: its <paramref name="size"/> bytes (masked: the bytes past the size
        ///     are not zero in every cell), repeated by doubling until 16 bytes are filled — byte <c>j</c> is
        ///     <c>text[j % size]</c>.
        /// </summary>
        /// <param name="cell">The short cell.</param>
        /// <param name="size">Its text size (1–15).</param>
        /// <returns>The block, little-endian (byte 0 in the lowest bits).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static UInt128 ShortPattern(byte* cell, int size)
        {
            var r = new UInt128(Unsafe.ReadUnaligned<ulong>(cell + 8) & s_masks[16 + size], Unsafe.ReadUnaligned<ulong>(cell) & s_masks[size]);
            // Each step appends the first len bytes after themselves; the shift stays below 128 bits because the loop ends
            // as soon as 16 bytes hold the pattern.
            for (int len = size; len < 16; len <<= 1)
                r |= r << (8 * len);
            return r;
        }

        /// <summary>
        ///     Fills <paramref name="total"/> bytes of an arena text with copies of a short text: the 16-byte pattern block
        ///     (<see cref="ShortPattern"/>) stored at every multiple of the largest whole number of repeats a block holds while a
        ///     whole store fits, then the last bytes copied exactly — the bytes of NumPy's one-<c>memcpy</c>-per-repeat loop,
        ///     never a byte past the text (the arena may end there).
        /// </summary>
        /// <param name="dst">The destination text (<paramref name="total"/> bytes).</param>
        /// <param name="cell">The short source cell.</param>
        /// <param name="size">Its text size (1–15).</param>
        /// <param name="total">The result size, a multiple of <paramref name="size"/>, more than 15.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static void FillRepeatShort(byte* dst, byte* cell, int size, long total)
        {
            UInt128 block = ShortPattern(cell, size);
            // A block holds 16 / size whole repeats; storing it every that many repeats keeps every store aligned to the
            // pattern, so each store writes correct bytes over the previous store's partial repeat.
            long step = size * (16 / size);
            long pos = 0;
            for (; total - pos >= 16; pos += step)
                Unsafe.WriteUnaligned(dst + pos, block);
            if (pos < total)
                Buffer.MemoryCopy(&block, dst + pos, total - pos, total - pos);
        }

        /// <summary>Writes <c>factor</c> copies of a short text into a fresh cell (the result at most 15 bytes).</summary>
        /// <param name="s">The text cell.</param>
        /// <param name="size">The text size (1–15).</param>
        /// <param name="factor">The count (1–15, <c>size * factor</c> at most 15).</param>
        /// <param name="o">The output cell (all zero).</param>
        /// <remarks>Built in a register (<see cref="ShortPattern"/>, then masked to the result size) instead of one copy per
        ///     repeat; the cell's bytes past the result stay zero, as NumPy's copies into a never-written cell leave them.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static void WriteShortRepeat(byte* s, int size, int factor, byte* o)
        {
            int n = size * factor;
            UInt128 r = ShortPattern(s, size);
            ulong lo = (ulong)r & s_masks[n];
            ulong hi = ((ulong)(r >> 64) & s_masks[16 + n]) | ((ulong)(ShortFlags | (uint)n) << 56);
            Unsafe.WriteUnaligned(o, lo);
            Unsafe.WriteUnaligned(o + 8, hi);
        }

        /// <summary>
        ///     The per-element path of NumPy's <c>multiply_loop_core</c>: the text loaded, the NA rule, the overflow check of
        ///     <c>size * count</c> as <c>size_t</c> arithmetic, then the output string (<c>load_new_string</c>, or a temporary
        ///     and <c>NpyString_pack</c> in place) filled with the copies (doubling, the same bytes as NumPy's copy loop).
        /// </summary>
        /// <param name="s">The text cell.</param>
        /// <param name="factor">The count as NumPy's <c>size_t</c> (a negative int64 is <c>2^64 + count</c>).</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <exception cref="TypeError">An NA under the <c>None</c>-like rule: <c>Cannot multiply null that is not a nan-like value</c>.</exception>
        /// <exception cref="OverflowException"><c>Overflow encountered in string multiply</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to load string in multiply</c>, <c>Failed to deallocate string in
        ///     multiply</c>, <c>Failed to allocate string in multiply</c>, <c>Failed to pack string in multiply</c>.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static void MultiplySlow(byte* s, ulong factor, byte* o, VStringWriteContext ctx)
        {
            int isNull = ctx.InA.load(s, out byte* buf, out long size);
            if (isNull < 0)
                throw new OutOfMemoryException("Failed to load string in multiply");
            if (isNull != 0)
            {
                if (ctx.Rule == VStringNARule.NaN)
                {
                    if (ctx.Out.pack_null(o) < 0)
                        throw new OutOfMemoryException("Failed to deallocate string in multiply");
                    return;
                }
                if (ctx.Rule == VStringNARule.Null)
                    throw new TypeError("Cannot multiply null that is not a nan-like value");
                buf = ctx.DefaultText;
                size = ctx.DefaultSize;
            }
            ulong high = Math.BigMul((ulong)size, factor, out ulong newsize);
            if (high != 0 || newsize > long.MaxValue)
                throw new OverflowException("Overflow encountered in string multiply");
            if (ctx.InPlace)
            {
                byte* tmp = (byte*)NativeMemory.Alloc((nuint)Math.Max(newsize, 1UL));
                try
                {
                    FillRepeat(tmp, buf, size, (long)newsize);
                    if (ctx.Out.pack(o, tmp, (long)newsize) < 0)
                        throw new OutOfMemoryException("Failed to pack string in multiply");
                }
                finally
                {
                    NativeMemory.Free(tmp);
                }
                return;
            }
            byte* dst = LoadNewString(o, (long)newsize, ctx.Out, "multiply");
            FillRepeat(dst, buf, size, (long)newsize);
        }

        /// <summary>
        ///     Fills <paramref name="total"/> bytes of <paramref name="dst"/> with copies of a <paramref name="size"/>-byte
        ///     text: one copy, then doubling copies of what is already written (the bytes of NumPy's one-copy-per-repeat loop,
        ///     in O(log count) calls).
        /// </summary>
        /// <param name="dst">The destination (<paramref name="total"/> bytes).</param>
        /// <param name="text">The text.</param>
        /// <param name="size">Its size (0 writes nothing).</param>
        /// <param name="total">The result size, a multiple of <paramref name="size"/>.</param>
        private static void FillRepeat(byte* dst, byte* text, long size, long total)
        {
            if (size == 0 || total == 0)
                return;
            Buffer.MemoryCopy(text, dst, total, size);
            long done = size;
            while (done < total)
            {
                long chunk = Math.Min(done, total - done);
                Buffer.MemoryCopy(dst, dst + done, total - done, chunk);
                done += chunk;
            }
        }

        /// <summary>
        ///     The size pass of <c>multiply</c>: the arena bytes the element's result takes in a fresh output (nothing for an NA,
        ///     a result of at most 15 bytes, or an overflowing product, which raises in the write pass); the kernel sums them
        ///     into <see cref="VStringWriteContext.Reserve"/>.
        /// </summary>
        /// <param name="s">The text cell.</param>
        /// <param name="i">The count.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="unsigned">True for the uint64 loop.</param>
        /// <returns>The arena bytes, prefix included.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long MultiplySize(byte* s, byte* i, VStringWriteContext ctx, bool unsigned)
        {
            ulong factor = unsigned ? Unsafe.ReadUnaligned<ulong>(i) : (ulong)Unsafe.ReadUnaligned<long>(i);
            long size = SizeOrDefault(s, ctx, out bool na);
            if (na && ctx.Rule is VStringNARule.NaN or VStringNARule.Null)
                return 0;
            ulong high = Math.BigMul((ulong)size, factor, out ulong newsize);
            if (high != 0 || newsize > long.MaxValue)
                return 0;
            return VStringAllocator.ArenaBytesForText((long)newsize);
        }

        // =============================================================================================================
        // maximum / minimum
        // =============================================================================================================

        /// <summary>
        ///     Computes <c>np.maximum</c> / <c>np.minimum(x1, x2, out=out, where=where)</c> over two <c>StringDType</c> operands
        ///     after <see cref="NDStringUfuncs.BinaryGuard"/> accepted the call: per element the winner of NumPy's
        ///     <c>_compare</c> (the first operand's NA rule) is duplicated into the output; a tie takes the first operand under
        ///     <c>maximum</c> and the second under <c>minimum</c> (NumPy's <c>(cmp &lt; 0) ^ invert</c>).
        /// </summary>
        /// <param name="isMax">True for <c>maximum</c>.</param>
        /// <param name="x1">The first operand.</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The bool <c>where=</c> mask, or null.</param>
        /// <param name="output">The loop's output descriptor.</param>
        /// <returns><paramref name="out"/> itself, or the fresh result.</returns>
        /// <exception cref="ValueError">An NA under the <c>None</c>-like rule (<c>Cannot compare null that is not a nan-like
        ///     value</c>).</exception>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal static NDArray MaximumMinimum(bool isMax, NDArray x1, NDArray x2, NDArray @out, NDArray where, DType output)
        {
            var resultDescr = BinaryOutputDescr(x1.dtype, x2.dtype);
            string name = isMax ? "maximum" : "minimum";
            return Run(isMax ? WriteOp.Maximum : WriteOp.Minimum, WriteOp.MaxMinSize, x1, x2, x1, x2, @out, where, resultDescr,
                       x1.dtype, name, textIndex: 0);
        }

        /// <summary>
        ///     One element of NumPy's <c>minimum_maximum_strided_loop</c>: <c>_compare(A, B)</c>, a tie into an aliased output
        ///     skipped, else the winner duplicated (<c>free_and_copy</c>) unless it IS the output. Two short cells into a fresh
        ///     output compare in the cell and copy the winner's 16 bytes (NumPy's <c>NpyString_dup</c> of a short string).
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="isMax">True for <c>maximum</c> (NumPy's <c>invert</c>).</param>
        /// <exception cref="ValueError">See <see cref="Compare"/>.</exception>
        /// <exception cref="OutOfMemoryException">A failed duplicate (NumPy's <c>Failed to allocate string in maximum</c>, …).</exception>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void MaxMinElement(byte* a, byte* b, byte* o, VStringWriteContext ctx, bool isMax)
        {
            ulong ha = Unsafe.ReadUnaligned<ulong>(a + 8), hb = Unsafe.ReadUnaligned<ulong>(b + 8);
            uint fa = (uint)(ha >> 56), fb = (uint)(hb >> 56);
            if (ctx.FreshOut && (fa & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags && (fb & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags)
            {
                // Branch-free: the order, the winner and the source of the copy are flag arithmetic and mask selects
                // (random text makes each a coin flip, which a branch mispredicts half the time).
                bool takeA = NDVStringCells.ShortLess(a, ha, fa, b, hb, fb) ^ isMax;
                byte* w = NDVStringCells.Select(takeA, a, b);
                uint m = 0u - (takeA ? 1u : 0u);
                uint fw = (fa & m) | (fb & ~m);
                // NpyString_dup: a short winner is its 16-byte cell; an EMPTY one is pack_empty, which leaves a
                // never-written cell all zero — so the copy reads the zero cell instead and stays unconditional.
                byte* src = NDVStringCells.Select((fw & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK) != 0, w, s_zeroCell);
                Unsafe.WriteUnaligned(o, Unsafe.ReadUnaligned<ulong>(src));
                Unsafe.WriteUnaligned(o + 8, Unsafe.ReadUnaligned<ulong>(src + 8));
                return;
            }
            MaxMinNotShort(a, b, o, ctx, isMax);
        }

        /// <summary>
        ///     <see cref="MaxMinElement"/> for every pair that is not two short cells into a fresh output: the fresh arena form
        ///     (<see cref="MaxMinFresh"/>), else NumPy's general path (<see cref="MaxMinSlow"/>).
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="isMax">True for <c>maximum</c>.</param>
        /// <exception cref="ValueError">See <see cref="Compare"/>.</exception>
        /// <exception cref="OutOfMemoryException">A failed duplicate.</exception>
        /// <remarks>Out of line on purpose: inlined beside the short form, the arena form's locals made the JIT spill the
        ///     short form's registers on every element (13 ns an element for two short cells, the copy included).</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void MaxMinNotShort(byte* a, byte* b, byte* o, VStringWriteContext ctx, bool isMax)
        {
            if (ctx.FreshOut && MaxMinFresh(a, b, o, ctx, isMax))
                return;
            MaxMinSlow(a, b, o, ctx, isMax);
        }

        /// <summary>The all-zero cell an empty winner copies from (a never-written cell stays all zero in NumPy).</summary>
        private static readonly byte* s_zeroCell = (byte*)NativeMemory.AllocZeroed(VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING);

        /// <summary>
        ///     The arena form of <see cref="MaxMinElement"/> into a fresh output: both texts loaded through the inputs' arena
        ///     bases, NumPy's <c>NpyString_cmp</c>, then the winner duplicated as <c>NpyString_dup</c> does — a short winner's
        ///     16-byte cell, nothing for an empty one, an arena / heap winner appended to the output's arena
        ///     (<see cref="VStringAllocator.AppendFresh"/>) and copied.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The fresh output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="isMax">True for <c>maximum</c>.</param>
        /// <returns>False for an element this form does not take (an NA, a corrupt cell, a winner past the reserved
        ///     capacity, a scheduled test failure): the caller runs <see cref="MaxMinSlow"/>, which writes nothing before
        ///     deciding, so the element is computed once.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static bool MaxMinFresh(byte* a, byte* b, byte* o, VStringWriteContext ctx, bool isMax)
        {
            if ((NDVStringCells.Load(a, ctx.ArenaA, out byte* ba, out long sa) | NDVStringCells.Load(b, ctx.ArenaB, out byte* bb, out long sb)) != 0)
                return false;
            // A tie takes A under maximum and B under minimum (NumPy's (cmp < 0) ^ invert).
            bool takeA = (NDVStringCells.Order(ba, sa, bb, sb) < 0) ^ isMax;
            byte* w = NDVStringCells.Select(takeA, a, b);
            uint fw = w[15];
            if ((fw & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags)
            {
                if ((fw & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK) != 0)
                {
                    Unsafe.WriteUnaligned(o, Unsafe.ReadUnaligned<ulong>(w));
                    Unsafe.WriteUnaligned(o + 8, Unsafe.ReadUnaligned<ulong>(w + 8));
                }
                return true;
            }
            long ws = takeA ? sa : sb;
            // An emptied arena winner (size 0) is pack_empty: the fresh cell stays all zero.
            if (ws == 0)
                return true;
            byte* dst = ctx.Out.AppendFresh(o, ws);
            if (dst == null)
                return false;
            Buffer.MemoryCopy(NDVStringCells.Select(takeA, ba, bb), dst, ws, ws);
            return true;
        }

        /// <summary>
        ///     The order of two short cells (NumPy's <c>NpyString_cmp</c> of their texts): the masked cells as 128-bit
        ///     big-endian numbers.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="fa">A's byte 15.</param>
        /// <param name="ha">A's high word.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="fb">B's byte 15.</param>
        /// <param name="hb">B's high word.</param>
        /// <returns>-1, 0 or 1.</returns>
        /// <remarks>A NUL inside A's text can end NumPy's <c>strncmp</c> before the masked cells' first difference (the sizes
        ///     then decide, see <see cref="NDVStringCells.Order"/>): that pair, rare in real text, takes the exact order.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int ShortOrder(byte* a, uint fa, ulong ha, byte* b, uint fb, ulong hb)
        {
            int sa = (int)(fa & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK), sb = (int)(fb & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
            if (NDVStringCells.ShortHasNul(a, ha, sa))
                return NDVStringCells.Order(a, sa, b, sb);
            ulong la = Unsafe.ReadUnaligned<ulong>(a) & s_masks[sa], lb = Unsafe.ReadUnaligned<ulong>(b) & s_masks[sb];
            if (la != lb)
                return System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(la) < System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(lb) ? -1 : 1;
            // The text tail and the size (byte 15, both flags nibbles alike).
            ulong ka = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness((ha & s_masks[16 + sa]) | (ha & 0xFF00_0000_0000_0000UL));
            ulong kb = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness((hb & s_masks[16 + sb]) | (hb & 0xFF00_0000_0000_0000UL));
            return ka.CompareTo(kb);
        }

        /// <summary>
        ///     The per-element path of NumPy's <c>minimum_maximum_strided_loop</c>, through the allocators.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="o">The output cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <param name="isMax">True for <c>maximum</c>.</param>
        /// <exception cref="ValueError">See <see cref="Compare"/>.</exception>
        /// <exception cref="OutOfMemoryException">A failed duplicate.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static void MaxMinSlow(byte* a, byte* b, byte* o, VStringWriteContext ctx, bool isMax)
        {
            int cmp = Compare(a, ctx.InA, b, ctx.InB, ctx);
            if (cmp == 0 && (a == o || b == o))
                return;
            if ((cmp < 0) ^ isMax)
            {
                // if in and out are the same address, do nothing to avoid a use-after-free
                if (a != o)
                    VStringAllocator.FreeAndCopy(ctx.InA, ctx.Out, a, o, ctx.Name);
            }
            else if (b != o)
            {
                VStringAllocator.FreeAndCopy(ctx.InB, ctx.Out, b, o, ctx.Name);
            }
        }

        /// <summary>
        ///     NumPy's <c>_compare</c> (<c>stringdtype/dtype.c</c>) with the deciding descriptor's NA rule: a NaN-like NA is
        ///     greater than everything (an NA FIRST operand answers 1 even against another NA, the asymmetry NumPy's sorts and
        ///     arg-reductions inherit), a <c>None</c>-like NA raises, a string NA (or an NA of an operand without an NA object)
        ///     reads as the default string.
        /// </summary>
        /// <param name="a">The first cell.</param>
        /// <param name="aa">Its allocator.</param>
        /// <param name="b">The second cell.</param>
        /// <param name="ab">Its allocator.</param>
        /// <param name="ctx">The context holding the rule and the default string.</param>
        /// <returns>Negative, zero or positive.</returns>
        /// <exception cref="ValueError"><c>Cannot compare null that is not a nan-like value</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to load string in string comparison</c>.</exception>
        internal static int Compare(byte* a, VStringAllocator aa, byte* b, VStringAllocator ab, VStringWriteContext ctx)
            => CompareCells(a, aa, b, ab, ctx.Rule, ctx.DefaultText, ctx.DefaultSize);

        /// <summary>
        ///     <see cref="Compare"/> with the rule and the default string given directly (the reductions call it too).
        /// </summary>
        /// <param name="a">The first cell.</param>
        /// <param name="aa">Its allocator (locked).</param>
        /// <param name="b">The second cell.</param>
        /// <param name="ab">Its allocator (locked).</param>
        /// <param name="rule">The deciding descriptor's NA rule.</param>
        /// <param name="defaultText">Its default string (UTF-8), or null when empty.</param>
        /// <param name="defaultSize">The default string's byte length.</param>
        /// <returns>Negative, zero or positive.</returns>
        /// <exception cref="ValueError"><c>Cannot compare null that is not a nan-like value</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to load string in string comparison</c>.</exception>
        internal static int CompareCells(byte* a, VStringAllocator aa, byte* b, VStringAllocator ab, VStringNARule rule,
                                         byte* defaultText, long defaultSize)
        {
            int na = aa.load(a, out byte* ba, out long sa);
            int nb = ab.load(b, out byte* bb, out long sb);
            if (na < 0 || nb < 0)
                throw new OutOfMemoryException("Failed to load string in string comparison");
            if (na != 0 || nb != 0)
            {
                if (rule == VStringNARule.NaN)
                    return na != 0 ? 1 : -1;
                if (rule == VStringNARule.Null)
                    throw new ValueError("Cannot compare null that is not a nan-like value");
                if (na != 0)
                {
                    ba = defaultText;
                    sa = defaultSize;
                }
                if (nb != 0)
                {
                    bb = defaultText;
                    sb = defaultSize;
                }
            }
            return NDVStringCells.Order(ba, sa, bb, sb);
        }

        /// <summary>
        ///     The size pass of <c>maximum</c> / <c>minimum</c>: an upper bound of the winner's arena bytes in a fresh output
        ///     (the larger of the two texts' — no comparison is made here; the reservation only sizes the arena's capacity).
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="ctx">The call's context.</param>
        /// <returns>The arena bytes, prefix included (the kernel sums them into <see cref="VStringWriteContext.Reserve"/>).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static long MaxMinSize(byte* a, byte* b, VStringWriteContext ctx)
        {
            long sa = SizeOrDefault(a, ctx, out _), sb = SizeOrDefault(b, ctx, out _);
            // max(sa, sb) by a sign mask (Math.Max compiled to a branch on random sizes)
            long d = sa - sb;
            return VStringAllocator.ArenaBytesForText(sa - (d & (d >> 63)));
        }

        // =============================================================================================================
        // isnan
        // =============================================================================================================

        /// <summary>
        ///     Writes NumPy's <c>isnan</c> of every cell of <paramref name="x"/> (broadcast to <paramref name="flags"/>) into
        ///     the bool array: the NA bit. The caller runs it only for a descriptor whose NA object is NaN-like.
        /// </summary>
        /// <param name="x">The <c>StringDType</c> array.</param>
        /// <param name="flags">The bool output (zero-filled), of the broadcast shape.</param>
        internal static void FillIsNaN(NDArray x, NDArray flags)
        {
            var kernel = ILKernelGenerator.GetVStringIsNaNKernel() ?? ManagedIsNaNKernel();
            var allocator = x.Storage.VStringSlice.Allocator;
            allocator.acquire_allocator();
            try
            {
                using var iter = NDIterRef.AdvancedNew(2, new[] { x, flags }, NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.REFS_OK,
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.NO_BROADCAST },
                    opDescrs: new DType[2]);
                iter.ForEach(kernel);
            }
            finally
            {
                allocator.release_allocator();
            }
        }

        /// <summary>The isnan kernel as a managed loop (hosts without dynamic code).</summary>
        /// <returns>The kernel.</returns>
        private static NDInnerLoopFunc ManagedIsNaNKernel()
            => (dataptrs, strides, count, auxdata) =>
            {
                byte* p = (byte*)dataptrs[0], po = (byte*)dataptrs[1];
                long sp = strides[0], so = strides[1];
                for (long i = 0; i < count; i++, p += sp, po += so)
                    *po = (byte)(p[15] >> 7);
            };

        // =============================================================================================================
        // The driver
        // =============================================================================================================

        /// <summary>
        ///     The common driver of the writing loops: the result (a fresh <c>StringDType</c> array in NumPy's ufunc output
        ///     layout, a <c>StringDType</c> <c>out=</c>, or a temporary cast into an <c>S</c> / <c>U</c> <c>out=</c>), the
        ///     overlap copies, the locks, the context, the size pass and reservation for a fresh output, then the write pass
        ///     (the trivial loop, or NDIter with a <c>where=</c> mask as the ARRAYMASK operand).
        /// </summary>
        /// <param name="op">The write operation.</param>
        /// <param name="sizeOp">Its size pass.</param>
        /// <param name="x1">The first operand as given (its layout votes for a fresh result's).</param>
        /// <param name="x2">The second operand as given.</param>
        /// <param name="a1">The first operand as the kernel reads it (promoted to <c>StringDType</c>, or the int64 counts).</param>
        /// <param name="a2">The second operand as the kernel reads it.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="resultDescr">The loop's output descriptor (a fresh result's dtype).</param>
        /// <param name="ruleDescr">The descriptor whose NA rule and default string the loop reads.</param>
        /// <param name="name">The ufunc NumPy names in its errors.</param>
        /// <param name="textIndex">The slot of the (first) text operand: 0, or 1 for <c>count * text</c>.</param>
        /// <returns>The result (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="Exception">The loop's errors (see the public members).</exception>
        /// <remarks>
        ///     The output descriptor follows NumPy's <c>resolve_descriptors</c>: a <c>StringDType</c> out= is the loop's
        ///     output as it is (an NA the loop packs reads back as THAT descriptor's NA object), and any other out= becomes
        ///     the cast's default <c>StringDType()</c> — so the temporary cast into an <c>S</c> / <c>U</c> out= holds its NA
        ///     cells as nulls of a descriptor without an NA object, which the cast writes as <c>''</c> (probed:
        ///     <c>np.add(['ab', nan], ..., out=np.empty(2, 'U5'))</c> is <c>['abab', '']</c>).
        /// </remarks>
        private static NDArray Run(WriteOp op, WriteOp sizeOp, NDArray x1, NDArray x2, NDArray a1, NDArray a2, NDArray @out,
                                   NDArray where, DType resultDescr, DType ruleDescr, string name, int textIndex)
        {
            // A fixed-width (S / U) out= receives the StringDType result through NumPy's output cast: the temporary spans
            // the whole iteration (the out= included, which never broadcasts) and is computed under where= like NumPy's
            // buffered loop, so a masked-off element raises nothing (an NA the rule refuses included) and stays ''.
            bool castOut = @out is not null && !NDStringUfuncs.IsVString(@out);
            var dims = NDStringUfuncs.BroadcastDims(x1, x2, @out, where);
            NDArray result = @out is not null && !castOut
                ? @out
                : new NDArray(castOut ? DTypeRegistry.VString.default_descr() : resultDescr,
                              NDVStringComparison.ResultShape(dims, x1, x2, where), true);
            NDArray c1 = null, c2 = null;
            try
            {
                if (result.size != 0)
                {
                    // COPY_IF_OVERLAP: an out= sharing memory with a text input other than element for element gets that
                    // input copied first (the copy has its own allocator, recorded in the context below).
                    c1 = OverlapCopy(result, a1, ReferenceEquals(result, @out));
                    c2 = OverlapCopy(result, a2, ReferenceEquals(result, @out));
                    RunLocked(op, sizeOp, c1 ?? a1, c2 ?? a2, result, where, ruleDescr, name, textIndex,
                              fresh: !ReferenceEquals(result, @out));
                }
                if (!castOut)
                    return result;
                UnmanagedStorage.Copyto(@out, result, NPY_CASTING.NPY_SAME_KIND_CASTING, where);
                return @out;
            }
            catch
            {
                if (!ReferenceEquals(result, @out))
                    result.Dispose();
                result = null;
                throw;
            }
            finally
            {
                c1?.Dispose();
                c2?.Dispose();
                if (castOut && result is not null)
                    result.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's COPY_IF_OVERLAP for a text input against a caller's <c>StringDType</c> output: a copy of the input when
        ///     the two may share memory and are not the same elements (an exact alias is elementwise and needs none).
        /// </summary>
        /// <param name="result">The output.</param>
        /// <param name="input">A kernel input.</param>
        /// <param name="callerOut">True when <paramref name="result"/> is the caller's <c>out=</c> (a fresh one shares nothing).</param>
        /// <returns>The copy (owned by the caller), or null when none is needed.</returns>
        private static NDArray OverlapCopy(NDArray result, NDArray input, bool callerOut)
        {
            if (!callerOut || !NDStringUfuncs.IsVString(input))
                return null;
            if (NDMemOverlap.SolveMayShareMemory(result, input, maxWork: 0) == MemOverlap.No)
                return null;
            if (SameElements(result, input))
                return null;
            return np.copy(input);
        }

        /// <summary>Whether two arrays are the same elements: one address, one shape, one set of strides.</summary>
        /// <param name="a">The first array.</param>
        /// <param name="b">The second array.</param>
        /// <returns>True for an exact alias.</returns>
        private static bool SameElements(NDArray a, NDArray b)
        {
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            if ((byte*)a.Address + a.Shape.offset * (long)cell != (byte*)b.Address + b.Shape.offset * (long)cell)
                return false;
            var sa = a.Shape;
            var sb = b.Shape;
            if (sa.NDim != sb.NDim)
                return false;
            for (int i = 0; i < sa.NDim; i++)
                if (sa.dimensions[i] != sb.dimensions[i] || sa.dimensions[i] != 1 && sa.strides[i] != sb.strides[i])
                    return false;
            return true;
        }

        /// <summary>
        ///     Locks every allocator the loop touches (NumPy's <c>NpyString_acquire_allocators</c>), builds the context, runs
        ///     the size pass and the reservation for a fresh output, then the write pass.
        /// </summary>
        /// <param name="op">The write operation.</param>
        /// <param name="sizeOp">Its size pass.</param>
        /// <param name="a1">The first kernel input.</param>
        /// <param name="a2">The second kernel input.</param>
        /// <param name="result">The <c>StringDType</c> output.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="ruleDescr">The descriptor whose NA rule and default string the loop reads.</param>
        /// <param name="name">The ufunc name.</param>
        /// <param name="textIndex">The slot of the first text operand.</param>
        /// <param name="fresh">True when <paramref name="result"/> is a fresh (all-zero) array.</param>
        private static void RunLocked(WriteOp op, WriteOp sizeOp, NDArray a1, NDArray a2, NDArray result, NDArray where,
                                      DType ruleDescr, string name, int textIndex, bool fresh)
        {
            var locks = new List<VStringAllocator>(3);
            VStringAllocator in1 = NDStringUfuncs.IsVString(a1) ? a1.Storage.VStringSlice.Allocator : null;
            VStringAllocator in2 = NDStringUfuncs.IsVString(a2) ? a2.Storage.VStringSlice.Allocator : null;
            var outAllocator = result.Storage.VStringSlice.Allocator;
            if (in1 is not null) locks.Add(in1);
            if (in2 is not null) locks.Add(in2);
            locks.Add(outAllocator);
            byte[] defaultText = NDUtf8.Encode(ruleDescr.VString.default_string);
            var ctx = new VStringWriteContext
            {
                InA = textIndex == 0 ? in1 : in2,
                InB = textIndex == 0 ? in2 : in1,
                Out = outAllocator,
                Rule = NDVStringCells.RuleOf(ruleDescr),
                DefaultSize = defaultText.Length,
                InPlace = ReferenceEquals(outAllocator, in1) || ReferenceEquals(outAllocator, in2),
                FreshOut = fresh,
                Name = name,
            };
            var handle = GCHandle.Alloc(ctx);
            VStringAllocator.acquire_allocators(locks);
            try
            {
                fixed (byte* def = defaultText)
                {
                    ctx.DefaultText = defaultText.Length == 0 ? null : def;
                    // The inputs' arena bases, read under the locks; only the fresh fast paths use them (a fresh output's
                    // allocations never move an input's arena).
                    ctx.ArenaA = ctx.InA is null ? null : ctx.InA.ArenaBase;
                    ctx.ArenaB = ctx.InB is null ? null : ctx.InB.ArenaBase;
                    IntPtr h = GCHandle.ToIntPtr(handle);
                    // maximum / minimum copy one of their inputs, so over inputs that hold no arena or heap string at all
                    // every winner is a cell (short, empty or NA) and there is nothing to reserve: the size pass is skipped.
                    bool winnersInCells = sizeOp == WriteOp.MaxMinSize
                                          && !(in1 is not null && in1.HoldsLongStrings) && !(in2 is not null && in2.HoldsLongStrings);
                    if (fresh && outAllocator.ArenaUsed == 0 && !winnersInCells)
                    {
                        // The exact arena total of the outputs, reserved once (strings plan D12: no 25 % growth steps).
                        var size = ILKernelGenerator.GetVStringWriteKernel(sizeOp, textIndex) ?? ManagedWriteKernel(sizeOp, textIndex);
                        RunPass(size, a1, a2, null, null, &h);
                        outAllocator.TryReserve(ctx.Reserve);
                    }
                    var kernel = ILKernelGenerator.GetVStringWriteKernel(op, textIndex) ?? ManagedWriteKernel(op, textIndex);
                    RunPass(kernel, a1, a2, result, where, &h);
                }
            }
            finally
            {
                VStringAllocator.release_allocators(locks);
                handle.Free();
            }
        }

        /// <summary>
        ///     One pass of a writing (or size) kernel: the trivial loop when every operand walks one linear order next to the
        ///     output, else NDIter (a size pass iterates the two inputs alone, broadcast together).
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="a1">The first input.</param>
        /// <param name="a2">The second input.</param>
        /// <param name="result">The output, or null for a size pass (then <paramref name="where"/> is null too).</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="aux">The kernel's auxdata: a pointer to the context's handle.</param>
        private static void RunPass(NDInnerLoopFunc kernel, NDArray a1, NDArray a2, NDArray result, NDArray where, IntPtr* aux)
        {
            if (where is null && result is not null && TryTrivial(kernel, a1, a2, result, aux))
                return;
            if (result is null)
            {
                using var sizeIter = NDIterRef.AdvancedNew(2, new[] { a1, a2 }, NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.REFS_OK,
                    NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY }, opDescrs: new DType[2]);
                sizeIter.ForEach(kernel, aux);
                return;
            }
            var ops = where is null ? new[] { a1, a2, result } : new[] { a1, a2, result, where };
            var flags = where is null
                ? new[]
                {
                    NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.NO_BROADCAST,
                }
                : new[]
                {
                    NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY,
                    NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEMASKED | NDIterPerOpFlags.NO_BROADCAST,
                    NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK,
                };
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops, NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.REFS_OK,
                NPY_ORDER.NPY_KEEPORDER,
                NPY_CASTING.NPY_SAFE_CASTING, flags, opDescrs: new DType[ops.Length]);
            iter.ForEach(kernel, aux);
        }

        /// <summary>
        ///     NumPy's trivial loop for a writing kernel: no mask, each input one element (stride 0) or of the output's
        ///     dimensions in its contiguous order (any 1-D input at its own stride for a 0-d / 1-D output).
        /// </summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="a1">The first input.</param>
        /// <param name="a2">The second input.</param>
        /// <param name="result">The output.</param>
        /// <param name="aux">The kernel's auxdata.</param>
        /// <returns>True when the pass ran here.</returns>
        private static bool TryTrivial(NDInnerLoopFunc kernel, NDArray a1, NDArray a2, NDArray result, IntPtr* aux)
        {
            var rs = result.Shape;
            void** ptrs = stackalloc void*[3];
            long* strides = stackalloc long[3];
            const int cell = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;
            long i1 = a1.dtype.itemsize, i2 = a2.dtype.itemsize;
            if (rs.NDim <= 1)
            {
                if (!Linear1D(a1, rs) || !Linear1D(a2, rs))
                    return false;
                strides[0] = a1.size == 1 ? 0 : a1.Shape.strides[0] * i1;
                strides[1] = a2.size == 1 ? 0 : a2.Shape.strides[0] * i2;
                strides[2] = rs.NDim == 0 || result.size == 1 ? 0 : rs.strides[0] * (long)cell;
            }
            else
            {
                bool c = rs.IsContiguous, f = rs.IsFContiguous;
                if (!c && !f)
                    return false;
                if (!LinearOperand(a1, rs, ref c, ref f) || !LinearOperand(a2, rs, ref c, ref f) || !(c || f))
                    return false;
                strides[0] = a1.size == 1 ? 0 : i1;
                strides[1] = a2.size == 1 ? 0 : i2;
                strides[2] = cell;
            }
            ptrs[0] = (byte*)a1.Address + a1.Shape.offset * i1;
            ptrs[1] = (byte*)a2.Address + a2.Shape.offset * i2;
            ptrs[2] = (byte*)result.Address + rs.offset * (long)cell;
            kernel(ptrs, strides, result.size, aux);
            GC.KeepAlive(a1);
            GC.KeepAlive(a2);
            GC.KeepAlive(result);
            return true;
        }

        /// <summary>Whether an input walks linearly next to a multi-dimensional output (see the comparison's twin).</summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The output's shape.</param>
        /// <param name="c">In / out: every operand so far is C-contiguous.</param>
        /// <param name="f">In / out: every operand so far is F-contiguous.</param>
        /// <returns>False when the input needs the iterator.</returns>
        private static bool LinearOperand(NDArray x, Shape rs, ref bool c, ref bool f)
        {
            if (x.size == 1)
                return true;
            var xs = x.Shape;
            if (xs.NDim != rs.NDim)
                return false;
            for (int i = 0; i < xs.NDim; i++)
                if (xs.dimensions[i] != rs.dimensions[i])
                    return false;
            c &= xs.IsContiguous;
            f &= xs.IsFContiguous;
            return c || f;
        }

        /// <summary>Whether an input walks linearly next to a 0-d / 1-D output.</summary>
        /// <param name="x">The input.</param>
        /// <param name="rs">The output's shape.</param>
        /// <returns>False when the input needs the iterator.</returns>
        private static bool Linear1D(NDArray x, Shape rs)
            => x.size == 1 || x.ndim == 1 && rs.NDim == 1 && x.Shape.dimensions[0] == rs.dimensions[0];

        /// <summary>
        ///     A writing (or size) kernel as a managed loop (hosts without dynamic code; the IL kernels are tested against it):
        ///     the same element helpers.
        /// </summary>
        /// <param name="op">The operation.</param>
        /// <param name="textIndex">The slot of the first text operand.</param>
        /// <returns>The kernel.</returns>
        internal static NDInnerLoopFunc ManagedWriteKernel(WriteOp op, int textIndex)
            => (dataptrs, strides, count, auxdata) =>
            {
                var ctx = VStringWriteContext.FromAux(auxdata);
                byte* p0 = (byte*)dataptrs[textIndex], p1 = (byte*)dataptrs[1 - textIndex];
                long s0 = strides[textIndex], s1 = strides[1 - textIndex];
                bool sizePass = op is WriteOp.AddSize or WriteOp.MultiplySize or WriteOp.MultiplySizeUnsigned or WriteOp.MaxMinSize;
                byte* po = sizePass ? null : (byte*)dataptrs[2];
                long so = sizePass ? 0 : strides[2];
                for (long i = 0; i < count; i++, p0 += s0, p1 += s1, po += so)
                {
                    switch (op)
                    {
                        case WriteOp.Add: AddElement(p0, p1, po, ctx); break;
                        case WriteOp.AddSize: ctx.Reserve += AddSize(p0, p1, ctx); break;
                        case WriteOp.Multiply: MultiplyElement(p0, p1, po, ctx, false); break;
                        case WriteOp.MultiplyUnsigned: MultiplyElement(p0, p1, po, ctx, true); break;
                        case WriteOp.MultiplySize: ctx.Reserve += MultiplySize(p0, p1, ctx, false); break;
                        case WriteOp.MultiplySizeUnsigned: ctx.Reserve += MultiplySize(p0, p1, ctx, true); break;
                        case WriteOp.Maximum: MaxMinElement(p0, p1, po, ctx, true); break;
                        case WriteOp.Minimum: MaxMinElement(p0, p1, po, ctx, false); break;
                        default: ctx.Reserve += MaxMinSize(p0, p1, ctx); break;
                    }
                }
            };
    }
}
