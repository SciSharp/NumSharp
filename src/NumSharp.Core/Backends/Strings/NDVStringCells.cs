using System;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     How a <c>StringDType</c> loop treats a missing (NA) cell, decided once per call from ONE descriptor — NumPy's loops
    ///     read <c>has_null</c> (<c>na_object != NULL</c>), <c>has_nan_na</c> and <c>has_string_na</c> from the descriptor of
    ///     their FIRST operand (<c>descr1</c> / <c>s1descr</c> / <c>idescr</c>), whatever operand holds the NA.
    /// </summary>
    /// <remarks>
    ///     The first operand decides, which is observable: a <c>U</c> first operand is cast to a default
    ///     <c>StringDType()</c> (no NA object), so <c>np.equal(u, t)</c> reads an NA of <c>t</c> as <c>''</c> while
    ///     <c>np.equal(t, u)</c> follows <c>t</c>'s own rule (probed against 2.4.2).
    /// </remarks>
    internal enum VStringNARule : byte
    {
        /// <summary>No NA object (<c>has_null</c> false): an NA cell — reachable only through an operand of another
        ///     descriptor — reads as the default string, <c>''</c>.</summary>
        None = 0,

        /// <summary>A NaN-like NA object (<c>has_nan_na</c>): a comparison with an NA is <c>False</c> (<c>!=</c> too), a
        ///     concatenation or repetition propagates the NA, and the NA is the greatest element of an ordering.</summary>
        NaN = 1,

        /// <summary>Any other non-string NA object (<c>None</c>, a custom object): under <c>==</c> / <c>!=</c> an NA reads as
        ///     <c>''</c> (NumPy's loop stores the null-ness answer and then overwrites it with the compare of the loaded
        ///     strings, an NA loading as an empty one); every ordering and every concatenation with an NA raises NumPy's
        ///     <c>ValueError</c>.</summary>
        Null = 2,

        /// <summary>A string NA object (<c>has_string_na</c>): an NA cell reads as that string (the descriptor's
        ///     <c>default_string</c>).</summary>
        String = 3,
    }

    /// <summary>
    ///     The per-call context of a <c>StringDType</c> pair kernel (comparisons), passed as the kernel's auxdata: the
    ///     arena base of each operand's allocator (an arena cell holds an OFFSET into it), the default string of the
    ///     descriptor whose NA rule the call follows, and that rule.
    /// </summary>
    /// <remarks>
    ///     The arena bases are read once, with both allocators locked, before the loop: an arena moves only when its
    ///     allocator allocates, and a comparison never allocates in an input's allocator, so the bases stay valid for the
    ///     whole call. A call whose kernels allocate (add, multiply, maximum, minimum) uses its own context instead.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct VStringPairAux
    {
        /// <summary>The arena base of operand A's allocator (the kernel's first input), or null when it has no arena.</summary>
        internal byte* ArenaA;

        /// <summary>The arena base of operand B's allocator, or null.</summary>
        internal byte* ArenaB;

        /// <summary>The UTF-8 text an NA cell reads as under <see cref="VStringNARule.None"/> / <see cref="VStringNARule.String"/>
        ///     (the deciding descriptor's <c>default_string</c>); null when it is empty.</summary>
        internal byte* DefaultText;

        /// <summary>The byte length of <see cref="DefaultText"/>.</summary>
        internal long DefaultSize;

        /// <summary>The NA rule of the call.</summary>
        internal VStringNARule Rule;

        /// <summary>
        ///     The comparison NumPy names in its errors, as the index of <see cref="NDVStringCells.ComparisonNames"/> (0
        ///     <c>equal</c> … 5 <c>greater_equal</c>) — the ufunc of the CALL, not the kernel's operation after a swap.
        /// </summary>
        internal byte Ufunc;
    }

    /// <summary>
    ///     The cell-level primitives of NumSharp's <c>StringDType</c> (<c>T</c>) loops: NumPy's <c>NpyString_load</c> given an
    ///     arena base, <c>NpyString_cmp</c>, and the per-pair bodies the IL kernels of
    ///     <see cref="ILKernelGenerator.GetVStringCompareKernel"/> and <see cref="ILKernelGenerator.GetVStringTruthKernel"/>
    ///     call (strings plan D11 / D12).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>The short-cell fast path (D12).</b> A text of at most 15 bytes lives IN its 16-byte cell: bytes 0–14 hold it and
    ///     byte 15 is <c>0x60 | size</c> (<c>NPY_STRING_INITIALIZED | NPY_STRING_OUTSIDE_ARENA</c> and the length). The bytes
    ///     past the size are not guaranteed zero (a cell rewritten from an arena string keeps old offset bytes there), so the
    ///     fast path MASKS the text to its size. Two masked short cells then compare as NumPy's <c>NpyString_cmp</c> does
    ///     without reading anything else: equality is "same byte 15 (same size) and equal masked text", and the order is the
    ///     unsigned big-endian order of the masked cell read as a 128-bit number — the first differing byte decides, and for
    ///     a proper prefix the zero padding ties until byte 15, where the shorter size is smaller. Any other cell (an arena or
    ///     heap string, an NA, the all-zero cell of a never-written element) takes the per-pair slow path.
    ///     </para>
    ///     <para>
    ///     <b>No allocator object.</b> The primitives take raw arena bases, so an IL kernel calls them with no managed state;
    ///     the caller reads the bases with the allocators locked (<see cref="VStringAllocator.ArenaBase"/>).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDVStringCells
    {
        /// <summary>The 56 size bits of a vstring cell's size field (byte 15 is the flags byte).</summary>
        internal const ulong SizeMask = 0x00FF_FFFF_FFFF_FFFFUL;

        /// <summary>The flags-and-size byte of a short cell, with the size nibble cleared.</summary>
        private const uint ShortFlags = VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA;

        /// <summary>
        ///     The comparison ufuncs, in the order of <see cref="VStringPairAux.Ufunc"/> — what NumPy's errors name
        ///     (<c>'less' not supported for null values …</c>).
        /// </summary>
        internal static readonly string[] ComparisonNames =
        {
            "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        };

        /// <summary>
        ///     The text masks of a short cell, by size (0–15), in one unmanaged table of 48 words: entries 0–15 keep the text
        ///     bytes of the LOW word (bytes 0–7), entries 16–31 the text bytes of the HIGH word (bytes 8–14, byte 15 cleared),
        ///     entries 32–47 the high word's text bytes AND byte 15 (the order compare reads the size there).
        /// </summary>
        /// <remarks>An unmanaged table (built once, never freed — 384 bytes for the process): a static readonly pointer is a
        ///     JIT-time constant in optimized code, and the size nibble indexes it with no bounds check.</remarks>
        private static readonly ulong* s_masks = BuildMasks();

        /// <summary>A static empty text for empty vstring cells (NumPy's <c>EMPTY_STRING</c>): a non-null pointer to nothing.</summary>
        /// <remarks>The allocator's one shared constant (<see cref="VStringAllocator.EmptyString"/>), not a block of its own:
        ///     NumPy has a single <c>EMPTY_STRING</c>, and a reader-local copy was one more raw native allocation for the
        ///     chokepoint gate to pin. Never written through, never freed.</remarks>
        private static readonly byte* s_empty = VStringAllocator.EmptyString;

        /// <summary>Builds <see cref="s_masks"/>.</summary>
        /// <returns>The 48-word table.</returns>
        private static ulong* BuildMasks()
        {
            var t = (ulong*)NativeMemory.Alloc(48, sizeof(ulong));
            for (int s = 0; s < 16; s++)
            {
                // Bytes 0..min(s, 8)-1 of the low word; bytes 8..s-1 of the high word (s - 8 bytes from its bottom).
                t[s] = s >= 8 ? ulong.MaxValue : (1UL << (s * 8)) - 1;
                ulong hi = s <= 8 ? 0UL : (1UL << ((s - 8) * 8)) - 1;
                t[16 + s] = hi;
                t[32 + s] = hi | 0xFF00_0000_0000_0000UL;
            }
            return t;
        }

        /// <summary>
        ///     The NA rule of a <c>StringDType</c> descriptor (see <see cref="VStringNARule"/>): NumPy's <c>has_null</c>,
        ///     <c>has_string_na</c> and <c>has_nan_na</c>, tested in its order.
        /// </summary>
        /// <param name="descr">A <c>StringDType</c> descriptor.</param>
        /// <returns>The rule.</returns>
        /// <exception cref="ArgumentException"><paramref name="descr"/> is not a <c>StringDType</c> descriptor.</exception>
        internal static VStringNARule RuleOf(DType descr)
        {
            var p = descr?.VString ?? throw new ArgumentException($"{descr?.ToString(true)} is not a StringDType descriptor.", nameof(descr));
            if (!p.HasNAObject)
                return VStringNARule.None;
            if (p.has_string_na)
                return VStringNARule.String;
            return p.has_nan_na ? VStringNARule.NaN : VStringNARule.Null;
        }

        /// <summary>
        ///     NumPy's <c>NpyString_load</c> with the arena base given instead of the allocator: the text pointer and size of a
        ///     cell, or the NA flag.
        /// </summary>
        /// <param name="cell">The 16-byte cell.</param>
        /// <param name="arena">The arena base of the cell's allocator (null when it has no arena yet).</param>
        /// <param name="buf">Receives the first text byte (the cell itself for a short string; a non-null pointer to nothing
        ///     for <c>''</c>; null for an NA).</param>
        /// <param name="size">Receives the text size in bytes (0 for an NA).</param>
        /// <returns>0 for a string, 1 for an NA, -1 for an arena cell whose allocator has no arena (a corrupt cell: NumPy's
        ///     <c>Failed to load string</c>).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int Load(byte* cell, byte* arena, out byte* buf, out long size)
        {
            uint f = cell[15];
            if ((f & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags)
            {
                buf = cell;
                size = f & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK;
                return 0;
            }
            if ((f & VStringAllocator.NPY_STRING_MISSING) != 0)
            {
                buf = null;
                size = 0;
                return 1;
            }
            ulong vsize = Unsafe.ReadUnaligned<ulong>(cell + 8) & SizeMask;
            if (vsize == 0)
            {
                buf = s_empty;
                size = 0;
                return 0;
            }
            if ((f & VStringAllocator.NPY_STRING_OUTSIDE_ARENA) != 0)
            {
                // A heap string: the cell holds the pointer itself.
                buf = (byte*)Unsafe.ReadUnaligned<ulong>(cell);
            }
            else
            {
                if (arena == null)
                {
                    buf = null;
                    size = 0;
                    return -1;
                }
                buf = arena + Unsafe.ReadUnaligned<ulong>(cell);
            }
            size = (long)vsize;
            return 0;
        }

        /// <summary>
        ///     NumPy's <c>NpyString_cmp</c>: the common prefix byte by byte (unsigned — UTF-8 byte order is code-point order),
        ///     then the shorter text is the smaller.
        /// </summary>
        /// <param name="a">The first text.</param>
        /// <param name="sa">Its size.</param>
        /// <param name="b">The second text.</param>
        /// <param name="sb">Its size.</param>
        /// <returns>Negative, zero or positive.</returns>
        /// <remarks>One vectorized <see cref="MemoryExtensions.SequenceCompareTo{T}(ReadOnlySpan{T}, ReadOnlySpan{T})"/> for a
        ///     common prefix that fits a span (every real text); <see cref="OrderLong"/> past <see cref="int.MaxValue"/>
        ///     bytes. Measured on 24-byte arena texts with a 21-byte common prefix: 5.6 ns a pair with the cells' loads, where
        ///     an 8-byte word loop took 11 ns (its data-dependent exit mispredicts).</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static int Order(byte* a, long sa, byte* b, long sb)
        {
            long common = Math.Min(sa, sb);
            if (common > int.MaxValue)
                return OrderLong(a, sa, b, sb);
            int c = new ReadOnlySpan<byte>(a, (int)common).SequenceCompareTo(new ReadOnlySpan<byte>(b, (int)common));
            return c != 0 ? c : sa.CompareTo(sb);
        }

        /// <summary><see cref="Order"/> for a common prefix past <see cref="int.MaxValue"/> bytes: chunk by chunk.</summary>
        /// <param name="a">The first text.</param>
        /// <param name="sa">Its size.</param>
        /// <param name="b">The second text.</param>
        /// <param name="sb">Its size.</param>
        /// <returns>Negative, zero or positive.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int OrderLong(byte* a, long sa, byte* b, long sb)
        {
            long common = Math.Min(sa, sb);
            long done = 0;
            while (done < common)
            {
                int chunk = (int)Math.Min(common - done, int.MaxValue);
                int c = new ReadOnlySpan<byte>(a + done, chunk).SequenceCompareTo(new ReadOnlySpan<byte>(b + done, chunk));
                if (c != 0)
                    return c;
                done += chunk;
            }
            return sa.CompareTo(sb);
        }

        /// <summary>
        ///     Whether a cell can take the short fast path: a short string (byte 15's flags nibble is
        ///     <c>INITIALIZED | OUTSIDE_ARENA</c>, never with <c>MISSING</c>).
        /// </summary>
        /// <param name="flags">Byte 15 of the cell.</param>
        /// <returns>True for a short string.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static bool IsShort(uint flags) => (flags & VStringAllocator.NPY_STRING_FLAG_MASK) == ShortFlags;

        /// <summary>
        ///     <c>A == B</c> (or <c>A != B</c>) for one pair of cells: the short fast path, else <see cref="ComparePairSlow"/>.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="aux">The call's context (arena bases, NA rule, default string).</param>
        /// <param name="negate">True for <c>!=</c> (an emit-time constant of the calling kernel).</param>
        /// <returns>1 or 0.</returns>
        /// <exception cref="OutOfMemoryException">A corrupt arena cell (NumPy's <c>Failed to load string in equal</c>).</exception>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte Equal(byte* a, byte* b, VStringPairAux* aux, bool negate)
        {
            ulong ha = Unsafe.ReadUnaligned<ulong>(a + 8), hb = Unsafe.ReadUnaligned<ulong>(b + 8);
            uint fa = (uint)(ha >> 56), fb = (uint)(hb >> 56);
            if (IsShort(fa) && IsShort(fb))
            {
                // Byte 15 carries the size: two sizes differ exactly when the bytes differ. With equal sizes one mask
                // covers both texts, and the bytes past the size (stale arena bytes possibly) never count.
                int s = (int)(fa & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
                ulong d = ((Unsafe.ReadUnaligned<ulong>(a) ^ Unsafe.ReadUnaligned<ulong>(b)) & s_masks[s])
                          | ((ha ^ hb) & s_masks[16 + s]);
                bool equal = fa == fb & d == 0;
                return (byte)((equal ? 1 : 0) ^ (negate ? 1 : 0));
            }
            return ComparePairSlow(a, b, aux, negate ? TextCompareOp.NotEqual : TextCompareOp.Equal);
        }

        /// <summary>
        ///     <c>A &lt; B</c> (or <c>A &lt;= B</c>) for one pair of cells: the short fast path (the masked cells as 128-bit
        ///     big-endian numbers), else <see cref="ComparePairSlow"/>.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="aux">The call's context.</param>
        /// <param name="orEqual">True for <c>&lt;=</c> (an emit-time constant of the calling kernel).</param>
        /// <returns>1 or 0.</returns>
        /// <exception cref="ValueError">An NA under the <see cref="VStringNARule.Null"/> rule (NumPy's
        ///     <c>'less' not supported for null values that are not nan-like or strings.</c>, naming the call's ufunc).</exception>
        /// <exception cref="OutOfMemoryException">A corrupt arena cell.</exception>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte Less(byte* a, byte* b, VStringPairAux* aux, bool orEqual)
        {
            ulong ha = Unsafe.ReadUnaligned<ulong>(a + 8), hb = Unsafe.ReadUnaligned<ulong>(b + 8);
            uint fa = (uint)(ha >> 56), fb = (uint)(hb >> 56);
            if (IsShort(fa) && IsShort(fb))
            {
                int sa = (int)(fa & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
                int sb = (int)(fb & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
                ulong la = Unsafe.ReadUnaligned<ulong>(a) & s_masks[sa];
                ulong lb = Unsafe.ReadUnaligned<ulong>(b) & s_masks[sb];
                // Bytes 0-7 decide when they differ; otherwise bytes 8-15 (the text tail, then the size in byte 15).
                // Both keys are formed and one is selected, so the per-element branch the data would steer is a select.
                bool low = la != lb;
                ulong ka = low ? BinaryPrimitives.ReverseEndianness(la) : BinaryPrimitives.ReverseEndianness(ha & s_masks[32 + sa]);
                ulong kb = low ? BinaryPrimitives.ReverseEndianness(lb) : BinaryPrimitives.ReverseEndianness(hb & s_masks[32 + sb]);
                bool r = orEqual ? ka <= kb : ka < kb;
                return r ? (byte)1 : (byte)0;
            }
            // Two non-NA cells, at least one outside its cell: their first eight bytes as big-endian words decide most pairs
            // of different texts without the out-of-line path (an arena text against a short scalar, random arena texts).
            if (((fa | fb) & VStringAllocator.NPY_STRING_MISSING) == 0
                && TryFirstWord(a, fa, ha, aux->ArenaA, out ulong wa) & TryFirstWord(b, fb, hb, aux->ArenaB, out ulong wb)
                && wa != wb)
                return wa < wb ? (byte)1 : (byte)0;
            return ComparePairSlow(a, b, aux, orEqual ? TextCompareOp.LessEqual : TextCompareOp.Less);
        }

        /// <summary>
        ///     The first eight bytes of a non-NA cell's text as a big-endian number, a text shorter than eight bytes zero-padded
        ///     (so a proper prefix orders first, as <c>NpyString_cmp</c> orders it): two different words decide the order of
        ///     their texts; equal words decide nothing.
        /// </summary>
        /// <param name="cell">The cell (not NA).</param>
        /// <param name="f">Its byte 15.</param>
        /// <param name="h">Its high word.</param>
        /// <param name="arena">Its allocator's arena base.</param>
        /// <param name="word">Receives the word (0 for an empty text).</param>
        /// <returns>False for an arena cell whose allocator has no arena (a corrupt cell: the caller's out-of-line path raises
        ///     NumPy's error for it).</returns>
        /// <remarks>A short cell has 16 readable bytes and an arena or heap text at least 16, so the 8-byte read never leaves
        ///     its text; an emptied vstring (size 0) reads nothing.</remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static bool TryFirstWord(byte* cell, uint f, ulong h, byte* arena, out ulong word)
        {
            if (IsShort(f))
            {
                word = BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(cell) & s_masks[Math.Min(f & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK, 8u)]);
                return true;
            }
            word = 0;
            if ((h & SizeMask) == 0)
                return true;
            byte* text;
            if ((f & VStringAllocator.NPY_STRING_OUTSIDE_ARENA) != 0)
                text = (byte*)Unsafe.ReadUnaligned<ulong>(cell);
            else if (arena != null)
                text = arena + Unsafe.ReadUnaligned<ulong>(cell);
            else
                return false;
            word = BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(text));
            return true;
        }

        /// <summary>
        ///     <c>A &lt; B</c> for two SHORT cells, branch-free: the masked cells as 128-bit big-endian numbers (bytes 0–7
        ///     decide when they differ, else the high word's text bytes and then byte 15, the size — the flags nibbles of two
        ///     short cells are alike), both keys formed and one selected, so the comparison the data would steer is a select.
        /// </summary>
        /// <param name="a">Operand A's cell (short).</param>
        /// <param name="ha">A's high word (bytes 8–15).</param>
        /// <param name="fa">A's byte 15.</param>
        /// <param name="b">Operand B's cell (short).</param>
        /// <param name="hb">B's high word.</param>
        /// <param name="fb">B's byte 15.</param>
        /// <returns>True when A's text orders before B's (<c>NpyString_cmp &lt; 0</c>).</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static bool ShortLess(byte* a, ulong ha, uint fa, byte* b, ulong hb, uint fb)
        {
            int sa = (int)(fa & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
            int sb = (int)(fb & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
            ulong xa = BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(a) & s_masks[sa]);
            ulong xb = BinaryPrimitives.ReverseEndianness(Unsafe.ReadUnaligned<ulong>(b) & s_masks[sb]);
            ulong ya = BinaryPrimitives.ReverseEndianness(ha & s_masks[32 + sa]);
            ulong yb = BinaryPrimitives.ReverseEndianness(hb & s_masks[32 + sb]);
            // The 128-bit order (xa:ya) < (xb:yb) with the non-short-circuit operators: three flags combined, no branch
            // (a select between the two key pairs compiled to a branch on the data, which random text mispredicts).
            return (xa < xb) | ((xa == xb) & (ya < yb));
        }

        /// <summary>
        ///     <c>c ? x : y</c> for two pointers as mask arithmetic, so the choice the data steers cannot become a branch (the
        ///     JIT compiles a pointer ternary inside a large inlined kernel to a conditional jump).
        /// </summary>
        /// <param name="c">The condition.</param>
        /// <param name="x">The pointer for true.</param>
        /// <param name="y">The pointer for false.</param>
        /// <returns><paramref name="x"/> or <paramref name="y"/>.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte* Select(bool c, byte* x, byte* y)
        {
            nuint m = (nuint)0 - (nuint)(c ? 1 : 0);
            return (byte*)(((nuint)x & m) | ((nuint)y & ~m));
        }

        /// <summary>
        ///     The per-pair path of NumPy's <c>string_comparison_strided_loop</c>: both cells loaded, the NA rule of the call,
        ///     then <c>NpyString_cmp</c>.
        /// </summary>
        /// <param name="a">Operand A's cell.</param>
        /// <param name="b">Operand B's cell.</param>
        /// <param name="aux">The call's context.</param>
        /// <param name="op">The kernel's operation (after the operand swap of <c>greater</c> / <c>greater_equal</c>).</param>
        /// <returns>1 or 0.</returns>
        /// <exception cref="ValueError">An NA in an ordering under the <see cref="VStringNARule.Null"/> rule.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt arena cell: <c>Failed to load string in {ufunc}</c>.</exception>
        /// <remarks>
        ///     Lean on purpose: two texts compare inline here, and every pair with an NA or a corrupt cell goes on to
        ///     <see cref="ComparePairNA"/>. The NA handling's interpolated messages used to sit in this method, whose frame then
        ///     zeroed a string handler on every call: 17.4 ns a pair of 24-byte arena texts, against 5.6 ns for the loads and
        ///     the compare alone. The size and first-word tests serve <c>==</c> / <c>!=</c> only: an ordering reaches this
        ///     method after <see cref="Less"/> found the first words equal, so a first-word test could only repeat that work
        ///     (the mutation pass showed the earlier form never decided an ordering), and a size difference does not order.
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static byte ComparePairSlow(byte* a, byte* b, VStringPairAux* aux, TextCompareOp op)
        {
            int na = Load(a, aux->ArenaA, out byte* ba, out long sa);
            int nb = Load(b, aux->ArenaB, out byte* bb, out long sb);
            if ((na | nb) != 0)
                return ComparePairNA(na, nb, ba, sa, bb, sb, aux, op);
            // NpyString_cmp is 0 only for two texts of one size: an EQUALITY pair of different sizes is unequal without
            // reading either text (a short text against one outside its cell always is), and two texts of one size of eight
            // bytes or more whose first eight bytes differ are unequal without the span compare. An ordering skips both
            // tests: Less compared the first words inline before calling here, so an ordering arrives with equal first words
            // (or with an NA or a corrupt cell, handled above).
            if (op <= TextCompareOp.NotEqual
                && (sa != sb || (sa >= 8 && Unsafe.ReadUnaligned<ulong>(ba) != Unsafe.ReadUnaligned<ulong>(bb))))
                return op == TextCompareOp.NotEqual ? (byte)1 : (byte)0;
            return Decide(Order(ba, sa, bb, sb), op);
        }

        /// <summary>
        ///     The flag a comparison writes for one <c>NpyString_cmp</c> result, branch-free on the data: the sign of
        ///     <paramref name="cmp"/> (-1, 0, 1) indexes a three-bit truth table per operation (bit 0 less, bit 1 equal, bit 2
        ///     greater — <c>==</c> 0b010, <c>!=</c> 0b101, <c>&lt;</c> 0b001, <c>&lt;=</c> 0b011, packed as one nibble per
        ///     <see cref="TextCompareOp"/> value).
        /// </summary>
        /// <param name="cmp">The order of A against B.</param>
        /// <param name="op">The kernel's operation.</param>
        /// <returns>1 or 0.</returns>
        /// <remarks>
        ///     A switch on <paramref name="op"/> whose arms test <paramref name="cmp"/> compiled to a jump table and a
        ///     conditional branch on the order itself, which random text mispredicts half the time: 17.2 ns a pair of 24-byte
        ///     arena texts, against 7.7 ns with the branch-free form (measured in one process, same loads and compare).
        /// </remarks>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        private static byte Decide(int cmp, TextCompareOp op)
        {
            // sign(cmp) without a branch: -1 for a negative cmp (int.MinValue included), 1 for a positive one, 0 for 0.
            int sign = (cmp >> 31) | (int)((uint)-cmp >> 31);
            int table = (0x3152 >> ((int)op * 4)) & 0xF;
            return (byte)((table >> (sign + 1)) & 1);
        }

        /// <summary>
        ///     <see cref="ComparePairSlow"/>'s pairs with an NA or a corrupt cell: the load error, then the call's NA rule (see
        ///     <see cref="VStringNARule"/>), then the compare of what the NA reads as.
        /// </summary>
        /// <param name="na">A's load result (1 NA, -1 corrupt, 0 text).</param>
        /// <param name="nb">B's load result.</param>
        /// <param name="ba">A's text (when <paramref name="na"/> is 0).</param>
        /// <param name="sa">A's size.</param>
        /// <param name="bb">B's text.</param>
        /// <param name="sb">B's size.</param>
        /// <param name="aux">The call's context.</param>
        /// <param name="op">The kernel's operation.</param>
        /// <returns>1 or 0.</returns>
        /// <exception cref="ValueError">An NA in an ordering under the <see cref="VStringNARule.Null"/> rule.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt arena cell: <c>Failed to load string in {ufunc}</c>.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static byte ComparePairNA(int na, int nb, byte* ba, long sa, byte* bb, long sb, VStringPairAux* aux, TextCompareOp op)
        {
            if (na < 0 || nb < 0)
                throw new OutOfMemoryException($"Failed to load string in {ComparisonNames[aux->Ufunc]}");
            if (na != 0 || nb != 0)
            {
                switch (aux->Rule)
                {
                    case VStringNARule.NaN:
                        // Every comparison with a NaN-like NA is False, != included.
                        return 0;
                    case VStringNARule.Null:
                        // NumPy's eq / ne branch stores the null-ness answer and then FALLS THROUGH to NpyString_cmp, which
                        // overwrites it with the compare of the loaded strings — an NA loads as {size 0, NULL} — so the NA
                        // reads as '' (probed: '' == None is True, 'ab' == None False, None == None True).
                        if (op is TextCompareOp.Equal or TextCompareOp.NotEqual)
                            break;
                        throw new ValueError(
                            $"'{ComparisonNames[aux->Ufunc]}' not supported for null values that are not nan-like or strings.");
                    default:
                        // No NA object (the NA belongs to an operand of another descriptor) or a string NA: the default string.
                        if (na != 0)
                        {
                            ba = aux->DefaultText;
                            sa = aux->DefaultSize;
                        }
                        if (nb != 0)
                        {
                            bb = aux->DefaultText;
                            sb = aux->DefaultSize;
                        }
                        break;
                }
            }
            return Decide(Order(ba, sa, bb, sb), op);
        }

        /// <summary>
        ///     The truth of one cell: a non-NA text is true when non-empty; an NA is <paramref name="naTruth"/> (the cast and
        ///     the nonzero route of NumPy differ there, see <see cref="NaTruth"/>). Branch-free over the cell kinds.
        /// </summary>
        /// <param name="cell">The cell.</param>
        /// <param name="naTruth">The truth of an NA cell (0 or 1).</param>
        /// <returns>1 or 0.</returns>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static byte Truth(byte* cell, byte naTruth)
        {
            ulong h = Unsafe.ReadUnaligned<ulong>(cell + 8);
            uint f = (uint)(h >> 56);
            // A short cell's size is its low nibble; any other cell's size is the 56-bit field (0 for the all-zero
            // never-written cell and for an emptied arena cell).
            ulong size = IsShort(f) ? f & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK : h & SizeMask;
            byte t = size != 0 ? (byte)1 : (byte)0;
            return (f & VStringAllocator.NPY_STRING_MISSING) != 0 ? naTruth : t;
        }

        /// <summary>
        ///     The four bytes of four cells' flags, indexed by the 4-bit move mask of an <c>unpackhi</c> of two cell pairs:
        ///     bit 0 is cell 0, bit 1 cell 2, bit 2 cell 1, bit 3 cell 3 (<c>vpunpckhqdq</c> takes the high word of each
        ///     128-bit lane, pair by pair), and the entry holds the bytes for cells 0, 1, 2, 3 in order (0 or 1 each).
        /// </summary>
        /// <remarks>An unmanaged table (built once, never freed — 64 bytes): a static readonly pointer is a JIT-time constant
        ///     in optimized code.</remarks>
        private static readonly uint* s_unpackedFlagBytes = BuildUnpackedFlagBytes();

        /// <summary>Builds <see cref="s_unpackedFlagBytes"/>.</summary>
        /// <returns>The 16-entry table.</returns>
        private static uint* BuildUnpackedFlagBytes()
        {
            var t = (uint*)NativeMemory.Alloc(16, sizeof(uint));
            for (uint m = 0; m < 16; m++)
            {
                uint c0 = m & 1, c2 = (m >> 1) & 1, c1 = (m >> 2) & 1, c3 = (m >> 3) & 1;
                t[m] = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
            }
            return t;
        }

        /// <summary>
        ///     <see cref="Truth"/> of <paramref name="n"/> CONTIGUOUS cells into <paramref name="n"/> contiguous bools, four cells
        ///     a step on AVX2 hosts: the high words of two cell pairs gathered by one unpack, the flags, the size (the nibble of
        ///     a short cell, the 56-bit field of any other), the NA test and the NA truth all lane-wise, then one move mask and
        ///     a table lookup write four bools. The tail, and every element on other hosts, takes the scalar <see cref="Truth"/>.
        /// </summary>
        /// <param name="cells">The first cell.</param>
        /// <param name="flags">The first bool.</param>
        /// <param name="n">The number of cells.</param>
        /// <param name="naTruth">The truth of an NA cell (0 or 1).</param>
        /// <remarks>Reads only the cells (no arena), like <see cref="Truth"/>; the per-element scalar form ran 1.1 ns an
        ///     element over 100,000 cells, where NumPy's <c>string_to_bool</c> takes 1.7.</remarks>
        internal static void TruthRun(byte* cells, byte* flags, long n, byte naTruth)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var flagNibble = Vector256.Create((ulong)VStringAllocator.NPY_STRING_FLAG_MASK);
                var shortFlags = Vector256.Create((ulong)ShortFlags);
                var sizeNibble = Vector256.Create((ulong)VStringAllocator.NPY_SHORT_STRING_SIZE_MASK);
                var size56 = Vector256.Create(SizeMask);
                var missing = Vector256.Create((ulong)VStringAllocator.NPY_STRING_MISSING);
                var naVec = naTruth != 0 ? Vector256<ulong>.AllBitsSet : Vector256<ulong>.Zero;
                for (; i + 4 <= n; i += 4)
                {
                    byte* p = cells + i * SIZEOF_CELL;
                    // [hi0, hi2, hi1, hi3]: the high word of each 128-bit lane of the two pairs
                    var hi = Avx2.UnpackHigh(Avx.LoadVector256((ulong*)p), Avx.LoadVector256((ulong*)(p + 2 * SIZEOF_CELL)));
                    var f = Avx2.ShiftRightLogical(hi, 56);
                    var isShort = Avx2.CompareEqual(f & flagNibble, shortFlags);
                    // a byte blend on whole-lane masks picks a whole 64-bit lane: the short size where the cell is short
                    var size = Avx2.BlendVariable(hi & size56, f & sizeNibble, isShort);
                    var nonEmpty = ~Avx2.CompareEqual(size, Vector256<ulong>.Zero);
                    var isNA = Avx2.CompareEqual(f & missing, missing);
                    var truth = Avx2.BlendVariable(nonEmpty, naVec, isNA);
                    *(uint*)(flags + i) = s_unpackedFlagBytes[Avx.MoveMask(truth.AsDouble())];
                }
            }
            for (; i < n; i++)
                flags[i] = Truth(cells + i * SIZEOF_CELL, naTruth);
        }

        /// <summary>
        ///     NumPy's <c>isnan</c> of <paramref name="n"/> CONTIGUOUS cells into contiguous bools (the NA bit, the top bit of
        ///     byte 15), four cells a step on AVX2 hosts: the high words of two cell pairs gathered by one unpack, whose sign
        ///     bits ARE the NA bits, one move mask and a table lookup. The caller runs it only for a NaN-like NA descriptor.
        /// </summary>
        /// <param name="cells">The first cell.</param>
        /// <param name="flags">The first bool.</param>
        /// <param name="n">The number of cells.</param>
        internal static void NARun(byte* cells, byte* flags, long n)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 4 <= n; i += 4)
                {
                    byte* p = cells + i * SIZEOF_CELL;
                    var hi = Avx2.UnpackHigh(Avx.LoadVector256((ulong*)p), Avx.LoadVector256((ulong*)(p + 2 * SIZEOF_CELL)));
                    *(uint*)(flags + i) = s_unpackedFlagBytes[Avx.MoveMask(hi.AsDouble())];
                }
            }
            for (; i < n; i++)
                flags[i] = (byte)(cells[i * SIZEOF_CELL + 15] >> 7);
        }

        /// <summary>The size of a cell, in bytes (NumPy's <c>sizeof(npy_packed_static_string)</c>).</summary>
        private const int SIZEOF_CELL = VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING;

        /// <summary>
        ///     Whether any of <paramref name="n"/> cells is a non-empty ARENA string (not NA, not outside the arena, size above
        ///     0) — the one kind of cell <c>NpyString_load</c> cannot load when its allocator has no arena. A caller whose
        ///     allocator has no arena runs it to keep NumPy's <c>Failed to load string in …</c> for such a (corrupt) cell
        ///     while reading only the cells otherwise; four contiguous cells a step on AVX2 hosts.
        /// </summary>
        /// <param name="cells">The first cell.</param>
        /// <param name="stride">The byte stride between cells.</param>
        /// <param name="n">The number of cells.</param>
        /// <returns>True when such a cell exists.</returns>
        internal static bool AnyArenaCell(byte* cells, long stride, long n)
        {
            long i = 0;
            if (Avx2.IsSupported && stride == SIZEOF_CELL)
            {
                // NA and OUTSIDE_ARENA both clear (a short cell has OUTSIDE_ARENA set) and a non-zero size field
                var naOrOutside = Vector256.Create((ulong)(VStringAllocator.NPY_STRING_MISSING | VStringAllocator.NPY_STRING_OUTSIDE_ARENA) << 56);
                var size56 = Vector256.Create(SizeMask);
                for (; i + 4 <= n; i += 4)
                {
                    byte* p = cells + i * SIZEOF_CELL;
                    var hi = Avx2.UnpackHigh(Avx.LoadVector256((ulong*)p), Avx.LoadVector256((ulong*)(p + 2 * SIZEOF_CELL)));
                    var inArena = Avx2.CompareEqual(hi & naOrOutside, Vector256<ulong>.Zero);
                    var sized = ~Avx2.CompareEqual(hi & size56, Vector256<ulong>.Zero);
                    if (Avx.MoveMask((inArena & sized).AsDouble()) != 0)
                        return true;
                }
            }
            for (; i < n; i++)
            {
                byte* c = cells + i * stride;
                ulong h = Unsafe.ReadUnaligned<ulong>(c + 8);
                uint f = (uint)(h >> 56);
                if ((f & (VStringAllocator.NPY_STRING_MISSING | VStringAllocator.NPY_STRING_OUTSIDE_ARENA)) == 0 && (h & SizeMask) != 0)
                    return true;
            }
            return false;
        }

        /// <summary>
        ///     The truth of an NA cell of <paramref name="descr"/> for one of NumPy's two routes: the string-to-bool CAST
        ///     (<c>string_to_bool</c>: a NaN-like NA is True, another non-string NA False, and — NumPy's own inverted test — a
        ///     string NA is True exactly when that string is EMPTY) or the dtype's NONZERO function (<c>nonzero</c>: a NaN-like
        ///     NA is True, any other NA False).
        /// </summary>
        /// <param name="descr">The array's <c>StringDType</c> descriptor.</param>
        /// <param name="cast">True for the cast route (<c>astype(bool)</c>, <c>any</c> / <c>all</c>, the logical ufuncs,
        ///     <c>where(c, x, y)</c>), false for the nonzero route (<c>nonzero</c>, <c>count_nonzero</c> without an axis,
        ///     <c>argwhere</c>, <c>flatnonzero</c>, <c>where(c)</c>).</param>
        /// <returns>1 or 0.</returns>
        internal static byte NaTruth(DType descr, bool cast)
        {
            var p = descr.VString;
            if (p.HasNAObject && !p.has_string_na)
                return p.has_nan_na ? (byte)1 : (byte)0;
            // The cast reads `default_string->size == 0` (NumPy's inverted test); the nonzero function reads the NA cell's
            // size, which is always 0.
            return cast && p.default_string.Length == 0 ? (byte)1 : (byte)0;
        }
    }
}
