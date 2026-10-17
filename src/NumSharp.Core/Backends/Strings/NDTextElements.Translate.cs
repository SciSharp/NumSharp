using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// NDTextElements.Translate.cs — the S / U element bodies of np.strings.translate (strings phase 2, P2.6 milestone 3)
// =============================================================================
//
// One output element per input element, of the same width: the element's scalar (np.bytes_ / np.str_: trailing NULs
// dropped) through bytes.translate / str.translate (NDTranslate.cs), the result cut to the width and zero-padded, as
// NumPy's setitem stores it.
//
// S: bytes.translate's trans_table[256] (-1 = delete): a map loop without deletions (the output keeps the length), else a
//    branch-free compaction (every byte written, the cursor advanced past kept ones). When NUL maps to NUL or is deleted
//    (NulPreserved, nearly every table) the whole width is mapped: translating the trailing NULs gives exactly the zero
//    padding, so no search for the text's end and no trip count that varies per element. bytes.translate's errors are the
//    call's (table / delete buffers), raised by the driver before any element.
// U: with a table whose every character past its dense range is kept and whose NUL maps to itself (VectorOk), eight units
//    per AVX2 gather of the dense entries — an out-of-range lane keeps its unit — and a lane whose entry is negative
//    (delete, a replacement, an error) or whose unit is past U+10FFFF sends the element to the scalar loop. The scalar loop
//    (TranslatePoints) runs branch-free while the entries are code points or deletions, then reads replacements from the
//    table's pinned pool and anything past the dense range from the managed table (never, when the table keeps everything
//    there: HighIdentity); an element that raises sets the call's error word and the driver finds NumPy's first error by a
//    C-order walk (NDCharObjectPath.Translate.cs).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDTextElements
    {
        /// <summary>
        ///     One element of NumPy's <c>translate</c>: <paramref name="dst"/> (an element of the same width) receives the
        ///     translated text of <paramref name="src"/>, cut to the width and zero-padded (see the file header).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="byte"/> for <c>S</c> (<c>bytes.translate</c>), <see cref="uint"/> for <c>U</c>
        ///     (<c>str.translate</c>).</typeparam>
        /// <param name="src">The input element.</param>
        /// <param name="width">The element width in bytes (a constant in the IL shell).</param>
        /// <param name="dst">The output element (a fresh array's: never <paramref name="src"/>).</param>
        /// <param name="st">The call's state: the compiled table and the error word (set to 1 when the element raises).</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void Translate<TUnit>(byte* src, int width, byte* dst, TranslateState* st) where TUnit : unmanaged
        {
            if (typeof(TUnit) == typeof(byte))
            {
                TranslateBytes(src, width, dst, st);
                return;
            }
            if (Avx2.IsSupported && (st->Flags & (int)TranslateFlags.VectorOk) != 0 && TranslateGatherUnits(src, width, dst, st))
                return;
            TranslateUnits((uint*)src, width >> 2, (uint*)dst, st);
        }

        /// <summary>
        ///     <c>bytes.translate</c> of an <c>S</c> element: the bytes before the trailing NULs mapped (or deleted) through the
        ///     256 entries, the rest of the output zero.
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="st">The call's state (the entries and the no-delete flag).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void TranslateBytes(byte* src, int width, byte* dst, TranslateState* st)
        {
            int* trans = st->Dense;
            int k;
            if ((st->Flags & (int)TranslateFlags.NulPreserved) != 0)
            {
                // NUL maps to NUL or is deleted: the trailing NULs translated give zeros at the end or nothing, exactly what
                // the zero padding of the stripped text gives, so the whole width is mapped with no search for the text's
                // end (a constant trip count: no length-dependent branch per element).
                if ((st->Flags & (int)TranslateFlags.NoDeletes) != 0)
                {
                    for (k = 0; k < width; k++)
                        dst[k] = (byte)trans[src[k]];
                    return;
                }
                k = 0;
                for (int i = 0; i < width; i++)
                {
                    int v = trans[src[i]];
                    dst[k] = (byte)v;
                    k += (v >> 31) + 1;
                }
                for (; k < width; k++)
                    dst[k] = 0;
                return;
            }
            // np.bytes_ drops the trailing NULs: they are not translated (a table mapping NUL to 'Z' changes only inner NULs).
            int n = width;
            while (n > 0 && src[n - 1] == 0)
                n--;
            if ((st->Flags & (int)TranslateFlags.NoDeletes) != 0)
            {
                for (k = 0; k < n; k++)
                    dst[k] = (byte)trans[src[k]];
            }
            else
            {
                k = 0;
                for (int i = 0; i < n; i++)
                {
                    int v = trans[src[i]];
                    // Written whatever it is; the cursor moves past a kept byte only (v >> 31 is -1 for a deletion), and the
                    // next byte or the zero fill overwrites a deleted one.
                    dst[k] = (byte)v;
                    k += (v >> 31) + 1;
                }
            }
            for (; k < width; k++)
                dst[k] = 0;
        }

        /// <summary>
        ///     The gather path of a <c>U</c> element (a <see cref="TranslateFlags.VectorOk"/> table): eight units per chunk
        ///     through the dense entries, a padded single chunk up to 32 bytes, else whole chunks and an overlapping last one
        ///     (a unit's output depends on the unit alone, so the overlap rewrites the same values).
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="dst">The output element.</param>
        /// <param name="st">The call's state.</param>
        /// <returns>False when a unit needs the scalar loop (a negative entry, a unit past U+10FFFF); the output may be
        ///     partly written, the scalar loop rewrites all of it.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TranslateGatherUnits(byte* src, int width, byte* dst, TranslateState* st)
        {
            var limit = Vector256.Create((uint)st->DenseLength);
            int* dense = st->Dense;
            if (width <= 32)
            {
                // The padded lanes are NUL, which a VectorOk table maps to NUL; StorePadded32 writes none of them.
                if (!TranslateChunk(LoadPadded32(src, width).AsUInt32(), limit, dense, out var r0))
                    return false;
                StorePadded32(dst, width, r0.AsByte());
                return true;
            }
            int o = 0;
            for (; o + 32 <= width; o += 32)
            {
                if (!TranslateChunk(Vector256.Load((uint*)(src + o)), limit, dense, out var r))
                    return false;
                r.Store((uint*)(dst + o));
            }
            if (o < width)
            {
                int t = width - 32;
                if (!TranslateChunk(Vector256.Load((uint*)(src + t)), limit, dense, out var r))
                    return false;
                r.Store((uint*)(dst + t));
            }
            return true;
        }

        /// <summary>
        ///     Eight units through the dense entries: a unit below the dense length takes its entry (one gather), any other is
        ///     kept; a negative entry or a unit past U+10FFFF fails the chunk.
        /// </summary>
        /// <param name="u">The units.</param>
        /// <param name="limit">The dense length in every lane.</param>
        /// <param name="dense">The dense entries.</param>
        /// <param name="r">Receives the translated units.</param>
        /// <returns>False when a lane needs the scalar loop.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool TranslateChunk(Vector256<uint> u, Vector256<uint> limit, int* dense, out Vector256<uint> r)
        {
            var inRange = Vector256.LessThan(u, limit);
            // Out-of-range lanes gather entry 0 (a real address) and are replaced by their unit below.
            var g = Avx2.GatherVector256(dense, (u & inRange).AsInt32(), 4).AsUInt32();
            r = Vector256.ConditionalSelect(inRange, g, u);
            var bad = Vector256.LessThan(r.AsInt32(), Vector256<int>.Zero).AsUInt32() | Vector256.GreaterThan(u, Vector256.Create(0x10FFFFu));
            return bad == Vector256<uint>.Zero;
        }

        /// <summary>
        ///     The scalar <c>str.translate</c> of a <c>U</c> element: its <c>np.str_</c> (trailing NULs dropped; one unit past
        ///     U+10FFFF is CPython's SystemError at the scalar), the characters translated into the output up to its width, the
        ///     rest zero.
        /// </summary>
        /// <param name="s">The element's units.</param>
        /// <param name="units">Its width in units.</param>
        /// <param name="d">The output element's units.</param>
        /// <param name="st">The call's state (the error word is set to 1 when the element raises).</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void TranslateUnits(uint* s, int units, uint* d, TranslateState* st)
        {
            if ((st->Flags & (int)TranslateFlags.NulPreserved) != 0)
            {
                // NUL maps to NUL or is deleted: the whole width translates as the stripped text plus padding would, so
                // while every entry is a code point or a deletion the element is one branch-free pass of constant length
                // (the width is a constant in the IL shell). A unit needing more — a replacement, an error, the managed
                // table, a unit past U+10FFFF (which may be a one-unit text's SystemError) — falls back to the exact path.
                int* dense = st->Dense;
                uint dl = (uint)st->DenseLength;
                bool high = (st->Flags & (int)TranslateFlags.HighIdentity) != 0;
                int k = 0;
                int i = 0;
                for (; i < units; i++)
                {
                    uint c = s[i];
                    int e;
                    if (c < dl)
                        e = dense[c];
                    else if (high && c <= 0x10FFFF)
                        e = (int)c;
                    else
                        break;
                    if (e < StrTranslateTable.Delete)
                        break;
                    d[k] = (uint)e;
                    k += (e >> 31) + 1;
                }
                if (i == units)
                {
                    for (; k < units; k++)
                        d[k] = 0;
                    return;
                }
            }
            int n = units;
            while (n > 0 && s[n - 1] == 0)
                n--;
            if (n == 1 && s[0] > 0x10FFFF)
            {
                st->Status = 1;
                return;
            }
            long total = TranslatePoints(s, n, d, units, st);
            if (total < 0)
                return;
            for (long k = Math.Min(total, units); k < units; k++)
                d[k] = 0;
        }

        /// <summary>
        ///     CPython's <c>_PyUnicode_TranslateCharmap</c> over code points with a compiled table: each character's entry
        ///     (dense, the pool's replacement, or the managed table past the dense range) appended to <paramref name="dst"/>
        ///     up to <paramref name="cap"/> (characters past it counted, not written), stopping at the first error.
        /// </summary>
        /// <param name="s">The code points (a <c>U</c> element's units may be past U+10FFFF).</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The output (null with a capacity of 0: a counting pass).</param>
        /// <param name="cap">The output capacity.</param>
        /// <param name="st">The call's state.</param>
        /// <returns>The whole result's length (it can exceed <paramref name="cap"/>), or -1 after an error (the error word
        ///     set).</returns>
        /// <remarks>The scan does not stop at the capacity: CPython builds the whole string before setitem cuts it, so an
        ///     error past the cut still raises.</remarks>
        internal static long TranslatePoints(uint* s, long n, uint* dst, long cap, TranslateState* st)
        {
            int* dense = st->Dense;
            uint dl = (uint)st->DenseLength;
            bool high = (st->Flags & (int)TranslateFlags.HighIdentity) != 0;
            long i = 0;
            long k = 0;
            if (cap >= n)
            {
                // While every entry is a code point or a deletion, each character writes at most one unit, so the output
                // never passes n <= cap: branch-free, every entry written and the cursor moved past a kept one only (a
                // deletion's -1 is overwritten by the next unit, or lies past the result). The first entry of any other kind
                // (a replacement, an error, a lookup past the dense range) hands the rest to the loop below.
                for (; i < n; i++)
                {
                    uint c = s[i];
                    int e;
                    if (c < dl)
                        e = dense[c];
                    else if (high && c <= 0x10FFFF)
                        e = (int)c;
                    else
                        break;
                    if (e < StrTranslateTable.Delete)
                        break;
                    dst[k] = (uint)e;
                    k += (e >> 31) + 1;
                }
            }
            for (; i < n; i++)
            {
                uint c = s[i];
                int e = c < dl ? dense[c] : high ? StrTranslateTable.Identity : TranslateLookupSlow(st, c);
                if (e >= 0)
                {
                    if (k < cap)
                        dst[k] = (uint)e;
                    k++;
                    continue;
                }
                if (e == StrTranslateTable.Delete)
                    continue;
                if (e == StrTranslateTable.Identity)
                {
                    // A kept character past U+10FFFF: the writer cannot hold it (a mapped or deleted one never reaches it).
                    if (c > 0x10FFFF)
                    {
                        st->Status = 1;
                        return -1;
                    }
                    if (k < cap)
                        dst[k] = c;
                    k++;
                    continue;
                }
                long x = -3L - e;
                long length = st->Index[2 * x + 1];
                if (length < 0)
                {
                    st->Status = 1;
                    return -1;
                }
                uint* r = st->Pool + st->Index[2 * x];
                for (long j = 0; j < length; j++)
                {
                    if (k < cap)
                        dst[k] = r[j];
                    k++;
                }
            }
            return k;
        }

        /// <summary>The entry of a character past the dense range, from the managed table (dict keys, a long table's tail,
        ///     an unsubscriptable table's error).</summary>
        /// <param name="st">The call's state (its table handle).</param>
        /// <param name="c">The code point.</param>
        /// <returns>The entry.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static int TranslateLookupSlow(TranslateState* st, uint c)
            => ((StrTranslateTable)GCHandle.FromIntPtr(st->Table).Target).Lookup(c);
    }
}
