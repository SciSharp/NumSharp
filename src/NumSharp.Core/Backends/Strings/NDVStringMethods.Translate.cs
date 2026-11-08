using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDVStringMethods.Translate.cs — the StringDType element body of np.strings.translate (strings phase 2, P2.6 milestone 3)
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py translate, multiarraymodule.c _vec_string_with_args, stringdtype/dtype.c)
//   For a StringDType operand translate is _vec_string(a, a.dtype, 'translate', [table] + _clean_args(deletechars)): a
//   fresh C-ordered array with a's NA object and coercion, filled in C order with str.translate(item, table):
//   * item = stringdtype_getitem: the text as a Python str (trailing NULs KEPT: they are characters of a T string), the NA
//     object for an NA cell, '' for a null cell of a descriptor without one;
//   * str.translate of an NA that is not a str (NaN, None, ...) raises CPython's descriptor TypeError ("descriptor
//     'translate' for 'str' objects doesn't apply to a 'NoneType' object"); a str NA object is translated like any text;
//   * the result is stored by stringdtype_setitem: under a STRING NA object a result equal to it is the NA; a result holding
//     a surrogate code point cannot be encoded as UTF-8, and _vec_string reports that failed setitem as "TypeError: result
//     array type does not match underlying function".
//   With deletechars every element raises (str.translate takes one argument); the driver walks those calls.
//
// HOW AN ELEMENT IS COMPUTED HERE (the table compiled once per call: NDTranslate.cs)
//   * A table that touches ASCII only (every non-ASCII character kept, every ASCII character to ASCII text — the
//     str.maketrans tables of ASCII text): ANY text byte for byte, its bytes of 0x80 or more copied (they belong to kept
//     characters) — mapped at its own size straight into the output when no length changes, else compacted / expanded into
//     the call's scratch and stored.
//   * An ASCII text with a table mapping every ASCII character to one ASCII character (or deleting it): the same byte maps.
//   * Any other text: ONE pass from UTF-8 to UTF-8 — each character decoded (ASCII without decoding), looked up, and its
//     result encoded into the scratch (sized by the longest replacement; a counting pass first when that bound is large),
//     surrogates refused when the table can make one — then stored. No code-point buffer in between.
//   A string NA's text is compared with a result of its size before anything is reserved. An element that raises sets the
//   TranslateState's error word; the driver then walks the calls in C order (NDCharObjectPath.Translate.cs) to raise
//   NumPy's first error.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDVStringMethods
    {
        /// <summary>
        ///     The <c>StringDType</c> loop of <c>translate</c> over a compiled table: every element of <paramref name="a"/>
        ///     translated into <paramref name="result"/> (a fresh C-ordered array with <paramref name="a"/>'s NA object and
        ///     coercion), elements that raise flagged in <paramref name="st"/>.
        /// </summary>
        /// <param name="a">The operand (<c>StringDType</c>).</param>
        /// <param name="result">The fresh result (all-zero cells; see <see cref="SameNADescr"/>).</param>
        /// <param name="st">The call's state: the compiled table (with its handle set) and the error word.</param>
        /// <exception cref="OutOfMemoryException">A result no string can hold (NumPy's MemoryError), or a corrupt input cell.</exception>
        /// <remarks>The error word is the caller's to read: a translate error is raised by its C-order walk, never here.</remarks>
        internal static void TranslateKernel(NDArray a, NDArray result, TranslateState* st)
        {
            var input = a.Storage.VStringSlice;
            byte[] naText = NDVStringCells.RuleOf(a.dtype) == VStringNARule.String ? NDUtf8.Encode(a.dtype.VString.default_string) : null;
            // The state is the caller's stack block: carried as an address (a lambda cannot capture a pointer local).
            IntPtr state = (IntPtr)st;
            RunLocked(new[] { a }, new[] { result }, a.dtype, aux =>
            {
                ((VStringMethodAux*)aux)->Extra = (void*)state;
                // As the case maps: the input's arena footprint reserves the result's arena once (a translation that keeps
                // sizes is the common case; a shortfall grows the arena from there).
                if (input.Allocator.HoldsLongStrings || NDVStringCopy.HasLongNAText(naText))
                {
                    using var cells = NDVStringMove.CellArray(a);
                    result.Storage.VStringSlice.Allocator.TryReserve(NDVStringCopy.ArenaBytesForCopy(cells, naText));
                }
                var key = new TextMethodKey(TextMethodOp.VTranslate, 1, Cell, Cell);
                NDTextLoop.Execute(key, new[] { a, result }, 1, null, (void*)aux, callerOut: false);
            }, "translate");
        }

        /// <summary>A fresh descriptor with an operand's NA object and coercion: the descriptor of a translate result.</summary>
        /// <param name="d">The operand's descriptor (<c>StringDType</c>).</param>
        /// <returns>The result descriptor.</returns>
        internal static DType SameNADescr(DType d) => SameNA(d);

        /// <summary>
        ///     One element of the <c>StringDType</c> translate: the NA rule, then the translated text into the fresh output
        ///     cell (or the NA, under a string NA object, when the result equals the NA's text).
        /// </summary>
        /// <param name="cell">The input cell.</param>
        /// <param name="outCell">The output cell (all zero: a fresh array's).</param>
        /// <param name="aux">The call's state; <see cref="VStringMethodAux.Extra"/> is the <see cref="TranslateState"/>.</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VTranslate(byte* cell, byte* outCell, VStringMethodAux* aux)
        {
            var st = (TranslateState*)aux->Extra;
            if (Load(cell, aux->ArenaA, out byte* buf, out long size, aux))
            {
                // getitem gives the NA object: str.translate refuses any that is not a str (the descriptor check), which the
                // driver's walk raises at the first such element; a str NA (and the '' of a descriptor without NA object) is
                // translated as text.
                if (aux->Rule is VStringNARule.NaN or VStringNARule.Null)
                {
                    st->Status = 1;
                    return;
                }
                Default(aux, out buf, out size);
            }
            if ((st->Flags & (int)TranslateFlags.AsciiOnly) != 0)
            {
                // The table touches ASCII only: any text translates byte for byte (its non-ASCII bytes copied: the byte
                // map's upper half is the identity).
                if ((st->Flags & (int)TranslateFlags.AsciiOneToOne) != 0)
                    TranslateAscii(buf, size, outCell, aux, st->AsciiMap);
                else if ((st->Flags & (int)TranslateFlags.AsciiCompact) != 0)
                    TranslateAsciiCompact(buf, size, outCell, aux, st->AsciiMap);
                else
                    TranslateAsciiBytes(buf, size, outCell, aux, st);
                return;
            }
            if (st->AsciiMap != null && IsAsciiRun(buf, size))
            {
                if ((st->Flags & (int)TranslateFlags.AsciiOneToOne) != 0)
                    TranslateAscii(buf, size, outCell, aux, st->AsciiMap);
                else
                    TranslateAsciiCompact(buf, size, outCell, aux, st->AsciiMap);
                return;
            }
            TranslateTextUtf8(buf, size, outCell, aux, st);
        }

        /// <summary>
        ///     The translate of any text through a table that touches ASCII only and changes some lengths (deletions,
        ///     replacements of ASCII text): byte for byte into the call's scratch — a byte of 0x80 or more copied, an ASCII
        ///     byte through its entry — then stored.
        /// </summary>
        /// <param name="buf">The text (valid UTF-8).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="aux">The call's state.</param>
        /// <param name="st">The translate state (an <see cref="TranslateFlags.AsciiOnly"/> table).</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void TranslateAsciiBytes(byte* buf, long size, byte* outCell, VStringMethodAux* aux, TranslateState* st)
        {
            // Every byte becomes at most MaxReplacement bytes (an ASCII replacement is one byte per code point).
            long rep = st->MaxReplacement;
            if (size > long.MaxValue / rep)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            byte* tmp = Scratch(aux, Math.Max(size * rep, 1));
            if (tmp == null)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            int* dense = st->Dense;
            long k = 0;
            for (long i = 0; i < size; i++)
            {
                uint b = buf[i];
                if (b >= 0x80)
                {
                    tmp[k++] = (byte)b;
                    continue;
                }
                int e = dense[b];
                if (e >= 0)
                {
                    tmp[k++] = (byte)e;
                    continue;
                }
                if (e == StrTranslateTable.Delete)
                    continue;
                long x = -3L - e;
                long length = st->Index[2 * x + 1];
                if (length < 0)
                {
                    // An ASCII character whose mapping raises: the driver's walk raises it.
                    st->Status = 1;
                    return;
                }
                uint* r = st->Pool + st->Index[2 * x];
                for (long j = 0; j < length; j++)
                    tmp[k++] = (byte)r[j];
            }
            StoreTranslated(tmp, k, outCell, aux);
        }

        /// <summary>
        ///     The translate of an ASCII text through an ASCII one-to-one table: every byte mapped, the size kept (or the NA,
        ///     under a string NA object, when the result equals the NA's text).
        /// </summary>
        /// <param name="buf">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="aux">The call's state.</param>
        /// <param name="map">The 256-entry byte map (ASCII bytes mapped, the upper half copied).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void TranslateAscii(byte* buf, long size, byte* outCell, VStringMethodAux* aux, byte* map)
        {
            if (aux->Rule == VStringNARule.String && size == aux->DefaultSize)
            {
                // The result may be the NA's text: map into scratch and compare before the output is reserved.
                byte* tmp = Scratch(aux, Math.Max(size, 1));
                if (tmp == null)
                {
                    Fail(aux, VStringMethodError.Memory);
                    return;
                }
                for (long i = 0; i < size; i++)
                    tmp[i] = map[buf[i]];
                StoreTranslated(tmp, size, outCell, aux);
                return;
            }
            byte* p = NewString(outCell, size, aux->Out0, aux);
            if (p != null)
                for (long i = 0; i < size; i++)
                    p[i] = map[buf[i]];
        }

        /// <summary>
        ///     The translate of an ASCII text through an ASCII table that deletes some characters: compacted branch-free into
        ///     the call's scratch (every byte written, the cursor moved past a kept one: 0xFF is a deletion), then stored.
        /// </summary>
        /// <param name="buf">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="aux">The call's state.</param>
        /// <param name="map">The 256-entry byte map (0xFF = delete; the upper half copied).</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void TranslateAsciiCompact(byte* buf, long size, byte* outCell, VStringMethodAux* aux, byte* map)
        {
            byte* tmp = Scratch(aux, Math.Max(size, 1));
            if (tmp == null)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            long k = 0;
            for (long i = 0; i < size; i++)
            {
                uint v = map[buf[i]];
                tmp[k] = (byte)v;
                // (v + 1) >> 8 is 1 for the deletion marker 0xFF only.
                k += 1 - ((v + 1) >> 8);
            }
            StoreTranslated(tmp, k, outCell, aux);
        }

        /// <summary>
        ///     The translate of any text in one pass from UTF-8 to UTF-8 into the call's scratch (see the file header), then
        ///     stored.
        /// </summary>
        /// <param name="buf">The text (valid UTF-8).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="aux">The call's state.</param>
        /// <param name="st">The translate state (the compiled table and the error word).</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void TranslateTextUtf8(byte* buf, long size, byte* outCell, VStringMethodAux* aux, TranslateState* st)
        {
            // Every input byte starts at most one character, which becomes at most MaxReplacement code points of at most 4
            // bytes each. Past 64 MiB that bound is a guess too costly to reserve: a counting pass measures the result.
            const long LargeBound = 1L << 26;
            long rep = st->MaxReplacement;
            long bound = size > (long.MaxValue / 4) / rep ? -1 : 4 * size * rep;
            if (bound < 0 || bound > LargeBound)
            {
                bound = TranslateUtf8(buf, size, null, st);
                if (bound < 0)
                    return;
            }
            // Three bytes of slack: a kept character is copied as its four-byte window.
            byte* tmp = Scratch(aux, bound + 4);
            if (tmp == null)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            long bytes = TranslateUtf8(buf, size, tmp, st);
            if (bytes < 0)
                return;
            StoreTranslated(tmp, bytes, outCell, aux);
        }

        /// <summary>
        ///     <c>stringdtype_setitem</c> of a translated text held in scratch: under a string NA object a result equal to the
        ///     NA's text is the NA, anything else is copied into the fresh output cell.
        /// </summary>
        /// <param name="text">The result (UTF-8, in the call's scratch).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StoreTranslated(byte* text, long size, byte* outCell, VStringMethodAux* aux)
        {
            if (aux->Rule == VStringNARule.String && size == aux->DefaultSize
                && new ReadOnlySpan<byte>(text, (int)size).SequenceEqual(new ReadOnlySpan<byte>(aux->DefaultText, (int)size)))
            {
                WriteNull(outCell);
                return;
            }
            WriteCell(outCell, text, size, aux->Out0, aux);
        }

        /// <summary>
        ///     CPython's <c>_PyUnicode_TranslateCharmap</c> from UTF-8 to UTF-8 with a compiled table: each character decoded
        ///     (an ASCII byte is its own code point), its entry looked up (dense, the pool's replacement, or the managed table
        ///     past the dense range unless the table keeps everything there) and its result encoded, stopping at the first
        ///     error.
        /// </summary>
        /// <param name="src">The text (valid UTF-8: every code point a scalar value).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="dst">The output (room for the result plus 3 bytes of slack), or null for a counting pass.</param>
        /// <param name="st">The translate state.</param>
        /// <returns>The result's byte size, or -1 after an error (the error word set): a mapping that raises, or — when the
        ///     table can make one — a surrogate code point, which no <c>StringDType</c> element can hold.</returns>
        /// <remarks>
        ///     An ASCII character takes a predicted branch, so the next character's address never waits for this one's load
        ///     (the branch-free decode below serializes every character on the previous load: ASCII-heavy text ran twice as
        ///     slow through it). While four bytes remain, a multi-byte character is decoded from its four-byte window without
        ///     branches (the length from the lead byte, every continuation byte's bits placed, the unused ones shifted out),
        ///     looked up with masks, and when it is KEPT — the common case — its own bytes are copied as the window (the
        ///     window's extra bytes land past it, where the next character or the end of the result covers them; hence the
        ///     slack). The last bytes, and the counting pass, decode one character at a time. On short mixed text the
        ///     per-character cost stays near 9 ns either way (measured in isolation): an out-of-line handler for changed
        ///     characters and a zero-padded window for the last bytes were both slower.
        /// </remarks>
        private static long TranslateUtf8(byte* src, long size, byte* dst, TranslateState* st)
        {
            int* dense = st->Dense;
            uint dl = (uint)st->DenseLength;
            bool high = (st->Flags & (int)TranslateFlags.HighIdentity) != 0;
            bool surrogates = (st->Flags & (int)TranslateFlags.SurrogateOutput) != 0;
            uint highBit = high ? 1u : 0u;
            byte* p = src;
            byte* end = src + size;
            long k = 0;
            // A kernel table's dense range covers ASCII (at least 256 entries), so an ASCII character is always in range.
            if (dst != null && dl >= 128)
            {
                while (end - p >= 4)
                {
                    uint a = *p;
                    if (a < 0x80)
                    {
                        int ea = dense[a];
                        p++;
                        if (ea == (int)a)
                        {
                            dst[k++] = (byte)a;
                            continue;
                        }
                        k = PutEntry(ea, a, dst, k, st, surrogates);
                        if (k < 0)
                            return -1;
                        continue;
                    }
                    uint raw = Unsafe.ReadUnaligned<uint>(p);
                    // Big-endian: the lead byte on top, each continuation byte's six bits below it.
                    uint w = System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(raw);
                    uint lead = w >> 24;
                    int nb = 1 + (lead >= 0xC0 ? 1 : 0) + (lead >= 0xE0 ? 1 : 0) + (lead >= 0xF0 ? 1 : 0);
                    // The lead byte's payload mask per length (0x7F, 0x1F, 0x0F, 0x07), picked arithmetically: written as a
                    // ternary it compiled to a branch, which mixed-width text mispredicts about half the time.
                    uint leadMask = (uint)(0x070F1F7FUL >> (8 * (nb - 1))) & 0xFFu;
                    uint c = (((lead & leadMask) << 18) | (((w >> 16) & 0x3F) << 12) | (((w >> 8) & 0x3F) << 6) | (w & 0x3F)) >> (24 - 6 * nb);
                    // The dense entry or the character itself, by masks (a select written with ternaries compiled to two
                    // branches): an out-of-range character reads entry 0 and keeps itself. A T text holds no unit past
                    // U+10FFFF, so a kept character past the dense range is simply itself.
                    uint inRange = c < dl ? 1u : 0u;
                    uint keep = 0u - inRange;
                    int d = dense[c & keep];
                    int e = (int)c + (int)((uint)(d - (int)c) & keep);
                    if ((inRange | highBit) == 0)
                        e = NDTextElements.TranslateLookupSlow(st, c);
                    p += nb;
                    if (e == (int)c)
                    {
                        Unsafe.WriteUnaligned(dst + k, raw);
                        k += nb;
                        continue;
                    }
                    k = PutEntry(e, c, dst, k, st, surrogates);
                    if (k < 0)
                        return -1;
                }
            }
            while (p < end)
            {
                uint c = *p;
                if (c < 0x80)
                {
                    p++;
                }
                else
                {
                    c = Decode(p);
                    int nb = CharBytes(*p);
                    // A continuation byte cannot start a character of valid UTF-8; step over it rather than loop on it.
                    p += nb == 0 ? 1 : nb;
                }
                int e = c < dl ? dense[c] : high ? (int)c : NDTextElements.TranslateLookupSlow(st, c);
                k = PutEntry(e, c, dst, k, st, surrogates);
                if (k < 0)
                    return -1;
            }
            return k;
        }

        /// <summary>
        ///     Writes one character's result (its entry): a code point encoded, nothing for a deletion, the character itself
        ///     for a kept one, a replacement's code points from the pool; an error entry, or a surrogate the table made, sets
        ///     the error word.
        /// </summary>
        /// <param name="e">The entry.</param>
        /// <param name="c">The character.</param>
        /// <param name="dst">The output, or null to measure.</param>
        /// <param name="k">The write position.</param>
        /// <param name="st">The translate state.</param>
        /// <param name="surrogates">Whether the table can make a surrogate (checked then).</param>
        /// <returns>The write position after the result, or -1 after an error. (Returned rather than passed by reference:
        ///     a <c>ref</c> position kept the caller's loop variable on the stack.)</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long PutEntry(int e, uint c, byte* dst, long k, TranslateState* st, bool surrogates)
        {
            if (e >= 0)
            {
                if (surrogates && (uint)e - 0xD800u < 0x800u)
                {
                    st->Status = 1;
                    return -1;
                }
                return k + PutUtf8((uint)e, dst, k);
            }
            if (e == StrTranslateTable.Delete)
                return k;
            if (e == StrTranslateTable.Identity)
                return k + PutUtf8(c, dst, k);
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
                uint cp = r[j];
                if (surrogates && cp - 0xD800u < 0x800u)
                {
                    st->Status = 1;
                    return -1;
                }
                k += PutUtf8(cp, dst, k);
            }
            return k;
        }

        /// <summary>Encodes one code point at <c>dst + k</c> (or only measures it for a counting pass).</summary>
        /// <param name="cp">The code point (a scalar value).</param>
        /// <param name="dst">The output, or null to measure.</param>
        /// <param name="k">The write position.</param>
        /// <returns>The encoding's byte count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PutUtf8(uint cp, byte* dst, long k)
        {
            if (cp < 0x80)
            {
                if (dst != null)
                    dst[k] = (byte)cp;
                return 1;
            }
            if (dst != null)
                return Encode(cp, dst + k);
            return cp <= 0x7FF ? 2 : cp <= 0xFFFF ? 3 : 4;
        }
    }
}
