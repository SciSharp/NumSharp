using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDVStringMethods.Case.cs — NumPy's StringDType case maps upper / lower / swapcase / capitalize / title (strings phase 2,
// P2.6 milestone 2)
// =============================================================================
//
// NUMPY'S STRUCTURE (numpy/_core/strings.py, multiarraymodule.c _vec_string, stringdtype/dtype.c)
//   upper(a) is `_vec_string(np.asarray(a), a.dtype, 'upper')`: a fresh C-ordered array whose descriptor is a new
//   StringDType instance with a's NA object and coercion (finalize_descr), filled in C order with `str.upper(item)`:
//   * item = stringdtype_getitem: the text as a Python str (trailing NULs KEPT: they are characters of a T string), the
//     NA object for an NA cell, '' for a null cell of a descriptor without one;
//   * str.upper of an NA that is not a str (NaN, None, pd.NA, ...) raises CPython's TypeError "descriptor 'upper' for
//     'str' objects doesn't apply to a 'float' object" (the object's type name); a str NA object is computed like any
//     text;
//   * the result is stored by stringdtype_setitem, which first asks na_eq_cmp(result, na_object): under a STRING NA object
//     a result equal to it is stored as the NA — whatever element it came from ('abc'.upper() under na_object='ABC' is
//     the NA, and so is ''.upper() under na_object='').
//   The text's mapping is CPython's str algorithm over its code points (NDTextElements.CaseMapCodePoints: full mappings,
//   the final sigma, title's cased state), so its UTF-8 size can grow or shrink ('ß' -> 'SS' keeps 2 bytes, 'ı' (2 bytes)
//   -> 'I' (1), 'ɐ' (2) -> 'Ɐ' (3)).
//
// HOW AN ELEMENT IS COMPUTED HERE
//   * A short ASCII cell (at most 15 bytes, all below 0x80): one 16-byte vector mapped by the bytes rules (ASCII str
//     methods equal them), written into the output cell with its size byte.
//   * A longer ASCII text: the output reserved at the same size (NewString) and mapped into it 16 bytes at a time.
//   * Any other text: decoded into code points in the call's scratch buffer, mapped by CPython's algorithm, its UTF-8 size
//     summed, the output reserved at that size and encoded into it.
//   * Under a string NA object, a result of the NA's size is compared with the NA's text before anything is reserved.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDVStringMethods
    {
        /// <summary>
        ///     NumPy's <c>upper</c> / <c>lower</c> / <c>swapcase</c> / <c>capitalize</c> / <c>title</c> over a
        ///     <c>StringDType</c> operand: a fresh C-ordered result with the operand's NA object and coercion, one mapped text
        ///     (or NA) per element.
        /// </summary>
        /// <param name="map">The map (<see cref="NDTextElements.UpperMap"/> ...).</param>
        /// <param name="a">The operand (<c>StringDType</c>).</param>
        /// <returns>The result.</returns>
        /// <exception cref="TypeError">An NA whose object is not a str (<c>descriptor 'upper' for 'str' objects doesn't apply
        ///     to a 'float' object</c>).</exception>
        /// <exception cref="OutOfMemoryException">A result no string can hold (NumPy's MemoryError).</exception>
        internal static NDArray CaseMap(int map, NDArray a)
        {
            // _vec_string allocates PyArray_SimpleNewFromDescr: C order whatever a's layout, cells all zero.
            var result = new NDArray(SameNA(a.dtype), new Shape(a.Shape.dimensions), true);
            try
            {
                var input = a.Storage.VStringSlice;
                // Under a string NA object an NA is mapped as that text: its footprint stands for the NA cells.
                byte[] naText = NDVStringCells.RuleOf(a.dtype) == VStringNARule.String ? NDUtf8.Encode(a.dtype.VString.default_string) : null;
                RunLocked(new[] { a }, new[] { result }, a.dtype, aux =>
                {
                    // A case map keeps an ASCII text's size and changes another's by little, so the input's own arena
                    // footprint reserves the result's arena once (the lock is held here), where ArenaMalloc would grow it by
                    // 25 % steps — each a reallocation that copies and zeroes the arena (measured: 53 % of a call over 100K
                    // texts of 16-32 bytes). A shortfall grows the arena from there; an excess is unused capacity, never an
                    // offset NumPy's sequence would not give.
                    if (input.Allocator.HoldsLongStrings || NDVStringCopy.HasLongNAText(naText))
                    {
                        using var cells = NDVStringMove.CellArray(a);
                        result.Storage.VStringSlice.Allocator.TryReserve(NDVStringCopy.ArenaBytesForCopy(cells, naText));
                    }
                    var key = new TextMethodKey(TextMethodOp.VCaseMap, 1, Cell, Cell, Extra: map);
                    NDTextLoop.Execute(key, new[] { a, result }, 1, null, (void*)aux, callerOut: false);
                }, NDTextElements.CaseMapName(map));
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     One element of the <c>StringDType</c> case map: the NA rule, then the mapped text into the fresh output cell
        ///     (or the NA, under a string NA object, when the result equals the NA's text).
        /// </summary>
        /// <param name="cell">The input cell.</param>
        /// <param name="outCell">The output cell (all zero: a fresh array's).</param>
        /// <param name="map">The map (a constant in the IL shell).</param>
        /// <param name="aux">The call's state (arena base, NA rule, default string, output allocator, scratch, error).</param>
        [MethodImpl(MethodImplOptionsConstants.OptimizeAndInline)]
        internal static void VCaseMap(byte* cell, byte* outCell, int map, VStringMethodAux* aux)
        {
            if (Avx2.IsSupported && IsShort(cell) && ShortCaseMap(cell, outCell, map, aux))
                return;
            if (Load(cell, aux->ArenaA, out byte* buf, out long size, aux))
            {
                // getitem gives the NA object: str.<map> refuses any that is not a str; a str NA (and the '' of a
                // descriptor without NA object) is computed as text.
                if (aux->Rule is VStringNARule.NaN or VStringNARule.Null)
                {
                    Fail(aux, VStringMethodError.CaseMapNull);
                    return;
                }
                Default(aux, out buf, out size);
            }
            CaseMapText(buf, size, outCell, map, aux);
        }

        /// <summary>
        ///     The case map of a short cell whose text is ASCII: one 16-byte vector mapped by the bytes rules and written
        ///     into the output cell with its size byte.
        /// </summary>
        /// <param name="cell">A short input cell.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="map">The map.</param>
        /// <param name="aux">The call's state.</param>
        /// <returns>False when the text is not ASCII (the general path maps it).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShortCaseMap(byte* cell, byte* outCell, int map, VStringMethodAux* aux)
        {
            int size = ShortSize(cell);
            // The cell's bytes past the text are not guaranteed zero: mask them (and byte 15, the size byte).
            var v = Vector128.Load(cell) & ByteLanes(size);
            if (v.ExtractMostSignificantBits() != 0)
                return false;
            var r = NDTextElements.CaseBytes128(v, map, Vector128<byte>.Zero, Vector128.CreateScalar((byte)0xFF));
            if (aux->Rule == VStringNARule.String && size == aux->DefaultSize && SameText(r, aux->DefaultText, size))
            {
                WriteNull(outCell);
                return true;
            }
            // The mapped bytes past the text stay zero (only letters flip), so byte 15 is free for the size byte; the empty
            // text leaves the all-zero cell, as NewString does.
            r.Store(outCell);
            if (size > 0)
                outCell[15] = (byte)(ShortFlags | size);
            return true;
        }

        /// <summary>Whether the first <paramref name="size"/> bytes of a vector equal a text (the NA comparison).</summary>
        /// <param name="r">The bytes.</param>
        /// <param name="text">The text (may be null when <paramref name="size"/> is 0).</param>
        /// <param name="size">The byte count (at most 16).</param>
        /// <returns>True when equal.</returns>
        private static bool SameText(Vector128<byte> r, byte* text, int size)
        {
            byte* tmp = stackalloc byte[16];
            r.Store(tmp);
            return new ReadOnlySpan<byte>(tmp, size).SequenceEqual(new ReadOnlySpan<byte>(text, size));
        }

        /// <summary>
        ///     The case map of a text into a fresh output cell: ASCII text mapped at its own size, any other decoded, mapped by
        ///     CPython's algorithm and re-encoded (see the file header).
        /// </summary>
        /// <param name="buf">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="map">The map.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CaseMapText(byte* buf, long size, byte* outCell, int map, VStringMethodAux* aux)
        {
            if (!IsAsciiRun(buf, size))
            {
                CaseMapTextDecoded(buf, size, outCell, map, aux);
                return;
            }
            if (aux->Rule == VStringNARule.String && size == aux->DefaultSize)
            {
                // The result may be the NA's text: map into scratch and compare before the output is reserved.
                byte* tmp = Scratch(aux, Math.Max(size, 1));
                if (tmp == null)
                {
                    Fail(aux, VStringMethodError.Memory);
                    return;
                }
                MapAsciiRun(buf, size, tmp, map);
                if (new ReadOnlySpan<byte>(tmp, (int)size).SequenceEqual(new ReadOnlySpan<byte>(aux->DefaultText, (int)size)))
                    WriteNull(outCell);
                else
                    WriteCell(outCell, tmp, size, aux->Out0, aux);
                return;
            }
            byte* p = NewString(outCell, size, aux->Out0, aux);
            if (p != null && size > 0)
                MapAsciiRun(buf, size, p, map);
        }

        /// <summary>Whether every byte of a text is ASCII (16-byte vectors, the last overlapping; bytes one at a time below
        ///     16, or without AVX2).</summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its size.</param>
        /// <returns>True when no byte is 0x80 or above.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsAsciiRun(byte* s, long size)
        {
            if (!Avx2.IsSupported || size < 16)
            {
                for (long i = 0; i < size; i++)
                    if (s[i] >= 0x80)
                        return false;
                return true;
            }
            var acc = Vector128<byte>.Zero;
            long o = 0;
            for (; o + 16 <= size; o += 16)
                acc |= Vector128.Load(s + o);
            if (o < size)
                acc |= Vector128.Load(s + size - 16);
            return acc.ExtractMostSignificantBits() == 0;
        }

        /// <summary>The bytes rules over an ASCII run: the vector form with AVX2, the scalar body otherwise.</summary>
        /// <param name="src">The text.</param>
        /// <param name="size">Its size.</param>
        /// <param name="dst">The output (its own buffer).</param>
        /// <param name="map">The map.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MapAsciiRun(byte* src, long size, byte* dst, int map)
        {
            if (Avx2.IsSupported)
                NDTextElements.CaseMapByteRun(src, size, dst, map);
            else
                NDTextElements.CaseMapBytesScalar(src, size, dst, map);
        }

        /// <summary>
        ///     The case map of a non-ASCII text: its code points decoded into the call's scratch buffer, mapped by CPython's
        ///     algorithm (<see cref="NDTextElements.CaseMapCodePoints"/>), the UTF-8 size summed, then the output reserved and
        ///     encoded (or the NA, under a string NA object, when the result equals the NA's text).
        /// </summary>
        /// <param name="buf">The text (valid UTF-8).</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="outCell">The output cell (all zero).</param>
        /// <param name="map">The map.</param>
        /// <param name="aux">The call's state.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void CaseMapTextDecoded(byte* buf, long size, byte* outCell, int map, VStringMethodAux* aux)
        {
            // Scratch: the code points (at most one per byte, 4 bytes each), the mapping (at most three per code point), and
            // the encoded result for the NA comparison (at most 4 bytes per mapped code point).
            if (size > int.MaxValue / 28)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            byte* scratch = Scratch(aux, 28 * size + 16);
            if (scratch == null)
            {
                Fail(aux, VStringMethodError.Memory);
                return;
            }
            uint* cps = (uint*)scratch;
            long n = 0;
            byte* end = buf + size;
            for (byte* c = buf; c < end;)
            {
                cps[n++] = Decode(c);
                int bytes = CharBytes(*c);
                // A continuation byte cannot start a character of valid UTF-8; step over it rather than loop on it.
                c += bytes == 0 ? 1 : bytes;
            }
            uint* mapped = cps + n;
            uint* one = stackalloc uint[PyUnicodeData.MaxMapping];
            long k = NDTextElements.CaseMapCodePoints(cps, n, map, mapped, 3 * n, one);
            long outSize = 0;
            for (long i = 0; i < k; i++)
            {
                uint cp = mapped[i];
                outSize += cp <= 0x7F ? 1 : cp <= 0x7FF ? 2 : cp <= 0xFFFF ? 3 : 4;
            }
            if (aux->Rule == VStringNARule.String && outSize == aux->DefaultSize)
            {
                byte* encoded = (byte*)(mapped + k);
                byte* e = encoded;
                for (long i = 0; i < k; i++)
                    e += Encode(mapped[i], e);
                if (new ReadOnlySpan<byte>(encoded, (int)outSize).SequenceEqual(new ReadOnlySpan<byte>(aux->DefaultText, (int)outSize)))
                    WriteNull(outCell);
                else
                    WriteCell(outCell, encoded, outSize, aux->Out0, aux);
                return;
            }
            byte* p = NewString(outCell, outSize, aux->Out0, aux);
            if (p == null)
                return;
            for (long i = 0; i < k; i++)
                p += Encode(mapped[i], p);
        }
    }
}
