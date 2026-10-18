using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using NumSharp.IO;

// =============================================================================
// NDCharObjectPath.CodecFast.cs — the fast runs of np.strings.encode / decode (strings phase 2, P2.6 milestone 4)
// =============================================================================
//
// A fast run codes a stretch of elements with everything per call hoisted: it starts only once the result rows exist, the
// codec is known and the arguments — which do not move inside the chunk — were read and checked at their addresses
// (CodecWalk.FastReady), so no per-element argument work is left. Every run has the same two levels:
//
//   * an INNER loop with no call in it (its state stays in registers), over the elements it writes straight into their
//     rows with whole-vector stores — it stops at the first element it does not take, or at the result's last rows, where
//     a whole-vector store would pass the result's end (CodecRows.VectorStoreLimit);
//   * an OUTER loop that codes that one element on its own — straight into its row with exact writes when it can, else
//     through the codec into the arena and its row (EncodeThroughArena / DecodeThroughArena: the general route's coding
//     without its per-element dispatch) — and enters the inner loop again.
//
// A run stops only BEFORE an element the general route must see: a U one-unit text past U+10FFFF (CPython's SystemError at
// the scalar) or a StringDType NA (its NA object). Every element gets exactly the bytes the general route writes (the
// CodecFastPathsDisabled hook replays a call through the general route for the tests).
//
//   U → S, UTF-8 / ASCII / Latin-1   AVX2, an element of up to 32 code points, rows up to 128 bytes: up to four masked loads
//                                    (lanes past the element never touched), the trimmed length from the non-zero lanes, the
//                                    OR of the units against the codec's one-byte limit (0x80, Latin-1 0x100), two unsigned
//                                    packs and a permute into 32 bytes stored over the row (the element's bytes, then its
//                                    zero lanes as the padding). UTF-8 past ASCII without surrogates or units past U+10FFFF
//                                    is sized from the lanes (n + the lanes past 0x7F, 0x7FF, 0xFFFF) and written over the
//                                    row zeroed first, without a branch per code point: 16-bit words compacted by one
//                                    vpshufb from a 256-entry table (up to eight code points below U+0800), per-lane
//                                    three-byte forms packed by one vpshufb (up to eight code points all three bytes long),
//                                    or 32-bit words (any). The one-byte encoders' inline handlers are written in the run:
//                                    replace as a blend of '?' into the lanes at or past the limit; ignore /
//                                    xmlcharrefreplace / backslashreplace / surrogateescape sized per lane in vectors, then
//                                    written over the zeroed row by walking only the unencodable positions (tzcnt), each run
//                                    of encodable bytes moved by one 32-byte copy of the narrowed bytes.
//   U → S, namereplace               ASCII / Latin-1, AVX2, an element of up to 32 code points in the rows the exact pre-pass
//                                    sized (CodecWalk.EstimateRows): the row zeroed by whole-vector stores, the runs of
//                                    encodable bytes moved as above, each unencodable BMP character's cached \N{NAME} (or
//                                    backslash form) copied in 32-byte chunks of its zero-padded bytes; a character past
//                                    U+FFFF or one whose replacement is not built yet stops the run (the outer loop builds
//                                    and caches it).
//   U → S, UTF-16 / UTF-32           AVX2, an element of up to 32 code points with no surrogate (UTF-16: nothing past
//                                    U+FFFF either): the units packed to 16 bits (UTF-16) or kept (UTF-32), byte-swapped for
//                                    big-endian, stored after the BOM of the BOM codecs.
//   StringDType → S                  a SHORT cell (its text inside its 16 bytes) read with one 16-byte load masked to its
//                                    size: copied under UTF-8 (and an ASCII text under ASCII / Latin-1), decoded without a
//                                    branch under Latin-1 when its code points are below U+0100; arena / heap cells are read
//                                    through the allocator, an NA cell stops the run.
//   S → U, UTF-8 / ASCII / Latin-1   AVX2, an element of up to 32 bytes read with one 32-byte load (contiguous elements
//                                    away from the chunk's end; exact 8-byte groups otherwise), the trimmed length from the
//                                    highest non-zero byte, each 8-byte group widened to eight code points (vpmovzxbd) and
//                                    stored over the row; Latin-1 takes any byte. UTF-8 past ASCII is validated with
//                                    byte-class masks (every continuation byte exactly where a lead needs one, no C0 / C1 /
//                                    F5..FF, the second-byte ranges of E0 / ED / F0 / F4 — exactly CPython's strict
//                                    decoder's verdict) and decoded into the row zeroed first: values computed at every
//                                    position and compacted to the code-point starts by one vpshufb from a 256-entry table
//                                    per half (one- and two-byte text up to 16 bytes), three bytes per lane gathered by one
//                                    vpshufb (up to eight code points all three bytes long), or 32-bit words (any). Under
//                                    replace / surrogateescape / ignore / backslashreplace, an element whose bytes past 0x7F
//                                    never adjoin has exactly one error range per such byte (no valid sequence can start):
//                                    U+FFFD / U+DC00 + the byte as a blend, ignore / backslashreplace per byte; under ASCII
//                                    every byte past 0x7F is its own error range.
//   S → U, UTF-16 / UTF-32           AVX2, an element of up to 32 bytes of whole units without a surrogate (UTF-32: none past
//                                    U+10FFFF either): the BOM of the BOM codecs read and skipped, the units byte-swapped for
//                                    big-endian and widened (vpmovzxwd) or kept.
//   every other element / codec      the outer loop: exact writes into the row when the element needs no codec (the result's
//                                    last rows, an element past 32 characters or bytes, a host without AVX2), else the codec
//                                    into the arena and the row (utf-8-sig, the code pages, the generic handlers other than
//                                    namereplace, an error the run's handler writes cannot take, a UTF-16 / UTF-32 element
//                                    with a surrogate or a truncated unit).
// Row stores of whole vectors spill into the following rows, which C order writes later; the last rows of the result take
// the outer loop's exact writes instead, so nothing is written past the result's end.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDCharObjectPath
    {
        /// <summary>
        ///     Codes elements from <paramref name="pa"/> with the fast run of the call's input kind and codec (see the file
        ///     header), stopping before the first element the general route must see.
        /// </summary>
        /// <param name="w">The call's state (rows active, codec known, arguments checked).</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride in bytes.</param>
        /// <param name="count">The number of elements left in the chunk.</param>
        /// <returns>The number of elements coded (0 .. <paramref name="count"/>).</returns>
        /// <exception cref="Exception">Whatever the codec raises for an element (the error the general route raises).</exception>
        private static long FastRun(CodecWalk w, byte* pa, long sa, long count)
        {
            var kind = w.Codec.Kind;
            switch (w.AReader.Kind)
            {
                case OperandKind.Unicode:
                    if (kind is PythonCodecKind.Utf8 or PythonCodecKind.Ascii or PythonCodecKind.Latin1)
                        return FastEncodeNarrow(w, pa, sa, count);
                    if (IsUtfWide(kind))
                        return FastEncodeWide(w, pa, sa, count);
                    return FastEncodeUnicode(w, pa, sa, count);
                case OperandKind.VString:
                    return FastEncodeVString(w, pa, sa, count);
                default:
                    if (kind is PythonCodecKind.Utf8 or PythonCodecKind.Ascii or PythonCodecKind.Latin1)
                        return FastDecodeWiden(w, pa, sa, count);
                    if (IsUtfWide(kind))
                        return FastDecodeWide(w, pa, sa, count);
                    return FastDecodeBytes(w, pa, sa, count);
            }
        }

        /// <summary>Whether a codec is one of the six UTF-16 / UTF-32 codecs (with a BOM, little- or big-endian).</summary>
        /// <param name="kind">The codec.</param>
        /// <returns>True for UTF-16 / UTF-32.</returns>
        private static bool IsUtfWide(PythonCodecKind kind)
            => kind is PythonCodecKind.Utf16 or PythonCodecKind.Utf16LE or PythonCodecKind.Utf16BE
                or PythonCodecKind.Utf32 or PythonCodecKind.Utf32LE or PythonCodecKind.Utf32BE;

        /// <summary>One <c>U</c> element through the codec into the arena and its row (the rows may be re-laid out).</summary>
        /// <param name="w">The call's state.</param>
        /// <param name="units">The element's code points.</param>
        /// <param name="n">Its trimmed length.</param>
        /// <exception cref="Exception">The codec's error.</exception>
        private static void EncodeThroughArena(CodecWalk w, uint* units, int n)
        {
            var arena = w.Rows.Arena;
            long start = arena.Count;
            w.Codec.Encode(units, n, w.Errors, arena);
            w.Rows.StoreRow(arena.Count - start);
        }

        /// <summary>One non-empty <c>S</c> element through the codec into the arena and its row.</summary>
        /// <param name="w">The call's state.</param>
        /// <param name="bytes">The element's bytes.</param>
        /// <param name="n">Its trimmed length (at least 1).</param>
        /// <exception cref="Exception">The codec's error.</exception>
        private static void DecodeThroughArena(CodecWalk w, byte* bytes, int n)
        {
            var arena = w.Rows.Arena;
            long start = arena.Count;
            w.Codec.Decode(bytes, n, w.Errors, arena);
            w.Rows.StoreRow(arena.Count - start);
        }

        // =============================================================================================================
        // U → S, UTF-8 / ASCII / Latin-1
        // =============================================================================================================

        /// <summary>An inner run's verdict when it stops before the chunk's end: the element it stopped at takes the outer loop's
        ///     exact-write path (the result's last rows, a row too narrow for it).</summary>
        private const int RunExact = 0;

        /// <summary>
        ///     How far past a row's 32-byte stores the narrow encode run writes: the handler writes' 32-byte copy of a run of
        ///     encodable bytes starts anywhere inside the row (the UTF-8 writers' words end at most 3 bytes past it).
        /// </summary>
        private const int NarrowRunSlack = 32;

        /// <summary>An inner run's verdict when it stops before the chunk's end: the element it stopped at needs the codec (an
        ///     error the run does not write, or a one-unit text past U+10FFFF the outer loop leaves to the general route).</summary>
        private const int RunCodec = 1;

        /// <summary>The narrow run's handler codes (<see cref="RunHandlerOf"/>): the errors the run writes itself for ASCII /
        ///     Latin-1 — CPython's inline cases of <c>unicode_encode_ucs1</c> — 0 when the codec must.</summary>
        private const int HandlerNone = 0, HandlerReplace = 1, HandlerIgnore = 2, HandlerXml = 3, HandlerBackslash = 4,
            HandlerSurrogateEscape = 5;

        /// <summary>
        ///     The narrow run's code for an errors policy: <c>replace</c>, <c>ignore</c>, <c>xmlcharrefreplace</c>,
        ///     <c>backslashreplace</c> and <c>surrogateescape</c>, which <c>unicode_encode_ucs1</c> writes inline (each
        ///     unencodable character on its own, so per character here too), else <see cref="HandlerNone"/> (strict raises,
        ///     every other handler is the generic call: the codec).
        /// </summary>
        /// <param name="handler">The policy's inline classification.</param>
        /// <returns>The run's handler code.</returns>
        private static int RunHandlerOf(NDErrorHandler handler) => handler switch
        {
            NDErrorHandler.Replace => HandlerReplace,
            NDErrorHandler.Ignore => HandlerIgnore,
            NDErrorHandler.XmlCharRefReplace => HandlerXml,
            NDErrorHandler.BackslashReplace => HandlerBackslash,
            NDErrorHandler.SurrogateEscape => HandlerSurrogateEscape,
            _ => HandlerNone,
        };

        /// <summary>
        ///     The UTF-8 / ASCII / Latin-1 encode run over <c>U</c> elements (see the file header): the vector inner run, and
        ///     each element it refuses on its own — straight to the codec when the run says so, else the exact-write path.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded.</returns>
        /// <exception cref="Exception">The codec's error for an element it codes through the arena.</exception>
        private static long FastEncodeNarrow(CodecWalk w, byte* pa, long sa, long count)
        {
            var rows = w.Rows;
            bool utf8 = w.Codec.Kind == PythonCodecKind.Utf8;
            uint limit = w.Codec.Kind == PythonCodecKind.Latin1 ? 0x100u : 0x80u;
            // namereplace (a generic handler, selected by its exact name: an unknown name must raise only at the first error,
            // so the registry is not consulted here) has a run of its own, which looks the names up.
            if (!utf8 && IsNameReplace(w.Errors))
                return FastEncodeNameReplace(w, pa, sa, count, limit);
            // UTF-8's only errors are surrogates (the codec writes them); ASCII / Latin-1 write their inline handlers in the run.
            int handler = utf8 ? HandlerNone : RunHandlerOf(w.Errors.Inline);
            int chars = w.AReader.Chars;
            bool vector = Avx2.IsSupported && chars <= 32;
            // The narrowed bytes of an element the run writes with handler replacements: 32 bytes, then zeros (a run of
            // encodable characters is moved by a 32-byte copy from any of them).
            byte* nb = stackalloc byte[64];
            Vector256<byte>.Zero.Store(nb + 32);
            // The vector-store limit depends only on the rows' width: computed again only when a re-layout changed it (a 64-bit
            // division per inner-run entry would cost more than a short element).
            int limitWidth = -1;
            long vectorLimit = 0;
            long k = 0;
            while (k < count)
            {
                int refusal = RunExact;
                int width = rows.RowWidth;
                if (vector && width <= 128)
                {
                    if (width != limitWidth)
                    {
                        vectorLimit = rows.VectorStoreLimit(((width + 31) >> 5) * 32 + NarrowRunSlack);
                        limitWidth = width;
                    }
                    long pos = rows.Position, maxLen = rows.MaxLength;
                    long done = EncodeNarrowRun(pa, sa, count - k, chars, rows.RowBase, width, vectorLimit, utf8, limit, handler,
                        nb, ref pos, ref maxLen, out refusal);
                    rows.Position = pos;
                    rows.MaxLength = maxLen;
                    k += done;
                    pa += done * sa;
                    if (k == count)
                        break;
                }
                if (refusal == RunCodec)
                {
                    uint* units = (uint*)pa;
                    int n = NDCharArray.TrimmedUnits(pa, chars, true);
                    // A one-unit text past U+10FFFF ends the run for the general route's SystemError.
                    if (n == 1 && units[0] > 0x10FFFF)
                        break;
                    EncodeThroughArena(w, units, n);
                }
                else if (!EncodeNarrowElement(w, pa, utf8, limit))
                {
                    break;
                }
                k++;
                pa += sa;
            }
            return k;
        }

        /// <summary>
        ///     The narrow encode's inner run (no call in it): elements written straight into their rows (up to 128 bytes wide)
        ///     with whole-vector stores — narrowed when every code point is below the limit, UTF-8-encoded past ASCII, or with
        ///     the inline handler's replacements under ASCII / Latin-1 — up to the first element it does not take or the
        ///     result's last rows.
        /// </summary>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <param name="chars">The element width in code points (at most 32).</param>
        /// <param name="rowBase">The first row.</param>
        /// <param name="width">The rows' width in bytes (at most 128).</param>
        /// <param name="vectorLimit">The first row the row stores, the UTF-8 writers' words and the handler writes' 32-byte copies
        ///     could pass the result's end from (<see cref="NarrowRunSlack"/> past the row's stores).</param>
        /// <param name="utf8">True for UTF-8 (ASCII / Latin-1 otherwise).</param>
        /// <param name="limit">The one-byte limit: 0x80, Latin-1 0x100.</param>
        /// <param name="handler">The inline handler the run writes (<see cref="RunHandlerOf"/>), ASCII / Latin-1 only.</param>
        /// <param name="nb">64 bytes of scratch whose last 32 are zero (an element's narrowed bytes for the handler writes).</param>
        /// <param name="posRef">The next row; advanced past the elements written.</param>
        /// <param name="maxLenRef">The longest element so far; raised by the elements written.</param>
        /// <param name="refusal">Why the run stopped before the chunk's end (<see cref="RunExact"/> / <see cref="RunCodec"/>).</param>
        /// <returns>The number of elements written.</returns>
        private static long EncodeNarrowRun(byte* pa, long sa, long count, int chars, byte* rowBase, int width, long vectorLimit,
            bool utf8, uint limit, int handler, byte* nb, ref long posRef, ref long maxLenRef, out int refusal)
        {
            // Copied into locals (registers) for the loop, written back once at the end.
            long pos = posRef, maxLen = maxLenRef;
            refusal = RunExact;
            int vectors = (chars + 7) >> 3;
            int rowStores = (width + 31) >> 5;
            // Lane l of load j is active when 8j + l < chars: the masked loads never read past the element.
            var lane = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var cv = Vector256.Create(chars);
            var m0 = Avx2.CompareGreaterThan(cv, lane);
            var m1 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(8));
            var m2 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(16));
            var m3 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(24));
            var high = Vector256.Create((int)~(limit - 1));
            var limitVector = Vector256.Create(limit);
            var question = Vector256.Create((int)'?');
            var order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
            var zero = Vector256<int>.Zero;
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                if (pos >= vectorLimit)
                    break;
                int* p = (int*)pa;
                var v0 = Avx2.MaskLoad(p, m0);
                var v1 = vectors > 1 ? Avx2.MaskLoad(p + 8, m1) : zero;
                var v2 = vectors > 2 ? Avx2.MaskLoad(p + 16, m2) : zero;
                var v3 = vectors > 3 ? Avx2.MaskLoad(p + 24, m3) : zero;
                // The trimmed length: one past the last non-zero lane (trailing NUL units are not part of np.str_).
                uint nz = (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v0, zero).AsSingle()) & 0xFF)
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v1, zero).AsSingle()) & 0xFF) << 8
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v2, zero).AsSingle()) & 0xFF) << 16
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v3, zero).AsSingle()) & 0xFF) << 24;
                int n = 32 - BitOperations.LeadingZeroCount(nz);
                byte* row = rowBase + pos * width;
                bool oneByte = Avx.TestZ(v0 | v1 | v2 | v3, high);
                if (!oneByte && !utf8 && handler == HandlerReplace)
                {
                    // replace writes one '?' per unencodable character: the lanes at or past the limit become '?' and the
                    // element narrows like a one-byte one. A one-unit text past U+10FFFF is the general route's SystemError.
                    if (n == 1 && (uint)v0.ToScalar() > 0x10FFFF)
                    {
                        refusal = RunCodec;
                        break;
                    }
                    v0 = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(v0.AsUInt32(), limitVector).AsInt32(), question, v0);
                    v1 = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(v1.AsUInt32(), limitVector).AsInt32(), question, v1);
                    v2 = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(v2.AsUInt32(), limitVector).AsInt32(), question, v2);
                    v3 = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(v3.AsUInt32(), limitVector).AsInt32(), question, v3);
                    oneByte = true;
                }
                if (oneByte)
                {
                    // Every code point below the limit: its bytes ARE the encoding. packusdw / packuswb keep each 128-bit lane:
                    // dwords [a0-3 b0-3 c0-3 d0-3 | a4-7 b4-7 c4-7 d4-7] → order. A row past 32 bytes gets zeros after.
                    if (n > width)
                        break;
                    var packed = Avx2.PackUnsignedSaturate(Avx2.PackUnsignedSaturate(v0, v1).AsInt16(),
                        Avx2.PackUnsignedSaturate(v2, v3).AsInt16());
                    Avx2.PermuteVar8x32(packed.AsInt32(), order).AsByte().Store(row);
                    if (rowStores > 1)
                        ZeroRowTail(row, rowStores);
                    pos++;
                    if (n > maxLen)
                        maxLen = n;
                    continue;
                }
                uint* units = (uint*)pa;
                if (!utf8)
                {
                    // ASCII / Latin-1 with an inline handler the run writes: the element sized first (each unencodable
                    // character's replacement on its own), then written over the zeroed row.
                    if (handler == HandlerNone || (n == 1 && units[0] > 0x10FFFF))
                    {
                        refusal = RunCodec;
                        break;
                    }
                    // The unencodable lanes (unsigned: a unit past 0x7FFFFFFF of a raw U view is unencodable too).
                    var b0 = Vector256.GreaterThanOrEqual(v0.AsUInt32(), limitVector);
                    var b1 = Vector256.GreaterThanOrEqual(v1.AsUInt32(), limitVector);
                    var b2 = Vector256.GreaterThanOrEqual(v2.AsUInt32(), limitVector);
                    var b3 = Vector256.GreaterThanOrEqual(v3.AsUInt32(), limitVector);
                    uint bad = b0.ExtractMostSignificantBits() | b1.ExtractMostSignificantBits() << 8
                               | b2.ExtractMostSignificantBits() << 16 | b3.ExtractMostSignificantBits() << 24;
                    long len = ReplacedLength(handler, n, bad, vectors, v0.AsUInt32(), v1.AsUInt32(), v2.AsUInt32(),
                        v3.AsUInt32(), b0, b1, b2, b3);
                    if (len < 0)
                    {
                        // A replacement the run does not write (surrogateescape outside U+DC80..U+DCFF, an XML reference of
                        // a unit past 0x7FFFFFFF): the codec.
                        refusal = RunCodec;
                        break;
                    }
                    if (len > width)
                        break;
                    // Each run of encodable characters moved by one 32-byte copy of the narrowed bytes (the bytes past it are
                    // rewritten by what follows; after the last run they are the copy's zero tail), each unencodable one
                    // replaced: one iteration per unencodable character, no branch per character.
                    ZeroRow(row, rowStores);
                    var narrowed = Avx2.PackUnsignedSaturate(Avx2.PackUnsignedSaturate(v0, v1).AsInt16(),
                        Avx2.PackUnsignedSaturate(v2, v3).AsInt16());
                    Avx2.PermuteVar8x32(narrowed.AsInt32(), order).AsByte().Store(nb);
                    int c = 0, prev = 0;
                    uint rest = bad;
                    while (rest != 0)
                    {
                        int i = BitOperations.TrailingZeroCount(rest);
                        rest &= rest - 1;
                        Vector256.Load(nb + prev).Store(row + c);
                        c += i - prev;
                        c = WriteHandler(handler, units[i], row, c);
                        prev = i + 1;
                    }
                    Vector256.Load(nb + prev).Store(row + c);
                    pos++;
                    if (len > maxLen)
                        maxLen = len;
                    continue;
                }
                if (!Utf8Plain(v0, v1, v2, v3))
                {
                    refusal = RunCodec;
                    break;
                }
                int a7F = LanesAbove(v0, v1, v2, v3, 0x7F);
                int a7FF = LanesAbove(v0, v1, v2, v3, 0x7FF);
                int aFFFF = LanesAbove(v0, v1, v2, v3, 0xFFFF);
                int len8 = n + a7F + a7FF + aFFFF;
                if (len8 > width)
                    break;
                // The row zeroed (its padding), then the UTF-8 written over it: the writers' words end at most 3 bytes past
                // the text, inside the reach the limit was computed for.
                ZeroRow(row, rowStores);
                if (a7FF == 0)
                {
                    if (vectors == 1)
                    {
                        // At most eight code points below U+0800: each lane's UTF-8 as a 16-bit word (the ASCII lanes' high byte
                        // empty), the words packed to 16 bytes, then one vpshufb from the expand table keeps each word's first
                        // byte and the two-byte lanes' second (the table pads with zeros, so the bytes after the text are zero).
                        var u = v0.AsUInt32();
                        var isTwo = Vector256.GreaterThanOrEqual(u, Vector256.Create(0x80u));
                        var two = (Vector256.Create(0xC0u) | (u >> 6)) | ((Vector256.Create(0x80u) | (u & Vector256.Create(0x3Fu))) << 8);
                        var words = Vector256.ConditionalSelect(isTwo, two, u).AsInt32();
                        var packedWords = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(words, zero).AsInt64(), 0xD8).AsByte().GetLower();
                        uint twoMask = isTwo.ExtractMostSignificantBits();
                        Ssse3.Shuffle(packedWords, Utf8Shuffles.Expand(twoMask)).Store(row);
                    }
                    else
                    {
                        WriteUtf8UpTo2(units, n, row);
                    }
                }
                else if (aFFFF == 0 && a7F == n && a7FF == n)
                {
                    if (n <= 8)
                    {
                        // Every code point three bytes long and at most eight of them: each lane's three bytes built in the
                        // lane (E0 | u >> 12, 80 | u >> 6 & 3F, 80 | u & 3F), lanes past the element cleared, then vpshufb packs
                        // four lanes into 12 bytes per half; the high half's 12 bytes go right after the low half's (its store
                        // at byte 12 overwrites the low half's four zero bytes).
                        var u = v0.AsUInt32();
                        var b = ((u >> 12) | Vector256.Create(0xE0u))
                                | ((((u >> 6) & Vector256.Create(0x3Fu)) | Vector256.Create(0x80u)) << 8)
                                | (((u & Vector256.Create(0x3Fu)) | Vector256.Create(0x80u)) << 16);
                        b &= ~Vector256.Equals(u, Vector256<uint>.Zero);
                        var packed = Avx2.Shuffle(b.AsByte(), Vector256.Create(
                            (byte)0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80,
                            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80));
                        packed.GetLower().Store(row);
                        packed.GetUpper().Store(row + 12);
                    }
                    else
                    {
                        WriteUtf8All3(units, n, row);
                    }
                }
                else
                {
                    WriteUtf8Branchless(units, n, row);
                }
                pos++;
                if (len8 > maxLen)
                    maxLen = len8;
            }
            posRef = pos;
            maxLenRef = maxLen;
            return k;
        }

        /// <summary>
        ///     The vpshufb tables of the one- / two-byte UTF-8 vector paths, built at their first use (4 KB each): a 256-entry
        ///     table per direction, indexed by an 8-bit lane mask, each entry 16 shuffle bytes (0x80: a zero byte).
        /// </summary>
        private static class Utf8Shuffles
        {
            /// <summary>
            ///     Encode: for the mask of the code points (of eight) that take two bytes, the shuffle of their eight 16-bit UTF-8
            ///     words that keeps each word's first byte and, for a two-byte code point, its second — the UTF-8 of the eight,
            ///     packed, then zeros.
            /// </summary>
            private static readonly byte[] ExpandTable = Build(expand: true);

            /// <summary>
            ///     Decode: for the mask of the byte positions (of eight) where a code point starts, the shuffle of their eight
            ///     16-bit values that gathers the starts' values to the front, then zeros.
            /// </summary>
            private static readonly byte[] CompactTable = Build(expand: false);

            /// <summary>The expand shuffle of a two-byte mask (see <see cref="ExpandTable"/>).</summary>
            /// <param name="mask">The mask (0..255).</param>
            /// <returns>The shuffle.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal static Vector128<byte> Expand(uint mask)
                => Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(ExpandTable), mask * 16);

            /// <summary>The compact shuffle of a start mask (see <see cref="CompactTable"/>).</summary>
            /// <param name="mask">The mask (0..255).</param>
            /// <returns>The shuffle.</returns>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal static Vector128<byte> Compact(uint mask)
                => Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(CompactTable), mask * 16);

            /// <summary>Builds one table (see the field docs).</summary>
            /// <param name="expand">True for the encode table, false for the decode table.</param>
            /// <returns>256 entries of 16 shuffle bytes.</returns>
            private static byte[] Build(bool expand)
            {
                var table = new byte[256 * 16];
                for (int mask = 0; mask < 256; mask++)
                {
                    int at = mask * 16, o = 0;
                    for (int lane = 0; lane < 8; lane++)
                    {
                        bool set = ((mask >> lane) & 1) != 0;
                        if (expand)
                        {
                            table[at + o++] = (byte)(2 * lane);
                            if (set)
                                table[at + o++] = (byte)(2 * lane + 1);
                        }
                        else if (set)
                        {
                            table[at + o++] = (byte)(2 * lane);
                            table[at + o++] = (byte)(2 * lane + 1);
                        }
                    }
                    while (o < 16)
                        table[at + o++] = 0x80;
                }
                return table;
            }
        }

        /// <summary>Whether an errors policy is <c>namereplace</c> (the exact name: CPython looks handler names up without
        ///     normalizing them).</summary>
        /// <param name="errors">The policy.</param>
        /// <returns>True for <c>namereplace</c>.</returns>
        private static bool IsNameReplace(NDCodecErrors errors)
            => errors.Inline == NDErrorHandler.Other && string.Equals(errors.Name, "namereplace", StringComparison.Ordinal);

        /// <summary>
        ///     The ASCII / Latin-1 encode run under <c>namereplace</c> (CPython's generic call, <c>PyCodec_NameReplaceErrors</c>:
        ///     each unencodable character as <c>\N{NAME}</c> where unicodedata names it, else its backslash form; both ASCII, so
        ///     the ASCII codec takes them): the vector inner run (<see cref="EncodeNameRun"/>) over the rows the exact pre-pass
        ///     sized, and each element it refuses on its own — sized from the memoized name lengths when the pre-pass did not
        ///     size it, then written straight into its row when it fits (the names copied in place), else through the codec,
        ///     which re-lays the rows out; a one-unit text past U+10FFFF stops the run (the general route raises CPython's
        ///     SystemError).
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <param name="limit">The one-byte limit: 0x80, Latin-1 0x100.</param>
        /// <returns>The number of elements coded.</returns>
        /// <exception cref="Exception">The codec's error for an element it codes through the arena.</exception>
        private static long FastEncodeNameReplace(CodecWalk w, byte* pa, long sa, long count, uint limit)
        {
            var rows = w.Rows;
            int chars = w.AReader.Chars;
            bool vector = Avx2.IsSupported && chars <= 32;
            // An element's narrowed bytes for the inner run: 32 bytes, then zeros (a run of encodable characters is moved by
            // a 32-byte copy from any of them, and the element's last copy ends in the zeros).
            byte* nb = stackalloc byte[64];
            Vector256<byte>.Zero.Store(nb + 32);
            // The vector-store limit depends only on the rows' width (see FastEncodeNarrow).
            int limitWidth = -1;
            long vectorLimit = 0;
            long k = 0;
            while (k < count)
            {
                int width = rows.RowWidth;
                // The inner run writes only rows the exact pre-pass sized (EstimateRows: every element of the chunk fits its
                // row, so the run never sizes one), and copies only replacements already built: the cache exists once a
                // namereplace call built its first one (the first element with an unencodable character builds it below).
                long sized = w.SizedThrough - rows.Position;
                var cache = Volatile.Read(ref _nameReplacements);
                if (vector && sized > 0 && cache is not null)
                {
                    if (width != limitWidth)
                    {
                        // The run's stores reach the row's whole-vector zeroing, or 32 bytes past the element's last byte —
                        // at most the row's width — whichever is further.
                        vectorLimit = rows.VectorStoreLimit((((width + 31) >> 5) << 5) + NarrowRunSlack);
                        limitWidth = width;
                    }
                    long pos = rows.Position, maxLen = rows.MaxLength;
                    long done = EncodeNameRun(pa, sa, Math.Min(count - k, sized), chars, rows.RowBase, width, vectorLimit,
                        limit, cache, nb, ref pos, ref maxLen);
                    rows.Position = pos;
                    rows.MaxLength = maxLen;
                    k += done;
                    pa += done * sa;
                    if (k == count)
                        break;
                }
                // One element with exact writes: the element the run stopped at (a character without a built replacement, a
                // character past U+FFFF, the result's last rows), every element on a host without AVX2 or past 32 characters,
                // and the elements past the sized rows.
                uint* units = (uint*)pa;
                int n = NDCharArray.TrimmedUnits(pa, chars, true);
                if (n == 1 && units[0] > 0x10FFFF)
                    break;
                // An element the exact pre-pass sized (EstimateRows) fits its row; any other is sized here first.
                if (rows.Position >= w.SizedThrough && NameReplacedLength(units, n, limit) > width)
                {
                    EncodeThroughArena(w, units, n);
                }
                else
                {
                    byte* row = rows.RowBase + rows.Position * width;
                    int c = 0;
                    for (int i = 0; i < n; i++)
                    {
                        uint u = units[i];
                        if (u < limit)
                        {
                            row[c++] = (byte)u;
                            continue;
                        }
                        // Builds (and for a BMP character caches) the replacement the inner run then copies.
                        var replacement = NameReplacementOf(u);
                        int length = replacement[0];
                        new ReadOnlySpan<byte>(replacement, 1, length).CopyTo(new Span<byte>(row + c, length));
                        c += length;
                    }
                    new Span<byte>(row + c, width - c).Clear();
                    rows.CommitDirect(c);
                }
                k++;
                pa += sa;
            }
            return k;
        }

        /// <summary>
        ///     The <c>namereplace</c> encode's inner run over ASCII / Latin-1 (no call in it): each element's row zeroed with
        ///     whole-vector stores, then written over — every run of encodable characters moved by one 32-byte copy of its
        ///     narrowed bytes, every unencodable character's cached replacement by 32-byte copies of its zero-padded bytes
        ///     (<see cref="NameReplacementOf"/>), walking only the unencodable positions — up to the first element with a
        ///     character whose replacement is not cached (one past U+FFFF, or one no call built yet), a one-unit text past
        ///     U+10FFFF, or the result's last rows.
        /// </summary>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements to write at most: every one inside the rows the exact pre-pass sized,
        ///     so each fits its row.</param>
        /// <param name="chars">The element width in code points (at most 32).</param>
        /// <param name="rowBase">The first row.</param>
        /// <param name="width">The rows' width in bytes.</param>
        /// <param name="vectorLimit">The first row whose zeroing stores or copies could pass the result's end (the copies end
        ///     at most 32 bytes past the element's last byte, which is inside the row).</param>
        /// <param name="limit">The one-byte limit: 0x80, Latin-1 0x100.</param>
        /// <param name="cache">The built BMP replacements (<see cref="_nameReplacements"/>; an entry is null until built).</param>
        /// <param name="nb">64 bytes of scratch whose last 32 are zero.</param>
        /// <param name="posRef">The next row; advanced past the elements written.</param>
        /// <param name="maxLenRef">The longest element so far; raised by the elements written.</param>
        /// <returns>The number of elements written.</returns>
        /// <remarks>
        ///     An element the run abandons for a missing replacement may leave partial bytes in its row and up to 32 bytes into
        ///     the next one: the outer loop then writes that row whole (its text, then zeros to the width), and the next row is
        ///     written later in C order, so nothing partial survives. The pre-pass's lengths (the memoized name lengths) are the
        ///     replacements' lengths, which is what keeps every element inside its row here, as in the outer loop.
        /// </remarks>
        private static long EncodeNameRun(byte* pa, long sa, long count, int chars, byte* rowBase, int width, long vectorLimit,
            uint limit, byte[][] cache, byte* nb, ref long posRef, ref long maxLenRef)
        {
            // Copied into locals (registers) for the loop, written back once at the end.
            long pos = posRef, maxLen = maxLenRef;
            int vectors = (chars + 7) >> 3;
            int rowStores = (width + 31) >> 5;
            // Lane l of load j is active when 8j + l < chars: the masked loads never read past the element.
            var lane = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var cv = Vector256.Create(chars);
            var m0 = Avx2.CompareGreaterThan(cv, lane);
            var m1 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(8));
            var m2 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(16));
            var m3 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(24));
            var limitVector = Vector256.Create(limit);
            var order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
            var zero = Vector256<int>.Zero;
            var zeroBytes = Vector256<byte>.Zero;
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                if (pos >= vectorLimit)
                    break;
                int* p = (int*)pa;
                var v0 = Avx2.MaskLoad(p, m0);
                var v1 = vectors > 1 ? Avx2.MaskLoad(p + 8, m1) : zero;
                var v2 = vectors > 2 ? Avx2.MaskLoad(p + 16, m2) : zero;
                var v3 = vectors > 3 ? Avx2.MaskLoad(p + 24, m3) : zero;
                // The trimmed length: one past the last non-zero lane (trailing NUL units are not part of np.str_).
                uint nz = (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v0, zero).AsSingle()) & 0xFF)
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v1, zero).AsSingle()) & 0xFF) << 8
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v2, zero).AsSingle()) & 0xFF) << 16
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v3, zero).AsSingle()) & 0xFF) << 24;
                int n = 32 - BitOperations.LeadingZeroCount(nz);
                // A one-unit text past U+10FFFF is the general route's SystemError (the outer loop stops before it).
                if (n == 1 && (uint)v0.ToScalar() > 0x10FFFF)
                    break;
                // The unencodable lanes (unsigned: a unit past 0x7FFFFFFF of a raw U view is unencodable too).
                uint bad = Vector256.GreaterThanOrEqual(v0.AsUInt32(), limitVector).ExtractMostSignificantBits()
                           | Vector256.GreaterThanOrEqual(v1.AsUInt32(), limitVector).ExtractMostSignificantBits() << 8
                           | Vector256.GreaterThanOrEqual(v2.AsUInt32(), limitVector).ExtractMostSignificantBits() << 16
                           | Vector256.GreaterThanOrEqual(v3.AsUInt32(), limitVector).ExtractMostSignificantBits() << 24;
                byte* row = rowBase + pos * width;
                // The row zeroed whole first (its width is the chunk's longest element, so most of it is padding); the copies
                // below write the element over the zeros. The trip count is the call's, so the loop predicts.
                for (int s = 0; s < rowStores; s++)
                    zeroBytes.Store(row + (s << 5));
                // The narrowed bytes (an unencodable lane narrows to garbage, which its replacement overwrites).
                var narrowed = Avx2.PackUnsignedSaturate(Avx2.PackUnsignedSaturate(v0, v1).AsInt16(),
                    Avx2.PackUnsignedSaturate(v2, v3).AsInt16());
                Avx2.PermuteVar8x32(narrowed.AsInt32(), order).AsByte().Store(nb);
                uint* units = (uint*)pa;
                int c = 0, prev = 0;
                uint rest = bad;
                bool missing = false;
                while (rest != 0)
                {
                    int i = BitOperations.TrailingZeroCount(rest);
                    rest &= rest - 1;
                    uint u = units[i];
                    // Only a BMP character's replacement is cached; one not built yet (or past U+FFFF) is the outer loop's.
                    // The entry is read with acquire semantics: its bytes were written before NameReplacementOf published it.
                    byte[] replacement = u < 0x10000 ? Volatile.Read(ref cache[u]) : null;
                    if (replacement is null)
                    {
                        missing = true;
                        break;
                    }
                    // The run of encodable characters before it (the copy's bytes past the run are overwritten next).
                    Vector256.Load(nb + prev).Store(row + c);
                    c += i - prev;
                    // The replacement in whole 32-byte chunks of its zero-padded bytes (byte 0 is its length): its padding
                    // lands where the element's next bytes, or the row's zeros, belong.
                    ref byte source = ref MemoryMarshal.GetArrayDataReference(replacement);
                    int length = source;
                    Vector256.LoadUnsafe(ref source, 1).Store(row + c);
                    for (int o = 32; o < length; o += 32)
                        Vector256.LoadUnsafe(ref source, (nuint)(1 + o)).Store(row + c + o);
                    c += length;
                    prev = i + 1;
                }
                if (missing)
                    break;
                // The last run of encodable characters, then zeros (the narrowed lanes past the element are zero).
                Vector256.Load(nb + prev).Store(row + c);
                c += n - prev;
                Debug.Assert(c <= width, "the exact pre-pass sized every row of the run for its element");
                pos++;
                if (c > maxLen)
                    maxLen = c;
            }
            posRef = pos;
            maxLenRef = maxLen;
            return k;
        }

        /// <summary>
        ///     The <c>namereplace</c> replacement of one character — <c>\N{NAME}</c> where unicodedata names it, else its
        ///     backslash form (the codec's generic handler writes the same bytes) — built once per BMP character and cached
        ///     (characters past U+FFFF are built each time).
        /// </summary>
        /// <param name="u">The character.</param>
        /// <returns>The replacement: byte 0 its length (at most <see cref="PyUnicodeNames.LongestName"/> + 4), then its ASCII
        ///     bytes, zero-padded to whole 32-byte chunks (<see cref="EncodeNameRun"/> copies it in 32-byte vectors). A cached
        ///     array: never written to.</returns>
        /// <exception cref="System.IO.InvalidDataException">The embedded name table is corrupt (a build defect).</exception>
        private static byte[] NameReplacementOf(uint u)
        {
            byte[][] cache = null;
            if (u < 0x10000)
            {
                // The cache of 65,536 references (512 KB) exists only once a namereplace call met a character: published
                // with a compare-exchange, entries with a release write (a racing thread builds and publishes the same bytes).
                cache = Volatile.Read(ref _nameReplacements);
                if (cache is null)
                {
                    Interlocked.CompareExchange(ref _nameReplacements, new byte[0x10000][], null);
                    cache = _nameReplacements;
                }
                var cached = Volatile.Read(ref cache[u]);
                if (cached is not null)
                    return cached;
            }
            byte* text = stackalloc byte[PyUnicodeNames.LongestName + 16];
            int nameLength = PyUnicodeNames.GetName(u, new Span<byte>(text + 3, PyUnicodeNames.LongestName));
            int length;
            if (nameLength > 0)
            {
                text[0] = (byte)'\\';
                text[1] = (byte)'N';
                text[2] = (byte)'{';
                length = 3 + nameLength;
                text[length++] = (byte)'}';
            }
            else
            {
                length = WriteHandler(HandlerBackslash, u, text, 0);
            }
            // Byte 0 the length (at most 88 + 4), then the bytes, then zeros to the next 32-byte boundary of the bytes: the
            // inner run's 32-byte loads stay inside the array, and their zeros are the padding it may write past the bytes.
            var replacement = new byte[1 + ((length + 31) & ~31)];
            replacement[0] = (byte)length;
            new ReadOnlySpan<byte>(text, length).CopyTo(new Span<byte>(replacement, 1, length));
            if (cache is not null)
                Volatile.Write(ref cache[u], replacement);
            return replacement;
        }

        /// <summary>The BMP <c>namereplace</c> replacements built so far (<see cref="NameReplacementOf"/>), null until the first.</summary>
        private static byte[][] _nameReplacements;

        /// <summary>
        ///     Forgets every built <c>namereplace</c> replacement (tests only): the next calls build them again, so the inner run
        ///     meets characters without a built replacement in the middle of a call, which a warm process never shows again.
        /// </summary>
        /// <remarks>Safe while another call runs: a run keeps the cache it started with (its entries stay valid), and the next
        ///     run starts over with a new cache.</remarks>
        internal static void ResetNameReplacementsForTests() => Volatile.Write(ref _nameReplacements, null);

        /// <summary>
        ///     The longest element of a chunk under ASCII / Latin-1 with <c>namereplace</c> — the exact pre-pass that sizes the
        ///     chunk's rows (<see cref="CodecWalk.EstimateRows"/>) — with each element's length from
        ///     <see cref="NameReplacedLength"/>'s rule: on AVX2, an element of up to 32 code points sized without a branch per
        ///     character (the unencodable lanes' memoized name lengths gathered, eight per <c>vpgatherdd</c>), an element with a
        ///     character the memo does not know yet (or one past U+FFFF) by the scalar rule, which fills the memo.
        /// </summary>
        /// <param name="pa">The chunk's first element.</param>
        /// <param name="sa">The element stride in bytes.</param>
        /// <param name="count">The number of elements.</param>
        /// <param name="chars">The element width in code points.</param>
        /// <param name="limit">The one-byte limit: 0x80, Latin-1 0x100.</param>
        /// <returns>The longest element's length in bytes (0 when every element is empty). A one-unit text past U+10FFFF is
        ///     skipped: the walk raises CPython's SystemError there, in C order.</returns>
        /// <exception cref="System.IO.InvalidDataException">The embedded name table is corrupt (a build defect).</exception>
        internal static long NameReplacedWidest(byte* pa, long sa, long count, int chars, uint limit)
        {
            long widest = 0;
            if (!Avx2.IsSupported || chars > 32)
            {
                for (long i = 0; i < count; i++, pa += sa)
                {
                    int n = NDCharArray.TrimmedUnits(pa, chars, true);
                    if (n == 1 && *(uint*)pa > 0x10FFFF)
                        continue;
                    widest = Math.Max(widest, NameReplacedLength((uint*)pa, n, limit));
                }
                return widest;
            }
            // The memo of BMP name lengths: byte cp holds the length + 1 (0 unknown). A 32-bit gather at memo + cp reads it in
            // its low byte (the memo has 4 bytes past U+FFFF's entry; only BMP lanes are gathered).
            int* memo = (int*)PyUnicodeNames.NameLengthsPointer;
            int vectors = (chars + 7) >> 3;
            var lane = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var cv = Vector256.Create(chars);
            var m0 = Avx2.CompareGreaterThan(cv, lane);
            var m1 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(8));
            var m2 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(16));
            var m3 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(24));
            var zero = Vector256<int>.Zero;
            for (long i = 0; i < count; i++, pa += sa)
            {
                int* p = (int*)pa;
                var v0 = Avx2.MaskLoad(p, m0);
                var v1 = vectors > 1 ? Avx2.MaskLoad(p + 8, m1) : zero;
                var v2 = vectors > 2 ? Avx2.MaskLoad(p + 16, m2) : zero;
                var v3 = vectors > 3 ? Avx2.MaskLoad(p + 24, m3) : zero;
                // The trimmed length: one past the last non-zero lane (the lanes after it are zero, so they add nothing below).
                uint nz = (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v0, zero).AsSingle()) & 0xFF)
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v1, zero).AsSingle()) & 0xFF) << 8
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v2, zero).AsSingle()) & 0xFF) << 16
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v3, zero).AsSingle()) & 0xFF) << 24;
                int n = 32 - BitOperations.LeadingZeroCount(nz);
                if (n == 1 && (uint)v0.ToScalar() > 0x10FFFF)
                    continue;
                // Each lane's bytes past the one n counts, and whether a lane needs the scalar rule — only over the vectors
                // the element width spans (a gather costs even with an empty mask; the call's width makes the test predict).
                var extra = NameExtra(v0.AsUInt32(), limit, memo, out var unknown);
                if (vectors > 1)
                {
                    extra += NameExtra(v1.AsUInt32(), limit, memo, out var unknown1);
                    unknown |= unknown1;
                    if (vectors > 2)
                    {
                        extra += NameExtra(v2.AsUInt32(), limit, memo, out var unknown2);
                        unknown |= unknown2;
                        if (vectors > 3)
                        {
                            extra += NameExtra(v3.AsUInt32(), limit, memo, out var unknown3);
                            unknown |= unknown3;
                        }
                    }
                }
                long length = unknown != Vector256<uint>.Zero
                    ? NameReplacedLength((uint*)pa, n, limit)
                    : n + Vector256.Sum(extra);
                if (length > widest)
                    widest = length;
            }
            return widest;
        }

        /// <summary>
        ///     The bytes eight characters add under <c>namereplace</c> past the one each counts in the element's length: 0 for
        ///     an encodable character, the name's length + 3 for a named one (<c>\N{</c> + name + <c>}</c>), 3 / 5 for an unnamed
        ///     one below / from U+0100 (<c>\xNN</c> / <c>\uNNNN</c>).
        /// </summary>
        /// <param name="u">The characters (zero past the element: encodable, adding nothing).</param>
        /// <param name="limit">The one-byte limit.</param>
        /// <param name="memo">The memo of BMP name lengths (<see cref="PyUnicodeNames.NameLengthsPointer"/>).</param>
        /// <param name="unknown">All-ones in the unencodable lanes the memo cannot size: a character past U+FFFF, or one whose
        ///     length is not memoized yet (the scalar rule then sizes the element).</param>
        /// <returns>The bytes per lane (meaningless in a lane of <paramref name="unknown"/>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> NameExtra(Vector256<uint> u, uint limit, int* memo, out Vector256<uint> unknown)
        {
            var bad = Vector256.GreaterThanOrEqual(u, Vector256.Create(limit));
            var bmp = Vector256.LessThan(u, Vector256.Create(0x10000u));
            // Only the unencodable BMP lanes are read (a lane outside the mask is never loaded, whatever its value).
            var memoed = Avx2.GatherMaskVector256(Vector256<int>.Zero, memo, u.AsInt32(), (bad & bmp).AsInt32(), 1).AsUInt32()
                         & Vector256.Create(0xFFu);
            unknown = bad & (~bmp | Vector256.Equals(memoed, Vector256<uint>.Zero));
            // A memo of 1 is a character without a name: its backslash form.
            var named = Vector256.GreaterThan(memoed, Vector256.Create(1u));
            var unnamed = Vector256.Create(3u) + (Vector256.GreaterThanOrEqual(u, Vector256.Create(0x100u)) & Vector256.Create(2u));
            return Vector256.ConditionalSelect(named, memoed + Vector256.Create(2u), unnamed) & bad;
        }

        /// <summary>
        ///     The length of an element under ASCII / Latin-1 with <c>namereplace</c>: each encodable character one byte, each
        ///     other one <c>\N{NAME}</c> (the name's length + 4) or its backslash form (4 / 6 / 10 bytes).
        /// </summary>
        /// <param name="units">The element's code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="limit">The one-byte limit.</param>
        /// <returns>The length in bytes.</returns>
        private static long NameReplacedLength(uint* units, int n, uint limit)
        {
            long len = 0;
            for (int i = 0; i < n; i++)
            {
                uint u = units[i];
                if (u < limit)
                {
                    len++;
                    continue;
                }
                int nameLength = PyUnicodeNames.GetNameLength(u);
                len += nameLength > 0 ? nameLength + 4 : u >= 0x10000 ? 10 : u >= 0x100 ? 6 : 4;
            }
            return len;
        }

        /// <summary>Zeroes a row of up to <paramref name="rowStores"/> 32-byte stores (the whole row and the start of the
        ///     following rows, which C order writes later).</summary>
        /// <param name="row">The row.</param>
        /// <param name="rowStores">The stores covering it (1..4).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ZeroRow(byte* row, int rowStores)
        {
            var z = Vector256<byte>.Zero;
            z.Store(row);
            if (rowStores > 1)
                ZeroRowTail(row, rowStores);
        }

        /// <summary>Zeroes the 32-byte stores of a row after its first (rows wider than 32 bytes).</summary>
        /// <param name="row">The row.</param>
        /// <param name="rowStores">The stores covering it (2..4).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ZeroRowTail(byte* row, int rowStores)
        {
            var z = Vector256<byte>.Zero;
            z.Store(row + 32);
            if (rowStores > 2)
                z.Store(row + 64);
            if (rowStores > 3)
                z.Store(row + 96);
        }

        /// <summary>
        ///     The encoded length of an element of <paramref name="n"/> code points under a run handler, computed per lane
        ///     without a branch: each encodable character one byte, each unencodable one its replacement
        ///     (<see cref="HandlerLength"/>) — <c>ignore</c> drops it, <c>surrogateescape</c> writes one byte for U+DC80..U+DCFF,
        ///     <c>backslashreplace</c> 4 / 6 / 10 bytes from U+0000 / U+0100 / U+10000, <c>xmlcharrefreplace</c> 3 + its decimal
        ///     digits.
        /// </summary>
        /// <param name="handler">The run handler (not <see cref="HandlerNone"/> or <see cref="HandlerReplace"/>).</param>
        /// <param name="n">The element's trimmed length.</param>
        /// <param name="bad">The mask of its unencodable code points (bit i: code point i).</param>
        /// <param name="vectors">The vectors the element width spans (1..4): the ones past it are zero and are skipped.</param>
        /// <param name="u0">Code points 0..7 (zero past the element).</param>
        /// <param name="u1">Code points 8..15.</param>
        /// <param name="u2">Code points 16..23.</param>
        /// <param name="u3">Code points 24..31.</param>
        /// <param name="b0">All-ones in the unencodable lanes of <paramref name="u0"/>.</param>
        /// <param name="b1">Those of <paramref name="u1"/>.</param>
        /// <param name="b2">Those of <paramref name="u2"/>.</param>
        /// <param name="b3">Those of <paramref name="u3"/>.</param>
        /// <returns>The length, or -1 for a replacement the run does not write (<c>surrogateescape</c> outside U+DC80..U+DCFF;
        ///     an XML reference of a unit past 9,999,999, which only a raw U view holds).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ReplacedLength(int handler, int n, uint bad, int vectors, Vector256<uint> u0, Vector256<uint> u1,
            Vector256<uint> u2, Vector256<uint> u3, Vector256<uint> b0, Vector256<uint> b1, Vector256<uint> b2, Vector256<uint> b3)
        {
            // Only the vectors the element width spans are evaluated: the lanes past them are zero (encodable, adding
            // nothing), and the per-lane formulas cost tens of vector instructions each (the call's width makes the tests
            // predict).
            switch (handler)
            {
                case HandlerIgnore:
                    return n - BitOperations.PopCount(bad);
                case HandlerSurrogateEscape:
                {
                    uint escapable = EscapableBits(u0);
                    if (vectors > 1)
                    {
                        escapable |= EscapableBits(u1) << 8;
                        if (vectors > 2)
                        {
                            escapable |= EscapableBits(u2) << 16;
                            if (vectors > 3)
                                escapable |= EscapableBits(u3) << 24;
                        }
                    }
                    return (bad & ~escapable) != 0 ? -1 : n;
                }
                case HandlerBackslash:
                {
                    var extra = BackslashExtra(u0) & b0;
                    if (vectors > 1)
                    {
                        extra += BackslashExtra(u1) & b1;
                        if (vectors > 2)
                        {
                            extra += BackslashExtra(u2) & b2;
                            if (vectors > 3)
                                extra += BackslashExtra(u3) & b3;
                        }
                    }
                    return n + Vector256.Sum(extra);
                }
                default:
                {
                    // xmlcharrefreplace. A unit from 10,000,000 (raw U views only) is left to the codec.
                    var tooLong = XmlTooLong(u0) & b0;
                    var extra = XmlExtra(u0) & b0;
                    if (vectors > 1)
                    {
                        tooLong |= XmlTooLong(u1) & b1;
                        extra += XmlExtra(u1) & b1;
                        if (vectors > 2)
                        {
                            tooLong |= XmlTooLong(u2) & b2;
                            extra += XmlExtra(u2) & b2;
                            if (vectors > 3)
                            {
                                tooLong |= XmlTooLong(u3) & b3;
                                extra += XmlExtra(u3) & b3;
                            }
                        }
                    }
                    if (tooLong != Vector256<uint>.Zero)
                        return -1;
                    return n + Vector256.Sum(extra);
                }
            }
        }

        /// <summary>The lanes <c>surrogateescape</c> can write (U+DC80..U+DCFF: the byte 0x80..0xFF), as a bit mask.</summary>
        /// <param name="u">The units.</param>
        /// <returns>Bit l set when unit l is in U+DC80..U+DCFF.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint EscapableBits(Vector256<uint> u)
            => Vector256.LessThanOrEqual(u - Vector256.Create(0xDC80u), Vector256.Create(0x7Fu)).ExtractMostSignificantBits();

        /// <summary>The bytes a <c>backslashreplace</c> form adds past one per lane: 3 (<c>\xNN</c>), 5 from U+0100
        ///     (<c>\uNNNN</c>), 9 from U+10000 (<c>\UNNNNNNNN</c>); the caller masks the encodable lanes out.</summary>
        /// <param name="u">The units.</param>
        /// <returns>3, 5 or 9 per lane.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> BackslashExtra(Vector256<uint> u)
            => Vector256.Create(3u) + (Vector256.GreaterThanOrEqual(u, Vector256.Create(0x100u)) & Vector256.Create(2u))
               + (Vector256.GreaterThanOrEqual(u, Vector256.Create(0x10000u)) & Vector256.Create(4u));

        /// <summary>The lanes whose XML character reference the run does not write: a unit from 10,000,000 (eight or more
        ///     digits, or negative as a C <c>int</c>), which only a raw <c>U</c> view holds.</summary>
        /// <param name="u">The units.</param>
        /// <returns>All-ones in those lanes (the caller masks the encodable lanes out).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> XmlTooLong(Vector256<uint> u)
            => Vector256.GreaterThanOrEqual(u, Vector256.Create(10_000_000u));

        /// <summary>The bytes an XML character reference adds past one per lane: 2 ("&amp;#" and ";" less the counted byte)
        ///     + the decimal digits of a unit below 10,000,000.</summary>
        /// <param name="u">The units.</param>
        /// <returns>3 + one for each power of ten from 10 to 1,000,000 a unit reaches.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<uint> XmlExtra(Vector256<uint> u)
        {
            var one = Vector256.Create(1u);
            return Vector256.Create(3u)
                   + (Vector256.GreaterThanOrEqual(u, Vector256.Create(10u)) & one)
                   + (Vector256.GreaterThanOrEqual(u, Vector256.Create(100u)) & one)
                   + (Vector256.GreaterThanOrEqual(u, Vector256.Create(1000u)) & one)
                   + (Vector256.GreaterThanOrEqual(u, Vector256.Create(10000u)) & one)
                   + (Vector256.GreaterThanOrEqual(u, Vector256.Create(100000u)) & one)
                   + (Vector256.GreaterThanOrEqual(u, Vector256.Create(1000000u)) & one);
        }

        /// <summary>
        ///     The length of one unencodable character's replacement under a run handler (CPython's inline forms of
        ///     <c>unicode_encode_ucs1</c>): <c>replace</c> one <c>?</c>, <c>ignore</c> nothing, <c>xmlcharrefreplace</c>
        ///     <c>&amp;#</c> + the unit printed as a C <c>int</c> + <c>;</c>, <c>backslashreplace</c> <c>\xXX</c> /
        ///     <c>\uXXXX</c> / <c>\UXXXXXXXX</c>, <c>surrogateescape</c> one byte for U+DC80..U+DCFF.
        /// </summary>
        /// <param name="handler">The run handler (not <see cref="HandlerNone"/>).</param>
        /// <param name="u">The character (at or past the codec's limit).</param>
        /// <returns>The replacement's length in bytes, or -1 when <c>surrogateescape</c> cannot write it (the generic call
        ///     raises).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HandlerLength(int handler, uint u)
        {
            switch (handler)
            {
                case HandlerReplace:
                    return 1;
                case HandlerIgnore:
                    return 0;
                case HandlerXml:
                    return 3 + CIntLength((int)u);
                case HandlerBackslash:
                    return u >= 0x10000 ? 10 : u >= 0x100 ? 6 : 4;
                default:
                    return u - 0xDC80u <= 0x7Fu ? 1 : -1;
            }
        }

        /// <summary>
        ///     Writes one unencodable character's replacement under a run handler (see <see cref="HandlerLength"/>; the same
        ///     bytes as the codec's <c>PutXmlCharRefReplace</c> / <c>PutBackslashReplace</c>).
        /// </summary>
        /// <param name="handler">The run handler.</param>
        /// <param name="u">The character.</param>
        /// <param name="dst">The row.</param>
        /// <param name="c">Where to write.</param>
        /// <returns>The position after the replacement.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int WriteHandler(int handler, uint u, byte* dst, int c)
        {
            switch (handler)
            {
                case HandlerReplace:
                    dst[c] = (byte)'?';
                    return c + 1;
                case HandlerIgnore:
                    return c;
                case HandlerXml:
                {
                    dst[c] = (byte)'&';
                    dst[c + 1] = (byte)'#';
                    c += 2;
                    int v = (int)u;
                    uint mag = (uint)v;
                    if (v < 0)
                    {
                        // printf("%d") of the unit read as a C int (a unit past 0x7FFFFFFF prints negative).
                        dst[c++] = (byte)'-';
                        mag = (uint)(-(long)v);
                    }
                    int digits = UIntLength(mag);
                    for (int d = digits - 1; d >= 0; d--)
                    {
                        dst[c + d] = (byte)('0' + mag % 10);
                        mag /= 10;
                    }
                    c += digits;
                    dst[c] = (byte)';';
                    return c + 1;
                }
                case HandlerBackslash:
                {
                    // \UXXXXXXXX from U+10000 (all eight hex digits of the 32-bit unit), \uXXXX from U+0100, \xXX below;
                    // lowercase hex.
                    dst[c] = (byte)'\\';
                    int digits;
                    if (u >= 0x10000)
                    {
                        dst[c + 1] = (byte)'U';
                        digits = 8;
                    }
                    else if (u >= 0x100)
                    {
                        dst[c + 1] = (byte)'u';
                        digits = 4;
                    }
                    else
                    {
                        dst[c + 1] = (byte)'x';
                        digits = 2;
                    }
                    c += 2;
                    for (int d = digits - 1; d >= 0; d--)
                    {
                        uint h = u & 0xF;
                        // '0'..'9', then 'a'..'f' ('a' - '0' - 10 = 39).
                        dst[c + d] = (byte)('0' + h + ((9 - (int)h) >> 31 & 39));
                        u >>= 4;
                    }
                    return c + digits;
                }
                default:
                    // surrogateescape: (char)(ch - 0xdc00), the byte 0x80..0xFF, written even by the ASCII codec.
                    dst[c] = (byte)(u - 0xDC00);
                    return c + 1;
            }
        }

        /// <summary>The number of characters <c>printf("%d")</c> writes for a C <c>int</c> (the sign included).</summary>
        /// <param name="v">The value.</param>
        /// <returns>1..11.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CIntLength(int v) => v < 0 ? 1 + UIntLength((uint)(-(long)v)) : UIntLength((uint)v);

        /// <summary>The number of decimal digits of an unsigned value (1 for zero).</summary>
        /// <param name="u">The value.</param>
        /// <returns>1..10.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int UIntLength(uint u)
            => 1 + (u >= 10 ? 1 : 0) + (u >= 100 ? 1 : 0) + (u >= 1000 ? 1 : 0) + (u >= 10000 ? 1 : 0) + (u >= 100000 ? 1 : 0)
               + (u >= 1000000 ? 1 : 0) + (u >= 10000000 ? 1 : 0) + (u >= 100000000 ? 1 : 0) + (u >= 1000000000 ? 1 : 0);

        /// <summary>
        ///     One <c>U</c> element of the narrow encode with exact writes (the result's last rows, an element wider than 32
        ///     code points or rows wider than 128 bytes, a host without AVX2, a row too narrow for the element) — narrowed or
        ///     UTF-8-encoded straight into its row when it fits, else through the codec into the arena.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The element.</param>
        /// <param name="utf8">True for UTF-8.</param>
        /// <param name="limit">The one-byte limit (0x80, Latin-1 0x100).</param>
        /// <returns>False for a one-unit element past U+10FFFF (left to the general route), else true (coded).</returns>
        /// <exception cref="Exception">The codec's error.</exception>
        private static bool EncodeNarrowElement(CodecWalk w, byte* pa, bool utf8, uint limit)
        {
            var rows = w.Rows;
            uint* units = (uint*)pa;
            int n = NDCharArray.TrimmedUnits(pa, w.AReader.Chars, true);
            if (n == 1 && units[0] > 0x10FFFF)
                return false;
            int width = rows.RowWidth;
            byte* row = rows.RowBase + rows.Position * width;
            uint acc = 0;
            bool plain = true;
            long len = 0;
            for (int i = 0; i < n; i++)
            {
                uint u = units[i];
                acc |= u;
                if (u - 0xD800u <= 0x7FFu || u > 0x10FFFF)
                    plain = false;
                len += u < 0x80 ? 1 : u < 0x800 ? 2 : u < 0x10000 ? 3 : 4;
            }
            if (acc < limit && n <= width)
            {
                for (int i = 0; i < n; i++)
                    row[i] = (byte)units[i];
                new Span<byte>(row + n, width - n).Clear();
                rows.CommitDirect(n);
                return true;
            }
            if (utf8 && plain && len <= width)
            {
                int written = WriteUtf8(units, n, row);
                new Span<byte>(row + written, width - written).Clear();
                rows.CommitDirect(written);
                return true;
            }
            EncodeThroughArena(w, units, n);
            return true;
        }

        /// <summary>Whether no lane of four vectors is a surrogate or past U+10FFFF (UTF-8 then needs neither an error handler
        ///     nor CPython's arithmetic on an invalid unit).</summary>
        /// <param name="v0">Units 0..7.</param>
        /// <param name="v1">Units 8..15.</param>
        /// <param name="v2">Units 16..23.</param>
        /// <param name="v3">Units 24..31.</param>
        /// <returns>True when every lane is a plain code point.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Utf8Plain(Vector256<int> v0, Vector256<int> v1, Vector256<int> v2, Vector256<int> v3)
        {
            var maxValid = Vector256.Create(0x10FFFFu);
            var surrogate = Vector256.Create(0xD800u);
            var span = Vector256.Create(0x7FFu);
            var u0 = v0.AsUInt32();
            var u1 = v1.AsUInt32();
            var u2 = v2.AsUInt32();
            var u3 = v3.AsUInt32();
            // A lane is refused when it is past U+10FFFF or within 0x7FF above U+D800 (unsigned arithmetic).
            var bad = Vector256.GreaterThan(u0, maxValid) | Vector256.LessThanOrEqual(u0 - surrogate, span)
                      | Vector256.GreaterThan(u1, maxValid) | Vector256.LessThanOrEqual(u1 - surrogate, span)
                      | Vector256.GreaterThan(u2, maxValid) | Vector256.LessThanOrEqual(u2 - surrogate, span)
                      | Vector256.GreaterThan(u3, maxValid) | Vector256.LessThanOrEqual(u3 - surrogate, span);
            return bad == Vector256<uint>.Zero;
        }

        /// <summary>The number of lanes of four vectors above a threshold (each one more UTF-8 byte of its code point; the
        ///     zero lanes past the element count nothing).</summary>
        /// <param name="v0">Units 0..7.</param>
        /// <param name="v1">Units 8..15.</param>
        /// <param name="v2">Units 16..23.</param>
        /// <param name="v3">Units 24..31.</param>
        /// <param name="threshold">The threshold (0x7F, 0x7FF or 0xFFFF; the lanes are at most U+10FFFF, so a signed compare
        ///     is exact).</param>
        /// <returns>The count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int LanesAbove(Vector256<int> v0, Vector256<int> v1, Vector256<int> v2, Vector256<int> v3, int threshold)
        {
            var t = Vector256.Create(threshold);
            uint bits = (uint)Avx2.MoveMask(Avx2.CompareGreaterThan(v0, t).AsSingle())
                        | (uint)Avx2.MoveMask(Avx2.CompareGreaterThan(v1, t).AsSingle()) << 8
                        | (uint)Avx2.MoveMask(Avx2.CompareGreaterThan(v2, t).AsSingle()) << 16
                        | (uint)Avx2.MoveMask(Avx2.CompareGreaterThan(v3, t).AsSingle()) << 24;
            return BitOperations.PopCount(bits);
        }

        /// <summary>
        ///     The UTF-8 of plain code points below U+0800 without a branch per code point: the one- and two-byte forms
        ///     selected with a mask and stored as one 16-bit word (a one-byte form's high byte zero, overwritten by the next
        ///     code point).
        /// </summary>
        /// <param name="units">The code points (all below U+0800).</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The destination: room for the UTF-8 and 1 byte past it.</param>
        /// <returns>The number of bytes of the UTF-8.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int WriteUtf8UpTo2(uint* units, int n, byte* dst)
        {
            int c = 0;
            for (int i = 0; i < n; i++)
            {
                uint u = units[i];
                // All-ones when the code point needs a second byte (an arithmetic shift of 0x7F - u).
                uint two = (uint)((int)(0x7Fu - u) >> 31);
                uint b2 = 0x80C0u | (u >> 6) | ((u & 0x3F) << 8);
                *(ushort*)(dst + c) = (ushort)((u & ~two) | (b2 & two));
                c += 1 + (int)(two & 1);
            }
            return c;
        }

        /// <summary>
        ///     The UTF-8 of code points that all take three bytes (U+0800..U+FFFF, no surrogate): each one's three bytes as a
        ///     32-bit word at a fixed offset (three bytes per code point), so no store waits for the previous one's length.
        /// </summary>
        /// <param name="units">The code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The destination: room for 3n bytes and 1 byte past them.</param>
        /// <returns>The number of bytes (3n).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int WriteUtf8All3(uint* units, int n, byte* dst)
        {
            for (int i = 0; i < n; i++)
            {
                uint u = units[i];
                *(uint*)(dst + 3 * i) = 0x8080E0u | (u >> 12) | (((u >> 6) & 0x3F) << 8) | ((u & 0x3F) << 16);
            }
            return 3 * n;
        }

        /// <summary>
        ///     The UTF-8 of plain code points (no surrogate, none past U+10FFFF) without a branch per code point: its one- to
        ///     four-byte forms computed and the right one selected with masks, then stored as one 32-bit word (its unused high
        ///     bytes zero, overwritten by the next code point) — random text mispredicts no branch.
        /// </summary>
        /// <param name="units">The code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The destination: room for the UTF-8 and 3 bytes past it.</param>
        /// <returns>The number of bytes of the UTF-8.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int WriteUtf8Branchless(uint* units, int n, byte* dst)
        {
            int c = 0;
            for (int i = 0; i < n; i++)
            {
                uint u = units[i];
                // All-ones when the code point needs a second, third, fourth byte (an arithmetic shift of 0x7F - u ...).
                uint m2 = (uint)((int)(0x7Fu - u) >> 31);
                uint m3 = (uint)((int)(0x7FFu - u) >> 31);
                uint m4 = (uint)((int)(0xFFFFu - u) >> 31);
                uint b2 = 0x80C0u | (u >> 6) | ((u & 0x3F) << 8);
                uint b3 = 0x8080E0u | (u >> 12) | (((u >> 6) & 0x3F) << 8) | ((u & 0x3F) << 16);
                uint b4 = 0x808080F0u | (u >> 18) | (((u >> 12) & 0x3F) << 8) | (((u >> 6) & 0x3F) << 16) | ((u & 0x3F) << 24);
                uint b = (u & ~m2) | (b2 & m2 & ~m3) | (b3 & m3 & ~m4) | (b4 & m4);
                *(uint*)(dst + c) = b;
                c += 1 + (int)(m2 & 1) + (int)(m3 & 1) + (int)(m4 & 1);
            }
            return c;
        }

        /// <summary>The UTF-8 of plain code points with a branch per code point (the exact-write element path).</summary>
        /// <param name="units">The code points.</param>
        /// <param name="n">Their number.</param>
        /// <param name="dst">The destination (room for the UTF-8).</param>
        /// <returns>The number of bytes written.</returns>
        private static int WriteUtf8(uint* units, int n, byte* dst)
        {
            int c = 0;
            for (int i = 0; i < n; i++)
            {
                uint ch = units[i];
                if (ch < 0x80)
                {
                    dst[c++] = (byte)ch;
                }
                else if (ch < 0x800)
                {
                    dst[c++] = (byte)(0xC0 | (ch >> 6));
                    dst[c++] = (byte)(0x80 | (ch & 0x3F));
                }
                else if (ch < 0x10000)
                {
                    dst[c++] = (byte)(0xE0 | (ch >> 12));
                    dst[c++] = (byte)(0x80 | ((ch >> 6) & 0x3F));
                    dst[c++] = (byte)(0x80 | (ch & 0x3F));
                }
                else
                {
                    dst[c++] = (byte)(0xF0 | (ch >> 18));
                    dst[c++] = (byte)(0x80 | ((ch >> 12) & 0x3F));
                    dst[c++] = (byte)(0x80 | ((ch >> 6) & 0x3F));
                    dst[c++] = (byte)(0x80 | (ch & 0x3F));
                }
            }
            return c;
        }

        // =============================================================================================================
        // U → S, UTF-16 / UTF-32
        // =============================================================================================================

        /// <summary>
        ///     The UTF-16 / UTF-32 encode run over <c>U</c> elements (see the file header): the vector inner run, and each
        ///     element it refuses through the codec.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded.</returns>
        /// <exception cref="Exception">The codec's error for an element it codes through the arena.</exception>
        private static long FastEncodeWide(CodecWalk w, byte* pa, long sa, long count)
        {
            var rows = w.Rows;
            var kind = w.Codec.Kind;
            bool utf16 = kind is PythonCodecKind.Utf16 or PythonCodecKind.Utf16LE or PythonCodecKind.Utf16BE;
            int byteorder = w.Codec.ByteOrder;
            int chars = w.AReader.Chars;
            int vectors = (chars + 7) >> 3;
            int bomBytes = byteorder == 0 ? (utf16 ? 2 : 4) : 0;
            // The bytes a row's stores cover: the BOM, then 32 bytes per 16 units (UTF-16) or per 8 units (UTF-32). A row wider
            // than that (an error handler widened the rows) would keep a tail the stores never zero: those rows take the codec.
            int stored = bomBytes + (utf16 ? 32 * ((vectors + 1) >> 1) : 32 * vectors);
            bool vector = Avx2.IsSupported && chars <= 32;
            // The stores' reach is fixed for the call, so the vector-store limit changes only with the rows' width.
            int limitWidth = -1;
            long vectorLimit = 0;
            long k = 0;
            while (k < count)
            {
                int width = rows.RowWidth;
                if (vector && width <= stored)
                {
                    if (width != limitWidth)
                    {
                        vectorLimit = rows.VectorStoreLimit(stored);
                        limitWidth = width;
                    }
                    long pos = rows.Position, maxLen = rows.MaxLength;
                    long done = EncodeWideRun(pa, sa, count - k, chars, rows.RowBase, width, vectorLimit, utf16, byteorder, ref pos,
                        ref maxLen);
                    rows.Position = pos;
                    rows.MaxLength = maxLen;
                    k += done;
                    pa += done * sa;
                    if (k == count)
                        break;
                }
                uint* units = (uint*)pa;
                int n = NDCharArray.TrimmedUnits(pa, chars, true);
                // A one-unit text past U+10FFFF is the general route's SystemError.
                if (n == 1 && units[0] > 0x10FFFF)
                    break;
                EncodeThroughArena(w, units, n);
                k++;
                pa += sa;
            }
            return k;
        }

        /// <summary>
        ///     The UTF-16 / UTF-32 encode's inner run (no call in it): elements without a surrogate (UTF-16: without a code
        ///     point past U+FFFF either) written straight into their rows — the BOM of the BOM codecs, then the units packed to
        ///     16 bits or kept, byte-swapped for big-endian — up to the first element it does not take or the result's last rows.
        /// </summary>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <param name="chars">The element width in code points (at most 32).</param>
        /// <param name="rowBase">The first row.</param>
        /// <param name="width">The rows' width in bytes (at most the bytes the stores cover).</param>
        /// <param name="vectorLimit">The first row the stores could pass the result's end from.</param>
        /// <param name="utf16">True for UTF-16, false for UTF-32.</param>
        /// <param name="byteorder">0 (the BOM codec: a little-endian BOM, little-endian units), -1 little-, 1 big-endian.</param>
        /// <param name="posRef">The next row; advanced past the elements written.</param>
        /// <param name="maxLenRef">The longest element so far; raised by the elements written.</param>
        /// <returns>The number of elements written.</returns>
        private static long EncodeWideRun(byte* pa, long sa, long count, int chars, byte* rowBase, int width, long vectorLimit,
            bool utf16, int byteorder, ref long posRef, ref long maxLenRef)
        {
            long pos = posRef, maxLen = maxLenRef;
            int vectors = (chars + 7) >> 3;
            var lane = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var cv = Vector256.Create(chars);
            var m0 = Avx2.CompareGreaterThan(cv, lane);
            var m1 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(8));
            var m2 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(16));
            var m3 = Avx2.CompareGreaterThan(cv, lane + Vector256.Create(24));
            var zero = Vector256<int>.Zero;
            bool big = byteorder > 0;
            int bom = byteorder == 0 ? (utf16 ? 2 : 4) : 0;
            // The byte-swap shuffles of a 16-bit and a 32-bit lane (within each 128-bit half, as vpshufb works).
            var swap16 = Vector256.Create((byte)1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
            var swap32 = Vector256.Create((byte)3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            // UTF-16 refuses a lane past U+FFFF (a surrogate pair: the codec writes it); UTF-32 writes such units as they are.
            var aboveBmp = Vector256.Create(unchecked((int)0xFFFF0000));
            var surrogateMask = Vector256.Create(unchecked((int)0xFFFFF800));
            var surrogateValue = Vector256.Create(0xD800);
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                if (pos >= vectorLimit)
                    break;
                int* p = (int*)pa;
                var v0 = Avx2.MaskLoad(p, m0);
                var v1 = vectors > 1 ? Avx2.MaskLoad(p + 8, m1) : zero;
                var v2 = vectors > 2 ? Avx2.MaskLoad(p + 16, m2) : zero;
                var v3 = vectors > 3 ? Avx2.MaskLoad(p + 24, m3) : zero;
                uint nz = (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v0, zero).AsSingle()) & 0xFF)
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v1, zero).AsSingle()) & 0xFF) << 8
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v2, zero).AsSingle()) & 0xFF) << 16
                          | (uint)(~Avx2.MoveMask(Avx2.CompareEqual(v3, zero).AsSingle()) & 0xFF) << 24;
                int n = 32 - BitOperations.LeadingZeroCount(nz);
                // A one-unit text past U+10FFFF is CPython's SystemError at the scalar, before any codec (UTF-32 would
                // otherwise write it as it is): the general route raises it, in C order.
                if (n == 1 && (uint)v0.ToScalar() > 0x10FFFF)
                    break;
                // A surrogate lane (U+D800..U+DFFF; a unit past U+10FFFF masked to its low bits is no surrogate here, as the
                // codec writes it as it is for UTF-32 and UTF-16 refuses it below).
                var sur = Avx2.CompareEqual(v0 & surrogateMask, surrogateValue) | Avx2.CompareEqual(v1 & surrogateMask, surrogateValue)
                          | Avx2.CompareEqual(v2 & surrogateMask, surrogateValue) | Avx2.CompareEqual(v3 & surrogateMask, surrogateValue);
                if (!Avx.TestZ(sur, sur))
                    break;
                byte* row = rowBase + pos * width;
                long len;
                if (utf16)
                {
                    if (!Avx.TestZ(v0 | v1 | v2 | v3, aboveBmp))
                        break;
                    len = 2L * n + bom;
                    if (len > width)
                        break;
                    // packusdw keeps each 128-bit lane ([a0-3 b0-3 | a4-7 b4-7]); vpermq 0xD8 restores unit order.
                    var lo = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(v0, v1).AsInt64(), 0xD8).AsByte();
                    if (big)
                        lo = Avx2.Shuffle(lo, swap16);
                    if (bom != 0)
                        *(ushort*)row = 0xFEFF;   // the BOM codec: little-endian BOM (bytes FF FE), little-endian units
                    lo.Store(row + bom);
                    if (vectors > 2)
                    {
                        var hi = Avx2.Permute4x64(Avx2.PackUnsignedSaturate(v2, v3).AsInt64(), 0xD8).AsByte();
                        if (big)
                            hi = Avx2.Shuffle(hi, swap16);
                        hi.Store(row + bom + 32);
                    }
                }
                else
                {
                    len = 4L * n + bom;
                    if (len > width)
                        break;
                    if (bom != 0)
                        *(uint*)row = 0xFEFF;   // bytes FF FE 00 00
                    if (big)
                    {
                        v0 = Avx2.Shuffle(v0.AsByte(), swap32).AsInt32();
                        v1 = Avx2.Shuffle(v1.AsByte(), swap32).AsInt32();
                        v2 = Avx2.Shuffle(v2.AsByte(), swap32).AsInt32();
                        v3 = Avx2.Shuffle(v3.AsByte(), swap32).AsInt32();
                    }
                    v0.Store((int*)(row + bom));
                    if (vectors > 1)
                        v1.Store((int*)(row + bom + 32));
                    if (vectors > 2)
                        v2.Store((int*)(row + bom + 64));
                    if (vectors > 3)
                        v3.Store((int*)(row + bom + 96));
                }
                pos++;
                if (len > maxLen)
                    maxLen = len;
            }
            posRef = pos;
            maxLenRef = maxLen;
            return k;
        }

        /// <summary>
        ///     The encode run over <c>U</c> elements for a codec without a vector route (utf-8-sig, the code pages,
        ///     <c>undefined</c>): each element through the codec into the arena and its row; a one-unit element past U+10FFFF
        ///     stops the run (the general route raises CPython's SystemError).
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded.</returns>
        /// <exception cref="Exception">The codec's error for an element.</exception>
        private static long FastEncodeUnicode(CodecWalk w, byte* pa, long sa, long count)
        {
            int chars = w.AReader.Chars;
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                uint* units = (uint*)pa;
                int n = NDCharArray.TrimmedUnits(pa, chars, true);
                if (n == 1 && units[0] > 0x10FFFF)
                    break;
                EncodeThroughArena(w, units, n);
            }
            return k;
        }

        // =============================================================================================================
        // StringDType → S
        // =============================================================================================================

        /// <summary>
        ///     The encode run over <c>StringDType</c> elements (see the file header): the vector inner run over short cells, and
        ///     each cell it refuses on its own (<see cref="EncodeVStringElement"/>); an NA cell stops the run (the general route
        ///     reads the NA object and raises for one that is not a str).
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first cell.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded.</returns>
        /// <exception cref="Exception">The codec's error for an element.</exception>
        private static long FastEncodeVString(CodecWalk w, byte* pa, long sa, long count)
        {
            var rows = w.Rows;
            var kind = w.Codec.Kind;
            bool utf8 = kind == PythonCodecKind.Utf8;
            bool latin1 = kind == PythonCodecKind.Latin1;
            bool asciiCopies = kind is PythonCodecKind.Ascii or PythonCodecKind.Latin1;
            bool vector = Avx2.IsSupported && (utf8 || asciiCopies);
            // A cell's bytes for the Latin-1 decode, which reads a 16-bit word at any of them: 32 bytes, then zeros.
            byte* tmp = stackalloc byte[64];
            Vector256<byte>.Zero.Store(tmp + 32);
            // The vector-store limit depends only on the rows' width: computed again only when a re-layout changed it.
            int limitWidth = -1;
            long vectorLimit = 0;
            long k = 0;
            while (k < count)
            {
                int width = rows.RowWidth;
                if (vector && width <= 32)
                {
                    if (width != limitWidth)
                    {
                        vectorLimit = rows.VectorStoreLimit(32);
                        limitWidth = width;
                    }
                    long pos = rows.Position, maxLen = rows.MaxLength;
                    long done = EncodeVStringRun(pa, sa, count - k, rows.RowBase, width, vectorLimit, utf8, latin1, tmp, ref pos,
                        ref maxLen);
                    rows.Position = pos;
                    rows.MaxLength = maxLen;
                    k += done;
                    pa += done * sa;
                    if (k == count)
                        break;
                }
                if (!EncodeVStringElement(w, pa, utf8, latin1, asciiCopies))
                    break;
                k++;
                pa += sa;
            }
            return k;
        }

        /// <summary>
        ///     The StringDType encode's inner run (no call in it): SHORT cells (<c>is_short_string</c>: the text, at most 15
        ///     bytes, inside the 16-byte cell itself) read with one 16-byte load masked to their size and written straight into
        ///     their rows — copied under UTF-8 (and an ASCII text under ASCII / Latin-1), decoded under Latin-1 when every code
        ///     point is below U+0100 — up to the first cell it does not take (an NA, arena or heap cell, a text the row does not
        ///     hold, a code point the codec cannot write) or the result's last rows.
        /// </summary>
        /// <param name="pa">The first cell.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <param name="rowBase">The first row.</param>
        /// <param name="width">The rows' width in bytes (at most 32).</param>
        /// <param name="vectorLimit">The first row a 32-byte store could pass the result's end from.</param>
        /// <param name="utf8">True for UTF-8 (otherwise ASCII or Latin-1).</param>
        /// <param name="latin1">True for Latin-1.</param>
        /// <param name="tmp">64 bytes of scratch whose last 32 are zero.</param>
        /// <param name="posRef">The next row; advanced past the elements written.</param>
        /// <param name="maxLenRef">The longest element so far; raised by the elements written.</param>
        /// <returns>The number of elements written.</returns>
        private static long EncodeVStringRun(byte* pa, long sa, long count, byte* rowBase, int width, long vectorLimit, bool utf8,
            bool latin1, byte* tmp, ref long posRef, ref long maxLenRef)
        {
            long pos = posRef, maxLen = maxLenRef;
            // The lane numbers 0..15 (Vector128<T>.Indices is .NET 9+; this file builds for net8.0 too).
            var lanes = Vector128.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            const byte shortFlags = VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA;
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                if (pos >= vectorLimit)
                    break;
                byte flags = pa[15];
                if ((flags & VStringAllocator.NPY_STRING_FLAG_MASK) != shortFlags)
                    break;
                int size = flags & VStringAllocator.NPY_SHORT_STRING_SIZE_MASK;
                // The 16-byte cell holds the text from its first byte; the bytes past the size (a longer text written there
                // before, the flags byte) are masked away.
                var v = (Vector128.Load(pa) & Vector128.LessThan(lanes, Vector128.Create((byte)size))).ToVector256();
                uint high = v.ExtractMostSignificantBits();
                byte* row = rowBase + pos * width;
                if (utf8 || high == 0)
                {
                    // Its bytes ARE the encoding: one store writes them and the row's zero padding (spilling into rows C
                    // order writes later).
                    if (size > width)
                        break;
                    v.Store(row);
                    pos++;
                    if (size > maxLen)
                        maxLen = size;
                    continue;
                }
                if (!latin1)
                    break;
                int m = Utf8CellToLatin1Vector(v, high, size, row, width, tmp);
                if (m < 0)
                    break;
                pos++;
                if (m > maxLen)
                    maxLen = m;
            }
            posRef = pos;
            maxLenRef = maxLen;
            return k;
        }

        /// <summary>
        ///     One StringDType cell of the encode with exact writes (a cell the inner run refused, rows wider than 32 bytes, the
        ///     result's last rows, a host without AVX2): copied straight into its row under UTF-8 (an ASCII text under ASCII /
        ///     Latin-1) or decoded under Latin-1 when it fits, else decoded to code points and coded through the arena.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The cell.</param>
        /// <param name="utf8">True for UTF-8.</param>
        /// <param name="latin1">True for Latin-1.</param>
        /// <param name="asciiCopies">True for ASCII and Latin-1 (an ASCII text is its own encoding).</param>
        /// <returns>False for an NA cell (left to the general route), else true (coded).</returns>
        /// <exception cref="Exception">The codec's error.</exception>
        private static bool EncodeVStringElement(CodecWalk w, byte* pa, bool utf8, bool latin1, bool asciiCopies)
        {
            var rows = w.Rows;
            if (w.AReader.Allocator.load(pa, out byte* buf, out long size) != 0)
                return false;
            if (utf8 || (asciiCopies && new ReadOnlySpan<byte>(buf, (int)size).IndexOfAnyExceptInRange((byte)0, (byte)0x7F) < 0))
            {
                int width = rows.RowWidth;
                if (size <= width)
                {
                    byte* row = rows.RowBase + rows.Position * width;
                    Buffer.MemoryCopy(buf, row, size, size);
                    new Span<byte>(row + size, (int)(width - size)).Clear();
                    rows.CommitDirect(size);
                    return true;
                }
                // Wider than the rows: through the arena, which re-lays the rows out.
                rows.Arena.PutBytes(new ReadOnlySpan<byte>(buf, (int)size));
                rows.StoreRow(size);
                return true;
            }
            if (latin1)
            {
                int width = rows.RowWidth;
                byte* row = rows.RowBase + rows.Position * width;
                int m = Utf8CellToLatin1(buf, size, row, width);
                if (m >= 0)
                {
                    new Span<byte>(row + m, width - m).Clear();
                    rows.CommitDirect(m);
                    return true;
                }
                // A code point past U+00FF (the codec raises or runs the handler), or more code points than the row holds
                // (the codec's arena re-lays the rows out): what was written is rewritten by StoreRow.
            }
            var points = w.Points;
            points.Count = 0;
            AppendUtf8Points(buf, size, points);
            var arena = rows.Arena;
            long start = arena.Count;
            w.Codec.Encode((uint*)points.Data, points.Count, w.Errors, arena);
            rows.StoreRow(arena.Count - start);
            return true;
        }

        /// <summary>
        ///     The Latin-1 of a StringDType cell of at most 32 bytes whose code points are all below U+0100, without a branch per
        ///     byte: the cell holds valid UTF-8 (no validation needed), so no byte from <c>C4</c> means every lead is <c>C2</c> /
        ///     <c>C3</c>; the first byte of each code point comes from the continuation mask, and each one's Latin-1 byte is the
        ///     ASCII byte or the two-byte sequence's payload, selected with a mask. The row is zeroed by one 32-byte store first.
        /// </summary>
        /// <param name="v">The cell's bytes (zero past them).</param>
        /// <param name="high">The mask of its bytes past 0x7F (not zero: an ASCII cell is copied before).</param>
        /// <param name="size">Its length in bytes.</param>
        /// <param name="row">The row (a 32-byte store from it stays inside the result).</param>
        /// <param name="width">The row's width in bytes (at most 32).</param>
        /// <param name="tmp">64 bytes of scratch whose last 32 are zero.</param>
        /// <returns>The number of bytes written, or -1 for a code point past U+00FF or more code points than the row holds (the
        ///     row is then untouched).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Utf8CellToLatin1Vector(Vector256<byte> v, uint high, int size, byte* row, int width, byte* tmp)
        {
            if (Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xC4)) != Vector256<byte>.Zero)
                return -1;
            uint cont = high & ~Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xC0)).ExtractMostSignificantBits();
            ulong starts = ~(ulong)cont & ((1UL << size) - 1);
            int m = BitOperations.PopCount(starts);
            if (m > width)
                return -1;
            v.Store(tmp);
            Vector256<byte>.Zero.Store(row);
            int j = 0;
            while (starts != 0)
            {
                int i = BitOperations.TrailingZeroCount(starts);
                starts &= starts - 1;
                uint wd = *(ushort*)(tmp + i);
                uint b0 = wd & 0xFF;
                // All-ones for a C2 / C3 lead: (lead & 3) << 6 | (next & 3F), else the ASCII byte.
                uint two = (uint)((int)(0x7Fu - b0) >> 31);
                uint latin = ((wd & 0x03) << 6) | ((wd >> 8) & 0x3F);
                row[j++] = (byte)((b0 & ~two) | (latin & two));
            }
            return m;
        }

        /// <summary>
        ///     The Latin-1 of a StringDType cell whose code points are all below U+0100: its ASCII bytes as they are, each two-byte
        ///     sequence of a <c>C2</c> / <c>C3</c> lead as the byte it encodes (the cell holds valid UTF-8, so no other byte
        ///     below <c>C4</c> leads a sequence).
        /// </summary>
        /// <param name="src">The cell's UTF-8.</param>
        /// <param name="size">Its length in bytes.</param>
        /// <param name="dst">The row.</param>
        /// <param name="width">The row's width in bytes.</param>
        /// <returns>The number of bytes written, or -1 for a code point past U+00FF or a text longer than the row (the row then
        ///     holds part of the text, which the caller's codec route overwrites).</returns>
        private static int Utf8CellToLatin1(byte* src, long size, byte* dst, int width)
        {
            int m = 0;
            long i = 0;
            while (i < size)
            {
                if (m >= width)
                    return -1;
                byte b = src[i];
                if (b < 0x80)
                {
                    dst[m++] = b;
                    i++;
                    continue;
                }
                if (b > 0xC3)
                    return -1;
                dst[m++] = (byte)(((b & 0x03) << 6) | (src[i + 1] & 0x3F));
                i += 2;
            }
            return m;
        }

        // =============================================================================================================
        // S → U: reading an element
        // =============================================================================================================

        /// <summary>
        ///     Reads exactly <paramref name="count"/> (0..8) bytes as a little-endian 64-bit value (the bytes past them zero):
        ///     an element's 8-byte group without reading past the element.
        /// </summary>
        /// <param name="p">The bytes.</param>
        /// <param name="count">How many to read.</param>
        /// <returns>The value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ReadGroup(byte* p, int count)
        {
            switch (count)
            {
                case 8: return *(ulong*)p;
                case 7: return *(uint*)p | (ulong)*(ushort*)(p + 4) << 32 | (ulong)p[6] << 48;
                case 6: return *(uint*)p | (ulong)*(ushort*)(p + 4) << 32;
                case 5: return *(uint*)p | (ulong)p[4] << 32;
                case 4: return *(uint*)p;
                case 3: return *(ushort*)p | (ulong)p[2] << 16;
                case 2: return *(ushort*)p;
                case 1: return *p;
                default: return 0;
            }
        }

        /// <summary>
        ///     The bytes of an element of <paramref name="chars"/> (at most 32) bytes as one vector, zero past them, read
        ///     EXACTLY (whole 8-byte groups, then the last partial group byte by byte): never past the element.
        /// </summary>
        /// <param name="p">The element.</param>
        /// <param name="chars">Its width in bytes (1..32).</param>
        /// <returns>The vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> ReadElementExact(byte* p, int chars)
        {
            // The branches depend only on the width, the same for every element of a call: they are predicted.
            ulong x0, x1 = 0, x2 = 0, x3 = 0;
            if (chars < 8)
            {
                x0 = ReadGroup(p, chars);
            }
            else
            {
                x0 = *(ulong*)p;
                if (chars < 16)
                {
                    x1 = ReadGroup(p + 8, chars - 8);
                }
                else
                {
                    x1 = *(ulong*)(p + 8);
                    if (chars < 24)
                    {
                        x2 = ReadGroup(p + 16, chars - 16);
                    }
                    else
                    {
                        x2 = *(ulong*)(p + 16);
                        x3 = ReadGroup(p + 24, chars - 24);
                    }
                }
            }
            return Vector256.Create(x0, x1, x2, x3).AsByte();
        }

        /// <summary>
        ///     The mask of the lanes below <paramref name="chars"/> (all ones there, zero past them): what a 32-byte load of an
        ///     element keeps of its own bytes.
        /// </summary>
        /// <param name="chars">The element width in bytes (1..32).</param>
        /// <returns>The mask.</returns>
        private static Vector256<byte> LowLanes(int chars)
        {
            // The lane numbers 0..31 (Vector256<T>.Indices is .NET 9+; this file builds for net8.0 too).
            var lanes = Vector256.Create((byte)0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31);
            return Vector256.LessThan(lanes, Vector256.Create((byte)chars));
        }

        /// <summary>
        ///     The element at <paramref name="pa"/> as one vector of its bytes (zero past them): one 32-byte load masked to the
        ///     element when the load stays inside the chunk's elements (contiguous elements, <paramref name="pa"/> + 32 at or
        ///     before <paramref name="loadEnd"/>), the exact read otherwise.
        /// </summary>
        /// <param name="pa">The element.</param>
        /// <param name="chars">Its width in bytes (1..32).</param>
        /// <param name="loadEnd">The end of the chunk's contiguous elements (null when they are not contiguous).</param>
        /// <param name="keep">The mask of the element's own lanes (<see cref="LowLanes"/>).</param>
        /// <returns>The vector.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<byte> ReadElement(byte* pa, int chars, byte* loadEnd, Vector256<byte> keep)
            => pa + 32 <= loadEnd ? Vector256.Load(pa) & keep : ReadElementExact(pa, chars);

        // =============================================================================================================
        // S → U, UTF-8 / ASCII / Latin-1
        // =============================================================================================================

        /// <summary>
        ///     The UTF-8 / ASCII / Latin-1 decode run over <c>S</c> elements (see the file header): the vector inner run, and
        ///     each element it refuses on its own.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded (all of them, unless the codec raises).</returns>
        /// <exception cref="Exception">The codec's error for an element it codes through the arena.</exception>
        private static long FastDecodeWiden(CodecWalk w, byte* pa, long sa, long count)
        {
            var rows = w.Rows;
            bool latin1 = w.Codec.Kind == PythonCodecKind.Latin1;
            bool utf8 = w.Codec.Kind == PythonCodecKind.Utf8;
            // The handlers the run writes for a one-byte error (xmlcharrefreplace is not one of them: a decode error with it is
            // CPython's TypeError, the codec's to raise).
            int handler = w.Errors.Inline is NDErrorHandler.Replace or NDErrorHandler.Ignore or NDErrorHandler.BackslashReplace
                or NDErrorHandler.SurrogateEscape ? RunHandlerOf(w.Errors.Inline) : HandlerNone;
            int chars = w.AReader.Chars;
            bool vector = Avx2.IsSupported && chars <= 32;
            // The element's bytes for the UTF-8 decode, which reads a 32-bit word at any of them: 32 bytes, then zeros.
            byte* buf = stackalloc byte[64];
            Vector256<byte>.Zero.Store(buf + 32);
            // A 32-byte load from an element stays inside the chunk while it ends at or before the chunk's last element's end
            // (contiguous elements only; a reversed or padded stride takes the exact reads).
            byte* loadEnd = sa == chars ? pa + count * sa : null;
            // The vector-store limit depends only on the rows' width: computed again only when a re-layout changed it.
            int limitWidth = -1;
            long vectorLimit = 0;
            long k = 0;
            while (k < count)
            {
                int refusal = RunExact;
                int width = rows.RowWidth;
                if (vector && width <= 32)
                {
                    if (width != limitWidth)
                    {
                        // The row's stores, and at least 64 bytes: the two-byte UTF-8 path stores its second half at up to
                        // eight code points past the row's start.
                        vectorLimit = rows.VectorStoreLimit(Math.Max(((width + 7) >> 3) * 32, 64));
                        limitWidth = width;
                    }
                    long pos = rows.Position, maxLen = rows.MaxLength;
                    long done = DecodeWidenRun(pa, sa, count - k, chars, loadEnd, (uint*)rows.RowBase, width, vectorLimit, latin1,
                        utf8, handler, buf, ref pos, ref maxLen, out refusal);
                    rows.Position = pos;
                    rows.MaxLength = maxLen;
                    k += done;
                    pa += done * sa;
                    if (k == count)
                        break;
                }
                if (refusal == RunCodec)
                {
                    // Invalid UTF-8, or a byte past 0x7F under ASCII: the codec (its error or handler) - never an empty element.
                    DecodeThroughArena(w, pa, NDCharArray.TrimmedUnits(pa, chars, false));
                }
                else
                {
                    DecodeWidenElement(w, pa, latin1, utf8);
                }
                k++;
                pa += sa;
            }
            return count;
        }

        /// <summary>
        ///     The widen decode's inner run (no call in it): elements written straight into their rows with whole-vector stores
        ///     — ASCII (any bytes under Latin-1) widened, valid UTF-8 decoded — up to the first element it does not take (an
        ///     ASCII codec's byte past 0x7F, invalid UTF-8, an element longer than the rows) or the result's last rows.
        /// </summary>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <param name="chars">The element width in bytes (at most 32).</param>
        /// <param name="loadEnd">The end of the chunk's contiguous elements, or null (see <see cref="ReadElement"/>).</param>
        /// <param name="rowBase">The first row.</param>
        /// <param name="width">The rows' width in code points (at most 32).</param>
        /// <param name="vectorLimit">The first row the whole-vector stores could pass the result's end from.</param>
        /// <param name="latin1">True for Latin-1.</param>
        /// <param name="utf8">True for UTF-8.</param>
        /// <param name="handler">The decode handler the run writes (<see cref="RunHandlerOf"/>: replace, ignore,
        ///     backslashreplace, surrogateescape), or <see cref="HandlerNone"/>.</param>
        /// <param name="buf">64 bytes of scratch whose last 32 are zero (the UTF-8 decode's padded copy of the element).</param>
        /// <param name="posRef">The next row; advanced past the elements written.</param>
        /// <param name="maxLenRef">The longest element so far; raised by the elements written.</param>
        /// <param name="refusal">Why the run stopped before the chunk's end (<see cref="RunExact"/> / <see cref="RunCodec"/>).</param>
        /// <returns>The number of elements written.</returns>
        private static long DecodeWidenRun(byte* pa, long sa, long count, int chars, byte* loadEnd, uint* rowBase, int width,
            long vectorLimit, bool latin1, bool utf8, int handler, byte* buf, ref long posRef, ref long maxLenRef, out int refusal)
        {
            long pos = posRef, maxLen = maxLenRef;
            refusal = RunExact;
            // Stores per row: eight code points each, covering the row (the bytes of an element at most 32, so x0..x3 hold it).
            int stores = (width + 7) >> 3;
            var keep = LowLanes(chars);
            var zero = Vector256<byte>.Zero;
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                if (pos >= vectorLimit)
                    break;
                var v = ReadElement(pa, chars, loadEnd, keep);
                // The trimmed length (one past the last non-zero byte) and the bytes past 0x7F.
                uint nonZero = ~Vector256.Equals(v, zero).ExtractMostSignificantBits();
                int n = 32 - BitOperations.LeadingZeroCount(nonZero);
                uint high = v.ExtractMostSignificantBits();
                uint* row = rowBase + pos * width;
                if (high == 0 || latin1)
                {
                    if (n > width)
                        break;
                    // Each 8-byte group widened to eight code points (vpmovzxbd takes the low 8 bytes of a half; the next group
                    // is that half shifted down 8 bytes): the element's units, then zeros as the row's padding (and zeros spilled
                    // into rows C order writes later).
                    var lo = v.GetLower();
                    Avx2.ConvertToVector256Int32(lo).AsUInt32().Store(row);
                    if (stores > 1)
                        Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo, 8)).AsUInt32().Store(row + 8);
                    if (stores > 2)
                    {
                        var hi = v.GetUpper();
                        Avx2.ConvertToVector256Int32(hi).AsUInt32().Store(row + 16);
                        if (stores > 3)
                            Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi, 8)).AsUInt32().Store(row + 24);
                    }
                    pos++;
                    if (n > maxLen)
                        maxLen = n;
                    continue;
                }
                int m;
                if (utf8)
                {
                    m = DecodeUtf8Vector(v, high, n, buf, row, width, stores);
                    if (m == -2)
                        break;   // valid, but more code points than the row holds: the exact path
                    // Invalid UTF-8 is the run's to write only with a handler and no two adjacent bytes past 0x7F: no valid
                    // sequence can start then, and CPython's error range is exactly each such byte (an invalid start byte, a lead
                    // before a byte that does not continue it: end = start + 1, or a lead that ends the element).
                    if (m == -1 && (handler == HandlerNone || (high & (high >> 1)) != 0))
                    {
                        refusal = RunCodec;
                        break;
                    }
                }
                else
                {
                    // ASCII: every byte past 0x7F is its own error (PyUnicode_DecodeASCII calls the handler per byte).
                    m = -1;
                    if (handler == HandlerNone)
                    {
                        refusal = RunCodec;
                        break;
                    }
                }
                if (m == -1)
                {
                    m = DecodeOneByteErrors(v, high, n, handler, buf, row, width, stores);
                    if (m < 0)
                        break;   // more code points than the row holds: the exact path (through the codec)
                }
                pos++;
                if (m > maxLen)
                    maxLen = m;
            }
            posRef = pos;
            maxLenRef = maxLen;
            return k;
        }

        /// <summary>
        ///     Decodes an element of up to 32 bytes holding a byte past 0x7F straight into its row when it is valid UTF-8 —
        ///     CPython's strict decoder's verdict, checked on byte-class masks: every continuation byte is exactly where a lead
        ///     byte needs one, no <c>C0</c> / <c>C1</c> / <c>F5..FF</c>, no overlong three- / four-byte form (<c>E0</c> before
        ///     <c>A0</c>, <c>F0</c> before <c>90</c>), no surrogate (<c>ED</c> before <c>A0..BF</c>), nothing past U+10FFFF
        ///     (<c>F4</c> before <c>90..BF</c>). The row is zeroed by whole-vector stores, then each code point is computed from
        ///     the 32-bit word at its first byte without a branch.
        /// </summary>
        /// <param name="v">The element's bytes (zero past them).</param>
        /// <param name="high">The mask of its bytes past 0x7F (not zero).</param>
        /// <param name="n">Its trimmed length.</param>
        /// <param name="buf">64 bytes of scratch whose last 32 are zero.</param>
        /// <param name="row">The row (room for <paramref name="stores"/> vectors).</param>
        /// <param name="width">The row's width in code points.</param>
        /// <param name="stores">The vectors covering the row.</param>
        /// <returns>The number of code points written; -1 when the bytes are not valid UTF-8, -2 when they decode to more code
        ///     points than the row holds (the row is then untouched).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int DecodeUtf8Vector(Vector256<byte> v, uint high, int n, byte* buf, uint* row, int width, int stores)
        {
            uint geC0 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xC0)).ExtractMostSignificantBits();
            uint geC2 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xC2)).ExtractMostSignificantBits();
            uint geE0 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xE0)).ExtractMostSignificantBits();
            uint cont = high & ~geC0;               // 80..BF
            uint bad = geC0 & ~geC2;                // C0, C1: always overlong
            ulong expected;                         // where the lead bytes need their continuation bytes
            bool all3 = false;                      // every code point three bytes long (BMP text past U+07FF, e.g. CJK)
            if (geE0 == 0)
            {
                // Only two-byte sequences: each lead needs the next byte.
                expected = (ulong)geC2 << 1;
            }
            else
            {
                uint geF0 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xF0)).ExtractMostSignificantBits();
                uint geF5 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xF5)).ExtractMostSignificantBits();
                uint lead2 = geC2 & ~geE0, lead3 = geE0 & ~geF0, lead4 = geF0 & ~geF5;
                // No ASCII byte, no two- or four-byte lead, at most eight code points: the vector decode below (the structure
                // check that follows still has to pass).
                all3 = lead2 == 0 && lead4 == 0 && high == (uint)((1UL << n) - 1) && n <= 24;
                bad |= geF5;
                expected = ((ulong)lead2 << 1) | ((ulong)lead3 << 1) | ((ulong)lead3 << 2)
                           | ((ulong)lead4 << 1) | ((ulong)lead4 << 2) | ((ulong)lead4 << 3);
                // The second byte's range for E0 / ED / F0 / F4 (shifting a lead's bit to its second byte's position).
                uint geA0 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0xA0)).ExtractMostSignificantBits();
                uint ge90 = Vector256.GreaterThanOrEqual(v, Vector256.Create((byte)0x90)).ExtractMostSignificantBits();
                uint e0 = Vector256.Equals(v, Vector256.Create((byte)0xE0)).ExtractMostSignificantBits();
                uint ed = Vector256.Equals(v, Vector256.Create((byte)0xED)).ExtractMostSignificantBits();
                uint f0 = Vector256.Equals(v, Vector256.Create((byte)0xF0)).ExtractMostSignificantBits();
                uint f4 = Vector256.Equals(v, Vector256.Create((byte)0xF4)).ExtractMostSignificantBits();
                bad |= ((e0 << 1) & ~geA0) | ((ed << 1) & geA0) | ((f0 << 1) & ~ge90) | ((f4 << 1) & ge90);
            }
            // A continuation byte nowhere a lead needs one, a lead whose continuation is missing (or past the element: the
            // expected bits past 31 match no byte), an invalid lead or second byte: not valid UTF-8.
            if (bad != 0 || expected != cont)
                return -1;
            // The first byte of every code point.
            ulong starts = ~(ulong)cont & ((1UL << n) - 1);
            int m = BitOperations.PopCount(starts);
            if (m > width)
                return -2;
            var z = Vector256<uint>.Zero;
            if (all3)
            {
                // Code points 0..3 from bytes 0..11 (the low half), 4..7 from bytes 12..27 (vpalignr of the two halves, in
                // registers): vpshufb gathers each code point's three bytes into a 32-bit lane (b0 | b1 << 8 | b2 << 16), then
                // the payload bits are combined per lane. Lanes past the element gather zero bytes: code point 0.
                var lower = v.GetLower();
                var gathered = Avx2.Shuffle(Vector256.Create(lower, Ssse3.AlignRight(v.GetUpper(), lower, 12)), Vector256.Create(
                    (byte)0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
                    0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80)).AsUInt32();
                var cp = ((gathered & Vector256.Create(0x0Fu)) << 12) | ((gathered & Vector256.Create(0x3F00u)) >> 2)
                         | ((gathered >> 16) & Vector256.Create(0x3Fu));
                cp.Store(row);
                if (stores > 1)
                    z.Store(row + 8);
                if (stores > 2)
                    z.Store(row + 16);
                if (stores > 3)
                    z.Store(row + 24);
                return m;
            }
            if (geE0 == 0 && n <= 16)
            {
                // Only one- and two-byte code points in at most 16 bytes: each byte position's value computed in a 16-bit lane
                // (a two-byte lead's payload with the next byte's, else the byte itself; continuation positions get junk the
                // compaction drops), then each half's code-point starts gathered to its front by a vpshufb from the compact
                // table, widened to 32 bits, and stored — the high half right after the low half's code points. The row is
                // zeroed first (the high half's store covers at most 16 code points from the row's start; the caller's limit
                // allows 64 bytes).
                var lower = v.GetLower();
                var bytes = Avx2.ConvertToVector256Int16(lower).AsUInt16();
                var next = Avx2.ConvertToVector256Int16(Sse2.ShiftRightLogical128BitLane(lower, 1)).AsUInt16();
                var isLead = Vector256.GreaterThanOrEqual(bytes, Vector256.Create((ushort)0xC0));
                var pair = ((bytes & Vector256.Create((ushort)0x1F)) << 6) | (next & Vector256.Create((ushort)0x3F));
                var values = Vector256.ConditionalSelect(isLead, pair, bytes);
                uint startBits = (uint)starts;
                var lowStarts = Ssse3.Shuffle(values.GetLower().AsByte(), Utf8Shuffles.Compact(startBits & 0xFF)).AsUInt16();
                var highStarts = Ssse3.Shuffle(values.GetUpper().AsByte(), Utf8Shuffles.Compact(startBits >> 8)).AsUInt16();
                z.Store(row);
                if (stores > 1)
                    z.Store(row + 8);
                if (stores > 2)
                    z.Store(row + 16);
                if (stores > 3)
                    z.Store(row + 24);
                Avx2.ConvertToVector256Int32(lowStarts).AsUInt32().Store(row);
                Avx2.ConvertToVector256Int32(highStarts).AsUInt32().Store(row + BitOperations.PopCount(startBits & 0xFF));
                return m;
            }
            // The per-code-point decode reads 16- / 32-bit words at any byte: from the copy, whose zero tail pads the last.
            v.Store(buf);
            z.Store(row);
            if (stores > 1)
                z.Store(row + 8);
            if (stores > 2)
                z.Store(row + 16);
            if (stores > 3)
                z.Store(row + 24);
            int j = 0;
            if (geE0 == 0)
            {
                while (starts != 0)
                {
                    int i = BitOperations.TrailingZeroCount(starts);
                    starts &= starts - 1;
                    uint wd = *(ushort*)(buf + i);
                    uint b0 = wd & 0xFF;
                    // All-ones for a two-byte lead: its payload with the next byte's, else the ASCII byte.
                    uint two = (uint)((int)(0x7Fu - b0) >> 31);
                    uint cp2 = ((wd & 0x1F) << 6) | ((wd >> 8) & 0x3F);
                    row[j++] = (b0 & ~two) | (cp2 & two);
                }
            }
            else
            {
                while (starts != 0)
                {
                    int i = BitOperations.TrailingZeroCount(starts);
                    starts &= starts - 1;
                    uint wd = *(uint*)(buf + i);
                    int b0 = (int)(wd & 0xFF);
                    // The sequence length from the lead byte (1 + one for each of C0 / E0 / F0 it reaches), then the payload as
                    // if four bytes long, shifted right by six bits per byte it does not have (the next code point's bytes it
                    // read fall out with them).
                    int len = 1 - ((0xBF - b0) >> 31) - ((0xDF - b0) >> 31) - ((0xEF - b0) >> 31);
                    // The lead byte's payload bits: 7 for ASCII, 7 - len for a lead of len bytes (len ones, then a zero).
                    uint leadMask = len == 1 ? 0x7Fu : (1u << (7 - len)) - 1;
                    uint cp4 = ((wd & leadMask) << 18) | ((wd & 0x3F00u) << 4) | ((wd >> 10) & 0xFC0u) | ((wd >> 24) & 0x3Fu);
                    row[j++] = cp4 >> (24 - 6 * len);
                }
            }
            return m;
        }

        /// <summary>
        ///     Decodes an element whose every byte past 0x7F is a one-byte error under a run handler (CPython's inline
        ///     <c>replace</c> / <c>surrogateescape</c> / <c>ignore</c> of the UTF-8 and ASCII decoders, the generic
        ///     <c>backslashreplace</c>): each such byte becomes U+FFFD, U+DC00 + the byte, nothing, or <c>\xNN</c> (four code
        ///     points, lowercase hex); every other byte is its ASCII code point.
        /// </summary>
        /// <param name="v">The element's bytes (zero past them).</param>
        /// <param name="high">The mask of its bytes past 0x7F (not zero).</param>
        /// <param name="n">Its trimmed length.</param>
        /// <param name="handler">The run handler (replace, ignore, backslashreplace or surrogateescape).</param>
        /// <param name="buf">64 bytes of scratch whose last 32 are zero.</param>
        /// <param name="row">The row (room for <paramref name="stores"/> vectors).</param>
        /// <param name="width">The row's width in code points.</param>
        /// <param name="stores">The vectors covering the row.</param>
        /// <returns>The number of code points written, or -1 when they would not fit the row (the row is then untouched).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int DecodeOneByteErrors(Vector256<byte> v, uint high, int n, int handler, byte* buf, uint* row, int width,
            int stores)
        {
            if (handler is HandlerReplace or HandlerSurrogateEscape)
            {
                // One code point per byte: the bytes widened eight at a time, the lanes past 0x7F replaced by U+FFFD, or by
                // U+DC00 + the byte (the zero lanes past the element stay zero).
                if (n > width)
                    return -1;
                var at80 = Vector256.Create(0x80u);
                var lo = v.GetLower();
                var hi = v.GetUpper();
                StoreDecodedErrors(Avx2.ConvertToVector256Int32(lo).AsUInt32(), at80, handler, row);
                if (stores > 1)
                    StoreDecodedErrors(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo, 8)).AsUInt32(), at80, handler, row + 8);
                if (stores > 2)
                    StoreDecodedErrors(Avx2.ConvertToVector256Int32(hi).AsUInt32(), at80, handler, row + 16);
                if (stores > 3)
                    StoreDecodedErrors(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi, 8)).AsUInt32(), at80, handler, row + 24);
                return n;
            }
            int bad = BitOperations.PopCount(high);
            int m = handler == HandlerIgnore ? n - bad : n + 3 * bad;
            if (m > width)
                return -1;
            v.Store(buf);
            var z = Vector256<uint>.Zero;
            z.Store(row);
            if (stores > 1)
                z.Store(row + 8);
            if (stores > 2)
                z.Store(row + 16);
            if (stores > 3)
                z.Store(row + 24);
            int c = 0;
            if (handler == HandlerIgnore)
            {
                // Every byte written, the position advanced past the ASCII ones only (no branch per byte); the last write past
                // the text is cleared (it lies inside the stores' reach).
                for (int i = 0; i < n; i++)
                {
                    uint b = buf[i];
                    row[c] = b;
                    c += (int)((b >> 7) ^ 1);
                }
                row[c] = 0;
                return m;
            }
            for (int i = 0; i < n; i++)
            {
                uint b = buf[i];
                if (b < 0x80)
                {
                    row[c++] = b;
                    continue;
                }
                // \xNN with lowercase hex digits ('0'..'9', then 'a'..'f': 'a' - '0' - 10 = 39).
                uint h1 = b >> 4, h0 = b & 0xF;
                row[c] = '\\';
                row[c + 1] = 'x';
                row[c + 2] = '0' + h1 + (uint)((9 - (int)h1) >> 31 & 39);
                row[c + 3] = '0' + h0 + (uint)((9 - (int)h0) >> 31 & 39);
                c += 4;
            }
            return m;
        }

        /// <summary>Stores eight decoded code points with the lanes past 0x7F replaced: U+FFFD for <c>replace</c>, U+DC00 + the
        ///     byte for <c>surrogateescape</c>.</summary>
        /// <param name="values">The bytes widened to code points.</param>
        /// <param name="at80">0x80 in every lane.</param>
        /// <param name="handler">Replace or surrogateescape.</param>
        /// <param name="dst">Where to store the eight code points.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StoreDecodedErrors(Vector256<uint> values, Vector256<uint> at80, int handler, uint* dst)
        {
            var replacement = handler == HandlerReplace ? Vector256.Create(0xFFFDu) : values | Vector256.Create(0xDC00u);
            Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(values, at80), replacement, values).Store(dst);
        }

        /// <summary>
        ///     One <c>S</c> element of the widen decode with exact writes (the result's last rows, an element wider than 32
        ///     bytes or rows wider than 32 code points, a host without AVX2, an element the inner run refused): widened or
        ///     decoded straight into its row when it fits, else through the codec into the arena.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The element.</param>
        /// <param name="latin1">True for Latin-1.</param>
        /// <param name="utf8">True for UTF-8.</param>
        /// <exception cref="Exception">The codec's error.</exception>
        private static void DecodeWidenElement(CodecWalk w, byte* pa, bool latin1, bool utf8)
        {
            var rows = w.Rows;
            int n = NDCharArray.TrimmedUnits(pa, w.AReader.Chars, false);
            int width = rows.RowWidth;
            uint* row = (uint*)rows.RowBase + rows.Position * width;
            byte acc = 0;
            for (int i = 0; i < n; i++)
                acc |= pa[i];
            if ((latin1 || acc < 0x80) && n <= width)
            {
                for (int i = 0; i < n; i++)
                    row[i] = pa[i];
                new Span<uint>(row + n, width - n).Clear();
                rows.CommitDirect(n);
                return;
            }
            if (utf8 && acc >= 0x80 && n <= width)
            {
                // Decoding writes at most one code point per byte, so a row as wide as the bytes holds it.
                int m = DecodeUtf8Valid(pa, n, row);
                if (m >= 0)
                {
                    new Span<uint>(row + m, width - m).Clear();
                    rows.CommitDirect(m);
                    return;
                }
            }
            if (n == 0)
                rows.StoreRow(0);
            else
                DecodeThroughArena(w, pa, n);
        }

        /// <summary>
        ///     The code points of VALID UTF-8 (CPython's utf8_decode without its error paths): -1 at the first sequence it
        ///     would report (so the codec's error handling takes the element), else the number written.
        /// </summary>
        /// <param name="s">The bytes.</param>
        /// <param name="size">Their number.</param>
        /// <param name="dst">The destination (room for one code point per byte).</param>
        /// <returns>The number of code points, or -1 for invalid UTF-8.</returns>
        private static int DecodeUtf8Valid(byte* s, int size, uint* dst)
        {
            int i = 0, n = 0;
            while (i < size)
            {
                uint ch = s[i];
                if (ch < 0x80)
                {
                    dst[n++] = ch;
                    i++;
                    continue;
                }
                if (ch < 0xC2 || ch > 0xF4)
                    return -1;
                if (ch < 0xE0)
                {
                    if (size - i < 2)
                        return -1;
                    uint c2 = s[i + 1];
                    if (c2 - 0x80u >= 0x40u)
                        return -1;
                    dst[n++] = ((ch & 0x1F) << 6) | (c2 & 0x3F);
                    i += 2;
                    continue;
                }
                if (ch < 0xF0)
                {
                    if (size - i < 3)
                        return -1;
                    uint c2 = s[i + 1], c3 = s[i + 2];
                    if (c2 - 0x80u >= 0x40u || c3 - 0x80u >= 0x40u || (ch == 0xE0 && c2 < 0xA0) || (ch == 0xED && c2 >= 0xA0))
                        return -1;
                    dst[n++] = ((ch & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
                    i += 3;
                    continue;
                }
                {
                    if (size - i < 4)
                        return -1;
                    uint c2 = s[i + 1], c3 = s[i + 2], c4 = s[i + 3];
                    if (c2 - 0x80u >= 0x40u || c3 - 0x80u >= 0x40u || c4 - 0x80u >= 0x40u || (ch == 0xF0 && c2 < 0x90)
                        || (ch == 0xF4 && c2 >= 0x90))
                        return -1;
                    dst[n++] = ((ch & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F);
                    i += 4;
                }
            }
            return n;
        }

        // =============================================================================================================
        // S → U, UTF-16 / UTF-32
        // =============================================================================================================

        /// <summary>
        ///     The UTF-16 / UTF-32 decode run over <c>S</c> elements (see the file header): the vector inner run, and each
        ///     element it refuses through the codec (an empty one is an empty row).
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded (all of them, unless the codec raises).</returns>
        /// <exception cref="Exception">The codec's error for an element it codes through the arena.</exception>
        private static long FastDecodeWide(CodecWalk w, byte* pa, long sa, long count)
        {
            var rows = w.Rows;
            var kind = w.Codec.Kind;
            bool utf16 = kind is PythonCodecKind.Utf16 or PythonCodecKind.Utf16LE or PythonCodecKind.Utf16BE;
            int byteorder = w.Codec.ByteOrder;
            int chars = w.AReader.Chars;
            bool vector = Avx2.IsSupported && chars <= 32;
            // The element's bytes, then zeros: the units are loaded from it after the BOM (a 32-byte load from byte 2 or 4).
            byte* buf = stackalloc byte[64];
            Vector256<byte>.Zero.Store(buf + 32);
            byte* loadEnd = sa == chars ? pa + count * sa : null;
            // The vector-store limit depends only on the rows' width: computed again only when a re-layout changed it.
            int limitWidth = -1;
            long vectorLimit = 0;
            long k = 0;
            while (k < count)
            {
                int width = rows.RowWidth;
                if (vector && width <= 32)
                {
                    if (width != limitWidth)
                    {
                        vectorLimit = rows.VectorStoreLimit(((width + 7) >> 3) * 32);
                        limitWidth = width;
                    }
                    long pos = rows.Position, maxLen = rows.MaxLength;
                    long done = DecodeWideRun(pa, sa, count - k, chars, loadEnd, (uint*)rows.RowBase, width, vectorLimit, utf16,
                        byteorder, buf, ref pos, ref maxLen);
                    rows.Position = pos;
                    rows.MaxLength = maxLen;
                    k += done;
                    pa += done * sa;
                    if (k == count)
                        break;
                }
                int n = NDCharArray.TrimmedUnits(pa, chars, false);
                if (n == 0)
                    rows.StoreRow(0);
                else
                    DecodeThroughArena(w, pa, n);
                k++;
                pa += sa;
            }
            return count;
        }

        /// <summary>
        ///     The UTF-16 / UTF-32 decode's inner run (no call in it): elements of whole units without a surrogate (UTF-32:
        ///     without a unit past U+10FFFF either) written straight into their rows — the BOM of the BOM codecs read (it picks
        ///     the order) and skipped, the units byte-swapped for big-endian and widened — up to the first element it does not
        ///     take (a truncated unit, a surrogate, an invalid unit, more units than the rows hold) or the result's last rows.
        /// </summary>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <param name="chars">The element width in bytes (at most 32).</param>
        /// <param name="loadEnd">The end of the chunk's contiguous elements, or null (see <see cref="ReadElement"/>).</param>
        /// <param name="rowBase">The first row.</param>
        /// <param name="width">The rows' width in code points (at most 32).</param>
        /// <param name="vectorLimit">The first row the whole-vector stores could pass the result's end from.</param>
        /// <param name="utf16">True for UTF-16, false for UTF-32.</param>
        /// <param name="byteorder">0 (read a BOM; native little-endian without one), -1 little-, 1 big-endian.</param>
        /// <param name="buf">64 bytes of scratch whose last 32 are zero.</param>
        /// <param name="posRef">The next row; advanced past the elements written.</param>
        /// <param name="maxLenRef">The longest element so far; raised by the elements written.</param>
        /// <returns>The number of elements written.</returns>
        private static long DecodeWideRun(byte* pa, long sa, long count, int chars, byte* loadEnd, uint* rowBase, int width,
            long vectorLimit, bool utf16, int byteorder, byte* buf, ref long posRef, ref long maxLenRef)
        {
            long pos = posRef, maxLen = maxLenRef;
            int stores = (width + 7) >> 3;
            var keep = LowLanes(chars);
            var zero = Vector256<byte>.Zero;
            var swap16 = Vector256.Create((byte)1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
            var swap32 = Vector256.Create((byte)3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            int unitBytes = utf16 ? 2 : 4;
            long k = 0;
            for (; k < count; k++, pa += sa)
            {
                if (pos >= vectorLimit)
                    break;
                var v = ReadElement(pa, chars, loadEnd, keep);
                uint nonZero = ~Vector256.Equals(v, zero).ExtractMostSignificantBits();
                int n = 32 - BitOperations.LeadingZeroCount(nonZero);
                // A trailing partial unit is CPython's "truncated data" (the codec raises or runs the handler).
                if ((n & (unitBytes - 1)) != 0)
                    break;
                bool big = byteorder > 0;
                int q = 0;
                if (byteorder == 0 && n >= unitBytes)
                {
                    // The BOM codec: a leading U+FEFF read little-endian picks little-endian, its swap big-endian; either is
                    // skipped. Without one the units are native little-endian.
                    uint first = utf16 ? v.AsUInt16().ToScalar() : v.AsUInt32().ToScalar();
                    if (first == 0xFEFF)
                    {
                        q = unitBytes;
                    }
                    else if (first == (utf16 ? 0xFFFEu : 0xFFFE0000u))
                    {
                        q = unitBytes;
                        big = true;
                    }
                }
                int units = (n - q) / unitBytes;
                if (units > width)
                    break;
                // The units from after the BOM (the copy's zero tail makes the lanes past the element zero code points).
                v.Store(buf);
                var u = Vector256.Load(buf + q);
                if (big)
                    u = Avx2.Shuffle(u, utf16 ? swap16 : swap32);
                uint* row = rowBase + pos * width;
                if (utf16)
                {
                    var u16 = u.AsUInt16();
                    // Any surrogate unit (a pair or a lone one): the codec decodes or reports it.
                    if (Vector256.Equals(u16 & Vector256.Create((ushort)0xF800), Vector256.Create((ushort)0xD800)) != Vector256<ushort>.Zero)
                        break;
                    Avx2.ConvertToVector256Int32(u16.GetLower()).AsUInt32().Store(row);
                    if (stores > 1)
                        Avx2.ConvertToVector256Int32(u16.GetUpper()).AsUInt32().Store(row + 8);
                    if (stores > 2)
                        Vector256<uint>.Zero.Store(row + 16);
                    if (stores > 3)
                        Vector256<uint>.Zero.Store(row + 24);
                }
                else
                {
                    var u32 = u.AsUInt32();
                    // A surrogate or a unit past U+10FFFF: the codec reports it (unsigned arithmetic for the surrogate range).
                    var invalid = Vector256.LessThan(u32 - Vector256.Create(0xD800u), Vector256.Create(0x800u))
                                  | Vector256.GreaterThan(u32, Vector256.Create(0x10FFFFu));
                    if (invalid != Vector256<uint>.Zero)
                        break;
                    u32.Store(row);
                    var z = Vector256<uint>.Zero;
                    if (stores > 1)
                        z.Store(row + 8);
                    if (stores > 2)
                        z.Store(row + 16);
                    if (stores > 3)
                        z.Store(row + 24);
                }
                pos++;
                if (units > maxLen)
                    maxLen = units;
            }
            posRef = pos;
            maxLenRef = maxLen;
            return k;
        }

        /// <summary>
        ///     The decode run over <c>S</c> elements for a codec without a vector route (utf-8-sig, the code pages,
        ///     <c>undefined</c>): an empty element is an empty row (no codec, as <c>PyUnicode_FromEncodedObject</c> returns ''
        ///     first), every other through the codec into the arena and its row.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="pa">The first element.</param>
        /// <param name="sa">The element stride.</param>
        /// <param name="count">The number of elements left.</param>
        /// <returns>The number of elements coded (all of them, unless the codec raises).</returns>
        /// <exception cref="Exception">The codec's error for an element.</exception>
        private static long FastDecodeBytes(CodecWalk w, byte* pa, long sa, long count)
        {
            var rows = w.Rows;
            int chars = w.AReader.Chars;
            for (long k = 0; k < count; k++, pa += sa)
            {
                int n = NDCharArray.TrimmedUnits(pa, chars, false);
                if (n == 0)
                    rows.StoreRow(0);
                else
                    DecodeThroughArena(w, pa, n);
            }
            return count;
        }
    }
}
