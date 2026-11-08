using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Printing;

// =============================================================================
// NDCharObjectPath.ModFast.cs — np.strings.mod's fast paths: a format that takes ONE argument by position, formatted
// straight into the result (strings phase 2, P2.6 milestone 3)
// =============================================================================
//
// WHEN THEY RUN
//   ModFormat.FastSpec marks a format with exactly one complete spec, no %(key), no '*' width or precision (np.strings.mod's
//   common call: one format such as '%d', '%.2f' or 'id_%04d' over an array of numbers or texts). Such a format consumes a
//   non-tuple value whole, whatever the value's type, so CPython's argument state (getnextarg, the mapping test, the
//   "not all arguments converted" check) cannot raise: the element is the format's literal runs around ONE conversion.
//   * The CHUNK path (TryModFastChunk) runs when the format does not move within an NDIter chunk (one format broadcast over
//     the values) and the values are numbers or plain texts: the literal runs are rendered once, and each element is read
//     straight from its typed pointer. Once a U / S result's width is chosen (ModWalkState's first RowSample elements) an
//     element is written STRAIGHT INTO ITS ROW — no arena, no copy — while its longest possible length stays inside the
//     result; the sampled elements, a StringDType result (packed per element) and the last rows go through the arena. An
//     element this path cannot write (NaN under %d, digits out of FloatDigits' range) is formatted by the general element
//     routine instead — in its place, so C order and the first error are kept.
//   * The ELEMENT path (TryModFastElement) serves the same formats element by element (a per-element format array, a value
//     kind the chunk path leaves out): literal runs appended, the conversion specialized for numbers, the general
//     conversion routine (ConvertStr / ConvertBytes) for everything else.
//   A Python object value keeps the general path (a tuple's items are the arguments; a dict feeds keys), and so do a
//   conversion character the formatter refuses (its error) and a width or precision past FastFieldLimit (the general
//   path's allocation behavior).
//
// WHAT IS SPECIALIZED (each byte-identical to PrintfFormatter's CPython conversions, which the strmod / tmod oracle gates)
//   * integer values (int8 … uint64):  d i u (decimal), x X o (with '#'s prefix), s (str), r / a (np.int64(…)),
//                                      e E f F g G (float(v));
//   * float values (float16/32/64):    d i u (int(v), truncated; NaN / inf / |v| ≥ 2^63 take the general route, which raises
//                                      or uses a Python int), s / r / a (NumPy's scalar str / repr: shortest digits in the
//                                      value's own precision), e E f F g G (FloatDigits' exact digits; a value outside their
//                                      range takes the general route);
//   * text values:                     %s of a U value (str formatting) and %s / %b of an S value (bytes formatting), the
//                                      text cut to the precision and padded to the width; any other conversion of a text
//                                      takes the general routine.
//   Bytes formatting (an S format) writes the same ASCII as bytes; its %s / %b of a number is the raw buffer and %c a byte
//   (the general routines), and 'i' reads as 'd' as _PyBytes_FormatEx makes it.
//
// WHY
//   The general path builds a .NET string per conversion (PrintfFormatter boxes the scalar, formats through BigInteger and
//   .NET's double formatter — 100–200 ns for a double —, then pads by concatenation) and walks CPython's argument state per
//   spec: 63–450 ns per element against NumPy's 360–1,280. A field here is digits written straight into the arena.
//   The writers are generic over the arena's unit (uint code points for str formatting, byte for bytes formatting); the
//   JIT compiles one copy per unit type.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static unsafe partial class NDCharObjectPath
    {
        /// <summary>
        ///     The widest width or precision the fast paths write themselves. Larger fields take the general path, which
        ///     reproduces CPython's allocation behavior for huge fields (an <see cref="OutOfMemoryException"/> past .NET's
        ///     longest string); 4096 is far above any field a real format asks for.
        /// </summary>
        private const int FastFieldLimit = 4096;

        /// <summary>
        ///     Room for the longest body a fast conversion builds — <c>%f</c> of a value below 2^64 at precision 19 (20 + 1 +
        ///     19 characters), <c>%r</c> of a float64 (<c>np.float64(</c> + 24 + <c>)</c>), an octal uint64 (22 digits) —
        ///     with <see cref="NDTextFormat.MaxTextLength"/> to spare after a repr prefix, the room
        ///     <see cref="NDTextFormat.FormatFloatAscii"/> asks for. Stack memory, not zeroed (<c>[SkipLocalsInit]</c>).
        /// </summary>
        private const int FastBodyCapacity = 96;

        /// <summary>The longest literal text (units) the chunk path renders on the stack; a longer format runs element by
        ///     element.</summary>
        private const int FastLiteralLimit = 1024;

        /// <summary>
        ///     Test hook: true sends every element of <c>np.strings.mod</c> on THIS thread through CPython's argument machinery
        ///     and the general conversion routines (no chunk path, no element fast path), so a test can hold the fast paths to
        ///     the route the strmod / tmod oracle gates on the same inputs. Per thread (the test suite runs in parallel).
        /// </summary>
        [ThreadStatic]
        internal static bool ModFastPathsDisabled;

        /// <summary>The value classes the chunk path writes itself.</summary>
        private enum FastValueClass : byte
        {
            /// <summary>Not served (the per-element loop runs).</summary>
            None,

            /// <summary>An integer dtype (int8 … uint64).</summary>
            Integer,

            /// <summary>A float dtype (float16 / float32 / float64).</summary>
            Float,

            /// <summary><c>%s</c> of a <c>U</c> value under str formatting.</summary>
            UnicodeText,

            /// <summary><c>%s</c> / <c>%b</c> of an <c>S</c> value under bytes formatting.</summary>
            BytesText,
        }

        // =============================================================================================================
        // The chunk path
        // =============================================================================================================

        /// <summary>
        ///     Formats a whole NDIter chunk whose format does not move (see the file header): returns false, with nothing
        ///     written, when the chunk's value kind or conversion is not served (the caller's per-element loop runs).
        /// </summary>
        /// <param name="program">The chunk's compiled format (its <see cref="ModFormat.FastSpec"/> is set).</param>
        /// <param name="format">The chunk's format element (a text: the caller checked str formatting's descriptor).</param>
        /// <param name="pv">The first value element.</param>
        /// <param name="sv">The values' byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="w">The call's state.</param>
        /// <returns>True when the chunk was formatted.</returns>
        /// <exception cref="Exception">The first element's CPython error, in C order.</exception>
        private static bool TryModFastChunk(ModFormat program, in Elem format, byte* pv, long sv, long n, ModWalkState w)
        {
            ref readonly var spec = ref program.Ops[program.FastSpec];
            int conv = spec.Conversion;
            if (w.Bytes ? !IsBytesConversion(conv) : !IsStrConversion(conv))
                return false;
            if (spec.Width > FastFieldLimit || spec.Precision > FastFieldLimit || program.UnitCount > FastLiteralLimit)
                return false;
            if (w.Bytes && conv == 'i')
                conv = 'd';   // _PyBytes_FormatEx converts 'i' as 'd'
            var reader = w.Readers[1];
            var cls = ChunkClass(reader, conv, w.Bytes);
            if (cls == FastValueClass.None)
                return false;
            return w.Bytes
                ? FastChunk<byte>(program, in format, in spec, conv, cls, reader, pv, sv, n, w)
                : FastChunk<uint>(program, in format, in spec, conv, cls, reader, pv, sv, n, w);
        }

        /// <summary>The value class the chunk path serves for a values operand and a conversion, or
        ///     <see cref="FastValueClass.None"/>.</summary>
        /// <param name="reader">The values operand.</param>
        /// <param name="conv">The conversion character ('i' already 'd' for bytes formatting).</param>
        /// <param name="bytes">Bytes formatting.</param>
        /// <returns>The class.</returns>
        private static FastValueClass ChunkClass(Operand reader, int conv, bool bytes)
        {
            switch (reader.Kind)
            {
                case OperandKind.Number:
                    switch (reader.Number)
                    {
                        case NPTypeCode.SByte:
                        case NPTypeCode.Byte:
                        case NPTypeCode.Int16:
                        case NPTypeCode.UInt16:
                        case NPTypeCode.Int32:
                        case NPTypeCode.UInt32:
                        case NPTypeCode.Int64:
                        case NPTypeCode.UInt64:
                            // %c (a code point / byte) and bytes formatting's %s / %b (the raw buffer) are the general's.
                            return conv == 'c' || (bytes && (conv == 's' || conv == 'b'))
                                ? FastValueClass.None
                                : FastValueClass.Integer;
                        case NPTypeCode.Half:
                        case NPTypeCode.Single:
                        case NPTypeCode.Double:
                            // x X o c refuse a float (the general routine's TypeError); bytes %s / %b is the buffer.
                            return conv is 'x' or 'X' or 'o' or 'c' || (bytes && (conv == 's' || conv == 'b'))
                                ? FastValueClass.None
                                : FastValueClass.Float;
                        default:
                            return FastValueClass.None;   // bool, char, decimal, complex
                    }
                case OperandKind.Unicode:
                    return !bytes && conv == 's' ? FastValueClass.UnicodeText : FastValueClass.None;
                case OperandKind.Bytes:
                    return bytes && (conv == 's' || conv == 'b') ? FastValueClass.BytesText : FastValueClass.None;
                default:
                    return FastValueClass.None;
            }
        }

        /// <summary>
        ///     The chunk loop over one arena unit type: the literal runs rendered once, then per element the prefix, the field
        ///     and the suffix — written straight into the element's <c>U</c> / <c>S</c> row once the rows exist (zeros to the
        ///     row's width, or a re-layout for a longer element), else into the arena and moved into the result by
        ///     <see cref="RecordLength"/>. An element the field writers refuse runs the general element routine in its place.
        /// </summary>
        /// <remarks>
        ///     The data-dependent decisions of a field (the sign, the rounding) are branch-free in the writers: on real data
        ///     they are coin flips per element, and a mispredicted branch costs as much as the rest of an integer field.
        /// </remarks>
        /// <typeparam name="TUnit"><see cref="uint"/> (str formatting) or <see cref="byte"/> (bytes formatting).</typeparam>
        /// <param name="program">The compiled format.</param>
        /// <param name="format">The format element.</param>
        /// <param name="spec">The format's one spec.</param>
        /// <param name="conv">Its conversion character.</param>
        /// <param name="cls">The values' class.</param>
        /// <param name="reader">The values operand.</param>
        /// <param name="pv">The first value element.</param>
        /// <param name="sv">The values' byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="w">The call's state.</param>
        /// <returns>True (the chunk is always formatted once entered).</returns>
        /// <exception cref="Exception">The first element's CPython error.</exception>
        [SkipLocalsInit]
        private static bool FastChunk<TUnit>(ModFormat program, in Elem format, in ModFormat.Op spec, int conv,
            FastValueClass cls, Operand reader, byte* pv, long sv, long n, ModWalkState w) where TUnit : unmanaged
        {
            // The literal runs before and after the spec, in the arena's unit ('%%' is one '%').
            TUnit* literals = stackalloc TUnit[program.UnitCount + 1];
            int prefixLength = RenderLiterals(program, format.Units, 0, program.FastSpec, literals);
            TUnit* suffix = literals + prefixLength;
            int suffixLength = RenderLiterals(program, format.Units, program.FastSpec + 1, program.Ops.Length, suffix);

            int width = spec.Width < 0 ? 0 : (int)spec.Width;
            int precision = spec.Precision;
            // An upper bound of one element's units: the field is at most its width, or sign + prefix + precision zeros +
            // the longest body; a text value adds its own width.
            long bound = prefixLength + suffixLength + width + Math.Max(precision, 0) + FastBodyCapacity +
                         (cls is FastValueClass.UnicodeText or FastValueClass.BytesText ? reader.Chars : 0);
            NPTypeCode tc = reader.Number;
            int chars = reader.Chars;
            // An integer value under d i u x X o is the commonest call: its field is written by IntegerField directly (no
            // dispatch through IntegerValueField per element).
            bool integerField = cls == FastValueClass.Integer && conv is 'd' or 'i' or 'u' or 'x' or 'X' or 'o';

            // The U / S rows' geometry, cached while it holds: directLimit is the first row whose longest possible element
            // (bound) could pass the result's end — rows before it are formatted straight into the result. -1: not known
            // yet (the width is still sampled, or a re-layout moved the rows), re-read before the next element.
            long directLimit = -1;
            TUnit* rowBase = null;
            int rowWidth = 0;

            for (long i = 0; i < n; i++, pv += sv)
            {
                if (directLimit < 0 && w.RowsActive)
                {
                    rowBase = (TUnit*)w.RowBase;
                    rowWidth = w.RowWidth;
                    directLimit = w.Size - (bound + rowWidth - 1) / rowWidth;
                }
                // Once the U / S rows exist, an element is formatted straight into its row — no arena, no copy — while its
                // longest possible length stays inside the result even if it overruns the row (CommitDirectRow re-lays the
                // rows out then). The sampled elements, a StringDType result, and the last rows go through the arena.
                long position = w.Position;
                bool direct = position < directLimit;
                TUnit* start;
                if (direct)
                {
                    start = rowBase + position * rowWidth;
                }
                else
                {
                    w.Reserve(bound);
                    start = (TUnit*)w.Arena + w.Count;
                }
                TUnit* p = PutUnits(start, literals, prefixLength);
                bool written;
                if (integerField)
                {
                    ReadIntegerValue(tc, pv, out bool negative, out ulong magnitude);
                    p = IntegerField(p, negative, magnitude, conv, in spec, width, precision);
                    written = true;
                }
                else
                {
                    switch (cls)
                    {
                        case FastValueClass.Integer:
                            written = IntegerValueField(ref p, tc, pv, conv, in spec, width, precision, typeof(TUnit) == typeof(byte));
                            break;
                        case FastValueClass.Float:
                            written = FloatValueField(ref p, tc, pv, conv, in spec, width, precision, typeof(TUnit) == typeof(byte));
                            break;
                        case FastValueClass.UnicodeText:
                        {
                            int len = NDCharArray.TrimmedUnits(pv, chars, true);
                            // np.str_ of a one-unit text above U+10FFFF: PyUnicode_New refuses it (the reader's check).
                            if (len == 1 && *(uint*)pv > 0x10FFFF)
                                throw new SystemError(InvalidMaxCharText);
                            p = TextUnitsField(p, (uint*)pv, len, spec.Left, width, precision);
                            written = true;
                            break;
                        }
                        default:
                        {
                            int len = NDCharArray.TrimmedUnits(pv, chars, false);
                            p = TextBytesField(p, pv, len, spec.Left, width, precision);
                            written = true;
                            break;
                        }
                    }
                }
                if (!written)
                {
                    // The general element routine writes this element into the arena (or raises its error) — nothing of
                    // ours was committed — and RecordLength moves it into the result, which may re-lay the rows out.
                    reader.Read(pv, out var value);
                    long s0 = w.Count;
                    if (w.Bytes)
                        ModBytesElement(in format, program, in value, w);
                    else
                        ModStrElement(in format, program, in value, w);
                    RecordLength(w, w.Count - s0);
                    directLimit = -1;
                    continue;
                }
                p = PutUnits(p, suffix, suffixLength);
                long length = p - start;
                if (direct)
                {
                    if (length > w.MaxLength)
                        w.MaxLength = length;
                    if (length <= rowWidth)
                    {
                        // Zeros to the row's width (a few units: a store loop beats a Clear call).
                        for (TUnit* z = start + length, end = start + rowWidth; z < end; z++)
                            *z = default;
                    }
                    else
                    {
                        // Longer than the rows: re-lay them out (the element moves into its wider row).
                        w.CommitDirectRow(length);
                        directLimit = -1;
                    }
                    w.Position = position + 1;
                    continue;
                }
                w.Count += length;
                RecordLength(w, length);
                // A stored row may have started or re-laid out the rows.
                directLimit = -1;
            }
            return true;
        }

        /// <summary>
        ///     Moves the element just formatted (the last <paramref name="length"/> units of the arena) into the result — its
        ///     <c>StringDType</c> cell, or its <c>U</c> / <c>S</c> row — and advances the walk's C-order position.
        /// </summary>
        /// <param name="w">The call's state.</param>
        /// <param name="length">The element's length in units.</param>
        /// <exception cref="OutOfMemoryException">The result cannot be allocated or grown.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void RecordLength(ModWalkState w, long length)
        {
            if (w.Packing)
                w.PackElement(length);
            else
                w.StoreRow(length);
            w.Position++;
        }

        /// <summary>Renders the literal steps <c>[from, to)</c> of a format (text runs, <c>%%</c> as <c>%</c>) into units.</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="program">The compiled format.</param>
        /// <param name="units">The format element's units (code points for str, bytes for bytes).</param>
        /// <param name="from">The first step.</param>
        /// <param name="to">One past the last step (no spec in between).</param>
        /// <param name="dst">Receives the units.</param>
        /// <returns>The number of units written.</returns>
        private static int RenderLiterals<TUnit>(ModFormat program, byte* units, int from, int to, TUnit* dst) where TUnit : unmanaged
        {
            int k = 0;
            for (int s = from; s < to; s++)
            {
                ref readonly var op = ref program.Ops[s];
                if (op.Kind == ModFormat.OpKind.Percent)
                {
                    dst[k++] = Unit<TUnit>('%');
                    continue;
                }
                for (int j = 0; j < op.Length; j++)
                {
                    // A str format's units are code points (any value, a unit past U+10FFFF included), a bytes format's bytes.
                    dst[k++] = typeof(TUnit) == typeof(uint)
                        ? Unsafe.BitCast<uint, TUnit>(((uint*)units)[op.Start + j])
                        : Unsafe.BitCast<byte, TUnit>(units[op.Start + j]);
                }
            }
            return k;
        }

        // =============================================================================================================
        // The element path
        // =============================================================================================================

        /// <summary>
        ///     Formats one element through the fast path when its format qualifies (<see cref="ModFormat.FastSpec"/>) and the
        ///     value is not a Python object: the format's literal runs and its one conversion, appended to the arena.
        /// </summary>
        /// <param name="program">The compiled format (its <see cref="ModFormat.FastSpec"/> is set).</param>
        /// <param name="src">The format's units (code points for str formatting, bytes for bytes formatting).</param>
        /// <param name="value">The element's value.</param>
        /// <param name="w">The call's state.</param>
        /// <returns>False, with nothing appended, when the element must take the general path.</returns>
        /// <exception cref="Exception">The conversion's CPython error (raised by the general conversion routines this
        ///     path calls for the conversions it does not specialize; <see cref="PrintfArgumentException"/> is translated by
        ///     the caller).</exception>
        private static bool TryModFastElement(ModFormat program, byte* src, in Elem value, ModWalkState w)
        {
            ref readonly var spec = ref program.Ops[program.FastSpec];
            int conv = spec.Conversion;
            // Decided before anything is appended: a refusal leaves the arena as the general path expects it.
            if (value.Kind == ElemKind.Object)
                return false;
            if (w.Bytes ? !IsBytesConversion(conv) : !IsStrConversion(conv))
                return false;
            if (spec.Width > FastFieldLimit || spec.Precision > FastFieldLimit)
                return false;

            foreach (ref readonly var op in program.Ops.AsSpan())
            {
                switch (op.Kind)
                {
                    case ModFormat.OpKind.Text:
                        if (w.Bytes)
                            w.AppendBytes(src + op.Start, op.Length);
                        else
                            w.AppendPoints((uint*)src + op.Start, op.Length);
                        break;
                    case ModFormat.OpKind.Percent:
                        w.AppendUnit((byte)'%');
                        break;
                    default:
                        if (w.Bytes)
                            ElementField<byte>(conv, in op, in value, w);
                        else
                            ElementField<uint>(conv, in op, in value, w);
                        break;
                }
            }
            return true;
        }

        /// <summary>
        ///     The one conversion of a fast element: the specialized writers for integer and float values, the general
        ///     conversion routines for everything else (texts, bool, char, complex, a value out of the writers' range).
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="conv">The conversion character (one the formatter supports).</param>
        /// <param name="op">The spec (flags, width, precision).</param>
        /// <param name="v">The value (not an object).</param>
        /// <param name="w">The call's state.</param>
        /// <exception cref="Exception">The conversion's CPython error.</exception>
        private static void ElementField<TUnit>(int conv, in ModFormat.Op op, in Elem v, ModWalkState w) where TUnit : unmanaged
        {
            int width = op.Width < 0 ? 0 : (int)op.Width;
            int precision = op.Precision;
            bool bytes = typeof(TUnit) == typeof(byte);
            if (bytes && conv == 'i')
                conv = 'd';   // _PyBytes_FormatEx converts 'i' as 'd' (its error texts name %d)
            if (v.Kind == ElemKind.Number)
            {
                bool integer = v.Number is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                    or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;
                bool real = v.Number is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;
                if (integer || real)
                {
                    w.Reserve(width + Math.Max(precision, 0) + FastBodyCapacity);
                    TUnit* p = (TUnit*)w.Arena + w.Count;
                    TUnit* start = p;
                    bool written = integer
                        ? IntegerValueField(ref p, v.Number, v.Raw, conv, in op, width, precision, bytes)
                        : FloatValueField(ref p, v.Number, v.Raw, conv, in op, width, precision, bytes);
                    if (written)
                    {
                        w.Count += p - start;
                        return;
                    }
                }
            }
            if (bytes)
                ConvertBytes((char)conv, op.Left, in op, width, precision, in v, w);
            else
                ConvertStr((char)conv, op.Left, in op, width, precision, in v, w);
        }

        // =============================================================================================================
        // Field writers (the caller reserved room; each returns or advances the write pointer)
        // =============================================================================================================

        /// <summary>One ASCII character as an arena unit.</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="c">The character.</param>
        /// <returns>The unit.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TUnit Unit<TUnit>(int c) where TUnit : unmanaged
            => typeof(TUnit) == typeof(uint) ? Unsafe.BitCast<uint, TUnit>((uint)c) : Unsafe.BitCast<byte, TUnit>((byte)c);

        /// <summary>Writes <paramref name="count"/> copies of an ASCII character.</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="c">The character.</param>
        /// <param name="count">The number of copies (nothing when not positive).</param>
        /// <returns>The position after them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TUnit* PutFill<TUnit>(TUnit* p, int c, int count) where TUnit : unmanaged
        {
            if (count <= 0)
                return p;
            new Span<TUnit>(p, count).Fill(Unit<TUnit>(c));
            return p + count;
        }

        /// <summary>Copies units.</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="src">The units.</param>
        /// <param name="count">Their number.</param>
        /// <returns>The position after them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TUnit* PutUnits<TUnit>(TUnit* p, TUnit* src, int count) where TUnit : unmanaged
        {
            for (int i = 0; i < count; i++)
                p[i] = src[i];
            return p + count;
        }

        /// <summary>Writes ASCII bytes as units (widened for str formatting).</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="src">The ASCII bytes.</param>
        /// <param name="count">Their number.</param>
        /// <returns>The position after them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TUnit* PutAsciiUnits<TUnit>(TUnit* p, byte* src, int count) where TUnit : unmanaged
        {
            for (int i = 0; i < count; i++)
                p[i] = Unit<TUnit>(src[i]);
            return p + count;
        }

        /// <summary>
        ///     An integer value's fast conversions: <c>d i u x X o</c>, <c>s</c> (str only; bytes formatting's <c>%s</c> is the
        ///     buffer), <c>r a</c> (NumPy's repr), and the float conversions of <c>float(v)</c>.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position (advanced only when written).</param>
        /// <param name="tc">The value's dtype (an integer dtype).</param>
        /// <param name="raw">The value's element bytes.</param>
        /// <param name="conv">The conversion character.</param>
        /// <param name="op">The spec.</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="bytes">Bytes formatting.</param>
        /// <returns>False, with nothing written, when the conversion is not specialized (the caller's general routine runs).</returns>
        [SkipLocalsInit]
        private static bool IntegerValueField<TUnit>(ref TUnit* p, NPTypeCode tc, byte* raw, int conv, in ModFormat.Op op,
            int width, int precision, bool bytes) where TUnit : unmanaged
        {
            ReadIntegerValue(tc, raw, out bool negative, out ulong magnitude);
            switch (conv)
            {
                case 'd':
                case 'i':
                case 'u':
                case 'x':
                case 'X':
                case 'o':
                    p = IntegerField(p, negative, magnitude, conv, in op, width, precision);
                    return true;
                case 's':
                    if (bytes)
                        return false;
                    goto case 'r';
                case 'r':
                case 'a':
                {
                    byte* body = stackalloc byte[FastBodyCapacity];
                    int len = 0;
                    if (conv != 's')
                        len = PutAscii(body, ScalarReprPrefix(tc));
                    if (negative)
                        body[len++] = (byte)'-';
                    int nd = FloatDigits.DecimalLength(magnitude);
                    FloatDigits.WriteDecimalBackwards(magnitude, body + len + nd);
                    len += nd;
                    if (conv != 's')
                        body[len++] = (byte)')';
                    p = TextAsciiField(p, body, len, op.Left, width, precision);
                    return true;
                }
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                {
                    // float(v): exact below 2^53, rounded once above (an int64 by .NET's correctly rounded conversion; a
                    // uint64 past int64's range through Python's int → float rule, as the general path does).
                    double d;
                    if (tc == NPTypeCode.UInt64 && magnitude > long.MaxValue)
                        d = NumSharp.PythonInt.ToDouble(new System.Numerics.BigInteger(magnitude));
                    else
                        d = negative ? -(double)magnitude : magnitude;
                    return RealField(ref p, d, conv, in op, width, precision);
                }
                default:
                    return false;   // c: the general routine (a code point / one byte)
            }
        }

        /// <summary>An integer NumPy scalar's sign and magnitude (exact for every width, int64's minimum included).</summary>
        /// <param name="tc">The dtype (an integer dtype).</param>
        /// <param name="raw">The element's bytes.</param>
        /// <param name="negative">Receives whether it is below zero.</param>
        /// <param name="magnitude">Receives its absolute value.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ReadIntegerValue(NPTypeCode tc, byte* raw, out bool negative, out ulong magnitude)
        {
            long s;
            switch (tc)
            {
                case NPTypeCode.SByte:
                    s = *(sbyte*)raw;
                    break;
                case NPTypeCode.Int16:
                    s = *(short*)raw;
                    break;
                case NPTypeCode.Int32:
                    s = *(int*)raw;
                    break;
                case NPTypeCode.Int64:
                    s = *(long*)raw;
                    break;
                case NPTypeCode.Byte:
                    negative = false;
                    magnitude = *raw;
                    return;
                case NPTypeCode.UInt16:
                    negative = false;
                    magnitude = *(ushort*)raw;
                    return;
                case NPTypeCode.UInt32:
                    negative = false;
                    magnitude = *(uint*)raw;
                    return;
                default:
                    negative = false;
                    magnitude = *(ulong*)raw;
                    return;
            }
            negative = s < 0;
            // -(s + 1) + 1 cannot overflow, so long.MinValue's magnitude 2^63 comes out exactly.
            magnitude = negative ? (ulong)(-(s + 1)) + 1 : (ulong)s;
        }

        /// <summary>
        ///     An integer conversion's field (<c>_PyUnicode_FormatLong</c> + <c>unicode_format_arg_output</c>, as
        ///     PrintfFormatter's FormatInteger / PadInteger write it): the digits in base 10, 16 or 8, a precision as the
        ///     MINIMUM number of digits, the sign (<c>-</c>, else <c>+</c> / space by flag), the <c>#</c> prefix of
        ///     <c>x X o</c>, and the width — zeros between the sign / prefix and the digits under the <c>0</c> flag, spaces
        ///     after under <c>-</c>, spaces before otherwise. The digits are written in place, right to left.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="negative">The value is below zero.</param>
        /// <param name="magnitude">Its magnitude.</param>
        /// <param name="conv">The conversion character (<c>d i u x X o</c>).</param>
        /// <param name="op">The spec (flags).</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The minimum digit count (-1 for none).</param>
        /// <returns>The position after the field.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static TUnit* IntegerField<TUnit>(TUnit* p, bool negative, ulong magnitude, int conv, in ModFormat.Op op,
            int width, int precision) where TUnit : unmanaged
        {
            int radixBits = conv is 'x' or 'X' ? 4 : conv == 'o' ? 3 : 0;
            int nd = radixBits == 0 ? FloatDigits.DecimalLength(magnitude) : RadixLength(magnitude, radixBits);
            int zeros = precision > nd ? precision - nd : 0;
            // The sign character is chosen without a branch on the sign (a coin flip per element on real data), written
            // unconditionally and kept only when there is one (the next write covers the slot otherwise).
            int flagSign = op.Plus ? '+' : op.Space ? ' ' : 0;
            int sign = negative ? '-' : flagSign;
            bool prefix = op.Alt && radixBits != 0;
            int length = (sign != 0 ? 1 : 0) + (prefix ? 2 : 0) + zeros + nd;
            int pad = width - length;
            if (pad > 0 && !op.Left && !op.Zero)
                p = PutFill(p, ' ', pad);
            *p = Unit<TUnit>(sign);
            p += sign != 0 ? 1 : 0;
            if (prefix)
            {
                *p++ = Unit<TUnit>('0');
                *p++ = Unit<TUnit>(conv);   // 'x', 'X' or 'o': the prefix letter is the conversion's own
            }
            if (pad > 0 && !op.Left && op.Zero)
                p = PutFill(p, '0', pad);
            p = PutFill(p, '0', zeros);
            if (radixBits == 0)
                WriteDecimalUnits(magnitude, p + nd);
            else
                WriteRadixUnits(magnitude, p + nd, radixBits, conv == 'X');
            p += nd;
            if (pad > 0 && op.Left)
                p = PutFill(p, ' ', pad);
            return p;
        }

        /// <summary>The number of base-16 (<paramref name="bits"/> = 4) or base-8 (3) digits of a magnitude (1 for 0).</summary>
        /// <param name="v">The magnitude.</param>
        /// <param name="bits">Bits per digit.</param>
        /// <returns>The digit count.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RadixLength(ulong v, int bits)
        {
            int significant = 64 - System.Numerics.BitOperations.LeadingZeroCount(v | 1);
            return (significant + bits - 1) / bits;
        }

        /// <summary>Writes a magnitude's decimal digits as units ending just before <paramref name="end"/> (two per step).</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="v">The magnitude.</param>
        /// <param name="end">One past the last digit's slot.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteDecimalUnits<TUnit>(ulong v, TUnit* end) where TUnit : unmanaged
        {
            TUnit* q = end;
            while (v >= 100)
            {
                ulong d = v / 100;
                int r = (int)(v - d * 100);
                ulong tens = (ulong)(r * 205) >> 11;   // r / 10 for r < 100 (205 / 2048 ≈ 1 / 10)
                q -= 2;
                q[0] = Unit<TUnit>('0' + (int)tens);
                q[1] = Unit<TUnit>('0' + r - (int)tens * 10);
                v = d;
            }
            if (v >= 10)
            {
                int r = (int)v;
                int tens = (r * 205) >> 11;
                q -= 2;
                q[0] = Unit<TUnit>('0' + tens);
                q[1] = Unit<TUnit>('0' + r - tens * 10);
            }
            else
            {
                *--q = Unit<TUnit>('0' + (int)v);
            }
        }

        /// <summary>Writes a magnitude in base 16 or 8 as units ending just before <paramref name="end"/>.</summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="v">The magnitude.</param>
        /// <param name="end">One past the last digit's slot.</param>
        /// <param name="bits">Bits per digit (4 or 3).</param>
        /// <param name="upper">Upper-case hex letters.</param>
        private static void WriteRadixUnits<TUnit>(ulong v, TUnit* end, int bits, bool upper) where TUnit : unmanaged
        {
            TUnit* q = end;
            ulong mask = (1UL << bits) - 1;
            int letter = upper ? 'A' - 10 : 'a' - 10;
            do
            {
                int d = (int)(v & mask);
                *--q = Unit<TUnit>(d < 10 ? '0' + d : letter + d);
                v >>= bits;
            } while (v != 0);
        }

        /// <summary>The <c>np.&lt;dtype&gt;(</c> prefix of NumPy 2's scalar repr for a numeric dtype.</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>The prefix as ASCII.</returns>
        private static ReadOnlySpan<byte> ScalarReprPrefix(NPTypeCode tc) => tc switch
        {
            NPTypeCode.SByte => "np.int8("u8,
            NPTypeCode.Byte => "np.uint8("u8,
            NPTypeCode.Int16 => "np.int16("u8,
            NPTypeCode.UInt16 => "np.uint16("u8,
            NPTypeCode.Int32 => "np.int32("u8,
            NPTypeCode.UInt32 => "np.uint32("u8,
            NPTypeCode.Int64 => "np.int64("u8,
            NPTypeCode.UInt64 => "np.uint64("u8,
            NPTypeCode.Half => "np.float16("u8,
            NPTypeCode.Single => "np.float32("u8,
            _ => "np.float64("u8,
        };

        /// <summary>Copies an ASCII literal into a body buffer.</summary>
        /// <param name="dst">The buffer.</param>
        /// <param name="text">The literal.</param>
        /// <returns>Its length.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PutAscii(byte* dst, ReadOnlySpan<byte> text)
        {
            text.CopyTo(new Span<byte>(dst, text.Length));
            return text.Length;
        }

        /// <summary>
        ///     A text conversion's field over an ASCII body (PrintfFormatter's PadText): the body cut to the precision, then
        ///     padded with spaces to the width (after it under <c>-</c>); no sign or zero handling (<c>%s</c> ignores them).
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="body">The <c>str()</c> / <c>repr()</c> text.</param>
        /// <param name="length">Its length.</param>
        /// <param name="left">Left-justify.</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The maximum number of characters kept (-1 for all).</param>
        /// <returns>The position after the field.</returns>
        private static TUnit* TextAsciiField<TUnit>(TUnit* p, byte* body, int length, bool left, int width, int precision)
            where TUnit : unmanaged
        {
            int keep = precision >= 0 && precision < length ? precision : length;
            int pad = width - keep;
            if (pad > 0 && !left)
                p = PutFill(p, ' ', pad);
            p = PutAsciiUnits(p, body, keep);
            if (pad > 0 && left)
                p = PutFill(p, ' ', pad);
            return p;
        }

        /// <summary>
        ///     <c>%s</c> of a <c>U</c> value (str formatting, TextIntoStr's rule): its code points cut to the precision, padded
        ///     with spaces to the width.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit (<see cref="uint"/>: str formatting).</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="units">The value's code points.</param>
        /// <param name="length">Their number (trailing NULs removed).</param>
        /// <param name="left">Left-justify.</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The maximum number of code points kept (-1 for all).</param>
        /// <returns>The position after the field.</returns>
        private static TUnit* TextUnitsField<TUnit>(TUnit* p, uint* units, int length, bool left, int width, int precision)
            where TUnit : unmanaged
        {
            int keep = precision >= 0 && precision < length ? precision : length;
            int pad = width - keep;
            if (pad > 0 && !left)
                p = PutFill(p, ' ', pad);
            for (int i = 0; i < keep; i++)
                p[i] = Unsafe.BitCast<uint, TUnit>(units[i]);
            p += keep;
            if (pad > 0 && left)
                p = PutFill(p, ' ', pad);
            return p;
        }

        /// <summary>
        ///     <c>%s</c> / <c>%b</c> of an <c>S</c> value (bytes formatting, ConvertBytes' buffer rule): its bytes cut to the
        ///     precision, padded with spaces to the width.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit (<see cref="byte"/>: bytes formatting).</typeparam>
        /// <param name="p">The write position.</param>
        /// <param name="bytes">The value's bytes.</param>
        /// <param name="length">Their number (trailing NULs removed).</param>
        /// <param name="left">Left-justify.</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The maximum number of bytes kept (-1 for all).</param>
        /// <returns>The position after the field.</returns>
        private static TUnit* TextBytesField<TUnit>(TUnit* p, byte* bytes, int length, bool left, int width, int precision)
            where TUnit : unmanaged
        {
            int keep = precision >= 0 && precision < length ? precision : length;
            int pad = width - keep;
            if (pad > 0 && !left)
                p = PutFill(p, ' ', pad);
            for (int i = 0; i < keep; i++)
                p[i] = Unsafe.BitCast<byte, TUnit>(bytes[i]);
            p += keep;
            if (pad > 0 && left)
                p = PutFill(p, ' ', pad);
            return p;
        }

        /// <summary>
        ///     A float value's fast conversions: <c>d i u</c> (int(v), within int64), <c>s</c> (str only) / <c>r a</c>
        ///     (NumPy's scalar str / repr in the value's own precision), and <c>e E f F g G</c>.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position (advanced only when written).</param>
        /// <param name="tc">The value's dtype (float16 / float32 / float64).</param>
        /// <param name="raw">The value's element bytes.</param>
        /// <param name="conv">The conversion character.</param>
        /// <param name="op">The spec.</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The precision (-1 for none).</param>
        /// <param name="bytes">Bytes formatting.</param>
        /// <returns>False, with nothing written, when the general routine must run (an unspecialized conversion, NaN /
        ///     infinity / |v| ≥ 2^63 under an integer conversion, digits out of <see cref="FloatDigits"/>' range).</returns>
        [SkipLocalsInit]
        private static bool FloatValueField<TUnit>(ref TUnit* p, NPTypeCode tc, byte* raw, int conv, in ModFormat.Op op,
            int width, int precision, bool bytes) where TUnit : unmanaged
        {
            FloatKind kind;
            double d;
            switch (tc)
            {
                case NPTypeCode.Half:
                    kind = FloatKind.Half;
                    d = (double)*(Half*)raw;
                    break;
                case NPTypeCode.Single:
                    kind = FloatKind.Single;
                    d = *(float*)raw;
                    break;
                default:
                    kind = FloatKind.Double;
                    d = *(double*)raw;
                    break;
            }
            switch (conv)
            {
                case 'd':
                case 'i':
                case 'u':
                {
                    // int(v) truncates toward zero; NaN and infinity raise in the general routine, and a magnitude of 2^63
                    // or more needs a Python int (the general routine's BigInteger).
                    double t = Math.Truncate(d);
                    if (!(Math.Abs(t) < 9223372036854775808.0))
                        return false;
                    long s = (long)t;
                    bool negative = s < 0;
                    p = IntegerField(p, negative, negative ? (ulong)(-(s + 1)) + 1 : (ulong)s, conv, in op, width, precision);
                    return true;
                }
                case 's':
                    if (bytes)
                        return false;
                    goto case 'r';
                case 'r':
                case 'a':
                {
                    byte* body = stackalloc byte[FastBodyCapacity];
                    int len = 0;
                    if (conv != 's')
                        len = PutAscii(body, ScalarReprPrefix(tc));
                    // NumPy's scalar str in the value's own precision (shortest digits, the positional window per dtype); at
                    // most 24 bytes, after a repr prefix of at most 11.
                    len += NDTextFormat.FormatFloatAscii(d, kind, body + len);
                    if (conv != 's')
                        body[len++] = (byte)')';
                    p = TextAsciiField(p, body, len, op.Left, width, precision);
                    return true;
                }
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                    return RealField(ref p, d, conv, in op, width, precision);
                default:
                    return false;   // x X o c: the general routine's TypeError
            }
        }

        /// <summary>
        ///     A float conversion's field (<c>formatfloat</c> as PrintfFormatter's FormatFixed / FormatScientific /
        ///     FormatGeneral writes it): CPython's digits from <see cref="FloatDigits"/>, the sign (<c>-</c> for a negative
        ///     value and -0.0; <c>+</c> / space by flag), <c>nan</c> / <c>inf</c> (upper-case for <c>E F G</c>; a NaN never
        ///     shows <c>-</c>), and the width — zeros between the sign and the body under <c>0</c> (also before
        ///     <c>inf</c>, as CPython writes <c>00inf</c>). A finite <c>%f</c> / <c>%e</c> is written straight into the
        ///     destination (<see cref="DirectRealField"/>); <c>%g</c> (whose trailing zeros are stripped) and the non-finite
        ///     words go through a body buffer.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position (advanced only when written).</param>
        /// <param name="d">The value.</param>
        /// <param name="conv">The conversion character (<c>e E f F g G</c>).</param>
        /// <param name="op">The spec (flags).</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The precision (-1 for CPython's default 6).</param>
        /// <returns>False, with nothing written, when the digits are out of <see cref="FloatDigits"/>' range (the general
        ///     routine formats the value).</returns>
        [SkipLocalsInit]
        private static bool RealField<TUnit>(ref TUnit* p, double d, int conv, in ModFormat.Op op, int width, int precision)
            where TUnit : unmanaged
        {
            bool upper = conv == 'E' || conv == 'F' || conv == 'G';
            if (double.IsFinite(d) && conv != 'g' && conv != 'G')
                return DirectRealField(ref p, d, conv, in op, width, precision);
            byte* body = stackalloc byte[FastBodyCapacity];
            int len;
            int sign;
            if (!double.IsFinite(d))
            {
                // NonFinite: a NaN's sign comes from the flags only; an infinity shows its own '-'.
                sign = !double.IsNaN(d) && d < 0 ? '-' : op.Plus ? '+' : op.Space ? ' ' : 0;
                ReadOnlySpan<byte> word = double.IsNaN(d) ? (upper ? "NAN"u8 : "nan"u8) : (upper ? "INF"u8 : "inf"u8);
                len = PutAscii(body, word);
            }
            else
            {
                // A finite %g (the finite %f / %e took the direct route above).
                sign = double.IsNegative(d) ? '-' : op.Plus ? '+' : op.Space ? ' ' : 0;
                if (!TryWriteGeneral(Math.Abs(d), precision, op.Alt, upper, body, out len))
                    return false;
            }

            // Pad: sign + body as one field; zeros go between them, spaces around them.
            int length = (sign != 0 ? 1 : 0) + len;
            int pad = width - length;
            TUnit* q = p;
            if (pad > 0 && !op.Left && !op.Zero)
                q = PutFill(q, ' ', pad);
            if (sign != 0)
                *q++ = Unit<TUnit>(sign);
            if (pad > 0 && !op.Left && op.Zero)
                q = PutFill(q, '0', pad);
            q = PutAsciiUnits(q, body, len);
            if (pad > 0 && op.Left)
                q = PutFill(q, ' ', pad);
            p = q;
            return true;
        }

        /// <summary>
        ///     <c>%f</c> / <c>%e</c> of a finite value written straight into the destination: the digits come from
        ///     <see cref="FloatDigits"/> as one integer, the body's length follows from their count before anything is
        ///     written (so the width's padding goes first), and the digits are laid out right to left around the point — no
        ///     intermediate body buffer and no copy (the fraction digits one division by ten each, the integer part two at a
        ///     time). Byte-identical to the buffered route of <see cref="RealField"/> and to PrintfFormatter.
        /// </summary>
        /// <typeparam name="TUnit">The arena unit.</typeparam>
        /// <param name="p">The write position (advanced only when written).</param>
        /// <param name="d">The value (finite).</param>
        /// <param name="conv">The conversion character (<c>f F e E</c>).</param>
        /// <param name="op">The spec (flags).</param>
        /// <param name="width">The width (0 for none).</param>
        /// <param name="precision">The precision (-1 for CPython's default 6).</param>
        /// <returns>False, with nothing written, when the digits are out of <see cref="FloatDigits"/>' range.</returns>
        private static bool DirectRealField<TUnit>(ref TUnit* p, double d, int conv, in ModFormat.Op op, int width, int precision)
            where TUnit : unmanaged
        {
            // The sign bit (a finite value: -0.0 included); the sign character is chosen without a branch on it, as the
            // sign is a coin flip per element on real data.
            int flagSign = op.Plus ? '+' : op.Space ? ' ' : 0;
            int sign = double.IsNegative(d) ? '-' : flagSign;
            double abs = Math.Abs(d);
            int pr = precision < 0 ? 6 : precision;
            bool point = pr > 0 || op.Alt;
            bool scientific = conv == 'e' || conv == 'E';
            ulong q;
            int exponent = 0;
            int bodyLength;
            if (!scientific)
            {
                if (!FloatDigits.TryFixed(abs, pr, out q))
                    return false;
                // At least one integer digit: a value below 1 is 0.xxx.
                int intDigits = Math.Max(FloatDigits.DecimalLength(q) - pr, 1);
                bodyLength = intDigits + (point ? 1 : 0) + pr;
            }
            else
            {
                if (!FloatDigits.TryScientific(abs, pr, out q, out exponent))
                    return false;
                bodyLength = 1 + (point ? 1 : 0) + pr + (Math.Abs(exponent) >= 100 ? 5 : 4);
            }

            // Pad: sign + body as one field; zeros go between them, spaces around them (PrintfFormatter's Pad).
            int length = (sign != 0 ? 1 : 0) + bodyLength;
            int pad = width - length;
            TUnit* t = p;
            if (pad > 0 && !op.Left && !op.Zero)
                t = PutFill(t, ' ', pad);
            // Written unconditionally and kept only when there is a sign: without one the next write covers the slot.
            *t = Unit<TUnit>(sign);
            t += sign != 0 ? 1 : 0;
            if (pad > 0 && !op.Left && op.Zero)
                t = PutFill(t, '0', pad);

            TUnit* bodyEnd = t + bodyLength;
            TUnit* w = bodyEnd;
            if (scientific)
            {
                // The exponent suffix: e / E, its sign, at least two digits.
                int e = Math.Abs(exponent);
                int expDigits = e >= 100 ? 3 : 2;
                w -= 2 + expDigits;
                w[0] = Unit<TUnit>(conv == 'E' ? 'E' : 'e');
                w[1] = Unit<TUnit>(exponent < 0 ? '-' : '+');
                for (int k = expDigits + 1; k >= 2; k--)
                {
                    w[k] = Unit<TUnit>('0' + e % 10);
                    e /= 10;
                }
            }
            // The fraction digits right to left (zeros for a short q), the point, then what is left of q: the integer
            // part of %f (0 for a value below 1) or the one leading digit of %e.
            ulong v = q;
            for (int k = 0; k < pr; k++)
            {
                ulong tens = v / 10;
                *--w = Unit<TUnit>('0' + (int)(v - tens * 10));
                v = tens;
            }
            if (point)
                *--w = Unit<TUnit>('.');
            WriteDecimalUnits(v, w);
            t = bodyEnd;
            if (pad > 0 && op.Left)
                t = PutFill(t, ' ', pad);
            p = t;
            return true;
        }

        /// <summary>
        ///     <c>%g</c>'s body: P significant digits (6 by default, 0 means 1), fixed notation when the exponent X of the
        ///     rounded value is in <c>[-4, P)</c> with <c>P - 1 - X</c> fraction digits, else <c>%e</c> with <c>P - 1</c>;
        ///     trailing zeros and a bare point stripped unless <c>#</c> (which also keeps a point with no fraction digits).
        /// </summary>
        /// <param name="abs">The magnitude.</param>
        /// <param name="precision">The precision (-1 for 6).</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="upper"><c>%G</c>.</param>
        /// <param name="dst">The body buffer.</param>
        /// <param name="len">Receives the body's length.</param>
        /// <returns>False when out of <see cref="FloatDigits"/>' range (a precision above 18 included).</returns>
        [SkipLocalsInit]
        private static bool TryWriteGeneral(double abs, int precision, bool alt, bool upper, byte* dst, out int len)
        {
            len = 0;
            int p = precision < 0 ? 6 : precision == 0 ? 1 : precision;
            if (p - 1 > FloatDigits.MaxScientificPrecision)
                return false;
            if (!FloatDigits.TryScientific(abs, p - 1, out ulong q, out int x))
                return false;
            byte* digits = stackalloc byte[24];
            FloatDigits.WriteDecimalBackwards(q, digits + p, p);   // the p significant digits (zeros for 0.0)

            if (x >= -4 && x < p)
            {
                // Fixed: X ≥ 0 puts X + 1 digits before the point; X < 0 writes "0." and -X - 1 zeros first.
                int fracStart;
                if (x >= 0)
                {
                    for (int i = 0; i <= x; i++)
                        dst[len++] = digits[i];
                    dst[len++] = (byte)'.';
                    fracStart = len;
                    for (int i = x + 1; i < p; i++)
                        dst[len++] = digits[i];
                }
                else
                {
                    dst[len++] = (byte)'0';
                    dst[len++] = (byte)'.';
                    fracStart = len;
                    for (int i = 0; i < -x - 1; i++)
                        dst[len++] = (byte)'0';
                    for (int i = 0; i < p; i++)
                        dst[len++] = digits[i];
                }
                if (!alt)
                {
                    while (len > fracStart && dst[len - 1] == '0')
                        len--;
                    if (len == fracStart)
                        len--;   // no fraction digit left: drop the point too
                }
                return true;
            }

            // Exponent form: d[.ddd] e±XX.
            dst[len++] = digits[0];
            int pointAt = len;
            dst[len++] = (byte)'.';
            for (int i = 1; i < p; i++)
                dst[len++] = digits[i];
            if (!alt)
            {
                while (len > pointAt + 1 && dst[len - 1] == '0')
                    len--;
                if (len == pointAt + 1)
                    len = pointAt;   // only the point left
            }
            len += WriteExponent(x, upper, dst + len);
            return true;
        }

        /// <summary>The exponent suffix CPython writes: <c>e</c> / <c>E</c>, the sign, at least two digits.</summary>
        /// <param name="exponent">The decimal exponent.</param>
        /// <param name="upper">Upper-case <c>E</c>.</param>
        /// <param name="dst">Where to write.</param>
        /// <returns>The suffix length (4 or 5).</returns>
        private static int WriteExponent(int exponent, bool upper, byte* dst)
        {
            int n = 0;
            dst[n++] = upper ? (byte)'E' : (byte)'e';
            dst[n++] = exponent < 0 ? (byte)'-' : (byte)'+';
            int e = Math.Abs(exponent);
            if (e >= 100)
            {
                dst[n++] = (byte)('0' + e / 100);
                e %= 100;
            }
            dst[n++] = (byte)('0' + e / 10);
            dst[n++] = (byte)('0' + e % 10);
            return n;
        }
    }
}
